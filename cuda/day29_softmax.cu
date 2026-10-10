#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)              \
                      << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)

constexpr int ROWS = 1024;
constexpr int COLS = 256;

constexpr int WARMUP = 10;
constexpr int REPEAT = 100;

// ============================================================
// Fused Softmax
//
// 一个 Block 处理一行。
// 一个线程处理一个元素。
// ============================================================

__global__ void softmax_fused(
    const float* X,
    float* Y,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    // 保存当前归约阶段的数据
    __shared__ float shared[256];

    // 当前线程负责的输入元素
    float x = X[row * cols + tid];

    // ========================================================
    // 第一阶段：Max Reduction
    // ========================================================

    shared[tid] = x;
    __syncthreads();

    // 树形归约：256 → 128 → ... → 1
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            shared[tid] = fmaxf(
                shared[tid],
                shared[tid + stride]
            );
        }

        __syncthreads();
    }

    // 此时 shared[0] 是整行最大值
    float m = shared[0];
    __syncthreads();

    // ========================================================
    // 第二阶段：计算指数
    // ========================================================

    // TODO 1：
    // 使用当前线程的 x 和整行最大值 m，
    // 计算数值稳定的指数值 e。
    //
    // 提示：使用 expf()

    float e = expf(x - m); 

    // ========================================================
    // 第三阶段：Sum Reduction
    // ========================================================

    // TODO 2：
    // 将当前线程计算的 e 写入 Shared Memory，
    // 并确保所有线程完成写入。

    shared[tid] = e;
    __syncthreads();


    // 树形求和：256 → 128 → ... → 1
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            // TODO 3：
            // 将 shared[tid + stride] 加到 shared[tid]
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    // 此时 shared[0] 是整行指数总和
    float s = shared[0];

    // ========================================================
    // 第四阶段：归一化并写回
    // ========================================================

    // TODO 4：
    // 计算当前线程的 Softmax 输出，
    // 并写回 Y 对应位置。

    Y[row * cols + tid] = e / s;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value = fmaxf(value, other);
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value += other;
    }
    return value;
}

__device__ __forceinline__ float block_reduce_max(float value) {
    __shared__ float warp_results[8];

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // 第一级：每个 Warp 内求最大值
    value = warp_reduce_max(value);

    // 每个 Warp 只让 Lane 0 写入局部结果
    if (lane == 0) {
        warp_results[warp_id] = value;
    }

    __syncthreads();

    // 第二级：Warp 0 读取 8 个局部结果
    float result = -INFINITY;

    if (warp_id == 0) {
        if (lane < 8) {
            result = warp_results[lane];
        }

        result = warp_reduce_max(result);

        // 将整行最大值写回 Shared Memory
        if (lane == 0) {
            warp_results[0] = result;
        }
    }

    __syncthreads();

    // 所有线程获得整行最大值
    return warp_results[0];
}

__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float warp_results[8];

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // 第一级：每个 Warp 求局部和
    value = warp_reduce_sum(value);

    // 每个 Warp 只让 Lane 0 写入局部结果
    if (lane == 0) {
        warp_results[warp_id] = value;
    }

    __syncthreads();

    // 第二级：Warp 0 读取 8 个局部和 并负责求整体和
    float result = 0.0f;

    if (warp_id == 0) {
        if (lane < 8) {
            result = warp_results[lane];
        }

        result = warp_reduce_sum(result);

        // 将整体和写回 Shared Memory
        if (lane == 0) {
            warp_results[0] = result;
        }
    }

    __syncthreads();

    // 所有线程获得整体和
    return warp_results[0];
}

__global__ void softmax_warp(
    const float* X,
    float* Y,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    float x = X[row * cols + tid];

    float m = block_reduce_max(x);

    float e = expf(x - m);

    float s = block_reduce_sum(e);

    Y[row * cols + tid] = e / s;
}

// ============================================================
// CPU Reference
// ============================================================

void softmax_cpu(
    const std::vector<float>& X,
    std::vector<float>& Y,
    int rows,
    int cols
) {
    for (int row = 0; row < rows; row++) {

        float m = -INFINITY;

        for (int col = 0; col < cols; col++) {
            m = std::max(m, X[row * cols + col]);
        }

        double sum = 0.0;

        for (int col = 0; col < cols; col++) {
            sum += std::exp(
                static_cast<double>(X[row * cols + col] - m)
            );
        }

        for (int col = 0; col < cols; col++) {
            double e = std::exp(
                static_cast<double>(X[row * cols + col] - m)
            );

            Y[row * cols + col] =
                static_cast<float>(e / sum);
        }
    }
}

// ============================================================
// Main：内存管理、Benchmark、正确性验证
// ============================================================

float benchmark_softmax(
    const float* d_X,
    float* d_Y,
    int rows,
    int cols,
    bool optimized
) {
    dim3 block(256);
    dim3 grid(rows);

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        if (optimized) {
            softmax_warp<<<grid, block>>>(d_X, d_Y, rows, cols);
        } else {
            softmax_fused<<<grid, block>>>(d_X, d_Y, rows, cols);
        }
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; i++) {
        if (optimized) {
            softmax_warp<<<grid, block>>>(d_X, d_Y, rows, cols);
        } else {
            softmax_fused<<<grid, block>>>(d_X, d_Y, rows, cols);
        }
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float total_ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(
        &total_ms, start, stop
    ));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / REPEAT;
}

int main() {

    const int rows = ROWS;
    const int cols = COLS;

    const size_t count = static_cast<size_t>(rows) * cols;
    const size_t bytes = count * sizeof(float);

    std::vector<float> h_X(count);
    std::vector<float> h_Y(count);
    std::vector<float> h_ref(count);

    // 故意包含较大的数值，测试数值稳定性
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            h_X[row * cols + col] =
                1000.0f +
                static_cast<float>((row * 7 + col * 13) % 101);
        }
    }

    softmax_cpu(h_X, h_ref, rows, cols);

    float* d_X = nullptr;
    float* d_Y = nullptr;

    CHECK_CUDA(cudaMalloc(&d_X, bytes));
    CHECK_CUDA(cudaMalloc(&d_Y, bytes));

    CHECK_CUDA(cudaMemcpy(
        d_X,
        h_X.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    dim3 block(256);
    dim3 grid(rows);

    float baseline_ms = benchmark_softmax(
        d_X, d_Y, rows, cols, false
    );

    float optimized_ms = benchmark_softmax(
        d_X, d_Y, rows, cols, true
    );

    CHECK_CUDA(cudaMemcpy(
        h_Y.data(),
        d_Y,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    double speedup = baseline_ms / optimized_ms;

    std::cout << "\nPerformance:\n";

    std::cout << "  Baseline:  "
            << baseline_ms << " ms\n";

    std::cout << "  Optimized: "
            << optimized_ms << " ms\n";

    std::cout << "  Speedup:   "
            << speedup << "x\n";

    // 逐元素验证
    bool pass = true;
    float max_error = 0.0f;

    for (size_t i = 0; i < count; i++) {

        float error = std::abs(h_Y[i] - h_ref[i]);

        max_error = std::max(max_error, error);

        if (!std::isfinite(h_Y[i]) || error > 1e-5f) {
            pass = false;
        }
    }

    // 验证每行输出之和约等于 1
    bool row_sum_pass = true;

    for (int row = 0; row < rows; row++) {

        double sum = 0.0;

        for (int col = 0; col < cols; col++) {
            sum += h_Y[row * cols + col];
        }

        if (!std::isfinite(sum) ||
            std::abs(sum - 1.0) > 1e-4) {
            row_sum_pass = false;
        }
    }

    // 每次计算读取 X、写入 Y
    double traffic_bytes = 2.0 * bytes;


    std::cout << "Matrix: "
              << rows << " x " << cols << "\n\n";

    std::cout << "Fused Softmax:\n";


    std::cout << "\nVerification:\n";

    std::cout << "  Elementwise: "
              << (pass ? "PASS" : "FAIL") << "\n";

    std::cout << "  Row Sum: "
              << (row_sum_pass ? "PASS" : "FAIL") << "\n";

    std::cout << "  Max Error: "
              << max_error << "\n";

    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));

    return (pass && row_sum_pass) ? 0 : 1;
}