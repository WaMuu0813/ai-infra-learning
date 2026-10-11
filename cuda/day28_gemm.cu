#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector> 
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " \
                  << cudaGetErrorString(err) \
                  << std::endl; \
        exit(1); \
    } \
} while (0)

__global__ void gemm_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    // TODO 1：定义局部累加变量
    float A_val,B_val;
    float C_val = 0.0f;

    // TODO 2：遍历 k = 0 到 K-1
    //          读取 A[row, k] 和 B[k, col]
    //          相乘并累加
    for(int k = 0;k < K;k++){
        A_val = A[row * K + k];
        B_val = B[k * N + col];
        C_val += A_val * B_val;
    }

    // TODO 3：将结果写入 C[row, col]    
    C[row * N + col] = C_val;
}

constexpr int TILE = 32;

__global__ void gemm_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    // TODO 1：
    // 循环遍历 K 维度的所有 Tile
    for(int t = 0;t < (K + TILE - 1) / TILE;t++){
        // TODO 2：
        // 从 Global Memory 加载 A Tile
        // 注意边界：超出矩阵范围时填 0
        if(row < M && t * TILE + tx < K)
            As[ty][tx] = A[row * K + t * TILE + tx];
        else
            As[ty][tx] = 0.0f;

        // TODO 3：
        // 从 Global Memory 加载 B Tile
        // 注意边界：超出矩阵范围时填 0
        if((t * TILE + ty) < K && col < N)
            Bs[ty][tx] = B[(t * TILE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        // TODO 4：
        // 同步线程
        __syncthreads();

        // TODO 5：
        // 使用 As、Bs 完成 TILE 次乘加
        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // TODO 6：
        // 再次同步线程
        __syncthreads();
    }
    // TODO 7：
    // 将 sum 写入 C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
    // 注意边界
}

__global__ void gemm_register_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row0 = blockIdx.y * 16 + ty * 2;
    int row1 = row0 + 1;
    int col = blockIdx.x * 16 + tx;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int t = 0; t < (K + 15) / 16; t++) {

        int a_col = t * 16 + tx;
        int b_row0 = t * 16 + ty * 2;
        int b_row1 = b_row0 + 1;

        // A Tile：每个线程加载两个元素
        As[ty * 2][tx] =
            (row0 < M && a_col < K)
            ? A[row0 * K + a_col] : 0.0f;

        As[ty * 2 + 1][tx] =
            (row1 < M && a_col < K)
            ? A[row1 * K + a_col] : 0.0f;

        // B Tile：每个线程加载两个元素
        Bs[ty * 2][tx] =
            (b_row0 < K && col < N)
            ? B[b_row0 * N + col] : 0.0f;

        Bs[ty * 2 + 1][tx] =
            (b_row1 < K && col < N)
            ? B[b_row1 * N + col] : 0.0f;

        __syncthreads();

        // TODO：使用 Shared Memory
        // 计算 sum0、sum1
        // 每轮沿 k 遍历 16 个元素
        for(int k = 0;k < 16;k++){
            float b = Bs[k][tx];
            sum0 += As[ty * 2][k] * b;
            sum1 += As[ty * 2 + 1][k] * b;
        }

        __syncthreads();
    }

    // TODO：把 sum0、sum1 写回 C
    // 注意 row0、row1、col 的边界
    if(col < N){
        if(row0 < M)
            C[row0 * N + col] = sum0;
        if(row1 < M)
            C[row1 * N + col] = sum1;
    }
}

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    constexpr int WARMUP = 10;
    constexpr int REPEAT = 50;

    size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> h_A(static_cast<size_t>(M) * K);
    std::vector<float> h_B(static_cast<size_t>(K) * N);

    std::vector<float> h_C_naive(static_cast<size_t>(M) * N);
    std::vector<float> h_C_tiled(static_cast<size_t>(M) * N);
    std::vector<float> h_C_register(static_cast<size_t>(M) * N);

    // 非均匀输入，避免全 1 数据掩盖索引错误
    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            h_A[row * K + k] = static_cast<float>((row + k) % 7) * 0.1f;
        }
    }

    for (int k = 0; k < K; k++) {
        for (int col = 0; col < N; col++) {
            h_B[k * N + col] = static_cast<float>((k + col) % 5) * 0.1f;
        }
    }

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA(cudaMemcpy(
        d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice
    ));

    std::cout << "Matrix: "
              << M << " x " << K
              << " * "
              << K << " x " << N << "\n\n";

    // Benchmark：复用相同的计时逻辑
    auto benchmark = [&](auto kernel, const char* name,
                     std::vector<float>& h_C,
                     dim3 kernel_grid,
                     dim3 kernel_block) -> float {

        // 预热
        for (int i = 0; i < WARMUP; i++) {
            kernel<<<kernel_grid, kernel_block>>>(d_A, d_B, d_C, M, N, K);
        }

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;

        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));

        for (int i = 0; i < REPEAT; i++) {
            kernel<<<kernel_grid, kernel_block>>>(d_A, d_B, d_C, M, N, K);
        }

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaGetLastError());

        float total_ms = 0.0f;

        CHECK_CUDA(cudaEventElapsedTime(
            &total_ms, start, stop
        ));

        float avg_ms = total_ms / REPEAT;

        CHECK_CUDA(cudaMemcpy(
            h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost
        ));

        double flops = 2.0 * M * N * K;

        double gflops =
            flops / (static_cast<double>(avg_ms) / 1000.0) / 1e9;

        std::cout << name << ":\n";
        std::cout << "  Average Time: " << avg_ms << " ms\n";
        std::cout << "  Performance: " << gflops << " GFLOPS\n";

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        return avg_ms;
    };

    // Naive GEMM：16×16 线程
    dim3 naive_block(16, 16);
    dim3 naive_grid(
        (N + 15) / 16,
        (M + 15) / 16
    );

    // Shared Memory Tiled GEMM：32×32 线程
    dim3 tiled_block(TILE, TILE);
    dim3 tiled_grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    // Register Tiling：16×8 线程
    // 每个 Block 计算 C 的 16×16 区域
    dim3 register_block(16, 8);
    dim3 register_grid(
        (N + 15) / 16,
        (M + 15) / 16
    );

    float naive_ms = benchmark(
        gemm_naive,
        "Naive GEMM",
        h_C_naive,
        naive_grid,
        naive_block
    );

    float tiled_ms = benchmark(
        gemm_tiled,
        "Tiled GEMM",
        h_C_tiled,
        tiled_grid,
        tiled_block
    );

    float register_ms = benchmark(
        gemm_register_tiled,
        "Register Tiling GEMM",
        h_C_register,
        register_grid,
        register_block
    );

    // CPU 参考计算：抽样检查多个输出位置
    // 避免完整 CPU GEMM 带来较长的等待时间
    bool naive_pass = true;
    bool tiled_pass = true;
    bool register_pass = true;

    for (int row = 0; row < M; row += 31) {
        for (int col = 0; col < N; col += 29) {

            double expected = 0.0;

            for (int k = 0; k < K; k++) {
                expected +=
                    static_cast<double>(h_A[row * K + k]) *
                    static_cast<double>(h_B[k * N + col]);
            }

            size_t idx = static_cast<size_t>(row) * N + col;

            if (std::fabs(h_C_naive[idx] - expected) > 1e-3) {
                naive_pass = false;
            }

            if (std::fabs(h_C_tiled[idx] - expected) > 1e-3) {
                tiled_pass = false;
            }

            if (std::fabs(h_C_register[idx] - expected) > 1e-3) {
                register_pass = false;
            }
        }
    }

    std::cout << "\nVerification:\n";
    std::cout << "  Naive: "
              << (naive_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "  Tiled: "
              << (tiled_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "  Register Tiling: "
            << (register_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "\nSpeedup (Naive / Tiled): "
              << naive_ms / tiled_ms << "x\n";

    std::cout << "Speedup (Tiled / Register): "
          << tiled_ms / register_ms << "x\n";

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return (naive_pass && tiled_pass && register_pass) ? EXIT_SUCCESS : EXIT_FAILURE;
}