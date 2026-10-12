#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>

#define CHECK_CUDA(call)                                           \
do {                                                               \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA Error: "                                \
                  << cudaGetErrorString(err)                       \
                  << " at line " << __LINE__ << std::endl;         \
        exit(1);                                                   \
    }                                                              \
} while (0)

#define TILE 16

// ============================================================
// Kernel 1: Shared Memory Tiled GEMM
// 每个线程计算 C 中的 1 个元素
// 每个 Block 负责 16×16 的输出区域
// ============================================================

__global__ void gemm_tiled(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {

        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        // 加载 A Tile
        if (row < M && a_col < K)
            As[ty][tx] = A[row * K + a_col];
        else
            As[ty][tx] = 0.0f;

        // 加载 B Tile
        if (b_row < K && col < N)
            Bs[ty][tx] = B[b_row * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Tile 内计算
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // 写回 C
    if (row < M && col < N)
        C[row * N + col] = sum;
}


// ============================================================
// Kernel 2: Shared Memory + Register Tiling GEMM
// 每个线程计算 C 中的 4 个元素
// 每个 Block 负责 32×32 的输出区域
// ============================================================

__global__ void gemm_register_tiling(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[2 * TILE][TILE];
    __shared__ float Bs[TILE][2 * TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 每个线程负责 4 个输出元素
    int row0 = blockIdx.y * blockDim.y * 2 + ty;
    int row1 = row0 + blockDim.y;

    int col0 = blockIdx.x * blockDim.x * 2 + tx;
    int col1 = col0 + blockDim.x;

    // 4 个寄存器累加器
    float c00 = 0.0f;
    float c01 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;

    // 沿 K 维遍历 Tile
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {

        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        // ----------------------------------------------------
        // 加载 A Tile: 32×16
        // 每个线程加载 2 个 A 元素
        // ----------------------------------------------------

        if (row0 < M && a_col < K)
            As[ty][tx] = A[row0 * K + a_col];
        else
            As[ty][tx] = 0.0f;

        if (row1 < M && a_col < K)
            As[ty + blockDim.y][tx] = A[row1 * K + a_col];
        else
            As[ty + blockDim.y][tx] = 0.0f;

        // ----------------------------------------------------
        // 加载 B Tile: 16×32
        // 每个线程加载 2 个 B 元素
        // ----------------------------------------------------

        if (b_row < K && col0 < N)
            Bs[ty][tx] = B[b_row * N + col0];
        else
            Bs[ty][tx] = 0.0f;

        if (b_row < K && col1 < N)
            Bs[ty][tx + blockDim.x] = B[b_row * N + col1];
        else
            Bs[ty][tx + blockDim.x] = 0.0f;

        __syncthreads();

        // ----------------------------------------------------
        // Register Tiling
        // 每轮读取 2 个 A、2 个 B，完成 4 次乘加
        // ----------------------------------------------------

        for (int k = 0; k < TILE; ++k) {

            float a0 = As[ty][k];
            float a1 = As[ty + blockDim.y][k];

            float b0 = Bs[k][tx];
            float b1 = Bs[k][tx + blockDim.x];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    // --------------------------------------------------------
    // 写回 4 个输出元素
    // --------------------------------------------------------

    if (row0 < M) {
        if (col0 < N)
            C[row0 * N + col0] = c00;

        if (col1 < N)
            C[row0 * N + col1] = c01;
    }

    if (row1 < M) {
        if (col0 < N)
            C[row1 * N + col0] = c10;

        if (col1 < N)
            C[row1 * N + col1] = c11;
    }
}


// ============================================================
// CPU Reference GEMM
// ============================================================

void gemm_cpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int N, int K
) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {

            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}


// ============================================================
// 正确性验证
// ============================================================

bool verify(
    const std::vector<float>& result,
    const std::vector<float>& reference,
    const char* name
) {
    bool correct = true;
    float max_error = 0.0f;

    for (size_t i = 0; i < result.size(); ++i) {

        float error = std::fabs(result[i] - reference[i]);

        max_error = std::max(max_error, error);

        if (!std::isfinite(result[i]) || error > 1e-4f) {
            correct = false;

            std::cout << "Mismatch at index " << i
                      << ": GPU = " << result[i]
                      << ", CPU = " << reference[i]
                      << std::endl;
            break;
        }
    }

    std::cout << name << ": "
              << (correct ? "PASS" : "FAIL")
              << std::endl;

    std::cout << "Max error: "
              << max_error << std::endl;

    return correct;
}


// ============================================================
// Main
// ============================================================

int main() {

    // 非 TILE 整数倍，验证边界处理
    const int M = 33;
    const int N = 30;
    const int K = 17;

    const size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    // Host 数据
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_ref(M * N, 0.0f);

    // 初始化
    for (int i = 0; i < M * K; ++i)
        h_A[i] = static_cast<float>(i % 7);

    for (int i = 0; i < K * N; ++i)
        h_B[i] = static_cast<float>(i % 5);

    // CPU 参考结果
    gemm_cpu(h_A, h_B, h_ref, M, N, K);

    // Device 内存
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    CHECK_CUDA(cudaMalloc(&d_B, bytesB));
    CHECK_CUDA(cudaMalloc(&d_C, bytesC));

    CHECK_CUDA(cudaMemcpy(
        d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice
    ));

    dim3 block(TILE, TILE);

    // ========================================================
    // Test 1: Shared Memory Tiled GEMM
    // ========================================================

    dim3 grid_tiled(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    gemm_tiled<<<grid_tiled, block>>>(
        d_A, d_B, d_C, M, N, K
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost
    ));

    bool pass_tiled = verify(
        h_C, h_ref, "Shared Memory Tiled GEMM"
    );

    std::cout << std::endl;

    // ========================================================
    // Test 2: Register Tiling GEMM
    // ========================================================

    dim3 grid_register(
        (N + 2 * block.x - 1) / (2 * block.x),
        (M + 2 * block.y - 1) / (2 * block.y)
    );

    // 清空输出，避免上一轮结果掩盖写回遗漏
    CHECK_CUDA(cudaMemset(d_C, 0, bytesC));

    gemm_register_tiling<<<grid_register, block>>>(
        d_A, d_B, d_C, M, N, K
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost
    ));

    bool pass_register = verify(
        h_C, h_ref, "Register Tiling GEMM"
    );

    std::cout << std::endl;

    // ========================================================
    // Summary
    // ========================================================

    std::cout << "==============================" << std::endl;

    if (pass_tiled && pass_register)
        std::cout << "ALL TESTS PASSED" << std::endl;
    else
        std::cout << "SOME TESTS FAILED" << std::endl;

    std::cout << "==============================" << std::endl;

    // 释放显存
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return (pass_tiled && pass_register) ? 0 : 1;
}