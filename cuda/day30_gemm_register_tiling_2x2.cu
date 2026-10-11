#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << cudaGetErrorString(err) << std::endl;\
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)

__global__ void gemm_register_tiling_2x2(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx;

    // 计算映射：每个线程负责 2×2 输出
    int row0 = blockIdx.y * 32 + ty * 2;
    int row1 = row0 + 1;

    // int col0 = blockIdx.x * 32 + tx * 2;
    // int col1 = col0 + 1;
    int col0 = blockIdx.x * 32 + tx;
    int col1 = col0 + 16;

    float c00 = 0.0f;
    float c01 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;

    for (int t = 0; t < (K + 31) / 32; t++) {

        // ===== TODO 1、2：加载 A、B Tile =====
        for (int i = 0; i < 4; i++) {
            int index = tid + i * 256;

            int load_row = index / 32;
            int load_col = index % 32;

            // 在这里计算全局索引
            int a_row = blockIdx.y * 32 + load_row;
            int a_col = t * 32 + load_col;

            int b_row = t * 32 + load_row;
            int b_col = blockIdx.x * 32 + load_col;

            // 在这里完成 Global → Shared 加载
            As[load_row][load_col] =
                (a_row < M && a_col < K)
                ? A[a_row * K + a_col]
                : 0.0f;

            Bs[load_row][load_col] =
                (b_row < K && b_col < N)
                ? B[b_row * N + b_col]
                : 0.0f;
                }

        __syncthreads();

        // ===== TODO 3：Register Tiling 计算 =====
        for (int k = 0; k < 32; k++) {
            float a0 = As[ty * 2][k];
            float a1 = As[ty * 2 + 1][k];

            // float b0 = Bs[k][tx * 2];
            // float b1 = Bs[k][tx * 2 + 1];
            float b0 = Bs[k][tx];
            float b1 = Bs[k][tx + 16];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    // ===== TODO 4：写回 C =====
    if (row0 < M) {
        if (col0 < N) C[row0 * N + col0] = c00;
        if (col1 < N) C[row0 * N + col1] = c01;
    }

    if (row1 < M) {
        if (col0 < N) C[row1 * N + col0] = c10;
        if (col1 < N) C[row1 * N + col1] = c11;
    }
}

bool verify(
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C,
    int M, int N, int K
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }

            float actual = C[i * N + j];
            float tolerance = 1e-3f + 1e-3f * std::fabs(sum);

            if (std::fabs(actual - sum) > tolerance) {
                std::cerr << "Mismatch at (" << i << ", " << j
                          << "): expected=" << sum
                          << ", actual=" << actual << std::endl;
                return false;
            }
        }
    }

    return true;
}

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    constexpr int WARMUP = 10;
    constexpr int REPEAT = 100;

    size_t bytesA = size_t(M) * K * sizeof(float);
    size_t bytesB = size_t(K) * N * sizeof(float);
    size_t bytesC = size_t(M) * N * sizeof(float);

    std::vector<float> hA(size_t(M) * K);
    std::vector<float> hB(size_t(K) * N);
    std::vector<float> hC(size_t(M) * N);

    // 使用确定性输入，便于重复实验
    for (size_t i = 0; i < hA.size(); i++)
        hA[i] = float(i % 17) / 17.0f;

    for (size_t i = 0; i < hB.size(); i++)
        hB[i] = float(i % 13) / 13.0f;

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 31) / 32, (M + 31) / 32);

    // 正确性测试
    gemm_register_tiling_2x2<<<grid, block>>>(
        dA, dB, dC, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        hC.data(), dC, bytesC, cudaMemcpyDeviceToHost
    ));

    bool passed = verify(hA, hB, hC, M, N, K);

    std::cout << "Correctness: "
              << (passed ? "PASS" : "FAIL") << std::endl;

    if (!passed) {
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return EXIT_FAILURE;
    }

    // Warmup：预热 GPU，减少首次启动等因素的影响
    for (int i = 0; i < WARMUP; i++) {
        gemm_register_tiling_2x2<<<grid, block>>>(
            dA, dB, dC, M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; i++) {
        gemm_register_tiling_2x2<<<grid, block>>>(
            dA, dB, dC, M, N, K
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    float avg_ms = total_ms / REPEAT;

    double flops = 2.0 * M * N * K;
    double gflops = flops / (avg_ms * 1e6);

    std::cout << "Matrix: " << M << "x" << N
              << "x" << K << std::endl;

    std::cout << "Average time: "
              << avg_ms << " ms" << std::endl;

    std::cout << "Performance: "
              << gflops << " GFLOPS" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}