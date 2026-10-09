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

constexpr int TILE = 16;

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

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    constexpr int WARMUP = 10;
    constexpr int REPEAT = 100;

    size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> h_A(static_cast<size_t>(M) * K);
    std::vector<float> h_B(static_cast<size_t>(K) * N);
    std::vector<float> h_C(static_cast<size_t>(M) * N);

    // 使用简单、可精确验证的输入
    for (size_t i = 0; i < h_A.size(); i++) {
        h_A[i] = 1.0f;
    }

    for (size_t i = 0; i < h_B.size(); i++) {
        h_B[i] = 1.0f;
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

    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    std::cout << "Matrix A: " << M << " x " << K << '\n';
    std::cout << "Matrix B: " << K << " x " << N << '\n';
    std::cout << "Grid: (" << grid.x << ", " << grid.y << ")\n";
    std::cout << "Block: (" << block.x << ", " << block.y << ")\n";

    // 预热 GPU
    for (int i = 0; i < WARMUP; i++) {
        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // CUDA Event 计时
    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; i++) {
        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float total_ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(
        &total_ms, start, stop
    ));

    float avg_ms = total_ms / REPEAT;

    // 拷贝结果回 CPU
    CHECK_CUDA(cudaMemcpy(
        h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost
    ));

    // 验证结果
    // A、B 全部为 1，所以 C 的每个元素应当等于 K
    bool pass = true;

    for (size_t i = 0; i < h_C.size(); i++) {
        if (std::fabs(h_C[i] - static_cast<float>(K)) > 1e-3f) {
            std::cerr << "Mismatch at index " << i
                      << ", expected: " << K
                      << ", actual: " << h_C[i] << '\n';
            pass = false;
            break;
        }
    }

    double flops =
        2.0 * static_cast<double>(M) * N * K;

    double gflops =
        flops / (static_cast<double>(avg_ms) / 1000.0) / 1e9;

    std::cout << "Verification: "
              << (pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Average Kernel Time: "
              << avg_ms << " ms\n";

    std::cout << "Performance: "
              << gflops << " GFLOPS\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}