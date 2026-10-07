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

#include <iostream>
#include <cuda_runtime.h>


__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(){
    const int REPEAT = 100;
    const int N = 20 * 1000 * 1000;
    const int THREADS = 256;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    size_t bytes = N * sizeof(float);

    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];
    const int MOD = 1 << 16;
    for(size_t i = 0;i < N;i++){
        h_A[i] = static_cast<float>(i % MOD);
        h_B[i] = static_cast<float>(2 * i % MOD);
    }


    float *d_A, *d_B, *d_C;

    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B,h_B,bytes,cudaMemcpyHostToDevice));

    for(size_t i = 0;i < 10;i++){
        vector_add<<<BLOCKS,THREADS>>>(d_A,d_B,d_C,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start,stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for(size_t i = 0;i < REPEAT;i++){
        vector_add<<<BLOCKS,THREADS>>>(d_A,d_B,d_C,N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    ms /= REPEAT;

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(h_C,d_C,bytes,cudaMemcpyDeviceToHost));

    bool correct = true;
    for(size_t i = 0;i < N;i++){
        if(h_C[i] != (h_A[i] + h_B[i])){
            correct = false;
            break;
        }
    }

    float eff_mem_bandwidth = 0.0f;
    eff_mem_bandwidth = 3 * bytes / ms * 1000 / 1e9;

    if(correct)
        std::cout << "结果正确。" << std::endl;
    else
        std::cout << "结果错误。" << std::endl;

    std::cout << "Average Kernel Time: " << ms << "ms" << std::endl;
    std::cout << "Effective Bandwidth: " << eff_mem_bandwidth << "GB/s" << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}