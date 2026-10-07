#include <iostream>
#include <cuda_runtime.h>

#define THREADS 32
#define ITERATIONS 2000
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

__global__ void bank_conflict_kernel(
    float* output,
    int stride
) {
    __shared__ volatile float tile[32 * 33];

    int tid = threadIdx.x;

    // 初始化 Shared Memory
    for (int i = tid; i < 32 * 33; i += THREADS) {
        tile[i] = static_cast<float>(i);
    }

    __syncthreads();

    float sum = 0.0f;

    for (int i = 0; i < ITERATIONS; i++) {
        int index = tid * stride + (i & 31);
        sum += tile[index];
    }

    int global_i =
        blockIdx.x * blockDim.x + threadIdx.x;

    output[global_i] = sum;
}

int main(){
    const int BLOCKS = 128;
    const int REPEAT = 20;
    float* d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_output,sizeof(float) * BLOCKS * THREADS));
    int strides[] = {1,2,3,4,8,16,31,32,33};
    int n = sizeof(strides) / sizeof(strides[0]);
    for(size_t i = 0;i < n;i++){
        int stride = strides[i];
        for(size_t j = 0;j < 5;j++){
            bank_conflict_kernel<<<BLOCKS,THREADS>>>(d_output,stride);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start,stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for(size_t k = 0;k < REPEAT;k++){
            bank_conflict_kernel<<<BLOCKS,THREADS>>>(d_output,stride);
        }
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
        
        ms /= REPEAT; 
        std::cout << "Stride " << stride << " : " << ms << " ms" << std::endl;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(d_output));

    return 0;
}