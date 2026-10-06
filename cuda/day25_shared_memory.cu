#include <iostream>
#include <cuda_runtime.h>

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

__global__ void shared_copy(
    const int* input,
    int* output,
    int n
) {
    __shared__ int shared_data[256];

    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;

    if(global_i < n)
        shared_data[local_i] = input[global_i];

     __syncthreads();

    if(global_i < n){
        if(local_i == 0)
            output[global_i] = shared_data[local_i];
        else
            output[global_i] = shared_data[local_i - 1];
    }
}

int main(){
    const int N = 10;
    const int THREADS = 4;
    size_t bytes = static_cast<size_t>(N) * sizeof(int);

    int* h_input = new int[N];
    for(size_t i = 0;i < N;i++){
        h_input[i] = i;
    }
    int* h_output = new int[N];

    int* d_input = nullptr;
    int* d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input,bytes));
    CHECK_CUDA(cudaMalloc(&d_output,bytes));

    CHECK_CUDA(cudaMemcpy(d_input,h_input,bytes,cudaMemcpyHostToDevice));

    int BLOCKS = (N + THREADS - 1) / THREADS;
    shared_copy<<<BLOCKS,THREADS>>>(d_input,d_output,N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output,d_output,bytes,cudaMemcpyDeviceToHost));

    for(size_t i = 0;i < N;i++){
        std::cout << h_output[i] << std::endl;
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    delete[] h_input;
    delete[] h_output;

    return 0;
}