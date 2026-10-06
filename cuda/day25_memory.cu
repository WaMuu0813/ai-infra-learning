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

__global__ void double_elements(int* data,int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        data[i] *= 2;
}

int main(){
    const int N = 8;
    size_t bytes = N * sizeof(int);
    int* h_data = new int[N];
    for(int i = 0;i < N;i++){
        h_data[i] = i;
    }

    int* d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    CHECK_CUDA(cudaMemcpy(
        d_data,
        h_data,
        bytes,
        cudaMemcpyHostToDevice
    ));

    double_elements<<<1,8>>>(d_data,N);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    for(int i = 0;i < N;i++){
        std::cout << h_data[i] << std::endl;
    }
    CHECK_CUDA(cudaMemcpy(
        h_data,
        d_data,
        bytes,
        cudaMemcpyDeviceToHost
    ));
    for(int i = 0;i < N;i++){
        std::cout << h_data[i] << std::endl;
    }

    CHECK_CUDA(cudaFree(d_data));
    delete[] h_data;

    return 0;
}   