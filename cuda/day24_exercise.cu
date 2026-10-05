#include <iostream>
#include <cuda_runtime.h>

__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        C[i] = A[i] + B[i];
    }
}

int main(){
    int N = 1000;
    int bytes = N * sizeof(float);
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    for(int i = 0;i < N;i++){
        h_A[i] = (float)i;
        h_B[i] = (float)(2*i);
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaMalloc(&d_A,bytes);
    cudaMalloc(&d_B,bytes);
    cudaMalloc(&d_C,bytes);

    cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,bytes,cudaMemcpyHostToDevice);

    const int threads_per_block = 128;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks_per_grid,threads_per_block>>>(d_A,d_B,d_C,N);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "Kernel Launch failed: "
            << cudaGetErrorString(err) << '\n';
    }
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        std::cerr << "Kernel Execution failed: "
            << cudaGetErrorString(err) << '\n';
    }

    cudaMemcpy(h_C,d_C,bytes,cudaMemcpyDeviceToHost);

    bool correct = true;
    for(int i = 0;i < N;i++){
        if(h_C[i] != 3 * i){   
            correct = false;
            break;
        }
    }

    //释放申请的所有内存空间以及GPU占用
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    if (correct) {
        std::cout << "Vector addition succeeded!" << std::endl;
    }
    return correct ? 0 : 1;
}