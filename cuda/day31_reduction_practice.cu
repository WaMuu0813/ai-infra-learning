#include <iostream>
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << cudaGetErrorString(err) << std::endl;\
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


const int elements_per_thread = 16;
__global__ void reduce_sum(
    const float* input,
    float* partial,
    int N
) {
    __shared__ float data[256];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_start = blockIdx.x * blockDim.x * elements_per_thread;

    float sum = 0.0f;
    for(size_t i = 0;i < elements_per_thread;i++){
        int global_i = block_start + tid + i * blockDim.x;
        if(global_i < N)
            sum += input[global_i];
    }
    data[tid] = sum;

    __syncthreads();

    for(int stride = 128;stride >= 32;stride /= 2){
        if(tid < stride)
            data[tid] += data[tid + stride];
        __syncthreads();
    }

    if(tid < 32){
        float value = data[tid];
        for(int offset = 16;offset >= 1;offset /= 2){
            float other = __shfl_down_sync(0xffffffff,value,offset);
            value += other;
        }
        if(tid == 0)
            partial[bid] = value;
    }
}

int main(){
    const int N = 1 << 20;
    float* h_input = new float[N];
    for(size_t i = 0;i < N;i++){
        h_input[i] = 1.0f;
    }
    float* h_output = new float[1];
    h_output[0] = 0.0f;

    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block * elements_per_thread - 1) 
                                / (threads_per_block * elements_per_thread);
    int bytes = N * sizeof(float);
    float* d_input = nullptr;
    float* d_output = nullptr;
    float* partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,bytes));
    CUDA_CHECK(cudaMalloc(&d_output,sizeof(float)));
    CUDA_CHECK(cudaMalloc(&partial,blocks_per_grid * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input,h_input,bytes,cudaMemcpyHostToDevice));

    reduce_sum<<<blocks_per_grid,threads_per_block>>>(d_input,partial,N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    reduce_sum<<<1,threads_per_block>>>(partial,d_output,blocks_per_grid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output,d_output,sizeof(float),cudaMemcpyDeviceToHost));

    bool correct = (h_output[0] == static_cast<float>(N));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(partial));

    delete[] h_input;
    delete[] h_output;

    std::cout << correct << std::endl;

    return correct ? 0 : 1;
}