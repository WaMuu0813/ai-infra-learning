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

__global__ void reduce_sum(
    const float* input,
    float* partial_sums,
    int N
) {
    // 1. 计算线程在 Block 内的索引 tid
    int tid = threadIdx.x;

    // 2. 计算全局索引 i
    int global_i = blockIdx.x * blockDim.x + threadIdx.x;

    // 3. 声明长度为 256 的 Shared Memory
    __shared__ float data[256];

    // 4. 从 Global Memory 加载数据
    //    越界线程写入 0
    if(global_i < N)
        data[tid] = input[global_i];
    else
        data[tid] = 0.0f;

    // 5. 同步，确保数据加载完成
    __syncthreads();

    // 6. 树形归约
    //    stride 从 128 开始，每轮减半
    //    只有 tid < stride 的线程执行加法
    //    每轮计算后正确同步
    int stride = 128;
    while(stride >= 1){
        if(tid < stride){
            data[tid] += data[tid + stride];
        }
        stride /= 2;
        __syncthreads();
    }

    // 7. Thread 0 将 Block Sum 写入 partial_sums
    if(tid == 0)
        partial_sums[blockIdx.x] = data[0];
}

int main(){
    const int N = 1000;
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    int bytes = N * sizeof(float);

    float* h_A = new float[N];
    for(size_t i = 0;i < N;i++){
        h_A[i] = static_cast<float>(i % 5);
    }
    float* h_sum = new float[BLOCKS];
    for(size_t i = 0;i < BLOCKS;i++){
        h_sum[i] = 0;
    }

    float *d_A, *d_sum;
    CHECK_CUDA(cudaMalloc(&d_A,bytes));
    CHECK_CUDA(cudaMalloc(&d_sum,BLOCKS * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice));

    reduce_sum<<<BLOCKS,THREADS>>>(d_A,d_sum,N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_sum,d_sum,BLOCKS * sizeof(float),cudaMemcpyDeviceToHost));

    float total_ans = 0;
    for(size_t i = 0;i < BLOCKS;i++){
        total_ans += h_sum[i];
    }
    std::cout << "The result of Reduction is: "
            << total_ans << std::endl;

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_sum));

    delete[] h_A;
    delete[] h_sum;

    return 0;
}
