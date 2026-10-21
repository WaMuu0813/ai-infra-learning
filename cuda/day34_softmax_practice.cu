#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>

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

__global__ void softmax_v1(float* input, float* output, int M, int N) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input + row * N;
    float* y = output + row * N;

    float max_val = -INFINITY;
    for(int i = tid;i < N;i += blockDim.x){
        max_val = fmax(max_val,x[i]);
    }
    smem[tid] = max_val;
    
    __syncthreads();

    for(int s = blockDim.x / 2;s > 0;s >>= 1){
        if(tid < s)
            smem[tid] = fmax(smem[tid],smem[tid + s]);
        __syncthreads();
    }
    
    max_val = smem[0];

    __syncthreads();

    float sum = 0.0f;
    for(int i = tid;i < N;i += blockDim.x){
        sum += expf(x[i] - max_val);
    }
    smem[tid] = sum;

    for(int s = blockDim.x / 2;s > 0;s >>= 1){
        if(tid < s){
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    sum = smem[0];
    float inv_sum = 1.0f / sum;

    for(int i = tid;i < N;i += blockDim.x){
        y[i] = inv_sum * expf(x[i] - max_val);
    }
}

//----------softmax_v2----------//
__device__ float warpReduceMax(float val){
    for(int offset = 16;offset > 0;offset >>= 1){
        val = fmaxf(val,__shfl_down_sync(0xffffffff,val,offset));
    }
    return val;
}

__device__ float warpReduceSum(float val){
    for(int offset = 16;offset > 0;offset >>= 1){
        val += __shfl_down_sync(0xfffffff,val,offset);
    }
    return val;
}

__device__ float blockReduceMaxShuffle(float val){
    __shared__ float warp_max[32];
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    val = warpReduceMax(val);
    if(lane == 0)
        warp_max[wid] = val;
    __syncthreads();
    int num_warps = blockDim.x / 32;
    val = (lane < num_warps) ? warp_max[lane] : -INFINITY;
    if(wid == 0)
        val = warpReduceMax(val);
    __shared__ float block_max;
    if(tid == 0)
        block_max = val;
    __syncthreads();
    return block_max;    
}

__device__ float blockReduceSumShuffle(float val){
    __shared__ float warp_sum[32];
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    val = warpReduceSum(val);
    if(lane == 0)
        warp_sum[wid] = val;
    __syncthreads();
    int num_warps = blockDim.x / 32;
    val = (lane < num_warps) ? warp_sum[lane] : 0.0f;
    __syncthreads();
    if(wid == 0)
        val = warpReduceSum(val);
    __shared__ float block_sum;
    if(tid == 0)
        block_sum = val;
    __syncthreads();
    return block_sum;
}

__global__ void softmax_v2(float* input, float* output, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input + row * N;
    float* y = output + row * N;

    float local_max = -INFINITY;
    for(int i = tid;i < N;i += blockDim.x){
        local_max = fmax(local_max,x[i]);
    }
    float max_val = blockReduceMaxShuffle(local_max);
    float local_sum = 0.0f;
    for(int i = tid;i < N;i += blockDim.x){
        local_sum = expf(x[i] - max_val);
    }
    float sum = blockReduceSumShuffle(local_sum);
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}   

int main(){
    const int M = 128;
    const int N = 257;
    const int THREADS = 256;
    size_t bytes = M * N * sizeof(float);
    std::vector<float> h_input(M * N,0);
    std::vector<float> h_output(M * N,-1);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input,bytes));
    CHECK_CUDA(cudaMalloc(&d_output,bytes));
    
    CHECK_CUDA(cudaMemcpy(d_input,h_input.data(),bytes,cudaMemcpyHostToDevice));

    softmax_v1<<<M,THREADS,THREADS * sizeof(float)>>>(d_input,d_output,M,N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output.data(),d_output,bytes,cudaMemcpyDeviceToHost));
    
    int unlimited = 0;
    float target = 1.0 / N;
    double max = 0;
    for(int i = 0;i < M * N;i++){
        if(!std::isfinite(h_output[i])){
            unlimited++;
            continue;
        }
        if(h_output[i] != target){
            max = fmax(max,std::abs(h_output[i] - target));
        }
    }

    if(unlimited == 0 && max <= 1e-6)
        std::cout << "PASSED" << std::endl;
    else
        std::cout << "FAILED" << std::endl;

    std::cout << "Unlimited: " << unlimited << std::endl;
    std::cout << "Max Error" << max << std::endl;

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}