#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>

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

    __syncthreads();

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
        val += __shfl_down_sync(0xffffffff,val,offset);
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
        local_sum += expf(x[i] - max_val);
    }
    float sum = blockReduceSumShuffle(local_sum);
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}   

template<int ITEMS>
__global__ void softmax_v3(float* input, float* output, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input + row * N;
    float* y = output + row * N;

    float private_elm[ITEMS];

    float local_max = -INFINITY;
    #pragma unroll
    for(int i = 0;i < ITEMS;i++){
        int idx = tid + i * blockDim.x;
        private_elm[i] = (idx < N) ? x[idx] : -INFINITY;
        local_max = fmax(local_max,private_elm[i]);
    }
    float max_val = blockReduceMaxShuffle(local_max);

    float local_sum = 0.0f;
    #pragma unroll
    for(int i = 0;i < ITEMS;i++){
        private_elm[i] = expf(private_elm[i] - max_val);
        local_sum += private_elm[i];
    }
    float sum = blockReduceSumShuffle(local_sum);
    float inv_sum = 1.0f / sum;

    #pragma unroll
    for(int i = 0;i < ITEMS;i++){
        int idx = tid + i * blockDim.x;
        if(idx < N)
            y[idx] = private_elm[i] * inv_sum;
    }
}   

std::vector<double> softmax_cpu(const std::vector<float>& input,int m,int n){
    std::vector<double> ans(m * n,0);
    double tmp_max = -INFINITY;
    double tmp_sum = 0;
    for(int i = 0;i < m;i++){
        tmp_max = -INFINITY;
        tmp_sum = 0;
        for(int j = 0;j < n;j++){
            tmp_max = fmax(tmp_max,input[i * n + j]);
        }
        
        for(int k = 0;k < n;k++){
            tmp_sum += exp(input[i * n + k] - tmp_max);
        }
        for(int v = 0;v < n;v++){
            ans[i * n + v] = exp(input[i * n + v] - tmp_max) / tmp_sum;
        }
    }
    return ans;
}

int main(){
    const int M = 1024;
    constexpr int N = 4096;
    constexpr int THREADS = 256;
    const int WARM_UP = 100;
    const int REPEAT = 1000;
    constexpr int ITEMS = (N + THREADS - 1) / THREADS;
    size_t bytes = M * N * sizeof(float);
    std::vector<float> h_input(M * N,0);
    std::vector<float> h_output(M * N,-1);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);

    for (float& value : h_input) {
        value = distribution(generator);
    }

    std::vector<double> h_reference(M * N,0);
    h_reference = softmax_cpu(h_input,M,N);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input,bytes));
    CHECK_CUDA(cudaMalloc(&d_output,bytes));
    
    CHECK_CUDA(cudaMemcpy(d_input,h_input.data(),bytes,cudaMemcpyHostToDevice));

    for(int i = 0;i < WARM_UP;i++){
        softmax_v1<<<M,THREADS,THREADS * sizeof(float)>>>(d_input,d_output,M,N);
        softmax_v2<<<M,THREADS>>>(d_input,d_output,M,N);
        softmax_v3<ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start,stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0;i < REPEAT;i++){
        softmax_v1<<<M,THREADS,THREADS * sizeof(float)>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    float avg_ms_v1 = ms / REPEAT;
    CHECK_CUDA(cudaMemcpy(h_output.data(),d_output,bytes,cudaMemcpyDeviceToHost));

    int unlimited = 0;
    double max_error = 0.0f;
    for(int i = 0;i < M * N;i++){
        if(!std::isfinite(h_output[i])){
            unlimited++;
            continue;
        }

        max_error = fmax(max_error,std::abs(h_output[i] - h_reference[i]));
    }

    if(unlimited == 0 && max_error <= 1e-6)
        std::cout << "V1 PASSED" << std::endl;
    else
        std::cout << "V1 FAILED" << std::endl;
    std::cout << "V1 Unlimited: " << unlimited << std::endl;
    std::cout << "V1 Max Error: " << max_error << std::endl;

    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0;i < REPEAT;i++){
        softmax_v2<<<M,THREADS>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    float avg_ms_v2 = ms / REPEAT;
    CHECK_CUDA(cudaMemcpy(h_output.data(),d_output,bytes,cudaMemcpyDeviceToHost));

    unlimited = 0;
    max_error = 0.0f;
    for(int i = 0;i < M * N;i++){
        if(!std::isfinite(h_output[i])){
            unlimited++;
            continue;
        }

        max_error = fmax(max_error,std::abs(h_output[i] - h_reference[i]));
    }

    if(unlimited == 0 && max_error <= 1e-6)
        std::cout << "V2 PASSED" << std::endl;
    else
        std::cout << "V2 FAILED" << std::endl;

    std::cout << "V2 Unlimited: " << unlimited << std::endl;
    std::cout << "V2 Max Error: " << max_error << std::endl;

    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0;i < REPEAT;i++){
        softmax_v3<ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    float avg_ms_v3 = ms / REPEAT;
    CHECK_CUDA(cudaMemcpy(h_output.data(),d_output,bytes,cudaMemcpyDeviceToHost));

    unlimited = 0;
    max_error = 0.0f;
    for(int i = 0;i < M * N;i++){
        if(!std::isfinite(h_output[i])){
            unlimited++;
            continue;
        }

        max_error = fmax(max_error,std::abs(h_output[i] - h_reference[i]));
    }

    if(unlimited == 0 && max_error <= 1e-6)
        std::cout << "V3 PASSED" << std::endl;
    else
        std::cout << "V3 FAILED" << std::endl;

    std::cout << "V3 Unlimited: " << unlimited << std::endl;
    std::cout << "V3 Max Error: " << max_error << std::endl;
    
    std::cout << "V1 Average Time: " << avg_ms_v1 << "ms" << std::endl;
    std::cout << "V2 Average Time: " << avg_ms_v2 << "ms" << std::endl;
    std::cout << "V3 Average Time: " << avg_ms_v3 << "ms" << std::endl;

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}