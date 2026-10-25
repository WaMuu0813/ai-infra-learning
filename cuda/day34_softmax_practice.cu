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
        if(tid < s)
            smem[tid] += smem[tid + s];
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

template<int ITEMS>
__global__ void softmax_v4(float* input, float* output, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input + row * N;
    float* y = output + row * N;

    float m = -INFINITY;
    float d = 0.0f;
    for(int i = 0;i < ITEMS;i++){
        int idx = tid + i * blockDim.x;
        if(idx >= N)
            break;
        float cur = x[idx];
        float m_new = fmax(cur, m);
        d = d * expf(m - m_new) + expf(cur - m_new);
        m = m_new;
    }
    
    float m_block = blockReduceMaxShuffle(m);
    d = d * expf(m - m_block);

    float d_block = blockReduceSumShuffle(d);
    float inv_sum = 1 / d_block;
    for(int i = 0;i < ITEMS;i++){
        int idx = tid + i * blockDim.x;
        if(idx < N){
            y[idx] = expf(x[idx] - m_block) * inv_sum; 
        }
    }
}

template<int ITEMS>
__global__ void softmax_v4_opt(float* input, float* output, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input + row * N;
    float* y = output + row * N;

    float m = -INFINITY;
    float d = 0.0f;
    for(int i = 0;i < ITEMS;i++){
        int idx = tid + i * blockDim.x;
        if(idx >= N)
            break;
        float cur = x[idx];
        float e = expf(-fabsf(cur - m));
        if(cur <= m)
            d = d + e;
        else{
            d = d * e + 1;
            m = cur;
        }
    }
    
    float m_block = blockReduceMaxShuffle(m);
    d = d * expf(m - m_block);

    float d_block = blockReduceSumShuffle(d);
    float inv_sum = 1 / d_block;
    for(int i = 0;i < ITEMS;i++){
        int idx = tid + i * blockDim.x;
        if(idx < N){
            y[idx] = expf(x[idx] - m_block) * inv_sum; 
        }
    }
}

__device__ void warpReduceOnline(float& m,float& d,const int THREADS_PER_WARP){
    for(int s = THREADS_PER_WARP / 2;s > 0; s >>= 1){
        float m2 = __shfl_down_sync(0xffffffff,m,s);
        float d2 = __shfl_down_sync(0xffffffff,d,s);
        float m_new = fmax(m,m2);
        if(d2 == 0)
            continue;
        d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
        m = m_new;
    }
}

template<int ITEMS>
__global__ void softmax_v5(float* input, float* output, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    int wid = tid / 32;
    int lane = tid % 32;

    float* x = input + row * N;
    float* y = output + row * N;

    float m = -INFINITY;
    float d = 0.0f;
    for(int i = 0;i < ITEMS;i++){
        int idx = tid + i * blockDim.x;
        if(idx >= N)
            break;
        float cur = x[idx];
        float e = expf(-fabsf(cur - m));
        if(cur <= m)
            d = d + e;
        else{
            d = d * e + 1;
            m = cur;
        }
    }
    warpReduceOnline(m,d,32);
    int num_warps = blockDim.x / 32;
    __shared__ float m_warp[32];
    __shared__ float d_warp[32];
    if(lane == 0){
        m_warp[wid] = m;
        d_warp[wid] = d;
    }
    __syncthreads();
    __shared__ float row_max;
    __shared__ float row_sum;
    if(wid == 0){
        m = (lane < num_warps) ? m_warp[lane] : -INFINITY;
        d = (lane < num_warps) ? d_warp[lane] : 0.0f;
        warpReduceOnline(m,d,32);
    }
    
    if(tid == 0){
        row_max = m;
        row_sum = d;
    }
    __syncthreads();
    float inv_sum = 1.0f / row_sum;
    for(int i = 0;i < ITEMS;i++){
        int idx = tid + i * blockDim.x;
        if(idx < N)
            y[idx] = inv_sum * expf(x[idx] - row_max);
    }
}


template<int VEC_ITEMS>
__global__ void softmax_v6(float* input, float* output, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    int wid = tid / 32;
    int lane = tid % 32;

    float* x = input + row * N;
    float* y = output + row * N;

    float m = -INFINITY;
    float d = 0.0f;
    


    // 基地址对齐时，行偏移为 4 个 float 的整数倍即可。
    bool aligned = (static_cast<size_t>(row) * N) % 4 == 0;
    if (aligned) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        int N4 = N / 4;
        
        for (int i = 0; i < VEC_ITEMS; i++) {
            int idx = tid + i * blockDim.x;
            if (idx >= N4)
                break;

            float4 data = x4[idx];
            float val[4] = {data.x, data.y, data.z, data.w};

            for (int j = 0; j < 4; j++) {
                float cur = val[j];
                float e = expf(-fabsf(cur - m));

                if (cur <= m) {
                    d += e;
                } else {
                    d = d * e + 1.0f;
                    m = cur;
                }
            }
        }

        // 不足四个元素的尾部
        for (int idx = N4 * 4 + tid; idx < N; idx += blockDim.x) {
            float cur = x[idx];
            float e = expf(-fabsf(cur - m));

            if (cur <= m) {
                d += e;
            } else {
                d = d * e + 1.0f;
                m = cur;
            }
        }
    } else {
        // 行首未对齐：整行回退为标量加载
        for (int idx = tid; idx < N; idx += blockDim.x) {
            float cur = x[idx];
            float e = expf(-fabsf(cur - m));

            if (cur <= m) {
                d += e;
            } else {
                d = d * e + 1.0f;
                m = cur;
            }
        }
    }

    warpReduceOnline(m,d,32);
    int num_warps = blockDim.x / 32;
    __shared__ float m_warp[32];
    __shared__ float d_warp[32];
    if(lane == 0){
        m_warp[wid] = m;
        d_warp[wid] = d;
    }
    __syncthreads();
    __shared__ float row_max;
    __shared__ float row_sum;
    if(wid == 0){
        m = (lane < num_warps) ? m_warp[lane] : -INFINITY;
        d = (lane < num_warps) ? d_warp[lane] : 0.0f;
        warpReduceOnline(m,d,32);
    }
    
    if(tid == 0){
        row_max = m;
        row_sum = d;
    }
    __syncthreads();
    float inv_sum = 1.0f / row_sum;
    for (int idx = tid; idx < N; idx += blockDim.x) {
        y[idx] = expf(x[idx] - row_max) * inv_sum;
    }
}


template<int VEC_ITEMS>
__global__ void softmax_v6_double(float* input, float* output, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    int wid = tid / 32;
    int lane = tid % 32;

    float* x = input + row * N;
    float* y = output + row * N;

    float m = -INFINITY;
    float d = 0.0f;
    
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int N4 = N / 4;

    // 基地址对齐时，行偏移为 4 个 float 的整数倍即可。
    bool aligned = (static_cast<size_t>(row) * N) % 4 == 0;
    if (aligned) {
        // 完整的 float4
        for (int i = 0; i < VEC_ITEMS; i++) {
            int idx = tid + i * blockDim.x;
            if (idx >= N4)
                break;

            float4 data = x4[idx];
            float val[4] = {data.x, data.y, data.z, data.w};

            for (int j = 0; j < 4; j++) {
                float cur = val[j];
                float e = expf(-fabsf(cur - m));

                if (cur <= m) {
                    d += e;
                } else {
                    d = d * e + 1.0f;
                    m = cur;
                }
            }
        }

        // 不足四个元素的尾部
        for (int idx = N4 * 4 + tid; idx < N; idx += blockDim.x) {
            float cur = x[idx];
            float e = expf(-fabsf(cur - m));

            if (cur <= m) {
                d += e;
            } else {
                d = d * e + 1.0f;
                m = cur;
            }
        }
    } else {
        // 行首未对齐：整行回退为标量加载
        for (int idx = tid; idx < N; idx += blockDim.x) {
            float cur = x[idx];
            float e = expf(-fabsf(cur - m));

            if (cur <= m) {
                d += e;
            } else {
                d = d * e + 1.0f;
                m = cur;
            }
        }
    }

    warpReduceOnline(m,d,32);
    int num_warps = blockDim.x / 32;
    __shared__ float m_warp[32];
    __shared__ float d_warp[32];
    if(lane == 0){
        m_warp[wid] = m;
        d_warp[wid] = d;
    }
    __syncthreads();
    __shared__ float row_max;
    __shared__ float row_sum;
    if(wid == 0){
        m = (lane < num_warps) ? m_warp[lane] : -INFINITY;
        d = (lane < num_warps) ? d_warp[lane] : 0.0f;
        warpReduceOnline(m,d,32);
    }
    
    if(tid == 0){
        row_max = m;
        row_sum = d;
    }
    __syncthreads();
    float inv_sum = 1.0f / row_sum;
    if(aligned){
        for (int i = tid; i < N4; i += blockDim.x) {
            float4 data = x4[i];
            float4 result;

            result.x = expf(data.x - row_max) * inv_sum;
            result.y = expf(data.y - row_max) * inv_sum;
            result.z = expf(data.z - row_max) * inv_sum;
            result.w = expf(data.w - row_max) * inv_sum;

            y4[i] = result;
        }
        // 完整向量之后的标量尾部
        for (int idx = N4 * 4 + tid; idx < N; idx += blockDim.x) {
            y[idx] = expf(x[idx] - row_max) * inv_sum;
        }
    }
    else{
        // 行首未对齐，回退为标量读写
        for (int idx = tid; idx < N; idx += blockDim.x) {
            y[idx] = expf(x[idx] - row_max) * inv_sum;
        }
    }
}

template<int VEC_ITEMS>
__global__ void softmax_v6_fetch(float* input, float* output, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    int wid = tid / 32;
    int lane = tid % 32;

    float* x = input + row * N;
    float* y = output + row * N;

    float m = -INFINITY;
    float d = 0.0f;
    
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int N4 = N / 4;

    // 基地址对齐时，行偏移为 4 个 float 的整数倍即可。
    bool aligned = (static_cast<size_t>(row) * N) % 4 == 0;
    if (aligned) {
        // 完整的 float4
        for (int i = 0; i < VEC_ITEMS; i++) {
            int idx = tid + i * blockDim.x;
            if (idx >= N4)
                break;

            float4 data = x4[idx];
            float val[4] = {data.x, data.y, data.z, data.w};

            for (int j = 0; j < 4; j++) {
                float cur = val[j];
                float e = expf(-fabsf(cur - m));

                if (cur <= m) {
                    d += e;
                } else {
                    d = d * e + 1.0f;
                    m = cur;
                }
            }
        }

        // 不足四个元素的尾部
        for (int idx = N4 * 4 + tid; idx < N; idx += blockDim.x) {
            float cur = x[idx];
            float e = expf(-fabsf(cur - m));

            if (cur <= m) {
                d += e;
            } else {
                d = d * e + 1.0f;
                m = cur;
            }
        }
    } else {
        // 行首未对齐：整行回退为标量加载
        for (int idx = tid; idx < N; idx += blockDim.x) {
            float cur = x[idx];
            float e = expf(-fabsf(cur - m));

            if (cur <= m) {
                d += e;
            } else {
                d = d * e + 1.0f;
                m = cur;
            }
        }
    }

    warpReduceOnline(m,d,32);
    int num_warps = blockDim.x / 32;
    __shared__ float m_warp[32];
    __shared__ float d_warp[32];
    if(lane == 0){
        m_warp[wid] = m;
        d_warp[wid] = d;
    }
    __syncthreads();
    __shared__ float row_max;
    __shared__ float row_sum;
    if(wid == 0){
        m = (lane < num_warps) ? m_warp[lane] : -INFINITY;
        d = (lane < num_warps) ? d_warp[lane] : 0.0f;
        warpReduceOnline(m,d,32);
    }
    
    if(tid == 0){
        row_max = m;
        row_sum = d;
    }
    __syncthreads();
    float max_val = row_max;
    float inv_sum = 1.0f / row_sum;

    if (aligned) {
        int idx = tid;

        // 本线程至少有一个完整向量时，才进行首次加载。
        if (idx < N4) {
            float4 current = x4[idx];

            while (true) {
                int next_idx = idx + blockDim.x;
                bool has_next = next_idx < N4;
                float4 next;

                // 先发起下一组加载，再计算当前组。
                if (has_next) {
                    next = x4[next_idx];
                }

                float4 result;
                result.x = expf(current.x - max_val) * inv_sum;
                result.y = expf(current.y - max_val) * inv_sum;
                result.z = expf(current.z - max_val) * inv_sum;
                result.w = expf(current.w - max_val) * inv_sum;

                y4[idx] = result;

                // 当前组已写完；没有下一组就结束。
                if (!has_next) {
                    break;
                }

                current = next;
                idx = next_idx;
            }
        }

        // 不足四个元素的尾部。
        for (int idx = N4 * 4 + tid; idx < N; idx += blockDim.x) {
            y[idx] = expf(x[idx] - max_val) * inv_sum;
        }
    } else {
        // 行首未对齐，仍使用原来的标量回退。
        for (int idx = tid; idx < N; idx += blockDim.x) {
            y[idx] = expf(x[idx] - max_val) * inv_sum;
        }
    }
}


template<int VEC_ITEMS>
__global__ void softmax_v7(float* input, float* output, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M || N <= 0)
        return;

    int wid = tid / 32;
    int lane = tid % 32;

    size_t row_offset = static_cast<size_t>(row) * N;
    const float* x = input + row_offset;
    float* y = output + row_offset;

    bool aligned = row_offset % 4 == 0;
    int N4 = N / 4;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    // N < 4 时 VEC_ITEMS 可能为 0，避免声明零长度数组。
    float4 cache[VEC_ITEMS > 0 ? VEC_ITEMS : 1];

    float m = -INFINITY;
    float d = 0.0f;

    // 第一遍：加载完整的四元素组，缓存原始值并更新 Online 状态。
    #pragma unroll
    for (int i = 0; i < VEC_ITEMS; i++) {
        int vec_idx = tid + i * blockDim.x;

        if (vec_idx < N4) {
            float4 data;

            if (aligned) {
                data = x4[vec_idx];
            } else {
                // 四个元素都有效，但地址不满足 float4 对齐。
                int base = vec_idx * 4;
                data.x = x[base];
                data.y = x[base + 1];
                data.z = x[base + 2];
                data.w = x[base + 3];
            }

            cache[i] = data;

            float values[4] = {
                data.x, data.y, data.z, data.w
            };

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float cur = values[j];
                float e = expf(-fabsf(cur - m));

                if (cur <= m) {
                    d += e;
                } else {
                    d = d * e + 1.0f;
                    m = cur;
                }
            }
        }
    }

    // 尾部最多三个元素。
    // 在线程数为 32 的整数倍时，每个线程最多负责一个尾部元素。
    int tail_idx = N4 * 4 + tid;
    bool has_tail = tail_idx < N;
    float tail_value = 0.0f;

    if (has_tail) {
        tail_value = x[tail_idx];

        float e = expf(-fabsf(tail_value - m));
        if (tail_value <= m) {
            d += e;
        } else {
            d = d * e + 1.0f;
            m = tail_value;
        }
    }

    // 第一级：Warp 内合并。
    warpReduceOnline(m, d, 32);

    __shared__ float m_warp[32];
    __shared__ float d_warp[32];

    if (lane == 0) {
        m_warp[wid] = m;
        d_warp[wid] = d;
    }
    __syncthreads();

    // 第二级：第 0 个 Warp 合并各 Warp 的结果。
    int num_warps = blockDim.x / 32;

    __shared__ float row_max;
    __shared__ float row_sum;

    if (wid == 0) {
        m = lane < num_warps ? m_warp[lane] : -INFINITY;
        d = lane < num_warps ? d_warp[lane] : 0.0f;

        warpReduceOnline(m, d, 32);
    }

    if (tid == 0) {
        row_max = m;
        row_sum = d;
    }
    __syncthreads();

    float max_val = row_max;
    float inv_sum = 1.0f / row_sum;

    // 第二遍：只使用缓存，不再读取输入 x。
    #pragma unroll
    for (int i = 0; i < VEC_ITEMS; i++) {
        int vec_idx = tid + i * blockDim.x;

        if (vec_idx < N4) {
            float4 data = cache[i];
            float4 result;

            result.x = expf(data.x - max_val) * inv_sum;
            result.y = expf(data.y - max_val) * inv_sum;
            result.z = expf(data.z - max_val) * inv_sum;
            result.w = expf(data.w - max_val) * inv_sum;

            if (aligned) {
                y4[vec_idx] = result;
            } else {
                int base = vec_idx * 4;
                y[base]     = result.x;
                y[base + 1] = result.y;
                y[base + 2] = result.z;
                y[base + 3] = result.w;
            }
        }
    }

    if (has_tail) {
        y[tail_idx] = expf(tail_value - max_val) * inv_sum;
    }
}


__global__ void softmax_v8(float* input, float* output, int M, int N){
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    int num_warps = blockDim.x / 32;

    int row = blockIdx.x * num_warps + wid;

    // 同一 Warp 的线程一起退出。
    if (row >= M || N <= 0)
        return;

    size_t row_offset = static_cast<size_t>(row) * N;
    const float* x = input + row_offset;
    float* y = output + row_offset;

    float m = -INFINITY;
    float d = 0.0f;

    // 第一遍：每个线程扫描本行中分配给自己的元素。
    for (int idx = lane; idx < N; idx += 32) {
        float cur = x[idx];
        float e = expf(-fabsf(cur - m));

        if (cur <= m) {
            d += e;
        } else {
            d = d * e + 1.0f;
            m = cur;
        }
    }

    // 每个 Warp 独立合并自己负责的一行。
    warpReduceOnline(m, d, 32);

    // lane 0 持有完整结果，广播到本 Warp。
    m = __shfl_sync(0xffffffff, m, 0);
    d = __shfl_sync(0xffffffff, d, 0);

    float inv_sum = 1.0f / d;

    // 第二遍：重新读取输入并归一化。
    for (int idx = lane; idx < N; idx += 32) {
        y[idx] = expf(x[idx] - m) * inv_sum;
    }
}


__global__ void softmax_v9(float* input, float* output, int M, int N) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    int num_warps = blockDim.x / 32;

    __shared__ float warp_m[32];
    __shared__ float warp_d[32];
    __shared__ float final_m;
    __shared__ float final_d;

    // 整个 block 一起退出。
    if (N <= 0)
        return;

    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        size_t row_offset = static_cast<size_t>(row) * N;

        const float* x = input + row_offset;
        float* y = output + row_offset;

        // 前提：input、output 均为 cudaMalloc 返回的对齐基地址。
        bool aligned = (row_offset % 4 == 0);

        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4* y4 = reinterpret_cast<float4*>(y);
        int N4 = N / 4;

        float local_m = -INFINITY;
        float local_d = 0.0f;

        // 第 1 遍：计算线程局部 Online 状态。
        if (aligned) {
            for (int i = tid; i < N4; i += blockDim.x) {
                float4 data = x4[i];
                float vals[4] = {
                    data.x, data.y, data.z, data.w
                };

                for (int k = 0; k < 4; k++) {
                    float cur = vals[k];
                    float m_new = fmaxf(local_m, cur);

                    local_d =
                        local_d * expf(local_m - m_new)
                        + expf(cur - m_new);

                    local_m = m_new;
                }
            }

            // 完整 float4 之后的标量尾部。
            for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
                float cur = x[i];
                float m_new = fmaxf(local_m, cur);

                local_d =
                    local_d * expf(local_m - m_new)
                    + expf(cur - m_new);

                local_m = m_new;
            }
        } else {
            // 行首未对齐：整行使用标量加载。
            for (int i = tid; i < N; i += blockDim.x) {
                float cur = x[i];
                float m_new = fmaxf(local_m, cur);

                local_d =
                    local_d * expf(local_m - m_new)
                    + expf(cur - m_new);

                local_m = m_new;
            }
        }

        // 第一级：每个 Warp 内合并。
        warpReduceOnline(local_m, local_d, 32);

        if (lane == 0) {
            warp_m[wid] = local_m;
            warp_d[wid] = local_d;
        }
        __syncthreads();

        // 第二级：第一个 Warp 合并各 Warp 的结果。
        if (wid == 0) {
            local_m = (lane < num_warps)
                ? warp_m[lane] : -INFINITY;

            local_d = (lane < num_warps)
                ? warp_d[lane] : 0.0f;

            warpReduceOnline(local_m, local_d, 32);
        }

        if (tid == 0) {
            final_m = local_m;
            final_d = local_d;
        }
        __syncthreads();

        float row_max = final_m;
        float inv_sum = 1.0f / final_d;

        // 第 2 遍：重新加载输入并归一化输出。
        if (aligned) {
            for (int i = tid; i < N4; i += blockDim.x) {
                float4 data = x4[i];
                float4 result;

                result.x = expf(data.x - row_max) * inv_sum;
                result.y = expf(data.y - row_max) * inv_sum;
                result.z = expf(data.z - row_max) * inv_sum;
                result.w = expf(data.w - row_max) * inv_sum;

                y4[i] = result;
            }

            for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
                y[i] = expf(x[i] - row_max) * inv_sum;
            }
        } else {
            // 行首未对齐：整行使用标量读写。
            for (int i = tid; i < N; i += blockDim.x) {
                y[i] = expf(x[i] - row_max) * inv_sum;
            }
        }

        // 所有线程完成当前行，再进入下一行复用共享内存。
        __syncthreads();
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
    const int M = 1025;
    constexpr int N = 257;
    constexpr int THREADS = 256;
    const int WARM_UP = 100;
    const int REPEAT = 1000;
    constexpr int ITEMS = (N + THREADS - 1) / THREADS;
    constexpr int VEC_ITEMS = (N / 4 + THREADS - 1) / THREADS;
    constexpr int WARPS_PER_BLOCK = THREADS / 32;
    int grid_v8 = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int grid_v9 = 256;
    std::cout << "M=" << M
          << ", N=" << N
          << ", THREADS=" << THREADS
          << ", ITEMS=" << ITEMS
          << ", WARM_UP=" << WARM_UP
          << ", REPEAT=" << REPEAT
          << ", VEC_ITEMS=" << VEC_ITEMS
          << ", grid_v9=" << grid_v9
          << '\n';
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
        softmax_v4<ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
        softmax_v4_opt<ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
        softmax_v5<ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
        softmax_v6<VEC_ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
        softmax_v6_double<VEC_ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
        softmax_v6_fetch<VEC_ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
        softmax_v7<VEC_ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
        softmax_v8<<<grid_v8,THREADS>>>(d_input,d_output,M,N);
        softmax_v9<<<grid_v9,THREADS>>>(d_input,d_output,M,N);
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

    
    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0;i < REPEAT;i++){
        softmax_v4<ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    float avg_ms_v4 = ms / REPEAT;
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
        std::cout << "V4 PASSED" << std::endl;
    else
        std::cout << "V4 FAILED" << std::endl;

    std::cout << "V4 Unlimited: " << unlimited << std::endl;
    std::cout << "V4 Max Error: " << max_error << std::endl;


    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0;i < REPEAT;i++){
        softmax_v4_opt<ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    float avg_ms_v4_opt = ms / REPEAT;
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
        std::cout << "V4_OPT PASSED" << std::endl;
    else
        std::cout << "V4_OPT FAILED" << std::endl;

    std::cout << "V4_OPT Unlimited: " << unlimited << std::endl;
    std::cout << "V4_OPT Max Error: " << max_error << std::endl;


    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0;i < REPEAT;i++){
        softmax_v5<ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    float avg_ms_v5 = ms / REPEAT;
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
        std::cout << "V5 PASSED" << std::endl;
    else
        std::cout << "V5 FAILED" << std::endl;

    std::cout << "V5 Unlimited: " << unlimited << std::endl;
    std::cout << "V5 Max Error: " << max_error << std::endl;


    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0;i < REPEAT;i++){
        softmax_v6<VEC_ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    float avg_ms_v6 = ms / REPEAT;
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
        std::cout << "V6 PASSED" << std::endl;
    else
        std::cout << "V6 FAILED" << std::endl;

    std::cout << "V6 Unlimited: " << unlimited << std::endl;
    std::cout << "V6 Max Error: " << max_error << std::endl;

    
    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0;i < REPEAT;i++){
        softmax_v6_double<VEC_ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    float avg_ms_v6_double = ms / REPEAT;
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
        std::cout << "V6_DOUBLE PASSED" << std::endl;
    else
        std::cout << "V6_DOUBLE FAILED" << std::endl;

    std::cout << "V6_DOUBLE Unlimited: " << unlimited << std::endl;
    std::cout << "V6_DOUBLE Max Error: " << max_error << std::endl;


    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0;i < REPEAT;i++){
        softmax_v6_fetch<VEC_ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    float avg_ms_v6_fetch = ms / REPEAT;
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
        std::cout << "V6_FETCH PASSED" << std::endl;
    else
        std::cout << "V6_FETCH FAILED" << std::endl;

    std::cout << "V6_FETCH Unlimited: " << unlimited << std::endl;
    std::cout << "V6_FETCH Max Error: " << max_error << std::endl;


    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0;i < REPEAT;i++){
        softmax_v7<VEC_ITEMS><<<M,THREADS>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    float avg_ms_v7 = ms / REPEAT;
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
        std::cout << "V7 PASSED" << std::endl;
    else
        std::cout << "V7 FAILED" << std::endl;

    std::cout << "V7 Unlimited: " << unlimited << std::endl;
    std::cout << "V7 Max Error: " << max_error << std::endl;


    CHECK_CUDA(cudaMemset(d_output, 0xff, bytes));
    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0;i < REPEAT;i++){
        softmax_v8<<<grid_v8,THREADS>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    float avg_ms_v8 = ms / REPEAT;
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
        std::cout << "V8 PASSED" << std::endl;
    else
        std::cout << "V8 FAILED" << std::endl;

    std::cout << "V8 Unlimited: " << unlimited << std::endl;
    std::cout << "V8 Max Error: " << max_error << std::endl;


    CHECK_CUDA(cudaMemset(d_output, 0xff, bytes));
    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0;i < REPEAT;i++){
        softmax_v9<<<grid_v9,THREADS>>>(d_input,d_output,M,N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    float avg_ms_v9 = ms / REPEAT;
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
        std::cout << "V9 PASSED" << std::endl;
    else
        std::cout << "V9 FAILED" << std::endl;

    std::cout << "V9 Unlimited: " << unlimited << std::endl;
    std::cout << "V9 Max Error: " << max_error << std::endl;


    std::cout << "V1 Average Time: " << avg_ms_v1 << "ms" << std::endl;
    std::cout << "V2 Average Time: " << avg_ms_v2 << "ms" << std::endl;
    std::cout << "V3 Average Time: " << avg_ms_v3 << "ms" << std::endl;
    std::cout << "V4 Average Time: " << avg_ms_v4 << "ms" << std::endl;
    std::cout << "V4_OPT Average Time: " << avg_ms_v4_opt << "ms" << std::endl;
    std::cout << "V5 Average Time: " << avg_ms_v5 << "ms" << std::endl;
    std::cout << "V6 Average Time: " << avg_ms_v6 << "ms" << std::endl;
    std::cout << "V6_DOUBLE Average Time: " << avg_ms_v6_double << "ms" << std::endl;
    std::cout << "V6_FETCH Average Time: " << avg_ms_v6_fetch << "ms" << std::endl;
    std::cout << "V7 Average Time: " << avg_ms_v7 << "ms" << std::endl;
    std::cout << "V8 Average Time: " << avg_ms_v8 << "ms" << std::endl;
    std::cout << "V9 Average Time: " << avg_ms_v9 << "ms" << std::endl;

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}