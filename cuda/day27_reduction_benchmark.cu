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

constexpr int ITEMS_PER_THREAD = 16;

__global__ void reduce_sum(
    const float* input,
    float* partial_sums,
    int N
) {
    __shared__ float data[256];
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x * ITEMS_PER_THREAD;
    float value = 0.0f;
    for(size_t i = 0;i < ITEMS_PER_THREAD;i++){
        int global_i = block_start + tid + i * blockDim.x;

        if(global_i < N)
            value += input[global_i];
    }
    data[tid] = value;

    __syncthreads();

    int stride = 128;
    while(stride >= 32){
        if(tid < stride){
            data[tid] += data[tid + stride];
        }
        stride /= 2;
        __syncthreads();
    }
    if (tid < 32) {
        float value = data[tid];

        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }

        if (tid == 0) {
            partial_sums[blockIdx.x] = value;
        }
    }
}

int main() {
    const int N = 1 << 23;
    const int THREADS = 256;
    // const int BLOCKS = (N + THREADS - 1) / THREADS;
    const int BLOCKS = (N + THREADS * ITEMS_PER_THREAD - 1) / (THREADS * ITEMS_PER_THREAD);
    const int WARMUP = 10;
    const int REPEAT = 100;

    const size_t input_bytes = N * sizeof(float);
    const size_t output_bytes = BLOCKS * sizeof(float);

    // Host Memory
    float* h_A = new float[N];
    float* h_sum = new float[BLOCKS];

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i % 5);
    }

    // Device Memory
    float *d_A = nullptr, *d_sum = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_sum, output_bytes));

    CHECK_CUDA(cudaMemcpy(
        d_A, h_A, input_bytes, cudaMemcpyHostToDevice
    ));

    // CUDA Events
    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    for (int i = 0; i < WARMUP; i++) {
        reduce_sum<<<BLOCKS, THREADS>>>(d_A, d_sum, N);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; i++) {
        reduce_sum<<<BLOCKS, THREADS>>>(d_A, d_sum, N);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float total_ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(
        &total_ms, start, stop
    ));

    double avg_ms = static_cast<double>(total_ms) / REPEAT;

    // Copy partial sums back to CPU
    CHECK_CUDA(cudaMemcpy(
        h_sum, d_sum, output_bytes, cudaMemcpyDeviceToHost
    ));

    // CPU final reduction
    double actual_sum = 0.0;

    for (int i = 0; i < BLOCKS; i++) {
        actual_sum += static_cast<double>(h_sum[i]);
    }

    // Expected result
    double expected_sum = 0.0;

    for (int i = 0; i < N; i++) {
        expected_sum += static_cast<double>(h_A[i]);
    }

    // Effective Bandwidth
    double bytes = static_cast<double>(input_bytes + output_bytes);
    double seconds = avg_ms / 1000.0;
    double bandwidth_gbs = bytes / seconds / 1e9;

    std::cout << "N: " << N << '\n';
    std::cout << "Blocks: " << BLOCKS << '\n';
    std::cout << "Threads per Block: " << THREADS << '\n';

    std::cout << "Expected Sum: " << expected_sum << '\n';
    std::cout << "Actual Sum: " << actual_sum << '\n';

    std::cout << "Verification: "
              << (actual_sum == expected_sum ? "PASS" : "FAIL")
              << '\n';

    std::cout << "Average Kernel Time: "
              << avg_ms << " ms\n";

    std::cout << "Effective Bandwidth: "
              << bandwidth_gbs << " GB/s\n";

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_sum));

    delete[] h_A;
    delete[] h_sum;

    return 0;
}