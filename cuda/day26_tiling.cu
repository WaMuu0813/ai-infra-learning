#define THREADS 256
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

#include <iostream>
#include <cuda_runtime.h>

__global__ void neighbor_sum_global(
    const float* input,
    float* output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i + 1 < n) {
        output[i] = input[i] + input[i + 1];
    }
}

__global__ void neighbor_sum_shared(
    const float* input,
    float* output,
    int n
) {
    __shared__ float tile[THREADS + 1];

    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;

    // ① 每个有效线程加载自己的元素
    if(global_i < n)
        tile[local_i] = input[global_i];

    // ② 最后一个线程额外加载 Halo
    if(local_i == THREADS - 1 && global_i + 1 < n)
        tile[THREADS] = input[global_i + 1];

    // ③ 同步
    __syncthreads();

    // ④ 使用 Shared Memory 计算
    if(global_i + 1 < n)
        output[global_i] = tile[local_i] + tile[local_i + 1];
}

int main(){
    const int N = 1 << 22;
    size_t input_bytes = static_cast<size_t>(N) * sizeof(float);
    size_t output_bytes = static_cast<size_t>(N - 1) * sizeof(float);

    float* h_input = new float[N];
    for(size_t i = 0;i < N;i++){
        h_input[i] = i;
    }
    float* h_output_global = new float[N-1];
    float* h_output_shared = new float[N-1];

    float* d_input = nullptr;
    float* d_output_global = nullptr;
    float* d_output_shared = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input,input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output_global,(N - 1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_shared,(N - 1) * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input,h_input,input_bytes,cudaMemcpyHostToDevice));

    int blocks = (N + THREADS - 1) / THREADS;
    const int REPEAT = 100;

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热两个 Kernel
    for (int i = 0; i < 10; i++) {
        neighbor_sum_global<<<blocks, THREADS>>>(
            d_input, d_output_global, N
        );

        neighbor_sum_shared<<<blocks, THREADS>>>(
            d_input, d_output_shared, N
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; i++) {
        neighbor_sum_global<<<blocks, THREADS>>>(
            d_input, d_output_global, N
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float global_ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(
        &global_ms, start, stop
    ));

    global_ms /= REPEAT;

    std::cout << "Global: " << global_ms << " ms\n";

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; i++) {
        neighbor_sum_shared<<<blocks, THREADS>>>(
            d_input, d_output_shared, N
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float shared_ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(
        &shared_ms, start, stop
    ));

    shared_ms /= REPEAT;

    std::cout << "Shared: " << shared_ms << " ms\n";
    // neighbor_sum_global<<<blocks,THREADS>>>(d_input,d_output_global,N);
    // CHECK_CUDA(cudaGetLastError());
    // CHECK_CUDA(cudaDeviceSynchronize());
    // neighbor_sum_shared<<<blocks,THREADS>>>(d_input,d_output_shared,N);
    // CHECK_CUDA(cudaGetLastError());
    // CHECK_CUDA(cudaDeviceSynchronize());

    // CHECK_CUDA(cudaMemcpy(h_output_global,d_output_global,output_bytes,cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaMemcpy(h_output_shared,d_output_shared,output_bytes,cudaMemcpyDeviceToHost));

    // std::cout << "Global: ";
    // for(int i = 0;i < N - 1;i++){
    //     std::cout << h_output_global[i] << " ";
    // }

    // std::cout << std::endl << "Shared: ";
    // for(int i = 0;i < N - 1;i++){
    //     std::cout << h_output_shared[i] << " ";
    // }

    // bool correct = true;

    // for (int i = 0; i < N - 1; i++) {
    //     float expected = h_input[i] + h_input[i + 1];

    //     if (h_output_global[i] != expected ||
    //         h_output_shared[i] != expected) {
    //         std::cout << "Mismatch at " << i << std::endl;
    //         correct = false;
    //         break;
    //     }
    // }

    // std::cout << (correct ? "Correct!" : "Failed!") << std::endl;
    // std::cout << std::endl;

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_global));
    CHECK_CUDA(cudaFree(d_output_shared));

    delete[] h_input;
    delete[] h_output_global;
    delete[] h_output_shared;

    return 0;
}