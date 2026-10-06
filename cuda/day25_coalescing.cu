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

__global__ void coalesced_copy(
    const float* input, 
    float* output, 
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        output[i] = input[i];
    }
}

__global__ void strided_copy(
    const float* input,
    float* output,
    int n,
    int stride
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        output[i] = input[i * stride];
    }
}

int main(){
    const int N = 1 << 20;
    const int STRIDE = 32;

    float* h_input = new float[N * STRIDE];
    for (size_t i = 0; i < static_cast<size_t>(N) * STRIDE; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    float* h_output = new float[N];

    float* d_input = nullptr;
    float* d_output = nullptr;

    size_t input_bytes =
        static_cast<size_t>(N) * STRIDE * sizeof(float);

    size_t output_bytes =
        static_cast<size_t>(N) * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_input,input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output,output_bytes));

    CHECK_CUDA(cudaMemcpy(d_input,h_input,input_bytes,cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start,stop;

    for (int i = 0; i < 10; ++i) {
        coalesced_copy<<<blocks, threads>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int REPEAT = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < REPEAT; ++i) {
        coalesced_copy<<<blocks, threads>>>(d_input, d_output, N);
    }

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaMemcpy(h_output,d_output,output_bytes,cudaMemcpyDeviceToHost));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    float avg_ms = total_ms / REPEAT;

    std::cout << "Coalesced average: "
        << avg_ms << " ms\n";

    // Strided warm-up
    for (int i = 0; i < 10; ++i) {
        strided_copy<<<blocks, threads>>>(
            d_input, d_output, N, STRIDE
        );
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Strided timing
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i) {
        strided_copy<<<blocks, threads>>>(
            d_input, d_output, N, STRIDE
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float strided_total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(
        &strided_total_ms, start, stop
    ));

    float strided_avg_ms = strided_total_ms / REPEAT;

    std::cout << "Strided average: "
            << strided_avg_ms << " ms\n";
    
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    delete[] h_input;
    delete[] h_output;

    return 0;
}