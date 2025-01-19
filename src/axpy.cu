#include <cuda.h>
#include <cuda_runtime_api.h> 
#include <cublas_v2.h>
#include <omp.h>
#include <cstdio>
#include <string>
#include <random>
#include <iostream>
#include <iomanip>

#include "utils.hpp"

#ifndef DATA_T
#define DATA_T          float
#endif
#define STRINGIFY(s)    #s
#define MACRO_STR(m)   STRINGIFY(m)

template <typename T>
__global__ void axpy_d_v0(int N, T alpha, const T * X, T * Y) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < N)
        Y[i] = alpha * X[i] + Y[i];
}

template <typename T>
__global__ void axpy_d_v1(int N, T alpha, const T * X, T * Y) {
    int base = (N / gridDim.x) * blockIdx.x + min(N % gridDim.x, blockIdx.x);
    int len = (N / gridDim.x) + (blockIdx.x < N % gridDim.x ? 1 : 0);

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        Y[base + i] = alpha * X[base + i] + Y[base + i];
    }
}

template <typename T>
__global__ void axpy_d_v2(int N, T alpha, const T * X, T * Y) {
    /**
     * @brief This kernel (v2) aligns the address to improve performance compared to the v1 kernel.
     *
     * The main difference between this kernel (v2) and the v1 kernel is that v2 includes optimizations
     * for memory alignment. By aligning the memory addresses, this kernel can take advantage of 
     * coalesced memory accesses, which significantly improves the performance on the GPU.
     * 
     * Memory alignment ensures that the data accesses are more efficient, reducing the number of 
     * memory transactions required and increasing the overall throughput of the kernel execution.
     */
    int section_size = 8;
    int max_len = ceilf((float)N / gridDim.x / section_size) * section_size;
    int base = max_len * blockIdx.x;
    int len = min(max_len, N - base);

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        Y[base + i] = alpha * X[base + i] + Y[base + i];
    }
}

template<typename T>
__host__ void axpy_h(int N, T alpha, const T * X, T * Y) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        Y[i] = alpha * X[i] + Y[i];
    }
}

int main(int argc, const char * argv[]) {
    DATA_T * X_h, * Y_h;
    DATA_T * X_d, * Y_d;
    DATA_T * A_d, * B_d;
    DATA_T alpha;
    int N;

    bool is_data_type_valid = 
        std::is_same<DATA_T, float>::value || std::is_same<DATA_T, double>::value;

    if (argc > 1 && is_data_type_valid) {
        N = std::stoi(argv[1]);
    }
    else {
        perror("Usage: ./axpy <vector size>\n");
        exit(1);
    }

    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    std::string bytes_s = utils::formatBytes(N * sizeof(DATA_T));
    cublasHandle_t handle;
    cublasCreate(&handle);

    /* version 0*/
    // int num_threads_per_block = std::gcd(props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor);
    // int num_blocks = std::ceil((float)N / num_threads_per_block);
    /* version 1 & 2 */
    int num_threads_per_block = std::gcd(props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor);
    int num_active_blocks_per_sm = props.maxThreadsPerMultiProcessor / num_threads_per_block;
    int num_blocks_per_sm = (props.maxThreadsPerMultiProcessor / num_active_blocks_per_sm) * num_active_blocks_per_sm;
    int num_blocks = props.multiProcessorCount * num_blocks_per_sm;

    printf("BLAS `axpy` (level-1) Function\n");
    printf("- data type: %s\n", MACRO_STR(DATA_T));
    printf("- vector size: %d (%s)\n", N, bytes_s.c_str());
    printf("- #SMs: %d\n", props.multiProcessorCount);
    printf("- #threads per block: %d\n", num_threads_per_block);
    printf("- #blocks: %d\n", num_blocks);

    X_h = new DATA_T[N];
    Y_h = new DATA_T[N];
    cudaMalloc((void **)&X_d, sizeof(DATA_T) * N);
    cudaMalloc((void **)&Y_d, sizeof(DATA_T) * N);
    cudaMalloc((void **)&A_d, sizeof(DATA_T) * N);
    cudaMalloc((void **)&B_d, sizeof(DATA_T) * N);

    alpha = utils::get_random_number();
    utils::random_fill_d<DATA_T>(X_d, N);
    utils::random_fill_d<DATA_T>(Y_d, N);
    cudaMemcpyAsync(X_h, X_d, sizeof(DATA_T) * N, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(Y_h, Y_d, sizeof(DATA_T) * N, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(A_d, X_d, sizeof(DATA_T) * N, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(B_d, Y_d, sizeof(DATA_T) * N, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    float tdiff;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    axpy_h<DATA_T>(N, alpha, X_h, Y_h);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tdiff, start, stop); 
    printf("- Elasped time (axpy_h): %.6f (ms)\n", tdiff);

    cudaEventRecord(start);
    // axpy_d_v0<DATA_T><<<num_blocks, num_threads_per_block>>>(N, alpha, X_d, Y_d);
    // axpy_d_v1<DATA_T><<<num_blocks, num_threads_per_block>>>(N, alpha, X_d, Y_d);
    axpy_d_v2<DATA_T><<<num_blocks, num_threads_per_block>>>(N, alpha, X_d, Y_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tdiff, start, stop); 
    printf("- Elasped time (axpy_d): %.6f (ms)\n", tdiff);

    cudaMemcpy(X_h, Y_d, sizeof(DATA_T) * N, cudaMemcpyDeviceToHost);

    if (utils::is_equal_vector_h<DATA_T>(X_h, Y_h, N)) {
        printf("- calculation (axpy_h - axpy_d): correct\n");
    }
    else {
        printf("- calculation (axpy_h - axpy_d): incorrect\n");
        exit(1);
    }

    cudaEventRecord(start);
    cublasSaxpy(handle, N, &alpha, A_d, 1, B_d, 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tdiff, start, stop); 
    printf("- Elasped time (cuBLAS axpy): %.6f (ms)\n", tdiff);

    cudaMemcpy(X_h, B_d, sizeof(DATA_T) * N, cudaMemcpyDeviceToHost);

    if (utils::is_equal_vector_h<DATA_T>(X_h, Y_h, N)) {
        printf("- calculation (axpy_h - cuBLAS axpy): correct\n");
    }
    else {
        printf("- calculation (axpy_h - cuBLAS axpy): incorrect\n");
        exit(1);
    }

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] X_h;
    delete[] Y_h;
    cudaFree(X_d);
    cudaFree(Y_d);
    cudaFree(A_d);
    cudaFree(B_d);

    return 0;
}