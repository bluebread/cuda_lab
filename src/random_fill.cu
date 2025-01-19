#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h> 
#include <omp.h>

#include "utils.hpp"

#define NUM_ITERATION   10

#ifndef DATA_T
#define DATA_T          float
#endif
#define STRINGIFY(s)    #s
#define MACRO_STR(m)   STRINGIFY(m)

int main(int argc, const char * argv[]) {
    DATA_T * X_h, * X_d;
    int N;

    bool is_data_type_valid = 
        std::is_same<DATA_T, float>::value || std::is_same<DATA_T, double>::value;

    if (argc > 1 && is_data_type_valid) {
        N = std::stoi(argv[1]);
    }
    else {
        perror("Usage: ./fma_vector <vector size>\n");
        exit(1);
    }

    cudaDeviceProp prop = utils::get_device_properties();
    int num_threads_per_block = std::gcd(prop.maxThreadsPerBlock, prop.maxThreadsPerMultiProcessor);
    // why 4? Increasing the number of blocks per SM can hide latency. 
    // Larger number is also better for throughput, especially for longer inputs.
    int num_blocks_per_sm = (prop.maxThreadsPerMultiProcessor / num_threads_per_block) * 4;
    int num_blocks = prop.multiProcessorCount * num_blocks_per_sm;

    std::string bytes_s = utils::formatBytes(N * sizeof(DATA_T));

    printf("Filling Vector with Random Values\n");
    printf("- data type: %s\n", MACRO_STR(DATA_T));
    printf("- vector size: %d (%s)\n", N, bytes_s.c_str());
    printf("- #blocks: %d\n", num_blocks);
    printf("- #threads per block: %d\n", num_threads_per_block);

    if (num_blocks * num_threads_per_block > N) {
        perror("The vector size is too small for the current device configuration.\n");
        exit(1);
    }

    X_h = new DATA_T[N];

    cudaMalloc((void **)&X_d, sizeof(DATA_T) * N);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    utils::random_fill_d<DATA_T>(X_d, N);
    cudaMemcpyAsync(X_h, X_d, sizeof(DATA_T) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double total_time = 0.0;

    for (int i = 0; i < NUM_ITERATION; i++) {
        cudaEventRecord(start);

        utils::random_fill_d<DATA_T>(X_d, N);
        // utils::random_fill_h<DATA_T>(X_h, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float t;
        cudaEventElapsedTime(&t, start, stop); 

        total_time += t;
    }

    printf("Average time: %.6f (ms)\n", total_time / NUM_ITERATION);

    delete[] X_h;
    cudaFree(X_d);

    return 0;
}