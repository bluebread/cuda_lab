#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/universal_ptr.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <omp.h>
#include <cstdio>
#include <string>
#include <random>
#include <iostream>
#include <iomanip>
#include <typeinfo>

#include "utils.hpp"

using data_t = int; // float or int

cudaEvent_t start, stop;

__host__ void test_reduce_openmp(const data_t * X_h, int N) {
    data_t sum = 0.0;
    cudaEventRecord(start);
    #pragma omp parallel for reduction(+:sum) 
    for (int i = 0; i < N; i++) {
        sum += X_h[i];
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t = 0;
    cudaEventElapsedTime(&t, start, stop);
    std::cout << "- Elasped time (OpenMP): " << t << " (ms) "
        << "[ret. " << std::fixed << std::setprecision(6) << sum << "]\n";
}

__host__ void test_reduce_thrust(const data_t * X_h, int N) {
    // Transfer to device and compute the sum.
    data_t * X_d;
    cudaMalloc(&X_d, N * sizeof(data_t));
    cudaMemcpy(X_d, X_h, N * sizeof(data_t), cudaMemcpyHostToDevice);
    data_t zero = 0;

    cudaEventRecord(start);
    data_t x = thrust::reduce(thrust::device, X_d, X_d + N, zero);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t = 0;
    cudaEventElapsedTime(&t, start, stop);
    std::cout << "- Elasped time (Thrust): " << t << " (ms) "
        << "[ret. " << std::fixed << std::setprecision(6) << x << "]\n";

    cudaFree(X_d);
}

__host__ void test_reduce_cub(const data_t * X_h, int N) {
    // Transfer to device and compute the sum.
    data_t * X_d, * result_d;
    cudaMalloc(&X_d, N * sizeof(data_t));
    cudaMalloc(&result_d, sizeof(data_t));
    cudaMemcpy(X_d, X_h, N * sizeof(data_t), cudaMemcpyHostToDevice);

    void * d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, X_d, result_d, N);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cudaEventRecord(start);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, X_d, result_d, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t = 0;
    data_t result_h;
    cudaEventElapsedTime(&t, start, stop);
    cudaMemcpy(&result_h, result_d, sizeof(data_t), cudaMemcpyDeviceToHost);

    std::cout << "- Elasped time (CUB): " << t << " (ms) "
        << "[ret. " << std::fixed << std::setprecision(6) << result_h 
        << ", temp. storage " << temp_storage_bytes << " bytes]\n";

    cudaFree(X_d);
    cudaFree(result_d);
    cudaFree(d_temp_storage);
}

__global__ void reduce_d_v0(const data_t * X, data_t * result, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(result, X[i]);
    }
}

__global__ void reduce_d_v1(const data_t * X, data_t * result, int N) {
    extern __shared__ data_t sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[threadIdx.x] = (idx < N) ? X[idx] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, sdata[0]);
    }
}

__global__ void reduce_d_v2(const data_t * X, data_t * result, int N) {
    extern __shared__ data_t sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    data_t sum = (i < N) ? X[i] : 0;

    sum += __shfl_down_sync(__activemask(), sum, 16);
    sum += __shfl_down_sync(__activemask(), sum, 8);
    sum += __shfl_down_sync(__activemask(), sum, 4);
    sum += __shfl_down_sync(__activemask(), sum, 2);
    sum += __shfl_down_sync(__activemask(), sum, 1);

    if (threadIdx.x % 32 == 0) {
        sdata[threadIdx.x / 32] = sum;
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x / 32) {
        sum = sdata[threadIdx.x];
        
        sum += __shfl_down_sync(__activemask(), sum, 16);
        sum += __shfl_down_sync(__activemask(), sum, 8);
        sum += __shfl_down_sync(__activemask(), sum, 4);
        sum += __shfl_down_sync(__activemask(), sum, 2);
        sum += __shfl_down_sync(__activemask(), sum, 1);
        
        if (threadIdx.x == 0) {
            atomicAdd(result, sum);
        }
    }
}

__global__ void reduce_d_v3(const data_t * X, data_t * out, int N) {
    extern __shared__ data_t sdata[];

    int grid_size = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data_t sum = 0;

    for (int j = i; j < N; j += grid_size) {
        sum += X[j];
    }

    if (std::is_integral<data_t>::value) {
        sum = __reduce_add_sync(__activemask(), (int)sum);
    }
    else {
        sum += __shfl_down_sync(__activemask(), sum, 16);
        sum += __shfl_down_sync(__activemask(), sum, 8);
        sum += __shfl_down_sync(__activemask(), sum, 4);
        sum += __shfl_down_sync(__activemask(), sum, 2);
        sum += __shfl_down_sync(__activemask(), sum, 1);
    }

    if (threadIdx.x % 32 == 0) {
        sdata[threadIdx.x / 32] = sum;
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x / 32) {
        sum = sdata[threadIdx.x];

        if (std::is_integral<data_t>::value) {
            sum = __reduce_add_sync(__activemask(), (int)sum);
        }
        else {
            sum += __shfl_down_sync(__activemask(), sum, 16);
            sum += __shfl_down_sync(__activemask(), sum, 8);
            sum += __shfl_down_sync(__activemask(), sum, 4);
            sum += __shfl_down_sync(__activemask(), sum, 2);
            sum += __shfl_down_sync(__activemask(), sum, 1);
        }
        
        if (threadIdx.x == 0) {
            out[blockIdx.x] = sum;
        }
    }
}

__host__ void test_reduce_handcraft(const data_t * X_h, int N) {
    data_t * X_d;
    cudaMalloc(&X_d, N * sizeof(data_t));
    cudaMemcpy(X_d, X_h, N * sizeof(data_t), cudaMemcpyHostToDevice);

    data_t * result_d;
    data_t result_h = 0;
    cudaMalloc(&result_d, sizeof(data_t));
    cudaMemcpy(result_d, &result_h, sizeof(data_t), cudaMemcpyHostToDevice);

    int num_threads_per_block, num_blocks, shared_memory_size;
    const char * version;
    cudaDeviceProp props = utils::get_device_properties();

    data_t * temp_storage;
    cudaMalloc(&temp_storage, props.multiProcessorCount * props.maxBlocksPerMultiProcessor * sizeof(data_t));

    cudaEventRecord(start);

    /* version 0 */
    // int num_threads_per_block = 256;
    // int num_blocks = (N + num_threads - 1) / num_threads;
    // const char * version = "v0";
    // reduce_d_v0<<<num_blocks, num_threads>>>(X_d, result_d, N);

    /* version 1 */
    // num_threads_per_block = std::gcd(props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor);
    // num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;
    // shared_memory_size = num_threads_per_block * sizeof(data_t);
    // version = "v1";
    // reduce_d_v1<<<num_blocks, num_threads_per_block, shared_memory_size>>>(X_d, result_d, N);

    /* version 2 */
    // num_threads_per_block = std::gcd(props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor);
    // num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;
    // shared_memory_size = num_threads_per_block * sizeof(data_t) / 32;
    // version = "v2";
    // reduce_d_v2<<<num_blocks, num_threads_per_block, shared_memory_size>>>(X_d, result_d, N);

    /* version 3 */
    num_threads_per_block = props.maxThreadsPerBlock;
    num_blocks = props.maxThreadsPerBlock;
    shared_memory_size = num_threads_per_block * sizeof(data_t) / 32;
    version = "v3";
    reduce_d_v3<<<num_blocks, num_threads_per_block, shared_memory_size>>>(X_d, temp_storage, N);
    reduce_d_v3<<<1, num_threads_per_block, shared_memory_size>>>(temp_storage, result_d, num_blocks);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t = 0;
    cudaEventElapsedTime(&t, start, stop);
    cudaMemcpy(&result_h, result_d, sizeof(data_t), cudaMemcpyDeviceToHost);

    std::cout << "- Elasped time (handcraft " << version << "): " << t << " (ms) ["
              << "ret. "  << std::fixed << std::setprecision(6) << result_h << ", "
              << "#threads " << num_threads_per_block << ", "
              << "#blocks " << num_blocks << ", "
              << "shared mem. " << shared_memory_size << " bytes/block]\n";

    cudaFree(X_d);
    cudaFree(result_d);
    cudaFree(temp_storage);
}

__host__ void print_info(int N) {
    cudaDeviceProp props = utils::get_device_properties();
    std::string bytes_s = utils::formatBytes(N * sizeof(data_t));
    std::string type_s = utils::get_type_name<data_t>();
    
    printf("Test `reduce` parallel algorithms\n");
    printf("- data type: %s\n", type_s.c_str());
    printf("- vector size: %d (%s)\n", N, bytes_s.c_str());
    printf("- #SMs: %d\n", props.multiProcessorCount);
}

int main(int argc, const char * argv[]) {
    int N;

    if (argc > 1) {
        N = std::stoi(argv[1]);
    }
    else {
        perror("Usage: ./reduce <vector size>\n");
        exit(1);
    }

    print_info(N);

    data_t * X_h = new data_t[N];

    if (std::is_integral<data_t>::value) {
        data_t max_v = std::numeric_limits<data_t>::max();
        utils::random_fill_h<data_t>(X_h, N, - max_v / 3, max_v / 3); 
    }
    else {
        utils::random_fill_h<data_t>(X_h, N, -1.0, 1.0); 
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    test_reduce_openmp(X_h, N);
    test_reduce_thrust(X_h, N);
    test_reduce_cub(X_h, N);
    test_reduce_handcraft(X_h, N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}