#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include <omp.h>
#include <cstdio>
#include <string>
#include <random>
#include <iostream>
#include <iomanip>
#include <typeinfo>

#include "utils.hpp"

using data_t = int; 

static_assert(std::is_integral<data_t>::value, "data_t must be an integral type");
static_assert(sizeof(data_t) <= 4, "data_t must be no more than 4 bytes");

cudaEvent_t start, stop;

union AggregateDescriptor {
    struct {
        bool flag;
        data_t val;
    };

    uint64_t u64;
};

union InclusivePrefixDescriptor {
    struct {
        bool flag;
        data_t val;
    };

    uint64_t u64;
};

__global__ void check_results_kernel(const data_t * Y_d, const data_t * ans_d, int N, volatile bool* is_correct) {
    int align = 8;
    int max_len = ceilf((float)N / gridDim.x / align) * align;
    int base = max_len * blockIdx.x;
    int len = min(max_len, N - base);

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        if (*is_correct == false) {
            return;
        }
        if (Y_d[base + i] != ans_d[base + i]) {
            *is_correct = false;
            return;
        }
    }
}

__host__ bool check_results(const data_t * Y_d, const data_t * ans_d, int N) {
    bool is_correct_h = true;
    bool * is_correct_d;
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;
    cudaMalloc(&is_correct_d, sizeof(bool));
    cudaMemcpy(is_correct_d, &is_correct_h, sizeof(bool), cudaMemcpyHostToDevice); 

    check_results_kernel<<<num_blocks, num_threads>>>(Y_d, ans_d, N, is_correct_d);
    cudaDeviceSynchronize();
    cudaMemcpy(&is_correct_h, is_correct_d, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(is_correct_d);

    return is_correct_h;
}

__host__ void test_scan_openmp(const data_t * X_h, data_t * Y_h, int N) {
    data_t scan_h = 0;

    #pragma omp simd reduction(inscan, +:scan_h)
    for (int i = 0; i < N; i++) {
        Y_h[i] = scan_h;
        #pragma omp scan exclusive(scan_h)
        scan_h += X_h[i];
    }
}

__host__ void test_memcpy_d2d(const data_t * X_h, const data_t * ans_h, int N) {
    data_t * Y_d, * ans_d;
    cudaMalloc(&Y_d, N * sizeof(data_t));
    cudaMalloc(&ans_d, N * sizeof(data_t));
    cudaMemcpy(ans_d, ans_h, N * sizeof(data_t), cudaMemcpyHostToDevice);

    cudaEventRecord(start);

    cudaMemcpy(Y_d, ans_d, N * sizeof(data_t), cudaMemcpyDeviceToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t;
    cudaEventElapsedTime(&t, start, stop);

    const char * is_correct_s = check_results(Y_d, ans_d, N) ? "correct" : "incorrect";
    float throughput = (float)(N * sizeof(data_t) * 2) / (t * 1e-3) / 1e9;
    std::cout << "- Throughput (memcpy D2D): " << throughput << " GB/s"
        << " [" << is_correct_s << ", time: " << t << " ms]\n"; 

    cudaFree(Y_d);
    cudaFree(ans_d);
}

__host__ void test_scan_cub(const data_t * X_h, const data_t * ans_h, int N) {
    data_t * X_d, * Y_d, * ans_d;
    cudaMalloc(&X_d, N * sizeof(data_t));
    cudaMalloc(&Y_d, N * sizeof(data_t));
    cudaMalloc(&ans_d, N * sizeof(data_t));
    cudaMemcpy(X_d, X_h, N * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ans_d, ans_h, N * sizeof(data_t), cudaMemcpyHostToDevice);

    // Determine temporary device storage requirements & allocate storage
    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, X_d, Y_d, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaEventRecord(start);

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, X_d, Y_d, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t;
    cudaEventElapsedTime(&t, start, stop);

    const char * is_correct_s = check_results(Y_d, ans_d, N) ? "correct" : "incorrect";
    float throughput = (float)(N * sizeof(data_t) * 2) / (t * 1e-3) / 1e9;
    std::cout << "- Throughput (CUB): " << throughput << " GB/s"
        << " [" << is_correct_s << ", time: " << t << " ms]\n"; 

    cudaFree(X_d);
    cudaFree(Y_d);
    cudaFree(ans_d);
}

__global__ void scan_d_kernel(
    const data_t *              X, 
    data_t *                    Y, 
    int                         N, 
    unsigned int *              block_counter,
    AggregateDescriptor *       aggregate_descriptor_table,
    InclusivePrefixDescriptor * inclusive_prefix_descriptor_table) {
        
    __shared__ data_t Y_s[1024];
    __shared__ data_t sdata[32];
    __shared__ unsigned int bid_s;

    const int warpIdx = threadIdx.x / warpSize;
    const int laneIdx = threadIdx.x % warpSize;

    int base, len;

    do {
        /* Dynamic block index assignment */

        if (threadIdx.x == 0) {
            bid_s = atomicAdd(block_counter, 1);
        }

        __syncthreads();

        base = bid_s * warpSize * warpSize;
        len = min(warpSize * warpSize, N - base);

        if (base >= N) {
            return;
        }

        /* The Kogge-Stone algorithm (Warp level) */

        data_t aggregate = 0;

        for (int i = threadIdx.x; i < warpSize * warpSize; i += blockDim.x) {
            data_t xi = (i < len) ? X[base + i] : 0;
            data_t yi = xi;
            data_t yi_prev = 0;

            yi_prev = __shfl_up_sync(__activemask(), yi, 1);
            yi += (laneIdx >= 1) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(__activemask(), yi, 2);
            yi += (laneIdx >= 2) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(__activemask(), yi, 4);
            yi += (laneIdx >= 4) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(__activemask(), yi, 8);
            yi += (laneIdx >= 8) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(__activemask(), yi, 16);
            yi += (laneIdx >= 16) ? yi_prev : 0;

            Y_s[i] = yi - xi;

            if (laneIdx == warpSize - 1) {
                sdata[i / warpSize] = yi;
            }
        }
        
        __syncthreads();

        /* Coarsened scan (Block level) */

        if (threadIdx.x < warpSize) {
            data_t xi = sdata[threadIdx.x];
            data_t yi = xi;
            data_t yi_prev = 0;

            yi_prev = __shfl_up_sync(__activemask(), yi, 1);
            yi += (laneIdx >= 1) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(__activemask(), yi, 2);
            yi += (laneIdx >= 2) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(__activemask(), yi, 4);
            yi += (laneIdx >= 4) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(__activemask(), yi, 8);
            yi += (laneIdx >= 8) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(__activemask(), yi, 16);
            yi += (laneIdx >= 16) ? yi_prev : 0;

            sdata[threadIdx.x] = yi - xi;
            aggregate = yi;
        }

        __syncthreads();

        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            Y_s[i] += sdata[i / warpSize];
            Y[base + i] = Y_s[i]; // debug
        }

        return;

        __syncthreads();

        /* The Merrill-Garland Algorithm (Device level)  */
        // paper: https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf

        data_t pred_sum = 0;

        if (warpIdx == 0 && laneIdx == warpSize - 1) {
            union AggregateDescriptor aggr_desr;

            // Optimization: Fence-free descriptor update
            aggr_desr.flag = true;
            aggr_desr.val = aggregate;
            aggregate_descriptor_table[bid_s].u64 = aggr_desr.u64;
        }

        return;

        int inclpref_idx = -1;

        // Optimization: Parallelized look-back
        for (int k = bid_s - blockDim.x + threadIdx.x; inclpref_idx < 0; k -= blockDim.x) {
            union AggregateDescriptor pred_ag;
            union InclusivePrefixDescriptor pred_ip;

            inclpref_idx = -1;
            pred_ag.u64 = 0;
            pred_ip.u64 = 0;
            
            if (k >= 0) {
                do {
                    pred_ag.u64 = atomicAdd(
                        (unsigned long long int *)(&aggregate_descriptor_table[k].u64), 0);
                } while (pred_ag.flag == false);
                __threadfence();

                pred_ip.u64 = inclusive_prefix_descriptor_table[k].u64;
                if (pred_ip.flag) {
                    inclpref_idx = k;
                }
            }
            
            inclpref_idx = __reduce_max_sync(__activemask(), inclpref_idx);
            if (threadIdx.x % warpSize == 0)
                sdata[threadIdx.x / warpSize] = inclpref_idx;
            __syncthreads();

            if (threadIdx.x < blockDim.x / warpSize) {
                sdata[threadIdx.x] = __reduce_max_sync(__activemask(), sdata[threadIdx.x]);
            }
            __syncthreads();

            inclpref_idx = sdata[threadIdx.x / warpSize];

            // TODO: it is possible to have no predecessors with status "P". fix this.
            // Case 1: All predecessors have status "A"
            if (inclpref_idx < 0 && k - threadIdx.x > 0) {
                // Since all predecessors have status "A", we can safely add the value 
                // and continue to the next iteration. 
                pred_sum += pred_ag.val;
                continue;
            }
    
            // Case 2: At least one predecessor has status "P"
            if (k > inclpref_idx) 
                pred_sum += pred_ag.val;
            else if (k == inclpref_idx) 
                pred_sum += pred_ip.val;
                
            pred_sum = __reduce_add_sync(__activemask(), pred_sum);
            if (threadIdx.x % warpSize == 0)
                sdata[threadIdx.x / warpSize] = pred_sum;
            __syncthreads();

            if (threadIdx.x < blockDim.x / warpSize) {
                sdata[threadIdx.x] = __reduce_add_sync(__activemask(), sdata[threadIdx.x]);
            }
            __syncthreads();

            pred_sum = sdata[threadIdx.x / warpSize];

            for (int i = threadIdx.x; i < len; i += blockDim.x) {
                Y_s[i] += pred_sum;
                Y[base + i] = Y_s[i];
            }

            inclpref_idx = 0;
        }
        
        __syncthreads();

        if (threadIdx.x == 0) {
            union InclusivePrefixDescriptor ip_desr;

            ip_desr.flag = true;
            ip_desr.val = Y_s[len - 1];
            inclusive_prefix_descriptor_table[bid_s].u64 = ip_desr.u64;
        }

    } while (true);
}

template<typename T>
__host__ void malloc_persisting_data(T ** ptr_addr, unsigned int n, float hit_ratio = 1.0f) {
    cudaMalloc(ptr_addr, n * sizeof(T));
    cudaMemset(*ptr_addr, 0, n * sizeof(T));

    cudaStreamAttrValue stream_attribute;   
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr_addr);
    stream_attribute.accessPolicyWindow.num_bytes = n * sizeof(T);
    stream_attribute.accessPolicyWindow.hitRatio  = hit_ratio;
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(cudaStreamDefault, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
}

__host__ void test_scan_handcraft(const data_t * X_h, const data_t * ans_h, int N) {
    data_t * X_d, * Y_d, * ans_d;
    cudaMalloc(&X_d, N * sizeof(data_t));
    cudaMalloc(&Y_d, N * sizeof(data_t));
    cudaMalloc(&ans_d, N * sizeof(data_t));
    cudaMemcpy(X_d, X_h, N * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ans_d, ans_h, N * sizeof(data_t), cudaMemcpyHostToDevice);

    // `num_threads` should be no more than 1024 (warpSize * warpSize) and multiple of 32 (warpSize)
    cudaDeviceProp prop = utils::get_device_properties();
    int num_threads = 256; 
    int num_blocks = prop.maxBlocksPerMultiProcessor * prop.multiProcessorCount;
    int partition_size = prop.warpSize * prop.warpSize;
    int num_partitions = (N + partition_size - 1) / partition_size;

    unsigned int * block_counter_d;
    union AggregateDescriptor * aggregate_descriptor_table_d;
    union InclusivePrefixDescriptor * inclusive_prefix_descriptor_table_d;
     
    malloc_persisting_data<AggregateDescriptor>(&aggregate_descriptor_table_d, num_partitions);
    malloc_persisting_data<InclusivePrefixDescriptor>(&inclusive_prefix_descriptor_table_d, num_partitions);
    malloc_persisting_data<unsigned int>(&block_counter_d, 1);

    cudaEventRecord(start);

    scan_d_kernel<<<num_blocks, num_threads>>>(
        X_d, 
        Y_d, 
        N, 
        block_counter_d, 
        aggregate_descriptor_table_d, 
        inclusive_prefix_descriptor_table_d
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t;
    cudaEventElapsedTime(&t, start, stop);

    // debug
    data_t * Y_h = new data_t[N];
    int * block_counter_h = new int[1];
    cudaMemcpy(Y_h, Y_d, N * sizeof(data_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_counter_h, block_counter_d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("num_threads: %d\n", num_threads);
    printf("num_blocks: %d\n", num_blocks);
    printf("partition_size: %d\n", partition_size);
    printf("num_partitions: %d\n", num_partitions);
    printf("block_counter: %d\n", *block_counter_h);
    for (int i = 0; i < 1024; i++) {
        std::cout << Y_h[i] << " ";
        if (i % 32 == 31) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;


    const char * is_correct_s = check_results(Y_d, ans_d, N) ? "correct" : "incorrect";
    float throughput = (float)(N * sizeof(data_t) * 2) / (t * 1e-3) / 1e9;
    std::cout << "- Throughput (handcraft): " << throughput << " GB/s"
        << " [" << is_correct_s << ", time: " << t << " ms]\n"; 

    cudaFree(X_d);
    cudaFree(Y_d);
    cudaFree(ans_d);
    cudaFree(block_counter_d);
    cudaFree(aggregate_descriptor_table_d);
    cudaFree(inclusive_prefix_descriptor_table_d);
}

__host__ void print_info(int N) {
    cudaDeviceProp props = utils::get_device_properties();
    std::string bytes_s = utils::formatBytes(N * sizeof(data_t));
    std::string type_s = utils::get_type_name<data_t>();
    
    float bandwidth = (float)(2 * props.memoryClockRate * 1e3 * props.memoryBusWidth) / 8 / 1e9;
    
    printf("Test `exclusive scan` parallel algorithms\n");
    printf("- data type: %s\n", type_s.c_str());;
    printf("- vector size: %d (%s)\n", N, bytes_s.c_str());
    printf("- #SMs: %d\n", props.multiProcessorCount);
    printf("- bandwidth: %.3f GB/s\n", bandwidth);
    printf("- L2 Cache Size: %d bytes\n", props.l2CacheSize);
    printf("- Persisting L2 Cache Max Size: %d bytes\n", props.persistingL2CacheMaxSize);
}

__host__ int main(int argc, char ** argv) {
    int N;

    if (argc > 1) {
        N = std::stoi(argv[1]);
    }
    else {
        perror("Usage: ./scan <vector size>\n");
        exit(1);
    }

    print_info(N);

    data_t * X_h = new data_t[N];
    data_t * Y_h = new data_t[N];
    utils::random_fill_h<data_t>(X_h, N, 0, 10); 
    test_scan_openmp(X_h, Y_h, N);

    // for(int i = 0; i < 256; i++) {
    //     std::cout << X_h[i] << " ";
    //     if (i % 32 == 31) {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    for(int i = 0; i <  1024; i++) {
        std::cout << Y_h[i] << " ";
        if (i % 32 == 31) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // test_scan_openmp(X_h, Y_h, N);
    test_memcpy_d2d(X_h, Y_h, N);
    test_scan_cub(X_h, Y_h, N);
    test_scan_handcraft(X_h, Y_h, N);
 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] X_h;
    delete[] Y_h;

    return 0;
}