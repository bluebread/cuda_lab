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

union Descriptor {
    enum class State {
        Invalid = 0,
        AggregateAvailable,
        PrefixAvailable
    };

    struct {
        enum State state;
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
            int idx = threadIdx.x + blockIdx.x * blockDim.x;    
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

    cudaEventRecord(start);

    #pragma omp simd reduction(inscan, +:scan_h)
    for (int i = 0; i < N; i++) {
        Y_h[i] = scan_h;
        #pragma omp scan exclusive(scan_h)
        scan_h += X_h[i];
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t;
    cudaEventElapsedTime(&t, start, stop);

    std::cout << "- Elasped time (OpenMP): " << t << " ms\n";
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
        << " [" << is_correct_s << ", time: " << t << " ms, " 
        << "temp. space " << utils::formatBytes(temp_storage_bytes) << "]\n"; 

    cudaFree(X_d);
    cudaFree(Y_d);
    cudaFree(ans_d);
}

__device__ __forceinline__ data_t warp_exclusive_scan(data_t val) {
    const int warpIdx = threadIdx.x / warpSize;
    const int laneIdx = threadIdx.x % warpSize;

    data_t yi = val;
    data_t yi_prev = 0;

    /* The Kogge-Stone algorithm */

    yi_prev = __shfl_up_sync(0xffffffff, yi, 1);
    yi += (laneIdx >= 1) ? yi_prev : 0;
    yi_prev = __shfl_up_sync(0xffffffff, yi, 2);
    yi += (laneIdx >= 2) ? yi_prev : 0;
    yi_prev = __shfl_up_sync(0xffffffff, yi, 4);
    yi += (laneIdx >= 4) ? yi_prev : 0;
    yi_prev = __shfl_up_sync(0xffffffff, yi, 8);
    yi += (laneIdx >= 8) ? yi_prev : 0;
    yi_prev = __shfl_up_sync(0xffffffff, yi, 16);
    yi += (laneIdx >= 16) ? yi_prev : 0;

    return yi - val;
}

template<typename T>
__device__ __forceinline__ data_t block_reduce_max(T val, T * tmp_space) {
    const int warpIdx = threadIdx.x / warpSize;
    const int laneIdx = threadIdx.x % warpSize;
    const int num_warps = blockDim.x / warpSize;
    
    val = __reduce_max_sync(0xffffffff, val);

    if (laneIdx == 0)
        tmp_space[warpIdx] = val;

    __syncthreads();

    if (threadIdx.x < warpSize) {
        T temp = (threadIdx.x < num_warps) ? tmp_space[threadIdx.x] : -1;
        tmp_space[threadIdx.x] = __reduce_max_sync(0xffffffff, temp);
    }

    __syncthreads();

    return tmp_space[warpIdx];
}

template<typename T>
__device__ __forceinline__ data_t block_reduce_add(T val, T * tmp_space) {
    const int warpIdx = threadIdx.x / warpSize;
    const int laneIdx = threadIdx.x % warpSize;
    const int num_warps = blockDim.x / warpSize;
    
    val = __reduce_add_sync(0xffffffff, val);

    if (laneIdx == 0)
        tmp_space[warpIdx] = val;

    __syncthreads();

    if (threadIdx.x < warpSize) {
        T temp = (threadIdx.x < num_warps) ? tmp_space[threadIdx.x] : 0;
        tmp_space[threadIdx.x] = __reduce_add_sync(0xffffffff, temp);
    }

    __syncthreads();

    return tmp_space[warpIdx];
}

__global__ void scan_d_kernel_v2(
    const data_t *      X, 
    data_t *            Y, 
    int                 N, 
    unsigned int *      block_counter,
    Descriptor *        descriptor_table) {

    // In this kernel, we use the Kogge-Stone algorithm for the warp level scan,
    // the coarsened scan for the block level scan, and the Merrill-Garland algorithm
    // for the device level scan.

    // The number of threads per block is fixed to 256.
    // Each thread holds 16 elements, so the partition size is 16 * 256 = 4096.
    const int partition_size = 4 * warpSize * warpSize;

    // The eight of the elemtns are saved in the shared memory, and the rest are saved in the registers.
    __shared__ data_t Y_s[2048];
    __shared__ data_t sdata[128]; // used for saving the sum of the elements in the warp
    __shared__ unsigned int bid_s; // dynamic block index
    
    const int warpIdx = threadIdx.x / warpSize;
    const int laneIdx = threadIdx.x % warpSize;
    const int num_warps = blockDim.x / warpSize;
    
    int base, len;

    // Get the first block id
    if (threadIdx.x == 0) {
        bid_s = atomicAdd(block_counter, 1);
    }

    __syncthreads();

    base = bid_s * partition_size;
    len = min(partition_size, N - base);

    while (base < N) {

        /* Load data in registers */

        data_t x0 = (threadIdx.x + 0 * blockDim.x < len) ? X[base + threadIdx.x + 0 * blockDim.x] : 0;
        data_t x1 = (threadIdx.x + 1 * blockDim.x < len) ? X[base + threadIdx.x + 1 * blockDim.x] : 0;
        data_t x2 = (threadIdx.x + 2 * blockDim.x < len) ? X[base + threadIdx.x + 2 * blockDim.x] : 0;
        data_t x3 = (threadIdx.x + 3 * blockDim.x < len) ? X[base + threadIdx.x + 3 * blockDim.x] : 0;
        data_t x4 = (threadIdx.x + 4 * blockDim.x < len) ? X[base + threadIdx.x + 4 * blockDim.x] : 0;
        data_t x5 = (threadIdx.x + 5 * blockDim.x < len) ? X[base + threadIdx.x + 5 * blockDim.x] : 0;
        data_t x6 = (threadIdx.x + 6 * blockDim.x < len) ? X[base + threadIdx.x + 6 * blockDim.x] : 0;
        data_t x7 = (threadIdx.x + 7 * blockDim.x < len) ? X[base + threadIdx.x + 7 * blockDim.x] : 0;
        data_t x8 = (threadIdx.x + 8 * blockDim.x < len) ? X[base + threadIdx.x + 8 * blockDim.x] : 0;
        data_t x9 = (threadIdx.x + 9 * blockDim.x < len) ? X[base + threadIdx.x + 9 * blockDim.x] : 0;
        data_t x10 = (threadIdx.x + 10 * blockDim.x < len) ? X[base + threadIdx.x + 10 * blockDim.x] : 0;
        data_t x11 = (threadIdx.x + 11 * blockDim.x < len) ? X[base + threadIdx.x + 11 * blockDim.x] : 0;
        data_t x12 = (threadIdx.x + 12 * blockDim.x < len) ? X[base + threadIdx.x + 12 * blockDim.x] : 0;
        data_t x13 = (threadIdx.x + 13 * blockDim.x < len) ? X[base + threadIdx.x + 13 * blockDim.x] : 0;
        data_t x14 = (threadIdx.x + 14 * blockDim.x < len) ? X[base + threadIdx.x + 14 * blockDim.x] : 0;
        data_t x15 = (threadIdx.x + 15 * blockDim.x < len) ? X[base + threadIdx.x + 15 * blockDim.x] : 0;

        data_t y0, y1, y2, y3, y4, y5, y6, y7;
        data_t & y8 = Y_s[threadIdx.x + 0 * blockDim.x];
        data_t & y9 = Y_s[threadIdx.x + 1 * blockDim.x];
        data_t & y10 = Y_s[threadIdx.x + 2 * blockDim.x];
        data_t & y11 = Y_s[threadIdx.x + 3 * blockDim.x];
        data_t & y12 = Y_s[threadIdx.x + 4 * blockDim.x];
        data_t & y13 = Y_s[threadIdx.x + 5 * blockDim.x];
        data_t & y14 = Y_s[threadIdx.x + 6 * blockDim.x];
        data_t & y15 = Y_s[threadIdx.x + 7 * blockDim.x];

        /* The Kogge-Stone algorithm (Warp level) */

        y0 = warp_exclusive_scan(x0);
        y1 = warp_exclusive_scan(x1);
        y2 = warp_exclusive_scan(x2);
        y3 = warp_exclusive_scan(x3);
        y4 = warp_exclusive_scan(x4);
        y5 = warp_exclusive_scan(x5);
        y6 = warp_exclusive_scan(x6);
        y7 = warp_exclusive_scan(x7);
        y8 = warp_exclusive_scan(x8);
        y9 = warp_exclusive_scan(x9);
        y10 = warp_exclusive_scan(x10);
        y11 = warp_exclusive_scan(x11);
        y12 = warp_exclusive_scan(x12);
        y13 = warp_exclusive_scan(x13);
        y14 = warp_exclusive_scan(x14);
        y15 = warp_exclusive_scan(x15);

        if (laneIdx == warpSize - 1) {
            // Save the sum of the elements in the warp
            sdata[warpIdx + 0 * num_warps] = x0 + y0;
            sdata[warpIdx + 1 * num_warps] = x1 + y1;
            sdata[warpIdx + 2 * num_warps] = x2 + y2;
            sdata[warpIdx + 3 * num_warps] = x3 + y3;
            sdata[warpIdx + 4 * num_warps] = x4 + y4;
            sdata[warpIdx + 5 * num_warps] = x5 + y5;
            sdata[warpIdx + 6 * num_warps] = x6 + y6;
            sdata[warpIdx + 7 * num_warps] = x7 + y7;
            sdata[warpIdx + 8 * num_warps] = x8 + y8;
            sdata[warpIdx + 9 * num_warps] = x9 + y9;
            sdata[warpIdx + 10 * num_warps] = x10 + y10;
            sdata[warpIdx + 11 * num_warps] = x11 + y11;
            sdata[warpIdx + 12 * num_warps] = x12 + y12;
            sdata[warpIdx + 13 * num_warps] = x13 + y13;
            sdata[warpIdx + 14 * num_warps] = x14 + y14;
            sdata[warpIdx + 15 * num_warps] = x15 + y15;
        }

        __syncthreads();

        /* Coarsened scan (Block level) */

        data_t aggregate = 0;

        if (threadIdx.x < warpSize) {
            for (int i = 0; i < partition_size / (warpSize * warpSize); i++) {
                data_t xi = sdata[threadIdx.x + i * warpSize];
                data_t yi = warp_exclusive_scan(xi);
                data_t s = __shfl_sync(0xffffffff, xi + yi, warpSize - 1); // broadcast
                sdata[threadIdx.x + i * warpSize] = yi + aggregate;
                aggregate += s;
            }
        }

        __syncthreads();

        y0 += sdata[warpIdx + 0 * num_warps];
        y1 += sdata[warpIdx + 1 * num_warps];
        y2 += sdata[warpIdx + 2 * num_warps];
        y3 += sdata[warpIdx + 3 * num_warps];
        y4 += sdata[warpIdx + 4 * num_warps];
        y5 += sdata[warpIdx + 5 * num_warps];
        y6 += sdata[warpIdx + 6 * num_warps];
        y7 += sdata[warpIdx + 7 * num_warps];
        y8 += sdata[warpIdx + 8 * num_warps];
        y9 += sdata[warpIdx + 9 * num_warps];
        y10 += sdata[warpIdx + 10 * num_warps];
        y11 += sdata[warpIdx + 11 * num_warps];
        y12 += sdata[warpIdx + 12 * num_warps];
        y13 += sdata[warpIdx + 13 * num_warps];
        y14 += sdata[warpIdx + 14 * num_warps];
        y15 += sdata[warpIdx + 15 * num_warps];

        __syncthreads();

        /* The Merrill-Garland Algorithm (Device level)  */
        // paper: https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf

        data_t pred_sum = 0;
        int inclpref_idx = -1;

        if (threadIdx.x == warpSize - 1) {
            union Descriptor desr;

            // Optimization: Fence-free descriptor update
            desr.state = Descriptor::State::AggregateAvailable;
            desr.val = aggregate;
            descriptor_table[bid_s].u64 = desr.u64;
        }

        // Optimization: Parallelized look-back
        // "inclpref_idx < 0" indicates that there is no predecessor with status "P"
        for (int k = bid_s - blockDim.x + threadIdx.x; inclpref_idx < 0 && (int)(k - threadIdx.x + blockDim.x) > 0; k -= blockDim.x) {
            union Descriptor pred_desr;

            inclpref_idx = -1;
            pred_desr.u64 = 0;
            
            if (k >= 0) {
                // spinning until the predecessor's descriptor is valid
                do {
                    pred_desr.u64 = atomicAdd(
                        (unsigned long long int *)(&descriptor_table[k].u64), 0);
                } while (pred_desr.state == Descriptor::State::Invalid);
                __threadfence();

                if (pred_desr.state == Descriptor::State::PrefixAvailable) {
                    inclpref_idx = k;
                }
            }

            // the rightmost predecessor with status "P"
            inclpref_idx = block_reduce_max(inclpref_idx, sdata);
            
            // Case 1: All predecessors have status "A"
            if (inclpref_idx < 0 && (int)(k - threadIdx.x) > 0) {
                // Since all predecessors have status "A", we can safely add the value 
                // and continue to the next iteration. 
                pred_sum += pred_desr.val;
                continue;
            }

            // Case 2: At least one predecessor has status "P"
            if (k >= inclpref_idx) 
                // incorporate the value of the predecessor with status "A"
                pred_sum += pred_desr.val;

            // compute the partition-wide inclusive prefix sum
            pred_sum = block_reduce_add(pred_sum, sdata);

            break;
        }

        /* Save results */

        if (threadIdx.x + 0 * blockDim.x < len)
            Y[base + threadIdx.x + 0 * blockDim.x] = y0 + pred_sum;
        if (threadIdx.x + 1 * blockDim.x < len)
            Y[base + threadIdx.x + 1 * blockDim.x] = y1 + pred_sum;
        if (threadIdx.x + 2 * blockDim.x < len)
            Y[base + threadIdx.x + 2 * blockDim.x] = y2 + pred_sum;
        if (threadIdx.x + 3 * blockDim.x < len)
            Y[base + threadIdx.x + 3 * blockDim.x] = y3 + pred_sum;
        if (threadIdx.x + 4 * blockDim.x < len)
            Y[base + threadIdx.x + 4 * blockDim.x] = y4 + pred_sum;
        if (threadIdx.x + 5 * blockDim.x < len)
            Y[base + threadIdx.x + 5 * blockDim.x] = y5 + pred_sum;
        if (threadIdx.x + 6 * blockDim.x < len)
            Y[base + threadIdx.x + 6 * blockDim.x] = y6 + pred_sum;
        if (threadIdx.x + 7 * blockDim.x < len)
            Y[base + threadIdx.x + 7 * blockDim.x] = y7 + pred_sum;
        if (threadIdx.x + 8 * blockDim.x < len)
            Y[base + threadIdx.x + 8 * blockDim.x] = y8 + pred_sum;
        if (threadIdx.x + 9 * blockDim.x < len)
            Y[base + threadIdx.x + 9 * blockDim.x] = y9 + pred_sum;
        if (threadIdx.x + 10 * blockDim.x < len)
            Y[base + threadIdx.x + 10 * blockDim.x] = y10 + pred_sum;
        if (threadIdx.x + 11 * blockDim.x < len)
            Y[base + threadIdx.x + 11 * blockDim.x] = y11 + pred_sum;
        if (threadIdx.x + 12 * blockDim.x < len)
            Y[base + threadIdx.x + 12 * blockDim.x] = y12 + pred_sum;
        if (threadIdx.x + 13 * blockDim.x < len)
            Y[base + threadIdx.x + 13 * blockDim.x] = y13 + pred_sum;
        if (threadIdx.x + 14 * blockDim.x < len)
            Y[base + threadIdx.x + 14 * blockDim.x] = y14 + pred_sum;
        if (threadIdx.x + 15 * blockDim.x < len)
            Y[base + threadIdx.x + 15 * blockDim.x] = y15 + pred_sum;

        /* Save prefix sum in the descriptor table */

        if (threadIdx.x == warpSize - 1) {
            union Descriptor desr;

            desr.state = Descriptor::State::PrefixAvailable;
            desr.val = pred_sum + aggregate;
            descriptor_table[bid_s].u64 = desr.u64;
        }

        /* get next block id */

        if (threadIdx.x == 0) {
            bid_s = atomicAdd(block_counter, 1);
        }

        __syncthreads();

        base = bid_s * partition_size;
        len = min(partition_size, N - base);
    }
}

__global__ void scan_d_kernel_v1(
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

            yi_prev = __shfl_up_sync(0xffffffff, yi, 1);
            yi += (laneIdx >= 1) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(0xffffffff, yi, 2);
            yi += (laneIdx >= 2) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(0xffffffff, yi, 4);
            yi += (laneIdx >= 4) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(0xffffffff, yi, 8);
            yi += (laneIdx >= 8) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(0xffffffff, yi, 16);
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

            yi_prev = __shfl_up_sync(0xffffffff, yi, 1);
            yi += (laneIdx >= 1) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(0xffffffff, yi, 2);
            yi += (laneIdx >= 2) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(0xffffffff, yi, 4);
            yi += (laneIdx >= 4) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(0xffffffff, yi, 8);
            yi += (laneIdx >= 8) ? yi_prev : 0;
            yi_prev = __shfl_up_sync(0xffffffff, yi, 16);
            yi += (laneIdx >= 16) ? yi_prev : 0;

            sdata[threadIdx.x] = yi - xi;
            aggregate = yi;
        }

        __syncthreads();

        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            Y_s[i] += sdata[i / warpSize];
        }

        __syncthreads();

        /* The Merrill-Garland Algorithm (Device level)  */
        // paper: https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf

        data_t pred_sum = 0;

        if (threadIdx.x == warpSize - 1) {
            union AggregateDescriptor aggr_desr;

            // Optimization: Fence-free descriptor update
            aggr_desr.flag = true;
            aggr_desr.val = aggregate;
            aggregate_descriptor_table[bid_s].u64 = aggr_desr.u64;
        }

        int inclpref_idx = -1;

        // Optimization: Parallelized look-back
        for (int k = bid_s - blockDim.x + threadIdx.x; inclpref_idx < 0 && (int)(k - threadIdx.x + blockDim.x) > 0; k -= blockDim.x) {
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
                if (pred_ip.flag == true) {
                    inclpref_idx = k;
                }
            }

            inclpref_idx = __reduce_max_sync(0xffffffff, inclpref_idx);
            if (laneIdx == 0)
                sdata[warpIdx] = inclpref_idx;
            __syncthreads();
            if (threadIdx.x < warpSize) {
                int temp = (threadIdx.x < blockDim.x / warpSize) ? sdata[threadIdx.x] : -1;
                sdata[threadIdx.x] = __reduce_max_sync(0xffffffff, temp);
            }
            __syncthreads();
            inclpref_idx = sdata[warpIdx];

            // Case 1: All predecessors have status "A"
            if (inclpref_idx < 0 && (int)(k - threadIdx.x) > 0) {
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
                
            pred_sum = __reduce_add_sync(0xffffffff, pred_sum);
            if (laneIdx == 0)
                sdata[warpIdx] = pred_sum;
            __syncthreads();
            if (threadIdx.x < warpSize) {
                int temp = (threadIdx.x < blockDim.x / warpSize) ? sdata[threadIdx.x] : 0;
                sdata[threadIdx.x] = __reduce_add_sync(0xffffffff, temp);
            }
            __syncthreads();
            pred_sum = sdata[warpIdx];

            inclpref_idx = 0; // equivalent to break
        }
        
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            Y[base + i] = Y_s[i] + pred_sum; 
        }

        if (threadIdx.x == warpSize - 1) {
            union InclusivePrefixDescriptor ip_desr;

            ip_desr.flag = true;
            ip_desr.val = pred_sum + aggregate;
            inclusive_prefix_descriptor_table[bid_s].u64 = ip_desr.u64;
        }

    } while (true);
}

template<typename T>
__host__ void malloc_persisting_data(T ** ptr_addr, unsigned int n, float hit_ratio = 1.0f) {
    cudaMalloc(ptr_addr, n * sizeof(T));
    cudaMemset(*ptr_addr, 0, n * sizeof(T));

    cudaStreamAttrValue stream_attribute;   
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(*ptr_addr);
    stream_attribute.accessPolicyWindow.num_bytes = n * sizeof(T);
    stream_attribute.accessPolicyWindow.hitRatio  = hit_ratio;
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(cudaStreamDefault, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
}

__host__ std::string descriptor_to_string(AggregateDescriptor agd, InclusivePrefixDescriptor ipd) {
    std::string res = "(";
    
    if (agd.flag == false) {
        res += "X)";
    }
    else {
        if (ipd.flag == false) {
            res += "A,";
            res += std::to_string(agd.val) + ")";
        }
        else {
            res += "P,";
            res += std::to_string(agd.val) + ",";
            res += std::to_string(ipd.val) + ")";
        }
    }

    return res;
}

__host__ void test_scan_handcraft(const data_t * X_h, const data_t * ans_h, int N) {
    data_t * X_d, * Y_d, * ans_d;
    cudaMalloc(&X_d, N * sizeof(data_t));
    cudaMalloc(&Y_d, N * sizeof(data_t));
    cudaMalloc(&ans_d, N * sizeof(data_t));
    cudaMemcpy(X_d, X_h, N * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ans_d, ans_h, N * sizeof(data_t), cudaMemcpyHostToDevice);

    /* scan_d_kernel_v1 parameters */

    // cudaDeviceProp prop = utils::get_device_properties();
    // int num_threads = 256; 
    // int num_blocks = prop.maxBlocksPerMultiProcessor * prop.multiProcessorCount;
    // int partition_size = prop.warpSize * prop.warpSize;
    // int num_partitions = (N + partition_size - 1) / partition_size;
    // unsigned int * block_counter_d;
    // union AggregateDescriptor * aggregate_descriptor_table_d;
    // union InclusivePrefixDescriptor * inclusive_prefix_descriptor_table_d;
    // malloc_persisting_data<AggregateDescriptor>(&aggregate_descriptor_table_d, num_partitions);
    // malloc_persisting_data<InclusivePrefixDescriptor>(&inclusive_prefix_descriptor_table_d, num_partitions);
    // malloc_persisting_data<unsigned int>(&block_counter_d, 1);
    // cudaDeviceSynchronize();

    /* scan_d_kernel_v2 parameters */

    cudaDeviceProp prop = utils::get_device_properties();
    int num_threads = 256; 
    int num_blocks = prop.maxBlocksPerMultiProcessor * prop.multiProcessorCount;
    int partition_size = 4 * prop.warpSize * prop.warpSize;
    int num_partitions = (N + partition_size - 1) / partition_size;
    unsigned int * block_counter_d;
    union Descriptor * descriptor_table_d;
    malloc_persisting_data<unsigned int>(&block_counter_d, 1);
    malloc_persisting_data<Descriptor>(&descriptor_table_d, num_partitions);
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    // scan_d_kernel_v1<<<num_blocks, num_threads>>>(
    //     X_d, 
    //     Y_d, 
    //     N, 
    //     block_counter_d, 
    //     aggregate_descriptor_table_d, 
    //     inclusive_prefix_descriptor_table_d
    // );

    scan_d_kernel_v2<<<num_blocks, num_threads>>>(
        X_d, 
        Y_d, 
        N, 
        block_counter_d, 
        descriptor_table_d
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t;
    cudaEventElapsedTime(&t, start, stop);

    const char * is_correct_s = check_results(Y_d, ans_d, N) ? "correct" : "incorrect";
    float throughput = (float)(N * sizeof(data_t) * 2) / (t * 1e-3) / 1e9;
    std::cout << "- Throughput (handcraft): " << throughput << " GB/s"
        << " [" << is_correct_s << ", time: " << t << " ms, "
        << "#partitions: " << num_partitions << "]\n"; 

    cudaFree(X_d);
    cudaFree(Y_d);
    cudaFree(ans_d);
    cudaFree(block_counter_d);
    // cudaFree(aggregate_descriptor_table_d);
    // cudaFree(inclusive_prefix_descriptor_table_d);
    cudaFree(descriptor_table_d);
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
    printf("- max block per SM: %d\n", props.maxBlocksPerMultiProcessor);
    printf("- shared mem. size: %lu bytes/sm\n", props.sharedMemPerMultiprocessor);
    printf("- L2 Cache Size: %d bytes\n", props.l2CacheSize);
    printf("- Persisting L2 Cache Max Size: %d bytes\n", props.persistingL2CacheMaxSize);
}

// compile command: nvcc -g -O3 --generate-line-info -lcublas -arch=sm_89 -diag-suppress=177 --maxrregcount 80 -Xcompiler -fopenmp,-Wextra src/scan.cu -o bin/scan.out
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

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    test_scan_openmp(X_h, Y_h, N);
    test_memcpy_d2d(X_h, Y_h, N);
    test_scan_cub(X_h, Y_h, N);
    test_scan_handcraft(X_h, Y_h, N);
 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] X_h;
    delete[] Y_h;

    return 0;
}