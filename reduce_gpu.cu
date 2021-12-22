#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <type_traits>
#include <cuda_runtime_api.h> 
#include <cuda_runtime.h>
#include <cuda.h> 
#include <cooperative_groups.h>

#include "utils.h"

#define KERNEL_VERSION_NUM 6

__global__ void vec_reduce_v0_d(TYPE * g_in_data, TYPE * g_out_data)
{
    extern __shared__ TYPE s_data[];
    u_int tid = threadIdx.x;
    u_int i = blockIdx.x * blockDim.x + threadIdx.x;

    s_data[tid] =  g_in_data[i];
    __syncthreads();

    for(u_int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_out_data[blockIdx.x] = s_data[0];
    }
}

__global__ void vec_reduce_v1_d(TYPE * g_in_data, TYPE * g_out_data)
{
    extern __shared__ TYPE s_data[];
    u_int tid = threadIdx.x;
    u_int i = blockIdx.x * blockDim.x + threadIdx.x;

    s_data[tid] =  g_in_data[i];
    __syncthreads();

    for(u_int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            s_data[index] += s_data[index + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_out_data[blockIdx.x] = s_data[0];
    }
}

__global__ void vec_reduce_v2_d(TYPE * g_in_data, TYPE * g_out_data)
{
    extern __shared__ TYPE s_data[];
    u_int tid = threadIdx.x;
    u_int i = blockIdx.x * blockDim.x + threadIdx.x;

    s_data[tid] =  g_in_data[i];
    __syncthreads();

    for(u_int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_out_data[blockIdx.x] = s_data[0];
    }
}

__global__ void vec_reduce_v3_d(TYPE * g_in_data, TYPE * g_out_data)
{
    extern __shared__ TYPE s_data[];
    u_int tid = threadIdx.x;
    u_int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    s_data[tid] =  g_in_data[i] + g_in_data[i + blockDim.x];;
    __syncthreads();

    for(u_int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_out_data[blockIdx.x] = s_data[0];
    }
}

__global__ void vec_reduce_v4_d(TYPE * g_in_data, TYPE * g_out_data)
{
    extern __shared__ TYPE s_data[];
    u_int tid = threadIdx.x;
    u_int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    s_data[tid] =  g_in_data[i] + g_in_data[i + blockDim.x];
    __syncthreads();

    for(u_int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        s_data[tid] += s_data[tid + 32];
        __syncwarp();
        s_data[tid] += s_data[tid + 16];
        __syncwarp();
        s_data[tid] += s_data[tid + 8];
        __syncwarp();
        s_data[tid] += s_data[tid + 4];
        __syncwarp();
        s_data[tid] += s_data[tid + 2];
        __syncwarp();
        s_data[tid] += s_data[tid + 1];
        __syncwarp();
    }

    if (tid == 0)
    {
        g_out_data[blockIdx.x] = s_data[0];
    }
}

template<u_int blockSize>
__global__ void vec_reduce_v5_d(TYPE * g_in_data, TYPE * g_out_data)
{
    extern __shared__ TYPE s_data[];
    u_int tid = threadIdx.x;
    u_int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    s_data[tid] =  g_in_data[i] + g_in_data[i + blockDim.x];
    __syncthreads();

    if (blockSize >= 1024)
    {
        if (tid < 512)
            s_data[tid] += s_data[tid + 512];
        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
            s_data[tid] += s_data[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
            s_data[tid] += s_data[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
            s_data[tid] += s_data[tid + 64];
        __syncthreads();
    }

    if (tid < 32)
    {
        if (blockSize >= 64) { s_data[tid] += s_data[tid + 32]; __syncwarp(); }
        if (blockSize >= 32) { s_data[tid] += s_data[tid + 16]; __syncwarp(); }
        if (blockSize >= 16) { s_data[tid] += s_data[tid + 8]; __syncwarp(); }
        if (blockSize >= 8) { s_data[tid] += s_data[tid + 4]; __syncwarp(); }
        if (blockSize >= 4) { s_data[tid] += s_data[tid + 2]; __syncwarp(); }
        if (blockSize >= 2) { s_data[tid] += s_data[tid + 1]; __syncwarp(); }
    }

    if (tid == 0)
    {
        g_out_data[blockIdx.x] = s_data[0];
    }
}

#define SWAP(__type, __A, __B)  \
    do {                        \
        __type __tmp = __A;     \
        __A = __B;              \
        __B = __tmp;            \
    } while(0);                 \

__host__ void vec_reduce_gpu(TYPE ** result_d, TYPE * A_d, int size_d, int version)
{
    static bool static_initialized = false;
    static TYPE * B_d, * C_d;
    if (static_initialized == false)
    {
        // initialize the random vector on GPU
        cudaMalloc((void **)&B_d, (size_d / 2) * sizeof(TYPE));
        cudaMalloc((void **)&C_d, (size_d / 4) * sizeof(TYPE));

        static_initialized = true;
    }

    // A -> C, B -> A, C -> B
    SWAP(TYPE *, A_d, B_d);
    SWAP(TYPE *, A_d, C_d);

    bool first_do = true;
    do
    {
        int threads = (size_d < 1024) ? size_d : 1024;
        int smem_size = (threads <= 32 && version >= 3) ? 2 * threads * sizeof(TYPE) : threads * sizeof(TYPE);
        int gmem_per_block = (version >= 3) ? (2 * 1024) : 1024;
        int blocks = int(size_d - 0.5) / gmem_per_block + 1;
        // printf("done(blocks = %d, size = %d)\n", blocks, size_d);

        switch (version)
        {
        case 0: vec_reduce_v0_d<<<blocks, threads, smem_size>>>(B_d, C_d); break;
        case 1: vec_reduce_v1_d<<<blocks, threads, smem_size>>>(B_d, C_d); break;
        case 2: vec_reduce_v2_d<<<blocks, threads, smem_size>>>(B_d, C_d); break;
        case 3: vec_reduce_v3_d<<<blocks, threads, smem_size>>>(B_d, C_d); break;
        case 4: vec_reduce_v4_d<<<blocks, threads, smem_size>>>(B_d, C_d); break;
        case 5: 
            switch (threads)
            {
            case 1024: vec_reduce_v5_d<1024><<<blocks, threads, smem_size>>>(B_d, C_d); break;
            case 512: vec_reduce_v5_d<512><<<blocks, threads, smem_size>>>(B_d, C_d); break;
            case 256: vec_reduce_v5_d<256><<<blocks, threads, smem_size>>>(B_d, C_d); break;
            case 128: vec_reduce_v5_d<128><<<blocks, threads, smem_size>>>(B_d, C_d); break;
            case 64: vec_reduce_v5_d<64><<<blocks, threads, smem_size>>>(B_d, C_d); break;
            case 32: vec_reduce_v5_d<32><<<blocks, threads, smem_size>>>(B_d, C_d); break;
            case 8: vec_reduce_v5_d<8><<<blocks, threads, smem_size>>>(B_d, C_d); break;
            case 4: vec_reduce_v5_d<4><<<blocks, threads, smem_size>>>(B_d, C_d); break;
            case 2: vec_reduce_v5_d<2><<<blocks, threads, smem_size>>>(B_d, C_d); break;
            case 1: vec_reduce_v5_d<1><<<blocks, threads, smem_size>>>(B_d, C_d); break;
            default:
                exit(1);
            }
            break;
        default:
            exit(1);
        }

        if (first_do)
        {
            SWAP(TYPE *, A_d, C_d);
            SWAP(TYPE *, A_d, B_d);
            first_do = false;
        }
        else
        {
            SWAP(TYPE *, B_d, C_d);
        }

        size_d = blocks;
    } while (size_d > 1);

    *result_d = B_d;
    // NOTE: We don't free up the B_d, C_d for the performance reasons.
    // If someone want to get the total sum of A_d, he could just visit
    // the first element in result_d.
}

unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

// @bread: randomly fill in 'size' floats between (-1,1) in the array 'd'
void random_fill(TYPE * d, int size)
{
	for (int i = 0; i < size; ++i)
	{
        if (std::is_same<TYPE, int>::value)
            d[i] = (rand() - RAND_MAX / 2);
        else
    		d[i] = (2.0 * (TYPE)rand() / RAND_MAX) - 1.0;
	}
}

TYPE vec_reduce_cpu(TYPE * A_h, int size)
{
    TYPE result = 0.0;
    for(int i = 0; i < size; i++)
    {
        result += A_h[i];
    }
    return result;
}

void check_correctness(int size, int version)
{
    TYPE * A_h, * A_d, * result_d;
    
    // initialize the random vector on CPU
    A_h = new TYPE[size];
    random_fill(A_h, size);

    // initialize the ranom vector GPU
    u_int size_d = nextPow2(size);
    cudaMalloc((void **)&A_d, size_d * sizeof(TYPE));
    cudaMemset((void *)A_d, 0, size_d * sizeof(TYPE));
    cudaMemcpy(A_d, A_h, size * sizeof(TYPE), cudaMemcpyHostToDevice);

    // GPU test
    TYPE res_gpu;
    vec_reduce_gpu(&result_d, A_d, size_d, version);
    cudaMemcpy(&res_gpu, result_d, sizeof(TYPE), cudaMemcpyDeviceToHost);

    // CPU test
    TYPE res_cpu = vec_reduce_cpu(A_h, size);

    // output results
    if (std::is_same<TYPE, int>::value)
    {
        printf("[gpu]%d - [cpu]%d: ", res_gpu, res_cpu);

        int error = abs(res_cpu - res_gpu);
        if (error == 0)
            printf("correct\n");
        else
            printf("wrong(error = %d)\n", error);
    }
    else
    {
        printf("[gpu]%.6f - [cpu]%.6f: ", res_gpu, res_cpu);

        double epsilon = 1e-5;
        double error = fabs(res_cpu - res_gpu);
        if (error < epsilon)
            printf("correct\n");
        else
            printf("wrong(error = %f)\n", error);
    }
    
    delete A_h;
    cudaFree(A_d);
}

int get_memory_rw_work(int size)
{
    int work = 0;
    do {
        work += (size + size / 1024);
        size /= 1024; 
    } while(size >= 1024);

    if (size > 1)
        work += size + 1;
    return work;
}

void benchmark_gpu(int size_h, int version)
{
    TYPE * A_d, * A_h, * result_d;

    u_int size_d = nextPow2(size_h);

    A_h = new TYPE[size_h];
    random_fill(A_h, size_h);
    cudaMalloc((void **)&A_d, size_d * sizeof(TYPE));
    cudaMemset((void *)A_d, 0, size_d * sizeof(TYPE));
    cudaMemcpy(A_d, A_h, size_h * sizeof(TYPE), cudaMemcpyHostToDevice);

    // warm up
    vec_reduce_gpu(&result_d, A_d, size_d, version);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float min_tdiff = FLT_MAX , max_tdiff = FLT_MIN, total_tdiff = 0.0;
    for(int i = 0; i < ITER_NUM; i++)
    {
        cudaEventRecord(start);

        vec_reduce_gpu(&result_d, A_d, size_d, version);

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float tdiff;
        cudaEventElapsedTime(&tdiff, start, stop); 

        min_tdiff = (tdiff < min_tdiff) ? tdiff : min_tdiff;
        max_tdiff = (tdiff > max_tdiff) ? tdiff : max_tdiff;
        total_tdiff += tdiff;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double gflops = (size_h - 1) / min_tdiff / 1e9 * 1e3;
    double bandwidth = get_memory_rw_work(size_d) * sizeof(TYPE) / min_tdiff / 1e9 * 1e3;
    printf("reduce[gpu ver.%d]: (min) %f ms, (max) %f ms, (avg) %f ms, %f GOPS, %f GB/s\n", version, min_tdiff, max_tdiff, total_tdiff / ITER_NUM, gflops, bandwidth);

    cudaFree(A_d);
    delete A_h;
}

int main(int argc, char * argv[])
{
    if (argc <= 2) return 1;

    const u_int size = atoi(argv[1]) * 1024 * 1024; // MB
    const u_int version = atoi(argv[2]);

    char * type_string;
    if (std::is_same<TYPE, int>::value)
        type_string = "int";
    else if (std::is_same<TYPE, float>::value)
        type_string = "float";
    else if (std::is_same<TYPE, double>::value)
        type_string = "double";

    int device = 0; // default
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int memoryClockRate = props.memoryClockRate; // KHz
    int memoryBusWidth = props.memoryBusWidth;  // bits
    double theory_bandwidth = memoryClockRate * 1e3 * memoryBusWidth / 8 * 2 / 1e9; // GB/sec

    printf("%s: (theoretical memory-bandwidth) %f GB/sec\n", props.name, theory_bandwidth);
    printf("type(%s), size(%d), iter(%d), version(%d)\n", type_string, size, ITER_NUM, version);

    check_correctness(size, version);
    benchmark_gpu(size, version);

    return 0;
}