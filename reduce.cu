#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <cooperative_groups.h>

#include "utils.h"

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

__global__ void vec_reduce_v0_d(double * g_in_data, double * g_out_data)
{
    __shared__ double s_data[MAX_THREADS_PER_BLOCK];
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

__global__ void vec_reduce_v1_d(double * g_in_data, double * g_out_data)
{
    __shared__ double s_data[MAX_THREADS_PER_BLOCK];
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

__global__ void vec_reduce_v2_d(double * g_in_data, double * g_out_data)
{
    __shared__ double s_data[MAX_THREADS_PER_BLOCK];
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

__global__ void vec_reduce_v3_d(double * g_in_data, double * g_out_data)
{
    extern __shared__ double s_data[];
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

__global__ void vec_reduce_v4_d(double * g_in_data, double * g_out_data)
{
    extern __shared__ double s_data[];
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
__global__ void vec_reduce_v5_d(double * g_in_data, double * g_out_data)
{
    extern __shared__ double s_data[];
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

double vec_reduce_gpu(double * A_d, int size_d)
{
    double * B_d;

    // initialize the random vector on GPU
    cudaMalloc((void **)&B_d, size_d * sizeof(double));
    cudaMemset((void *)B_d, 0, size_d * sizeof(double));

    int blocks = int(size_d - 0.5) / MAX_THREADS_PER_BLOCK + 1;
    // vec_reduce_v0_d<<<blocks, MAX_THREADS_PER_BLOCK>>>(A_d, B_d);
    // vec_reduce_v1_d<<<blocks, MAX_THREADS_PER_BLOCK>>>(A_d, B_d);
    vec_reduce_v2_d<<<blocks, MAX_THREADS_PER_BLOCK>>>(A_d, B_d);

    double * B_h = new double[size_d];
    cudaMemcpy(B_h, B_d, size_d * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(B_d);

    double result = 0.0;
    for(int i = 0; i < blocks; i++)
    {
        result += B_h[i];
    }

    return result;
}

double vec_reduce_gpu_o1(double * A_d, int size_d)
{
    double * B_d, * C_d;
    // initialize the random vector on GPU
    cudaMalloc((void **)&B_d, size_d * sizeof(double));
    cudaMemset((void *)B_d, 0, size_d * sizeof(double));
    cudaMalloc((void **)&C_d, size_d * sizeof(double));
    cudaMemset((void *)C_d, 0, size_d * sizeof(double));

    int blocks = int(size_d - 0.5) / MAX_THREADS_PER_BLOCK + 1;
    // vec_reduce_v0_d<<<blocks, MAX_THREADS_PER_BLOCK>>>(A_d, B_d);
    // vec_reduce_v1_d<<<blocks, MAX_THREADS_PER_BLOCK>>>(A_d, B_d);
    vec_reduce_v2_d<<<blocks, MAX_THREADS_PER_BLOCK>>>(A_d, B_d);

    size_d = blocks;
    while(size_d > 1)
    {
        blocks = int(size_d - 0.5) / MAX_THREADS_PER_BLOCK + 1;
        u_int threads = (size_d < MAX_THREADS_PER_BLOCK) ? size_d : MAX_THREADS_PER_BLOCK;
        // printf("done(blocks = %d, size = %d)\n", blocks, size_d);

        // vec_reduce_v0_d<<<blocks, threads>>>(B_d, C_d);
        // vec_reduce_v1_d<<<blocks, threads>>>(B_d, C_d);
        vec_reduce_v2_d<<<blocks, threads>>>(B_d, C_d);

        double * tmp = B_d;
        B_d = C_d;
        C_d = tmp;

        size_d = blocks;
    }
    double result;
    cudaMemcpy(&result, B_d, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(B_d);
    cudaFree(C_d);

    return result;
}

double vec_reduce_gpu_o2(double * A_d, int size_d)
{
    double * B_d, * C_d;
    // initialize the random vector on GPU
    cudaMalloc((void **)&B_d, size_d * sizeof(double));
    cudaMemset((void *)B_d, 0, size_d * sizeof(double));
    cudaMalloc((void **)&C_d, size_d * sizeof(double));
    cudaMemset((void *)C_d, 0, size_d * sizeof(double));

    int blocks = int(size_d - 0.5) / (2 * MAX_THREADS_PER_BLOCK) + 1;
    int threads = (size_d < MAX_THREADS_PER_BLOCK) ? size_d : MAX_THREADS_PER_BLOCK;
    int smem_size = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);
    // vec_reduce_v3_d<<<blocks, threads, smem_size>>>(A_d, B_d);
    vec_reduce_v4_d<<<blocks, threads, smem_size>>>(A_d, B_d);

    size_d = blocks;
    while(size_d > 1)
    {
        blocks = int(size_d - 0.5) / (2 * MAX_THREADS_PER_BLOCK) + 1;
        threads = (size_d < MAX_THREADS_PER_BLOCK) ? size_d : MAX_THREADS_PER_BLOCK;
        smem_size = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);
        // printf("done(blocks = %d, size = %d)\n", blocks, size_d);

        // vec_reduce_v3_d<<<blocks, threads, smem_size>>>(B_d, C_d);
        vec_reduce_v4_d<<<blocks, threads, smem_size>>>(B_d, C_d);

        double * tmp = B_d;
        B_d = C_d;
        C_d = tmp;

        size_d = blocks;
    }
    double result;
    cudaMemcpy(&result, B_d, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(B_d);
    cudaFree(C_d);

    return result;
}

double vec_reduce_gpu_o3(double * A_d, int size_d)
{
    bool static_initialized = false;
    static double * B_d, * C_d;
    if (static_initialized == false)
    {
        // initialize the random vector on GPU
        cudaMalloc((void **)&B_d, (size_d / 2) * sizeof(double));
        // cudaMemset((void *)B_d, 0, size_d * sizeof(double));
        cudaMalloc((void **)&C_d, (size_d / 4) * sizeof(double));
        // cudaMemset((void *)C_d, 0, size_d * sizeof(double));
    }

    int blocks = int(size_d - 0.5) / (2 * MAX_THREADS_PER_BLOCK) + 1;
    int threads = (size_d < MAX_THREADS_PER_BLOCK) ? size_d : MAX_THREADS_PER_BLOCK;
    int smem_size = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);
    vec_reduce_v5_d<MAX_THREADS_PER_BLOCK><<<blocks, threads, smem_size>>>(A_d, B_d);

    size_d = blocks;
    while(size_d > 1)
    {
        blocks = int(size_d - 0.5) / (2 * MAX_THREADS_PER_BLOCK) + 1;
        threads = (size_d < MAX_THREADS_PER_BLOCK) ? size_d : MAX_THREADS_PER_BLOCK;
        smem_size = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);
        // printf("done(blocks = %d, size = %d)\n", blocks, size_d);

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

        double * tmp = B_d;
        B_d = C_d;
        C_d = tmp;

        size_d = blocks;
    }
    double result = 0.0;
    // cudaMemcpy(&result, B_d, sizeof(double), cudaMemcpyDeviceToHost);

    // cudaFree(B_d);
    // cudaFree(C_d);

    return result;
}

double vec_reduce_cpu(double * A_h, int size)
{
    double result = 0.0;
    for(int i = 0; i < size; i++)
    {
        result += A_h[i];
    }
    return result;
}

#define _UNROLL_1(ARR, OP, i) (ARR[i])
#define _UNROLL_2(ARR, OP, i) (ARR[i] OP ARR[i+1])
#define _UNROLL_3(ARR, OP, i) ((_UNROLL_2(ARR, OP, i)) OP (_UNROLL_1(ARR, OP, i+2)))
#define _UNROLL_4(ARR, OP, i) ((_UNROLL_2(ARR, OP, i)) OP (_UNROLL_2(ARR, OP, i+2)))
#define _UNROLL_5(ARR, OP, i) ((_UNROLL_4(ARR, OP, i)) OP (_UNROLL_1(ARR, OP, i+4)))
#define _UNROLL_6(ARR, OP, i) ((_UNROLL_4(ARR, OP, i)) OP (_UNROLL_2(ARR, OP, i+4)))
#define _UNROLL_7(ARR, OP, i) ((_UNROLL_4(ARR, OP, i)) OP (_UNROLL_3(ARR, OP, i+4)))
#define _UNROLL_8(ARR, OP, i) ((_UNROLL_4(ARR, OP, i)) OP (_UNROLL_4(ARR, OP, i+4)))
#define _UNROLL_9(ARR, OP, i) ((_UNROLL_8(ARR, OP, i)) OP (_UNROLL_1(ARR, OP, i+8)))
#define _UNROLL_10(ARR, OP, i) ((_UNROLL_8(ARR, OP, i)) OP (_UNROLL_2(ARR, OP, i+8)))
#define _UNROLL_11(ARR, OP, i) ((_UNROLL_8(ARR, OP, i)) OP (_UNROLL_3(ARR, OP, i+8)))
#define _UNROLL_12(ARR, OP, i) ((_UNROLL_8(ARR, OP, i)) OP (_UNROLL_4(ARR, OP, i+8)))
#define _UNROLL_13(ARR, OP, i) ((_UNROLL_8(ARR, OP, i)) OP (_UNROLL_5(ARR, OP, i+8)))
#define _UNROLL_14(ARR, OP, i) ((_UNROLL_8(ARR, OP, i)) OP (_UNROLL_6(ARR, OP, i+8)))
#define _UNROLL_15(ARR, OP, i) ((_UNROLL_8(ARR, OP, i)) OP (_UNROLL_7(ARR, OP, i+8)))
#define _UNROLL_16(ARR, OP, i) ((_UNROLL_8(ARR, OP, i)) OP (_UNROLL_8(ARR, OP, i+8)))
#define _UNROLL_32(ARR, OP, i) ((_UNROLL_16(ARR, OP, i)) OP (_UNROLL_16(ARR, OP, i+8)))

#define UNROLL(ARR, OP, unroll, i) (_UNROLL_ ## unroll (ARR, OP, i))

double vec_reduce_cpu_unroll(double * A_h, int size)
{
    double result = 0.0;
    int i = 0;
    for(; i <= size - 16; i += 16)
    {
        result += UNROLL(A_h, +, 16, i);
    }
    for(; i < size; i++)
    {
        result += A_h[i];
    }
    return result;
}

// @bread: randomly fill in 'size' floats between (-1,1) in the array 'd'
void random_fill(double * d, int size)
{
	for (int i = 0; i < size; ++i)
	{
		d[i] = (2.0 * (double)rand() / RAND_MAX) - 1.0;
	}
}

void benchmark_cpu(int size)
{
    double * A;
    
    // initialize the random vector on CPU
    A = new double[size];
    random_fill(A, size);

    // warm up
    // vec_reduce_cpu(A, size);
    vec_reduce_cpu_unroll(A, size);

    double min_tdiff = DBL_MAX;
    for(int i = 0; i < ITER_NUM; i++)
    {
        StartCounter();

        // vec_reduce_cpu(A, size);
        vec_reduce_cpu_unroll(A, size);

        double tdiff = GetCounter();
        min_tdiff = (tdiff < min_tdiff) ? tdiff : min_tdiff;
    }

    double gflops = (size - 1) / min_tdiff / 1e9 * 1e3;
    printf("reduce[cpu]: %f ms, %f Gflops\n", min_tdiff, gflops);

    delete A;
}

void benchmark_gpu(int size_h)
{
    double * A_d, * A_h;

    u_int size_d = nextPow2(size_h);

    A_h = new double[size_h];
    random_fill(A_h, size_h);
    cudaMalloc((void **)&A_d, size_d * sizeof(double));
    cudaMemset((void *)A_d, 0, size_d * sizeof(double));
    cudaMemcpy(A_d, A_h, size_h * sizeof(double), cudaMemcpyHostToDevice);

    // warm up
    // vec_reduce_gpu(A_d, size_d);
    // vec_reduce_gpu_o1(A_d, size_d);
    // vec_reduce_gpu_o2(A_d, size_d);
    vec_reduce_gpu_o3(A_d, size_d);


    double min_tdiff = DBL_MAX;
    for(int i = 0; i < ITER_NUM; i++)
    {
        StartCounter();

        // vec_reduce_gpu(A_d, size_d);
        // vec_reduce_gpu_o1(A_d, size_d);
        // vec_reduce_gpu_o2(A_d, size_d);
        vec_reduce_gpu_o3(A_d, size_d);

        double tdiff = GetCounter();
        min_tdiff = (tdiff < min_tdiff) ? tdiff : min_tdiff;
    }

    double gflops = (size_h - 1) / min_tdiff / 1e9 * 1e3;
    printf("reduce[gpu]: %f ms, %f Gflops\n", min_tdiff, gflops);

    cudaFree(A_d);
    delete A_h;
}

void check_correctness(int size)
{
    double * A_h, * A_d;
    
    // initialize the random vector on CPU
    A_h = new double[size];
    random_fill(A_h, size);

    // initialize the ranom vector GPU
    u_int size_d = nextPow2(size);
    cudaMalloc((void **)&A_d, size_d * sizeof(double));
    cudaMemset((void *)A_d, 0, size_d * sizeof(double));
    cudaMemcpy(A_d, A_h, size * sizeof(double), cudaMemcpyHostToDevice);

    // GPU test
    double res_gpu = vec_reduce_gpu_o3(A_d, size);

    // CPU test
    double res_cpu = vec_reduce_cpu(A_h, size);

    // output results
    printf("[gpu]%.6f - [cpu]%.6f: ", res_gpu, res_cpu);

    double epsilon = 1e-5;
    double error = fabs(res_cpu - res_gpu);
    if (error < epsilon)
        printf("correct\n");
    else
        printf("wrong(error = %f)\n", error);
    
    delete A_h;
    cudaFree(A_d);
}


int main(int argc, char * argv[])
{
    if (argc <= 1) return 1;

    const u_int size = atoi(argv[1]);

    printf("size(%d), iter(%d)\n", size, ITER_NUM);

    check_correctness(size);

    benchmark_cpu(size);
    benchmark_gpu(size);

    return 0;
}