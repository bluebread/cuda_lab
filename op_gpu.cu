#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <windows.h>

// #include "utils.h"

#define TYPE int
#define ITER_NUM 50
#define OP +

#define STMT(A, B, C, i, n, s) C[i + (n) * s] = A[i + (n) * s] OP B[i + (n) * s]

#define UNROLL_STMT_1(A, B, C, i, n, s) STMT(A, B, C, i, n, s)
#define UNROLL_STMT_2(A, B, C, i, n, s) UNROLL_STMT_1(A, B, C, i, n, s); UNROLL_STMT_1(A, B, C, i, n + 1, s)
#define UNROLL_STMT_4(A, B, C, i, n, s) UNROLL_STMT_2(A, B, C, i, n, s); UNROLL_STMT_2(A, B, C, i, n + 2, s)
#define UNROLL_STMT_8(A, B, C, i, n, s) UNROLL_STMT_4(A, B, C, i, n, s); UNROLL_STMT_4(A, B, C, i, n + 4, s)
#define UNROLL_STMT_16(A, B, C, i, n, s) UNROLL_STMT_8(A, B, C, i, n, s); UNROLL_STMT_8(A, B, C, i, n + 8, s)
#define UNROLL_STMT_32(A, B, C, i, n, s) UNROLL_STMT_16(A, B, C, i, n, s); UNROLL_STMT_16(A, B, C, i, n + 16, s)
#define UNROLL_STMT_64(A, B, C, i, n, s) UNROLL_STMT_32(A, B, C, i, n, s); UNROLL_STMT_32(A, B, C, i, n + 32, s)
#define UNROLL_STMT_128(A, B, C, i, n, s) UNROLL_STMT_64(A, B, C, i, n, s); UNROLL_STMT_64(A, B, C, i, n + 64, s)
#define UNROLL_STMT_256(A, B, C, i, n, s) UNROLL_STMT_128(A, B, C, i, n, s); UNROLL_STMT_128(A, B, C, i, n + 128, s)
#define UNROLL_STMT_512(A, B, C, i, n, s) UNROLL_STMT_256(A, B, C, i, n, s); UNROLL_STMT_256(A, B, C, i, n + 256, s)
#define UNROLL_STMT_1024(A, B, C, i, n, s) UNROLL_STMT_512(A, B, C, i, n, s); UNROLL_STMT_512(A, B, C, i, n + 512, s)

#define UNROLL_STMT(unroll, A, B, C, i, n, s) UNROLL_STMT_ ## unroll (A, B, C, i, n, s)

__global__ void vec_op_d(TYPE * A, TYPE * B, TYPE * C)
{
    int s = blockDim.x;
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // UNROLL_STMT(2, A, B, C, i, 0, s);
    C[i] = A[i] OP B[i];
}

__host__ void vec_op_gpu(TYPE * A, TYPE * B, TYPE * C, int size_d)
{
    int blocks = size_d / 1024;
    vec_op_d<<<blocks, 1024>>>(A, B, C);
}

void vec_op_cpu(TYPE * A, TYPE * B, TYPE * C, int size_h)
{
    for(int i = 0; i < size_h; i++)
    {
        C[i] = A[i] OP B[i]; 
    }
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

bool is_vec_same(TYPE * C_h, TYPE * C_d, int size)
{
    for(int i = 0; i < size; i++)
    {
        if (std::is_same<TYPE, int>::value)
        {
            if (C_h[i] != C_d[i])
                return false;
        }
        else
        {
            double epsilon = 1e-6;
            if (fabs(C_h[i] - C_d[i]) >= epsilon)
                return false;
        }
    }
    return true;
}

void benchmark(int size_h)
{
    TYPE * A_h, * B_h, * C_h, * D_h;
    A_h = new TYPE[size_h];
    B_h = new TYPE[size_h];
    C_h = new TYPE[size_h];
    D_h = new TYPE[size_h];

    TYPE * A_d, * B_d, * C_d;
    int size_d = nextPow2(size_h);
    cudaMalloc((void **)&A_d, size_d * sizeof(TYPE));
    cudaMalloc((void **)&B_d, size_d * sizeof(TYPE));
    cudaMalloc((void **)&C_d, size_d * sizeof(TYPE));

    cudaMemcpy(A_d, A_h, size_h * sizeof(TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_h * sizeof(TYPE), cudaMemcpyHostToDevice);

    vec_op_cpu(A_h, B_h, C_h, size_h);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm up
    vec_op_gpu(A_d, B_d, C_d, size_d);
    cudaMemcpy(D_h, C_d, size_h * sizeof(TYPE), cudaMemcpyDeviceToHost);

    if (is_vec_same(C_h, D_h, size_h))
        printf("op: correct\n");
    else printf("op: wrong\n");

    float min_tdiff = FLT_MAX , max_tdiff = FLT_MIN, total_tdiff = 0.0;
    for(int i = 0; i < ITER_NUM; i++)
    {
        cudaEventRecord(start);

        vec_op_gpu(A_d, B_d, C_d, size_d);

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

    double gflops = size_d / min_tdiff / 1e9 * 1e3;
    double bandwidth = 3 * size_d * sizeof(TYPE) / min_tdiff / 1e9 * 1e3;
    printf("op[gpu]: (min) %f ms, (max) %f ms, (avg) %f ms, %f GOPS, %f GB/s\n", min_tdiff, max_tdiff, total_tdiff / ITER_NUM, gflops, bandwidth);

    delete A_h;
    delete B_h;
    delete C_h;
    delete D_h;
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(int argc, char * argv[])
{
    if (argc <= 1) return 1;

    const u_int size = atoi(argv[1]) * 1024 * 1024; // MB

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
    printf("type(%s), size(%d), iter(%d)\n", type_string, size, ITER_NUM);

    benchmark(size);
}