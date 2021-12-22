#include <math.h>
#include <stdio.h>
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <cooperative_groups.h>

#include "utils.h"

__global__ void vec_multiply_d(float * A, float * B, float * C, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= N) return;

    C[i] = A[i] * B[i];
}

__global__ void vec_sum_partial(float * A, float * B, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= N) return;

    B[i] = A[2 * i] + A[2 * i + 1];
}

float vec_dot_cuda(float * A_h, float * B_h, int N)
{
    float * A_d, * B_d;
    int N_log = (int)log2(N) + 1;
    int vec_size = (1 << N_log);
    
    cudaMalloc((void **)&A_d, vec_size * sizeof(float));
    cudaMalloc((void **)&B_d, vec_size * sizeof(float));

    cudaMemset((void *)A_d, 0, vec_size * sizeof(float));
    cudaMemset((void *)B_d, 0, vec_size * sizeof(float));

    cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = int(N - 0.5) / MAX_THREADS_PER_BLOCK + 1;
    vec_multiply_d<<<blocks, MAX_THREADS_PER_BLOCK>>>(A_d, B_d, A_d, N);

    int end_p = 10;
    for(int pow = N_log - 1; pow > 0; pow--)
    {
        int n = (1 << pow);
        blocks = int(n - 0.5) / MAX_THREADS_PER_BLOCK + 1;
        vec_sum_partial<<<blocks, MAX_THREADS_PER_BLOCK>>>(A_d, B_d, n);
        float * tmp = A_d;
        A_d = B_d;
        B_d = tmp;
    }

    float result_h;
    cudaMemcpy(&result_h, A_d, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);

    return result_h;
}

float vec_dot_cpu(float * A, float * B, int N)
{
    float result = 0.0;
    for(int i = 0; i < N; i++)
    {
        result += A[i] * B[i];
    }
    return result;
}

// @bread: randomly fill in 'size' floats between [0,1) in the array 'd'
void random_fill(float * d, int size)
{
	for (int i = 0; i < size; ++i)
	{
		d[i] = (float)rand() / RAND_MAX;
	}
}

int main(int argc, char * argv[])
{
    if (argc <= 1) return 1;

    const int VEC_LEN = atoi(argv[1]);

    float * A, * B;
    
    A = new float[VEC_LEN];
    B = new float[VEC_LEN];

    random_fill(A, VEC_LEN);
    random_fill(B, VEC_LEN);

    float res_cpu, res_cuda;

    double tdiff1, tdiff2;
    StartCounter();
    res_cuda = vec_dot_cuda(A, B, VEC_LEN);
    tdiff1 = GetCounter();

    StartCounter();
    res_cpu = vec_dot_cpu(A, B, VEC_LEN);
    tdiff2 = GetCounter();

    float epsilon = 1e-5;
    float error = fabs(res_cpu - res_cuda);
    if (error < epsilon)
        printf("Dot is correct\n");
    else
        printf("Dot is wrong(error = %f)\n", error);

    printf("dot[cuda]: %.3f ms\n", tdiff1);
    printf("dot[cpu]: %.3f ms\n", tdiff2);

    return 0;
}