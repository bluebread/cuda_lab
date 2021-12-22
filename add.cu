#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils.h"

__global__ void vec_add_d(float * A, float * B, float * C, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= N)
        return;

    C[i] = A[i] + B[i];
}

void vec_add_h(float * A, float * B, float * C, int N)
{
    for(int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool is_vector_same(float * C, float * D, int N)
{
    const float epsilon = 1e-6;

    for(int i = 0; i < N; i++)
    {
        if (fabs(C[i] - D[i]) > epsilon)
            return false;
    }
    return true;
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
    if (argc < 1) return 1;

    const int VEC_LEN = atoi(argv[1]);

    float * A_h, * B_h, * C_h;
    float * A_d, * B_d, * C_d;
    float * C_ans_h;

    A_h = new float[VEC_LEN];
    B_h = new float[VEC_LEN];
    C_h = new float[VEC_LEN];

    C_ans_h = new float[VEC_LEN];

    cudaMalloc((void **)&A_d, VEC_LEN * sizeof(float));
    cudaMalloc((void **)&B_d, VEC_LEN * sizeof(float));
    cudaMalloc((void **)&C_d, VEC_LEN * sizeof(float));

    random_fill(A_h, VEC_LEN);
    random_fill(B_h, VEC_LEN);
    random_fill(C_h, VEC_LEN);

    cudaMemcpy(A_d, A_h, VEC_LEN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, VEC_LEN * sizeof(float), cudaMemcpyHostToDevice);

    double tdiff1, tdiff2;

    int blocks = int(VEC_LEN - 0.5) / MAX_THREADS_PER_BLOCK + 1;
    StartCounter();
    vec_add_d<<<blocks, MAX_THREADS_PER_BLOCK>>>(A_d, B_d, C_d, VEC_LEN);
    tdiff1 = GetCounter();
    cudaMemcpy(C_h, C_d, VEC_LEN * sizeof(float), cudaMemcpyDeviceToHost);

    StartCounter();
    vec_add_h(A_h, B_h, C_ans_h, VEC_LEN);
    tdiff2 = GetCounter();

    if(is_vector_same(C_h, C_ans_h, VEC_LEN))
        printf("Adding is correct\n");
    else
        printf("Adding is wrong\n");

    printf("add[cuda<%d,%d>]: %.3f ms\n", blocks, MAX_THREADS_PER_BLOCK, tdiff1);
    printf("add[cpu]: %.3f ms\n", tdiff2);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A_h);
    free(B_h);
    free(C_h);
    free(C_ans_h);

    return 0;
}