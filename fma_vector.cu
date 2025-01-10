#include <cuda.h>
#include <cuda_runtime_api.h> 
#include <omp.h>
#include <cstdio>
#include <string>
#include <random>
#include <iostream>

#define NUM_ITERATION   10

#ifndef DATA_T
#define DATA_T          float
#endif
#define STRINGIFY(s)    #s
#define MACRO_STR(m)   STRINGIFY(m)

__global__ void fma_vector_d(DATA_T * C, DATA_T * A, DATA_T * B, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < N)
        C[i] += A[i] * B[i];

    /*
    * single-precision FMA operation for round-to-nearest:
    *       C[i] = __fmaf_rn(A[i], B[i], C[i]);
    * 
    * double-precision FMA operation for round-to-nearest.
    *       C[i] = __fma_rn(A[i], B[i], C[i]);
    */
}

__host__ void fma_vector_h(DATA_T * C, DATA_T * A, DATA_T * B, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        C[i] += A[i] * B[i];
    }
}

__host__ void random_fill(DATA_T * d, int N) {
    #pragma omp parallel 
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<> dist(0, 1); 

        #pragma omp for
        for (int i = 0; i < N; ++i) {
            d[i] = dist(rng);
        }
    }
}

__host__ bool is_equal_vector(DATA_T * X, DATA_T * Y, int N) {
    volatile bool is_equal = true;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        if (! is_equal) 
            continue;

        if (std::abs(X[i] - Y[i]) >= 1e-6) {
            is_equal = false;
        }
    }

    return is_equal;
}

// compile command: nvcc -g -O3 -Xcompiler -fopenmp fma_vector.cu
int main(int argc, const char * argv[]) {
    DATA_T * A_h, * B_h, * C_h;
    DATA_T * A_d, * B_d, * C_d;
    int N;

    bool is_data_type_valid = 
        std::is_same<DATA_T, float>::value || std::is_same<DATA_T, double>::value;

    if (argc > 1 && is_data_type_valid) {
        N = std::stoi(argv[1]);
    }
    else {
        exit(1);
    }

    A_h = new DATA_T[N];
    B_h = new DATA_T[N];
    C_h = new DATA_T[N];

    random_fill(A_h, N);
    random_fill(B_h, N);
    random_fill(C_h, N);

    cudaMalloc((void **)&A_d, sizeof(DATA_T) * N);
    cudaMalloc((void **)&B_d, sizeof(DATA_T) * N);
    cudaMalloc((void **)&C_d, sizeof(DATA_T) * N);
    cudaMemcpy(A_d, A_h, sizeof(DATA_T) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(DATA_T) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, sizeof(DATA_T) * N, cudaMemcpyHostToDevice);

    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    int num_threads_per_block = props.maxThreadsPerBlock;
    int num_blocks = std::ceil((float)N / props.maxThreadsPerBlock);

    fma_vector_h(C_h, A_h, B_h, N);
    fma_vector_d<<<num_blocks, num_threads_per_block>>>(C_d, A_d, B_d, N);
    cudaMemcpy(A_h, C_d, sizeof(DATA_T) * N, cudaMemcpyDeviceToHost);

    printf("Fused Multiply-Add Vector Operation\n");
    printf("- data type: %s\n", MACRO_STR(DATA_T));
    printf("- vector size: %lu bytes\n", N * sizeof(DATA_T));

    if (is_equal_vector(C_h, A_h, N)) {
        printf("- calculation: correct\n");
    }
    else {
        printf("- calculation: incorrect\n");
        exit(1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int k = 0; k < NUM_ITERATION; k++) {
        cudaEventRecord(start);

        fma_vector_d<<<num_blocks, num_threads_per_block>>>(C_d, A_d, B_d, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float tdiff;
        cudaEventElapsedTime(&tdiff, start, stop); 

        printf("- test %d: %.3f (ms)\n", k, tdiff);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] A_h;
    delete[] B_h;
    delete[] C_h;
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}