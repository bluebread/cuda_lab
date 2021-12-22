#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdlib>

#define TYPE int
#define ITER_NUM 50

void benchmark_op(int size)
{
    // generate 32M random numbers serially
    thrust::host_vector<TYPE> h_A(size);
    thrust::host_vector<TYPE> h_B(size);
    std::generate(h_A.begin(), h_A.end(), rand);
    std::generate(h_B.begin(), h_B.end(), rand);
    // TYPE x = thrust::reduce(h_vec.begin(), h_vec.end(), 0, thrust::plus<TYPE>());

    // transfer data to the device
    thrust::device_vector<TYPE> d_A = h_A;
    thrust::device_vector<TYPE> d_B = h_B;
    thrust::device_vector<TYPE> d_C(size);
    thrust::fill(d_C.begin(), d_C.end(), 0);

    // warm up
    thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_C.begin(), thrust::plus<TYPE>());

    // double epsilon = 1e-6;
    // if (fabs(x - y) < epsilon)
    // printf("reduce[thrust]: correct\n");
    // else printf("reduce[thrust]: wrong(%f, %f)\n", x, y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float min_tdiff = FLT_MAX , max_tdiff = FLT_MIN, total_tdiff = 0.0;
    for(int i = 0; i < ITER_NUM; i++)
    {
        cudaEventRecord(start);

        thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_C.begin(), thrust::plus<TYPE>());
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

    double gflops = (size - 1) / min_tdiff / 1e9 * 1e3;
    printf("op[thrust]: (min) %f ms, (max) %f ms, (avg) %f ms, %f GOPS\n", min_tdiff, max_tdiff, total_tdiff / ITER_NUM, gflops);
}

void benchmark_reduce(int size)
{
    // generate 32M random numbers serially
    thrust::host_vector<TYPE> h_vec(size);
    std::generate(h_vec.begin(), h_vec.end(), rand);
    // TYPE x = thrust::reduce(h_vec.begin(), h_vec.end(), 0, thrust::plus<TYPE>());

    // transfer data to the device
    thrust::device_vector<TYPE> d_vec = h_vec;

    // warm up
    TYPE y = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<TYPE>());

    // double epsilon = 1e-6;
    // if (fabs(x - y) < epsilon)
    // printf("reduce[thrust]: correct\n");
    // else printf("reduce[thrust]: wrong(%f, %f)\n", x, y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float min_tdiff = FLT_MAX , max_tdiff = FLT_MIN, total_tdiff = 0.0;
    for(int i = 0; i < ITER_NUM; i++)
    {
        cudaEventRecord(start);

        TYPE y = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<TYPE>());

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

    double gflops = (size - 1) / min_tdiff / 1e9 * 1e3;
    printf("reduce[thrust]: (min) %f ms, (max) %f ms, (avg) %f ms, %f GOPS\n", min_tdiff, max_tdiff, total_tdiff / ITER_NUM, gflops);
}

int main(int argc, char * argv[])
{
    if(argc <= 1) return 1;

    const int size = atoi(argv[1]) * 1024 * 1024;

    char * type_string;
    if (std::is_same<TYPE, int>::value)
        type_string = "int";
    else if (std::is_same<TYPE, float>::value)
        type_string = "float";
    else if (std::is_same<TYPE, double>::value)
        type_string = "double";

    printf("type(%s), size(%d), iter(%d)\n", type_string, size, ITER_NUM);

    benchmark_op(size);
    benchmark_reduce(size);
}