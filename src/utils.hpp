#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <random>
#include <iostream>
#include <iomanip>
#include <string>
#include <numeric>
#include <cassert>
#include <typeinfo>
#include <cxxabi.h>

#include <omp.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace utils {
    template<typename T>
    __host__ const std::string get_type_name(const T variable = 0) {
        const char* const name = typeid(variable).name();
        int status = -4;
        char* const demangled_name = abi::__cxa_demangle(name, NULL, NULL, &status);
        std::string ret{name};
        if (status == 0) {
            ret = std::string(demangled_name);
            free(demangled_name);
        }
        return ret;
    }

    __host__ double get_random_number() {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<> dist(0, 1); 

        return dist(rng);
    }

    template<typename T>
    __host__ void random_fill_h(T * X, int N) {
        static_assert(std::is_floating_point<T>::value, "input X must be floating-point type pointer");

        #pragma omp parallel 
        {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_real_distribution<> dist(0, 1); 

            #pragma omp for private(dist)
            for (int i = 0; i < N; ++i) {
                X[i] = 2 * dist(rng) - 1;
            }
        }
    }

    template<typename T>
    __global__ void random_fill_d_kernel(T * X, int N) {
        static_assert(std::is_floating_point<T>::value, "input X must be floating-point type pointer");
        int base = (N / gridDim.x) * blockIdx.x + min(blockIdx.x, N % gridDim.x);
        int len = (N / gridDim.x) + (blockIdx.x < N % gridDim.x);
        int idx = blockDim.x * blockIdx.x + threadIdx.x;

        curandState state;
        curand_init(clock64(), idx, 0, &state); 

        for (int i = 0; i < len / blockDim.x + (len % blockDim.x > 0); i++) {
            if (i * blockDim.x + threadIdx.x < len) 
                X[base + i * blockDim.x + threadIdx.x] = curand_uniform(&state);
        }
    }

    __host__ cudaDeviceProp get_device_properties() {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        return prop;
    }

    template<typename T>
    __host__ void random_fill_d(T * X, int N) {
        static_assert(std::is_floating_point<T>::value, "input X must be floating-point type pointer");

        cudaDeviceProp prop = get_device_properties();
        int num_threads_per_block = std::gcd(prop.maxThreadsPerBlock, prop.maxThreadsPerMultiProcessor);
        int num_blocks_per_sm = (prop.maxThreadsPerMultiProcessor / num_threads_per_block) * 4;
        int num_blocks = prop.multiProcessorCount * num_blocks_per_sm;

        assert(num_blocks * num_threads_per_block <= N);

        random_fill_d_kernel<<<num_blocks, num_threads_per_block>>>(X, N);
    }

    template<typename T>
    __host__ bool is_equal_vector_h(T * X, T * Y, int N) {
        static_assert(std::is_floating_point<T>::value, "input X must be floating-point type pointer");
        volatile bool is_equal = true;

        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            if (! is_equal) 
                continue;

            if (std::abs(X[i] - Y[i]) >= 1e-6) {
                is_equal = false;
                printf("X[%d] = %f, Y[%d] = %f\n", i, X[i], i, Y[i]);
            }
        }

        return is_equal;
    }

    __host__ std::string formatBytes(size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unitIndex = 0;
        double size = bytes;

        while (size >= 1024 && unitIndex < 4) {
            size /= 1024;
            unitIndex++;
        }

        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << size << " " << units[unitIndex];
        return out.str();
    }
}

#endif // __UTILS_HPP__