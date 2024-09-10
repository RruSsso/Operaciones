#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

// Addition function on GPU
__global__ void add_gpu(int* a, int* b, int* result)
{
    *result = *a + *b;
}

// Addition function on CPU
void add_cpu(int* a, int* b, int* result)
{
    *result = *a + *b;
}

// Subtraction function on GPU
__global__ void subtract_gpu(int* a, int* b, int* result)
{
    *result = *b - *a;
}

// Subtraction function on CPU
void subtract_cpu(int* a, int* b, int* result)
{
    *result = *b - *a;
}

// Multiplication function on GPU
__global__ void multiply_gpu(int* a, int* b, int* result)
{
    *result = *a * *b;
}

// Multiplication function on CPU
void multiply_cpu(int* a, int* b, int* result)
{
    *result = *a * *b;
}

// Division function on GPU
__global__ void divide_gpu(int* a, int* b, int* result)
{
    if (*a != 0)
        *result = *b / *a;
    else
        *result = 0;  // Avoid division by zero
}

// Division function on CPU
void divide_cpu(int* a, int* b, int* result)
{
    if (*a != 0)
        *result = *b / *a;
    else
        *result = 0;  // Avoid division by zero
}

int main()
{
    int a = 2;
    int b = 5;
    int result;

    // CPU Operations
    std::cout << "CPU Operations:" << std::endl;

    add_cpu(&a, &b, &result);
    std::cout << "Addition (CPU): " << result << std::endl;

    subtract_cpu(&a, &b, &result);
    std::cout << "Subtraction (CPU): " << result << std::endl;

    multiply_cpu(&a, &b, &result);
    std::cout << "Multiplication (CPU): " << result << std::endl;

    divide_cpu(&a, &b, &result);
    std::cout << "Division (CPU): " << result << std::endl;

    // GPU Memory Allocation
    int* a_gpu, * b_gpu, * result_gpu;
    int size = sizeof(int);

    cudaMalloc((void**)&a_gpu, size);
    cudaMalloc((void**)&b_gpu, size);
    cudaMalloc((void**)&result_gpu, size);

    cudaMemcpy(a_gpu, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, &b, size, cudaMemcpyHostToDevice);

    // GPU Operations
    std::cout << "\nGPU Operations:" << std::endl;

    add_gpu << <1, 1 >> > (a_gpu, b_gpu, result_gpu);
    cudaMemcpy(&result, result_gpu, size, cudaMemcpyDeviceToHost);
    std::cout << "Addition (GPU): " << result << std::endl;

    subtract_gpu << <1, 1 >> > (a_gpu, b_gpu, result_gpu);
    cudaMemcpy(&result, result_gpu, size, cudaMemcpyDeviceToHost);
    std::cout << "Subtraction (GPU): " << result << std::endl;

    multiply_gpu << <1, 1 >> > (a_gpu, b_gpu, result_gpu);
    cudaMemcpy(&result, result_gpu, size, cudaMemcpyDeviceToHost);
    std::cout << "Multiplication (GPU): " << result << std::endl;

    divide_gpu << <1, 1 >> > (a_gpu, b_gpu, result_gpu);
    cudaMemcpy(&result, result_gpu, size, cudaMemcpyDeviceToHost);
    std::cout << "Division (GPU): " << result << std::endl;

    // Free GPU memory
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(result_gpu);

    return 0;
}
