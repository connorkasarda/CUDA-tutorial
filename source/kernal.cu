#include <iostream>
#include <cuda.h>
#include "kernal.h"

__global__ void AddIntsCUDA(int* a, int* b)
{
    a[0] += b[0];
}

void RunMyFirstKernal()
{
    std::cout << "Running first kernal example..." << std::endl;

    int a = 5, b = 9;
    
    // These are device pointers, hence the d_ prefix
    int *d_a, *d_b;

    // Typically want to detect if these are successful
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    
    AddIntsCUDA<<<1, 1>>>(d_a, d_b);

    cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "The answer is " << a << std::endl;

    // Make sure to free memory that was allocated
    cudaFree(d_a);
    cudaFree(d_b);

    std::cout << std::endl;
}
