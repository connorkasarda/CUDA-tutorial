#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include "threads.h"

__global__ void AddInts(int* a, int* b, int count)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < count)
    {
        a[thread_id] += b[thread_id];
    }
}

void RunThreadsExample()
{
    std::cout << "Running thread example..." << std::endl;

    // Create host (hence the h_ prefix) arrays through random generation
    srand(time(NULL));
    int count = 100;
    int *h_a = new int[count];
    int *h_b = new int[count];
    for (int i = 0; i < count; ++i)
    {
        h_a[i] = rand() % 1000;
	h_b[i] = rand() % 1000;
    }

    // Reveal the results for the first 5 elements of both arrays
    for (int i = 0; i < 5; ++i)
    {
        std::cout << h_a[i] << " " << h_b[i] << std::endl;
    }

    // Device copies of the arrays
    int *d_a, *d_b;

    // Check if memory allocation for device arrays fails
    if (cudaMalloc(&d_a, sizeof(int) * count) != cudaSuccess)
    {
        std::cout << "device copy of array a failed!" << std::endl;
        return;
    }
    if (cudaMalloc(&d_b, sizeof(int) * count) != cudaSuccess)
    {
	std::cout << "device copy of array b failed!" << std::endl;
	cudaFree(d_a);
        return;	    
    }

    // Check if copying host arrays to GPU/device arrays failed
    if (cudaMemcpy(d_a, h_a, sizeof(int) * count, cudaMemcpyHostToDevice) !=
        cudaSuccess)
    {
        std::cout << "Copying host array a to "
	    << "GPU/device array a failed!"
	    << std::endl;
	cudaFree(d_a);
	cudaFree(d_b);
	return;
    } 
    if (cudaMemcpy(d_b, h_b, sizeof(int) * count, cudaMemcpyHostToDevice) !=
        cudaSuccess)
    {
        std::cout << "Copying host array b to "
	    << "GPU/device array b failed!"
	    << std::endl;
	cudaFree(d_a);
	cudaFree(d_b);
	return;
    }
    
    // Launch configuration
    AddInts<<<count / 256 + 1, 256>>>(d_a, d_b, count);

    // Did GPU generated results copy from device to host or did it fail?
    if (cudaMemcpy(h_a, d_a, sizeof(int) * count, cudaMemcpyDeviceToHost) !=
        cudaSuccess)
    {
        std::cout << "" << std::endl;
        delete[] h_a;
        delete[] h_b;
	cudaFree(d_a);
	cudaFree(d_b);
        return;
    }

    // Print addition results
    for (int i = 0; i < 5; ++i)
    {
        std::cout << "element @ " << i << ": " << h_a[i] << std::endl;
    }

    // Make sure to delete host arrays and memory
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;

    std::cout << std::endl;
} // void RunThreadsExample()
