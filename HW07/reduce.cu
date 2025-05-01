#include "reduce.cuh"
#include <cuda.h>
#include <stdio.h>

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sh_mem[];
    
    // Set up thread and block indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Perform first level of reduction upon reading from global memory
    // Each thread loads and processes two elements
    sh_mem[tid] = (idx < n) ? g_idata[idx] : 0.0f;
    if (idx + blockDim.x < n) {
        sh_mem[tid] += g_idata[idx + blockDim.x];
    }
    __syncthreads();
    
    // Perform in-shared-memory reduction
    for (int s = blockDim.x / 2; s > 0; s = s >> 1) {
        if (tid < s) {
            sh_mem[tid] += sh_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sh_mem[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N, 
                    unsigned int threads_per_block) {
    float *current_input = *input;
    float *current_output = *output;
    unsigned int current_n = N;

    while (current_n > 1) {
        // Calculate number of blocks needed (half as many due to first-add optimization)
        unsigned int num_blocks = (current_n + (threads_per_block * 2 - 1)) / (threads_per_block * 2);
        num_blocks = (num_blocks > 0) ? num_blocks : 1;
        
        // Launch kernel with dynamically allocated shared memory
        size_t shared_mem_size = threads_per_block * sizeof(float);
        reduce_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(current_input, current_output, current_n);
        
        // Update for next iteration
        current_n = num_blocks;
        
        // Swap input and output pointers for the next iteration
        std::swap(current_input, current_output);
    }
    
    // Synchronize to ensure all reduction steps are completed
    cudaDeviceSynchronize();
    
    // If the final result is not in the original input array, copy it there
    if (current_input != *input) {
        cudaMemcpy(*input, current_input, sizeof(float), cudaMemcpyDeviceToDevice);
    }
}