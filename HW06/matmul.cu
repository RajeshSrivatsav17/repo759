#include<iostream>
#include<cuda.h>
#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    extern __shared__ float shmem[];
    float * sA = shmem;
    float * sB = (float*)&sA[blockDim.x*blockDim.x];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    
    float sum = 0;
    
    // Number of tiles = N / BLOCK_SIZE
    for(int k = 0; k < n; k += blockDim.x) {
        if (row < n && (k + tx) < n)
            sA[ty * blockDim.x + tx] = A[row * n + k + tx];
        else
            sA[ty * blockDim.x + tx] = 0;

        if ((k + ty) < n && col < n)
            sB[ty * blockDim.x + tx] = B[(k + ty) * n + col];
        else
            sB[ty * blockDim.x + tx] = 0;
        __syncthreads();
        
        for(int i = 0; i < blockDim.x; i++)
            sum += sA[ty * blockDim.x + i] * sB[i * blockDim.x + tx];
        
        __syncthreads();
    }
    
    if(row < n && col < n)
        C[row*n + col] = sum;
}
