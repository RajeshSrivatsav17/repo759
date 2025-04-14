#include<iostream>
#include<cuda.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;
    
    float sum = 0;
    
    // Number of tiles = N / BLOCK_SIZE
    for(int k = 0; k < N; k += BLOCK_SIZE) {
        sA[ty][tx] = A[row*N + k + tx]; 
        sB[ty][tx] = B[(k + ty)*N + col]; 
        __syncthreads();
        
        for(int i = 0; i < BLOCK_SIZE; i++)
            sum += sA[ty][i] * sB[i][tx];
        
        __syncthreads();
    }
    
    if(row < N && col < N)
        C[row*N + col] = sum;
}
