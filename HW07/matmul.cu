#include "matmul.cuh"
#include <cuda.h>
#include <stdio.h>

__global__ void matmul_kernel_float(float *A, float *B, float *C, unsigned int n) {
    extern __shared__ int sh_mem[]; // Shared memory for A and B
    float *As = sh_mem;
    float *Bs = &sh_mem[blockDim.x * blockDim.y];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    for (unsigned int tile_idx = 0; tile_idx < (n + blockDim.x - 1) / blockDim.x; ++tile_idx) {
        if (row < n && tile_idx * blockDim.x + threadIdx.x < n) {
            As[threadIdx.y * blockDim.x + threadIdx.x] = A[row * n + tile_idx * blockDim.x + threadIdx.x];
        } else {
            As[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        if (col < n && tile_idx * blockDim.y + threadIdx.y < n) {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(tile_idx * blockDim.y + threadIdx.y) * n + col];
        } else {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        __syncthreads(); 

        for (unsigned int k = 0; k < blockDim.x; ++k) {
            sum += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
        }

        __syncthreads(); 
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

__global__ void matmul_kernel_int(int *A, int *B, int *C, unsigned int n) {
    extern __shared__ int sh_mem[]; // Shared memory for A and B
    int *As = sh_mem;
    int *Bs = &sh_mem[blockDim.x * blockDim.y];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    for (unsigned int tile_idx = 0; tile_idx < (n + blockDim.x - 1) / blockDim.x; ++tile_idx) {
        if (row < n && tile_idx * blockDim.x + threadIdx.x < n) {
            As[threadIdx.y * blockDim.x + threadIdx.x] = A[row * n + tile_idx * blockDim.x + threadIdx.x];
        } else {
            As[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        if (col < n && tile_idx * blockDim.y + threadIdx.y < n) {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(tile_idx * blockDim.y + threadIdx.y) * n + col];
        } else {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        __syncthreads(); 

        for (unsigned int k = 0; k < blockDim.x; ++k) {
            sum += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
        }

        __syncthreads(); 
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

__global__ void matmul_kernel_int(double *A, double *B, double *C, unsigned int n) {
    extern __shared__ int sh_mem[]; // Shared memory for A and B
    double *As = sh_mem;
    double *Bs = &sh_mem[blockDim.x * blockDim.y];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    for (unsigned int tile_idx = 0; tile_idx < (n + blockDim.x - 1) / blockDim.x; ++tile_idx) {
        if (row < n && tile_idx * blockDim.x + threadIdx.x < n) {
            As[threadIdx.y * blockDim.x + threadIdx.x] = A[row * n + tile_idx * blockDim.x + threadIdx.x];
        } else {
            As[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        if (col < n && tile_idx * blockDim.y + threadIdx.y < n) {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(tile_idx * blockDim.y + threadIdx.y) * n + col];
        } else {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        __syncthreads(); 

        for (unsigned int k = 0; k < blockDim.x; ++k) {
            sum += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
        }

        __syncthreads(); 
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    dim3 threadsPerBlock(block_dim, block_dim);
    dim3 numBlocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t sh_mem = 2 * block_dim * block_dim * sizeof(int); 

    matmul_kernel<<<numBlocks, threadsPerBlock, sh_mem>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim)
{
    dim3 threadsPerBlock(block_dim, block_dim);
    dim3 numBlocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t sh_mem = 2 * block_dim * block_dim * sizeof(float); 

    matmul_kernel<<<numBlocks, threadsPerBlock, sh_mem>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim)
{
    dim3 threadsPerBlock(block_dim, block_dim);
    dim3 numBlocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t sh_mem = 2 * block_dim * block_dim * sizeof(double); 

    matmul_kernel<<<numBlocks, threadsPerBlock, sh_mem>>>(A, B, C, n);
    cudaDeviceSynchronize();
}