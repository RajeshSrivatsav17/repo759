#include "matmul.cuh"
#include <cuda_runtime.h>
#include <iostream>

template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, int n) {
    extern __shared__ unsigned char smem[];
    T* tile_A = reinterpret_cast<T*>(smem);
    T* tile_B = reinterpret_cast<T*>(smem + blockDim.x * blockDim.y * sizeof(T));

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = 0;

    for (int tile = 0; tile < (n + blockDim.x - 1) / blockDim.x; ++tile) {
        int tiled_row = row;
        int tiled_col = tile * blockDim.x + threadIdx.x;

        if (tiled_row < n && tiled_col < n)
            tile_A[threadIdx.y * blockDim.x + threadIdx.x] = A[tiled_row * n + tiled_col];
        else
            tile_A[threadIdx.y * blockDim.x + threadIdx.x] = 0;

        tiled_row = tile * blockDim.y + threadIdx.y;
        tiled_col = col;

        if (tiled_row < n && tiled_col < n)
            tile_B[threadIdx.y * blockDim.x + threadIdx.x] = B[tiled_row * n + tiled_col];
        else
            tile_B[threadIdx.y * blockDim.x + threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k) {
            sum += tile_A[threadIdx.y * blockDim.x + k] * tile_B[k * blockDim.x + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

template <typename T>
__host__ void matmul(const T* A, const T* B, T* C, unsigned int n, unsigned int block_dim) {
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);

    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(T);  // A and B tiles

    matmul_kernel<<<grid, block, shared_mem_size>>>(A, B, C, n);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess)
    std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    cudaDeviceSynchronize();
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    matmul<int>(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    matmul<float>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    matmul<double>(A, B, C, n, block_dim);
}
