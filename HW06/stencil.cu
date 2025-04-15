#include<iostream>
#include<cuda.h>
#include<cmath>

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    extern __shared__ float sh_mem[];

    // Layout in shared memory:
    float* sh_mask   = sh_mem;
    float* sh_img    = &sh_mask[2 * R + 1];
    float* sh_output = &sh_img[blockDim.x + 2 * R]; 

    int int_R = (int)R;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load mask (first 2*R+1 threads only)
    if (tid < (2 * int_R + 1)) {
        sh_mask[tid] = mask[tid];
    }

    // Load image into shared memory
    int block_start = blockIdx.x * blockDim.x;

    // Center 
    if (gid < n)
        sh_img[tid + int_R] = image[gid];
    else
        sh_img[tid + int_R] = 1.0f;

    // Left 
    if (tid < int_R) {
        int left_idx = block_start + tid - int_R;
        sh_img[tid] = (left_idx >= 0) ? image[left_idx] : 1.0f;
    }

    // Right 
    if (tid < int_R) {
        int right_idx = block_start + blockDim.x + tid;
        int halo_idx = blockDim.x + int_R + tid;
        sh_img[halo_idx] = (right_idx < n) ? image[right_idx] : 1.0f;
    }

    __syncthreads();  

    float result = 0.0f;
    if (gid < n) {
        for (int j = -int_R; j <= int_R; ++j) {
            result += sh_img[tid + int_R + j] * sh_mask[j + int_R];
        }
        sh_output[tid] = result;  // Store in shared memory (required by problem)
    }

    __syncthreads();  

    if (gid < n) {
        output[gid] = sh_output[tid];
    }
}

__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block){

    float* image_d;
    float* mask_d;
    float* output_d;
    cudaMalloc((void**)&image_d, n* sizeof(float));
    cudaMalloc((void**)&mask_d, (2*R+1)* sizeof(float));
    cudaMalloc((void**)&output_d, n* sizeof(float));

    cudaMemcpy(image_d, image, n* sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(mask_d, mask, (2*R+1)* sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks      = (n + threads_per_block - 1) / threads_per_block;
    int sharedMemSize   = ((2*R+1) + (threads_per_block+2*R) + threads_per_block) * sizeof(float);
    stencil_kernel<<<num_blocks, threads_per_block, sharedMemSize>>>(image_d, mask_d, output_d, n, R);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, n* sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(image_d);
    cudaFree(mask_d);
    cudaFree(output_d);
}
