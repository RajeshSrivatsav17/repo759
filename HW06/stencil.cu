#include<iostream>
#include<cuda.h>
#include<cmath>

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
    extern __shared__ float sh_mem[];
    float * sh_mask = sh_mem;
    float * sh_img = &sh_mask[2*R+1];
    float * sh_output = &sh_img[blockDim.x + 2*R];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    if(gid >= n) return;
    
    if(tid < 2*R+1) {
        sh_mask[tid] = mask[tid];
    }

    int block_start = blockIdx.x * blockDim.x;
    
    // Copies blockDim.x number of elements into the shared memeory
    // [R] to [R+blockDim.x-1] elements
    sh_img[tid + R] = image[gid];
    
    // Lower elements 
    // [0] to [R-1] elements
    if(tid < R) {
        int load_idx = block_start - R + tid;
        if(load_idx >= 0) {
            sh_img[tid] = image[load_idx];
        } else {
            sh_img[tid] = 1.0f;
        }
    }
    
    // Upper elements 
    // [R+blockDim.x] to [R+blockDim.x+R-1] elements
    if(tid < R) {
        int load_idx = block_start + blockDim.x + tid;
        if(load_idx < n) {
            sh_img[R + blockDim.x + tid] = image[load_idx];
        } else {
            sh_img[R + blockDim.x + tid] = 1.0f;
        }
    }
    
    __syncthreads();

    float result = 0.0f;
    for (int j = -R; j <= R; j++) {
        result += sh_img[tid + R + j] * sh_mask[j + R];
    }

    sh_output[tid] = result;
    output[gid] = sh_output[tid];
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
    
    cudaMemcpy(output, output_d, n* sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(image_d);
    cudaFree(mask_d);
    cudaFree(output_d);
}
