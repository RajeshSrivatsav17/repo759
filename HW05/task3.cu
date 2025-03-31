#include<iostream>
#include<cuda.h>
#include <cstdlib>
#include <random>
#include <ctime>
#include "vscale.cuh"

void genRandomFloat(float * arr, std::size_t size, float min, float max){
    std::srand(std::time(0));
    for(long unsigned int i =0;i<size;i++){
        arr[i] = min + (((float)rand())/(float)RAND_MAX)*(max-min);
    }
}

int main(int argc, char *argv[]){
    if(argc < 2)
    {
        std::cout<<"Provide a valid number N\n";
        return -1;
    }
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::size_t n = (std::size_t)std::stol(argv[1]);
    int numBlocks = 0;
    int numThreads = 512;
    float *d_a, *d_b;
    float *h_a = new float[n]; 
    float *h_b = new float[n];

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));

    genRandomFloat(h_a, n, -10, 10);
    genRandomFloat(h_b, n, 0, 1);

    numBlocks = (n+numThreads-1) / numThreads;
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    vscale<<<numBlocks,numThreads>>>(d_a,d_b,n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_b, d_b, sizeof(int) * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout<<ms << "\n" << h_b[0] << "\n" << h_b[n-1] << "\n";
    return 0;
}
