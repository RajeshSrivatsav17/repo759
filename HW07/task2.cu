#include<iostream>
#include<cuda.h>
#include<cmath>
#include<ctime>
#include "reduce.cuh"

double genRandomfloat(double min, double max){
    return (min + (((double)std::rand())/(double)RAND_MAX)*(max-min));
}

void assignRandomNumberToInputArrays(float * arr, std::size_t n){
    float random_num = 0.0;
    for(long unsigned int i =0;i<n;i++){
        random_num = genRandomfloat(-1,1);
        arr[i] = random_num;
    }  
}

int main(int argc, char* argv[]) {
    if(argc < 3)
    {
        std::cout <<"Please provide n and t arguments\n";
        return 0;
    }
    std::srand(std::time(0));

    int n = std::atoi(argv[1]);
    int threads_per_block = std::atoi(argv[2]);
    int size = n*n;
    int num_blocks = (n + threads_per_block * 2 - 1) / (threads_per_block * 2);

    float* input = new float[size];

    float* input_d;
    float* output_d;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&input_d, size* sizeof(float));
    cudaMalloc((void**)&output_d, num_blocks* sizeof(float));

    assignRandomNumberToInputArraysInt(input, size);
    cudaMemcpy(input_d, input, size* sizeof(float), cudaMemcpyHostToDevice); 

    cudaEventRecord(start);
    reduce(&input_d, &output_d, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(&input, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout<< input[0] <<"\n"<<ms << "\n";

    return 0;
}