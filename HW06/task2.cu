#include<iostream>
#include<cuda.h>
#include<cmath>
#include<ctime>
#include "stencil.cuh"

float genRandomfloat(float min, float max){
    return (min + (((float)std::rand())/(float)RAND_MAX)*(max-min));
}

void assignRandomNumberToInputArrays(float * arr, std::size_t n){
    float random_num = 0.0;
    for(long unsigned int i =0;i<n;i++){
        random_num = genRandomfloat(-1,1);
        arr[i] = random_num;
    }  
}

int main(int argc, char* argv[]) {
    if(argc < 4)
    {
        std::cout <<"Please provide n and t arguments\n";
        return 0;
    }
    std::srand(std::time(0));

    int n = std::atoi(argv[1]);
    int R = std::atoi(argv[2]);
    int threads_per_block = std::atoi(argv[3]);

    float* image = new float[n];
    float* mask = new float[2*R+1];
    float* output = new float[n];

    assignRandomNumberToInputArrays(image, n);
    assignRandomNumberToInputArrays(mask, 2*R+1);
    
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stencil(image, mask, output, n, R, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << output[n-1] << "\n" << ms << "\n";
    return 0;
}
