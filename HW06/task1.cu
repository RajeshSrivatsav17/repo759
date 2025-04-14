#include<iostream>
#include<cuda.h>
#include<cmath>

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
    if(argc < 3)
    {
        std::cout <<"Please provide n and t arguments\n";
        return 0;
    }
    std::srand(std::time(0));

    int n = std::atoi(argv[1]);
    int threads_per_block = std::atoi(argv[2]);
    int size = n*n;

    float* A = new float[size];
    float* B = new float[size];
    float* C = new float[size];

    float* Ad;
    float* Bd;
    float* Cd;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&Ad, size* sizeof(float));
    cudaMalloc((void**)&Bd, size* sizeof(float));
    cudaMalloc((void**)&Cd, size* sizeof(float));

    assignRandomNumberToInputArrays(A, size);
    assignRandomNumberToInputArrays(B, size);

    cudaMemcpy(Ad, A, size* sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(Bd, B, size* sizeof(float), cudaMemcpyHostToDevice);

    int block_dim = sqrt(threads_per_block);

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);

    cudaEventRecord(start);
    matmul_kernel<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaDeviceSynchronize();
    cudaMemcpy(C, Cd, size* sizeof(float), cudaMemcpyDeviceToHost);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout<< C[size-1] <<"\n"<<ms << "\n";

    return 0;
}
