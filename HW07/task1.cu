#include<iostream>
#include<cuda.h>
#include<cmath>
#include<ctime>
#include "matmul.cuh"

float genRandomfloat(float min, float max){
    return (min + (((float)std::rand())/(float)RAND_MAX)*(max-min));
}

double genRandomdouble(double min, double max){
    return (min + (((double)std::rand())/(double)RAND_MAX)*(max-min));
}

int genRandomInt(int min, int max) {
    return min + std::rand() % (max - min + 1);
}

void assignRandomNumberToInputArraysFloat(float * arr, std::size_t n){
    float random_num = 0.0;
    for(long unsigned int i =0;i<n;i++){
        random_num = genRandomfloat(-1,1);
        arr[i] = random_num;
    }  
}

void assignRandomNumberToInputArraysDouble(double * arr, std::size_t n){
    double random_num = 0.0;
    for(long unsigned int i =0;i<n;i++){
        random_num = genRandomdouble(-1,1);
        arr[i] = random_num;
    }  
}

void assignRandomNumberToInputArraysInt(int * arr, std::size_t n){
    int random_num = 0.0;
    for(long unsigned int i =0;i<n;i++){
        random_num = genRandomInt(-1,1);
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
    int block_dim = std::atoi(argv[2]);
    int block_dim_per_row_or_col = std::sqrt(block_dim);
    int size = n*n;

    int* A1 = new int[size];
    int* B1 = new int[size];
    int* C1 = new int[size];

    float* A2 = new float[size];
    float* B2 = new float[size];
    float* C2 = new float[size];

    double* A3 = new double[size];
    double* B3 = new double[size];
    double* C3 = new double[size];

    int* Ad1;
    int* Bd1;
    int* Cd1;

    float* Ad2;
    float* Bd2;
    float* Cd2;

    double* Ad3;
    double* Bd3;
    double* Cd3;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&Ad1, size* sizeof(int));
    cudaMalloc((void**)&Bd1, size* sizeof(int));
    cudaMalloc((void**)&Cd1, size* sizeof(int));

    cudaMalloc((void**)&Ad2, size* sizeof(float));
    cudaMalloc((void**)&Bd2, size* sizeof(float));
    cudaMalloc((void**)&Cd2, size* sizeof(float));

    cudaMalloc((void**)&Ad3, size* sizeof(double));
    cudaMalloc((void**)&Bd3, size* sizeof(double));
    cudaMalloc((void**)&Cd3, size* sizeof(double));

    assignRandomNumberToInputArraysInt(A1, size);
    assignRandomNumberToInputArraysInt(B1, size);
    assignRandomNumberToInputArraysFloat(A2, size);
    assignRandomNumberToInputArraysFloat(B2, size);
    assignRandomNumberToInputArraysDouble(A3, size);
    assignRandomNumberToInputArraysDouble(B3, size);

    cudaMemcpy(Ad1, A1, size* sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(Bd1, B1, size* sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(Ad2, A2, size* sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(Bd2, B2, size* sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(Ad3, A3, size* sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(Bd3, B3, size* sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    matmul_1(Ad1, Bd1, Cd1, n, block_dim_per_row_or_col);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(C1, Cd1, size* sizeof(int), cudaMemcpyDeviceToHost);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout<< C1[0] << "\n" << C1[size-1] <<"\n"<<ms << "\n";

    cudaEventRecord(start, 0);
    cudaEventRecord(stop, 0);

    cudaEventRecord(start);
    matmul_2(Ad2, Bd2, Cd2, n, block_dim_per_row_or_col);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(C2, Cd2, size* sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&ms, start, stop);

    std::cout<< C2[0] << "\n" << C2[size-1] <<"\n"<<ms << "\n";

    cudaEventRecord(start, 0);
    cudaEventRecord(stop, 0);

    cudaEventRecord(start);
    matmul_3(Ad3, Bd3, Cd3, n, block_dim_per_row_or_col);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(C3, Cd3, size* sizeof(double), cudaMemcpyDeviceToHost);

    std::cout<< C3[0] << "\n" << C3[size-1] <<"\n"<<ms << "\n";

    cudaEventRecord(start, 0);
    cudaEventRecord(stop, 0);
    delete[] A1; delete[] B1; delete[] C1;
    delete[] A2; delete[] B2; delete[] C2;
    delete[] A3; delete[] B3; delete[] C3;
    
    cudaFree(Ad1); cudaFree(Bd1); cudaFree(Cd1);
    cudaFree(Ad2); cudaFree(Bd2); cudaFree(Cd2);
    cudaFree(Ad3); cudaFree(Bd3); cudaFree(Cd3);

    return 0;
}