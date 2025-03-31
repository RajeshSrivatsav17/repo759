#include<iostream>
#include<cuda.h>
#include <random>

__global__ void kernel(int *data, int a)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    data[index] = a * threadIdx.x + blockIdx.x;
}

int main()
{
    const int numElems = 16;
    int hA[numElems],*dA;
    int a = std::rand();
    cudaMalloc((void**)&dA, sizeof(int) * numElems);
    cudaMemset(dA, 0, numElems * sizeof(int));
    kernel<<<2,8>>>(dA,a);
    cudaMemcpy(hA, dA, sizeof(int) * numElems, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0;i<numElems;i++){
        printf("%d ", hA[i]);
    }
    printf("\n");

    return 0;
}
