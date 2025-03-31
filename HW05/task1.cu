#include<iostream>
#include<cuda.h>

__global__ void factorial_kernel()
{
    int fact = 1;
    int num = threadIdx.x+1;
    for(int i=2;i<=num;i++){
        fact*= i;
    }
    printf("%d!=%d\n",threadIdx.x+1,fact);
}

int main()
{
    const int numThreads = 8;
    factorial_kernel<<<1,numThreads>>>();
    cudaDeviceSynchronize();
    return 0;
}
