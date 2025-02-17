#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

void genRandomFloat(float * arr, std::size_t size, float min, float max){
    std::srand(std::time(0));
    for(long unsigned int i =0;i<size;i++){
        arr[i] = min + (((float)rand())/(float)RAND_MAX)*(max-min);
    }
}

void scan(const float *arr, float *output, std::size_t n){
    output[0] = arr[0];
    for(long unsigned int i=1;i<n;i++){
        output[i] = arr[i]+output[i-1];
    }
}