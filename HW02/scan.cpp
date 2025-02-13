#include <cstdlib>
#include <ctime>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

void genRandomFloat(float * arr, int size, float min, float max){
    for(int i =0;i<size;i++){
        arr[i] = min + (rand()/RAND_MAX)*(max-min);
    }
}

void scan(const float *arr, float *output, std::size_t n){
    for(int i=1;i<n;i++){
        output[i] = arr[i]+arr[i-1];
    }
}