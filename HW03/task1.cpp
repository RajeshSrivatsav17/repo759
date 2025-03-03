#include <iostream>
#include <string> 
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <omp.h>
#include "matmul.h"
using std::chrono::high_resolution_clock;
using std::chrono::duration;

float genRandomfloat(float min, float max){
    return (min + (((float)std::rand())/(float)RAND_MAX)*(max-min));
}

int genRandomInt(int min, int max){
    std::srand(std::time(0));
    return (min + (std::rand() % (max-min+1)));
}

void assignRandomNumberToInputArrays(float * arr, std::size_t n){
    float random_num = 0.0;
    for(long unsigned int i =0;i<n;i++){
        random_num = genRandomfloat(0,10);
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

    size_t n = std::atoi(argv[1]);
    size_t t = std::atoi(argv[2]);
    high_resolution_clock::time_point start_time;
    high_resolution_clock::time_point end_time;
    duration<double, std::milli> total_time;
    
    size_t total_elements = (n*n);
    float * A = new float[total_elements];
    float * B = new float[total_elements];

    float * C = new float[total_elements];

    assignRandomNumberToInputArrays(A, total_elements);
    assignRandomNumberToInputArrays(B, total_elements);

    for(size_t i=0;i<total_elements;i++){
        C[i] = 0.0;
    }
    omp_set_num_threads(t);
    start_time = high_resolution_clock::now();
    mmul(A, B, C, n);
    end_time = high_resolution_clock::now();
    total_time = std::chrono::duration_cast< duration<double, std::milli> >(end_time - start_time);
    std::cout << C[0] << "\n" << C[total_elements-1] << "\n" <<total_time.count() << "\n";

    delete[] A;
    delete[] B;
    delete[] C;

}