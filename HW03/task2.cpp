#include <iostream>
#include <string> 
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <omp.h>


#include "convolution.h"
using std::chrono::high_resolution_clock;
using std::chrono::duration;

void genRandomFloat(float * arr, std::size_t size, float min, float max){
    std::srand(std::time(0));
    for(long unsigned int i =0;i<size;i++){
        arr[i] = min + (((float)rand())/(float)RAND_MAX)*(max-min);
    }
}

int main(int argc, char *argv[]){
    if(argc < 3)
    {
        std::cout <<"Please provide n and t arguments\n";
        return 0;
    }

    std::size_t n = (std::size_t)std::stoi(argv[1]);
    std::size_t t = (std::size_t)std::stoi(argv[2]);

    high_resolution_clock::time_point start_time;
    high_resolution_clock::time_point end_time;
    duration<float, std::milli> total_time;
    
    float * f = new float[n*n];
    float * w = new float[3*3];
    float * g = new float[n*n];

    genRandomFloat(f, n*n, -10, 10);
    genRandomFloat(w, 3*3, -1 , 1);

    for(size_t i=0;i<n*n;i++){
        g[i] = 0;
    }
    omp_set_num_threads(t);
    std::cout<<"Number of OMP threads" << omp_get_num_threads() << "\n";

    start_time = high_resolution_clock::now();
    convolve(f, g, n, w, 3);
    end_time = high_resolution_clock::now();
    total_time = std::chrono::duration_cast< duration<float, std::milli> >(end_time - start_time);
    std::cout << g[0] << "\n" << g[n*n-1] << "\n" << total_time.count() << "\n" ;
    delete[] f;
    delete[] g;
    delete[] w;
}