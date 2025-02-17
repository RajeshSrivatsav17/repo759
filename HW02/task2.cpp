#include <iostream>
#include <string> 
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>

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
        std::cout<<"Provide a valid number N\n";
        return -1;
    }

    std::size_t n = (std::size_t)std::stoi(argv[1]);
    std::size_t m = (std::size_t)std::stoi(argv[2]);

    high_resolution_clock::time_point start_time;
    high_resolution_clock::time_point end_time;
    duration<float, std::milli> total_time;
    
    float * f = new float[n*n];
    float * w = new float[m*m];
    float * g = new float[n*n];
    //Testing purposes
    // float f[] = {1.0,3.0,4.0,8.0,6.0,5.0,2.0,4.0,3.0,4.0,6.0,8.0,1.0,4.0,5.0,2.0};
    // float w[] = {0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0};
    genRandomFloat(f, n*n, -10, 10);
    genRandomFloat(w, m*m, -1 , 1);

    for(size_t i=0;i<n*n;i++){
        g[i] = 0;
    }
    start_time = high_resolution_clock::now();
    convolve(f, g, n, w, m);
    end_time = high_resolution_clock::now();
    total_time = std::chrono::duration_cast< duration<float, std::milli> >(end_time - start_time);
    std::cout << total_time.count() << "\n" << g[0] << "\n" << g[n*n-1] << "\n";
    delete[] f;
    delete[] g;
    delete[] w;
}
