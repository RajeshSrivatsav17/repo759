#include <iostream>
#include <string> 
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <omp.h>
#include "msort.h"
using std::chrono::high_resolution_clock;
using std::chrono::duration;

double genRandomDouble(double min, double max){
    std::srand(std::time(0));
    return (min + (((double)std::rand())/(double)RAND_MAX)*(max-min));
}

int genRandomInt(int min, int max){
    return (min + (std::rand() % (max-min+1)));
}

void assignRandomNumberToInputArrays(int * arr, std::size_t n){
    double random_num = 0.0;
    for(long unsigned int i =0;i<n;i++){
        random_num = genRandomInt(-1000,1000);
        arr[i] = random_num;
    }  
}

int main(int argc, char* argv[]) {
    if(argc < 4)
    {
        std::cout <<"Please provide n and t arguments\n";
        return 0;
    }
    size_t n = std::atoi(argv[1]);
    size_t t = std::atoi(argv[2]);
    size_t threshold = std::atoi(argv[2]);
    std::srand(std::time(0));
    std::cout<< n << " " << t << " " << threshold << "\n";
    omp_set_num_threads(t);
    
    high_resolution_clock::time_point start_time;
    high_resolution_clock::time_point end_time;
    duration<double, std::milli> total_time;
    
    int * arr = new int[n];
    assignRandomNumberToInputArrays(arr,n);

    start_time = high_resolution_clock::now();
    msort(arr, n, threshold);
    end_time = high_resolution_clock::now();
    // for(int i =0; i<n;i++){
    //     std::cout << arr[i] << "\n";
    // }
    // std::cout << "\n";
    total_time = std::chrono::duration_cast< duration<double, std::milli> >(end_time - start_time);
    std::cout << arr[0] << "\n" << arr[n-1] << "\n" <<total_time.count() << "\n";

    delete[] arr;
}