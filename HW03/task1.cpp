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

double genRandomDouble(double min, double max){
    std::srand(std::time(0));
    return (min + (((double)std::rand())/(double)RAND_MAX)*(max-min));
}

int genRandomInt(int min, int max){
    std::srand(std::time(0));
    return (min + (std::rand() % (max-min+1)));
}

void assignRandomNumberToInputArrays(double * arr, std::vector<double>& vec, std::size_t n){
    double random_num = 0.0;
    for(long unsigned int i =0;i<n;i++){
        random_num = genRandomDouble(0,10);
        arr[i] = random_num;
        vec.push_back(random_num);
    }  
}

int main(int argc, char* argv[]) {
    if(argc < 3)
    {
        cout <<"Please provide n and t arguments\n";
        return 0;
    }
    size_t n = std::atoi(argv[1]);
    size_t t = std::atoi(argv[2]);
    high_resolution_clock::time_point start_time;
    high_resolution_clock::time_point end_time;
    duration<double, std::milli> total_time;
    
    size_t total_elements = (n*n);
    double * A = new double[total_elements];
    double * B = new double[total_elements];
    std::vector<double> A_vector;
    std::vector<double> B_vector;

    double * C = new double[total_elements];

    assignRandomNumberToInputArrays(A, A_vector, total_elements);
    assignRandomNumberToInputArrays(B, B_vector, total_elements);

    for(size_t i=0;i<total_elements;i++){
        C[i] = 0.0;
    }
    omp_set_num_threads(t);
    cout<<"Number of OMP threads" << omp_get_num_threads() << "\n";
    start_time = high_resolution_clock::now();
    mmul(A, B, C, size);
    end_time = high_resolution_clock::now();
    total_time = std::chrono::duration_cast< duration<double, std::milli> >(end_time - start_time);
    std::cout << C[0] << "\n" << C[total_elements-1] << "\n" <<total_time.count() << "\n";

    delete[] A;
    delete[] B;
    delete[] C;

}