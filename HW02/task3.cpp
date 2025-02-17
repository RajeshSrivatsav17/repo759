#include <iostream>
#include <string> 
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>

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

int main(){
    high_resolution_clock::time_point start_time;
    high_resolution_clock::time_point end_time;
    duration<double, std::milli> total_time;
    
    size_t size = (int) genRandomInt(1000,2000);
    size_t total_elements = (size*size);
    double * A = new double[total_elements];
    double * B = new double[total_elements];
    std::vector<double> A_vector;
    std::vector<double> B_vector;

    double * C1 = new double[total_elements];
    double * C2 = new double[total_elements];
    double * C3 = new double[total_elements];
    double * C4 = new double[total_elements];

    assignRandomNumberToInputArrays(A, A_vector, total_elements);
    assignRandomNumberToInputArrays(B, B_vector, total_elements);

    for(size_t i=0;i<total_elements;i++){
        C1[i] = 0.0;
        C2[i] = 0.0;
        C3[i] = 0.0;
        C4[i] = 0.0;
    }
    std::cout << size << "\n" ;

    start_time = high_resolution_clock::now();
    mmul1(A, B, C1, size);
    end_time = high_resolution_clock::now();
    total_time = std::chrono::duration_cast< duration<double, std::milli> >(end_time - start_time);
    std::cout << total_time.count() << "\n" << C1[total_elements-1] << "\n";

    start_time = high_resolution_clock::now();
    mmul2(A, B, C2, size);
    end_time = high_resolution_clock::now();
    total_time = std::chrono::duration_cast< duration<double, std::milli> >(end_time - start_time);
    std::cout << total_time.count() << "\n" << C2[total_elements-1] << "\n";

    start_time = high_resolution_clock::now();
    mmul3(A, B, C3, size);
    end_time = high_resolution_clock::now();
    total_time = std::chrono::duration_cast< duration<double, std::milli> >(end_time - start_time);
    std::cout << total_time.count() << "\n" << C3[total_elements-1] << "\n";

    start_time = high_resolution_clock::now();
    mmul4(A_vector, B_vector, C4, size);
    end_time = high_resolution_clock::now();
    total_time = std::chrono::duration_cast< duration<double, std::milli> >(end_time - start_time);
    std::cout << total_time.count() << "\n" << C4[total_elements-1] << "\n";

    // for(size_t i=0;i<(total_elements);i++){
    //     // printf("%f ", C1[i]);
    //     if((C1[i] != C2[i]) || (C3[i] != C1[i]) || (C2[i] != C3[i]) || (C4[i] != C1[i]))
    //         std::cout << "Not working as expected\n";
    // }
    // printf("\n");
    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    delete[] C3;

}
