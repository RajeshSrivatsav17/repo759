#include <iostream>
#include <string> 
#include "scan.cpp"

int main(int argc, char *argv[]){
    if(argc == 1)
    {
        std::cout<<"Provide a valid number N\n";
        return -1;
    }
    high_resolution_clock::time_point start_time;
    high_resolution_clock::time_point end_time;
    duration<double, std::milli> total_time;

    std::size_t n = (std::size_t)std::stoi(argv[1]);
    float * arr = new float[n];
    genRandomFloat(arr, n, -1.0, 1.0);
    start_time = high_resolution_clock::now();
    scan(arr,arr,n);
    end_time = high_resolution_clock::now();
    total_time = std::chrono::duration_cast< duration<double, std::milli> >(end_time - start_time);
    std::cout << total_time.count() << "\n" << arr[0] << "\n" << arr[n-1];
    delete[] arr;
}