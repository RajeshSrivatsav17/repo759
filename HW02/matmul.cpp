#ifndef MATMUL
#define MATMUL 1

#include <cstddef>
#include <vector>

void mmul1(const double* A, const double* B, double* C, const unsigned int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            C[i*n+j] = 0;
            for(int k=0;k<n;k++){
                C[i*n+j] += A[i*n+k]*B[k*n+j];
            }
        }
    }
}

void mmul2(const double* A, const double* B, double* C, const unsigned int n){
    for(int i=0;i<n;i++){
        for(int k=0;k<n;k++){
            C[i*n+k] = 0;
            for(int j=0;j<n;j++){
                C[i*n+j] += A[i*n+k]*B[k*n+j];
            }
        }
    }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n){
    for(int j=0;j<n;j++){
        for(int k=0;k<n;k++){
            C[j*n+k] = 0;
            for(int i=0;i<n;i++){
                C[i*n+j] += A[i*n+k]*B[k*n+j];
            }
        }
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            C[i*n+j] = 0;
            for(int k=0;k<n;k++){
                C[i*n+j] += A[i*n+k]*B[k*n+j];
            }
        }
    }
}


#endif