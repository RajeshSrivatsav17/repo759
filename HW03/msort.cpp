#include <iostream>
#include <string> 
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <omp.h>
#include "msort.h"

// Insertion Sort for threshold length
void insertionSort(int* arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int num = arr[i];
        int j = i - 1;
        // Shift arr[j] to right until num < arr[j] 
        while (j >= left && arr[j] > num) {
            arr[j + 1] = arr[j]; 
            j--;
        }
        arr[j + 1] = num;
    }
}

// Merge function
void merge(int* arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = new int[n1];
    int* R = new int[n2];

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    delete[] L;
    delete[] R;
}

///Recursive function call
void mergesort(int* arr, int left, int right, int threshold) {
    if ((right - left + 1) <= threshold) {
        insertionSort(arr, left, right);
        return;
    }

    int mid = left + (right - left) / 2;
    #pragma omp parallel
    {
        #pragma omp single  // Ensure only one thread spawns tasks
        {
            #pragma omp task
            mergesort(arr, left, mid, threshold);

            #pragma omp task
            mergesort(arr, mid + 1, right, threshold);
        }
    }
    merge(arr, left, mid, right);
}

void msort(int* arr, const std::size_t n, const std::size_t threshold){
    mergesort(arr, 0, n - 1, threshold);
}