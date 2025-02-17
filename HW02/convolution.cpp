#ifndef CONVOLUTION
#define CONVOLUTION 1
#include "matmul.h"

#include <cstddef>
#include <vector>

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    size_t i_p=0;
    size_t j_p=0;
    for(size_t x=0;x<n;x++){
        for(size_t y=0;y<n;y++){
            for(size_t i=0;i<m;i++){
                for(size_t j=0;j<m;j++){
                    i_p = (x+i-((m-1)/2));
                    j_p = (y+j-((m-1)/2));
                    if(i_p <n && i_p >=0 && j_p <n && j_p >= 0)
                        output[x*n+y] += mask[i*m+j]*image[i_p*n +j_p];
                    else if(( (i_p <n && i_p >=0) && !(j_p <n && j_p >= 0)) || 
                        (!(i_p <n && i_p >=0) && (j_p <n && j_p >= 0)))
                        output[x*n+y] += mask[i*m+j]; // f[i,j] is 1 if one of the i, j <0
                    // else 
                    //     f[i,j] = 0;                    
                }
            }
        }
    }

}
#endif //CONVOLUTION