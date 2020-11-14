#ifndef MY_KERNEL_AUX_FUNCTIONS
#define MY_KERNEL_AUX_FUNCTIONS

#include "matrix.hpp"
#include <stdio.h>
#include <cuda_runtime.h>

#define ull unsigned long long

__global__ void launchKernel(const EigenCUDA::Matrix &MatA,
                             const EigenCUDA::Matrix &MatB, double *finalMat,
                             ull MATRIX_DIMENSIONS, int threads) {
  // const int i = blockIdx.x * blockDim.x + threadIdx.x;

  ull nrows = MATRIX_DIMENSIONS, ncols = MATRIX_DIMENSIONS;

  double *rowArray = NULL, *colArray = NULL;
  cudaMalloc(&rowArray, MATRIX_DIMENSIONS * sizeof(double));
  cudaMalloc(&colArray, MATRIX_DIMENSIONS * sizeof(double));

  for (ull i = 0; i < nrows; ++i) {
    for (ull j = 0; j < ncols; ++j) {
      // rowArray = getRowFromMatrix(MatA, i);
      // colArray = getColFromMatrix(MatB, i);
    }
  }

  cudaFree(rowArray);
  cudaFree(colArray);

  return ;
}

#endif