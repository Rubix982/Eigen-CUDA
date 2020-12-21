#ifndef KERNEL_AUX_H
#define KERNEL_AUX_H

#include "matrix.hpp"

#define ull unsigned long long

__device__ double *multiplyArrays(double *arrayA, double *ArrayB,
                                  ull arraySize) {

  double *result = new double[arraySize];

  for (ull i = 0; i < arraySize; ++i) {
    result[i] = arrayA[i] * ArrayB[i];
  }

  return result;
}

__global__ void launchKernel(const EigenCUDA::Matrix &MatA,
                             const EigenCUDA::Matrix &MatB, double **finalMat,
                             ull dim, int threads) {  
  printf("here\n");
  ull nrows = dim, ncols = dim;

  double *rowArray = NULL, *colArray = NULL;
  rowArray = new double[dim];
  colArray = new double[dim];

  for (ull i = 0; i < nrows; ++i) {

    double *resultArray = NULL;

    for (ull j = 0; j < ncols; ++j) {
      for (ull col = 0; col < dim; ++col) {
        rowArray[i] = MatA.__Mat[i][col];
      }

      printf("here\n");

      for (ull row = 0; row < dim; ++row) {
        colArray[i] = MatB.__Mat[row][j];
      }

      resultArray = multiplyArrays(rowArray, colArray, dim);
    }

    for (int j = 0; j < ncols; ++j)
      finalMat[i][j] = resultArray[j];

    delete[] resultArray;
    resultArray = NULL;
  }

  delete[] rowArray;
  rowArray = NULL;
  delete[] colArray;
  colArray = NULL;

  return;
}

#endif