#include "matrix.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 512
#define TPB 32
#define MATRIX_DIMENSIONS 10

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

int main(int argc, char const *argv[]) {

  // Declaring MatA, MatB
  EigenCUDA::Matrix MatA(MATRIX_DIMENSIONS, MATRIX_DIMENSIONS),
      MatB(MATRIX_DIMENSIONS, MATRIX_DIMENSIONS);

  // Intializing Matricies
  MatA.init__MatWithUniformDistribution(1, 100, false),
      MatB.init__MatWithUniformDistribution(1, 199, false);

  // MatA.printMatrix();
  // MatB.printMatrix();

  double **finalMat = new double *[MATRIX_DIMENSIONS];

  for (int i = 0; i < MATRIX_DIMENSIONS; ++i)
    finalMat[i] = new double[MATRIX_DIMENSIONS];

  // cudaMalloc(&finalMat, N * sizeof(double));

  dim3 gridSize(N / (2 * TPB), N / (2 * TPB), 1);
  dim3 blockSize(TPB / 2, TPB / 2, 1);
  launchKernel<<<gridSize, blockSize>>>(MatA, MatB, finalMat, MATRIX_DIMENSIONS,
                                        N);

  for (int i = 0; i < 10; ++i) {
    printf("[ ");
    for (int j = 0; j < 10; ++j) {
      printf("%lf ", finalMat[i][j]);
    }
    printf("]\n");
  }

  for (int i = 0; i < MATRIX_DIMENSIONS; ++i) {
    delete[] finalMat[i];
    finalMat[i] = NULL;
  }

  delete[] finalMat;
  finalMat = NULL;

  // cudaFree(finalMat);

  return 0;
}