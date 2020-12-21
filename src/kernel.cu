#include "kernelAux.cuh"
#include "matrix.hpp"
#include <stdio.h>
#include <stdlib.h>

#define N 512
#define TPB 32
#define MATRIX_DIMENSIONS 10

#define ull unsigned long long

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

  cudaMalloc(&finalMat, N * sizeof(double));

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

  cudaFree(finalMat);

  return 0;
}