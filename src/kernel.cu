#include "kernel.cuh"
#include "matrix.hpp"
#include <cstdio>
#include <cuda_runtime.h>

#define N 256
#define TPB 32
#define MATRIX_DIMENSIONS 256

int main(int argc, char const *argv[]) {
  
  // Declaring MatA, MatB
  EigenCUDA::Matrix MatA(MATRIX_DIMENSIONS, MATRIX_DIMENSIONS),
      MatB(MATRIX_DIMENSIONS, MATRIX_DIMENSIONS);

  // Intializing Matricies
  MatA.init__MatWithUniformDistribution(1, 100, false),
      MatB.init__MatWithUniformDistribution(1, 199, false);

  double *finalMat = NULL; 

  cudaMalloc(&finalMat, N * sizeof(double));

  dim3 gridSize(N / (2*TPB), N / (2*TPB), 1);
  dim3 blockSize(TPB / 2, TPB / 2, 1);
  launchKernel<<<gridSize, blockSize>>>(MatA, MatB, finalMat, MATRIX_DIMENSIONS, N);

  cudaFree(finalMat);

  return 0;
}