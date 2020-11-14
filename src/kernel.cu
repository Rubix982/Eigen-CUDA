#include "matrix.hpp"
#include "kernel.cuh"
#include <cstdio>

#define N 128
#define TPB 32

int main(int argc, char const *argv[]) {
  launchKernel<<<N/TPB, TPB>>>();

  EigenCUDA::Matrix MatA(4, 4);
  MatA.init__MatWithUniformDistribution(1, 100, true);
  MatA.printMatrix();

  return 0;
}