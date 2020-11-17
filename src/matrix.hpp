#ifndef MATRIX_CUDA_CPP
#define MATRIX_CUDA_CPP

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

typedef unsigned long long ull;

namespace EigenCUDA {
class Matrix {
public:
  explicit Matrix();

  explicit Matrix(ull nrows, ull ncols);

  Matrix(const Matrix &initializeWithObject);

  double at(ull ith, ull jth) const;

  ull getRows() const;

  ull getCols() const;

  void setRows(ull param);

  void setCols(ull param);

  // For testing purposes
  void printMatrix(void);

  void init__MatWithZeroDistribution(void);

  void init__MatWithRandomDistribution(int lowerRange, int upperRange,
                                       bool debugInfo);

  void init__MatWithUniformDistribution(int lowerRange, int upperRange,
                                        bool debugInfo);

  // For scaler operations
  EigenCUDA::Matrix operator+(double scalar);

  EigenCUDA::Matrix operator-(double scalar);

  EigenCUDA::Matrix operator*(double scalar);

  EigenCUDA::Matrix operator/(double scalar);

  EigenCUDA::Matrix transposeMatrix(void) const;

  EigenCUDA::Matrix operator*(const Matrix &rhs);

  bool operator==(const Matrix &rhs);

  void operator=(const Matrix &rhs);

  ~Matrix();

// private:
  double **__Mat;
  ull nrows;
  ull ncols;
}; // class Matrix
}; // namespace EigenCUDA

#endif