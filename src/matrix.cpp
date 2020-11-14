#include "matrix.hpp"
#include <cstdlib>
#include <stdexcept>

EigenCUDA::Matrix::Matrix() {
  nrows = 0u;
  ncols = 0u;
  __Mat = nullptr;
}

EigenCUDA::Matrix::Matrix(ull nrows, ull ncols) : nrows(nrows), ncols(ncols) {
  this->__Mat = new double *[nrows];

  for (ull i = 0; i < nrows; ++i)
    this->__Mat[i] = new double[ncols];
}

EigenCUDA::Matrix::Matrix(const Matrix &initializeWithObject)
    : nrows(initializeWithObject.getRows()),
      ncols(initializeWithObject.getCols()) {

  this->__Mat = new double *[this->nrows];

  for (ull i = 0; i < nrows; ++i)
    this->__Mat[i] = new double[this->ncols];

  clock_t __start = clock();

  for (ull i = 0; i < this->nrows; ++i)
    for (ull j = 0; j < this->ncols; ++j)
      this->__Mat[i][j] = initializeWithObject.__Mat[i][j];
  clock_t __end = clock();
  double __inputTimeUsed = ((double)(__end - __start)) / CLOCKS_PER_SEC;
  std::cout << "[MATRIX - <" << nrows << ", " << ncols
            << "> @ COPY CONSTRUCTOR] Time taken to get the matrix, "
            << __inputTimeUsed << "\n";
}

double EigenCUDA::Matrix::at(ull ith, ull jth) const {
  if (ith < this->nrows and jth < this->ncols) {
    return this->__Mat[ith][jth];
  } else {
    std::cerr << "[ ERR - at ]: Invalid indexing, at" << __LINE__ << " in file "
              << __FILE__ << " in function " << __FUNCTION__ << " \n";
    exit(EXIT_FAILURE);
  }
}

ull EigenCUDA::Matrix::getRows() const { return this->nrows; }

ull EigenCUDA::Matrix::getCols() const { return this->ncols; }

void EigenCUDA::Matrix::setRows(ull param) {

  if (this->nrows == param)
    return; /* do nothing */

  if (this->nrows < param) {

    /* Create new matrix */
    double **tempMatrix = nullptr;
    tempMatrix = new double *[param];

    for (size_t i = 0; i < param; ++i) {
      tempMatrix[i] = new double[this->ncols];
    }
    /* Finish creating a new matrix */

    /* Transfer data from old matrix to new matrix */
    for (size_t i = 0; i < this->nrows; ++i) {
      for (size_t j = 0; j < this->ncols; ++j) {
        tempMatrix[i][j] = this->__Mat[i][j];
      }
    }
    /* Finish transfer data from old matrix to new matrix */

    /* Zero out all the new rows */
    for (size_t i = nrows; i < param; ++i)
      for (size_t j = 0; j < this->ncols; ++j)
        tempMatrix[i][j] = 0;
    /* Finish zeroing out all the new cols */

    /* Delete old matrix */
    for (ull i = 0; i < this->nrows; ++i) {
      delete[] this->__Mat[i];
      this->__Mat[i] = nullptr;
    }

    delete this->__Mat;
    this->__Mat = nullptr;
    /* Finish deleting old matrix */

    /* Set pointer to new matrix */
    this->__Mat = tempMatrix;
    tempMatrix = nullptr;

  } else if (this->nrows > param) {

    /* Create new matrix */
    double **tempMatrix = nullptr;
    tempMatrix = new double *[param];

    for (size_t i = 0; i < param; ++i) {
      tempMatrix[i] = new double[this->ncols];
    }
    /* Finish creating a new matrix */

    /* Transfer data from old matrix to new matrix */
    for (size_t i = 0; i < param; ++i) {
      for (size_t j = 0; j < this->ncols; ++j) {
        tempMatrix[i][j] = this->__Mat[i][j];
      }
    }
    /* Finish transfer data from old matrix to new matrix */

    /* Delete old matrix */
    for (ull i = 0; i < this->nrows; ++i) {
      delete[] this->__Mat[i];
      this->__Mat[i] = nullptr;
    }

    delete this->__Mat;
    this->__Mat = nullptr;
    /* Finish deleting old matrix */

    /* Set pointer to new matrix */
    this->__Mat = tempMatrix;
    tempMatrix = nullptr;
  }

  /*
   * The following works for when,
   * - this->nrows == 0
   * - this->nrows > param
   * - this->nrows < param
   */
  this->nrows = param;

  return;
} // setRows()

void EigenCUDA::Matrix::setCols(ull param) {

  if (this->ncols == param)
    return; // Do nothing

  else if (this->ncols < param) {

    /* Create new matrix */
    double **tempMatrix = nullptr;
    tempMatrix = new double *[this->nrows];

    for (size_t i = 0; i < this->nrows; ++i) {
      tempMatrix[i] = new double[param];
    }
    /* Finish creating a new matrix */

    /* Transfer data from old matrix to new matrix */
    for (size_t i = 0; i < this->nrows; ++i) {
      for (size_t j = 0; j < this->ncols; ++j) {
        tempMatrix[i][j] = this->__Mat[i][j];
      }
    }
    /* Finish transfer data from old matrix to new matrix */

    /* Zero out all the new cols */
    for (size_t i = 0; i < this->nrows; ++i)
      for (size_t j = this->ncols; j < param; ++j)
        tempMatrix[i][j] = 0;
    /* Finish zeroing out all the new cols */

    /* Delete old matrix */
    for (ull i = 0; i < this->nrows; ++i) {
      delete[] this->__Mat[i];
      this->__Mat[i] = nullptr;
    }

    delete this->__Mat;
    this->__Mat = nullptr;
    /* Finish deleting old matrix */

    /* Set pointer to new matrix */
    this->__Mat = tempMatrix;
    tempMatrix = nullptr;

  } else if (this->ncols > param) {

    /* Create new matrix */
    double **tempMatrix = nullptr;
    tempMatrix = new double *[this->nrows];

    for (size_t i = 0; i < this->nrows; ++i) {
      tempMatrix[i] = new double[param];
    }
    /* Finish creating a new matrix */

    /* Transfer data from old matrix to new matrix */
    for (size_t i = 0; i < this->nrows; ++i) {
      for (size_t j = 0; j < param; ++j) {
        tempMatrix[i][j] = this->__Mat[i][j];
      }
    }
    /* Finish transfer data from old matrix to new matrix */

    /* Delete old matrix */
    for (ull i = 0; i < this->nrows; ++i) {
      delete[] this->__Mat[i];
      this->__Mat[i] = nullptr;
    }

    delete this->__Mat;
    this->__Mat = nullptr;
    /* Finish deleting old matrix */

    /* Set pointer to new matrix */
    this->__Mat = tempMatrix;
    tempMatrix = nullptr;
  }

  /*
   * The following works for when,
   * - this->ncols == 0
   * - this->ncols > param
   * - this->ncols < param
   */
  this->ncols = param;

  return;
} // setCols()

void EigenCUDA::Matrix::init__MatWithZeroDistribution(void) {
  try {
    if (this->nrows == 0 or this->ncols == 0) {
      throw std::invalid_argument(
          "Initialization without specifying proper dimensions");
    }

  } catch (std::invalid_argument &error) {
    std::cerr << "[ ERR - init__MatWithZeroDistribution ]: Initialization "
                 "called without setting either rows or columns "
                 "to be a number greater than 0 at line "
              << __LINE__ << " in file " << __FILE__ << " in function "
              << __FUNCTION__ << " \n";
    exit(EXIT_FAILURE);
  }

  clock_t __start = clock();

  for (ull i = 0; i < this->nrows; ++i)
    for (ull j = 0; j < this->ncols; ++j)
      this->__Mat[i][i] = 0;

  clock_t __end = clock();
  double __inputTimeUsed = ((double)(__end - __start)) / CLOCKS_PER_SEC;
  std::cout << "[MATRIX - <" << nrows << ", " << ncols
            << "> @ ZERO_INITIALIZATION ] Time taken to get the matrix, "
            << __inputTimeUsed << "\n";

  return;
}

void EigenCUDA::Matrix::init__MatWithRandomDistribution(int lowerRange = -1,
                                                        int upperRange = 1) {

  try {
    if (this->nrows == 0 or this->ncols == 0) {
      throw std::invalid_argument(
          "Initialization without specifying proper dimensions");
    }
  } catch (std::invalid_argument &error) {
    std::cerr << "[ ERR - init__MatWithRandomDistribution ]: Initialization "
                 "called without setting either rows or columns "
                 "to be a number greater than 0 at line "
              << __LINE__ << " in file " << __FILE__ << " in function "
              << __FUNCTION__ << " \n";
    exit(EXIT_FAILURE);
  }

  std::random_device __rd;
  // seed value is designed specifically to make initialization
  // parameters of std::mt19937 (instance of std::mersenne_twister_engine<>)
  // different across executions of application
  std::mt19937::result_type __seed =
      __rd() ^
      ((std::mt19937::result_type)
           std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
               .count() +
       (std::mt19937::result_type)
           std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
               .count());

  std::mt19937 __gen(__seed);
  std::mt19937::result_type n;

  /* Generating single pseudo-random number makes no sense
      even if you use std::mersenne_twister_engine instead of rand()
      and even when your seed quality is much better than time(NULL)
      See SO thread,
     https://stackoverflow.com/questions/13445688/how-to-generate-a-random-number-in-c
      for more reference */

  clock_t __start = clock();

  for (ull i = 0; i < this->nrows; ++i) {
    for (ull j = 0; j < this->ncols; ++j) {

      // reject readings that would make n%6 non-uniformly distributed
      while ((n = __gen()) >
             std::mt19937::max() -
                 (std::mt19937::max() - lowerRange) %
                     upperRange) { /* bad value retrieved so get next one */
      }

      this->__Mat[i][j] = n;
    }

    clock_t __end = clock();
    double __inputTimeUsed = ((double)(__end - __start)) / CLOCKS_PER_SEC;
    std::cout << "[MATRIX - <" << nrows << ", " << ncols
              << "> @ RANDOM_DISTRIBUTION ] Time taken to get the matrix, "
              << __inputTimeUsed << "\n";

    return;
  }
}

void EigenCUDA::Matrix::init__MatWithUniformDistribution(int lowerRange = -1,
                                                         int upperRange = 1) {
  try {
    if (this->nrows == 0 or this->ncols == 0) {
      throw std::invalid_argument(
          "Initialization without specifying proper dimensions");
    }
  } catch (std::invalid_argument &error) {
    std::cerr << "[ ERR - init__MatWithUniformDistribution ]: Initialization "
                 "called without setting either rows or columns "
                 "to be a number greater than 0 at line "
              << __LINE__ << " in file " << __FILE__ << " in function "
              << __FUNCTION__ << " \n";
    exit(EXIT_FAILURE);
  }

  std::random_device __rd;
  std::mt19937::result_type __seed =
      __rd() ^
      ((std::mt19937::result_type)
           std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
               .count() +
       (std::mt19937::result_type)
           std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
               .count());

  std::mt19937 __gen(__seed);
  std::uniform_int_distribution<unsigned> __distrib(1, 6);

  clock_t __start = clock();

  for (ull i = 0; i < nrows; ++i)
    for (ull j = 0; j < ncols; ++j)
      this->__Mat[i][j] = __distrib(__gen);

  clock_t __end = clock();
  double __inputTimeUsed = ((double)(__end - __start)) / CLOCKS_PER_SEC;
  std::cout << "[MATRIX - <" << nrows << ", " << ncols
            << "> @ UNIFORM_DISTRIBUTION ] Time taken to get the matrix, "
            << __inputTimeUsed << "\n";

  return;
}

EigenCUDA::Matrix EigenCUDA::Matrix::transposeMatrix(void) const {
  EigenCUDA::Matrix transposedMatrix(this->getCols(), this->getRows());

  for (ull i = 0; i < this->getRows(); ++i)
    for (ull j = 0; j < this->getCols(); ++j)
      transposedMatrix.__Mat[i][j] = this->__Mat[j][i];

  return transposedMatrix;
}

EigenCUDA::Matrix EigenCUDA::Matrix::operator+(double scalar) {

  EigenCUDA::Matrix tempMatrix(*this);

  for (ull i = 0; i < this->getRows(); ++i)
    for (ull j = 0; j < this->getCols(); ++j)
      tempMatrix.__Mat[i][j] += scalar;

  return tempMatrix;
}

EigenCUDA::Matrix EigenCUDA::Matrix::operator-(double scalar) {

  EigenCUDA::Matrix tempMatrix(*this);

  for (ull i = 0; i < this->getRows(); ++i)
    for (ull j = 0; j < this->getCols(); ++j)
      tempMatrix.__Mat[i][j] -= scalar;

  return tempMatrix;
}

EigenCUDA::Matrix EigenCUDA::Matrix::operator*(double scalar) {

  EigenCUDA::Matrix tempMatrix(*this);

  for (ull i = 0; i < this->getRows(); ++i)
    for (ull j = 0; j < this->getCols(); ++j)
      tempMatrix.__Mat[i][j] *= scalar;

  return tempMatrix;
}

EigenCUDA::Matrix EigenCUDA::Matrix::operator/(double scalar) {

  EigenCUDA::Matrix tempMatrix(*this);

  for (ull i = 0; i < this->getRows(); ++i)
    for (ull j = 0; j < this->getCols(); ++j)
      tempMatrix.__Mat[i][j] /= scalar;

  return tempMatrix;
}

EigenCUDA::Matrix EigenCUDA::Matrix::operator*(const Matrix &rhs) {

  int newMatrixRows = this->getRows();
  int newMatrixCols = rhs.getCols();

  EigenCUDA::Matrix productMatrix(newMatrixRows, newMatrixCols);

  productMatrix.init__MatWithZeroDistribution();

  for (ull i = 0; i < this->getRows(); ++i)
    for (ull j = 0; j < rhs.getCols(); ++j)
      for (ull k = 0; k < this->getCols(); ++k)
        productMatrix.__Mat[i][j] += (this->__Mat[i][k] * rhs.__Mat[k][j]);

  return productMatrix;
}

const bool EigenCUDA::Matrix::operator==(const Matrix &rhs) {
  if (this != &rhs) {
    if (this->getCols() != rhs.getCols() or this->getRows() != rhs.getRows()) {
      std::cout << "Invalid number of cols AND/OR rows\n";
      return false;
    }

    for (ull i = 0; i < this->nrows; ++i)
      for (ull j = 0; j < this->ncols; ++j)
        if (this->__Mat[i][j] != rhs.__Mat[i][j]) {
          std::cout << "Invalid value at [" << i << ", " << j << "]\n";
          return false;
        }

    return true;
  }

  return true;
}

void EigenCUDA::Matrix::operator=(const Matrix &rhs) {
  if (this == &rhs) {
    return;
  } else {

    if (this->getRows() != rhs.getRows() or this->getCols() != rhs.getCols()) {
      /* Deleting the old 2D matrix */
      for (ull i = 0; i < this->getRows(); ++i) {
        delete[] __Mat[i];
        __Mat[i] = nullptr;
      }

      delete[] __Mat;
      __Mat = nullptr;
      /* Deleted old 2D matrix */

      /* Getting the new information */
      this->nrows = rhs.getRows();
      this->ncols = rhs.getCols();
      /* Finish getting new information */

      /* Reallocating memory for the new matrix */
      __Mat = new double *[this->nrows];

      for (ull i = 0; i < this->nrows; ++i) {
        this->__Mat[i] = new double[this->ncols];
      }
      /* Finish reallocating the new matrix */
    }

    /* Start copying the parameter Matrix object into the newly allocated
     * object */
    for (ull i = 0; i < this->getRows(); ++i) {
      for (ull j = 0; j < this->getCols(); ++j) {
        this->__Mat[i][j] = rhs.__Mat[i][j];
      }
    }
    /* End copying the parameter Matrix object into the newly allocated
     * object */

    return;
  }
}

void EigenCUDA::Matrix::printMatrix(void) {
  for (ull i = 0; i < this->nrows; ++i) {
    printf("[ ");
    for (ull j = 0; j < this->ncols; ++j) {
      printf("%lf ", this->__Mat[i][j]);
    }
    printf("]\n");
  }
  puts("");

  return;
}

EigenCUDA::Matrix::~Matrix() {
  for (ull i = 0; i < nrows; ++i) {
    delete[] __Mat[i];
    __Mat[i] = nullptr;
  }

  delete __Mat;
  __Mat = nullptr;
}