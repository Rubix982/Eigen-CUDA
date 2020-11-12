#ifndef MATRIX_CUDA_CPP
#define MATRIX_CUDA_CPP

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <random>

typedef unsigned long long ull;

namespace EigenCUDA
{
    class Matrix
    {
    public:

        explicit Matrix()
            :   nrows{0}, ncols{0}, __Mat{nullptr} {}

        explicit Matrix(int nrows, int ncols)
            : nrows(nrows), ncols(ncols)
        {
            this->__Mat = new double *[nrows];

            for (register ull i = 0; i < nrows; ++i)
                this->__Mat[i] = new double[ncols];
        }

        explicit Matrix(const Matrix &initializeWithObject)
            : nrows(initializeWithObject.getRows()), ncols(initializeWithObject.getCols())
        {

            this->__Mat = new double *[this->nrows];

            for (register ull i = 0; i < nrows; ++i)
                this->__Mat[i] = new double[this->ncols];

            clock_t __start = clock();

            for (register ull i = 0; i < this->nrows; ++i)
                for (register ull j = 0; j < this->ncols; ++j)
                    this->__Mat[i][j] = initializeWithObject.__Mat[i][j];
            clock_t __end = clock();
            double __inputTimeUsed = ((double)(__end - __start)) / CLOCKS_PER_SEC;
            std::cout << "[MATRIX - <" << nrows << ", " << ncols << "> @ COPY CONSTRUCTOR] Time taken to get the matrix, " << __inputTimeUsed << "\n";
        }

        int getRows() const
        {
            return this->nrows;
        }

        int getCols() const
        {
            return this->ncols;
        }

        void init__MatWithZeroDistribution(void)
        {

            for (register int i = 0; i < this->nrows; ++i)
                for (register int j = 0; j < this->ncols; ++j)
                    __Mat[i][i] = 0;
        }

        void init__MatWithRandomDistribution(int lowerRange = -1, int upperRange = 1)
        {

            std::random_device __rd;
            // seed value is designed specifically to make initialization
            // parameters of std::mt19937 (instance of std::mersenne_twister_engine<>)
            // different across executions of application
            std::mt19937::result_type __seed = __rd() ^ ((std::mt19937::result_type)
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
                See SO thread, https://stackoverflow.com/questions/13445688/how-to-generate-a-random-number-in-c
                for more reference */

            clock_t __start = clock();

            for (register ull i = 0; i < this->nrows; ++i)
            {
                for (register ull j = 0; j < this->ncols; ++j)
                {

                    // reject readings that would make n%6 non-uniformly distributed
                    while ((n = __gen()) > std::mt19937::max() -
                                               (std::mt19937::max() - lowerRange) % upperRange)
                    { /* bad value retrieved so get next one */
                    }

                    this->__Mat[i][j] = n;
                }
            }

            clock_t __end = clock();
            double __inputTimeUsed = ((double)(__end - __start)) / CLOCKS_PER_SEC;
            std::cout << "[MATRIX - <" << nrows << ", " << ncols << "> @ RANDOM_DISTRIBUTION ] Time taken to get the matrix, " << __inputTimeUsed << "\n";

            return;
        }

        void init__MatWithUniformDistribution(int lowerRange = -1, int upperRange = 1)
        {
            std::random_device __rd;
            std::mt19937::result_type __seed = __rd() ^ ((std::mt19937::result_type)
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

            for (register ull i = 0; i < nrows; ++i)
                for (register ull j = 0; j < ncols; ++j)
                    this->__Mat[i][j] = __distrib(__gen);

            clock_t __end = clock();
            double __inputTimeUsed = ((double)(__end - __start)) / CLOCKS_PER_SEC;
            std::cout << "[MATRIX - <" << nrows << ", " << ncols << "> @ UNIFORM_DISTRIBUTION ] Time taken to get the matrix, " << __inputTimeUsed << "\n";

            return;
        }

        const bool operator==(const Matrix &rhs)
        {
            if (this != &rhs)
            {
                if (this->getCols() != rhs.getCols() or this->getRows() != rhs.getRows())
                {
                    std::cout << "Invalid number of cols AND/OR rows\n";
                    return false;
                }

                for (register ull i = 0; i < this->nrows; ++i)
                    for (register ull j = 0; j < this->ncols; ++j)
                        if (this->__Mat[i][j] != rhs.__Mat[i][j])
                        {
                            std::cout << "Invalid value at [" << i << ", " << j << "]\n";
                            return false;
                        }

                return true;
            }

            return true;
        }

        const Matrix& operator=(const Matrix &rhs) {
            if ( this == &rhs )
                return *this;
        }

        ~Matrix()
        {

            for (register ull i = 0; i < nrows; ++i)
                delete[] __Mat[i];

            for (register ull i = 0; i < nrows; ++i)
                __Mat[i] = nullptr;

            delete __Mat;
            __Mat = nullptr;
        }

    private:
        double **__Mat;
        int nrows;
        int ncols;

    protected:
    };
}; // namespace EigenCUDA

#endif