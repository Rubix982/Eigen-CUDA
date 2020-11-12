#include <iostream>
#include "matrix.hpp"

int main(int argc, char const *argv[])
{
    EigenCUDA::Matrix testing;

    testing.init__MatWithZeroDistribution();

    std::cout << "Hello, World!\n";
    return 0;
}
