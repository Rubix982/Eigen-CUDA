GCC = /usr/bin/gcc
NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

HPP_FILES = src/matrix.hpp cudaMultiply.hpp
CPP_FILES = src/matrix.cpp src/main.cpp src/cudaMultiply.cpp

main: $(CPP_FILES)
	@ $(NVCC) $(NVCC_FLAGS) $^ -o $@ &>> /dev/null

clean: main
	rm main