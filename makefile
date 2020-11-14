GCC = /usr/bin/gcc
NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
INC = -I/usr/local/cuda/samples/common/inc

SOURCEDIR = src
BUILDDIR = build

all: dir main.exe

dir:
	@ mkdir -p $(BUILDDIR)

main.exe: $(BUILDDIR)/cuda.o $(BUILDDIR)/matrix.o $(BUILDDIR)/kernel.o
	@ $(NVCC) $^ -o $@
	@ mv *.exe $(BUILDDIR)

$(BUILDDIR)/kernel.o: $(SOURCEDIR)/kernel.cu $(SOURCEDIR)/kernel.hpp
	@ $(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

$(BUILDDIR)/matrix.o: $(SOURCEDIR)/matrix.cpp $(SOURCEDIR)/matrix.hpp
	@ $(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

$(BUILDDIR)/cuda.o: $(SOURCEDIR)/cudaMultiply.cpp $(SOURCEDIR)/cudaMultiply.hpp
	@ $(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

run: all
	./build/main.exe

clean: 
	@ rm -rf $(BUILDDIR)