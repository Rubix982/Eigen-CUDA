NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
INC = -I/usr/local/cuda/samples/common/inc

SOURCEDIR = src
BUILDDIR = build

all: dir main.exe

dir:
	@ mkdir -p $(BUILDDIR)

# $(BUILDDIR)/main.o
main.exe: $(BUILDDIR)/kernel.o $(BUILDDIR)/matrix.o 
	@ $(NVCC) $(NVCC_FLAGS) $(INC) $^ -o $@
	@ mv *.exe $(BUILDDIR)

$(BUILDDIR)/kernel.o: $(SOURCEDIR)/kernel.cu $(SOURCEDIR)/kernelAux.cuh
	@ $(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

$(BUILDDIR)/matrix.o: $(SOURCEDIR)/matrix.cpp $(SOURCEDIR)/matrix.hpp
	@ $(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

run: all
	@ ./build/main.exe

clean: 
	@ rm -rf $(BUILDDIR)