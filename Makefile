CC=g++
NVCC=nvcc
CXXFLAGS= -Xcompiler -fopenmp,-Ofast,-Wextra,-g
CUDAFLAGS= # -keep

CU_FILES := $(wildcard *.cu)
EXE_FILES := $(CU_FILES:.cu=)

.PHONY: clean ptx all

all: $(EXE_FILES)

%: %.cu
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) $< -o $@.out

clean:
	rm -rf *.ii *.cubin *.ptx *.txt *.out *.o *fatbin* *.module_id *.gpu *.cudafe* *.reg*
ptx:
	$(NVCC) $(CUDAFLAGS) -ptx $(CU_FILES)
