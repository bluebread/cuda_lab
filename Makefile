CC=g++
NVCC=nvcc
CXXFLAGS= -Xcompiler -fopenmp,-Wextra
CUDAFLAGS= -g -O3 --generate-line-info -lcublas -arch=sm_89 -diag-suppress=177

SRC_DIR := src
BIN_DIR := bin
CU_FILES := $(wildcard $(SRC_DIR)/*.cu)
EXE_FILES := $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%.out, $(CU_FILES))

.PHONY: clean ptx all

all: $(EXE_FILES)

$(BIN_DIR)/%.out: $(SRC_DIR)/%.cu
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(BIN_DIR)/*.out *.ii *.cubin *.ptx *.txt *.o *fatbin* *.module_id *.gpu *.cudafe* *.reg*

ptx:
	$(NVCC) $(CUDAFLAGS) -ptx $(CU_FILES)
