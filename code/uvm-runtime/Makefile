CUDA_PATH = /usr/local/cuda-9.0/
GPU_ARCH = sm_70
#CUDA_PATH = /sw/summitdev/cuda/8.0.61-1/
#GPU_ARCH = sm_60
#CUDA_PATH = /software/cuda/8.0/
#GPU_ARCH = sm_35

CLANG_FLAG = --cuda-gpu-arch=$(GPU_ARCH) -I$(CUDA_PATH)/include

all:
	clang $(CLANG_FLAG) uvm-runtime.cu -c
	clang $(CLANG_FLAG) uvm-runtime.cu -c -S -emit-llvm
