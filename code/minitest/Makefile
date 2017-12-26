CUDA_PATH = /sw/summitdev/cuda/8.0.61-1/
GPU_ARCH = sm_60
#CUDA_PATH = /software/cuda/8.0/
#GPU_ARCH = sm_35

#PASS_NAME = UVMTransformPass
#PASS_NAME = MemAnalysisPass
PASS_NAME = OMPPass

CODE_PATH = `pwd`/../

ifndef OPT
COMPILE_FLAG = --cuda-gpu-arch=$(GPU_ARCH) -I$(CUDA_PATH)/include -I$(CODE_PATH)/uvm-runtime -O3
else
COMPILE_FLAG = --cuda-gpu-arch=$(GPU_ARCH) -I$(CUDA_PATH)/include -I$(CODE_PATH)/uvm-runtime -O$(OPT)
endif
OMP_COMPILE_FLAG = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -I$(CUDA_PATH)/include -O3
LINK_FLAG = -L$(CUDA_PATH)/lib64 -lcudart -ldl -lrt -pthread -lm $(CODE_PATH)/uvm-runtime/uvm-runtime.o
IR_FLAG = -c -S -emit-llvm
PASS_FLAG = -Xclang -load -Xclang $(CODE_PATH)/llvm-pass/build/uvm/lib$(PASS_NAME).so

all: managed

test: test_m.cu
	clang $(COMPILE_FLAG) $(IR_FLAG) $(PASS_FLAG) test_m.cu

managed: test_m.cu
	clang $(COMPILE_FLAG) $(PASS_FLAG) $(IR_FLAG) test_m.cu
	clang $(COMPILE_FLAG) $(PASS_FLAG) $(LINK_FLAG) test_m.cu -o test_m

func: func.cu
	clang $(COMPILE_FLAG) $(PASS_FLAG) $(IR_FLAG) func.cu
#	clang $(COMPILE_FLAG) $(PASS_FLAG) $(LINK_FLAG) func.cu -o func 

uvmapi: uvmapitest.cu uvmapitest_memcpy.cu
	clang $(COMPILE_FLAG) $(LINK_FLAG) uvmapitest.cu -o uvmapitest
	clang $(COMPILE_FLAG) $(LINK_FLAG) uvmapitest_memcpy.cu -o uvmapitest_memcpy

uvmapiir: uvmapitest.cu uvmapitest_memcpy.cu
	clang $(COMPILE_FLAG) $(IR_FLAG) uvmapitest.cu
	clang $(COMPILE_FLAG) $(IR_FLAG) uvmapitest_memcpy.cu

omp: omp_m.c
	clang $(OMP_COMPILE_FLAG) $(IR_FLAG) $(PASS_FLAG) omp_m.c
#	clang $(OMP_COMPILE_FLAG) $(LINK_FLAG) $(PASS_FLAG) omp_m.c -o omp_m

clean:
	rm core.* *.bc *.ll
