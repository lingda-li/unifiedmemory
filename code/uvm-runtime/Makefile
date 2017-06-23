all:
	clang --cuda-gpu-arch=sm_35 -I/software/cuda/8.0/include uvm-runtime.cu -c
	clang --cuda-gpu-arch=sm_35 -I/software/cuda/8.0/include uvm-runtime.cu -c -S -emit-llvm
