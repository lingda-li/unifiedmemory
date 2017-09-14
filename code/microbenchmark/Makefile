CUDA_PATH = /sw/summitdev/cuda/8.0.61-1/
GPU_ARCH = sm_60
#CUDA_PATH = /software/cuda/8.0/
#GPU_ARCH = sm_35

CLANG_FLAG = --cuda-gpu-arch=$(GPU_ARCH) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcudart -ldl -lrt -pthread -lm

all:
	clang $(CLANG_FLAG) latency.cu -o latency.out
	clang $(CLANG_FLAG) throughput_rw.cu -o throughput_rw.out
	clang $(CLANG_FLAG) throughput_r.cu -o throughput_r.out
	clang $(CLANG_FLAG) throughput_w.cu -o throughput_w.out

clean:
	-rm *.out
