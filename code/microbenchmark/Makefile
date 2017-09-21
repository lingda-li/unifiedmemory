CUDA_PATH = /sw/summitdev/cuda/8.0.61-1/
GPU_ARCH = sm_60
#CUDA_PATH = /software/cuda/8.0/
#GPU_ARCH = sm_35

GPUCC = nvcc
COMPILE_FLAG = -arch=$(GPU_ARCH) -O3 -Xptxas -v

#GPUCC = clang
#COMPILE_FLAG = --cuda-gpu-arch=$(GPU_ARCH) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcudart -ldl -lrt -pthread -lm -O3

all:
	$(GPUCC) $(COMPILE_FLAG) latency.cu -o latency.out
	$(GPUCC) $(COMPILE_FLAG) throughput_rw.cu -o throughput_rw.out
	$(GPUCC) $(COMPILE_FLAG) throughput_r.cu -o throughput_r.out
	$(GPUCC) $(COMPILE_FLAG) throughput_w.cu -o throughput_w.out

lat:
	$(GPUCC) $(COMPILE_FLAG) latency.cu -o latency.out

page:
	$(GPUCC) $(COMPILE_FLAG) page.cu -o page.out

page_ol:
	$(GPUCC) $(COMPILE_FLAG) page_overlap.cu -o page_overlap.out

page_sm:
	$(GPUCC) $(COMPILE_FLAG) page_sm.cu -o page_sm.out

clean:
	-rm *.out
