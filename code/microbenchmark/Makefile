CUDA_PATH = /sw/summitdev/cuda/9.0.69/
#CUDA_PATH = /sw/summitdev/cuda/8.0.61-1/
GPU_ARCH = sm_60
#CUDA_PATH = /software/cuda/8.0/
#GPU_ARCH = sm_35

GPUCC = nvcc
COMPILE_FLAG = -arch=$(GPU_ARCH) -O3 -Xptxas -v

#GPUCC = clang
#COMPILE_FLAG = --cuda-gpu-arch=$(GPU_ARCH) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcudart -ldl -lrt -pthread -lm -O3

ifdef STEP
  COMPILE_FLAG += -Xcompiler -DSTEP=$(STEP)
endif

ifdef SIZE
  COMPILE_FLAG += -Xcompiler -DSIZE=$(SIZE)
endif

ifdef DEV_SIZE
  COMPILE_FLAG += -Xcompiler -DDEV_SIZE=$(DEV_SIZE)
endif

ifdef ALLOC
  COMPILE_FLAG += -Xcompiler -D$(ALLOC)
endif

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

mgpu:
	$(GPUCC) $(COMPILE_FLAG) mgpu_lat.cu -o mgpu_lat.out

access_sparse:
	$(GPUCC) $(COMPILE_FLAG) access_sparse.cu -o access_sparse.out

access_locality:
	$(GPUCC) $(COMPILE_FLAG) access_locality.cu -o access_locality.out

clean:
	-rm *.out
