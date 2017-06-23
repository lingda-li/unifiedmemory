COMPILE_FLAG = --cuda-gpu-arch=sm_35 -I/software/cuda/8.0/include -O3
LINK_FLAG = -L/software/cuda/8.0/lib64 -lcudart -ldl -lrt -pthread -lm
CODE_PATH = `pwd`/../

uvmapi: uvmapitest.cu uvmapitest_memcpy.cu
	clang $(COMPILE_FLAG) -I$(CODE_PATH)/uvm-runtime $(LINK_FLAG) $(CODE_PATH)/uvm-runtime/uvm-runtime.o uvmapitest.cu -o uvmapitest
	clang $(COMPILE_FLAG) -I$(CODE_PATH)/uvm-runtime $(LINK_FLAG) $(CODE_PATH)/uvm-runtime/uvm-runtime.o uvmapitest_memcpy.cu -o uvmapitest_memcpy
