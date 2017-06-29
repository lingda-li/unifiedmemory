CODE_PATH = `pwd`/../
ifndef OPT
COMPILE_FLAG = --cuda-gpu-arch=sm_35 -I/software/cuda/8.0/include -I$(CODE_PATH)/uvm-runtime -O3
else
COMPILE_FLAG = --cuda-gpu-arch=sm_35 -I/software/cuda/8.0/include -I$(CODE_PATH)/uvm-runtime -O$(OPT)
endif
LINK_FLAG = -L/software/cuda/8.0/lib64 -lcudart -ldl -lrt -pthread -lm $(CODE_PATH)/uvm-runtime/uvm-runtime.o
IR_FLAG = -c -S -emit-llvm
PASS_FLAG = -Xclang -load -Xclang $(CODE_PATH)/llvm-pass/build/uvm/libUVMTransformPass.so

all: managed

test: test_m.cu
	clang $(COMPILE_FLAG) $(IR_FLAG) $(PASS_FLAG) test_m.cu

managed: test_m.cu
	clang $(COMPILE_FLAG) $(PASS_FLAG) $(IR_FLAG) test_m.cu
	clang $(COMPILE_FLAG) $(PASS_FLAG) $(LINK_FLAG) test_m.cu -o test_m

uvmapi: uvmapitest.cu uvmapitest_memcpy.cu
	clang $(COMPILE_FLAG) $(LINK_FLAG) uvmapitest.cu -o uvmapitest
	clang $(COMPILE_FLAG) $(LINK_FLAG) uvmapitest_memcpy.cu -o uvmapitest_memcpy

uvmapiir: uvmapitest.cu uvmapitest_memcpy.cu
	clang $(COMPILE_FLAG) $(IR_FLAG) uvmapitest.cu
	clang $(COMPILE_FLAG) $(IR_FLAG) uvmapitest_memcpy.cu

clean:
	rm core.* *.bc *.ll
