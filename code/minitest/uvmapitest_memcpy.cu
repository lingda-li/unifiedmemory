#include <stdio.h>
#include <stdlib.h>
#include "uvm-runtime.h"

//#define MYDEBUG
#ifdef MYDEBUG
#define DEBUG_PRINT printf("here: %d\n", __LINE__); fflush(stdout);
#else
#define DEBUG_PRINT
#endif

#define SIZE 10240

__global__ void MyKernel(int *a, int *b, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < size)
        b[id] = a[id] * 2;
}

int main()
{
    int i;
    int size = SIZE;
    int BlockSize = 256;
    int BlockNum = (size + BlockSize - 1) / BlockSize;
    int *d_a, *d_b;
    int sum_a, sum_b;
    sum_a = sum_b = 0;

    struct uvmMallocInfo a_info, b_info;
    a_info.size = size*sizeof(int);
    b_info.size = size*sizeof(int);
    DEBUG_PRINT

    __uvm_malloc(&a_info);
    __uvm_malloc(&b_info);
    DEBUG_PRINT

    d_a = (int*)a_info.devPtr;
    d_b = (int*)b_info.devPtr;
    int *h_a = (int*)a_info.hostPtr;
    int *h_b = (int*)b_info.hostPtr;

    for(i = 0; i < size; i++) {
        h_a[i] = rand() % 100;
        sum_a += h_a[i];
    }
    DEBUG_PRINT

    __uvm_memcpy(&a_info, cudaMemcpyHostToDevice);
    DEBUG_PRINT

    MyKernel<<<BlockNum, BlockSize>>>(d_a, d_b, size);

    __uvm_memcpy(&b_info, cudaMemcpyDeviceToHost);
    DEBUG_PRINT

    for(i = 0; i < size; i++)
        sum_b += h_b[i];
    DEBUG_PRINT

    __uvm_free(&a_info);
    __uvm_free(&b_info);
    DEBUG_PRINT

    printf("sum_a: %d, sum_b: %d\n", sum_a, sum_b);

    return 0;
}
