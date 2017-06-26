#include <stdio.h>
#include <stdlib.h>
#include "uvm-runtime.h"

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

    __uvm_malloc(&a_info);
    __uvm_malloc(&b_info);
    d_a = (int*)a_info.devPtr;
    d_b = (int*)b_info.devPtr;

    for(i = 0; i < size; i++) {
        d_a[i] = rand() % 100;
        sum_a += d_a[i];
    }

    MyKernel<<<BlockNum, BlockSize>>>(d_a, d_b, size);

    cudaDeviceSynchronize();

    for(i = 0; i < size; i++)
        sum_b += d_b[i];

    __uvm_free(&a_info);
    __uvm_free(&b_info);

    printf("sum_a: %d, sum_b: %d\n", sum_a, sum_b);

    return 0;
}
