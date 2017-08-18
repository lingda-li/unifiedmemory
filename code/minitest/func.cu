#include <stdio.h>
#include <stdlib.h>

#define MYDEBUG
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
        b[id] = (a[id] >> 1) + a[id];
}

int foo(int *a, int *b, int size)
{
    int i;
    int BlockSize = 256;
    int BlockNum = (size + BlockSize - 1) / BlockSize;
    int sum = 0;

    for(i = 0; i < size; i++) {
        a[i] += rand() % 50;
        sum += a[i];
    }

    MyKernel<<<BlockNum, BlockSize>>>(a, b, size);

    cudaDeviceSynchronize();
    DEBUG_PRINT
    return sum;
}

int main()
{
    int i;
    int size = SIZE;
    int *d_a, *d_b;
    int sum_a, sum_b;
    sum_a = sum_b = 0;
    DEBUG_PRINT

	cudaMallocManaged((void **)&d_a, size*sizeof(int));
	cudaMallocManaged((void **)&d_b, size*sizeof(int));
    DEBUG_PRINT
    for(i = 0; i < size; i++) {
        d_a[i] = rand() % 100;
    }
    DEBUG_PRINT

    sum_a = foo(d_a, d_b, size);

    for(i = 0; i < size; i++)
        sum_b += d_b[i];

    DEBUG_PRINT
    cudaFree(d_a);
    cudaFree(d_b);
    DEBUG_PRINT

    printf("sum_a: %d, sum_b: %d\n", sum_a, sum_b);

    return 0;
}
