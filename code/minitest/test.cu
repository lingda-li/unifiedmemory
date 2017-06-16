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
        b[id] = a[id] * 2;
}

int main()
{
    int i;
    int size = SIZE;
    int BlockSize = 256;
    int BlockNum = (size + BlockSize - 1) / BlockSize;
    int *a, *b; 
    int *d_a, *d_b;
    int sum_a, sum_b;
    sum_a = sum_b = 0;

    a = (int*)malloc(size*sizeof(int));
    b = (int*)malloc(size*sizeof(int));
    for(i = 0; i < size; i++) {
        a[i] = rand() % 100;
        sum_a += a[i];
    }
    DEBUG_PRINT

    cudaMalloc((void **)&d_a, size*sizeof(int));
    cudaMemcpy(d_a, a, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_b, size*sizeof(int));
    cudaMemcpy(d_b, b, size*sizeof(int), cudaMemcpyHostToDevice);
    DEBUG_PRINT

    MyKernel<<<BlockNum, BlockSize>>>(d_a, d_b, size);
    DEBUG_PRINT

    cudaMemcpy(b, d_b, size*sizeof(int), cudaMemcpyDeviceToHost);
    DEBUG_PRINT
    //cudaDeviceSynchronize();

    for(i = 0; i < size; i++)
        sum_b += b[i];
    DEBUG_PRINT
    cudaFree(d_a);
    cudaFree(d_b);

    printf("sum_a: %d, sum_b: %d\n", sum_a, sum_b);

    return 0;
}
