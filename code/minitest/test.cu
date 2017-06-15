#include <stdio.h>
#include <stdlib.h>

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

    cudaMalloc((void **)&d_a, size*sizeof(int));
    cudaMemcpy(d_a, a, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_b, size*sizeof(int));
    cudaMemcpy(d_b, b, size*sizeof(int), cudaMemcpyHostToDevice);

    MyKernel<<<BlockNum, BlockSize>>>(d_a, d_b, size);

    cudaMemcpy(b, d_b, size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);

    for(i = 0; i < size; i++)
        sum_b += b[i];

    printf("sum_a: %d, sum_b: %d\n", sum_a, sum_b);

    return 0;
}
