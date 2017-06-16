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
    int *d_a, *d_b;
    int sum_a, sum_b;
    sum_a = sum_b = 0;

	cudaMallocManaged((void **)&d_a, size*sizeof(int));
	cudaMallocManaged((void **)&d_b, size*sizeof(int));
    for(i = 0; i < size; i++) {
        d_a[i] = rand() % 100;
        sum_a += d_a[i];
    }

    MyKernel<<<BlockNum, BlockSize>>>(d_a, d_b, size);

    cudaDeviceSynchronize();

    for(i = 0; i < size; i++)
        sum_b += d_b[i];

    cudaFree(d_a);
    cudaFree(d_b);

    printf("sum_a: %d, sum_b: %d\n", sum_a, sum_b);

    return 0;
}
