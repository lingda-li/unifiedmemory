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

	cudaMallocManaged((void **)&d_a, size*sizeof(int));
	cudaMallocManaged((void **)&d_b, size*sizeof(int));
    for(i = 0; i < size; i++)
        d_a[i] = rand() % 100;

    MyKernel<<<BlockNum, BlockSize>>>(d_a, d_b, size);

    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
