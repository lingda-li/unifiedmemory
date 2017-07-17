#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define MYDEBUG
#ifdef MYDEBUG
#define DEBUG_PRINT printf("here: %d\n", __LINE__); fflush(stdout);
#else
#define DEBUG_PRINT
#endif

#define SIZE 10240

int main()
{
    int i;
    int size = SIZE;
    int BlockSize = 256;
    int BlockNum = (size + BlockSize - 1) / BlockSize;
    int *d_a, *d_b;
    int sum_a, sum_b;
    sum_a = sum_b = 0;
    DEBUG_PRINT

	cudaMallocManaged((void **)&d_a, size*sizeof(int), 1);
	cudaMallocManaged((void **)&d_b, size*sizeof(int), 1);
    DEBUG_PRINT
    for(i = 0; i < size; i++) {
        d_a[i] = rand() % 100;
        sum_a += d_a[i];
    }
    DEBUG_PRINT

//#pragma omp target data map(to:size) map(to:d_a[0:size]) map(from:d_b[0:size])
#pragma omp target data map(to:size)
#pragma omp target teams distribute parallel for is_device_ptr(d_a) is_device_ptr(d_b)
    for (int i = 0; i < size; i++) {
        d_b[i] = d_a[i] * 2;
    }
    cudaDeviceSynchronize();

    DEBUG_PRINT

    for(i = 0; i < size; i++)
        sum_b += d_b[i];

    printf("sum_a: %d, sum_b: %d\n", sum_a, sum_b);

    return 0;
}
