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
//#define SIZE (8 * 1024 * 1024 / sizeof(int) * 1024)

int main()
{
    uint64_t i;
    uint64_t size = SIZE;
    double *d_a, *d_b, *d_c;
    double sum_a, sum_c;
    sum_a = sum_c = 0.0;
    DEBUG_PRINT

	cudaMallocManaged((void **)&d_a, size*sizeof(double), 1);
	cudaMallocManaged((void **)&d_b, size*sizeof(double), 1);
	cudaMallocManaged((void **)&d_c, size*sizeof(double), 1);
    DEBUG_PRINT
    for(i = 0; i < size; i++) {
        d_a[i] = (rand() % 10) * 0.5;
        d_b[i] = (rand() % 2 + 1) * 1.0;
        sum_a += d_a[i];
    }
    DEBUG_PRINT

//#pragma omp target data map(to:size) map(to:d_a[0:size]) map(from:d_b[0:size])
#pragma omp target data map(to:size)
    {
#pragma omp target teams distribute parallel for is_device_ptr(d_a) is_device_ptr(d_b) is_device_ptr(d_c)
    for (uint64_t i = 0; i < size; i++) {
        d_c[i] = d_a[i] / d_b[i];
    }
    }
    cudaDeviceSynchronize();

    DEBUG_PRINT

    for(i = 0; i < size; i++)
        sum_c += d_c[i];

    printf("sum_a: %f, sum_c: %f\n", sum_a, sum_c);

    return 0;
}
