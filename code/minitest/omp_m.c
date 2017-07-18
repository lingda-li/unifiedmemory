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

//#define USE_UVM

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

	cudaMallocManaged((void **)&d_a, size*sizeof(double), cudaMemAttachHost);
	cudaMallocManaged((void **)&d_b, size*sizeof(double), cudaMemAttachHost);
	cudaMallocManaged((void **)&d_c, size*sizeof(double), cudaMemAttachHost);
	//cudaMalloc((void **)&d_a, size*sizeof(double));
	//cudaMalloc((void **)&d_b, size*sizeof(double));
	//cudaMalloc((void **)&d_c, size*sizeof(double));
    //d_a = (double*)malloc(size*sizeof(double));
    //d_b = (double*)malloc(size*sizeof(double));
    //d_c = (double*)malloc(size*sizeof(double));
    DEBUG_PRINT
    for(i = 0; i < size; i++) {
        d_a[i] = (rand() % 10) * 0.5;
        d_b[i] = (rand() % 2 + 1) * 1.0;
        sum_a += d_a[i];
    }
    DEBUG_PRINT

#ifdef USE_UVM
#pragma omp target data map(to:size)
#else
#pragma omp target data map(to:d_a[0:size]) map(to:d_b[0:size]) map(from:d_c[0:size])
#endif
    {
#ifdef USE_UVM
#pragma omp target teams distribute parallel for is_device_ptr(d_a) is_device_ptr(d_b) is_device_ptr(d_c)
#else
#pragma omp target teams distribute parallel for
#endif
    for (uint64_t i = 0; i < size; i++) {
        d_c[i] = d_a[i] / d_b[i];
    }
    }
    cudaDeviceSynchronize();

    DEBUG_PRINT

    for(i = 0; i < size; i++)
        sum_c += d_c[i];

    printf("sum_a: %f, sum_c: %f\n", sum_a, sum_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
