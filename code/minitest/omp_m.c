#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <omp.h>

#define MYDEBUG
#ifdef MYDEBUG
#define DEBUG_PRINT printf("here: %d\n", __LINE__); fflush(stdout);
#else
#define DEBUG_PRINT
#endif

#define OMP_ALLOC
//#define HYB_ALLOC
//#define UVM_ALLOC
//#define HOST_ALLOC
//#define DEVICE_ALLOC

#define MAP_ALL

//#define NO_HOST_ACCESS
#define SEC_CALL

#define SIZE 10240
//#define SIZE (18 / 3 * 1024 * 1024 / sizeof(double) * 1024)

int main()
{
    uint64_t i;
    uint64_t size = SIZE;
    double *d_a, *d_b, *d_c;
    double sum_a, sum_c;
    sum_a = sum_c = 0.0;
    DEBUG_PRINT

#if defined(OMP_ALLOC)
    d_a = (double *)omp_target_alloc(size * sizeof(double), omp_get_default_device());
    d_b = (double *)omp_target_alloc(size * sizeof(double), omp_get_default_device());
    d_c = (double *)omp_target_alloc(size * sizeof(double), omp_get_default_device());
#elif defined(HYB_ALLOC)
    d_a = (double *)malloc(size*sizeof(double));
    d_b = (double *)malloc(size*sizeof(double));
    d_c = (double *)omp_target_alloc(size * sizeof(double), omp_get_default_device());
#elif defined(UVM_ALLOC)
	cudaMallocManaged((void **)&d_a, size*sizeof(double), cudaMemAttachGlobal);
	cudaMallocManaged((void **)&d_b, size*sizeof(double), cudaMemAttachGlobal);
	cudaMallocManaged((void **)&d_c, size*sizeof(double), cudaMemAttachGlobal);
#elif defined(HOST_ALLOC)
	cudaMallocHost((void **)&d_a, size*sizeof(double));
	cudaMallocHost((void **)&d_b, size*sizeof(double));
	cudaMallocHost((void **)&d_c, size*sizeof(double));
#elif defined(DEVICE_ALLOC)
	cudaMalloc((void **)&d_a, size*sizeof(double));
	cudaMalloc((void **)&d_b, size*sizeof(double));
	cudaMalloc((void **)&d_c, size*sizeof(double));
#else
    d_a = (double *)malloc(size*sizeof(double));
    d_b = (double *)malloc(size*sizeof(double));
    d_c = (double *)malloc(size*sizeof(double));
#endif
    DEBUG_PRINT
#ifndef NO_HOST_ACCESS
    for(i = 0; i < size; i++) {
        d_a[i] = (rand() % 10) * 0.5;
        d_b[i] = (rand() % 2 + 1) * 1.0;
        sum_a += d_a[i];
    }
    DEBUG_PRINT
#endif

#if defined(MAP_ALL)
#pragma omp target data map(to:size) map(to:d_a[0:size]) map(to:d_b[0:size]) map(from:d_c[0:size])
#elif defined(OMP_ALLOC)
#pragma omp target data map(to:size)
#elif defined(HYB_ALLOC)
#pragma omp target data map(to:size) map(to:d_a[0:size]) map(to:d_b[0:size])
#else
#pragma omp target data map(to:size) map(to:d_a[0:size]) map(to:d_b[0:size]) map(from:d_c[0:size])
#endif
    {
#if defined(MAP_ALL)
#pragma omp target teams distribute parallel for
#elif defined(OMP_ALLOC)
#pragma omp target teams distribute parallel for is_device_ptr(d_a) is_device_ptr(d_b) is_device_ptr(d_c)
#elif defined(HYB_ALLOC)
#pragma omp target teams distribute parallel for is_device_ptr(d_c)
#else
#pragma omp target teams distribute parallel for
#endif
    for (uint64_t i = 0; i < size; i++) {
        d_c[i] = d_a[i] / d_b[i];
    }
#ifdef SEC_CALL
    cudaDeviceSynchronize();
#if defined(MAP_ALL)
#pragma omp target teams distribute parallel for
#elif defined(OMP_ALLOC)
#pragma omp target teams distribute parallel for is_device_ptr(d_a) is_device_ptr(d_b) is_device_ptr(d_c)
#elif defined(HYB_ALLOC)
#pragma omp target teams distribute parallel for is_device_ptr(d_c)
#else
#pragma omp target teams distribute parallel for
#endif
    for (uint64_t i = 0; i < size; i++) {
        d_c[i] += d_a[i] / d_b[i];
    }
#endif
    }
    cudaDeviceSynchronize();

    DEBUG_PRINT

#ifndef NO_HOST_ACCESS
    for(i = 0; i < size; i++)
        sum_c += d_c[i];
#endif

    printf("sum_a: %f, sum_c: %f\n", sum_a, sum_c);

#if defined(OMP_ALLOC)
    omp_target_free(d_a, omp_get_default_device());
    omp_target_free(d_b, omp_get_default_device());
    omp_target_free(d_c, omp_get_default_device());
#elif defined(HYB_ALLOC)
    free(d_a);
    free(d_b);
    omp_target_free(d_c, omp_get_default_device());
#elif defined(UVM_ALLOC) || defined(HOST_ALLOC) || defined(DEVICE_ALLOC)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
#else
    free(d_a);
    free(d_b);
    free(d_c);
#endif
    return 0;
}
