#include <stdio.h>
#include <stdlib.h>

#define BLOCK_PER_SM (8 * 2)
#define SM_NUM 56
#define BLOCK_NUM (SM_NUM * BLOCK_PER_SM)
#define THREAD_PER_BLOCK 256
#define TOTAL_NUM (BLOCK_NUM * THREAD_PER_BLOCK)

//#define WARP_AWARE

//#define DEVICE_ALLOC
#define UVM_ALLOC
//#define HOST_ALLOC

//#define SIZE (1024 * 1024 * 2L * TOTAL_NUM)
//#define SIZE (1024 * 1024 * 9 * 7 * 5)
#define SIZE (1024 * 1024 * 7 * 512L)
//#define STEP (512)
#define STEP (1024 * 16)

#define LAT_ARRAY_SIZE 12
#define LAT_LOWER_BOUND 10000
#define LAT_HIGHER_BOUND 20000

__global__ void kernel(int *input, double *total_lat)
{
  //unsigned t0, t1, lat;
  int tmp;
  __shared__ int s_tmp;

  s_tmp = 0;

#ifdef WARP_AWARE
  unsigned idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
#else
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
#endif
  unsigned long long begin = SIZE / TOTAL_NUM * idx;
  unsigned long long end = SIZE / TOTAL_NUM * (idx + 1);
  for (unsigned long long i = begin; i < end; i += STEP) {
    tmp = input[i];
    s_tmp += tmp;
  }
}

int main()
{
  int *d_input;
  double *total_lat, *h_total_lat;

  h_total_lat = (double*)malloc(LAT_ARRAY_SIZE * sizeof(double));
  cudaMalloc(&total_lat, LAT_ARRAY_SIZE*sizeof(double));
  for (int i = 0; i < LAT_ARRAY_SIZE; i++)
    h_total_lat[i] = 0.0;
  cudaMemcpy(total_lat, h_total_lat, LAT_ARRAY_SIZE*sizeof(double), cudaMemcpyHostToDevice);
#if defined(DEVICE_ALLOC)
  cudaMalloc(&d_input, SIZE*sizeof(int));
#elif defined(UVM_ALLOC)
  cudaMallocManaged(&d_input, SIZE*sizeof(int));
  cudaMemAdvise(d_input, SIZE*sizeof(int), cudaMemAdviseSetReadMostly, 0);
#elif defined(HOST_ALLOC)
  cudaMallocHost(&d_input, SIZE*sizeof(int));
#else
  return 0;
#endif

  // init
#if defined(DEVICE_ALLOC)
  int *h_input;
  h_input = (int*)malloc(SIZE*sizeof(int));
  for (unsigned long long i = 0; i < SIZE; i += STEP) {
    h_input[i] = rand();
  }
  cudaMemcpy(d_input, h_input, SIZE*sizeof(int), cudaMemcpyHostToDevice);
#elif defined(UVM_ALLOC) || defined(HOST_ALLOC)
  for (unsigned long long i = 0; i < SIZE; i += STEP) {
    d_input[i] = rand();
  }
#endif

#ifdef WARP_AWARE
  kernel<<<BLOCK_NUM, THREAD_PER_BLOCK * 32>>>(d_input, total_lat);
#else
  kernel<<<BLOCK_NUM, THREAD_PER_BLOCK>>>(d_input, total_lat);
#endif

  cudaMemcpy(h_total_lat, total_lat, LAT_ARRAY_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(total_lat);
  printf("Access #: %llu\n", SIZE / STEP);
#ifdef WARP_AWARE
  printf("Accesses per warp: %llu\n", SIZE / STEP / TOTAL_NUM);
#else
  printf("Accesses per thread: %llu\n", SIZE / STEP / TOTAL_NUM);
#endif
  return 0;
}
