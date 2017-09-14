#include <stdio.h>
#include <stdlib.h>

//#define DEVICE_ALLOC
#define UVM_ALLOC
//#define HOST_ALLOC

//#define SIZE (1024 * 8)
//#define STEP 512

//#define SIZE (1024 * 1024)
//#define STEP (1024 * 32)
//#define STEP 512

#define SIZE (1024 * 1024 * 1024)
#define STEP (1024 * 1024 * 32)

//#define SIZE (1024 * 1024 * 1024L * 2)
//#define STEP (1024 * 1024 * 32)

//#define PRINT_LAT

__global__ void kernel(int *input, double *total_lat)
{
  unsigned t0, t1, lat;
  __shared__ int s_tmp;
  int tmp;
  unsigned maxlat, minlat;

  s_tmp = 0;
  maxlat = minlat = 0;
  for (unsigned long long i = 0; i < SIZE; i += STEP) {
    t0 = clock();
    __syncthreads();
    tmp = input[i];
    __syncthreads();
    t1 = clock();
    lat = t1 - t0;
#ifdef PRINT_LAT
    printf("0x%10llx: %d\n", i, lat);
#endif
    s_tmp += lat;
    if (lat > maxlat)
      maxlat = lat;
    if (lat < minlat || minlat == 0)
      minlat = lat;
  }
  total_lat[0] = s_tmp;
  total_lat[1] = maxlat;
  total_lat[2] = minlat;
}

int main()
{
  int *d_input;
  double *total_lat, h_total_lat[3];

#if defined(DEVICE_ALLOC)
  cudaMalloc(&d_input, SIZE*sizeof(int));
#elif defined(UVM_ALLOC)
  cudaMallocManaged(&d_input, SIZE*sizeof(int));
#elif defined(HOST_ALLOC)
  cudaMallocHost(&d_input, SIZE*sizeof(int));
#else
  return 0;
#endif
  cudaMalloc(&total_lat, 3*sizeof(double));

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

  kernel<<<1, 1>>>(d_input, total_lat);

  cudaMemcpy(&h_total_lat[0], total_lat, 3*sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(total_lat);
  double AvgLat = h_total_lat[0] / (SIZE / STEP);
  printf("Average latency: %f\n", AvgLat);
  printf("Max latency: %f\n", h_total_lat[1]);
  printf("Min latency: %f\n", h_total_lat[2]);
  return 0;
}
