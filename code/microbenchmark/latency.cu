#include <stdio.h>

#define DEVICE_ALLOC
//#define UVM_ALLOC
//#define HOST_ALLOC

//#define SIZE (2048 * 4)
//#define STEP 512

//#define SIZE (1024 * 1024)
//#define STEP (1024 * 32)

//#define SIZE (1024 * 1024 * 1024)
//#define STEP (1024 * 1024 * 32)

#define SIZE (1024 * 1024 * 1024L * 2)
#define STEP (1024 * 1024 * 32)

//#define PRINT_LAT

__global__ void kernel(int *input, unsigned *total_lat)
{
  unsigned t0, t1, lat;
  __shared__ int s_tmp;
  int tmp;

  s_tmp = 0;
  for (unsigned long long i = 0; i < SIZE; i += STEP) {
    t0 = clock();
    __syncthreads();
    tmp = input[i];
    __syncthreads();
    t1 = clock();
    lat = t1 - t0;
#ifdef PRINT_LAT
    printf("%llu: %d\n", i, lat);
#endif
    s_tmp += lat;
  }
  *total_lat = s_tmp;
}

int main()
{
  int *d_input;
  unsigned *total_lat, h_total_lat;

#if defined(DEVICE_ALLOC)
  cudaMalloc(&d_input, SIZE*sizeof(int));
#elif defined(UVM_ALLOC)
  cudaMallocManaged(&d_input, SIZE*sizeof(int));
#elif defined(HOST_ALLOC)
  cudaMallocHost(&d_input, SIZE*sizeof(int));
#else
  return 0;
#endif
  cudaMalloc(&total_lat, sizeof(unsigned));

  kernel<<<1, 1>>>(d_input, total_lat);

  cudaMemcpy(&h_total_lat, total_lat, sizeof(unsigned), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(total_lat);
  double AvgLat = (double)h_total_lat / (SIZE / STEP);
  printf("Average latency: %f\n", AvgLat);
  return 0;
}
