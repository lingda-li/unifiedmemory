#include <stdio.h>
#include <stdlib.h>

//#define DEVICE_ALLOC
#define UVM_ALLOC
//#define HOST_ALLOC

//#define SIZE (1024 * 8)
//#define STEP 16
//#define SIZE (1024 * 32)
//#define STEP 128

#define SIZE (1024 * 1024)
#define STEP (1024 * 32)

//#define SIZE (1024 * 1024 * 1024)
//#define STEP (1024 * 1024 * 32)
//#define STEP (512)

//#define SIZE (1024 * 1024 * 1024L * 5)
//#define SIZE (1024 * 1024 * 512L * 7)
//#define STEP (1024 * 1024 * 32)

#define PRINT_LAT

#define LAT_ARRAY_SIZE 12
#define LAT_LOWER_BOUND 3000
#define LAT_HIGHER_BOUND 10000

__global__ void warmup(int *input, int gid, int gnum)
{
  __shared__ int s_tmp;
  s_tmp = 0;

  for (unsigned long long i = (SIZE / gnum) * gid; i < (SIZE / gnum) * (gid + 1); i += STEP) {
    s_tmp += input[i];
  }
}

__global__ void kernel(int *input, double *total_lat, unsigned long long size)
{
  unsigned t0, t1, lat;
  __shared__ int s_tmp;
  int tmp;
  double maxlat, minlat, totallat;
  double maxlat_l, minlat_l, totallat_l;
  double maxlat_s, minlat_s, totallat_s;
  double llat_num, slat_num;

  s_tmp = 0;
  totallat = maxlat = minlat = 0.0;
  totallat_l = maxlat_l = minlat_l = 0.0;
  totallat_s = maxlat_s = minlat_s = 0.0;
  llat_num = slat_num = 0.0;

  for (unsigned long long i = 0; i < size; i += STEP) {
    t0 = clock();
    __syncthreads();
    tmp = input[i];
    __syncthreads();
    t1 = clock();
    lat = t1 - t0;
    s_tmp = tmp;
#ifdef PRINT_LAT
    printf("0x%10llx: %d\n", i, lat);
#endif
    totallat += lat;
    if (lat > maxlat)
      maxlat = lat;
    if (lat < minlat || minlat == 0)
      minlat = lat;

    // classify lat
    if (lat >= LAT_LOWER_BOUND && lat <= LAT_HIGHER_BOUND)
      total_lat[3] += lat;
    else if (lat < LAT_LOWER_BOUND) {
      totallat_s += lat;
      if (lat > maxlat_s)
        maxlat_s = lat;
      if (lat < minlat_s || minlat_s == 0)
        minlat_s = lat;
      slat_num++;
    } else {
      totallat_l += lat;
      if (lat > maxlat_l)
        maxlat_l = lat;
      if (lat < minlat_l || minlat_l == 0)
        minlat_l = lat;
      llat_num++;
    }
  }
  total_lat[0] = totallat;
  total_lat[1] = maxlat;
  total_lat[2] = minlat;

  total_lat[4] = totallat_l;
  total_lat[5] = maxlat_l;
  total_lat[6] = minlat_l;

  total_lat[7] = totallat_s;
  total_lat[8] = maxlat_s;
  total_lat[9] = minlat_s;

  total_lat[10] = llat_num;
  total_lat[11] = slat_num;
}

int main()
{
  int numGPUs = 0;
  cudaGetDeviceCount(&numGPUs);
  printf("# GPUs: %d\n", numGPUs);
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

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    warmup<<<1, 1>>>(d_input, d, numGPUs);
    cudaDeviceSynchronize();
  }

  cudaSetDevice(0);
  kernel<<<1, 1>>>(d_input, total_lat, SIZE/2);
  cudaDeviceSynchronize();

  cudaMemcpy(h_total_lat, total_lat, LAT_ARRAY_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
  double AvgLat = h_total_lat[0] / (SIZE / STEP);
  printf("Average latency: %f (%f / %lld)\n", AvgLat, h_total_lat[0], SIZE / STEP);
  printf("Max latency: %f\n", h_total_lat[1]);
  printf("Min latency: %f\n", h_total_lat[2]);
  printf("\n");
  printf("Average latency (large): %f (%f / %f)\n", h_total_lat[4] / h_total_lat[10], h_total_lat[4], h_total_lat[10]);
  printf("Max latency (large): %f\n", h_total_lat[5]);
  printf("Min latency (large): %f\n", h_total_lat[6]);
  printf("\n");
  printf("Average latency (short): %f (%f / %f)\n", h_total_lat[7] / h_total_lat[11], h_total_lat[7], h_total_lat[11]);
  printf("Max latency (short): %f\n", h_total_lat[8]);
  printf("Min latency (short): %f\n", h_total_lat[9]);
  printf("\n");
  printf("Abnormal total: %f\n", h_total_lat[3]);

  kernel<<<1, 1>>>(&d_input[SIZE/2], total_lat, SIZE/2);
  cudaDeviceSynchronize();

  cudaMemcpy(h_total_lat, total_lat, LAT_ARRAY_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
  AvgLat = h_total_lat[0] / (SIZE / STEP);
  printf("Average latency: %f (%f / %lld)\n", AvgLat, h_total_lat[0], SIZE / STEP);
  printf("Max latency: %f\n", h_total_lat[1]);
  printf("Min latency: %f\n", h_total_lat[2]);
  printf("\n");
  printf("Average latency (large): %f (%f / %f)\n", h_total_lat[4] / h_total_lat[10], h_total_lat[4], h_total_lat[10]);
  printf("Max latency (large): %f\n", h_total_lat[5]);
  printf("Min latency (large): %f\n", h_total_lat[6]);
  printf("\n");
  printf("Average latency (short): %f (%f / %f)\n", h_total_lat[7] / h_total_lat[11], h_total_lat[7], h_total_lat[11]);
  printf("Max latency (short): %f\n", h_total_lat[8]);
  printf("Min latency (short): %f\n", h_total_lat[9]);
  printf("\n");
  printf("Abnormal total: %f\n", h_total_lat[3]);

  cudaFree(d_input);
  cudaFree(total_lat);
  return 0;
}
