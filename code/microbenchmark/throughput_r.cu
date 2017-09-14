#include <stdio.h>

//#define DEVICE_ALLOC
#define UVM_ALLOC
//#define HOST_ALLOC

//#define SIZE (2048 * 4)
//#define SIZE (1024 * 1024)
//#define SIZE (1024 * 1024 * 1024)
#define SIZE (1024 * 1024 * 1024L * 2)

__global__ void kernel(int *input, unsigned long long size)
{
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int s_tmp;

  if (i < size)
    s_tmp += input[i];
}

int main()
{
  int *d_input;

  cudaEvent_t start;
  cudaEvent_t end;

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
  for (unsigned long long i = 0; i < SIZE; i++) {
    h_input[i] = rand() % 10;
  }
  cudaMemcpy(d_input, h_input, SIZE*sizeof(int), cudaMemcpyHostToDevice);
#elif defined(UVM_ALLOC) || defined(HOST_ALLOC)
  for (unsigned long long i = 0; i < SIZE; i++) {
    d_input[i] = rand() % 10;
  }
#endif

  unsigned ThreadNum = 256;
  unsigned long long BlockNum = (SIZE - 1) / ThreadNum + 1;

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start,0);

  kernel<<<BlockNum, ThreadNum>>>(d_input, SIZE);

  cudaThreadSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float elapsed_time1;
  cudaEventElapsedTime(&elapsed_time1, start, end);

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start,0);

  kernel<<<BlockNum, ThreadNum>>>(d_input, SIZE);

  cudaThreadSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float elapsed_time2;
  cudaEventElapsedTime(&elapsed_time2, start, end);

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start,0);

  kernel<<<BlockNum, ThreadNum>>>(d_input, SIZE);

  cudaThreadSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float elapsed_time3;
  cudaEventElapsedTime(&elapsed_time3, start, end);

  double AvgTP1 = (double)SIZE*sizeof(int) / (elapsed_time1 / 1000.0) / 1e9;
  double AvgTP2 = (double)SIZE*sizeof(int) / (elapsed_time2 / 1000.0) / 1e9;
  double AvgTP3 = (double)SIZE*sizeof(int) / (elapsed_time3 / 1000.0) / 1e9;
  printf("Average throughput: %f GB/s, %f GB/s, %f GB/s\n", AvgTP1, AvgTP2, AvgTP3);
  printf("Time: %f ms, %f ms, %f ms\n", elapsed_time1, elapsed_time2, elapsed_time3);

  cudaFree(d_input);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return 0;
}
