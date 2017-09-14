#include <stdio.h>

#define DEVICE_ALLOC
//#define UVM_ALLOC
//#define HOST_ALLOC

//#define SIZE (2048 * 4)
//#define SIZE (1024 * 1024)
//#define SIZE (1024 * 1024 * 1024)
#define SIZE (1024 * 1024 * 1024L * 2)

__global__ void kernel(int *input)
{
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  input[i] += 1;
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

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start,0);

  kernel<<<1, 1>>>(d_input);

  cudaThreadSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, end);

  double AvgTP = (double)SIZE*sizeof(int)/(elapsed_time / 1000.0);
  printf("Average throughput: %f\n", AvgTP);

  cudaFree(d_input);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return 0;
}
