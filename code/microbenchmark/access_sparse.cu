#include <stdio.h>
#include <stdlib.h>

//#define DEVICE_ALLOC
//#define UVM_ALLOC

#define BLOCK_PER_SM 8
#define SM_NUM 56
#define BLOCK_NUM (SM_NUM * BLOCK_PER_SM)
#define THREAD_PER_BLOCK 256
#define TOTAL_NUM (BLOCK_NUM * THREAD_PER_BLOCK)

#define SIZE (1024 * 1024 * 9 * 7 * 5 * 12L)
//#define STEP (512)
//#define DEV_SIZE SIZE
#define STEP (1)

__global__ void kernel(int *input)
{
  int tmp;
  __shared__ int s_tmp;
  s_tmp = 0;

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long begin = DEV_SIZE / TOTAL_NUM * idx;
  unsigned long long end = DEV_SIZE / TOTAL_NUM * (idx + 1);
  for (unsigned long long i = begin; i < end; i += STEP) {
    tmp = input[i];
    s_tmp += tmp;
  }
}

class timer
{
  cudaEvent_t start;
  cudaEvent_t end;

public:
  timer()
  {   
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start,0);
  }   

  ~timer()
  {   
    cudaEventDestroy(start);
    cudaEventDestroy(end);
  }   

  float milliseconds_elapsed()
  {   
    float elapsed_time;
    cudaEventRecord(end, 0); 
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    return elapsed_time;
  }   
  float seconds_elapsed()
  {   
    return milliseconds_elapsed() / 1000.0;
  }   
};

int main()
{
  timer time_overall;
  int *d_input;
#if defined(DEVICE_ALLOC)
  cudaMalloc(&d_input, SIZE*sizeof(int));
#elif defined(UVM_ALLOC)
  cudaMallocManaged(&d_input, SIZE*sizeof(int));
#else
  return 0;
#endif

  // init
#if defined(DEVICE_ALLOC)
  int *h_input;
  h_input = (int*)malloc(SIZE*sizeof(int));
  for (unsigned long long i = 0; i < SIZE; i++) {
    h_input[i] = i;
  }
  timer time_in;
  cudaMemcpy(d_input, h_input, SIZE*sizeof(int), cudaMemcpyHostToDevice);
#elif defined(UVM_ALLOC)
  for (unsigned long long i = 0; i < SIZE; i++) {
    d_input[i] = i;
  }
  timer time_in;
#endif

  kernel<<<BLOCK_NUM, THREAD_PER_BLOCK>>>(d_input);
  cudaDeviceSynchronize();
  double in_time = time_in.seconds_elapsed();

  timer time_out;
  unsigned long long sum = 0;
#if defined(DEVICE_ALLOC)
  cudaMemcpy(h_input, d_input, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  for (unsigned long long i = 0; i < SIZE; i++) {
    sum += h_input[i];
  }
#elif defined(UVM_ALLOC)
  for (unsigned long long i = 0; i < SIZE; i++) {
    sum += d_input[i];
  }
#endif
  double out_time = time_out.seconds_elapsed();

  cudaFree(d_input);
#if defined(DEVICE_ALLOC)
  free(h_input);
#endif

  double total_time = time_overall.seconds_elapsed();
  printf("Time: %f sec\n", total_time);
  printf("In time: %f sec\n", in_time);
  printf("Out time: %f sec\n", out_time);
  printf("Sum: %llu\n", sum);
  return 0;
}
