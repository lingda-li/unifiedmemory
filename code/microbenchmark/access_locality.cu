#include <stdio.h>
#include <stdlib.h>

//#define DEVICE_ALLOC
#define UVM_ALLOC
//#define HYB_ALLOC

#define BLOCK_PER_SM 8
#define SM_NUM 56
#define BLOCK_NUM (SM_NUM * BLOCK_PER_SM)
#define THREAD_PER_BLOCK 256
#define TOTAL_NUM (BLOCK_NUM * THREAD_PER_BLOCK)
#define MEM_SIZE (1024 * 1024 * 256 * 14L)

#define OCCUPANCY_TUNE
#define SHMEM_SIZE 2048

#define ITER 1
#define OUT_ITER 1
#define SIZE (1024 * 1024 * 9 * 7 * 5 * 14L)
#define STEP (128)
//#define STEP (1)
#define DELAY_ITER 100000

#if defined(HYB_ALLOC)
__global__ void kernel(int *input, int *input2)
#else
__global__ void kernel(int *input)
#endif
{
  int tmp;
  __shared__ int s_tmp;
#ifdef OCCUPANCY_TUNE
  __shared__ int s_idle[SHMEM_SIZE-1];
#endif
  s_tmp = 0;

  //if (blockIdx.x <= BLOCK_NUM / 2)
  //  return;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long begin = SIZE / TOTAL_NUM * idx;
  unsigned long long end = SIZE / TOTAL_NUM * (idx + 1);
  for (unsigned t = 0; t < ITER; t++) {
    for (unsigned long long i = begin; i < end; i += STEP) {
#if defined(HYB_ALLOC)
      if (i >= MEM_SIZE)
        tmp = input2[i - MEM_SIZE];
      else
        tmp = input[i];
#else
      tmp = input[i];
#endif
      //double tmp_f = tmp / 10.0;
      //for (unsigned d = 0; d < DELAY_ITER; d++) {
      //  tmp_f = tmp_f / (tmp_f - 1.0) + 1.0;
      //}
      //tmp = (int)tmp_f;
      s_tmp += tmp;
#ifdef OCCUPANCY_TUNE
      s_idle[i % (SHMEM_SIZE-1)] = tmp;
#endif
    }
    //double tmp_f = s_tmp / 10.0;
    //for (unsigned d = 0; d < DELAY_ITER; d++) {
    //  tmp_f = tmp_f / (tmp_f - 1.0) + 1.0;
    //}
    //s_tmp += (int)tmp_f;
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
#elif defined(HYB_ALLOC)
  if (SIZE <= MEM_SIZE) {
    printf("Error: SIZE is not large enough for hybrid allocation.");
    return 0;
  }
  int *d_input_2;
  cudaMalloc(&d_input, MEM_SIZE*sizeof(int));
  cudaMallocManaged(&d_input_2, (SIZE-MEM_SIZE)*sizeof(int));
#else
  printf("Error: need to define allocation method.");
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
#elif defined(HYB_ALLOC)
  int *h_input;
  h_input = (int*)malloc(MEM_SIZE*sizeof(int));
  for (unsigned long long i = 0; i < MEM_SIZE; i++) {
    h_input[i] = i;
  }
  for (unsigned long long i = 0; i < SIZE - MEM_SIZE; i++) {
    d_input_2[i] = i + MEM_SIZE;
  }
  timer time_in;
  cudaMemcpy(d_input, h_input, MEM_SIZE*sizeof(int), cudaMemcpyHostToDevice);
#endif

  printf("start\n");
  for (unsigned i = 0; i < OUT_ITER; i++) {
#if defined(HYB_ALLOC)
    kernel<<<BLOCK_NUM, THREAD_PER_BLOCK>>>(d_input, d_input_2);
#else
    kernel<<<BLOCK_NUM, THREAD_PER_BLOCK>>>(d_input);
#endif
    cudaDeviceSynchronize();
  }
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
#elif defined(HYB_ALLOC)
  cudaMemcpy(h_input, d_input, MEM_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  for (unsigned long long i = 0; i < MEM_SIZE; i++) {
    sum += h_input[i];
  }
  for (unsigned long long i = 0; i < SIZE - MEM_SIZE; i++) {
    sum += d_input_2[i];
  }
#endif
  double out_time = time_out.seconds_elapsed();

  cudaFree(d_input);
#if defined(DEVICE_ALLOC)
  free(h_input);
#elif defined(HYB_ALLOC)
  free(h_input);
  cudaFree(d_input_2);
#endif

  double total_time = time_overall.seconds_elapsed();
  printf("Time: %f sec\n", total_time);
  printf("In time: %f sec\n", in_time);
  printf("Out time: %f sec\n", out_time);
  printf("Sum: %llu\n", sum);
  return 0;
}
