#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE (1024 * 1024 * 7 * 512L)
#define STEP (1024 * 16 * 2)

int main()
{
  int *h_input;
  h_input = (int*)malloc(SIZE*sizeof(int));
  struct timeval start, end;
  gettimeofday(&start, NULL);
  for (unsigned long long i = 0; i < SIZE; i += STEP) {
    h_input[i] = i;
  }
  gettimeofday(&end, NULL);
  printf("Access #: %llu\n", SIZE / STEP);
  unsigned long long time = (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec;
  printf("Time: %llu us\n", time);
  printf("Accesses per sec: %f\n", (SIZE / STEP * 1000000.0) / time);

  int sum = 0;
  for (unsigned long long i = 0; i < SIZE; i += STEP) {
    sum += h_input[(rand()%SIZE)/STEP*STEP];
  }
  return sum;
}
