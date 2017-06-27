#include "stdio.h"
#include "uvm-runtime.h"

#define ALL_MANAGED
#define GPU_PRE_PASCAL

#define UVM_DEBUG
#ifdef UVM_DEBUG
#define DEBUG_PRINT printf("here: %d\n", __LINE__); fflush(stdout);
#else
#define DEBUG_PRINT
#endif

void __uvm_malloc(struct uvmMallocInfo* uvmInfo)
{
  size_t size = uvmInfo->size;

#if defined (ALL_MANAGED)
  cudaMallocManaged(&uvmInfo->devPtr, size);
  uvmInfo->hostPtr = uvmInfo->devPtr;
  uvmInfo->isSame = true;
#else
  uvmInfo->hostPtr = malloc(size);
  cudaMalloc(&uvmInfo->devPtr, size);
  uvmInfo->isSame = false;
#endif
#ifdef UVM_DEBUG
  printf("Debug: __uvm_malloc 0x%lx 0x%lx (%lu)\n", (unsigned long)uvmInfo->hostPtr, (unsigned long)uvmInfo->devPtr, size); fflush(stdout);
#endif
}

void __uvm_free(struct uvmMallocInfo* uvmInfo)
{
  cudaFree(uvmInfo->devPtr);
  if (!uvmInfo->isSame)
    free(uvmInfo->hostPtr);
#ifdef UVM_DEBUG
  printf("Debug: __uvm_free 0x%lx 0x%lx\n", (unsigned long)uvmInfo->hostPtr, (unsigned long)uvmInfo->devPtr); fflush(stdout);
#endif
}

void __uvm_memcpy(struct uvmMallocInfo* uvmInfo, cudaMemcpyKind kind)
{
  if (uvmInfo->isSame) {
#ifdef GPU_PRE_PASCAL
    if (kind == cudaMemcpyDeviceToHost)
      cudaDeviceSynchronize();
#endif
    return;
  }
  void* devPtr = uvmInfo->devPtr;
  void* hostPtr = uvmInfo->hostPtr;
  size_t size = uvmInfo->size;

  if (kind == cudaMemcpyHostToDevice)
    cudaMemcpy(devPtr, hostPtr, size, kind);
  else
    cudaMemcpy(hostPtr, devPtr, size, kind);
#ifdef UVM_DEBUG
  printf("Debug: __uvm_memcpy 0x%lx 0x%lx\n", (unsigned long)uvmInfo->hostPtr, (unsigned long)uvmInfo->devPtr); fflush(stdout);
#endif
}
