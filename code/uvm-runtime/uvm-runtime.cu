#include "uvm-runtime.h"

#define ALL_MANAGED
#define GPU_PRE_PASCAL

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
}

void __uvm_free(struct uvmMallocInfo* uvmInfo)
{
  cudaFree(uvmInfo->devPtr);
  if (!uvmInfo->isSame)
    free(uvmInfo->hostPtr);
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
}
