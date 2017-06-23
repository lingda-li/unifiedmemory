#include "uvm-runtime.h"

//#define ALL_MANAGED

void uvmMalloc(struct uvmMallocInfo* uvmInfo)
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

void uvmFree(struct uvmMallocInfo* uvmInfo)
{
  cudaFree(uvmInfo->devPtr);
  if (!uvmInfo->isSame)
    free(uvmInfo->hostPtr);
}

void uvmMemcpy(struct uvmMallocInfo* uvmInfo, cudaMemcpyKind kind)
{
  if (uvmInfo->isSame)
    return;
  void* devPtr = uvmInfo->devPtr;
  void* hostPtr = uvmInfo->hostPtr;
  size_t size = uvmInfo->size;

  if (kind == cudaMemcpyHostToDevice)
    cudaMemcpy(devPtr, hostPtr, size, kind);
  else
    cudaMemcpy(hostPtr, devPtr, size, kind);
}
