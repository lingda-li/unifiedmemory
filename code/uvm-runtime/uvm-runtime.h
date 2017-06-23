#define ALL_MANAGED
#define GPU_PRE_PASCAL

struct uvmMallocInfo {
  void* devPtr;
  size_t size;
  void* hostPtr;
  bool isSame;
};

void uvmMalloc(struct uvmMallocInfo* uvmInfo);
void uvmFree(struct uvmMallocInfo* uvmInfo);
void uvmMemcpy(struct uvmMallocInfo* uvmInfo, cudaMemcpyKind kind);
