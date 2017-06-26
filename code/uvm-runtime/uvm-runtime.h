#ifndef UVM_RUNTIME_H
#define UVM_RUNTIME_H

extern "C" {
struct uvmMallocInfo {
  void* devPtr;
  size_t size;
  void* hostPtr;
  bool isSame;
};

extern void __uvm_malloc(struct uvmMallocInfo* uvmInfo);
extern void __uvm_free(struct uvmMallocInfo* uvmInfo);
extern void __uvm_memcpy(struct uvmMallocInfo* uvmInfo, cudaMemcpyKind kind);
}

#endif
