#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

class DataEntry {
public:
  Value *base_ptr;
  unsigned type; // 0: host, 1: device, 2: managed
  Value *size;
  Type *ptr_type;

  DataEntry *pair_entry;
  Value *reallocated_base_ptr;
  SmallVector<Value *, 8> alias_ptrs;
  SmallVector<Value *, 2> base_alias_ptrs;
  bool keep_me; // keep original allocation & data transfer

  DataEntry(Value *in_base_ptr, unsigned in_type, Value *in_size) {
    base_ptr = in_base_ptr;
    type = in_type;
    assert(type >= 0 && type <= 2);
    size = in_size;

    pair_entry = NULL;
    reallocated_base_ptr = NULL;
    keep_me = false;
  }
};

class DataInfo {
public:
  DenseMap<Value*, DataEntry*> DataMap; // keep all allocated memory space

  DataEntry* getBaseAliasEntry(Value *base_alias_ptr) {
    for (auto DMEntry : DataMap) {
      if (DMEntry.second->base_ptr == base_alias_ptr)
        return DMEntry.second;
      for (Value *CAPTR : DMEntry.second->base_alias_ptrs) {
        if(CAPTR == base_alias_ptr)
          return DMEntry.second;
      }
    }
    return NULL;
  }

  bool tryInsertBaseAliasEntry(DataEntry *data_entry, Value *base_alias_ptr) {
    for (Value *CAPTR : data_entry->base_alias_ptrs) {
      if(CAPTR == base_alias_ptr)
        return false;
    }
    data_entry->base_alias_ptrs.push_back(base_alias_ptr);
    return true;
  }

  DataEntry* getAliasEntry(Value *alias_ptr) {
    for (auto DMEntry : DataMap) {
      for (Value *CAPTR : DMEntry.second->alias_ptrs) {
        if(CAPTR == alias_ptr)
          return DMEntry.second;
      }
    }
    return NULL;
  }

  DataEntry* getAliasEntry(DataEntry *data_entry, Value *alias_ptr) {
    for (Value *CAPTR : data_entry->alias_ptrs) {
      if(CAPTR == alias_ptr)
        return data_entry;
    }
    return NULL;
  }

  bool tryInsertAliasEntry(DataEntry *data_entry, Value *alias_ptr) {
    for (Value *CAPTR : data_entry->alias_ptrs) {
      if(CAPTR == alias_ptr)
        return false;
    }
    data_entry->alias_ptrs.push_back(alias_ptr);
    return true;
  }
};
