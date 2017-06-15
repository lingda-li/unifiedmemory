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
  unsigned type; // 0: host, 1: device
  SmallVector<Value *, 4> alias_ptrs;
  DataEntry *pair_entry;
  Value *reallocated_base_ptr;
};

class DataInfo {
public:
  DenseMap<Value*, DataEntry*> DataMap;
  DataEntry* getAliasEntry(Value *alias_ptr) {
    for (auto DMEntry : DataMap) {
      for (Value *CAPTR: DMEntry.second->alias_ptrs) {
        if(CAPTR == alias_ptr)
          return DMEntry.second;
      }
    }
    return NULL;
  }

  DataEntry* getAliasEntry(DataEntry *data_entry, Value *alias_ptr) {
    for (Value *CAPTR: data_entry->alias_ptrs) {
      if(CAPTR == alias_ptr)
        return data_entry;
    }
    return NULL;
  }

  void insertAliasEntry(DataEntry *data_entry, Value *alias_ptr) {
    data_entry->alias_ptrs.push_back(alias_ptr);
  }
};
