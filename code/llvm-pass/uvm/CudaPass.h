#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

class FuncInfoEntry {
  public:
    Function *func;
    CallInst *call_point;
    Value *arg;
    Value *arg_value;
    Value *local_copy = NULL;
    unsigned type; // 1: alias, 2: base alias
    FuncInfoEntry *parent;

    FuncInfoEntry(Function *in_func) : func(in_func), parent(NULL) {}
    FuncInfoEntry(Function *in_func, CallInst *in_call_inst, Value *in_arg, Value *in_arg_value, unsigned in_type) : func(in_func), call_point(in_call_inst), arg(in_arg), arg_value(in_arg_value), type(in_type) {
      assert(type >= 1 && type <= 2);
    }

    void setParent(FuncInfoEntry *in_parent) {
      parent = in_parent;
    }

    void setLocalCopy(Value *in_local_copy) {
      assert(!local_copy);
      local_copy = in_local_copy;
    }

    void dump() {
      errs() << "FuncInfoEntry: call: ";
      call_point->dump();
      errs() << "                arg: ";
      arg->dump();
    }
  private:
};

class DataEntry {
  public:
    Value *base_ptr;
    unsigned type; // 0: host, 1: device, 2: managed
    Value *size;
    Type *ptr_type;
  
    DataEntry *pair_entry;
    Value *reallocated_base_ptr;
    SmallVector<Value*, 8> alias_ptrs;
    SmallVector<Value*, 2> base_alias_ptrs;
    bool keep_me; // keep original allocation & data transfer
  
    Instruction *alloc;
    SmallVector<Instruction*, 2> send2kernel;
    SmallVector<Instruction*, 2> kernel;
    Instruction *free;
  
    DenseMap<Function*, FuncInfoEntry*> func_map; // keep uvmMallocInfo copies for each function involved
    SmallVector<FuncInfoEntry*, 2> func_stack; // a stack version of above
  
    DataEntry(Value *in_base_ptr, unsigned in_type, Value *in_size) {
      base_ptr = in_base_ptr;
      type = in_type;
      assert(type >= 0 && type <= 2);
      size = in_size;
  
      pair_entry = NULL;
      reallocated_base_ptr = NULL;
      keep_me = false;
      alloc = NULL;
      free = NULL;
    }

    void insertFuncInfoEntry(FuncInfoEntry *in_fie) {
      Function *F = in_fie->func;
      func_map.insert(std::make_pair(F, in_fie));
      func_stack.push_back(in_fie);
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
  
    DataEntry* getDataEntry(FuncInfoEntry *fie) {
      Function *f = fie->func;
      for (auto DMEntry : DataMap) {
        DataEntry *DE = DMEntry.second;
        if (DE->func_map.find(f) != DE->func_map.end()) {
          assert(fie == DE->func_map.find(f)->second);
          return DE;
        }
      }
      return NULL;
    }
};
