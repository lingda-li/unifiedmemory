#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

#define DEBUG_PRINT {errs() << "Error: "<< __LINE__ << "\n";}

class FuncInfoEntry {
  private:

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

    // Load & store frequency
    double load_freq, store_freq;
  
    DataEntry() {
      base_ptr = NULL;
      size = NULL;

      pair_entry = NULL;
      reallocated_base_ptr = NULL;
      keep_me = false;
      alloc = NULL;
      free = NULL;

      load_freq = store_freq = 0.0;
    }

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

      load_freq = store_freq = 0.0;
    }

    void insertFuncInfoEntry(FuncInfoEntry *in_fie) {
      Function *F = in_fie->func;
      func_map.insert(std::make_pair(F, in_fie));
      func_stack.push_back(in_fie);
    }
};

class FuncArgEntry {
  private:
    Function *func;
    Value *arg;
    int arg_num;

    SmallVector<Value*, 8> alias_ptrs;
    SmallVector<Value*, 2> base_alias_ptrs;

    // Load & store frequency
    double load_freq, store_freq;
    bool valid;

  public:
    FuncArgEntry(Function *f, Value *a, int an)
      : func(f), arg(a), arg_num(an), load_freq(0.0), store_freq(0.0), valid(false) {}

    bool isMatch(Function *f, int an) {
      if (func == f && arg_num == an)
        return true;
      return false;
    }
    double getLoadFreq() { return load_freq; }
    double getStoreFreq() { return store_freq; }
    bool getValid() { return valid; }
    void setValid() { valid = true; }

    bool tryInsertAliasPtr(Value *alias_ptr) {
      for (Value *CAPTR : alias_ptrs) {
        if(CAPTR == alias_ptr)
          return false;
      }
      alias_ptrs.push_back(alias_ptr);
      return true;
    }
    bool tryInsertBaseAliasPtr(Value *alias_ptr) {
      for (Value *CAPTR : base_alias_ptrs) {
        if(CAPTR == alias_ptr)
          return false;
      }
      base_alias_ptrs.push_back(alias_ptr);
      return true;
    }
};

class MemAccessInfo {
  private:
    std::vector<DataEntry> Entries; // memory allocation info
    std::vector<FuncArgEntry> FuncArgs; // function argument access info

  public:
    std::vector<DataEntry>* getEntries() { return &Entries; }

    void newEntry(DataEntry data_entry) {
      Entries.push_back(data_entry);
    }

    DataEntry* getBaseAliasEntry(Value *base_alias_ptr) {
      for (auto &E : Entries) {
        if (E.base_ptr == base_alias_ptr)
          return &E;
        for (Value *CAPTR : E.base_alias_ptrs) {
          if(CAPTR == base_alias_ptr)
            return &E;
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
      for (auto &E : Entries) {
        for (Value *CAPTR : E.alias_ptrs) {
          if(CAPTR == alias_ptr)
            return &E;
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

    void newFuncArg(Function *f, Value *a, int an) {
      FuncArgEntry E(f, a, an);
      FuncArgs.push_back(E);
      bool Succeed = E.tryInsertAliasPtr(a);
      assert(Succeed);
    }

    FuncArgEntry* getFuncArgEntry(Function *f, int an) {
      for (auto &E : FuncArgs) {
        if (E.isMatch(f, an))
          return &E;
      }
      return NULL;
    }
};
