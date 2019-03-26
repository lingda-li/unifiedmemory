#ifndef LLVM_UVM_PASS_DATASTRUCTURE_H
#define LLVM_UVM_PASS_DATASTRUCTURE_H

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
    Instruction *call_point;
    Value *arg;
    Value *arg_value;
    Value *local_copy = NULL;
    unsigned type; // 1: alias, 2: base alias
    FuncInfoEntry *parent;

    FuncInfoEntry(Function *in_func) : func(in_func), parent(NULL) {}
    FuncInfoEntry(Function *in_func, Instruction *in_call_inst, Value *in_arg, Value *in_arg_value, unsigned in_type) : func(in_func), call_point(in_call_inst), arg(in_arg), arg_value(in_arg_value), type(in_type) {
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
      call_point->print(errs());
      errs() << "\n";
      errs() << "                arg: ";
      arg->print(errs());
      errs() << "\n";
    }
};

class DataEntry {
  private:
    unsigned rank;

  protected:
    Function *func; // birth function

  public:
    Value *base_ptr;
    unsigned type; // 0: host, 1: device, 2: managed
    Value *size;
    Type *ptr_type;
  
    DataEntry *pair_entry;
    Value *reallocated_base_ptr;
    SmallVector<Value*, 8> alias_ptrs;
    SmallVector<Value*, 2> base_alias_ptrs;
    SmallVector<std::pair<Value*, int64_t>, 2> base_offset_alias_ptrs;
    bool keep_me; // keep original allocation & data transfer
  
    Instruction *alloc;
    SmallVector<Instruction*, 2> send2kernel;
    SmallVector<Instruction*, 2> kernel;
    Instruction *free;
  
    DenseMap<Function*, FuncInfoEntry*> func_map; // keep uvmMallocInfo copies for each function involved
    SmallVector<FuncInfoEntry*, 2> func_stack; // a stack version of above

    // Load & store frequency
    double load_freq, store_freq;
    double tgt_load_freq, tgt_store_freq;
  
    DataEntry() {
      base_ptr = NULL;
      size = NULL;
      func = NULL;

      pair_entry = NULL;
      reallocated_base_ptr = NULL;
      keep_me = false;
      alloc = NULL;
      free = NULL;

      load_freq = store_freq = 0.0;
      tgt_load_freq = tgt_store_freq = 0.0;
    }

    DataEntry(Value *in_base_ptr, unsigned in_type, Value *in_size) {
      base_ptr = in_base_ptr;
      type = in_type;
      assert(type >= 0 && type <= 2);
      size = in_size;
  
      func = NULL;
      pair_entry = NULL;
      reallocated_base_ptr = NULL;
      keep_me = false;
      alloc = NULL;
      free = NULL;

      load_freq = store_freq = 0.0;
      tgt_load_freq = tgt_store_freq = 0.0;
    }

    DataEntry(Value *in_base_ptr, unsigned in_type, Value *in_size, Function *birth_func) {
      base_ptr = in_base_ptr;
      type = in_type;
      assert(type >= 0 && type <= 2);
      size = in_size;
      func = birth_func;
  
      pair_entry = NULL;
      reallocated_base_ptr = NULL;
      keep_me = false;
      alloc = NULL;
      free = NULL;

      load_freq = store_freq = 0.0;
      tgt_load_freq = tgt_store_freq = 0.0;
    }

    void insertFuncInfoEntry(FuncInfoEntry *in_fie) {
      Function *F = in_fie->func;
      func_map.insert(std::make_pair(F, in_fie));
      func_stack.push_back(in_fie);
    }

    DataEntry* getBaseAliasPtr(Value *base_alias_ptr) {
      if (base_ptr == base_alias_ptr)
        return this;
      for (Value *CAPTR : base_alias_ptrs) {
        if(CAPTR == base_alias_ptr)
          return this;
      }
      for (auto CAPTR : base_offset_alias_ptrs) {
        if(CAPTR.first == base_alias_ptr && CAPTR.second == 0)
          return this;
      }
      return NULL;
    }

    std::pair<DataEntry*, int64_t> getBaseOffsetAliasPtr(Value *base_alias_ptr) {
      for (auto CAPTR : base_offset_alias_ptrs) {
        if(CAPTR.first == base_alias_ptr)
          return std::make_pair(this, CAPTR.second);
      }
      return std::make_pair((DataEntry*)NULL, 0);
    }

    DataEntry* getBaseOffsetAliasPtr(Value *base_alias_ptr, int64_t offset) {
      for (auto CAPTR : base_offset_alias_ptrs) {
        if(CAPTR.first == base_alias_ptr && CAPTR.second == offset)
          return this;
      }
      return NULL;
    }

    bool tryInsertAliasPtr(Value *alias_ptr) {
      for (Value *CAPTR : alias_ptrs) {
        if(CAPTR == alias_ptr)
          return false;
      }
      alias_ptrs.push_back(alias_ptr);
      return true;
    }

    bool tryInsertBaseAliasPtr(Value *alias_ptr) {
      if (base_ptr == alias_ptr)
        return false;
      for (Value *CAPTR : base_alias_ptrs) {
        if(CAPTR == alias_ptr)
          return false;
      }
      base_alias_ptrs.push_back(alias_ptr);
      base_offset_alias_ptrs.push_back(std::make_pair(alias_ptr, 0));
      return true;
    }

    bool tryInsertBaseOffsetAlias(Value *base_alias_ptr, int64_t offset) {
      for (auto CAPTR : base_offset_alias_ptrs) {
        if(CAPTR.first == base_alias_ptr && CAPTR.second == offset)
          return false;
      }
      base_offset_alias_ptrs.push_back(std::make_pair(base_alias_ptr, offset));
      return true;
    }

    double getLoadFreq() { return load_freq; }
    double getStoreFreq() { return store_freq; }
    void addLoadFreq(double num) { load_freq += num; }
    void addStoreFreq(double num) { store_freq += num; }
    double getTgtLoadFreq() { return tgt_load_freq; }
    double getTgtStoreFreq() { return tgt_store_freq; }
    void addTgtLoadFreq(double num) { tgt_load_freq += num; }
    void addTgtStoreFreq(double num) { tgt_store_freq += num; }
    Function *getFunc() { return func; }
    void setFunc(Function * f) { func = f; }
    unsigned getRank() { return rank; }
    void setRank(unsigned r) { rank = r; }

    void dumpBase() {
      if (base_ptr) {
        base_ptr->print(errs());
        errs() << "\n";
      } else {
        assert(!alias_ptrs.empty());
        alias_ptrs[0]->print(errs());
        errs() << "\n";
      }
    }

    void dump() {
      errs() << "DataEntry: ";
      if (base_ptr) {
        base_ptr->print(errs());
        errs() << "\n";
      } else {
        assert(!alias_ptrs.empty());
        alias_ptrs[0]->print(errs());
        errs() << "\n";
      }
      errs() << "BaseAlias: ";
      for (Value *CAPTR : base_alias_ptrs) {
        CAPTR->print(errs());
        errs() << "\n";
      }
      errs() << "BaseOffsetAlias: ";
      for (auto CAPTR : base_offset_alias_ptrs) {
        errs() << "(" << CAPTR.second << ") ";
        CAPTR.first->print(errs());
        errs() << "\n";
      }
      errs() << "Alias: ";
      for (Value *CAPTR : alias_ptrs) {
        CAPTR->print(errs());
        errs() << "\n";
      }
    }
};

class FuncArgEntry : public DataEntry {
  private:
    Value *arg;
    int arg_num;
    bool valid;
    std::string func_name;

  public:
    FuncArgEntry(Function *f, Value *a, int an)
      : DataEntry(), arg(a), arg_num(an), valid(false) {
      setFunc(f); }
    FuncArgEntry(Function *f, Value *a, int an, std::string nm)
      : DataEntry(), arg(a), arg_num(an), valid(false), func_name(nm) {
      setFunc(f); }

    bool isMatch(Function *f, int an) {
      if (func == f && arg_num == an)
        return true;
      else if (func == f && an == -1)
        return true;
      return false;
    }
    bool isMatch(std::string name, int an) {
      if (name.find(func_name) == 0)
        if (arg_num == an || an == -1)
          return true;
      return false;
    }
    bool getValid() { return valid; }
    void setValid() { valid = true; }
    int getArgNum() { return arg_num; }
    std::string getName() { return func_name; }

    std::pair<FuncArgEntry*, int64_t> getBaseOffsetAliasPtr(Value *base_alias_ptr) {
      for (auto CAPTR : base_offset_alias_ptrs) {
        if(CAPTR.first == base_alias_ptr)
          return std::make_pair(this, CAPTR.second);
      }
      return std::make_pair((FuncArgEntry*)NULL, 0);
    }

    FuncArgEntry* getBaseOffsetAliasPtr(Value *base_alias_ptr, int64_t offset) {
      for (auto CAPTR : base_offset_alias_ptrs) {
        if(CAPTR.first == base_alias_ptr && CAPTR.second == offset)
          return this;
      }
      return NULL;
    }
    void dump() {
      errs() << "FuncArgEntry: " << func_name << " " << arg_num;
      errs() << "\n";
    }
};

template <class EntryTy>
class MemInfo {
  private:
    std::vector<EntryTy> Entries; // memory allocation info

  public:
    std::vector<EntryTy>* getEntries() { return &Entries; }

    void newEntry(EntryTy data_entry) {
      Entries.push_back(data_entry);
    }

    void eraseTail() {
      Entries.erase(Entries.end());
    }

    EntryTy* getBaseAliasEntry(Value *base_alias_ptr) {
      for (auto &E : Entries) {
        if (E.getBaseAliasPtr(base_alias_ptr))
          return &E;
      }
      return NULL;
    }
  
    SmallVector<std::pair<EntryTy*, int64_t>, 4> getBaseOffsetAliasEntries(Value *base_alias_ptr) {
      SmallVector<std::pair<EntryTy*, int64_t>, 4> Matches;
      for (auto &E : Entries) {
        auto EV = E.getBaseOffsetAliasPtr(base_alias_ptr);
        if (EV.first)
          Matches.push_back(EV);
      }
      return Matches;
    }
  
    EntryTy* getBaseOffsetAliasEntries(Value *base_alias_ptr, int64_t offset) {
      SmallVector<EntryTy*, 4> Matches;
      for (auto &E : Entries) {
        if (auto EV = E.getBaseOffsetAliasPtr(base_alias_ptr, offset))
          Matches.push_back(EV);
      }
      assert(Matches.size() <= 1 && "Two different allocations should not have the same base and offset");
      if (Matches.size() == 0)
        return NULL;
      else
        return Matches.front();
    }
  
    bool tryInsertBaseAliasEntry(EntryTy *data_entry, Value *base_alias_ptr) {
      return data_entry->tryInsertBaseAliasPtr(base_alias_ptr);
    }
  
    bool tryInsertBaseOffsetAliasEntry(EntryTy *data_entry, Value *base_alias_ptr, int64_t offset) {
      return data_entry->tryInsertBaseOffsetAlias(base_alias_ptr, offset);
    }
  
    EntryTy* getAliasEntry(Value *alias_ptr) {
      for (auto &E : Entries) {
        for (Value *CAPTR : E.alias_ptrs) {
          if(CAPTR == alias_ptr)
            return &E;
        }
      }
      return NULL;
    }
  
    EntryTy* getAliasEntry(EntryTy *data_entry, Value *alias_ptr) {
      for (Value *CAPTR : data_entry->alias_ptrs) {
        if(CAPTR == alias_ptr)
          return data_entry;
      }
      return NULL;
    }
  
    bool tryInsertAliasEntry(EntryTy *data_entry, Value *alias_ptr) {
      for (Value *CAPTR : data_entry->alias_ptrs) {
        if(CAPTR == alias_ptr)
          return false;
      }
      data_entry->alias_ptrs.push_back(alias_ptr);
      return true;
    }

    FuncArgEntry* getFuncArgEntry(Function *f, int an) {
      for (auto &E : Entries) {
        if (auto *FAE = dyn_cast<FuncArgEntry>(&E)) {
          if (FAE->isMatch(f, an))
            return FAE;
        } else
          return NULL;
      }
      return NULL;
    }

    FuncArgEntry* getFuncArgEntry(std::string name, int an) {
      for (auto &E : Entries) {
        if (auto *FAE = dyn_cast<FuncArgEntry>(&E)) {
          if (FAE->isMatch(name, an))
            return FAE;
        } else
          return NULL;
      }
      return NULL;
    }

    void dump() {
      for (auto &E : Entries)
        E.dump();
    }
};

#endif // LLVM_UVM_PASS_DATASTRUCTURE_H
