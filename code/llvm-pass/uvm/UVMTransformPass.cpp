#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "CudaPass.h"
using namespace llvm;

#define DEBUG_PRINT {errs() << "Error: "<< __LINE__ << "\n";}

namespace {
  struct UVMTransformPass : public ModulePass {
    static char ID;
    DataInfo *Info;
    UVMTransformPass() : ModulePass(ID) {
      Info = new DataInfo;
    }

    virtual bool runOnModule(Module &M) {
      errs() << "  ---- UVM Transform (" << M.getName() << ") ----\n";
      // Find all places that allocate memory
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;

        for (auto &BB : F) {
          for (auto &I : BB) {
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              auto *Callee = CI->getCalledFunction();
              if (Callee && Callee->getName() == "cudaMalloc") {
                errs() << "Error: this is for UVM only\n";
                auto *AllocPtr = CI->getArgOperand(0);
                if (auto *BCI = dyn_cast<BitCastInst>(AllocPtr)) {
                  assert(BCI->getNumOperands() == 1);
                  Value *BasePtr = BCI->getOperand(0);
                  if (auto *AI = dyn_cast<AllocaInst>(BasePtr)) {
                    Value *BasePtr = AI;
                    if (Info->DataMap.find(BasePtr) == Info->DataMap.end()) {
                      errs() << "new device entry ";
                      BasePtr->dump();
                      DataEntry *data_entry = new DataEntry(BasePtr, 1, CI->getArgOperand(1)); // device space
                      Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                    } else
                      errs() << "Error: redundant allocation?\n";
                  } else
                    DEBUG_PRINT
                } else if (auto *AI = dyn_cast<AllocaInst>(AllocPtr)) {
                  Value *BasePtr = AI;
                  if (Info->DataMap.find(BasePtr) == Info->DataMap.end()) {
                    errs() << "new device entry ";
                    BasePtr->dump();
                    DataEntry *data_entry = new DataEntry(BasePtr, 1, CI->getArgOperand(1)); // device space
                    Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                  } else
                    errs() << "Error: redundant allocation?\n";
                } else
                  DEBUG_PRINT
              } else if (Callee && Callee->getName() == "cudaMallocManaged") {
                errs() << "Error: cannot address cudaMallocManaged now\n";
              } else if (Callee && Callee->getName() == "malloc") {
                for (auto& U : CI->uses()) {
                  User* user = U.getUser();
                  if (auto *BCI = dyn_cast<BitCastInst>(user)) {
                    for (auto& UU : BCI->uses()) {
                      User* uuser = UU.getUser();
                      if (auto *SI = dyn_cast<StoreInst>(uuser)) {
                        Value *BasePtr = SI->getOperand(1);
                        if (Info->DataMap.find(BasePtr) == Info->DataMap.end()) {
                          errs() << "new host entry ";
                          BasePtr->dump();
                          DataEntry *data_entry = new DataEntry(BasePtr, 0, CI->getArgOperand(0)); // host space
                          data_entry->ptr_type = SI->getOperand(0)->getType();
                          data_entry->alias_ptrs.push_back(BCI);
                          data_entry->alias_ptrs.push_back(CI);
                          errs() << "  alias entry ";
                          BCI->dump();
                          errs() << "  alias entry ";
                          CI->dump();
                          Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                        } else
                          errs() << "Error: redundant allocation?\n";
                      } else
                        DEBUG_PRINT
                    }
                  } else if (auto *SI = dyn_cast<StoreInst>(user)) {
                    Value *BasePtr = SI->getOperand(1);
                    if (Info->DataMap.find(BasePtr) == Info->DataMap.end()) {
                      errs() << "new host entry ";
                      BasePtr->dump();
                      DataEntry *data_entry = new DataEntry(BasePtr, 0, CI->getArgOperand(0)); // host space
                      data_entry->ptr_type = SI->getOperand(0)->getType();
                      data_entry->alias_ptrs.push_back(CI);
                      errs() << "  alias entry ";
                      CI->dump();
                      Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                    } else
                      errs() << "Error: redundant allocation?\n";
                  } else
                    DEBUG_PRINT
                }
              }
            }
          }
        }
      }
      return false;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;

    DataInfo &getInfo() {
      return *Info;
    }
  };

  void UVMTransformPass::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
  }
}

char UVMTransformPass::ID = 0;

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
static void registerUVMTransformPassPass(const PassManagerBuilder &,
                         legacy::PassManagerBase &PM) {
  PM.add(new UVMTransformPass());
}
static RegisterStandardPasses
  RegisterMyPass(PassManagerBuilder::EP_EnabledOnOptLevel0,
  //RegisterMyPass(PassManagerBuilder::EP_ModuleOptimizerEarly,
                 registerUVMTransformPassPass);
