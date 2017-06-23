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
    bool Succeeded = true;

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
                Succeeded = false;
              } else if (Callee && Callee->getName() == "malloc") {
                errs() << "Error: this is for UVM only\n";
                Succeeded = false;
              } else if (Callee && Callee->getName() == "cudaMallocManaged") {
                auto *AllocPtr = CI->getArgOperand(0);
                if (auto *BCI = dyn_cast<BitCastInst>(AllocPtr)) {
                  Value *BasePtr = BCI->getOperand(0);
                  if (auto *AI = dyn_cast<AllocaInst>(BasePtr)) {
                    Value *BasePtr = AI;
                    if (Info->DataMap.find(BasePtr) == Info->DataMap.end()) {
                      errs() << "new entry ";
                      BasePtr->dump();
                      DataEntry *data_entry = new DataEntry(BasePtr, 2, CI->getArgOperand(1)); // managed space
                      Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                    } else
                      errs() << "Error: redundant allocation?\n";
                  } else
                    DEBUG_PRINT
                } else if (auto *AI = dyn_cast<AllocaInst>(AllocPtr)) {
                  Value *BasePtr = AI;
                  if (Info->DataMap.find(BasePtr) == Info->DataMap.end()) {
                    errs() << "new entry ";
                    BasePtr->dump();
                    DataEntry *data_entry = new DataEntry(BasePtr, 2, CI->getArgOperand(1)); // managed space
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

      // Find all pointers to allocated space
      // Iterate until no new functions that are discovered
      SmallVector<Function*, 8> Funcs;
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;
        Funcs.push_back(&F);
      }

      unsigned NumRounds = 0;
      while (!Funcs.empty()) {
        SmallVector<Function*, 4> AddedFuncs;
        for (Function *FP : Funcs) {
          auto &F = *FP;
          for (auto &BB : F) {
            for (auto &I : BB) {
              if (auto *LI = dyn_cast<LoadInst>(&I)) {
                assert(LI->getNumOperands() >= 1);
                Value *LoadAddr = LI->getOperand(0);
                if (DataEntry *InsertEntry = Info->getBaseAliasEntry(LoadAddr)) {
                  if(Info->tryInsertAliasEntry(InsertEntry, LI)) {
                    errs() << "  alias entry ";
                    LI->dump();
                  }
                }
              } else if (auto *BCI = dyn_cast<BitCastInst>(&I)) {
                Value *CastSource = BCI->getOperand(0);
                unsigned NumAlias = 0;
                if (auto *SourceEntry = Info->getAliasEntry(CastSource)) {
                  if (Info->tryInsertAliasEntry(SourceEntry, BCI)) {
                    errs() << "  alias entry ";
                    BCI->dump();
                    NumAlias++;
                  }
                }
                if (auto *SourceEntry = Info->getBaseAliasEntry(CastSource)) {
                  if (Info->tryInsertBaseAliasEntry(SourceEntry, BCI)) {
                    errs() << "  base alias entry ";
                    BCI->dump();
                    NumAlias++;
                  }
                }
                if (Info->DataMap.find(CastSource) != Info->DataMap.end()) {
                  auto *SourceEntry = Info->DataMap.find(CastSource)->second;
                  if (Info->tryInsertBaseAliasEntry(SourceEntry, BCI)) {
                    errs() << "  base alias entry ";
                    BCI->dump();
                    NumAlias++;
                  }
                }
                if (NumAlias > 1)
                  errs() << "Error: a value is alias for multiple entries\n";
              } else if (auto *CI = dyn_cast<CallInst>(&I)) {
                auto *Callee = CI->getCalledFunction();
                if (Callee && Callee->getName() == "cudaMallocManaged")
                  continue;
                else if (Callee && Callee->getName() == "cudaFree")
                  continue;
                bool Use = false;
                for (int i = 0; i < I.getNumOperands(); i++) {
                  Value *OPD = I.getOperand(i);
                  if (Info->getAliasEntry(OPD) || Info->getBaseAliasEntry(OPD)) {
                    Use = true;
                    int argcount = 0;
                    Function::ArgumentListType::iterator A;
                    for (A = Callee->getArgumentList().begin(); A != Callee->getArgumentList().end(); A++) {
                      if (argcount == i)
                        break;
                      argcount++;
                    }
                    assert(argcount == i);
                  }
                }
                if (Use) {
                  errs() << "Info: add function " << Callee->getName() << " in Round " << NumRounds << "\n";
                  AddedFuncs.push_back(Callee);
                }
              }
            }
          }
        }

        while (!Funcs.empty())
          Funcs.pop_back();
        for (Function *FP : AddedFuncs)
          Funcs.push_back(FP);
        NumRounds++;
      }

      if (!Succeeded)
        return false;

      // Find all usage of allocated space
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;
        for (auto &BB : F) {
          for (auto &I : BB) {
            bool Use = false;
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              auto *Callee = CI->getCalledFunction();
              if (Callee && Callee->getName() == "cudaMallocManaged")
                continue;
              else if (Callee && Callee->getName() == "cudaFree")
                continue;
              for (int i = 0; i < I.getNumOperands(); i++) {
                Value *OPD = I.getOperand(i);
                if (Info->getAliasEntry(OPD) || Info->getBaseAliasEntry(OPD)) {
                  errs() << "Mark\n";
                  int argcount = 0;
                  Function::ArgumentListType::iterator A;
                  for (A = Callee->getArgumentList().begin(); A != Callee->getArgumentList().end(); A++) {
                    if (argcount == i)
                      break;
                    argcount++;
                  }
                  assert(argcount == i);
                }
              }
            }
            for (int i = 0; i < I.getNumOperands(); i++) {
              Value *OPD = I.getOperand(i);
              if (Info->getAliasEntry(OPD))
                Use = true;
              if (Info->getBaseAliasEntry(OPD))
                Use = true;
            }
            if (Use) {
              I.dump();
              // Go into callee function
              if (auto *CI = dyn_cast<CallInst>(&I)) {
                Function *Callee = CI->getCalledFunction();
                Callee->dump();
                //for (auto A : Callee->getArgumentList())
                auto A = Callee->getArgumentList().begin();
                //for (auto A : Callee->getArgumentList())
                A->dump();
              }
            }
          }
        }
      }

      bool Changed = false;
      return Changed;
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
