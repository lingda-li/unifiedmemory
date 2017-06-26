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
                    if (Info->getBaseAliasEntry(BasePtr) == NULL) {
                      errs() << "new entry ";
                      BasePtr->dump();
                      DataEntry *data_entry = new DataEntry(BasePtr, 2, CI->getArgOperand(1)); // managed space
                      data_entry->alloc = CI;
                      Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                    } else
                      errs() << "Error: redundant allocation?\n";
                  } else
                    DEBUG_PRINT
                } else if (auto *AI = dyn_cast<AllocaInst>(AllocPtr)) {
                  Value *BasePtr = AI;
                  if (Info->getBaseAliasEntry(BasePtr) == NULL) {
                    errs() << "new entry ";
                    BasePtr->dump();
                    DataEntry *data_entry = new DataEntry(BasePtr, 2, CI->getArgOperand(1)); // managed space
                    data_entry->alloc = CI;
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
        errs() << "Round " << NumRounds << "\n";
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
              } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
                Value *StoreContent = SI->getOperand(0);
                Value *StoreAddr = SI->getOperand(1);
                if (DataEntry *InsertEntry = Info->getAliasEntry(StoreContent)) {
                  if(Info->tryInsertBaseAliasEntry(InsertEntry, StoreAddr)) {
                    errs() << "  base alias entry ";
                    StoreAddr->dump();
                  }
                }
                if (DataEntry *InsertEntry = Info->getBaseAliasEntry(StoreAddr)) {
                  DataEntry *InsertEntry2 = Info->getAliasEntry(StoreContent);
                  if (InsertEntry != InsertEntry2) {
                    errs() << "Warning: store a different alias pointer to a base pointer\n";
                    Succeeded = false;
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
                if (NumAlias > 1)
                  errs() << "Error: a value is alias for multiple entries\n";
              } else if (auto *CI = dyn_cast<CallInst>(&I)) {
                auto *Callee = CI->getCalledFunction();
                if (Callee && Callee->isIntrinsic())
                  continue;
                else if (Callee && Callee->getName() == "cudaMallocManaged")
                  continue;
                else if (Callee && Callee->getName() == "cudaFree")
                  continue;
                bool Use = false;
                for (int i = 0; i < I.getNumOperands(); i++) {
                  Value *OPD = CI->getOperand(i);
                  unsigned AliasTy = 0;
                  DataEntry *SourceEntry;
                  if (auto *E = Info->getAliasEntry(OPD)) {
                    SourceEntry = E;
                    AliasTy += 1;
                  }
                  if (auto *E = Info->getBaseAliasEntry(OPD)) {
                    SourceEntry = E;
                    AliasTy += 2;
                  }
                  assert(AliasTy < 3);
                  if (AliasTy > 0) {
                    if (Callee && Callee->isDeclaration()) {
                      if (Callee->getName() != "cudaSetupArgument")
                        errs() << "Warning: reach to function declaration " << Callee->getName();
                      break;
                    }
                    Use = true;
                    int argcount = 0;
                    Function::ArgumentListType::iterator A;
                    for (A = Callee->getArgumentList().begin(); A != Callee->getArgumentList().end(); A++) {
                      if (argcount == i)
                        break;
                      argcount++;
                    }
                    assert(argcount == i);
                    if (AliasTy == 1) {
                      if (Info->tryInsertAliasEntry(SourceEntry, &(*A))) {
                        errs() << "  alias entry (func arg) ";
                        A->dump();
                      }
                    } else {
                      if (Info->tryInsertBaseAliasEntry(SourceEntry, &(*A))) {
                        errs() << "  base alias entry (func arg) ";
                        A->dump();
                      }
                    }
                  }
                }
                if (Use) {
                  errs() << "Info: add function " << Callee->getName() << " to Round " << NumRounds+1 << "\n";
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
        while (!AddedFuncs.empty())
          AddedFuncs.pop_back();
        NumRounds++;
      }
      errs() << "Round end\n";

      // Find where data pointer is passed to GPU kernel
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;
        for (auto &BB : F) {
          for (auto &I : BB) {
            bool Use = false;
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              auto *Callee = CI->getCalledFunction();
              if (Callee && Callee->getName() == "cudaSetupArgument") {
                auto FirstArg = CI->getOperand(0);
                if (DataEntry *SourceEntry = Info->getBaseAliasEntry(&(*FirstArg))) {
                  errs() << "Info: data ";
                  SourceEntry->base_ptr->dump();
                  errs() << "  is passed to a kernel through ";
                  FirstArg->dump();
                  SourceEntry->send2kernel.push_back(CI);

                  // Find the corresponding kernel call
                  Instruction *NextInst = CI->getNextNode();
                  auto *CBB = &BB;
                  bool FoundKernel = false;
                  do {
                    if (auto *NCI = dyn_cast<CallInst>(NextInst)) {
                      auto *NextCallee = NCI->getCalledFunction();
                      if (NextCallee && NextCallee->getName() == "cudaLaunch") {
                        errs() << "  kernel call is ";
                        NCI->dump();
                        FoundKernel = true;
                        SourceEntry->kernel.push_back(NCI);
                        break;
                      }
                    }
                    NextInst = NextInst->getNextNode();
                    if (!NextInst) {
                      CBB = CBB->getNextNode(); // FIXME: go to the successor block
                      if (!CBB)
                        break;
                      NextInst = &(*CBB->begin());
                    }
                  } while (NextInst);
                  if (!FoundKernel) {
                    errs() << "Error: didn't find the kernel call\n";
                    Succeeded = false;
                  }
                }
              } else if (Callee && Callee->getName() == "cudaFree") {
                auto *FreePtr = CI->getArgOperand(0);
                DataEntry *data_entry = Info->getAliasEntry(FreePtr);
                assert(data_entry);
                assert(!data_entry->free);
                data_entry->free = CI;
                errs() << "Info: find free of ";
                data_entry->base_ptr->dump();
              }
            }
          }
        }
      }

      if (!Succeeded) {
        errs() << "Info: didn't perform transformation because data analysis fails\n";
        return false;
      }

      // Change memory allocation api calls
      //for (auto & : Info->DataMap)
      // Insert memory copy api calls
      // Change memory free api calls

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
