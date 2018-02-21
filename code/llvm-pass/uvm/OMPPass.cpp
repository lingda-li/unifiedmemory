#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/ADT/SmallSet.h"
#include "OMPPass.h"
using namespace llvm;

char OMPPass::ID = 0;

bool OMPPass::runOnModule(Module &M) {
  errs() << "  ---- OMP (" << M.getName() << ", " << M.getTargetTriple() << ") ----\n";
  if (!analyzeGPUAlloc(M)) {
    errs() << "Error: GPU access anlaysis fails\n";
    return false;
  }
  if (!analyzePointerPropagation(M)) {
    errs() << "Error: pointer propagation anlaysis fails\n";
    return false;
  }
  return optimizeDataMapping(M);
}

void OMPPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<BlockFrequencyInfoWrapperPass>();
  AU.addRequired<FuncArgAccessCGInfoPass>();
  AU.setPreservesCFG();
}

bool OMPPass::analyzeGPUAlloc(Module &M) {
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
            return false;
          } else if (Callee && Callee->getName() == "malloc") {
            errs() << "Info: ignore malloc\n";
            continue;
          } else if (Callee && Callee->getName() == "cudaMallocManaged") {
            auto *AllocPtr = CI->getArgOperand(0);
            if (auto *BCI = dyn_cast<BitCastInst>(AllocPtr)) {
              Value *BasePtr = BCI->getOperand(0);
              if (auto *AI = dyn_cast<AllocaInst>(BasePtr)) {
                Value *BasePtr = AI;
                if (MAI.getBaseAliasEntry(BasePtr) == NULL) {
                  errs() << "new entry ";
                  BasePtr->dump();
                  DataEntry data_entry(BasePtr, 2, CI->getArgOperand(1));
                  data_entry.alloc = CI;
                  FuncInfoEntry *FIE = new FuncInfoEntry(&F);
                  data_entry.func_map.insert(std::make_pair(&F, FIE));
                  data_entry.insertFuncInfoEntry(FIE);
                  MAI.newEntry(data_entry);
                } else
                  errs() << "Error: redundant allocation?\n";
              } else
                DEBUG_PRINT
            } else if (auto *AI = dyn_cast<AllocaInst>(AllocPtr)) {
              Value *BasePtr = AI;
              if (MAI.getBaseAliasEntry(BasePtr) == NULL) {
                errs() << "new entry ";
                BasePtr->dump();
                DataEntry data_entry(BasePtr, 2, CI->getArgOperand(1)); // managed space
                data_entry.alloc = CI;
                MAI.newEntry(data_entry);
              } else
                errs() << "Error: redundant allocation?\n";
            } else
              DEBUG_PRINT
          }
        }
      }
    }
  }
  return true;
}

bool OMPPass::analyzePointerPropagation(Module &M) {
  // Find all pointers to allocated space
  // Iterate until no new functions that are discovered
  SmallVector<Function*, 8> Funcs;
  SmallVector<FuncInfoEntry*, 4> ArgsByVal;
  SmallVector<FuncInfoEntry*, 4> ArgsByRef;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    Funcs.push_back(&F);
  }

  unsigned NumRounds = 0;
  SmallVector<Function*, 4> AddedFuncs;
  while (!Funcs.empty()) {
    errs() << "Round " << NumRounds << "\n";
    for (Function *FP : Funcs) {
      auto &F = *FP;
      unsigned NumNewAdded = 0;
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *LI = dyn_cast<LoadInst>(&I)) {
            assert(LI->getNumOperands() >= 1);
            Value *LoadAddr = LI->getOperand(0);
            if (DataEntry *InsertEntry = MAI.getBaseAliasEntry(LoadAddr)) {
              if(MAI.tryInsertAliasEntry(InsertEntry, LI)) {
                errs() << "  alias entry ";
                LI->dump();
                NumNewAdded++;
              }
            }
          } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
            Value *StoreContent = SI->getOperand(0);
            Value *StoreAddr = SI->getOperand(1);
            if (DataEntry *InsertEntry = MAI.getAliasEntry(StoreContent)) {
              if(MAI.tryInsertBaseAliasEntry(InsertEntry, StoreAddr)) {
                errs() << "  base alias entry ";
                StoreAddr->dump();
                NumNewAdded++;
              }
            }
            if (DataEntry *InsertEntry = MAI.getBaseAliasEntry(StoreAddr)) {
              DataEntry *InsertEntry2 = MAI.getAliasEntry(StoreContent);
              if (InsertEntry != InsertEntry2) {
                errs() << "Warning: store a different alias pointer to a base pointer\n";
                return false;
              }
            }
          } else if (auto *BCI = dyn_cast<BitCastInst>(&I)) {
            Value *CastSource = BCI->getOperand(0);
            unsigned NumAlias = 0;
            if (auto *SourceEntry = MAI.getAliasEntry(CastSource)) {
              if (MAI.tryInsertAliasEntry(SourceEntry, BCI)) {
                errs() << "  alias entry ";
                BCI->dump();
                NumNewAdded++;
                NumAlias++;
              }
            }
            if (auto *SourceEntry = MAI.getBaseAliasEntry(CastSource)) {
              if (MAI.tryInsertBaseAliasEntry(SourceEntry, BCI)) {
                errs() << "  base alias entry ";
                BCI->dump();
                NumNewAdded++;
                NumAlias++;
              }
            }
            if (auto *SourceEntry = MAI.getAliasEntry(BCI)) {
              if (MAI.tryInsertAliasEntry(SourceEntry, CastSource)) {
                errs() << "  alias entry ";
                CastSource->dump();
                NumNewAdded++;
                NumAlias++;
              }
            }
            if (auto *SourceEntry = MAI.getBaseAliasEntry(BCI)) {
              if (MAI.tryInsertBaseAliasEntry(SourceEntry, CastSource)) {
                errs() << "  base alias entry ";
                CastSource->dump();
                NumNewAdded++;
                NumAlias++;
              }
            }
            if (NumAlias > 1) {
              errs() << "Error: a value is alias for multiple entries\n";
              I.dump();
              return false;
            }
          } else if (auto *GEPI = dyn_cast<GetElementPtrInst>(&I)) {
            Value *BasePtr = GEPI->getOperand(0);
            unsigned NumAlias = 0;
            if (auto *SourceEntry = MAI.getAliasEntry(BasePtr)) {
              if (MAI.tryInsertAliasEntry(SourceEntry, GEPI)) {
                errs() << "  alias entry ";
                GEPI->dump();
                NumNewAdded++;
                NumAlias++;
              }
            }
            if (auto *SourceEntry = MAI.getBaseAliasEntry(BasePtr)) {
              if (MAI.tryInsertBaseAliasEntry(SourceEntry, GEPI)) {
                errs() << "  base alias entry ";
                GEPI->dump();
                NumNewAdded++;
                NumAlias++;
              }
            }
            if (auto *SourceEntry = MAI.getAliasEntry(GEPI)) {
              if (MAI.tryInsertAliasEntry(SourceEntry, BasePtr)) {
                errs() << "  alias entry ";
                BasePtr->dump();
                NumNewAdded++;
                NumAlias++;
              }
            }
            if (auto *SourceEntry = MAI.getBaseAliasEntry(GEPI)) {
              if (MAI.tryInsertBaseAliasEntry(SourceEntry, BasePtr)) {
                errs() << "  base alias entry ";
                BasePtr->dump();
                NumNewAdded++;
                NumAlias++;
              }
            }
            if (NumAlias > 1) {
              errs() << "Error: a value is alias for multiple entries\n";
              I.dump();
              return false;
            }
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
              if (auto *E = MAI.getAliasEntry(OPD)) {
                SourceEntry = E;
                AliasTy += 1;
              }
              if (auto *E = MAI.getBaseAliasEntry(OPD)) {
                SourceEntry = E;
                AliasTy += 2;
              }
              assert(AliasTy < 3);
              if (AliasTy > 0) {
                if (Callee && Callee->isDeclaration()) {
                  if (Callee->getName() != "cudaSetupArgument")
                    errs() << "Warning: reach to function declaration " << Callee->getName() << "\n";
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
                  if (MAI.tryInsertAliasEntry(SourceEntry, &(*A))) {
                    assert(SourceEntry->func_map.find(Callee) == SourceEntry->func_map.end());
                    FuncInfoEntry *FIE = new FuncInfoEntry(Callee, CI, &(*A), OPD, 1);
                    SourceEntry->insertFuncInfoEntry(FIE);
                    assert(SourceEntry->func_map.find(&F) != SourceEntry->func_map.end());
                    FIE->setParent(SourceEntry->func_map.find(&F)->second);
                    ArgsByVal.push_back(FIE);
                    errs() << "  alias entry (func arg) ";
                    A->dump();
                  }
                } else {
                  if (MAI.tryInsertBaseAliasEntry(SourceEntry, &(*A))) {
                    assert(SourceEntry->func_map.find(Callee) == SourceEntry->func_map.end());
                    FuncInfoEntry *FIE = new FuncInfoEntry(Callee, CI, &(*A), OPD, 2);
                    SourceEntry->insertFuncInfoEntry(FIE);
                    assert(SourceEntry->func_map.find(&F) != SourceEntry->func_map.end());
                    FIE->setParent(SourceEntry->func_map.find(&F)->second);
                    ArgsByRef.push_back(FIE);
                    errs() << "  base alias entry (func arg) ";
                    A->dump();
                    errs() << "Warning: cannot address base alias pointer as an argument yet";
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
      if (NumNewAdded)
        AddedFuncs.push_back(FP);
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
  return true;
}

bool OMPPass::optimizeDataMapping(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    errs() << "  func: " << F.getName() << "\n";
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          auto *Callee = CI->getCalledFunction();
          if (Callee && Callee->getName() == "__tgt_target_teams") {
            errs() << "  target call: ";
            CI->dump();
            if (auto *CE = dyn_cast<ConstantExpr>(CI->getArgOperand(6))) {
              if (auto *GV = dyn_cast<GlobalVariable>(CE->getOperand(0))) {
                GV->dump();
                if (auto *C = dyn_cast<ConstantDataArray>(GV->getOperand(0))) {
                  SmallVector<uint64_t, 16> MapTypes;
                  bool LocalChanged = false;
                  for (unsigned i = 0; i < C->getNumElements(); i++) {
                    auto *ConstantI = dyn_cast<ConstantInt>(C->getElementAsConstant(i));
                    assert(ConstantI && "Suppose to get constant integer");
                    int64_t V = ConstantI->getSExtValue();
                    if ((V & 0x01 || V & 0x02) && !(V & 0x400)) {
                      V |= 0x400;
                      LocalChanged = true;
                    }
                    MapTypes.push_back(V);
                  }
                  if (LocalChanged) {
                    auto *NCDA = ConstantDataArray::get(C->getContext(), MapTypes);
                    C->replaceAllUsesWith(NCDA);
                    errs() << "    map type changed: ";
                    GV->dump();
                    Changed = true;
                  }
                }
              }
            }
          } else if (Callee && Callee->getName() == "__tgt_target_data_begin") {
            errs() << "  target data call: ";
            CI->dump();
            if (auto *CE = dyn_cast<ConstantExpr>(CI->getArgOperand(5))) {
              if (auto *GV = dyn_cast<GlobalVariable>(CE->getOperand(0))) {
                GV->dump();
                if (auto *C = dyn_cast<ConstantDataArray>(GV->getOperand(0))) {
                  SmallVector<uint64_t, 16> MapTypes;
                  bool LocalChanged = false;
                  for (unsigned i = 0; i < C->getNumElements(); i++) {
                    auto *ConstantI = dyn_cast<ConstantInt>(C->getElementAsConstant(i));
                    assert(ConstantI && "Suppose to get constant integer");
                    int64_t V = ConstantI->getSExtValue();
                    if ((V & 0x01 || V & 0x02) && !(V & 0x400)) {
                      V |= 0x400;
                      LocalChanged = true;
                    }
                    MapTypes.push_back(V);
                  }
                  if (LocalChanged) {
                    auto *NCDA = ConstantDataArray::get(C->getContext(), MapTypes);
                    C->replaceAllUsesWith(NCDA);
                    errs() << "    map type changed: ";
                    GV->dump();
                    Changed = true;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return Changed;
}

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
static void registerOMPPass(const PassManagerBuilder &,
                         legacy::PassManagerBase &PM) {
  PM.add(new BlockFrequencyInfoWrapperPass());
  PM.add(new CallGraphWrapperPass());
  PM.add(new FuncArgAccessCGInfoPass());
  //PM.add(new OMPPass());
}
static RegisterStandardPasses
  //RegisterMyPass(PassManagerBuilder::EP_EnabledOnOptLevel0,
  RegisterMyPass(PassManagerBuilder::EP_OptimizerLast,
  //RegisterMyPass(PassManagerBuilder::EP_ModuleOptimizerEarly,
                 registerOMPPass);
