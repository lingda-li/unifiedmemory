#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "MemAnalysisDataStructure.h"
#include "CGMemAnalysis.h"
#include "MemAccessInfoPass.h"
using namespace llvm;

char MemAccessInfoPass::ID = 0;

bool MemAccessInfoPass::runOnModule(Module &M) {
  std::string TT = M.getTargetTriple();
  if (TT.find("cuda") != std::string::npos) {
    errs() << "  ---- GPU Analysis (" << M.getName() << ", " << TT << ") ----\n";
    if (!analyzeGPUAlloc(M)) {
      errs() << "Error: GPU access anlaysis fails\n";
      return false;
    }
    if (!analyzePointerPropagation(M)) {
      errs() << "Error: GPU pointer propagation anlaysis fails\n";
      return false;
    }
    calculateAccessFreq(M);
  } else {
    errs() << "  ---- CPU Analysis (" << M.getName() << ", " << TT << ") ----\n";
    if (!analyzeCPUAlloc(M)) {
      errs() << "Error: CPU access anlaysis fails\n";
      return false;
    }
    if (!analyzePointerPropagation(M)) {
      errs() << "Error: CPU pointer propagation anlaysis fails\n";
      return false;
    }
    calculateAccessFreq(M);
  }
  return false;
}

void MemAccessInfoPass::calculateAccessFreq(Module &M) {
  errs() << "  ---- Access Frequency Analysis ----\n";
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    //errs() << "On function " << F.getName() << "\n";
    BlockFrequencyInfo *BFI = &getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
    MemInfo<FuncArgEntry> *FAI = &getAnalysis<FuncArgAccessCGInfoPass>().getFAI();
    for (auto &BB : F) {
      double Freq = (double)BFI->getBlockFreq(&BB).getFrequency() / (double)BFI->getEntryFreq();
      //errs() << Freq << "\n";
      for (auto &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          Value *LoadAddr = LI->getOperand(0);
          if (DataEntry *E = MAI.getAliasEntry(LoadAddr)) {
            errs() << "  load (" << Freq << ") from ";
            E->dumpBase();
            E->load_freq += Freq;
          }
        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          Value *StoreAddr = SI->getOperand(1);
          if (DataEntry *E = MAI.getAliasEntry(StoreAddr)) {
            errs() << "  store (" << Freq << ") to ";
            E->dumpBase();
            E->store_freq += Freq;
          }
        } else if (auto *CI = dyn_cast<CallInst>(&I)) {
          for (int i = 0; i < I.getNumOperands(); i++) {
            Value *OPD = CI->getOperand(i);
            unsigned AliasTy = 0;
            DataEntry *E = MAI.getAliasEntry(OPD);
            if (E) {
              assert(MAI.getBaseAliasEntry(OPD) == NULL);
              FuncArgEntry *FAE = FAI->getFuncArgEntry(CI->getCalledFunction(), i);
              if (FAE && FAE->getValid()) {
                errs() << "  call (" << Freq << ", " << FAE->getLoadFreq()
                       << ", " << FAE->getStoreFreq() << ") using ";
                E->dumpBase();
                E->load_freq += Freq * FAE->getLoadFreq();
                E->store_freq += Freq * FAE->getStoreFreq();
              } else if (!CI->getCalledFunction()->isDeclaration()) // Could reach declaration here
                errs() << "Warning: wrong traversal order, or recursive call\n";
            }
          }
        }
      }
    }
  }

  for (auto &E : *MAI.getEntries()) {
    errs() << "Frequency of ";
    E.dumpBase();
    errs() << "  load is " << E.load_freq << "\n";
    errs() << "  store is " << E.store_freq << "\n";
  }
}

bool MemAccessInfoPass::analyzeGPUAlloc(Module &M) {
  errs() << "  ---- GPU Memory Analysis ----\n";
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    // Add an entry for every pointer argument
    for (auto &A : F.args()) {
      auto *PT = dyn_cast<PointerType>(A.getType());
      if (!PT)
        continue;
      A.dump();
      if (MAI.getAliasEntry(&A)) {
        continue;
      }
      if (MAI.getBaseAliasEntry(&A)) {
        errs() << "Warning: pass a base pointer to GPU?\n";
        return false;
      }
      DataEntry E;
      bool Succeed = MAI.tryInsertAliasEntry(&E, &A);
      assert(Succeed);
      MAI.newEntry(E);
    }
  }
  return true;
}

bool MemAccessInfoPass::analyzeCPUAlloc(Module &M) {
  errs() << "  ---- CPU Memory Analysis ----\n";
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
            errs() << "Error: this is for UVM only\n";
            return false;
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

bool MemAccessInfoPass::analyzePointerPropagation(Module &M) {
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
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *LI = dyn_cast<LoadInst>(&I)) {
            assert(LI->getNumOperands() >= 1);
            Value *LoadAddr = LI->getOperand(0);
            if (DataEntry *InsertEntry = MAI.getBaseAliasEntry(LoadAddr)) {
              if(MAI.tryInsertAliasEntry(InsertEntry, LI)) {
                errs() << "  alias entry ";
                LI->dump();
              }
            }
          } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
            Value *StoreContent = SI->getOperand(0);
            Value *StoreAddr = SI->getOperand(1);
            if (DataEntry *InsertEntry = MAI.getAliasEntry(StoreContent)) {
              if(MAI.tryInsertBaseAliasEntry(InsertEntry, StoreAddr)) {
                errs() << "  base alias entry ";
                StoreAddr->dump();
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
                NumAlias++;
              }
            }
            if (auto *SourceEntry = MAI.getBaseAliasEntry(CastSource)) {
              if (MAI.tryInsertBaseAliasEntry(SourceEntry, BCI)) {
                errs() << "  base alias entry ";
                BCI->dump();
                NumAlias++;
              }
            }
            if (NumAlias > 1)
              errs() << "Error: a value is alias for multiple entries\n";
          } else if (auto *GEPI = dyn_cast<GetElementPtrInst>(&I)) {
            Value *BasePtr = GEPI->getOperand(0);
            if (auto *SourceEntry = MAI.getAliasEntry(BasePtr)) {
              if (MAI.tryInsertAliasEntry(SourceEntry, GEPI)) {
                errs() << "  alias entry ";
                GEPI->dump();
              }
            }
            if (auto *SourceEntry = MAI.getBaseAliasEntry(BasePtr)) {
              errs() << "Warning: cast to a base pointer\n";
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
                    errs() << "Warning: reach to function declaration " << Callee->getName();
                  break;
                }
                Use = true;
                int argcount = 0;
                Function::arg_iterator A;
                for (A = Callee->arg_begin(); A != Callee->arg_end(); A++) {
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
