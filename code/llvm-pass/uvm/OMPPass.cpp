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
#include <fstream>
#include <sstream>
#include <algorithm>
#include "llvm/Support/Format.h"
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
  calculateAccessFreq(M);
  optimizeDataAllocation(M);
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
        if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
          CallSite CS(&I);
          auto *Callee = CS.getCalledFunction();
          if (Callee && Callee->getName() == "cudaMalloc") {
            errs() << "Error: this is for UVM only\n";
            return false;
          } else if (Callee && Callee->getName() == "malloc") {
            errs() << "Info: ignore malloc\n";
            continue;
          } else if (Callee && Callee->getName() == "cudaMallocManaged") {
            auto *AllocPtr = CS.getArgOperand(0);
            auto *Size = CS.getArgOperand(1);
            if (auto *BCI = dyn_cast<BitCastInst>(AllocPtr)) {
              Value *BasePtr = BCI->getOperand(0);
              if (auto *AI = dyn_cast<AllocaInst>(BasePtr)) {
                Value *BasePtr = AI;
                if (MAI.getBaseAliasEntry(BasePtr) == NULL) {
                  errs() << "new entry ";
                  BasePtr->dump();
                  DataEntry data_entry(BasePtr, 2, Size, &F);
                  data_entry.alloc = &I;
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
                DataEntry data_entry(BasePtr, 2, Size, &F); // managed space
                data_entry.alloc = &I;
                MAI.newEntry(data_entry);
              } else
                errs() << "Error: redundant allocation?\n";
            } else
              DEBUG_PRINT
          } else if (Callee && Callee->getName() == "omp_target_alloc") {
            auto *Size = CS.getArgOperand(0);
            auto *Device = CS.getArgOperand(1);
            if (auto *ConstantI = dyn_cast<ConstantInt>(Device)) {
              int64_t V = ConstantI->getSExtValue();
              if (V == -100 && MAI.getAliasEntry(&I) == NULL) {
                errs() << "new entry ";
                I.dump();
                DataEntry data_entry(NULL, 2, Size, &F); // managed space
                data_entry.alloc = &I;
                data_entry.tryInsertAliasPtr(&I);
                MAI.newEntry(data_entry);
              }
            }
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
            DataEntry *SourceEntry;
            if (SourceEntry = MAI.getAliasEntry(BasePtr)) {
              if (MAI.tryInsertAliasEntry(SourceEntry, GEPI)) {
                errs() << "  alias entry ";
                GEPI->dump();
                NumNewAdded++;
              }
            } else if (SourceEntry = MAI.getAliasEntry(GEPI)) {
              if (MAI.tryInsertAliasEntry(SourceEntry, BasePtr)) {
                errs() << "  alias entry ";
                BasePtr->dump();
                NumNewAdded++;
              }
            } else {
              SourceEntry = MAI.getBaseAliasEntry(GEPI);
              auto EVSet = MAI.getBaseOffsetAliasEntries(BasePtr);
              if (SourceEntry == NULL && EVSet.size() == 0)
                continue;

              auto *ConstantI = dyn_cast<ConstantInt>(GEPI->getOperand(1));
              if (!ConstantI) {
                errs() << "Warning: the first offset is not constant\n";
                continue;
              }
              int64_t V = ConstantI->getSExtValue();
              if (V != 0) {
                errs() << "Warning: the first offset is not 0\n";
                continue;
              }
              ConstantI = dyn_cast<ConstantInt>(GEPI->getOperand(2));
              if (!ConstantI)
                continue;
              V = ConstantI->getSExtValue();
              if (SourceEntry) {
                if (MAI.tryInsertBaseOffsetAliasEntry(SourceEntry, BasePtr, V)) {
                  errs() << "  base alias offset entry (" << V << ") ";
                  BasePtr->dump();
                  NumNewAdded++;
                }
              }
              for (auto EV : EVSet) {
                SourceEntry = EV.first;
                int64_t Diff = EV.second - V;
                if (MAI.tryInsertBaseOffsetAliasEntry(SourceEntry, GEPI, Diff)) {
                  errs() << "  base alias offset entry (" << Diff << ") ";
                  GEPI->dump();
                  NumNewAdded++;
                }
              }
            }
          } else if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
            CallSite CS(&I);
            auto *Callee = CS.getCalledFunction();
            if (Callee == NULL)
              continue;
            else if (Callee->isIntrinsic())
              continue;
            else if (Callee->getName() == "cudaMallocManaged")
              continue;
            else if (Callee->getName() == "cudaFree")
              continue;
            bool Use = false;
            for (int i = 0; i < I.getNumOperands(); i++) {
              Value *OPD = I.getOperand(i);
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
                    FuncInfoEntry *FIE = new FuncInfoEntry(Callee, &I, &(*A), OPD, 1);
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
                    FuncInfoEntry *FIE = new FuncInfoEntry(Callee, &I, &(*A), OPD, 2);
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

void OMPPass::calculateAccessFreq(Module &M) {
  errs() << "  ---- Access Frequency Analysis ----\n";
  SmallVector<Function*, 8> VisitedFuncs;
  MemInfo<FuncArgEntry> *FAI = &getAnalysis<FuncArgAccessCGInfoPass>().getFAI();

  std::string TT = M.getTargetTriple();
  if (TT.find("cuda") == std::string::npos) {
    std::ifstream CudaFuncFile("cuda_func_arg.lst");
    if (!CudaFuncFile.is_open()) {
      errs() << "Warning: No cuda file found\n";
    } else {
      std::string Line, Name;
      int ArgNum;
      double LoadFreq, StoreFreq;
      while (getline(CudaFuncFile, Line)) {
        std::stringstream ss(Line);
        ss >> Name >> ArgNum >> LoadFreq >> StoreFreq;
        FuncArgEntry E(NULL, NULL, ArgNum, Name);
        E.addTgtLoadFreq(LoadFreq);
        E.addTgtStoreFreq(StoreFreq);
        TFAI.newEntry(E);
      }
    }
  }

  for (auto &E : *MAI.getEntries()) {
    Function *F = E.getFunc();
    SmallVector<Function*, 8>::const_iterator I = VisitedFuncs.begin();
    for (; I != VisitedFuncs.end(); I++)
      if (*I == F)
        break;
    if (I != VisitedFuncs.end()) // Have traversed this function
      continue;
    assert(!F->isDeclaration());
    VisitedFuncs.push_back(F);

    BlockFrequencyInfo *BFI = &getAnalysis<BlockFrequencyInfoWrapperPass>(*F).getBFI();
    for (auto &BB : *F) {
      double Freq = (double)BFI->getBlockFreq(&BB).getFrequency() / (double)BFI->getEntryFreq();
      for (auto &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          Value *LoadAddr = LI->getOperand(0);
          DataEntry *E = MAI.getAliasEntry(LoadAddr);
          if (E && E->getFunc() == F) {
            errs() << "  load (" << Freq << ") from ";
            E->dumpBase();
            E->load_freq += Freq;
          }
        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          Value *StoreAddr = SI->getOperand(1);
          DataEntry *E = MAI.getAliasEntry(StoreAddr);
          if (E && E->getFunc() == F) {
            errs() << "  store (" << Freq << ") to ";
            E->dumpBase();
            E->store_freq += Freq;
          }
        } else if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
          CallSite CS(&I);
          auto *Callee = CS.getCalledFunction();
          // OpenMP target calls
          if (Callee && Callee->getName().find("__tgt_target") == 0 &&
              Callee->getName().find("__tgt_target_data") == std::string::npos) {
            auto *MapTypeCE = dyn_cast<ConstantExpr>(CS.getArgOperand(6));
            assert(MapTypeCE);
            auto *GV = dyn_cast<GlobalVariable>(MapTypeCE->getOperand(0));
            assert(GV);
            auto *C = dyn_cast<ConstantDataArray>(GV->getOperand(0));
            assert(C);
            unsigned NumArgs = C->getNumElements();
            std::string FuncName = CS.getArgOperand(1)->getName();
            auto *Args = CS.getArgOperand(4);
            for (unsigned i = 0; i < NumArgs; i++) {
              auto *E = MAI.getBaseOffsetAliasEntries(Args, i);
              FuncArgEntry *TFAE = TFAI.getFuncArgEntry(FuncName, i);
              if (E && TFAE) {
                errs() << "  target call (" << Freq << ", " << TFAE->getTgtLoadFreq()
                       << ", " << TFAE->getTgtStoreFreq() << ") using ";
                E->dumpBase();
                E->addTgtLoadFreq(Freq * TFAE->getTgtLoadFreq());
                E->addTgtStoreFreq(Freq * TFAE->getTgtStoreFreq());
              }
            }
          } else if (Callee) { // Other calls
            for (int i = 0; i < I.getNumOperands(); i++) {
              Value *OPD = I.getOperand(i);
              unsigned AliasTy = 0;
              DataEntry *E = MAI.getAliasEntry(OPD);
              if (E && E->getFunc() == F) {
                assert(MAI.getBaseAliasEntry(OPD) == NULL);
                FuncArgEntry *FAE = FAI->getFuncArgEntry(Callee, i);
                if (FAE && FAE->getValid()) {
                  errs() << "  call (" << Freq << ", " << FAE->getLoadFreq()
                         << ", " << FAE->getStoreFreq() << ", " << FAE->getTgtLoadFreq() 
                         << ", " << FAE->getTgtStoreFreq() << ") using ";
                  E->dumpBase();
                  E->load_freq += Freq * FAE->getLoadFreq();
                  E->store_freq += Freq * FAE->getStoreFreq();
                  E->addTgtLoadFreq(Freq * FAE->getTgtLoadFreq());
                  E->addTgtStoreFreq(Freq * FAE->getTgtStoreFreq());
                } else if (!Callee->isDeclaration()) // Could reach declaration here
                  errs() << "Warning: wrong traversal order, or recursive call\n";
              }
            }
          }
        }
      }
    }
  }

  for (auto &E : *MAI.getEntries()) {
    errs() << "Frequency of ";
    E.dumpBase();
    errs() << "  load: " << E.getLoadFreq() << "\t\t";
    errs() << "  store: " << E.getStoreFreq() << " (host)\n";
    errs() << "  load: " << E.getTgtLoadFreq() << "\t\t";
    errs() << "  store: " << E.getTgtStoreFreq() << " (target)\n";
  }
}

bool OMPPass::optimizeDataAllocation(Module &M) {
  bool Changed = false;
  errs() << "  ---- Data Allocation Optimization ----\n";

  // Rank by frequency
  std::sort(MAI.getEntries()->begin(), MAI.getEntries()->end(), compareAccessFreq);
  size_t Rank = MAI.getEntries()->size();
  for (auto &E : *MAI.getEntries()) {
    // Address rank overflow
    if (Rank < 0xff)
      E.setRank(Rank);
    else
      E.setRank(0xff);
    Rank--;
    errs() << "Rank " << E.getRank() << " for ";
    E.dumpBase();
    errs() << "  load: " << E.getLoadFreq() << "\t\t";
    errs() << "  store: " << E.getStoreFreq() << " (host)\n";
    errs() << "  load: " << E.getTgtLoadFreq() << "\t\t";
    errs() << "  store: " << E.getTgtStoreFreq() << " (target)\n";
  }
  return Changed;
}

bool OMPPass::optimizeDataMapping(Module &M) {
  bool Changed = false;
  errs() << "  ---- Data Mapping Optimization ----\n";
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
          CallSite CS(&I);
          auto *Callee = CS.getCalledFunction();
          if (Callee == NULL)
            continue;
          if (Callee->getName().find("__tgt_target") != 0)
            continue;
          ConstantExpr *CE;
          Value *Args;
          bool IsDataRegion;
          std::string FuncName;
          if (Callee->getName().find("__tgt_target_data") != std::string::npos) {
            errs() << "  target data call: ";
            Args = CS.getArgOperand(3);
            CE = dyn_cast<ConstantExpr>(CS.getArgOperand(5));
            IsDataRegion = true;
          } else {
            errs() << "  target call: ";
            Args = CS.getArgOperand(4);
            CE = dyn_cast<ConstantExpr>(CS.getArgOperand(6));
            IsDataRegion = false;
            FuncName = CS.getArgOperand(1)->getName();
            auto P = FuncName.find(".region_id");
            FuncName = FuncName.substr(0, P);
          }
          assert(Args && CE);
          I.dump();

          if (auto *GV = dyn_cast<GlobalVariable>(CE->getOperand(0))) {
            GV->dump();
            if (auto *C = dyn_cast<ConstantDataArray>(GV->getOperand(0))) {
              SmallVector<uint64_t, 16> MapTypes;
              bool LocalChanged = false;
              // iterate through all arguments
              for (unsigned i = 0; i < C->getNumElements(); i++) {
                auto *ConstantI = dyn_cast<ConstantInt>(C->getElementAsConstant(i));
                assert(ConstantI && "Suppose to get constant integer");
                int64_t V = ConstantI->getSExtValue();
                if (auto *Entry = MAI.getBaseOffsetAliasEntries(Args, i)) {
                  errs() << "  arg " << i << " (" << Entry->getLoadFreq() << ", " << Entry->getStoreFreq()
                         << "; " << Entry->getTgtLoadFreq() << ", " << Entry->getTgtStoreFreq() << ") is ";
                  Entry->dumpBase();
                  errs() << "    size is ";
                  Entry->size->dump();
                  double LocalReuse = 0.0;
                  //if ((V & 0x01 || V & 0x02) && !(V & 0x400)) {
                  //  V |= 0x400;
                  //  LocalChanged = true;
                  //}
                  // set global reuse mark
                  if (!(V & 0xff000)) {
                    V |= (Entry->getRank() << 12) & 0xff000;
                    LocalChanged = true;
                  }
                  // set local reuse mark
                  if (!IsDataRegion) {
                    FuncArgEntry *TFAE = TFAI.getFuncArgEntry(FuncName, i);
                    if (TFAE) {
                      LocalReuse = TFAE->getTgtLoadFreq() + TFAE->getTgtStoreFreq();
                      errs() << "    local reuse is " << LocalReuse << ";\t\t";
                      uint32_t LocalReuseScale;
                      LocalReuse *= 0xf;
                      if (LocalReuse > 0xff)
                        LocalReuseScale = 0xff;
                      else
                        LocalReuseScale = LocalReuse;
                      errs() << "    scaled local reuse is " << format_hex(LocalReuseScale, 4) << "\n";
                      if (!(V & 0xff00000)) {
                        V |= LocalReuseScale << 20;
                        LocalChanged = true;
                      }
                    }
                  }
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

  return Changed;
}

bool compareAccessFreq(DataEntry A, DataEntry B) {
  double AFreq = A.getTgtLoadFreq() + A.getTgtStoreFreq();
  double BFreq = B.getTgtLoadFreq() + B.getTgtStoreFreq();
  if (AFreq < BFreq)
    return true;
  else
    return false;
}

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
static void registerOMPPass(const PassManagerBuilder &,
                         legacy::PassManagerBase &PM) {
  PM.add(new BlockFrequencyInfoWrapperPass());
  //PM.add(new CallGraphWrapperPass());
  PM.add(new FuncArgAccessCGInfoPass());
  PM.add(new OMPPass());
}
static RegisterStandardPasses
  //RegisterMyPass(PassManagerBuilder::EP_EnabledOnOptLevel0,
  RegisterMyPass(PassManagerBuilder::EP_OptimizerLast,
  //RegisterMyPass(PassManagerBuilder::EP_ModuleOptimizerEarly,
                 registerOMPPass);
