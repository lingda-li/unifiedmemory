#include <fstream>
#include <string>
#include <sstream>
#include "CGMemAnalysis.h"
using namespace llvm;

char FuncArgAccessCGInfoPass::ID = 0;

bool FuncArgAccessCGInfoPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  errs() << "  ---- Function Argument Access Frequency CG Analysis ----\n";
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
        E.addLoadFreq(LoadFreq);
        E.addStoreFreq(StoreFreq);
        TFAI.newEntry(E);
      }
    }
  }
  //for (auto &E : *TFAI.getEntries()) {
  //  errs() << E.getName() << " ";
  //  errs() << E.getArgNum() << " ";
  //  errs() << E.getLoadFreq() << " ";
  //  errs() << E.getStoreFreq() << "\n";
  //}

  //CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  //for (scc_iterator<CallGraph *> I = scc_begin(&CG); !I.isAtEnd(); ++I) {
  //  if (I->size() != 1)
  //    continue;

  //  Function *F = I->front()->getFunction();
  //  if (F && !F->isDeclaration()/* && F->doesNotRecurse()*/) {
  //    errs() << "On function " << F->getName() << "\n";
  //    computeLocalAccessFreq(*F);
  //  }
  //}
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    errs() << "On function " << F.getName() << "\n";
    computeLocalAccessFreq(F);
  }

  // Output CUDA kernel analysis results
  if (TT.find("cuda") != std::string::npos) {
    std::ofstream Output("cuda_func_arg.lst");
    if (!Output.is_open())
      return false;
    for (auto &E : *FAI.getEntries()) {
      Output << E.getName() << " ";
      Output << E.getArgNum() << " ";
      Output << E.getLoadFreq() << " ";
      Output << E.getStoreFreq() << "\n";
    }
    Output.close();
  }

  return false;
}

bool FuncArgAccessCGInfoPass::computeLocalAccessFreq(Function &F) {
  // Add an entry for every pointer argument
  Function::ArgumentListType::iterator AIT;
  int i = 0;
  for (AIT = F.getArgumentList().begin(); AIT != F.getArgumentList().end(); AIT++, i++) {
    Value *A = &*AIT;
    auto *PT = dyn_cast<PointerType>(A->getType());
    if (!PT)
      continue;
    FuncArgEntry E(&F, A, i, F.getName());
    if (!E.tryInsertAliasPtr(A))
      assert(0);
    FAI.newEntry(E);
  }

  // Pointer aliasing analysis
  unsigned NumRounds = 0;
  unsigned NumNewAdded;
  do {
    errs() << "Round " << NumRounds << "\n";
    NumNewAdded = 0;
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          assert(LI->getNumOperands() >= 1);
          Value *LoadAddr = LI->getOperand(0);
          if (FuncArgEntry *InsertEntry = FAI.getBaseAliasEntry(LoadAddr)) {
            if(FAI.tryInsertAliasEntry(InsertEntry, LI)) {
              errs() << "    alias entry ";
              LI->dump();
              NumNewAdded++;
            }
          }
        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          Value *StoreContent = SI->getOperand(0);
          Value *StoreAddr = SI->getOperand(1);
          // To handle address space cast
          if (auto *CE = dyn_cast<ConstantExpr>(StoreAddr)) {
            //CE->getOperand(0)->dump();
            continue;
          }
          if (FuncArgEntry *InsertEntry = FAI.getAliasEntry(StoreContent)) {
            if (FAI.tryInsertBaseAliasEntry(InsertEntry, StoreAddr)) {
              errs() << "    base alias entry ";
              StoreAddr->dump();
              NumNewAdded++;
            }
          }
          if (FuncArgEntry *InsertEntry = FAI.getBaseAliasEntry(StoreAddr)) {
            FuncArgEntry *InsertEntry2 = FAI.getAliasEntry(StoreContent);
            if (InsertEntry != InsertEntry2) {
              InsertEntry->dump();
              InsertEntry2->dump();
              errs() << "Warning: store a different alias pointer to a base pointer\n";
              return false;
            }
          }
        } else if (auto *BCI = dyn_cast<BitCastInst>(&I)) {
          Value *CastSource = BCI->getOperand(0);
          unsigned NumAlias = 0;
          if (auto *SourceEntry = FAI.getAliasEntry(CastSource)) {
            if (FAI.tryInsertAliasEntry(SourceEntry, BCI)) {
              errs() << "    alias entry ";
              BCI->dump();
              NumNewAdded++;
              NumAlias++;
            }
          }
          if (auto *SourceEntry = FAI.getBaseAliasEntry(CastSource)) {
            if (FAI.tryInsertBaseAliasEntry(SourceEntry, BCI)) {
              errs() << "    base alias entry ";
              BCI->dump();
              NumNewAdded++;
              NumAlias++;
            }
          }
          if (auto *SourceEntry = FAI.getAliasEntry(BCI)) {
            if (FAI.tryInsertAliasEntry(SourceEntry, CastSource)) {
              errs() << "  alias entry ";
              CastSource->dump();
              NumNewAdded++;
              NumAlias++;
            }
          }
          if (auto *SourceEntry = FAI.getBaseAliasEntry(BCI)) {
            if (FAI.tryInsertBaseAliasEntry(SourceEntry, CastSource)) {
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
          FuncArgEntry *SourceEntry;
          if (SourceEntry = FAI.getAliasEntry(BasePtr)) {
            if (FAI.tryInsertAliasEntry(SourceEntry, GEPI)) {
              errs() << "  alias entry ";
              GEPI->dump();
              NumNewAdded++;
            }
          } else if (SourceEntry = FAI.getAliasEntry(GEPI)) {
            if (FAI.tryInsertAliasEntry(SourceEntry, BasePtr)) {
              errs() << "  alias entry ";
              BasePtr->dump();
              NumNewAdded++;
            }
          } else {
            SourceEntry = FAI.getBaseAliasEntry(GEPI);
            auto EVSet = FAI.getBaseOffsetAliasEntries(BasePtr);
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
              if (FAI.tryInsertBaseOffsetAliasEntry(SourceEntry, BasePtr, V)) {
                errs() << "  base alias offset entry (" << V << ") ";
                BasePtr->dump();
                NumNewAdded++;
              }
            } else {
              for (auto EV : EVSet) {
                SourceEntry = EV.first;
                int64_t Diff = EV.second - V;
                if (FAI.tryInsertBaseOffsetAliasEntry(SourceEntry, GEPI, Diff)) {
                  errs() << "  base alias offset entry (" << Diff << ") ";
                  GEPI->dump();
                  NumNewAdded++;
                }
              }
            }
          }
        }
      }
    }
    NumRounds++;
  } while (NumNewAdded);
  errs() << "Round end\n";

  // Access frequency analysis
  BlockFrequencyInfo *BFI = &getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
  for (auto &BB : F) {
    double Freq = (double)BFI->getBlockFreq(&BB).getFrequency() / (double)BFI->getEntryFreq();
    for (auto &I : BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        Value *LoadAddr = LI->getOperand(0);
        if (FuncArgEntry *E = FAI.getAliasEntry(LoadAddr)) {
          errs() << "    load (" << Freq << ") from ";
          E->dumpBase();
          E->load_freq += Freq;
        }
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        Value *StoreAddr = SI->getOperand(1);
        if (FuncArgEntry *E = FAI.getAliasEntry(StoreAddr)) {
          errs() << "    store (" << Freq << ") to ";
          E->dumpBase();
          E->store_freq += Freq;
        }
      } else if (auto *CI = dyn_cast<CallInst>(&I)) {
        auto *Callee = CI->getCalledFunction();
        // OpenMP target calls
        if (Callee && Callee->getName().find("__tgt_target") == 0 &&
            Callee->getName().find("__tgt_target_data") == std::string::npos) {
          auto *MapTypeCE = dyn_cast<ConstantExpr>(CI->getArgOperand(6));
          assert(MapTypeCE);
          auto *GV = dyn_cast<GlobalVariable>(MapTypeCE->getOperand(0));
          assert(GV);
          auto *C = dyn_cast<ConstantDataArray>(GV->getOperand(0));
          assert(C);
          unsigned NumArgs = C->getNumElements();
          std::string FuncName = CI->getOperand(1)->getName();
          auto *Args = CI->getArgOperand(4);
          for (unsigned i = 0; i < NumArgs; i++) {
            auto *E = FAI.getBaseOffsetAliasEntries(Args, i);
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
            Value *OPD = CI->getOperand(i);
            unsigned AliasTy = 0;
            FuncArgEntry*E = FAI.getAliasEntry(OPD);
            if (E) {
              assert(FAI.getBaseAliasEntry(OPD) == NULL);
              FuncArgEntry *FAE = FAI.getFuncArgEntry(CI->getCalledFunction(), i);
              if (FAE && FAE->getValid()) { // Could reach declaration here
                errs() << "    call (" << Freq << ", " << FAE->getLoadFreq()
                       << ", " << FAE->getStoreFreq() << ") using ";
                E->dumpBase();
                E->load_freq += Freq * FAE->getLoadFreq();
                E->store_freq += Freq * FAE->getStoreFreq();
              } else if (!CI->getCalledFunction()->isDeclaration())
                errs() << "Warning: wrong traversal order, or recursive call\n";
            }
          }
        }
      }
    }
  }

  for (auto &E : *FAI.getEntries()) {
    if (E.isMatch(&F, -1)) {
      errs() << "  Frequency of ";
      E.dumpBase();
      errs() << "    load is " << E.load_freq << "\n";
      errs() << "    store is " << E.store_freq << "\n";
      E.setValid();
    }
  }
}
