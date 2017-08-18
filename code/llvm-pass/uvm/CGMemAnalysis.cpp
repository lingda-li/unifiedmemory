#include "CGMemAnalysis.h"
using namespace llvm;

char FuncArgAccessCGInfoPass::ID = 0;

bool FuncArgAccessCGInfoPass::computeLocalAccessFreq(Function &F) {
  // Add an entry for every pointer argument
  Function::ArgumentListType::iterator AIT;
  int i = 0;
  for (AIT = F.getArgumentList().begin(); AIT != F.getArgumentList().end(); AIT++, i++) {
    Value *A = &*AIT;
    auto *PT = dyn_cast<PointerType>(A->getType());
    if (!PT)
      continue;
    FuncArgEntry E(&F, A, i);
    if (!E.tryInsertAliasPtr(A))
      assert(0);
    FAI.newEntry(E);
  }

  // Pointer aliasing analysis
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        assert(LI->getNumOperands() >= 1);
        Value *LoadAddr = LI->getOperand(0);
        if (FuncArgEntry *InsertEntry = FAI.getBaseAliasEntry(LoadAddr)) {
          if(FAI.tryInsertAliasEntry(InsertEntry, LI)) {
            errs() << "  alias entry ";
            LI->dump();
          }
        }
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        Value *StoreContent = SI->getOperand(0);
        Value *StoreAddr = SI->getOperand(1);
        if (FuncArgEntry *InsertEntry = FAI.getAliasEntry(StoreContent)) {
          if(FAI.tryInsertBaseAliasEntry(InsertEntry, StoreAddr)) {
            errs() << "  base alias entry ";
            StoreAddr->dump();
          }
        }
        if (FuncArgEntry *InsertEntry = FAI.getBaseAliasEntry(StoreAddr)) {
          FuncArgEntry *InsertEntry2 = FAI.getAliasEntry(StoreContent);
          if (InsertEntry != InsertEntry2) {
            errs() << "Warning: store a different alias pointer to a base pointer\n";
            return false;
          }
        }
      } else if (auto *BCI = dyn_cast<BitCastInst>(&I)) {
        Value *CastSource = BCI->getOperand(0);
        unsigned NumAlias = 0;
        if (auto *SourceEntry = FAI.getAliasEntry(CastSource)) {
          if (FAI.tryInsertAliasEntry(SourceEntry, BCI)) {
            errs() << "  alias entry ";
            BCI->dump();
            NumAlias++;
          }
        }
        if (auto *SourceEntry = FAI.getBaseAliasEntry(CastSource)) {
          if (FAI.tryInsertBaseAliasEntry(SourceEntry, BCI)) {
            errs() << "  base alias entry ";
            BCI->dump();
            NumAlias++;
          }
        }
        if (NumAlias > 1)
          errs() << "Error: a value is alias for multiple entries\n";
      } else if (auto *GEPI = dyn_cast<GetElementPtrInst>(&I)) {
        Value *BasePtr = GEPI->getOperand(0);
        if (auto *SourceEntry = FAI.getAliasEntry(BasePtr)) {
          if (FAI.tryInsertAliasEntry(SourceEntry, GEPI)) {
            errs() << "  alias entry ";
            GEPI->dump();
          }
        }
        if (auto *SourceEntry = FAI.getBaseAliasEntry(BasePtr)) {
          errs() << "Warning: cast to a base pointer\n";
          return false;
        }
      }
    }
  }

  // Access frequency analysis
  BlockFrequencyInfo *BFI = &getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
  for (auto &BB : F) {
    double Freq = (double)BFI->getBlockFreq(&BB).getFrequency() / (double)BFI->getEntryFreq();
    for (auto &I : BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        Value *LoadAddr = LI->getOperand(0);
        if (FuncArgEntry *E = FAI.getAliasEntry(LoadAddr)) {
          errs() << "  load (" << Freq << ") from ";
          E->dumpBase();
          E->load_freq += Freq;
        }
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        Value *StoreAddr = SI->getOperand(1);
        if (FuncArgEntry *E = FAI.getAliasEntry(StoreAddr)) {
          errs() << "  store (" << Freq << ") to ";
          E->dumpBase();
          E->store_freq += Freq;
        }
      } else if (auto *CI = dyn_cast<CallInst>(&I)) {
        for (int i = 0; i < I.getNumOperands(); i++) {
          Value *OPD = CI->getOperand(i);
          unsigned AliasTy = 0;
          DataEntry *E = FAI.getAliasEntry(OPD);
          if (E) {
            assert(FAI.getBaseAliasEntry(OPD) == NULL);
            FuncArgEntry *FAE = FAI.getFuncArgEntry(CI->getCalledFunction(), i);
            if (FAE && FAE->getValid()) { // Could reach declaration here
              errs() << "  call (" << Freq << ", " << FAE->getLoadFreq()
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

  for (auto &E : *FAI.getEntries()) {
    if (E.isMatch(&F, -1)) {
      errs() << "Frequency of ";
      E.dumpBase();
      errs() << "  load is " << E.load_freq << "\n";
      errs() << "  store is " << E.store_freq << "\n";
      E.setValid();
    }
  }
}
