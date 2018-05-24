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
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include "llvm/Support/Format.h"
#include "TFGPass.h"
using namespace llvm;

char TFGPass::ID = 0;

bool TFGPass::runOnModule(Module &M) {
  // Find target regions
  std::vector<Function*> FuncHasTarget;
  errs() << "  ---- Identify Target Regions ----\n";
  unsigned Idx = 0;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    bool hasTarget = false;
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
          CallSite CS(&I);
          auto *Callee = CS.getCalledFunction();
          if (Callee == NULL)
            continue;
          if (Callee->getName().find("__tgt_target") != 0)
            continue;
          if (Callee->getName().find("__tgt_target_data") != std::string::npos)
            continue;
          errs() << "  target call: ";
          I.dump();
          TargetRegions.push_back(&I);
          TargetRegionMap[&I] = Idx;
          Idx++;
          //Value *TargetRegionName = CS.getArgOperand(0);
          hasTarget = true;
        }
      }
    }
    if (hasTarget)
      FuncHasTarget.push_back(&F);
  }

  // Allocate space
  TargetNum = TargetRegions.size();
  PreDis = new BBTargetDisTy[TargetNum];
  PosDis = new BBTargetDisTy[TargetNum];

  // Calculate target region distance
  for (auto *F : FuncHasTarget) {
    for (unsigned TIdx = 0; TIdx < TargetNum; TIdx++)
      for (auto &BB : *F)
        PreDis[TIdx][&BB] = INFDIS;
    BranchProbabilityInfo &BPI =
      getAnalysis<BranchProbabilityInfoWrapperPass>(*F).getBPI();
    VisitMap.shrink_and_clear();
    // Traverse CFG
    bool R;
    unsigned NIter = 0;
    do {
      TotalDiff = 0.0;
      for (auto &BB : *F)
        VisitMap[&BB] = false;
      R = traverseDFS(&F->getEntryBlock(), BPI);
      NIter++;
      //dumpDis(F);
    } while (R && TotalDiff > 10.0);
    errs() << "" << F->getName() << " converges after " << NIter << " iterations\n";
  }

  // Derive the distance at each target region
  T2TDis = new Target2TargetDisTy[TargetNum];
  for (unsigned TIdx = 0; TIdx < TargetNum; TIdx++) {
    for (unsigned TTIdx = 0; TTIdx < TargetNum; TTIdx++)
      T2TDis[TIdx][TTIdx] = INFDIS;
    auto *TR = TargetRegions[TIdx];
    auto *BB = TR->getParent();
    // Find target regions after the current one
    bool IsFindTR = false;
    unsigned NumFoundTR = 0;
    DenseMap<unsigned, bool> UpdatedIndexes;
    for (auto &I : *BB) {
      if (!IsFindTR) {
        if (TR == &I)
          IsFindTR = true;
        continue;
      }
      if (TargetRegionMap.find(&I) == TargetRegionMap.end())
        continue;
      NumFoundTR++;
      unsigned TTIdx = TargetRegionMap[&I];
      T2TDis[TIdx][TTIdx] = NumFoundTR;
      UpdatedIndexes[TTIdx] = true;
    }
    for (unsigned TTIdx = 0; TTIdx < TargetNum; TTIdx++) {
      if (UpdatedIndexes.find(TTIdx) != UpdatedIndexes.end())
        continue;
      T2TDis[TIdx][TTIdx] = PosDis[TTIdx][BB] + NumFoundTR + 1;
    }
    errs() << "target " << TIdx << ": ";
    for (unsigned TTIdx = 0; TTIdx < TargetNum; TTIdx++) {
      errs() << "(" << TTIdx << ": " << T2TDis[TIdx][TTIdx] << ") ";
    }
    errs() << "\n";
  }

  // Clean up
  delete[] PreDis;
  delete[] PosDis;
  return false;
}

void TFGPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<BranchProbabilityInfoWrapperPass>();
  AU.setPreservesAll();
}

bool TFGPass::traverseDFS(BasicBlock *BB, const BranchProbabilityInfo &BPI) {
  if (VisitMap[BB])
    return true;
  bool Recursive = false;
  auto *TInst = BB->getTerminator();
  VisitMap[BB] = true;

  // Update PosDis
  unsigned NSucc = TInst->getNumSuccessors();
  for (unsigned TIdx = 0; TIdx < TargetNum; TIdx++) {
    if (NSucc == 0)
      PosDis[TIdx][BB] = INFDIS;
    else
      PosDis[TIdx][BB] = 0.0;
  }
  //for (unsigned I = 0; I < NSucc; I++) {
  //  auto *NBB = TInst->getSuccessor(I);
  //  auto P = BPI.getEdgeProbability(BB, I);
  //  double prob = (double)P.getNumerator() / P.getDenominator();
  //  if (traverseDFS(NBB, BPI))
  //    Recursive = true;
  //  for (unsigned TIdx = 0; TIdx < TargetNum; TIdx++) {
  //    PosDis[TIdx][BB] += prob * PreDis[TIdx][NBB];
  //  }
  //}
  std::vector<bool> SuccRecur;
  for (unsigned I = 0; I < NSucc; I++) {
    auto *NBB = TInst->getSuccessor(I);
    if (traverseDFS(NBB, BPI)) {
      SuccRecur.push_back(true);
      Recursive = true;
    } else
      SuccRecur.push_back(false);
  }
  double NonRecurProb = 0.0;
  double FreeProb = 0.0;
  double ReleaseFactor = 10.0;
  for (unsigned I = 0; I < NSucc; I++) {
    if (!SuccRecur[I]) {
      auto P = BPI.getEdgeProbability(BB, I);
      double prob = (double)P.getNumerator() / P.getDenominator();
      NonRecurProb += prob;
      prob /= ReleaseFactor;
      FreeProb = prob * (ReleaseFactor - 1.0);
      auto *NBB = TInst->getSuccessor(I);
      for (unsigned TIdx = 0; TIdx < TargetNum; TIdx++) {
        PosDis[TIdx][BB] += prob * PreDis[TIdx][NBB];
      }
    }
  }
  for (unsigned I = 0; I < NSucc; I++) {
    if (SuccRecur[I]) {
      auto P = BPI.getEdgeProbability(BB, I);
      double prob = (double)P.getNumerator() / P.getDenominator();
      prob += FreeProb * prob / (1.0 - NonRecurProb);
      auto *NBB = TInst->getSuccessor(I);
      for (unsigned TIdx = 0; TIdx < TargetNum; TIdx++) {
        PosDis[TIdx][BB] += prob * PreDis[TIdx][NBB];
      }
    }
  }

  // Update PreDis
  unsigned BBTRN = 0, CBBTRN = 0;
  for (auto *TR : TargetRegions)
    if (TR->getParent() == BB)
      BBTRN++;
  for (unsigned TIdx = 0; TIdx < TargetNum; TIdx++) {
    double OldPreDis = PreDis[TIdx][BB];
    auto *TR = TargetRegions[TIdx];
    if (TR->getParent() == BB) {
      PreDis[TIdx][BB] = CBBTRN;
      CBBTRN++;
    } else
      PreDis[TIdx][BB] = PosDis[TIdx][BB] + BBTRN;
    TotalDiff += fabs(OldPreDis - PreDis[TIdx][BB]);
  }

  return Recursive;
}

void TFGPass::dumpDis(Function *F) {
  for (unsigned TIdx = 0; TIdx < TargetNum; TIdx++) {
    auto *TR = TargetRegions[TIdx];
    errs() << "target " << TIdx << ": ";
    TR->dump();
    for (auto &BB : *F) {
      errs() << "BB " << &BB << ": ";
      errs() << PreDis[TIdx][&BB] << "  " << PosDis[TIdx][&BB] << "\n";
    }
    //errs() << "PreDis: ";
    //for (auto PDBB : PreDis[TIdx])
    //  errs() << "BB " << PDBB.first << " " << PDBB.second << " : ";
    //errs() << "\n";
    //errs() << "PosDis: ";
    //for (auto PDBB : PosDis[TIdx])
    //  errs() << "BB " << PDBB.first << " " << PDBB.second << " : ";
    //errs() << "\n";
  }
  errs() << "Diff: " << TotalDiff << "\n";
}

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
static void registerTFGPass(const PassManagerBuilder &,
                         legacy::PassManagerBase &PM) {
  PM.add(new BranchProbabilityInfoWrapperPass());
  PM.add(new TFGPass());
}
static RegisterStandardPasses
  //RegisterMyPass(PassManagerBuilder::EP_EnabledOnOptLevel0,
  RegisterMyPass(PassManagerBuilder::EP_OptimizerLast,
  //RegisterMyPass(PassManagerBuilder::EP_ModuleOptimizerEarly,
                 registerTFGPass);
