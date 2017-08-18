#ifndef LLVM_UVM_PASS_CGMEMANALYSIS_H
#define LLVM_UVM_PASS_CGMEMANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "MemAccessDataStructure.h"

struct FuncArgAccessCGInfoPass : public ModulePass {
  static char ID;
  FuncArgAccessCGInfoPass() : ModulePass(ID) {}
  MemInfo<FuncArgEntry> FAI; // store analysis result

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    errs() << "  ---- Function Argument Access Frequency CG Analysis ----\n";
    CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
    for (scc_iterator<CallGraph *> I = scc_begin(&CG); !I.isAtEnd(); ++I) {
      errs() << I->size() << "\n";
      if (I->size() != 1)
        continue;

      Function *F = I->front()->getFunction();
      if (F && !F->isDeclaration()/* && F->doesNotRecurse()*/) {
        errs() << "On function " << F->getName() << "\n";
        computeLocalAccessFreq(*F);
      }
    }
    return false;
  }

  bool computeLocalAccessFreq(Function &F);

  MemInfo<FuncArgEntry> &getFAI() { return FAI; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<CallGraphWrapperPass>();
  }
};

#endif // LLVM_UVM_PASS_CGMEMANALYSIS_H
