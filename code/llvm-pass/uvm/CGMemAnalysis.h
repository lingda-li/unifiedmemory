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
#include "MemAnalysisDataStructure.h"
using namespace llvm;

struct FuncArgAccessCGInfoPass : public ModulePass {
  static char ID;
  FuncArgAccessCGInfoPass() : ModulePass(ID) {}
  MemInfo<FuncArgEntry> FAI; // store analysis result

  bool runOnModule(Module &M) override {
    //if (skipModule(M))
    //  return false;

    //errs() << "  ---- Function Argument Access Frequency CG Analysis ----\n";
    //CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
    //CG.dump();
    //unsigned sccNum = 0;
    //for (scc_iterator<CallGraph *> I = scc_begin(&CG); !I.isAtEnd(); ++I) {
    //  const std::vector<CallGraphNode*> &SCC = *I;
    //  errs() << "\nSCC #" << ++sccNum << " " << I->size() << "\n";
    //  //continue;
    //  if (I->size() != 1)
    //    continue;

    //  for (auto *CGN : SCC)
    //    if (Function *F = CGN->getFunction())
    //   //   F->dump();
    //    errs() << "On function " << F->getName() << "\n";
    //  //Function *F = I->front()->getFunction();
    //  ////std::vector<CallGraphNode*>::const_iterator FI = nextSCC.begin();
    //  ////if (FI == nextSCC.end())
    //  ////  continue;
    //  ////Function *F = (*FI)->getFunction();
    //  //if (F && !F->isDeclaration()/* && F->doesNotRecurse()*/) {
    //  //F->dump();
    //  //  errs() << "On function " << F->getName() << "\n";
    //  //  //computeLocalAccessFreq(*F);
    //  //}
    //}
  CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  unsigned sccNum = 0;
  errs() << "SCCs for the program in PostOrder:";
  for (scc_iterator<CallGraph*> SCCI = scc_begin(&CG); !SCCI.isAtEnd();
       ++SCCI) {
    const std::vector<CallGraphNode*> &nextSCC = *SCCI;
    errs() << "\nSCC #" << ++sccNum << " : ";
    for (std::vector<CallGraphNode*>::const_iterator I = nextSCC.begin(),
           E = nextSCC.end(); I != E; ++I)
      errs() << ((*I)->getFunction() ? (*I)->getFunction()->getName()
                                     : "external node") << ", ";
    if (nextSCC.size() == 1 && SCCI.hasLoop())
      errs() << " (Has self-loop).";
  }
  errs() << "\n";
    return false;
  }

  bool computeLocalAccessFreq(Function &F);

  MemInfo<FuncArgEntry> &getFAI() { return FAI; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    //AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<CallGraphWrapperPass>();
  }
};

#endif // LLVM_UVM_PASS_CGMEMANALYSIS_H
