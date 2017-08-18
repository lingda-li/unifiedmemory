#ifndef LLVM_UVM_PASS_MEMACCESSINFO_H
#define LLVM_UVM_PASS_MEMACCESSINFO_H

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

struct MemAccessInfoPass : public ModulePass {
  static char ID;
  MemAccessInfoPass() : ModulePass(ID) {}
  MemInfo<DataEntry> MAI; // store analysis result

  virtual bool runOnModule(Module &M);

  void calculateAccessFreq(Module &M);
  bool analyzeGPUAlloc(Module &M);
  bool analyzeCPUAlloc(Module &M);
  bool analyzePointerPropagation(Module &M);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<FuncArgAccessCGInfoPass>();
  }
};

#endif // LLVM_UVM_PASS_MEMACCESSINFO_H
