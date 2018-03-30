#ifndef LLVM_OMP_PASS_H
#define LLVM_OMP_PASS_H

#include "llvm/Pass.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "MemAnalysisDataStructure.h"
#include "CGMemAnalysis.h"
using namespace llvm;

struct OMPPass : public ModulePass {
  static char ID;
  MemInfo<DataEntry> MAI; // store analysis result
  MemInfo<FuncArgEntry> TFAI; // Store target functions
  DataLayout *DL;
  OMPPass() : ModulePass(ID) {
  }

  virtual bool runOnModule(Module &M);

  bool analyzeGPUAlloc(Module &M);
  bool analyzePointerPropagation(Module &M);
  void calculateAccessFreq(Module &M);
  bool optimizeDataAllocation(Module &M);
  bool optimizeDataMapping(Module &M);

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

bool compareAccessFreq(DataEntry A, DataEntry B);

#endif // LLVM_OMP_PASS_H
