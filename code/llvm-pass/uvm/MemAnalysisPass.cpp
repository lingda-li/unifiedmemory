#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "MemAnalysisDataStructure.h"
#include "CGMemAnalysis.h"
#include "MemAccessInfoPass.h"
using namespace llvm;

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
static void registerMemAccessInfoPass(const PassManagerBuilder &,
                         legacy::PassManagerBase &PM) {
  PM.add(new BlockFrequencyInfoWrapperPass());
  PM.add(new CallGraphWrapperPass());
  PM.add(new FuncArgAccessCGInfoPass());
  PM.add(new MemAccessInfoPass());
}
static RegisterStandardPasses
  //RegisterMyPass(PassManagerBuilder::EP_EarlyAsPossible,
  RegisterMyPass(PassManagerBuilder::EP_EnabledOnOptLevel0,
  //RegisterMyPass(PassManagerBuilder::EP_ModuleOptimizerEarly,
                 registerMemAccessInfoPass);
