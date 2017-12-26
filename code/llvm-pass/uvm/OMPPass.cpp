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
using namespace llvm;

namespace {
  struct OMPPass : public ModulePass {
    static char ID;
    OMPPass() : ModulePass(ID) {
    }

    virtual bool runOnModule(Module &M) {
      errs() << "  ---- OMP (" << M.getName() << ", " << M.getTargetTriple() << ") ----\n";
      bool Changed = false;
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;
        errs() << "  " << F.getName() << "\n";
        //F.dump();
        for (auto &BB : F) {
          for (auto &I : BB) {
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              auto *Callee = CI->getCalledFunction();
              if (Callee && Callee->getName() == "__tgt_target_teams") {
                errs() << "  target call: ";
                CI->dump();
                CI->getArgOperand(1)->dump();
                if (auto *CE = dyn_cast<ConstantExpr>(CI->getArgOperand(6))) {
                  if (auto *GV = dyn_cast<GlobalVariable>(CE->getOperand(0))) {
                    GV->dump();
                    if (auto *C = dyn_cast<ConstantDataArray>(GV->getOperand(0))) {
                      C->dump();
                      SmallVector<unsigned, 16> MapTypes;
                      bool LocalChanged = false;
                      for (unsigned i = 0; i < C->getNumElements(); i++) {
                        auto *ConstantI = dyn_cast<ConstantInt>(C->getElementAsConstant(i));
                        assert(ConstantI && "Suppose to get constant integer");
                        int64_t V = ConstantI->getSExtValue();
                        if (V & 0x01 || V & 0x02) {
                          V &= ~0x03;
                          V |= 0x100;
                          auto *NCI = ConstantInt::get(ConstantI->getType(), V);
                          NCI->dump();
                          LocalChanged = true;
                        }
                        MapTypes.push_back(V);
                      }
                      if (LocalChanged) {
                        auto *NCDA = ConstantDataArray::get(C->getContext(), MapTypes);
                        C->replaceAllUsesWith(NCDA);
                        errs() << "  map type changed: ";
                        Changed = true;
                      }
                    }
                  }
                }
              } else if (Callee && Callee->getName() == "__tgt_target_data_begin") {
                errs() << "  target data call: ";
                CI->dump();
                if (auto *CE = dyn_cast<ConstantExpr>(CI->getArgOperand(5))) {
                  if (auto *GV = dyn_cast<GlobalVariable>(CE->getOperand(0))) {
                    GV->dump();
                    if (auto *C = dyn_cast<ConstantDataArray>(GV->getOperand(0))) {
                      C->dump();
                      SmallVector<unsigned, 16> MapTypes;
                      bool LocalChanged = false;
                      for (unsigned i = 0; i < C->getNumElements(); i++) {
                        auto *ConstantI = dyn_cast<ConstantInt>(C->getElementAsConstant(i));
                        assert(ConstantI && "Suppose to get constant integer");
                        int64_t V = ConstantI->getSExtValue();
                        if (V & 0x01 || V & 0x02) {
                          V &= ~0x03;
                          V |= 0x100;
                          auto *NCI = ConstantInt::get(ConstantI->getType(), V);
                          LocalChanged = true;
                        }
                        MapTypes.push_back(V);
                      }
                      if (LocalChanged) {
                        auto *NCDA = ConstantDataArray::get(C->getContext(), MapTypes);
                        C->replaceAllUsesWith(NCDA);
                        errs() << "  map type changed: ";
                        GV->dump();
                        Changed = true;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      return Changed;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;
  };

  void OMPPass::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
  }
}

char OMPPass::ID = 0;

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
static void registerOMPPass(const PassManagerBuilder &,
                         legacy::PassManagerBase &PM) {
  PM.add(new OMPPass());
}
static RegisterStandardPasses
  //RegisterMyPass(PassManagerBuilder::EP_EnabledOnOptLevel0,
  RegisterMyPass(PassManagerBuilder::EP_OptimizerLast,
  //RegisterMyPass(PassManagerBuilder::EP_ModuleOptimizerEarly,
                 registerOMPPass);
