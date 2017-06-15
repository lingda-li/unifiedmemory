#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

namespace {
  struct SkeletonPass : public FunctionPass {
    static char ID;
    SkeletonPass() : FunctionPass(ID) {}

    virtual bool runOnFunction(Function &F) {
      //errs() << "I saw a function called " << F.getName() << "!\n";
      bool Changed = false;
      LLVMContext& Ctx = F.getContext();
      auto* I8PPTy = PointerType::get(PointerType::get(Type::getInt8Ty(Ctx), 0), 0);
      Constant* cudaMallocManagedFunc = F.getParent()->getOrInsertFunction("cudaMallocManaged", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx), NULL);
      SmallVector<CallInst *, 2> CallsToDelete;

      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            auto *Callee = CI->getCalledFunction();
            if (Callee && Callee->getName() == "cudaLaunch") {
              errs() << "I saw a function call of " << Callee->getName() << " (" << CI->getNumArgOperands() << ")\n";
              auto *A = CI->getArgOperand(0);
              A->dump();
              if(auto *FA = dyn_cast<Function>(A))
                errs() << "haha\n";
            //} else if (Callee && Callee->getName() == "cudaSetupArgument") {
            //  errs() << "I saw a function call of " << Callee->getName() << " (" << CI->getNumArgOperands() << ")\n";
            //  auto *A = CI->getArgOperand(0);
            //  A->dump();
            } else if (Callee && Callee->getName() == "cudaMalloc") {
              errs() << "I saw a function call of " << Callee->getName() << " (" << CI->getNumArgOperands() << ")\n";
              CI->dump();
              for(int i = 0; i < CI->getNumArgOperands(); i++) {
                auto *A = CI->getArgOperand(i);
                A->dump();
              }

              IRBuilder<> builder(CI);
              Value* args[] = {CI->getArgOperand(0), CI->getArgOperand(1)};
              auto InsertCall = builder.CreateCall(cudaMallocManagedFunc, args);
              for(auto& U : CI->uses()) {
                User* user = U.getUser();
                user->setOperand(U.getOperandNo(), InsertCall);
              }
              CallsToDelete.push_back(CI);
            } else if (Callee && Callee->getName() == "cudaMemcpy") {
              errs() << "I saw a function call of " << Callee->getName() << " (" << CI->getNumArgOperands() << ")\n";
              for(int i = 0; i < CI->getNumArgOperands(); i++) {
                auto *A = CI->getArgOperand(i);
                A->dump();
              }

              CallsToDelete.push_back(CI);
            } else if (Callee && Callee->getName() == "cudaMallocManaged") {
              errs() << "I saw a function call of " << Callee->getName() << " (" << CI->getNumArgOperands() << ")\n";
              for(int i = 0; i < CI->getNumArgOperands(); i++) {
                auto *A = CI->getArgOperand(i);
                A->dump();
              }
            } else if (Callee && Callee->getName() == "malloc") {
              errs() << "I saw a function call of " << Callee->getName() << " (" << CI->getNumArgOperands() << ")\n";
              for(int i = 0; i < CI->getNumArgOperands(); i++) {
                auto *A = CI->getArgOperand(i);
                A->dump();
              }
            }
          }
        }
      }

      for (auto *CI : CallsToDelete) {
        CI->eraseFromParent();
        Changed = true;
      }
      return Changed;
    }
  };
}

char SkeletonPass::ID = 0;

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
static void registerSkeletonPass(const PassManagerBuilder &,
                         legacy::PassManagerBase &PM) {
  PM.add(new SkeletonPass());
}
static RegisterStandardPasses
  RegisterMyPass(PassManagerBuilder::EP_EarlyAsPossible,
                 registerSkeletonPass);
