#include "llvm/Pass.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "CudaPass.h"
#include "llvm/ADT/SmallSet.h"
using namespace llvm;

namespace {
  struct UVMTransformPass : public ModulePass {
    static char ID;
    DataInfo *Info;
    UVMTransformPass() : ModulePass(ID) {
      Info = new DataInfo;
    }
    bool Succeeded = true;

    virtual bool runOnModule(Module &M) {
      errs() << "  ---- UVM Transform (" << M.getName() << ", " << M.getTargetTriple() << ") ----\n";
      // Find all places that allocate memory
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;
        for (auto &BB : F) {
          for (auto &I : BB) {
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              auto *Callee = CI->getCalledFunction();
              if (Callee && Callee->getName() == "cudaMalloc") {
                errs() << "Error: this is for UVM only\n";
                Succeeded = false;
              } else if (Callee && Callee->getName() == "malloc") {
                errs() << "Error: this is for UVM only\n";
                Succeeded = false;
              } else if (Callee && Callee->getName() == "cudaMallocManaged") {
                auto *AllocPtr = CI->getArgOperand(0);
                if (auto *BCI = dyn_cast<BitCastInst>(AllocPtr)) {
                  Value *BasePtr = BCI->getOperand(0);
                  if (auto *AI = dyn_cast<AllocaInst>(BasePtr)) {
                    Value *BasePtr = AI;
                    if (Info->getBaseAliasEntry(BasePtr) == NULL) {
                      errs() << "new entry ";
                      BasePtr->dump();
                      DataEntry *data_entry = new DataEntry(BasePtr, 2, CI->getArgOperand(1)); // managed space
                      data_entry->alloc = CI;
                      FuncInfoEntry *FIE = new FuncInfoEntry(&F);
                      data_entry->func_map.insert(std::make_pair(&F, FIE));
                      data_entry->insertFuncInfoEntry(FIE);
                      Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                    } else
                      errs() << "Error: redundant allocation?\n";
                  } else
                    DEBUG_PRINT
                } else if (auto *AI = dyn_cast<AllocaInst>(AllocPtr)) {
                  Value *BasePtr = AI;
                  if (Info->getBaseAliasEntry(BasePtr) == NULL) {
                    errs() << "new entry ";
                    BasePtr->dump();
                    DataEntry *data_entry = new DataEntry(BasePtr, 2, CI->getArgOperand(1)); // managed space
                    data_entry->alloc = CI;
                    Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                  } else
                    errs() << "Error: redundant allocation?\n";
                } else
                  DEBUG_PRINT
              }
            }
          }
        }
      }

      // Find all pointers to allocated space
      // Iterate until no new functions that are discovered
      SmallVector<Function*, 8> Funcs;
      SmallVector<FuncInfoEntry*, 4> ArgsByVal;
      SmallVector<FuncInfoEntry*, 4> ArgsByRef;
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;
        Funcs.push_back(&F);
      }

      unsigned NumRounds = 0;
      SmallVector<Function*, 4> AddedFuncs;
      while (!Funcs.empty()) {
        errs() << "Round " << NumRounds << "\n";
        for (Function *FP : Funcs) {
          auto &F = *FP;
          for (auto &BB : F) {
            for (auto &I : BB) {
              if (auto *LI = dyn_cast<LoadInst>(&I)) {
                assert(LI->getNumOperands() >= 1);
                Value *LoadAddr = LI->getOperand(0);
                if (DataEntry *InsertEntry = Info->getBaseAliasEntry(LoadAddr)) {
                  if(Info->tryInsertAliasEntry(InsertEntry, LI)) {
                    errs() << "  alias entry ";
                    LI->dump();
                  }
                }
              } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
                Value *StoreContent = SI->getOperand(0);
                Value *StoreAddr = SI->getOperand(1);
                if (DataEntry *InsertEntry = Info->getAliasEntry(StoreContent)) {
                  if(Info->tryInsertBaseAliasEntry(InsertEntry, StoreAddr)) {
                    errs() << "  base alias entry ";
                    StoreAddr->dump();
                  }
                }
                if (DataEntry *InsertEntry = Info->getBaseAliasEntry(StoreAddr)) {
                  DataEntry *InsertEntry2 = Info->getAliasEntry(StoreContent);
                  if (InsertEntry != InsertEntry2) {
                    errs() << "Warning: store a different alias pointer to a base pointer\n";
                    Succeeded = false;
                  }
                }
              } else if (auto *BCI = dyn_cast<BitCastInst>(&I)) {
                Value *CastSource = BCI->getOperand(0);
                unsigned NumAlias = 0;
                if (auto *SourceEntry = Info->getAliasEntry(CastSource)) {
                  if (Info->tryInsertAliasEntry(SourceEntry, BCI)) {
                    errs() << "  alias entry ";
                    BCI->dump();
                    NumAlias++;
                  }
                }
                if (auto *SourceEntry = Info->getBaseAliasEntry(CastSource)) {
                  if (Info->tryInsertBaseAliasEntry(SourceEntry, BCI)) {
                    errs() << "  base alias entry ";
                    BCI->dump();
                    NumAlias++;
                  }
                }
                if (NumAlias > 1)
                  errs() << "Error: a value is alias for multiple entries\n";
              } else if (auto *CI = dyn_cast<CallInst>(&I)) {
                auto *Callee = CI->getCalledFunction();
                if (Callee && Callee->isIntrinsic())
                  continue;
                else if (Callee && Callee->getName() == "cudaMallocManaged")
                  continue;
                else if (Callee && Callee->getName() == "cudaFree")
                  continue;
                bool Use = false;
                for (int i = 0; i < I.getNumOperands(); i++) {
                  Value *OPD = CI->getOperand(i);
                  unsigned AliasTy = 0;
                  DataEntry *SourceEntry;
                  if (auto *E = Info->getAliasEntry(OPD)) {
                    SourceEntry = E;
                    AliasTy += 1;
                  }
                  if (auto *E = Info->getBaseAliasEntry(OPD)) {
                    SourceEntry = E;
                    AliasTy += 2;
                  }
                  assert(AliasTy < 3);
                  if (AliasTy > 0) {
                    if (Callee && Callee->isDeclaration()) {
                      if (Callee->getName() != "cudaSetupArgument")
                        errs() << "Warning: reach to function declaration " << Callee->getName();
                      break;
                    }
                    Use = true;
                    int argcount = 0;
                    Function::arg_iterator A;
                    for (A = Callee->arg_begin(); A != Callee->arg_end(); A++) {
                      if (argcount == i)
                        break;
                      argcount++;
                    }
                    assert(argcount == i);
                    if (AliasTy == 1) {
                      if (Info->tryInsertAliasEntry(SourceEntry, &(*A))) {
                        assert(SourceEntry->func_map.find(Callee) == SourceEntry->func_map.end());
                        FuncInfoEntry *FIE = new FuncInfoEntry(Callee, CI, &(*A), OPD, 1);
                        SourceEntry->insertFuncInfoEntry(FIE);
                        assert(SourceEntry->func_map.find(&F) != SourceEntry->func_map.end());
                        FIE->setParent(SourceEntry->func_map.find(&F)->second);
                        ArgsByVal.push_back(FIE);
                        errs() << "  alias entry (func arg) ";
                        A->dump();
                      }
                    } else {
                      if (Info->tryInsertBaseAliasEntry(SourceEntry, &(*A))) {
                        assert(SourceEntry->func_map.find(Callee) == SourceEntry->func_map.end());
                        FuncInfoEntry *FIE = new FuncInfoEntry(Callee, CI, &(*A), OPD, 2);
                        SourceEntry->insertFuncInfoEntry(FIE);
                        assert(SourceEntry->func_map.find(&F) != SourceEntry->func_map.end());
                        FIE->setParent(SourceEntry->func_map.find(&F)->second);
                        ArgsByRef.push_back(FIE);
                        errs() << "  base alias entry (func arg) ";
                        A->dump();
                      }
                    }
                  }
                }
                if (Use) {
                  errs() << "Info: add function " << Callee->getName() << " to Round " << NumRounds+1 << "\n";
                  AddedFuncs.push_back(Callee);
                }
              }
            }
          }
        }

        while (!Funcs.empty())
          Funcs.pop_back();
        for (Function *FP : AddedFuncs)
          Funcs.push_back(FP);
        while (!AddedFuncs.empty())
          AddedFuncs.pop_back();
        NumRounds++;
      }
      errs() << "Round end\n";

      // Find where data pointer is passed to GPU kernel
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;
        for (auto &BB : F) {
          for (auto &I : BB) {
            bool Use = false;
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              auto *Callee = CI->getCalledFunction();
              if (Callee && Callee->getName() == "cudaSetupArgument") {
                auto FirstArg = CI->getOperand(0);
                if (DataEntry *SourceEntry = Info->getBaseAliasEntry(&(*FirstArg))) {
                  errs() << "Info: data ";
                  SourceEntry->base_ptr->dump();
                  errs() << "  is passed to a kernel through ";
                  FirstArg->dump();
                  SourceEntry->send2kernel.push_back(CI);

                  // Find the corresponding kernel call
                  Instruction *NextInst = CI->getNextNode();
                  auto *CBB = &BB;
                  bool FoundKernel = false;
                  do {
                    if (auto *NCI = dyn_cast<CallInst>(NextInst)) {
                      auto *NextCallee = NCI->getCalledFunction();
                      if (NextCallee && NextCallee->getName() == "cudaLaunch") {
                        errs() << "  kernel call is ";
                        NCI->dump();
                        FoundKernel = true;
                        SourceEntry->kernel.push_back(NCI);
                        break;
                      }
                    }
                    NextInst = NextInst->getNextNode();
                    if (!NextInst) {
                      CBB = CBB->getNextNode(); // FIXME: go to the successor block
                      if (!CBB)
                        break;
                      NextInst = &(*CBB->begin());
                    }
                  } while (NextInst);
                  if (!FoundKernel) {
                    errs() << "Error: didn't find the kernel call\n";
                    Succeeded = false;
                  }
                }
              } else if (Callee && Callee->getName() == "cudaFree") {
                auto *FreePtr = CI->getArgOperand(0);
                DataEntry *data_entry = Info->getAliasEntry(FreePtr);
                if (!data_entry) // In case this cudaFree is for cudaMalloc
                  continue;
                assert(!data_entry->free);
                data_entry->free = CI;
                errs() << "Info: find free for ";
                data_entry->base_ptr->dump();
              }
            }
          }
        }
      }

      bool Changed = false;
      LLVMContext& Ctx = M.getContext();
      auto UVMMemInfoTy = StructType::create(Ctx, "struct.uvmMallocInfo");
      UVMMemInfoTy->setBody(PointerType::get(Type::getInt8Ty(Ctx), 0), Type::getInt64Ty(Ctx), PointerType::get(Type::getInt8Ty(Ctx), 0), Type::getInt8Ty(Ctx));
      auto *UVMMemInfoPTy = PointerType::get(UVMMemInfoTy, 0);
      auto uvmMallocFunc = M.getOrInsertFunction("__uvm_malloc", Type::getVoidTy(Ctx), UVMMemInfoPTy, nullptr);
      auto uvmMemcpyFunc = M.getOrInsertFunction("__uvm_memcpy", Type::getVoidTy(Ctx), UVMMemInfoPTy, Type::getInt32Ty(Ctx), nullptr);
      auto uvmFreeFunc = M.getOrInsertFunction("__uvm_free", Type::getVoidTy(Ctx), UVMMemInfoPTy, nullptr);
      SmallVector<Instruction*, 8> InstsToDelete;

      // Allocate initial UVM runtime data structures
      for (auto &DME : Info->DataMap) {
        DataEntry *DE = DME.second;
        if (!DE->free) {
          errs() << "Error: didn't find free for ";
          DE->base_ptr->dump();
          Succeeded = false;
          break;
        }

        // Change memory allocation api calls:
        // replace cudaMallocManaged with __uvm_malloc
        auto *AllocInst = dyn_cast<CallInst>(DE->alloc);
        assert(AllocInst);
        auto *BaseInst = dyn_cast<Instruction>(DE->base_ptr);
        assert(BaseInst);
        auto *AllocPtr = AllocInst->getArgOperand(0);
        auto *AllocSize = AllocInst->getArgOperand(1);
        IRBuilder<> builder(AllocInst);
        Value *AI = builder.CreateAlloca(UVMMemInfoTy); // uvmMallocInfo allocated instruction
        ConstantInt *Offset = ConstantInt::get(Type::getInt32Ty(Ctx), 1, false);
        Value *IndexList[2] = {ConstantInt::get(Type::getInt64Ty(Ctx), 0, false), Offset};
        auto *SizeGEPI = builder.CreateGEP(AI, ArrayRef<Value*>(IndexList, 2));
        auto *SI = builder.CreateStore(AllocSize, SizeGEPI);
        // Insert __uvm_malloc
        Value* args[] = {AI};
        auto *UVMMallocCI = builder.CreateCall(uvmMallocFunc, args);
        Offset = ConstantInt::get(Type::getInt32Ty(Ctx), 2, false);
        Value *IndexList2[2] = {ConstantInt::get(Type::getInt64Ty(Ctx), 0, false), Offset};
        auto *HostGEPI = builder.CreateGEP(AI, ArrayRef<Value*>(IndexList2, 2));
        Value *HostBCI;
        if (HostGEPI->getType() != BaseInst->getType())
          HostBCI = builder.CreateBitCast(HostGEPI, BaseInst->getType());
        else
          HostBCI = HostGEPI;
        // Replace usage
        SmallVector<User*, 6> Users;
        SmallVector<unsigned, 6> UsersNo;
        for (auto &U : BaseInst->uses()) {
          User* user = U.getUser();
          Users.push_back(user);
          UsersNo.push_back(U.getOperandNo());
        }
        while (!Users.empty()) {
          User* user = Users.back();
          user->setOperand(UsersNo.back(), HostBCI);
          Users.pop_back();
          UsersNo.pop_back();
        }
        Function *CurFunc = AllocInst->getParent()->getParent();
        assert(DE->func_map.find(CurFunc) != DE->func_map.end());
        DE->func_map.find(CurFunc)->second->setLocalCopy(AI);
        errs() << "Info: reallocate ";
        AllocInst->dump();
        errs() << "            with ";
        UVMMallocCI->dump();
        InstsToDelete.push_back(AllocInst);
      }

      if (!Succeeded) {
        errs() << "Info: didn't perform transformation because data analysis fails\n";
        return false;
      }

      // Transform all functions that need GPU memory 
      // FIXME: potentially transform more functions than needed
      // Change the argument to pass by refernce instead of by value
      for (FuncInfoEntry *FIE : ArgsByVal) {
        Value *A = FIE->arg;
        Value *AV = FIE->arg_value;
        Function *F = FIE->func;
        CallInst *CI = FIE->call_point;
        FuncInfoEntry *PFIE = FIE->parent;
        Value *ParentCopy = PFIE->local_copy;
        if (PFIE == NULL) {
          assert(FIE->local_copy);
        } else {
          if (ParentCopy == NULL) {
            errs() << "Error: parent " << PFIE->func->getName() << " didn't have a local copy\n";
            continue;
          }
          assert(!FIE->local_copy);
        }
        // Change the argument passed to function
        IRBuilder<> Callerbuilder(CI);
        Value *CallerBCI = Callerbuilder.CreateBitCast(ParentCopy, AV->getType());
        for (auto &U : AV->uses()) {
          User* user = U.getUser();
          if (user == CI) {
            user->setOperand(U.getOperandNo(), CallerBCI);
            break;
          }
        }
        // Change how function argument is used
        Instruction *CalleeFirstInst = &(*F->begin()->begin());
        IRBuilder<> Calleebuilder(CalleeFirstInst);
        Value *CalleeBCI = Calleebuilder.CreateBitCast(A, ParentCopy->getType());
        ConstantInt *Offset = ConstantInt::get(Type::getInt32Ty(Ctx), 2, false);
        Value *IndexList[2] = {ConstantInt::get(Type::getInt64Ty(Ctx), 0, false), Offset};
        auto *HostGEPI = Calleebuilder.CreateGEP(CalleeBCI, ArrayRef<Value*>(IndexList, 2));
        Value *HostBCI;
        if (HostGEPI->getType() != A->getType())
          HostBCI = Calleebuilder.CreateBitCast(HostGEPI, A->getType());
        else
          HostBCI = HostGEPI;
        SmallVector<User*, 6> Users;
        SmallVector<unsigned, 6> UsersNo;
        for (auto &U : A->uses()) {
          User* user = U.getUser();
          Users.push_back(user);
          UsersNo.push_back(U.getOperandNo());
        }
        while (!Users.empty()) {
          User* user = Users.back();
          if (user != CalleeBCI)
            user->setOperand(UsersNo.back(), HostBCI);
          Users.pop_back();
          UsersNo.pop_back();
        }
        // Assign local variable copy
        FIE->local_copy = CalleeBCI;
      }

      for (FuncInfoEntry *FIE : ArgsByRef) {
        Value *A = FIE->arg;
        Function *F = FIE->func;
        assert(0);
      }

      // Transform CUDA APIs to UVM runtime APIs
      for (auto &DME : Info->DataMap) {
        DataEntry *DE = DME.second;
        // Insert memory copy api calls
        {
          // Insert host2device __uvm_memcpy
          for (auto *SACI : DE->send2kernel) {
            Function *CurFunc = SACI->getParent()->getParent();
            auto FuncMapIt = DE->func_map.find(CurFunc);
            assert(FuncMapIt != DE->func_map.end());
            Value *AI = FuncMapIt->second->local_copy;
            if (!AI) {
              errs() << "Error: " << CurFunc->getName() << " didn't have a local copy for ";
              DE->base_ptr->dump();
              continue;
            }
            IRBuilder<> builder(SACI);
            ConstantInt *Direction = ConstantInt::get(Type::getInt32Ty(Ctx), 1, false);
            Value* args[] = {AI, Direction};
            auto *UVMMemcpyCI = builder.CreateCall(uvmMemcpyFunc, args);
            errs() << "Info: H2D transfer inserted ";
            UVMMemcpyCI->dump();
            // Get device memory pointer
            ConstantInt *Offset = ConstantInt::get(Type::getInt32Ty(Ctx), 0, false);
            Value *IndexList[2] = {ConstantInt::get(Type::getInt64Ty(Ctx), 0, false), Offset};
            auto *DeviceGEPI = builder.CreateGEP(AI, ArrayRef<Value*>(IndexList, 2));
            Value *DeviceBCI;
            if (DeviceGEPI->getType() != PointerType::get(Type::getInt8Ty(Ctx), 0))
              DeviceBCI = builder.CreateBitCast(DeviceGEPI, PointerType::get(Type::getInt8Ty(Ctx), 0));
            else
              DeviceBCI = DeviceGEPI;
            SACI->setOperand(0, DeviceBCI);
          }

          // Insert device2host __uvm_memcpy
          for (auto *KCI : DE->kernel) {
            Function *CurFunc = KCI->getParent()->getParent();
            auto FuncMapIt = DE->func_map.find(CurFunc);
            assert(FuncMapIt != DE->func_map.end());
            Value *AI = FuncMapIt->second->local_copy;
            if (!AI) {
              errs() << "Error: " << CurFunc->getName() << " didn't have a local copy for ";
              DE->base_ptr->dump();
              continue;
            }
            auto *InsertPoint = KCI->getNextNode();
            assert(InsertPoint);
            IRBuilder<> builder(InsertPoint);
            ConstantInt *Direction = ConstantInt::get(Type::getInt32Ty(Ctx), 2, false);
            Value* args[] = {AI, Direction};
            auto *UVMMemcpyCI = builder.CreateCall(uvmMemcpyFunc, args);
            errs() << "Info: D2H transfer inserted ";
            UVMMemcpyCI->dump();
          }
        }

        // Change memory free api calls
        {
          auto *FreeInst = dyn_cast<CallInst>(DE->free);
          assert(FreeInst);
          Function *CurFunc = FreeInst->getParent()->getParent();
          auto FuncMapIt = DE->func_map.find(CurFunc);
          assert(FuncMapIt != DE->func_map.end());
          Value *AI = FuncMapIt->second->local_copy;
          if (!AI) {
            errs() << "Error: " << CurFunc->getName() << " didn't have a local copy for ";
            DE->base_ptr->dump();
            continue;
          }
          // Insert __uvm_free
          IRBuilder<> builder(FreeInst);
          Value* args[] = {AI};
          auto *UVMFreeCI = builder.CreateCall(uvmFreeFunc, args);
          InstsToDelete.push_back(FreeInst);
          errs() << "  free is added for ";
          DE->base_ptr->dump();
        }

        // Change device reference
        Changed = true;
      }

      for (auto *I : InstsToDelete)
        I->eraseFromParent();
      return Changed;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;

    DataInfo &getInfo() {
      return *Info;
    }
  };

  void UVMTransformPass::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
  }
}

char UVMTransformPass::ID = 0;

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
static void registerUVMTransformPass(const PassManagerBuilder &,
                         legacy::PassManagerBase &PM) {
  PM.add(new UVMTransformPass());
}
static RegisterStandardPasses
  RegisterMyPass(PassManagerBuilder::EP_EnabledOnOptLevel0,
  //RegisterMyPass(PassManagerBuilder::EP_ModuleOptimizerEarly,
                 registerUVMTransformPass);
