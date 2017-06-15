#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "CudaPass.h"
using namespace llvm;

namespace {
  //struct CudaMallocAnalysisPass : public ModulePass {}
  struct CudaMallocAnalysisPass : public FunctionPass {
    static char ID;
    Module *PreModule = NULL;
    DataInfo *Info;
    //CudaMallocAnalysisPass() : ModulePass(ID) {}
    CudaMallocAnalysisPass() : FunctionPass(ID) {
      Info = new DataInfo;
    }

    //virtual bool runOnModule(Module &M) {}
    virtual bool runOnFunction(Function &F) {
      //errs() << "I saw a function called " << F.getName() << "!\n";
      if (F.getParent() != PreModule) {
        PreModule = F.getParent();
        errs() << "  ---- Malloc Analysis ----\n";
      }
      // Find all places that allocate memory
      bool Changed = false;
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            auto *Callee = CI->getCalledFunction();
            if (Callee && Callee->getName() == "cudaMalloc") {
              auto *AllocPtr = CI->getArgOperand(0);
              AllocPtr->dump();
              if (auto *BCI = dyn_cast<BitCastInst>(AllocPtr)) {
                assert(BCI->getNumOperands() == 1);
                Value *BasePtr = BCI->getOperand(0);
                if (auto *AI = dyn_cast<AllocaInst>(BasePtr)) {
                  Value *BasePtr = AI;
                  if (Info->DataMap.find(BasePtr) == Info->DataMap.end()) {
                    errs() << "new entry device" << "\n";
                    BasePtr->dump();
                    DataEntry *data_entry = new DataEntry;
                    data_entry->base_ptr = BasePtr;
                    data_entry->type = 1; // device space
                    data_entry->pair_entry = NULL;
                    data_entry->reallocated_base_ptr = NULL;
                    Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                  } else
                    errs() << "Error: redundant allocation?\n";
                } else
                  errs() << "Error\n";
              } else
                errs() << "Error\n";
            } else if (Callee && Callee->getName() == "cudaMemcpy") {
            } else if (Callee && Callee->getName() == "cudaMallocManaged") {
            } else if (Callee && Callee->getName() == "malloc") {
              for (auto& U : CI->uses()) {
                User* user = U.getUser();
                if (auto *BCI = dyn_cast<BitCastInst>(user)) {
                  for (auto& UU : BCI->uses()) {
                    User* uuser = UU.getUser();
                    if (auto *SI = dyn_cast<StoreInst>(uuser)) {
                      Value *BasePtr = SI->getOperand(1);
                      if (Info->DataMap.find(BasePtr) == Info->DataMap.end()) {
                        errs() << "new entry host" << "\n";
                        BasePtr->dump();
                        DataEntry *data_entry = new DataEntry;
                        data_entry->base_ptr = BasePtr;
                        data_entry->type = 0; // host space
                        data_entry->pair_entry = NULL;
                        data_entry->reallocated_base_ptr = NULL;
                        data_entry->alias_ptrs.push_back(BCI);
                        data_entry->alias_ptrs.push_back(CI);
                        errs() << " alias entry ";
                        BCI->dump();
                        errs() << " alias entry ";
                        CI->dump();
                        Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                      } else
                        errs() << "Error: redundant allocation?\n";
                    } else
                      errs() << "Error\n";
                  }
                } else
                  errs() << "Error\n";
              }
            }
          }
        }
      }
      return Changed;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;

    DataInfo &getInfo() {
      return *Info;
    }
  };

  struct CudaMemAliasAnalysisPass : public FunctionPass {
    static char ID;
    Module *PreModule = NULL;
    DataInfo *Info;
    CudaMemAliasAnalysisPass() : FunctionPass(ID) {}

    virtual bool runOnFunction(Function &F) {
      //if (F.getName() == "main")
      if (F.getParent() != PreModule) {
        PreModule = F.getParent();
        errs() << "  ---- Memory Alias Analysis ----\n";
      }
      Info = &getAnalysis<CudaMallocAnalysisPass>().getInfo();
      DenseMap<Value*, DataEntry*> *DataMapPtr = &Info->DataMap;
      // Find all pointers that could point to memory related to GPU
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *LI = dyn_cast<LoadInst>(&I)) {
            assert(LI->getNumOperands() >= 1);
            Value *LoadAddr = LI->getOperand(0);
            if (DataMapPtr->find(LoadAddr) != DataMapPtr->end()) {
              if (!Info->getAliasEntry(DataMapPtr->find(LoadAddr)->second, LI)) {
                Info->insertAliasEntry(DataMapPtr->find(LoadAddr)->second, LI);
                errs() << "new alias entry for ";
                LI->dump();
              }
            }
          } else if (auto *BCI = dyn_cast<BitCastInst>(&I)) {
            Value *CastSource = BCI->getOperand(0);
            if (auto *SourceEntry = Info->getAliasEntry(CastSource)) {
              if (!Info->getAliasEntry(SourceEntry, BCI)) {
                Info->insertAliasEntry(SourceEntry, BCI);
                errs() << "new alias entry for ";
                BCI->dump();
              }
            }
          }
        }
      }
      return false;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;
  };

  struct CudaHDMapAnalysisPass : public FunctionPass {
    static char ID;
    Module *PreModule = NULL;
    DataInfo *Info;
    CudaHDMapAnalysisPass() : FunctionPass(ID) {}

    virtual bool runOnFunction(Function &F) {
      //errs() << "I saw a function called " << F.getName() << "!\n";
      //if (F.getName() == "main")
      if (F.getParent() != PreModule) {
        PreModule = F.getParent();
        errs() << "  ---- Host-device Memory Map Analysis ----\n";
      }
      Info = &getAnalysis<CudaMallocAnalysisPass>().getInfo();
      // Relate host memory with device memory
      bool Changed = false;
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            auto *Callee = CI->getCalledFunction();
            if (Callee && Callee->getName() == "cudaMemcpy") {
              Value *HostData, *DeviceData;
              ConstantInt* DCI = dyn_cast<ConstantInt>(CI->getArgOperand(3));
              assert(DCI);
              auto direction = DCI->getValue();
              errs() << "map direction " << direction << "\n";
              if (direction == 1) {
                HostData = CI->getArgOperand(1);
                DeviceData = CI->getArgOperand(0);
              } else if (direction == 2) {
                HostData = CI->getArgOperand(0);
                DeviceData = CI->getArgOperand(1);
              }
              HostData->dump();
              DeviceData->dump();
              auto *HostEntry = Info->getAliasEntry(HostData);
              auto *DeviceEntry = Info->getAliasEntry(DeviceData);
              if (!HostEntry || !DeviceEntry)
                errs() << "Error: did not find alias pointers\n";
              if (HostEntry->pair_entry != NULL || DeviceEntry->pair_entry != NULL)
                if (HostEntry->pair_entry != DeviceEntry || DeviceEntry->pair_entry != HostEntry)
                  errs() << "Error: a data entry is mapped more than once\n";
              HostEntry->pair_entry = DeviceEntry;
              DeviceEntry->pair_entry = HostEntry;
              HostEntry->base_ptr->dump();
              DeviceEntry->base_ptr->dump();
            }
          }
        }
      }
      return Changed;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;
  };

  struct CudaManagedMemPass : public FunctionPass {
    static char ID;
    Module *PreModule = NULL;
    DataInfo *Info;
    CudaManagedMemPass() : FunctionPass(ID) {}

    virtual bool runOnFunction(Function &F) {
      if (F.getParent() != PreModule) {
        PreModule = F.getParent();
        errs() << "  ---- Managed Memory Transformation ----\n";
      }
      Info = &getAnalysis<CudaMallocAnalysisPass>().getInfo();

      bool Changed = false;
      LLVMContext& Ctx = F.getContext();
      auto* I8PPTy = PointerType::get(PointerType::get(Type::getInt8Ty(Ctx), 0), 0);
      Constant* cudaMallocManagedFunc = F.getParent()->getOrInsertFunction("cudaMallocManaged", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx), NULL);
      SmallVector<Instruction*, 8> InstsToDelete;

      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            auto *Callee = CI->getCalledFunction();
            if (Callee && Callee->getName() == "cudaLaunch") {
            } else if (Callee && Callee->getName() == "cudaSetupArgument") {
            } else if (Callee && Callee->getName() == "cudaMalloc") {
              InstsToDelete.push_back(CI);
              Changed = true;
            } else if (Callee && Callee->getName() == "cudaMemcpy") {
              InstsToDelete.push_back(CI);
              Changed = true;
            } else if (Callee && Callee->getName() == "cudaMallocManaged") {
            } else if (Callee && Callee->getName() == "malloc") {
              errs() << "address ";
              CI->dump();
              // Find corresponding entry in data map
              DataEntry *data_entry = NULL;
              Type* AllocType;
              for (auto& U : CI->uses()) {
                User* user = U.getUser();
                if (auto *BCI = dyn_cast<BitCastInst>(user)) {
                  for (auto& UU : BCI->uses()) {
                    User* uuser = UU.getUser();
                    if (auto *SI = dyn_cast<StoreInst>(uuser)) {
                      Value *BasePtr = SI->getOperand(1);
                      assert(Info->DataMap.find(BasePtr) != Info->DataMap.end());
                      data_entry = Info->DataMap.find(BasePtr)->second;
                      AllocType = SI->getOperand(0)->getType();
                    } else
                      errs() << "Error\n";
                  }
                } else
                  errs() << "Error\n";
              }
              assert(data_entry);
              IRBuilder<> builder(CI);
              // Allocate space for the pointer of cudaMallocManaged's first argument
              auto *AI = builder.CreateAlloca(AllocType);
              // Cast it to i8**
              auto *BCI = builder.CreateBitCast(AI, I8PPTy);
              // Insert cudaMallocManaged
              Value* args[] = {BCI, CI->getArgOperand(0)};
              auto *ICI = builder.CreateCall(cudaMallocManagedFunc, args);
              // Load the pointer of allocated space
              auto *LI = builder.CreateLoad(PointerType::get((Type::getInt8Ty(Ctx)), 0), BCI);
              // Remove malloc
              for(auto& U : CI->uses()) {
                User* user = U.getUser();
                user->setOperand(U.getOperandNo(), LI);
              }
              // Correspond host ptr with managed ptr
              assert(data_entry->reallocated_base_ptr == NULL);
              data_entry->reallocated_base_ptr = AI;
              errs() << "reallocate ";
              data_entry->base_ptr->dump();
              errs() << "      with ";
              data_entry->reallocated_base_ptr->dump();

              InstsToDelete.push_back(CI);
              Changed = true;
            }
          }
        }
      }

      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *AI = dyn_cast<AllocaInst>(&I)) {
            if (Info->DataMap.find(AI) != Info->DataMap.end()) {
              if (Info->DataMap.find(AI)->second->type == 1) { // device space
                DataEntry *data_entry = Info->DataMap.find(AI)->second;
                if (data_entry->pair_entry == NULL) {
                  errs() << "Info: this device ptr does not map to host ";
                } else {
                  // Replace usage of device ptrs with managed ptrs
                  Instruction *managed_base_ptr = dyn_cast<Instruction>(data_entry->pair_entry->reallocated_base_ptr);
                  assert(managed_base_ptr);
                  if (managed_base_ptr->getType() != AI->getType())
                    errs() << "Error: not the same type\n";
                  errs() << "replace ";
                  AI->dump();
                  errs() << "   with ";
                  managed_base_ptr->dump();
                  int i = 0;
                  SmallVector<User*, 6> Users;
                  SmallVector<unsigned, 6> UsersNo;
                  for (auto& U : AI->uses()) {
                    User* user = U.getUser();
                    Users.push_back(user);
                    UsersNo.push_back(U.getOperandNo());
                  }
                  while (!Users.empty()) {
                    User* user = Users.back();
                    user->setOperand(UsersNo.back(), managed_base_ptr);
                    Users.pop_back();
                    UsersNo.pop_back();
                  }
                }
                InstsToDelete.push_back(AI);
                Changed = true;
              } else if (Info->DataMap.find(AI)->second->type == 0) { // host space
                DataEntry *data_entry = Info->DataMap.find(AI)->second;
                // Replace usage of host ptrs with managed ptrs
                Instruction *managed_base_ptr = dyn_cast<Instruction>(data_entry->reallocated_base_ptr);
                assert(managed_base_ptr);
                if (managed_base_ptr->getType() != AI->getType())
                  errs() << "Error: not the same type\n";
                errs() << "replace ";
                AI->dump();
                errs() << "   with ";
                managed_base_ptr->dump();
                int i = 0;
                SmallVector<User*, 6> Users;
                SmallVector<unsigned, 6> UsersNo;
                for (auto& U : AI->uses()) {
                  User* user = U.getUser();
                  Users.push_back(user);
                  UsersNo.push_back(U.getOperandNo());
                }
                while (!Users.empty()) {
                  User* user = Users.back();
                  user->setOperand(UsersNo.back(), managed_base_ptr);
                  Users.pop_back();
                  UsersNo.pop_back();
                }
                InstsToDelete.push_back(AI);
                Changed = true;
              }
            }
          } else if (auto *CI = dyn_cast<CallInst>(&I)) {
            auto *Callee = CI->getCalledFunction();
            if (Callee && Callee->getName() == "cudaLaunch") {
            } else if (Callee && Callee->getName() == "cudaSetupArgument") {
            } else if (Callee && Callee->getName() == "cudaMalloc") {
            } else if (Callee && Callee->getName() == "cudaMemcpy") {
            } else if (Callee && Callee->getName() == "cudaMallocManaged") {
            } else if (Callee && Callee->getName() == "malloc") {
            }
          }
        }
      }

      for (auto *I : InstsToDelete)
        I->eraseFromParent();
      return Changed;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;
  };

  void CudaMallocAnalysisPass::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  void CudaMemAliasAnalysisPass::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<CudaMallocAnalysisPass>();

    AU.setPreservesAll();
  }

  void CudaHDMapAnalysisPass::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<CudaMallocAnalysisPass>();
    AU.addRequired<CudaMemAliasAnalysisPass>();

    AU.setPreservesAll();
  }

  void CudaManagedMemPass::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<CudaMallocAnalysisPass>();
    AU.addRequired<CudaMemAliasAnalysisPass>();
    AU.addRequired<CudaHDMapAnalysisPass>();

    AU.setPreservesCFG();
  }
}

char CudaMallocAnalysisPass::ID = 0;
char CudaMemAliasAnalysisPass::ID = 0;
char CudaHDMapAnalysisPass::ID = 0;
char CudaManagedMemPass::ID = 0;

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
static void registerCudaPass(const PassManagerBuilder &,
                         legacy::PassManagerBase &PM) {
  PM.add(new CudaMallocAnalysisPass());
  PM.add(new CudaMemAliasAnalysisPass());
  PM.add(new CudaHDMapAnalysisPass());
  PM.add(new CudaManagedMemPass());
}
static RegisterStandardPasses
  RegisterMyPass(PassManagerBuilder::EP_EarlyAsPossible,
                 registerCudaPass);
