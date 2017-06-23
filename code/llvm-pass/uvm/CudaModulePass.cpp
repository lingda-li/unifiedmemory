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

#define DEBUG_PRINT {errs() << "Error: "<< __LINE__ << "\n";}

namespace {
  struct CudaMallocAnalysisPass : public ModulePass {
    static char ID;
    DataInfo *Info;
    CudaMallocAnalysisPass() : ModulePass(ID) {
      Info = new DataInfo;
    }

    virtual bool runOnModule(Module &M) {
      errs() << "  ---- Malloc Analysis for " << M.getName() << " ----\n";
      // Find all places that allocate memory
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;

        for (auto &BB : F) {
          for (auto &I : BB) {
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              auto *Callee = CI->getCalledFunction();
              if (Callee && Callee->getName() == "cudaMalloc") {
                auto *AllocPtr = CI->getArgOperand(0);
                if (auto *BCI = dyn_cast<BitCastInst>(AllocPtr)) {
                  assert(BCI->getNumOperands() == 1);
                  Value *BasePtr = BCI->getOperand(0);
                  if (auto *AI = dyn_cast<AllocaInst>(BasePtr)) {
                    Value *BasePtr = AI;
                    if (Info->DataMap.find(BasePtr) == Info->DataMap.end()) {
                      errs() << "new device entry ";
                      BasePtr->dump();
                      DataEntry *data_entry = new DataEntry(BasePtr, 1, CI->getArgOperand(1)); // device space
                      Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                    } else
                      errs() << "Error: redundant allocation?\n";
                  } else
                    DEBUG_PRINT
                } else if (auto *AI = dyn_cast<AllocaInst>(AllocPtr)) {
                  Value *BasePtr = AI;
                  if (Info->DataMap.find(BasePtr) == Info->DataMap.end()) {
                    errs() << "new device entry ";
                    BasePtr->dump();
                    DataEntry *data_entry = new DataEntry(BasePtr, 1, CI->getArgOperand(1)); // device space
                    Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                  } else
                    errs() << "Error: redundant allocation?\n";
                } else
                  DEBUG_PRINT
              } else if (Callee && Callee->getName() == "cudaMallocManaged") {
                errs() << "Error: cannot address cudaMallocManaged now\n";
              } else if (Callee && Callee->getName() == "malloc") {
                for (auto& U : CI->uses()) {
                  User* user = U.getUser();
                  if (auto *BCI = dyn_cast<BitCastInst>(user)) {
                    for (auto& UU : BCI->uses()) {
                      User* uuser = UU.getUser();
                      if (auto *SI = dyn_cast<StoreInst>(uuser)) {
                        Value *BasePtr = SI->getOperand(1);
                        if (Info->DataMap.find(BasePtr) == Info->DataMap.end()) {
                          errs() << "new host entry ";
                          BasePtr->dump();
                          DataEntry *data_entry = new DataEntry(BasePtr, 0, CI->getArgOperand(0)); // host space
                          data_entry->ptr_type = SI->getOperand(0)->getType();
                          data_entry->alias_ptrs.push_back(BCI);
                          data_entry->alias_ptrs.push_back(CI);
                          errs() << "  alias entry ";
                          BCI->dump();
                          errs() << "  alias entry ";
                          CI->dump();
                          Info->DataMap.insert(std::make_pair(BasePtr, data_entry));
                        } else
                          errs() << "Error: redundant allocation?\n";
                      } else
                        DEBUG_PRINT
                    }
                  } else if (auto *SI = dyn_cast<StoreInst>(user)) {
                    Value *BasePtr = SI->getOperand(1);
                    if (Info->DataMap.find(BasePtr) == Info->DataMap.end()) {
                      errs() << "new host entry ";
                      BasePtr->dump();
                      DataEntry *data_entry = new DataEntry(BasePtr, 0, CI->getArgOperand(0)); // host space
                      data_entry->ptr_type = SI->getOperand(0)->getType();
                      data_entry->alias_ptrs.push_back(CI);
                      errs() << "  alias entry ";
                      CI->dump();
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
      }
      return false;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;

    DataInfo &getInfo() {
      return *Info;
    }
  };

  struct CudaMemAliasAnalysisPass : public ModulePass {
    static char ID;
    DataInfo *Info;
    CudaMemAliasAnalysisPass() : ModulePass(ID) {}

    virtual bool runOnModule(Module &M) {
      errs() << "  ---- Memory Alias Analysis for " << M.getName() << " ----\n";
      Info = &getAnalysis<CudaMallocAnalysisPass>().getInfo();
      // Find all pointers that could point to memory related to GPU
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;

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
              if (Info->DataMap.find(CastSource) != Info->DataMap.end()) {
                auto *SourceEntry = Info->DataMap.find(CastSource)->second;
                if (Info->tryInsertBaseAliasEntry(SourceEntry, BCI)) {
                  errs() << "  base alias entry ";
                  BCI->dump();
                  NumAlias++;
                }
              }
              if (NumAlias > 1)
                errs() << "Error: a value is alias for multiple entries\n";
            }
          }
        }
      }
      return false;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;
  };

  struct CudaHDMapAnalysisPass : public ModulePass {
    static char ID;
    DataInfo *Info;
    CudaHDMapAnalysisPass() : ModulePass(ID) {}

    virtual bool runOnModule(Module &M) {
      errs() << "  ---- Host-device Memory Map Analysis for " << M.getName() << " ----\n";
      Info = &getAnalysis<CudaMallocAnalysisPass>().getInfo();
      // Relate host memory with device memory
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;

        for (auto &BB : F) {
          for (auto &I : BB) {
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              auto *Callee = CI->getCalledFunction();
              if (Callee && Callee->getName() == "cudaMemcpy") {
                Value *HostData, *DeviceData;
                ConstantInt* DCI = dyn_cast<ConstantInt>(CI->getArgOperand(3));
                assert(DCI);
                auto direction = DCI->getValue();
                if (direction == 1) {
                  HostData = CI->getArgOperand(1);
                  DeviceData = CI->getArgOperand(0);
                } else if (direction == 2) {
                  HostData = CI->getArgOperand(0);
                  DeviceData = CI->getArgOperand(1);
                }
                DataEntry *HostEntry = Info->getAliasEntry(HostData);
                DataEntry *DeviceEntry = Info->getAliasEntry(DeviceData);
                if (!DeviceEntry)
                  errs() << "Error: a device space is not allocated using cudaMalloc\n";
                if (!HostEntry) {
                  errs() << "Info: a host space is not allocated using malloc\n";
                  DeviceEntry->keep_me = true;
                } else {
                  errs() << "map direction " << direction << "\n";
                  HostData->dump();
                  DeviceData->dump();
                  if (HostEntry->pair_entry != NULL || DeviceEntry->pair_entry != NULL)
                    if (HostEntry->pair_entry != DeviceEntry || DeviceEntry->pair_entry != HostEntry)
                      errs() << "Error: a data entry is mapped to different entries\n";
                  HostEntry->pair_entry = DeviceEntry;
                  DeviceEntry->pair_entry = HostEntry;
                  HostEntry->base_ptr->dump();
                  DeviceEntry->base_ptr->dump();
                }
              }
            }
          }
        }
      }
      return false;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;
  };

  struct CudaManagedMemPass : public ModulePass {
    static char ID;
    DataInfo *Info;
    CudaManagedMemPass() : ModulePass(ID) {}

    virtual bool runOnModule(Module &M) {
      errs() << "  ---- Managed Memory Transformation for " << M.getName() << " ----\n";
      Info = &getAnalysis<CudaMallocAnalysisPass>().getInfo();

      bool Changed = false;
      LLVMContext& Ctx = M.getContext();
      auto* I8PPTy = PointerType::get(PointerType::get(Type::getInt8Ty(Ctx), 0), 0);
      Constant* cudaMallocManagedFunc = M.getOrInsertFunction("cudaMallocManaged", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), NULL);
      Constant* cudaDeviceSynchronizeFunc = M.getOrInsertFunction("cudaDeviceSynchronize", Type::getInt32Ty(Ctx), NULL);
      Constant* cudaFreeFunc = M.getOrInsertFunction("cudaFree", Type::getInt32Ty(Ctx), PointerType::get(Type::getInt8Ty(Ctx), 0), NULL);
      Changed = true;
      SmallVector<Instruction*, 8> InstsToDelete;

      for (Function &F : M) {
        if (F.isDeclaration())
          continue;

        for (auto &BB : F) {
          for (auto &I : BB) {
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              auto *Callee = CI->getCalledFunction();
              if (Callee && Callee->getName() == "cudaMalloc") {
                // Find corresponding entry in data map
                auto *AllocPtr = CI->getArgOperand(0);
                DataEntry *data_entry = Info->getBaseAliasEntry(AllocPtr);
                assert(data_entry);
                if (data_entry->keep_me) {
                  errs() << "Info: did not remove ";
                  CI->dump();
                  continue;
                }

                if (data_entry->pair_entry == NULL) {
                  errs() << "Info: this device ptr does not map to host ";
                  data_entry->base_ptr->dump();

                  // Replace cudaMalloc with cudaMallocManaged
                  IRBuilder<> builder(CI);
                  // Insert cudaMallocManaged
                  ConstantInt *ThirdArg = ConstantInt::get(Type::getInt32Ty(Ctx), 1, false);
                  Value* args[] = {CI->getArgOperand(0), CI->getArgOperand(1), ThirdArg};
                  auto *ICI = builder.CreateCall(cudaMallocManagedFunc, args);
                  // Remove malloc
                  for(auto& U : CI->uses()) {
                    User* user = U.getUser();
                    user->setOperand(U.getOperandNo(), ICI);
                  }
                  errs() << "  reallocate ";
                  CI->dump();
                  errs() << "        with ";
                  ICI->dump();
                }
                InstsToDelete.push_back(CI);
                Changed = true;
              } else if (Callee && Callee->getName() == "cudaMemcpy") {
                ConstantInt* DCI = dyn_cast<ConstantInt>(CI->getArgOperand(3));
                assert(DCI);
                auto direction = DCI->getValue();
                Value *DeviceData;
                if (direction == 1)
                  DeviceData = CI->getArgOperand(0);
                else if (direction == 2)
                  DeviceData = CI->getArgOperand(1);
                DataEntry *data_entry = Info->getAliasEntry(DeviceData);
                assert(data_entry);
                if (data_entry->keep_me) {
                  errs() << "Info: did not remove ";
                  CI->dump();
                  continue;
                }

                // Insert cudaDeviceSynchronize() for data transfer from device to host
                // FIXME: potential redundant cudaDeviceSynchronize()
                if (direction == 2) {
                  IRBuilder<> builder(CI);
                  auto *CDSCI = builder.CreateCall(cudaDeviceSynchronizeFunc);
                }
                InstsToDelete.push_back(CI);
                Changed = true;
              } else if (Callee && Callee->getName() == "cudaMallocManaged") {
              } else if (Callee && Callee->getName() == "cudaFree") {
                auto *FreePtr = CI->getArgOperand(0);
                DataEntry *data_entry = Info->getAliasEntry(FreePtr);
                if (!data_entry)
                  continue;
                if (data_entry->keep_me) {
                  errs() << "Info: did not remove ";
                  CI->dump();
                  continue;
                }

                // Delete all cudaFree, since managed memory is released on free
                // FIXME: this only works for pure cudaMalloc version, need to preserve cudaFree for cudaMallocManaged
                InstsToDelete.push_back(CI);
                Changed = true;
              } else if (Callee && Callee->getName() == "malloc") {
                errs() << "address ";
                CI->dump();
                // Find corresponding entry in data map
                DataEntry *data_entry = Info->getAliasEntry(CI);
                assert(data_entry);
                Type* AllocType = data_entry->ptr_type;

                IRBuilder<> builder(CI);
                // Allocate space for the pointer of cudaMallocManaged's first argument
                auto *AI = builder.CreateAlloca(AllocType);
                // Cast it to i8**
                auto *BCI = builder.CreateBitCast(AI, I8PPTy);
                // Insert cudaMallocManaged
                ConstantInt *ThirdArg = ConstantInt::get(Type::getInt32Ty(Ctx), 1, false); // Not sure what is this number
                Value* args[] = {BCI, CI->getArgOperand(0), ThirdArg};
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
                errs() << "  reallocate ";
                data_entry->base_ptr->dump();
                errs() << "        with ";
                data_entry->reallocated_base_ptr->dump();

                InstsToDelete.push_back(CI);
                Changed = true;
              } else if (Callee && Callee->getName() == "free") {
                // Replace it with cudaFree
                errs() << "address ";
                CI->dump();
                IRBuilder<> builder(CI);
                Value* args[] = {CI->getArgOperand(0)};
                auto *CFCI = builder.CreateCall(cudaFreeFunc, args);
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
                  if (data_entry->pair_entry == NULL)
                    continue;
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
                } else
                  DEBUG_PRINT
              }
            } else if (auto *CI = dyn_cast<CallInst>(&I)) {
              auto *Callee = CI->getCalledFunction();
              if (Callee && Callee->getName() == "cudaLaunch") {
              } else if (Callee && Callee->getName() == "cudaSetupArgument") {
              } else if (Callee && Callee->getName() == "cudaMalloc") {
              } else if (Callee && Callee->getName() == "cudaMemcpy") {
                // Potential sanity check here
              } else if (Callee && Callee->getName() == "cudaMallocManaged") {
              } else if (Callee && Callee->getName() == "malloc") {
              }
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
static void registerCudaModulePass(const PassManagerBuilder &,
                         legacy::PassManagerBase &PM) {
  PM.add(new CudaMallocAnalysisPass());
  PM.add(new CudaMemAliasAnalysisPass());
  PM.add(new CudaHDMapAnalysisPass());
  PM.add(new CudaManagedMemPass());
}
static RegisterStandardPasses
  RegisterMyPass(PassManagerBuilder::EP_EnabledOnOptLevel0,
  //RegisterMyPass(PassManagerBuilder::EP_ModuleOptimizerEarly,
                 registerCudaModulePass);
