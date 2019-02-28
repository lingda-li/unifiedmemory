#ifndef LLVM_TFG_PASS_H
#define LLVM_TFG_PASS_H

using namespace llvm;

class TargetDistInfo {
public:
  DenseMap<const Instruction*, unsigned> TargetRegionMap; // Target region to index map
  typedef DenseMap<unsigned, double> Target2TargetDisTy;
  Target2TargetDisTy *T2TDis; // Target to target distance

  const double INFDIS = 100.0;

  double getDist(const Instruction *Src, const Instruction *Dst) {
    assert(TargetRegionMap.find(Src) != TargetRegionMap.end());
    assert(TargetRegionMap.find(Dst) != TargetRegionMap.end());
    unsigned SIdx = TargetRegionMap[Src];
    unsigned DIdx = TargetRegionMap[Dst];
    return T2TDis[SIdx][DIdx];
  }

  unsigned getInt(double DD) {
    if (DD > 0x3f)
      return 0x3f;
    return (unsigned)DD;
  }
};

struct TFGPass : public ModulePass {
  static char ID;
  std::vector<Instruction*> TargetRegions;
  unsigned TargetNum;
  std::vector<Function*> FuncHasTarget;
  DenseMap<const BasicBlock*, bool> VisitMap;
  double TotalDiff;
  // Special successor to address function calls
  DenseMap<const BasicBlock*, BasicBlock*> SuccMap;
  // successor for return blocks of a function
  // bool in the pair represents if this is true successor, or bridge successor
  DenseMap<const Function*, std::pair<bool, BasicBlock*>> RetSuccMap;

  typedef DenseMap<const BasicBlock*, double> BBTargetDisTy;
  BBTargetDisTy *PreDis; // At the very beginning of a BB, how soon it supposes to see a target region
  BBTargetDisTy *PosDis; // At the very end of a BB, how soon it supposes to see a target region

  TargetDistInfo Res;

  TFGPass() : ModulePass(ID) {
  }

  virtual bool runOnModule(Module &M);

  bool traverseDFS(BasicBlock *BB);
  void dumpDis(Function *F);
  TargetDistInfo &getT2T() { return Res; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

#endif // LLVM_TFG_PASS_H
