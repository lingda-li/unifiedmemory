#ifndef LLVM_TFG_PASS_H
#define LLVM_TFG_PASS_H

using namespace llvm;

class TargetDistInfo {
public:
  DenseMap<const Instruction*, unsigned> TargetRegionMap; // Target region to index map
  typedef DenseMap<unsigned, double> Target2TargetDisTy;
  Target2TargetDisTy *T2TDis; // Target to target distance

  const double INFDIS = 100.0;
};

struct TFGPass : public ModulePass {
  static char ID;
  std::vector<Instruction*> TargetRegions;
  unsigned TargetNum;
  DenseMap<const BasicBlock*, bool> VisitMap;
  double TotalDiff;

  typedef DenseMap<const BasicBlock*, double> BBTargetDisTy;
  BBTargetDisTy *PreDis; // At the very beginning of a BB, how soon it supposes to see a target region
  BBTargetDisTy *PosDis; // At the very end of a BB, how soon it supposes to see a target region

  TargetDistInfo Res;

  TFGPass() : ModulePass(ID) {
  }

  virtual bool runOnModule(Module &M);

  bool traverseDFS(BasicBlock *BB, const BranchProbabilityInfo &BPI);
  void dumpDis(Function *F);
  TargetDistInfo &getT2T() { return Res; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

#endif // LLVM_TFG_PASS_H
