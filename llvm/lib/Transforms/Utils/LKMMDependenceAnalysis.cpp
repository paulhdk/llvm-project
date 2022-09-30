//===- LKMMDependenceAnalaysis.cpp - LKMM Deps Implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements two passes to determine whether data, addr and ctrl
/// dependencies were preserved according to the Linux kernel memory model.
///
/// The first pass annotates relevant dependencies in unoptimized IR and the
/// second pass verifies that the dependenices still hold in optimized IR.
///
/// Linux kernel memory model:
/// https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/memory-model/Documentation/explanation.txt
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LKMMDependenceAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <list>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

// FIXME: Brakcets with multiple levels of conditionals

namespace llvm {
static cl::opt<bool> InjectBugs(
    "lkmm-enable-tests",
    cl::desc("Enable the LKMM dependency checker tests. Requires the tests "
             "to be present in the source tree of the kernel being compiled"),
    cl::Hidden, cl::init(false));

static cl::opt<bool> FullToPartialOpt(
    "enable-lkmm-addr-warnings",
    cl::desc("Enable warnings for LKMM addr dependencies based on full to "
             "partial addr dependency conversion"),
    cl::Hidden, cl::init(false));

namespace {
// Avoid the std:: qualifier if possible
using std::list;
using std::pair;
using std::string;
using std::to_string;
using std::unordered_map;
using std::unordered_set;

constexpr StringRef ADBStr = "LKMMDep: address dep begin";
constexpr StringRef ADEStr = "LKMMDep: address dep end";

// FIXME Is there a more elegant way of dealing with duplicate IDs
// (preferably getting eliminating the problem all together)?

// The IDReMap type alias represents the map of IDs to sets of alias IDs
// which verification contexts use for remapping duplicate IDs. Duplicate
// IDs appear when an annotated instruction is duplicated as part of
// optimizations.
using IDReMap = unordered_map<string, unordered_set<string>>;

// Represents a map of IDs to (potential) dependency halfs.
template <typename T> using DepHalfMap = unordered_map<string, T>;

// The DepChain type alias reprsents a dependency chain. Even though the term
// 'chain' suggests ordering, an unordered set fits better here as a), the
// algorithm doesn't require the dep chain to be ordered, and b), an unordered
// set allows constant-time lookup on average.
using DepChain = unordered_set<Value *>;

// The DepChainPair type alias reprsents the pair (DCUnion, DCInter) of
// dependency chains. It exists for every BB the dependency chains of a given
// potential address dependency beginning visit.
using DepChainPair = pair<DepChain, DepChain>;

// The DepChainMap type alias represents the map of BBs to dep chain pairs.
// Such a map exists for every potential addr dep beginning.
using DepChainMap = unordered_map<BasicBlock *, DepChainPair>;

using VerIDSet = unordered_set<string>;

using CallPathStack = list<CallInst *>;

using BBtoBBSetMap = unordered_map<BasicBlock *, unordered_set<BasicBlock *>>;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Checks if an instruction has already been annotated with a given dependency
/// half. This function is used for avoiding multiple annotations of the same
/// dependency. By checking the annotated metadata, this works across modules.
///
/// \param I the instruction whose metadata should be checked
/// \param A the annotation of the potentially new dependency half
///
/// \returns true if the instruction hasn't been annotated with the given
/// dependency half yet.
bool hasAnnotation(Instruction *I, string &A) {
  auto *MDAs = I->getMetadata("annotation");

  if (!MDAs)
    return false;

  for (auto &MDAOp : MDAs->operands())
    if (cast<MDString>(MDAOp.get())->getString().contains(A))
      return true;

  return false;
}

/// Returns a string representation of an instruction's location in the form:
/// <function_name>::<line>:<column>.
///
/// \param I the instruction whose location string should be returned.
/// \param viaFile set to true if the filename should be used instead of the
///  function name
/// \param Entering set to true if the location for a call is being requested
/// which control is entering right now. In that case, line and column info
/// will remain the same, but the function name will be replaced with the
/// called function to make for better reading when outputting broken
/// dependencies.
///
/// \returns a string represenation of \p I's location.
string getInstLocString(Instruction *I, bool ViaFile = false) {
  const llvm::DebugLoc &InstDebugLoc = I->getDebugLoc();

  if (!InstDebugLoc)
    return "no location";

  auto LiAndCol = "::" + to_string(InstDebugLoc.getLine()) + ":" +
                  to_string(InstDebugLoc.getCol());

  if (ViaFile)
    return InstDebugLoc.get()->getFilename().str() + LiAndCol;

  return (I->getFunction()->getName().str()) + LiAndCol;
}

//===----------------------------------------------------------------------===//
// The BFS BB Info Struct
//===----------------------------------------------------------------------===//

struct BFSBBInfo {
  // The BB the two fields below relate to.
  BasicBlock *BB;

  // Denotes the amount of predeceessors which must be visited before the BFS
  // can look at 'BB'.
  unsigned MaxHits;

  // Denotes the amount of predecessors the BFS has already seen (or how many
  // times 'BB' has been 'hit' by an edge from a predecessor).
  unsigned CurrentHits;

  BFSBBInfo(BasicBlock *BB, unsigned MaxHits)
      : BB(BB), MaxHits(MaxHits), CurrentHits(0) {}
};

//===----------------------------------------------------------------------===//
// The Dependency Half Hierarchy
//===----------------------------------------------------------------------===//

class DepHalf {
public:
  enum DepKind {
    DK_AddrBeg,
    DK_VerAddrBeg,
    DK_VerAddrEnd,
  };

  /// Returns the ID of this DepHalf.
  ///
  /// \returns the DepHalf's ID.
  string getID() const;

  /// Returns a string representation of the path the annotation pass took
  /// to discover this DepHalf. The difference is, that the path is expressed in
  /// terms of files and not functions.
  ///
  /// \returns a string representation of the path the annotation pass took
  ///  to discover this DepHalf.
  string getPathToViaFiles() const { return PathToViaFiles; }

  /// Returns a string representation of the path the annotation pass has taken
  /// since seeing this dependency half.
  ///
  /// \returns a string representation of the path the annotation pass took
  ///  since discovering this DepHalf.
  string getPathFrom() const { return PathFrom; }

  /// Sets the PathFrom member. Used for updating PathFrom when returning from
  /// interprocedural analysis.
  ///
  /// \param P the new PathFrom value.
  void setPathFrom(string P) { PathFrom = P; }

  /// Adds a function call to the path taken since discovering this dep half.
  ///
  /// \param FCall the location string of the function call to be added
  /// \param R set to true if control flow is returning to this function call,
  /// set to false if control flow is entering this function call
  void addStepToPathFrom(CallInst *FCall, bool R = false) {
    PathFrom += (getInstLocString(FCall) + (R ? "<-" : "->") +
                 FCall->getCalledFunction()->getName().str() + "()\n");
  }

  DepKind getKind() const { return Kind; }

protected:
  // Instruction which this potential dependency beginning / ending relates to.
  Instruction *const I;

  // An ID which makes this dependency half unique and is used for annotation /
  // verification of dependencies. IDs are represented by a string
  // representation of the calls the BFS took to reach Inst, including inst, and
  // are assumed to be unique within the BFS.
  const string ID;

  // FIXME: can this be removed?
  const string PathToViaFiles;

  string PathFrom;

  DepHalf(Instruction *I, string ID, string PathToViaFiles, DepKind Kind)
      : I(I), ID(ID), PathToViaFiles(PathToViaFiles), PathFrom("\n"),
        Kind(Kind){};

  virtual ~DepHalf() {}

private:
  DepKind Kind;
};

class PotAddrDepBeg : public DepHalf {
public:
  PotAddrDepBeg(Instruction *I, string ID, string PathToViaFiles, Value *V,
                bool FDep = true)
      : PotAddrDepBeg(I, ID, PathToViaFiles, DepChain{V}, FDep,
                      I->getParent()) {}

  PotAddrDepBeg(Instruction *I, string ID, string PathToViaFiles, DepChain DC,
                bool FDep, BasicBlock *BB)
      : DepHalf(I, ID, PathToViaFiles, DK_AddrBeg), DCM(), FDep(FDep) {
    DCM.emplace(BB, DepChainPair{DC, DC});
  }

  // Copy constructor for copying a returned PotAddrDepBeg into the calling
  // context.
  PotAddrDepBeg(PotAddrDepBeg &ADB, BasicBlock *BB, DepChain DC)
      : PotAddrDepBeg(ADB) {
    DCM.clear();
    DCM.emplace(BB, DepChainPair(DC, DC));
  }

  /// Checks whether a DepChainPair is currently at a given BB.
  ///
  /// \param BB the BB to be checked.
  ///
  /// \returns true if the PotAddrDepBeg has dep chains at \p BB.
  bool isAt(BasicBlock *BB) const { return DCM.find(BB) != DCM.end(); }

  DepChainPair getDCsAt(BasicBlock *BB) const {
    return isAt(BB) ? DCM.at(BB) : DepChainPair();
  }

  /// Checks whether this PotAddrDepBeg begins at a given instruction.
  ///
  /// \param I the instruction to be checked.
  ///
  /// \returns true if \p this begins at \p I.
  bool beginsAt(Instruction *I) const { return I == this->I; }

  /// Checks whether all DepChains of this PotAddrDepBeg are at a given
  /// BasicBlock. Useful for interprocedural analysis as it helps determine
  /// whether this PotAddrDepBeg can be completed as a full dependency in a
  /// called function.
  ///
  /// \param BB the BB to be checked.
  ///
  /// \returns true if all DepChains are at \p BB.
  bool areAllDepChainsAt(BasicBlock *BB) {
    return DCM.find(BB) != DCM.end() && DCM.size() == 1;
  };

  /// Updates the dep chain map after the BFS has visitied a given BB with a
  /// given succeeding BB.
  ///
  /// \param BB the BB the BFS just visited.
  /// \param SBB one of BB's successors.
  /// \param BEDsForBB the back edge destination map.
  void progressDCPaths(BasicBlock *BB, BasicBlock *SBB,
                       BBtoBBSetMap &BEDsForBB);

  /// Tries to delete DepChains if possible. Needed for a), keeping track of how
  /// many DepChains are still valid, and b), saving space.
  ///
  /// \param BB the BB the BFS just visited.
  /// \param BEDsForBB the back edge destination for \p BB.
  void deleteDCsAt(BasicBlock *BB, unordered_set<BasicBlock *> &BEDs);

  /// Tries to add a value to the intersection of all DepChains at \p BB.
  ///
  /// \param BB the BB to whose dep chain intersection \p V should be
  ///  added.
  /// \param V the value to be added.
  void addToDCInter(BasicBlock *BB, Value *V);

  /// Tries to add a value to the union of all dep chains at \p BB.
  ///
  /// \param BB the BB to whose dep chain union \p V should be added.
  /// \param V the value to be added.
  void addToDCUnion(BasicBlock *BB, Value *V);

  /// Checks if a counter-argument for this beginning being a full dependency
  /// has been found yet.
  ///
  /// \returns false if a counter-argument for this beginning being a full
  ///  dependency has been found.
  bool canBeFullDependency() const { return FDep; }

  /// This function is called when the BFS is able to prove that any
  /// instruction it encounters after this call is not able to complete a full
  /// dependency to this beginning. This might be the case when the BFS has just
  /// seen a DepChain running through a back edge.
  void cannotBeFullDependencyAnymore() { FDep = false; }

  /// Tries to continue the DepChain with a new value.
  ///
  /// \param I the instruction which is currently being checked.
  /// \param ValCmp the value which might or might not be part of a DepChain.
  /// \param ValAdd the value to add if \p ValCmp is part of a DepChain.
  bool tryAddValueToDepChains(Instruction *I, Value *VCmp, Value *VAdd);

  /// Checks if a value is part of all dep chains starting at this
  /// PotAddrDepBeg. Can be used for checking whether an address dependency
  /// ending marks a full dependency to this PotAddrDepBeg.
  ///
  /// \param BB the BB the BFS is currently visiting.
  /// \param VCmp the value which might or might not be part of all dep
  ///  chains.
  ///
  /// \returns true if \p VCmp is part of all of the beginning's dep chains.
  bool belongsToAllDepChains(BasicBlock *BB, Value *VCmp) const;

  /// Checks if a value is part of any dep chain of an addr dep beginning.
  ///
  /// \param BB the BB the BFS is currently at.
  /// \param VCmp the value which might or might not be part of a dep chain.
  ///
  /// \returns true if \p VCmp belongs to at least one of the beginning's dep
  ///  chains.
  bool belongsToDepChain(BasicBlock *BB, Value *VCmp) const;

  /// Checks if a value is part of some, but not all, dep chains, starting at
  /// this potential beginning. Can be used for checking whether an address
  /// dependency ending marks a partial dependency to this PotAddrDepBeg.
  ///
  /// \param BB the BB the BFS is currently at.
  /// \param VCmp the value which might or might not be part of all dep
  ///  chains.
  ///
  /// \returns true if \p VCmp belongs to at least one, but not all, of this
  ///  PotAddrDepBeg's DepChains.
  bool belongsToSomeNotAllDepChains(BasicBlock *BB, Value *VCmp) const;

  /// Annotates an address dependency from a given ending to this beginning.
  ///
  /// \param ID2 the ID of the ending.
  /// \param I2 the instruction where the address dependency ends.
  /// \param FDep set to true if this is a full address
  ///  dependency.
  void addAddrDep(string ID2, string PathToViaFiles2, Instruction *I2,
                  bool FDep) const;

  /// Resets the DepChainMap to a new state and potentially alteres the
  /// possibility of this PotAddrDepBeg being the beginning of a full
  /// dependency. This functionality is required for interprocedural analysis,
  /// where a DepChain carries over, but should not be cluttered with values
  /// from previous function(s). In the case where not all DepChains of this
  /// PotAddrDepBeg carry over, this cannot mark the beginning of a full
  /// dependency in the called function anymore.
  ///
  /// \param BB The BB to reset the DepChainMap to.
  /// \param FDep The new \p FDep state.
  /// \param DC A DepChain to be used as initial value for the new DepChainPair
  /// at \p BB. In the interprocedural analysis case, \p DC will contain all
  /// function arguments which are part of a DepChain in the calling function.
  void resetDCMTo(BasicBlock *BB, bool FDep, DepChain &DC) {
    this->FDep = FDep;
    DCM.clear();
    DCM.emplace(BB, DepChainPair(DC, DC));
  }

  /// Resets the DepChainMap
  void clearDCMap() { DCM.clear(); }

  /// Returns true if the DepChainMap is completely empty. This is useful for
  /// determining whether a dependency has started in the current function or
  /// was carried over from a previous function where its dependency chain
  /// didn't run into any of the function call's arguments, in which case its
  /// DepChainMap will be completely empty.
  ///
  /// \Returns true if the DepChainMap is completely empty.
  bool isDepChainMapEmpty() const { return DCM.empty(); }

  DepChainMap &&getDCM() { return std::move(DCM); }

  static bool classof(const DepHalf *VDH) {
    return VDH->getKind() == DK_AddrBeg;
  }

  void printDepChainAt(BasicBlock *BB) {
    errs() << "printing DCInter\n";
    for (auto &V : DCM.at(BB).first) {
      V->print(errs());
      errs() << "\n";
    }
  }

private:
  DepChainMap DCM;

  // Denotes whether a matching ending can be annotated as a full dependency. Is
  // set to false if the algorithm encounters something on the way that makes a
  // full dependency impossible, e.g. a backedge.
  bool FDep;

  /// Helper function for progressDCPaths(). Used for computing an intersection
  /// of dep chains.
  ///
  /// \param DCs the list of (BasicBlock, DepChain) pairs wheere the DCs might
  ///  all contain \p V
  /// \param V the value to be checked.
  ///
  /// \returns true if \p V is present in all of \p DCs' dep chains.
  bool depChainsShareValue(list<pair<BasicBlock *, DepChain *>> &DCs,
                           Value *V) const;
};

class VerDepHalf : public DepHalf {
public:
  enum BrokenByType { BrokenDC, FullToPart };

  void setBrokenBy(BrokenByType BB) { BrokenBy = BB; }

  string getBrokenBy() {
    switch (BrokenBy) {
    case BrokenDC:
      return "by breaking the dependency chain";
    case FullToPart:
      return "by converting a partial dependency to a full dependency";
    }
  }

  string const &getParsedDepHalfID() const { return ParsedDepHalfID; }

  string const &getParsedpathTOViaFiles() const { return ParsedPathToViaFiles; }

  Instruction *const &getInst() const { return I; };

  virtual ~VerDepHalf(){};

  static bool classof(const DepHalf *VDH) {
    return VDH->getKind() >= DK_VerAddrBeg && VDH->getKind() <= DK_VerAddrEnd;
  }

  string const &getParsedID() const { return ParsedID; }

protected:
  VerDepHalf(Instruction *I, string ParsedID, string DepHalfID,
             string PathToViaFiles, string ParsedDepHalfID,
             string ParsedPathToViaFiles, DepKind Kind)
      : DepHalf(I, DepHalfID, PathToViaFiles, Kind), ParsedID(ParsedID),
        ParsedDepHalfID(ParsedDepHalfID), ParsedPathToViaFiles{
                                              ParsedPathToViaFiles} {}

private:
  // Shows how this dependency got broken
  BrokenByType BrokenBy;

  // The ID which identifies the two metadata annotations for this dependency.
  const string ParsedID;

  // The PathTo which was attached to the metadata annotation, i.e. the
  // path to I in unoptimised IR.
  const string ParsedDepHalfID;

  const string ParsedPathToViaFiles;
};

class VerAddrDepBeg : public VerDepHalf {
public:
  VerAddrDepBeg(Instruction *I, string ParsedID, string DepHalfID,
                string PathToViaFiles, string ParsedPathTo,
                string ParsedPathToViaFiles)
      : VerDepHalf(I, ParsedID, DepHalfID, PathToViaFiles, ParsedPathTo,
                   ParsedPathToViaFiles, DK_VerAddrBeg) {}

  void setDCP(DepChainPair DCP) { this->DCP = DCP; }
  DepChainPair &getDCP() { return DCP; }

  static bool classof(const DepHalf *VDH) {
    return VDH->getKind() == DK_VerAddrBeg;
  }

private:
  // Gets populated at the end of the BFS and is used for printing the dep
  // chain to users.
  DepChainPair DCP;
};

class VerAddrDepEnd : public VerDepHalf {
public:
  VerAddrDepEnd(Instruction *I, string ParsedID, string DepHalfID,
                string PathToViaFiles, string ParsedDepHalfID,
                string ParsedPathToViaFiles, bool ParsedFDep)
      : VerDepHalf(I, ParsedID, DepHalfID, PathToViaFiles, ParsedDepHalfID,
                   ParsedPathToViaFiles, DK_VerAddrEnd),
        ParsedFDep(ParsedFDep) {}

  const bool &getParsedFullDep() const { return ParsedFDep; }

  static bool classof(const DepHalf *VDH) {
    return VDH->getKind() == DK_VerAddrEnd;
  }

private:
  // Denotes whether the address dependency was annotated as a full dependency
  // or a partial dependency. The value is obtained from the metadata
  // annotation.
  const bool ParsedFDep;
};

using InterprocBFSRes = DepHalfMap<PotAddrDepBeg>;

//===----------------------------------------------------------------------===//
// The BFS Context Hierarchy
//===----------------------------------------------------------------------===//

// A BFSCtx contains all the info the BFS requires to traverse the CFG.

class BFSCtx : public InstVisitor<BFSCtx> {
public:
  enum CtxKind { CK_Annot, CK_Ver };

  CtxKind getKind() const { return Kind; }

  BFSCtx(BasicBlock *BB, CtxKind CK)
      : BB(BB), ADBs(), ReturnedADBs(), CallPath(new CallPathStack()),
        Kind(CK){};

  virtual ~BFSCtx() {
    if (!CallPath->empty())
      CallPath->pop_back();
  }

  /// Runs the BFS algorithm in the given context. This function is called at
  /// the beginning of any function including those which are encountered
  /// through interprocedural analysis.
  void runBFS();

  /// Update all PotAddrDepBegs in the current context after a BasicBlock has
  /// been visited by the BFS. 'Updating' referes to moving the DepChains along
  /// to successors of the BB the BFS just visited.
  ///
  /// \param BB the BB the BFS just visited.
  /// \param SBB one of \p BB's successors
  /// \param BEDsForBB the back edge destination map.
  void progressAddrDepDCPaths(BasicBlock *BB, BasicBlock *SBB,
                              BBtoBBSetMap &BEDsForBB);

  /// Tries to delete unused DepChains for all PotAddrDepBegs in
  /// the current context.
  ///
  /// \param BB the BB the BFS just visited.
  /// \param BEDs the set of back edge destinations for \p BB.
  void deleteAddrDepDCsAt(BasicBlock *BB, unordered_set<BasicBlock *> &BEDs);

  /// Checks if a function call has arguments which are part of DepChains in the
  /// current context. This function is expected to be called at the beginning
  /// of an interprocedural analysis and might reset DepChains if they don't run
  /// through any of the call's arguments.
  ///
  /// \param CI the function call to be checked.
  /// \param ADBs the current ADBs.
  /// \param ADBsForCall the PotAddrDepBegs which will be
  ///  carried over to the called function. This map is left untouched if none
  ///  of the call's arguments are part of a DepChain.
  void handleDependentFunctionArgs(CallInst *CI, BasicBlock *BB);

  //===--------------------------------------------------------------------===//
  // Visitor Functions
  //===--------------------------------------------------------------------===//

  // In order for the BFS to traverse the CFG easily, BFSCtx implements the
  // InstVisitor pattern with a general instruction case, several concrete
  // cases as well as several excluded cases.

  /// Visits a Basic Block in the BFS. Updates the BB field in the current
  /// BFSCtx.
  ///
  /// \param BB the BasicBlock to be visited.
  void visitBasicBlock(BasicBlock &BB);

  /// Visits any instruction which does not match any concrete or excluded case
  /// below. Checks if any dependency chains run through \p I.
  ///
  /// \param I the instruction which did not match any concrete or excluded
  ///  case.
  void visitInstruction(Instruction &I);

  /// Visits a call instruction. Starts interprocedural analysis if possible.
  ///
  /// \param CallI the call instruction.
  void visitCallInst(CallInst &CallI);

  /// Visits a load instruction. Checks if a DepChain runs through \p LoadI
  /// or if \p LoadI marks a (potential) addr dep beginning / ending.
  ///
  /// \param LoadI the load instruction.
  void visitLoadInst(LoadInst &LoadI);

  /// Visits a store instruction. Checks if a DepChain runs through \p
  /// StoreI, \p StoreI redefines a value in an existing DepChain and if it
  /// marks the ending of a dependency.
  ///
  /// \param StoreI the store instruction.
  void visitStoreInst(StoreInst &StoreI);

  /// Visits a PHI instruction. Checks if a DepChain runs through the PHI
  /// instruction, and if that's the case, marks it as conditional if not all
  /// incoming values are part of the DepChain.
  ///
  /// \param PhiI the PHI instruction.
  void visitPHINode(PHINode &PhiI);

  /// Visits a return instruction. If visitReturnInst() is called in an
  /// interprocedural context, it handles the returned potential dependency
  /// beginnings. Assumes that only one ReturnInst exists per function.
  ///
  /// \param ReturnI the return instruction.
  void visitReturnInst(ReturnInst &ReturnI);

  void visitCallBase(CallBase &CB) {
    if (isa<CallInst>(CB))
      visitCallInst(*dyn_cast<CallInst>(&CB));
  }

  // Excluded cases. Instructions of the below types are not handled as they
  // don't affect DepChains.

  /// Excluded.
  void visitAllocaInst(AllocaInst &AllocaI) {}
  /// Excluded.
  void visitFenceInst(FenceInst &FenceI) {}
  /// Excluded.
  void visitAtomicCmpXchgInst(AtomicCmpXchgInst &AtomicCmpXchgI) {}
  /// Excluded.
  void visitAtomicRMWInst(AtomicRMWInst &AtomicRMWI) {}
  /// Excluded.
  void visitFuncletPadInst(FuncletPadInst &FuncletPadI) {}
  /// Excluded.
  void visitTerminator(Instruction &TermI) {}

  // For shared functionality between the visitor functions, BFSCtx
  // provides two internal helper functions.

  /// Shared functionality between visitLoadInstruction() and
  /// visitStoreInstruction().
  ///
  /// \param LoadStoreI the load / store Inst.
  void handleLoadStoreInst(Instruction &LoadStoreI);

protected:
  // The BB the BFS is currently checking.
  BasicBlock *BB;

  // We identify all potential dependency beginnings by the path the passes took
  // to reach them. Keeping track of the path is necessary as it wouldn't be
  // possible to disambiguate two dependency beginnings which are reached by two
  // different calls to the same function - both will begin with the same
  // instruction but will have different dependency chains. To avoid annotating
  // duplicates such as
  //
  // foo()::21 -> bar()::42: READ_ONCE()
  // foo()::21 -> bar()::63: READ_ONCE()
  //
  // and
  //
  // bar()::42: READ_ONCE()
  // bar()::63: READ_ONCE()
  //
  // we check before annotating if we have annotated a dependency
  // before.
  //
  // All potential address dependency beginnings (ADBs) which are being tracked.
  DepHalfMap<PotAddrDepBeg> ADBs;

  // Potential beginnings where the return value is part of the DepChain.
  DepHalfMap<PotAddrDepBeg> ReturnedADBs;

  // The path which the BFS took to reach BB.
  std::shared_ptr<CallPathStack> CallPath;

  /// Prepares a newly created BFSCtx for interprocedural analysis.
  ///
  /// \param BB the first BasicBlock in the called function.
  /// \param CallI the call instruction whose called function begins with \p BB.
  void prepareInterproc(BasicBlock *BB, CallInst *CallI);

  /// Spawns an interprocedural BFS from the current context.
  ///
  /// \param FirstBB the first BasicBlock of the called function.
  /// \param CallI the call instructions which calls the function beginning with
  /// \p FirstBB.
  InterprocBFSRes runInterprocBFS(BasicBlock *FirstBB, CallInst *CallI);

  /// Helper function for handleDependentFunctionArgs(). Checks if all arguments
  /// of a function call are part of all of a given PotAddrDepBeg's DepChains
  /// and adds all depenet arguments to a given set. This function is used for
  /// determining whether a PotAddrDepBeg can carry over as potential full
  /// dependency beginning into the interprocedural analysis.
  ///
  /// \param ADB the PotAddrDepBeg in question.
  /// \param CallI the call instruction whose arguments should be checked
  ///  against \p ADB's dep chains.
  /// \param DependentArgs the set which will contain all dependent function
  ///  arguments on return.
  ///
  /// \returns true if all of \p CallI's arguments are part of all of \p ADB's
  ///  DepChains.
  bool allFunctionArgsPartOfAllDepChains(PotAddrDepBeg &ADB, CallInst *CallI,
                                         unordered_set<Value *> &DependentArgs);

  /// Returns the current limit for interprocedural annotation / verification
  ///
  /// \returns the maximum recursion level
  constexpr unsigned currentLimit() const;

  /// Returns string representation of the full path to an instructions, i.e. a
  /// concatenation of the path of calls the BFS took to discover \p I and the
  /// string representation of \p I's location in source code. Such a string is
  /// supposed to uniquely identify an instruction within the BFS.
  ///
  /// \param I the instruction whose full path should be returned.
  /// \param viaFiles set to true if the path should be expressed in terms of
  ///  filename::line:col instead of functinoName::line:column
  ///
  /// \returns a string representation of \p I's full path.
  string getFullPath(Instruction *I, bool ViaFiles = false) {
    return convertPathToString(ViaFiles) + getInstLocString(I, ViaFiles);
  }

  /// Returns string representation of the full path to an instructions, i.e. a
  /// concatenation of the path of calls the BFS took to discover \p I and the
  /// string representation of \p I's location in source code. Such a string is
  /// supposed to uniquely identify an instruction within the BFS.
  ///
  /// \param I the instruction whose full path should be returned.
  ///
  /// \returns a string representation of \p I's full path.
  string getFullPathViaFiles(Instruction *I) {
    return convertPathToString() + getInstLocString(I);
  }

  /// Converts BFS's call path, i.e. a list of call instructions, to a string.
  ///
  /// \param viaFiles set to true if the filename should be used instead of the
  ///  function name
  ///
  /// \returns a string represenation of \p CallPath.
  string convertPathToString(bool ViaFiles = false) {
    string PathStr{""};

    for (auto &CallI : *CallPath)
      PathStr += (getInstLocString(CallI, ViaFiles) + "  ");

    return PathStr;
  }

  /// Returns the depth, i.e. number of calls, of the interprocedural
  /// analysis.
  ///
  /// \returns the number of call instructions the current BFS has already
  ///  followed.
  unsigned recLevel() { return CallPath->size(); }

  /// Checks whether the BFS can visit a given BB and adds it to the BFSQueue if
  /// this is the case.
  ///
  /// \param SBB the successor the BFS wants to visit.
  /// \param BFSInfo the BFS Info for the current function.
  /// \param BEDs the set of back edge destinations for the current BB.
  ///
  /// \returns true if the BFS has seen all of \p SBB's predecessors.
  bool bfsCanVisit(BasicBlock *SBB,
                   unordered_map<BasicBlock *, BFSBBInfo> &BFSInfo,
                   unordered_set<BasicBlock *> &BEDs);

  /// Parse a dependency annotation string into its individual components.
  ///
  /// \param Annot the dependency annotation string.
  ///
  /// \returns a vector of strings, representing the individual components of
  ///  the annotation string. (ID, type ...)
  void parseDepHalfString(StringRef Annot, SmallVector<string, 5> &AnnotData);

  /// Populates a map of BBs to a set of BBs, representing the back edge
  /// destinations.
  ///
  /// \param BEDsForBB the set to be populated.
  /// \param F the current function.
  void buildBackEdgeMap(BBtoBBSetMap *BEDsForBB, Function *F);

  /// Populates a BFSInfo map at the beginning of a function in a BFS.
  ///
  /// \param BFSInfo the map to be populated.
  /// \param BEDsForBB a map of BBs to their back edge
  ///  destinations in \p F.
  /// \param F the current function.
  void buildBFSInfo(unordered_map<BasicBlock *, BFSBBInfo> *BFSInfo,
                    BBtoBBSetMap *BEDsForBB, Function *F);

  /// Removes back edges from an unordered set of successors, i.e. BasicBlocks.
  ///
  /// \param BB the BB whose successors this function is supposed to look at.
  /// \param BEDsForBB all successors of BB which are connected
  ///  through backedges.
  /// \param SuccessorsWOBackEdges the set of successors without backedges.
  ///  Assumed to be empty.
  void removeBackEdgesFromSuccessors(
      BasicBlock *BB, unordered_set<BasicBlock *> *BEDs,
      unordered_set<BasicBlock *> *SuccessorsWOBackEdges) {
    for (auto *SBB : successors(BB))
      if (BEDs->find(SBB) == BEDs->end())
        SuccessorsWOBackEdges->insert(SBB);
  }

  /// Returns a string representation of how an instruction was inlined in the
  /// form of: <fileN>::<lineN>:<columnN>...<file1>::<line1>:<column1>
  ///
  /// For the algorithm it is important that this representation matches that of
  /// \p convert_path_to_str().
  ///
  /// \param I the instruction whose inline string should be returned.
  ///
  /// \returns a string represenation of how \p I was inlined. The string is
  ///  empty if \p I didn't get inlined.
  string buildInlineString(Instruction *I);

private:
  const CtxKind Kind;
};

class AnnotCtx : public BFSCtx {
public:
  static bool classof(const BFSCtx *C) { return C->getKind() == CK_Annot; }

  AnnotCtx(BasicBlock *BB) : BFSCtx(BB, CK_Annot) {}

  // Creates an AnnotCtx for exploring a called function.
  // FIXME Nearly identical to VerCtx's copy constructor. Can we template this?
  AnnotCtx(AnnotCtx &AC, BasicBlock *BB, CallInst *CallI) : AnnotCtx(AC) {
    prepareInterproc(BB, CallI);
    ReturnedADBs.clear();
  }

  /// Inserts the bugs in the testing functions. Will output to errs() if the
  /// desired annotation can't be found.
  ///
  /// \param F any testing function.
  /// \param IOpCode the type of Instruction whose dependency should be broken.
  ///  Can be Load or Store.
  /// \param AnnotationType the type of annotation to break, i.e. (addr + ctrl)
  ///  dep (beginning + ending).
  void insertBug(Function *F, Instruction::MemoryOps IOpCode,
                 string AnnotationType);
};

class VerCtx : public BFSCtx {
public:
  VerCtx(BasicBlock *BB, std::shared_ptr<DepHalfMap<VerAddrDepBeg>> BrokenADBs,
         std::shared_ptr<DepHalfMap<VerAddrDepEnd>> BrokenADEs,
         std::shared_ptr<IDReMap> RemappedIDs,
         std::shared_ptr<VerIDSet> VerifiedIDs)
      : BFSCtx(BB, CK_Ver), BrokenADBs(BrokenADBs), BrokenADEs(BrokenADEs),
        RemappedIDs(RemappedIDs), VerifiedIDs(VerifiedIDs) {}

  // Creates a VerCtx for exploring a called function.
  // FIXME Nearly identical to AnnotCtx's copy constructor. Can we template
  // this?
  VerCtx(VerCtx &VC, BasicBlock *BB, CallInst *CallI) : VerCtx(VC) {
    prepareInterproc(BB, CallI);
    ReturnedADBs.clear();
  }

  /// Responsible for handling an instruction with at least one '!annotation'
  /// type metadata node. Immediately returns if it doesn't find at least one
  /// dependency annotation.
  ///
  /// \param I the instruction which has at least one dependency annotation
  ///  attached.
  /// \param MDAnnotation a pointer to the \p MDNode containing the dependency
  ///  annotation(s).
  void handleDepAnnotations(Instruction *I, MDNode *MDAnnotation);

  void markIDAsVerified(string ParsedID) {
    auto DelId = [](auto &ID, auto &Bs, auto &Es, auto &RemappedIDs) {
      Bs->erase(ID);
      Es->erase(ID);

      if (RemappedIDs->find(ID) != RemappedIDs->end())
        for (auto const &ReID : RemappedIDs->at(ID)) {
          Bs->erase(ReID);
          Es->erase(ReID);
        }
    };

    DelId(ParsedID, BrokenADBs, BrokenADEs, RemappedIDs);

    VerifiedIDs->insert(ParsedID);
    RemappedIDs->erase(ParsedID);
  }

  static bool classof(const BFSCtx *C) { return C->getKind() == CK_Ver; }

private:
  // Contains all unverified address dependency beginning annotations.
  std::shared_ptr<DepHalfMap<VerAddrDepBeg>> BrokenADBs;
  // Contains all unverified address dependency ending annotations.
  std::shared_ptr<DepHalfMap<VerAddrDepEnd>> BrokenADEs;

  // All remapped IDs which were discovered from the current root function.
  std::shared_ptr<IDReMap> RemappedIDs;

  // Contains all IDs which have been verified in the current module.
  std::shared_ptr<VerIDSet> VerifiedIDs;

  /// Responsible for handling a single address dependency ending annotation.
  ///
  /// \param ID the ID of the address dependency.
  /// \param I the instruction the annotation was attached to, i.e. the
  ///  instruction where the address dependency ends.
  /// \param ParsedPathTo the path the annotation pass took to discover
  ///  \p Inst.
  /// \param ParsedFullDep set to true if the address dependency was annotated
  ///  as a full dependency.
  ///
  /// \returns true if the address dependency could be verified.
  bool handleAddrDepID(string const &ID, Instruction *I, string &ParsedPathTo,
                       string &ParsedPathToViaFiles, bool ParsedFullDep);

  /// Responsible for updating an ID if the verification pass has encountered it
  /// before. Will add the updated ID to \p RemappedIDs.
  ///
  /// \param ID a reference to the ID which should be updated.
  void updateID(string &ID) {
    if (RemappedIDs->find(ID) == RemappedIDs->end()) {
      RemappedIDs->emplace(ID, unordered_set<string>{ID + "-#1"});
      ID = ID + "-#1";
    } else {
      auto S = RemappedIDs->at(ID).size();
      RemappedIDs->at(ID).insert(ID + "-#" + to_string(S + 1));
      ID = ID + "-#" + to_string(S + 1);
    }
  }
};

//===----------------------------------------------------------------------===//
// DepHalf Implementations
//===----------------------------------------------------------------------===//

string DepHalf::getID() const {
  if (isa<PotAddrDepBeg>(this))
    return ID;
  if (const auto *VDH = dyn_cast<VerDepHalf>(this))
    return VDH->getParsedID();
  llvm_unreachable("unhandled case in getID");
}

//===----------------------------------------------------------------------===//
// PotAddrDepBeg Implementations
//===----------------------------------------------------------------------===//

void PotAddrDepBeg::progressDCPaths(BasicBlock *BB, BasicBlock *SBB,
                                    BBtoBBSetMap &BEDsForBB) {
  if (!isAt(BB))
    return;

  if (!isAt(SBB))
    DCM.insert({SBB, DepChainPair{}});

  auto &SDCP = DCM.at(SBB);

  // BB might not be the only predecessor of SBB. Build a list of all
  // preceeding dep chains.
  list<pair<BasicBlock *, DepChain *>> PDCs;

  // Populate PDCs and DCUnion.
  for (auto *Pred : predecessors(SBB)) {
    // If Pred is connected to SBB via a back edge, skip.
    if (BEDsForBB.at(Pred).find(SBB) != BEDsForBB.at(Pred).end())
      continue;

    // If the DepChain don't run through Pred, skip.
    if (!isAt(Pred))
      continue;

    // Previous, i.e. Pred's, DepChainPair.
    auto &PDCP = DCM.at(Pred);

    // Insert preceeding DCunion into succeeding DCUnion.
    SDCP.second.insert(PDCP.second.begin(), PDCP.second.end());

    // Save preceeding DepChain for intersection.
    PDCs.emplace_back(Pred, &PDCP.first);
  }

  // FIXME When this if doesn't fire, depChainsShareValue() will make one
  //  unneccesary loop iteration.

  // If PDCs is empty, we are at the function entry:
  if (PDCs.empty()) {
    // 1. Intiialise PDCs with current DCUnion.
    SDCP.second.insert(DCM.at(BB).second.begin(), DCM.at(BB).second.end());

    // 2. Initialise SDCP's DCUnion with the current DCUnion.
    PDCs.emplace_back(BB, &DCM.at(BB).first);
  }

  // Update DCInter. Only add a value if it's present in every
  // preceeding DepChain.
  DepChain FixedDC = *PDCs.begin()->second;

  // If SDCP's DCInter isn't empty. Intersect succeeding DCInter with
  // current DCInter.
  if (!SDCP.first.empty()) {
    FixedDC = SDCP.first;
    SDCP.first.clear();
  }

  // Compute intersection of all dep chains leading to SBB.
  for (auto &V : FixedDC)
    // Add a value if it is present in all preceeding DepChains.
    if (depChainsShareValue(PDCs, V))
      SDCP.first.insert(V);
}

void PotAddrDepBeg::deleteDCsAt(BasicBlock *BB,
                                unordered_set<BasicBlock *> &BEDs) {
  if (!isAt(BB))
    return;

  if (!BEDs.empty() || isa<ReturnInst>(BB->getTerminator())) {
    // Keep the entry in DCM to account for 'dead' DepChain, but clear
    // them to save space.
    DCM.at(BB).first.clear();
    DCM.at(BB).second.clear();
  } else
    // If there's no dead DepChain, erase the DCM entry for the current BB.
    DCM.erase(BB);
}

void PotAddrDepBeg::addToDCInter(BasicBlock *BB, Value *V) {
  if (!isAt(BB))
    return;

  DCM.at(BB).first.insert(V);
}

void PotAddrDepBeg::addToDCUnion(BasicBlock *BB, Value *V) {
  if (!isAt(BB))
    return;

  DCM.at(BB).second.insert(V);
}

bool PotAddrDepBeg::tryAddValueToDepChains(Instruction *I, Value *VCmp,
                                           Value *VAdd) {
  if (!isAt(I->getParent()))
    return false;

  if (isa<ConstantData>(VAdd))
    return false;

  auto Ret(false);

  auto &DCP = DCM.at(I->getParent());

  auto &DCInter = DCP.first;
  auto &DCUnion = DCP.second;

  // Add to DCinter and account for redefinition.
  if (DCInter.find(VCmp) != DCInter.end()) {
    DCInter.insert(VAdd);
    Ret = true;
  }

  // Add to DCUnion and account for redefinition
  if (DCUnion.find(VCmp) != DCUnion.end()) {
    DCUnion.insert(VAdd);
    Ret = true;
  }

  return Ret;
}

bool PotAddrDepBeg::belongsToAllDepChains(BasicBlock *BB, Value *VCmp) const {
  if (DCM.find(BB) == DCM.end())
    return false;

  auto &DCInter = DCM.at(BB).first;

  return DCInter.find(VCmp) != DCInter.end() && DCM.size() == 1;
}

bool PotAddrDepBeg::belongsToDepChain(BasicBlock *BB, Value *VCmp) const {
  if (DCM.find(BB) == DCM.end())
    return false;

  auto &DCUnion = DCM.at(BB).second;

  return DCUnion.find(VCmp) != DCUnion.end();
}

bool PotAddrDepBeg::belongsToSomeNotAllDepChains(BasicBlock *BB,
                                                 Value *VCmp) const {
  if (DCM.find(BB) == DCM.end())
    return false;

  return !belongsToAllDepChains(BB, VCmp) && belongsToDepChain(BB, VCmp);
}

void PotAddrDepBeg::addAddrDep(string ID2, string PathToViaFiles2,
                               Instruction *I2, bool FDep) const {
  auto DepID = getID() + PathFrom + ID2;

  auto BeginAnnotation = ADBStr.str() + ",\n" + DepID + ",\n" + getID() + ",\n";
  auto EndAnnotation = ADEStr.str() + ",\n" + DepID + ",\n" + ID2 + ",\n";

  // We only annotate if we haven't annotated this exact dependency before.
  if (hasAnnotation(I, BeginAnnotation) && hasAnnotation(I2, EndAnnotation))
    return;

  BeginAnnotation += getPathToViaFiles() + ";";
  EndAnnotation += PathToViaFiles2 + ",\n" + to_string(FDep) + ";";

  I->addAnnotationMetadata(BeginAnnotation);
  I2->addAnnotationMetadata(EndAnnotation);
}

bool PotAddrDepBeg::depChainsShareValue(
    list<pair<BasicBlock *, DepChain *>> &DCs, Value *V) const {
  for (auto &DCP : DCs)
    if (DCP.second->find(V) == DCP.second->end())
      return false;

  return true;
}

//===----------------------------------------------------------------------===//
// BFSCtx Implementations
//===----------------------------------------------------------------------===//

void BFSCtx::runBFS() {
  // BB might be null when runBFS gets called for a function with external
  // linkage for example
  if (!BB)
    return;

  // Maps a BB to the set of its back edge destinations (BEDs).
  BBtoBBSetMap BEDsForBB;

  buildBackEdgeMap(&BEDsForBB, BB->getParent());

  unordered_map<BasicBlock *, BFSBBInfo> BFSInfo;

  buildBFSInfo(&BFSInfo, &BEDsForBB, BB->getParent());

  std::queue<BasicBlock *> BFSQueue = {};

  BFSQueue.push(BB);

  while (!BFSQueue.empty()) {
    auto &BB = BFSQueue.front();

    visit(BB);

    unordered_set<BasicBlock *> SuccessorsWOBackEdges{};

    removeBackEdgesFromSuccessors(BB, &BEDsForBB.at(BB),
                                  &SuccessorsWOBackEdges);

    for (auto &SBB : SuccessorsWOBackEdges) {
      if (bfsCanVisit(SBB, BFSInfo, BEDsForBB.at(SBB)))
        BFSQueue.push(SBB);

      progressAddrDepDCPaths(BB, SBB, BEDsForBB);
    }

    deleteAddrDepDCsAt(BB, BEDsForBB.at(BB));

    BFSQueue.pop();
  }
}

void BFSCtx::progressAddrDepDCPaths(BasicBlock *BB, BasicBlock *SBB,
                                    BBtoBBSetMap &BEDsForBB) {
  for (auto &ADBP : ADBs)
    ADBP.second.progressDCPaths(BB, SBB, BEDsForBB);
}

void BFSCtx::deleteAddrDepDCsAt(BasicBlock *BB,
                                unordered_set<BasicBlock *> &BEDs) {
  for (auto &ADBP : ADBs)
    ADBP.second.deleteDCsAt(BB, BEDs);
}

void BFSCtx::handleDependentFunctionArgs(CallInst *CallI, BasicBlock *BB) {
  DepChain DependentArgs;
  auto *CalledF = CallI->getCalledFunction();

  for (auto &ADBP : ADBs) {
    auto &ParsedID = ADBP.first;
    auto &ADB = ADBP.second;

    bool FDep = allFunctionArgsPartOfAllDepChains(ADB, CallI, DependentArgs);

    // Instead of deleting an ADB if it doesn't run into a function, we keep it
    // with an empty DCM, thereby ensuring that no further items can be added to
    // the DepChain until control flow returns to this function, but still
    // allowing an ending to be mapped to it when verifying.
    if (DependentArgs.empty())
      ADB.clearDCMap();
    else {
      // Mark dependencies through external or empty functions as trivially
      // verified
      if (CalledF->hasExternalLinkage() || CalledF->isIntrinsic() ||
          CalledF->isVarArg() || CalledF->empty() || CallI->isIndirectCall()) {
        if (auto *VC = dyn_cast<VerCtx>(this))
          VC->markIDAsVerified(ParsedID);
      } else {
        ADB.resetDCMTo(BB, FDep, DependentArgs);
        ADB.addStepToPathFrom(CallI);
      }
    }

    DependentArgs.clear();
  }
}

void BFSCtx::prepareInterproc(BasicBlock *BB, CallInst *CallI) {
  handleDependentFunctionArgs(CallI, BB);

  CallPath->push_back(CallI);

  this->BB = BB;
}

// FIXME Duplciate code
InterprocBFSRes BFSCtx::runInterprocBFS(BasicBlock *FirstBB, CallInst *CallI) {
  if (auto *AC = dyn_cast<AnnotCtx>(this)) {
    AnnotCtx InterprocCtx = AnnotCtx(*AC, FirstBB, CallI);
    InterprocCtx.runBFS();
    return InterprocBFSRes(std::move(InterprocCtx.ReturnedADBs));
  }
  if (auto *VC = dyn_cast<VerCtx>(this)) {
    VerCtx InterprocCtx = VerCtx(*VC, FirstBB, CallI);
    InterprocCtx.runBFS();
    return InterprocBFSRes(std::move(InterprocCtx.ReturnedADBs));
  }
  llvm_unreachable("Called runInterprocBFS() with no BFSCtx child.");
}

constexpr unsigned BFSCtx::currentLimit() const {
  if (isa<AnnotCtx>(this))
    return 3;
  if (isa<VerCtx>(this))
    return 4;
  llvm_unreachable("called currentLimit with unhandled subclass.");
}

bool BFSCtx::allFunctionArgsPartOfAllDepChains(
    PotAddrDepBeg &ADB, CallInst *CallI,
    unordered_set<Value *> &DependentArgs) {
  bool FDep = ADB.canBeFullDependency();
  auto *CalledF = CallI->getCalledFunction();

  if (!ADB.areAllDepChainsAt(BB))
    FDep = false;

  for (unsigned Ind = 0; Ind < CallI->arg_size(); ++Ind) {
    auto *VCmp = CallI->getArgOperand(Ind);

    if (!ADB.belongsToDepChain(BB, VCmp))
      continue;

    if (!ADB.belongsToAllDepChains(BB, CallI->getArgOperand(Ind)))
      FDep = false;

    if (!CalledF->isVarArg())
      DependentArgs.insert(CalledF->getArg(Ind));
    else
      DependentArgs.insert(VCmp);
  }

  return FDep;
}

bool BFSCtx::bfsCanVisit(BasicBlock *SBB,
                         unordered_map<BasicBlock *, BFSBBInfo> &BFSInfo,
                         unordered_set<BasicBlock *> &BEDs) {
  auto &NextMaxHits{BFSInfo.at(SBB).MaxHits};
  auto &NextCurrentHits{BFSInfo.at(SBB).CurrentHits};

  if (NextMaxHits == 0 || ++NextCurrentHits == NextMaxHits)
    return true;

  return false;
}

void BFSCtx::parseDepHalfString(StringRef Annot,
                                SmallVector<string, 5> &AnnotData) {
  if (!Annot.consume_back(";"))
    return;

  while (!Annot.empty()) {
    auto P = Annot.split(",");
    AnnotData.push_back(P.first.str());
    Annot = P.second;
  }
}

void BFSCtx::buildBackEdgeMap(BBtoBBSetMap *BEDsForBB, Function *F) {
  // Initialise backEdges with all BB's and an empty set of back-edge
  // successors.
  for (auto &BB : *F)
    BEDsForBB->insert({&BB, {}});

  SmallVector<pair<const BasicBlock *, const BasicBlock *>> BackEdgeVector;
  FindFunctionBackedges(*F, BackEdgeVector);

  for (auto &BackEdge : BackEdgeVector) {
    BEDsForBB->at(const_cast<BasicBlock *>(BackEdge.first))
        .insert(const_cast<BasicBlock *>(BackEdge.second));
  }
}

void BFSCtx::buildBFSInfo(unordered_map<BasicBlock *, BFSBBInfo> *BFSInfo,
                          BBtoBBSetMap *BEDsForBB, Function *F) {
  for (auto &BB : *F) {
    unsigned MaxHits{pred_size(&BB)};

    // Every incoming edge which is a back edge, i.e. closes a loop, is not
    // considered in MaxHits.
    for (auto *Pred : predecessors(&BB))
      if (BEDsForBB->at(Pred).find(&BB) != BEDsForBB->at(Pred).end())
        --MaxHits;

    BFSInfo->emplace(&BB, BFSBBInfo(&BB, MaxHits));
  }
}

string BFSCtx::buildInlineString(Instruction *I) {
  auto InstDebugLoc = I->getDebugLoc();

  if (!InstDebugLoc)
    return "no debug loc when building inline string";

  string InlinePath = InstDebugLoc.get()->getFilename().str() +
                      "::" + to_string(InstDebugLoc.getLine()) + ":" +
                      to_string(InstDebugLoc.getCol());

  auto *InlinedAt = InstDebugLoc.getInlinedAt();

  while (InlinedAt) {
    // Column.
    InlinePath = ":" + to_string(InlinedAt->getColumn()) + "  " + InlinePath;
    // Line.
    InlinePath = "::" + to_string(InlinedAt->getLine()) + InlinePath;
    // File name.
    InlinePath = InlinedAt->getFilename().str() + InlinePath;

    // Move to next InlinedAt if it exists.
    InlinedAt = InlinedAt->getInlinedAt();
  }

  return InlinePath;
}

//===----------------------------------------------------------------------===//
// BFSCtx Visitor Functions
//===----------------------------------------------------------------------===//

void BFSCtx::visitBasicBlock(BasicBlock &BB) { this->BB = &BB; }

void BFSCtx::visitInstruction(Instruction &I) {
  for (auto &ADBP : ADBs)
    for (unsigned Ind = 0; Ind < I.getNumOperands(); ++Ind)
      ADBP.second.tryAddValueToDepChains(&I, I.getOperand(Ind),
                                         cast<Value>(&I));
}

void BFSCtx::visitCallInst(CallInst &CallI) {
  auto *CalledF = CallI.getCalledFunction();

  if (!CalledF)
    return;

  if (recLevel() > currentLimit())
    return;

  // Skip intrinsics, external functions ... of type void
  if (CalledF->empty() && CalledF->getReturnType()->isVoidTy())
    return;

  InterprocBFSRes Ret;

  auto *FirstBB = CalledF->empty() ? nullptr : &*CalledF->begin();

  // FIXME redundant checks?
  if (isa<AnnotCtx>(this))
    Ret = runInterprocBFS(FirstBB, &CallI);
  else if (isa<VerCtx>(this))
    Ret = runInterprocBFS(FirstBB, &CallI);

  auto &RADBsFromCall = Ret;

  // Handle returned addr deps.
  for (auto &RADBP : RADBsFromCall) {
    auto &ID = RADBP.first;
    auto &RADB = RADBP.second;

    if (ADBs.find(ID) != ADBs.end()) {
      auto &ADB = ADBs.at(ID);

      if (RADB.isDepChainMapEmpty())
        continue;

      ADB.addToDCUnion(BB, cast<Value>(&CallI));

      ADB.setPathFrom(RADB.getPathFrom());

      // If not all dep chains from the beginning got returned, FDep might
      // have changed.
      if (RADB.canBeFullDependency())
        ADB.addToDCInter(BB, cast<Value>(&CallI));
      else
        ADB.cannotBeFullDependencyAnymore();

      ADBs.at(ID).addStepToPathFrom(&CallI, true);
    } else if (RADB.isDepChainMapEmpty()) {
      ADBs.emplace(ID, RADB);
    } else {
      ADBs.emplace(ID, PotAddrDepBeg(RADB, BB, DepChain{cast<Value>(&CallI)}));
      // FIXME: can this identical second call be avoid by rearanging the
      // branches?
      ADBs.at(ID).addStepToPathFrom(&CallI, true);
    }
  }
}

void BFSCtx::visitLoadInst(LoadInst &LoadI) {
  handleLoadStoreInst(LoadI);

  if (!LoadI.isVolatile())
    return;

  if (!isa<AnnotCtx>(this))
    return;

  auto ID = getFullPath(&LoadI);

  if (ADBs.find(ID) == ADBs.end())
    ADBs.emplace(ID,
                 PotAddrDepBeg(&LoadI, getInstLocString(&LoadI),
                               getFullPath(&LoadI, true), cast<Value>(&LoadI)));
}

void BFSCtx::visitStoreInst(StoreInst &StoreI) { handleLoadStoreInst(StoreI); }

void BFSCtx::visitPHINode(PHINode &PhiI) {
  for (auto &ADBP : ADBs) {
    auto &ADB = ADBP.second;
    for (unsigned Ind = 0; Ind < PhiI.getNumIncomingValues(); ++Ind)
      if (!ADB.tryAddValueToDepChains(&PhiI, PhiI.getIncomingValue(Ind),
                                      cast<Value>(&PhiI)))
        ADB.cannotBeFullDependencyAnymore();
  }
}

void BFSCtx::visitReturnInst(ReturnInst &RetI) {
  auto *RetVal = RetI.getReturnValue();

  if (!recLevel())
    return;

  for (auto &ADBP : ADBs) {
    auto &ADB = ADBP.second;

    if (!RetVal || ADB.isDepChainMapEmpty() ||
        !ADB.belongsToDepChain(BB, RetVal)) {
      ReturnedADBs.emplace(ADBP.first, ADB);
      ReturnedADBs.at(ADBP.first).clearDCMap();
    } else {
      if (ADB.belongsToSomeNotAllDepChains(BB, RetVal))
        ADB.cannotBeFullDependencyAnymore();

      ReturnedADBs.emplace(ADBP.first, ADB);
    }
  }
}

void BFSCtx::handleLoadStoreInst(Instruction &I) {
  if (auto *VC = dyn_cast<VerCtx>(this))
    if (auto *MDAnnotation = I.getMetadata("annotation"))
      VC->handleDepAnnotations(&I, MDAnnotation);

  auto *VAdd = isa<StoreInst>(I) ? I.getOperand(1) : cast<Value>(&I);

  // Operand which can end an address dependency
  auto *VEnd = isa<StoreInst>(I) ? I.getOperand(1) : I.getOperand(0);

  for (auto &ADBP : ADBs) {
    auto &ADB = ADBP.second;

    if (I.isVolatile()) {
      if (isa<AnnotCtx>(this)) {
        if (ADB.belongsToAllDepChains(BB, VEnd) && ADB.canBeFullDependency())
          ADB.addAddrDep(getInstLocString(&I), getFullPath(&I, true), &I, true);
        else if (ADB.belongsToSomeNotAllDepChains(BB, VEnd))
          ADB.addAddrDep(getInstLocString(&I), getFullPath(&I, true), &I,
                         false);
      }
    }

    ADB.tryAddValueToDepChains(&I, I.getOperand(0), VAdd);
  }
}

//===----------------------------------------------------------------------===//
// AnnotCtx Implementations
//===----------------------------------------------------------------------===//

void AnnotCtx::insertBug(Function *F, Instruction::MemoryOps IOpCode,
                         string AnnotationType) {
  Instruction *InstWithAnnotation = nullptr;

  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    if (auto *MDN = I->getMetadata("annotation")) {
      for (auto &Op : MDN->operands()) {
        if (isa<MDString>(Op) &&
            cast<MDString>(Op)->getString().contains(AnnotationType) &&
            (I->getOpcode() == IOpCode)) {
          InstWithAnnotation = &*I;
          break;
        }
      }

      if (InstWithAnnotation)
        break;
    }
  }

  if (!InstWithAnnotation) {
    errs() << "No annotations in testing function " << F->getName()
           << ". No bug was inserted.\n";
    return;
  }

  auto &InstContext = InstWithAnnotation->getContext();

  auto *BugVal1 = new AllocaInst(Type::getInt32Ty(InstContext), 0,
                                 string("BugVal1"), &*F->begin()->begin());

  auto *BugVal2 = new AllocaInst(Type::getInt32PtrTy(InstContext), 0,
                                 string("BugVal2"), &*F->begin()->begin());

  auto *S1 = new StoreInst(ConstantInt::get(Type::getInt32Ty(InstContext), 42),
                           cast<Value>(BugVal1), true, InstWithAnnotation);

  new StoreInst(BugVal1, cast<Value>(BugVal2), true,
                S1->getNextNonDebugInstruction());

  if (AnnotationType == "dep begin") {
    for (auto InstIt = InstWithAnnotation->getIterator(),
              InstEnd = InstWithAnnotation->getParent()->end();
         InstIt != InstEnd; ++InstIt)
      if ((InstIt->getOpcode() == Instruction::Store) &&
          (InstIt->getOperand(0) == cast<Value>(InstWithAnnotation))) {
        InstWithAnnotation = &*InstIt;
        break;
      }

    // Replace the source of the store to break the dependency chain.
    InstWithAnnotation->setOperand(0, BugVal1);
  } else {
    if (IOpCode == Instruction::Load) {
      // Update the source of our annotated load to be the global BugVal.
      InstWithAnnotation->setOperand(0, BugVal1);
      // Set a new name.
      InstWithAnnotation->setName("new_ending");
    } else
      InstWithAnnotation->setOperand(1, BugVal1);
  }
}

//===----------------------------------------------------------------------===//
// VerCtx Implementations
//===----------------------------------------------------------------------===//

bool VerCtx::handleAddrDepID(string const &ID, Instruction *I,
                             string &ParsedDepHalfID,
                             string &ParsedPathToViaFiles, bool ParsedFullDep) {
  auto *VCmp = isa<StoreInst>(I) ? I->getOperand(1) : I->getOperand(0);

  if (ADBs.find(ID) != ADBs.end()) {
    auto &ADB = ADBs.at(ID);

    // We only add the current annotation as a broken ending if the current
    // BFS has seen the beginning ID. If we were to add unconditionally, we
    // might add endings which aren't actually reachable by the corresponding.
    // Such cases may be false positivies.

    // Check for fully broken dependency chain
    if (!ADB.belongsToDepChain(BB, VCmp)) {
      auto &VADB = BrokenADBs->at(ID);
      auto VADE =
          VerAddrDepEnd(I, ID, getFullPath(I), getFullPath(I, true),
                        ParsedDepHalfID, ParsedPathToViaFiles, ParsedFullDep);

      VADB.setDCP(ADB.getDCsAt(BB));
      VADE.setBrokenBy(VerDepHalf::BrokenByType::BrokenDC);

      BrokenADEs->emplace(ID, std::move(VADE));
      return false;
    }

    // Check for full to partial conversion iff cl opt is specified
    if (FullToPartialOpt) {
      if (ParsedFullDep && ADB.belongsToSomeNotAllDepChains(BB, VCmp)) {
        BrokenADEs->emplace(ID,
                            VerAddrDepEnd(I, ID, getFullPath(I),
                                          getFullPath(I, true), ParsedDepHalfID,
                                          ParsedPathToViaFiles, ParsedFullDep));
        BrokenADEs->at(ID).setBrokenBy(VerDepHalf::BrokenByType::FullToPart);
        return false;
      }
    }
  } else
    return false;

  return true;
}

void VerCtx::handleDepAnnotations(Instruction *I, MDNode *MDAnnotation) {
  // For non-greedy verification
  unordered_set<int> AddedEndings{};

  SmallVector<string, 5> AnnotData;

  for (auto &MDOp : MDAnnotation->operands()) {
    auto CurrentDepHalfStr = cast<MDString>(MDOp.get())->getString();

    if (!CurrentDepHalfStr.contains("LKMMDep"))
      continue;

    AnnotData.clear();

    parseDepHalfString(CurrentDepHalfStr, AnnotData);

    auto &ParsedDepHalfTypeStr = AnnotData[0];
    auto &ParsedID = AnnotData[1];

    if (VerifiedIDs->find(ParsedID) != VerifiedIDs->end())
      continue;

    auto &ParsedDepHalfID = AnnotData[2];
    auto &ParsedPathToViaFiles = CurrentDepHalfStr.contains("ctrl dep begin")
                                     ? AnnotData[4]
                                     : AnnotData[3];

    // Figure out if this is the instruction we originally attached the
    // annotation to. If it isn't, conintue.
    auto InlinePath = buildInlineString(I);

    if (!InlinePath.empty() && !ParsedPathToViaFiles.empty())
      if ((InlinePath.length() > ParsedPathToViaFiles.length()) ||
          // Does ParsedPathTo end with InlinePath?
          ParsedPathToViaFiles.compare(ParsedPathToViaFiles.length() -
                                           InlinePath.length(),
                                       InlinePath.length(), InlinePath) != 0)
        continue;

    if (ParsedDepHalfTypeStr.find("begin") != string::npos) {
      if (ADBs.find(ParsedID) != ADBs.end())
        updateID(ParsedID);

      // For tracking the dependency chain, always add a PotAddrDepBeg
      // beginning, no matter if the annotation concerns an address dependency
      // or control dependency beginning.
      ADBs.emplace(ParsedID,
                   PotAddrDepBeg(I, getFullPath(I), getFullPath(I, true),
                                 cast<Value>(I)));

      if (ParsedDepHalfTypeStr.find("address dep") != string::npos)
        // Assume broken until proven wrong.
        BrokenADBs->emplace(ParsedID,
                            VerAddrDepBeg(I, ParsedID, getFullPath(I),
                                          getFullPath(I, true), ParsedDepHalfID,
                                          ParsedPathToViaFiles));
    } else if (ParsedDepHalfTypeStr.find("end") != string::npos) {
      // If we are able to verify one pair in
      // {ORIGINAL_ID} \cup REMAPPED_IDS.at(ORIGINAL_ID) x {ORIGINAL_ID}
      // We consider ORIGINAL_ID verified; there only exists one dependency in
      // unoptimised IR, hence we only look for one dependency in optimised
      // IR.
      if (ParsedDepHalfTypeStr.find("address dep") != string::npos) {
        bool ParsedFullDep = std::stoi(AnnotData[4]);

        if (handleAddrDepID(ParsedID, I, ParsedDepHalfID, ParsedPathToViaFiles,
                            ParsedFullDep)) {
          markIDAsVerified(ParsedID);
          continue;
        }

        if (RemappedIDs->find(ParsedID) != RemappedIDs->end()) {
          for (auto const &RemappedID : RemappedIDs->at(ParsedID)) {
            if (handleAddrDepID(RemappedID, I, ParsedDepHalfID,
                                ParsedPathToViaFiles, ParsedFullDep)) {
              markIDAsVerified(ParsedID);
              break;
            }
          }
        }
      }
    }
  }
}
} // namespace

class LKMMAnnotator {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class LKMMVerifier {
public:
  LKMMVerifier();

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  // Contains all unverified address dependency beginning annotations.
  std::shared_ptr<DepHalfMap<VerAddrDepBeg>> BrokenADBs;

  // Contains all unverified address dependency ending annotations.
  std::shared_ptr<DepHalfMap<VerAddrDepEnd>> BrokenADEs;

  std::shared_ptr<IDReMap> RemappedIDs;

  std::shared_ptr<unordered_set<string>> VerifiedIDs;

  unordered_set<string> PrintedBrokenIDs;

  unordered_set<Module *> PrintedModules;

  /// Prints broken dependencies.
  void printBrokenDeps();

  void printBrokenDep(VerDepHalf &Beg, VerDepHalf &End, const string &ID);
};

PreservedAnalyses LKMMAnnotator::run(Module &M, ModuleAnalysisManager &AM) {
  // FIXME: Come up with a way of making bug insertion upstream compatible
  bool InsertedBugs = false;

  for (auto &F : M) {
    if (F.empty())
      continue;

    AnnotCtx AC(&*F.begin());

    // Annotate dependencies.
    AC.runBFS();

    if (InjectBugs) {
      if (!F.hasName())
        continue;

      auto FName = F.getName();

      // Insert bugs if the BFS just annotated a testing function.
      if (FName.contains("doitlk_rr_addr_dep_begin") ||
          FName.contains("doitlk_rw_addr_dep_begin") ||
          FName.contains("doitlk_ctrl_dep_begin")) {
        AC.insertBug(&F, Instruction::Load, "dep begin");
        InsertedBugs = true;
      }

      // Break read -> read addr dep endings.
      else if (FName.contains("doitlk_rr_addr_dep_end")) {
        AC.insertBug(&F, Instruction::Load, "dep end");
        InsertedBugs = true;
      }

      // Break read -> write addr dep and ctrl dep endings.
      else if (FName.contains("doitlk_rw_addr_dep_end") ||
               FName.contains("doitlk_ctrl_dep_end")) {
        AC.insertBug(&F, Instruction::Store, "dep end");
        InsertedBugs = true;
      }
    }
  }

  return InsertedBugs ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

LKMMVerifier::LKMMVerifier()
    : BrokenADBs(std::make_shared<DepHalfMap<VerAddrDepBeg>>()),
      BrokenADEs(std::make_shared<DepHalfMap<VerAddrDepEnd>>()),
      RemappedIDs(std::make_shared<IDReMap>()),
      VerifiedIDs(std::make_shared<unordered_set<string>>()),
      PrintedBrokenIDs(), PrintedModules() {}

PreservedAnalyses LKMMVerifier::run(Module &M, ModuleAnalysisManager &AM) {
  for (auto &F : M) {
    if (F.empty())
      continue;

    auto VC =
        VerCtx(&*F.begin(), BrokenADBs, BrokenADEs, RemappedIDs, VerifiedIDs);

    VC.runBFS();
  }

  printBrokenDeps();

  return PreservedAnalyses::all();
}

void LKMMVerifier::printBrokenDeps() {
  auto CheckDepPair = [this](auto &P, auto &E) {
    auto ID = P.first;

    // Exclude duplicate IDs by normalising them.
    // This means we only print one representative of each equivalence class.
    if (auto Pos = ID.find("-#"))
      ID = ID.substr(0, Pos);

    auto &VDB = P.second;

    auto VDEP = E->find(ID);

    if (VDEP == E->end())
      return;

    auto &VDE = VDEP->second;

    if (PrintedBrokenIDs.find(ID) != PrintedBrokenIDs.end())
      return;

    PrintedBrokenIDs.insert(ID);
    printBrokenDep(VDB, VDE, ID);
  };

  for (auto &VADBP : *BrokenADBs)
    CheckDepPair(VADBP, BrokenADEs);
}

void LKMMVerifier::printBrokenDep(VerDepHalf &Beg, VerDepHalf &End,
                                  const string &ID) {
  string DepKindStr{""};

  if (isa<VerAddrDepBeg>(Beg))
    DepKindStr = "Address dependency";
  else
    llvm_unreachable("Invalid beginning type when printing broken dependency.");

  errs() << "//===--------------------------Broken "
            "Dependency---------------------------===//\n";

  errs() << DepKindStr << " with ID: " << ID << "\n\n";

  errs() << "Dependency Beginning:\t" << Beg.getParsedDepHalfID() << "\n";
  errs() << "\nPath to (via files) from annotation: "
         << Beg.getParsedpathTOViaFiles() << "\n";

  errs() << "\nDependnecy Ending:\t" << End.getParsedDepHalfID() << "\n";
  errs() << "\nPath to (via files) from annotation: "
         << End.getParsedpathTOViaFiles() << "\n";

  if (auto *VADE = dyn_cast<VerAddrDepEnd>(&End))
    errs() << "\nFull dependency: " << (VADE->getParsedFullDep() ? "yes" : "no")
           << "\n";

  errs() << "Broken " << End.getBrokenBy() << "\n";

  if (auto *VADB = dyn_cast<VerAddrDepBeg>(&Beg)) {
    auto &DCP = VADB->getDCP();
    auto &DCInter = DCP.first;
    auto &DCUnion = DCP.second;

    errs() << "DCInter at ";
    End.getInst()->print(errs());
    errs() << "\n";

    for (auto *V : DCInter) {
      V->print(errs());
      errs() << "\n";
    }

    errs() << "\nDCUnion:\n";
    for (auto *V : DCUnion) {
      V->print(errs());
      errs() << "\n";
    }
  }

#define DEBUG_TYPE "lkmm-print-modules"
  LLVM_DEBUG(dbgs() << "\nFirst access in optimised IR\n\n"
                    << "inst:\n\t";);

  LLVM_DEBUG(Beg.getInst()->print(dbgs()));

  LLVM_DEBUG(dbgs() << "\noptimised IR function:\n\t"
                    << Beg.getInst()->getFunction()->getName() << "\n\n");

  LLVM_DEBUG(dbgs() << "\nSecond access in optimised IR\n\n"
                    << "inst:\n\t");

  LLVM_DEBUG(End.getInst()->print(dbgs()));

  LLVM_DEBUG(dbgs() << "\nOptimised IR function:\n\t"
                    << End.getInst()->getFunction()->getName() << "\n\n");

  if (PrintedModules.find(Beg.getInst()->getModule()) == PrintedModules.end()) {
    LLVM_DEBUG(dbgs() << "Optimised IR module:\n");
    LLVM_DEBUG(Beg.getInst()->getModule()->print(dbgs(), nullptr));

    PrintedModules.insert(Beg.getInst()->getModule());
  }
#undef DEBUG_TYPE

  errs() << "//"
            "===---------------------------------------------------------------"
            "-------===//\n\n";
}

//===----------------------------------------------------------------------===//
// The Annotation Pass
//===----------------------------------------------------------------------===//

PreservedAnalyses LKMMAnnotateDepsPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  auto A = LKMMAnnotator();

  return A.run(M, AM);
}

//===----------------------------------------------------------------------===//
// The Verification Pass
//===----------------------------------------------------------------------===//

PreservedAnalyses LKMMVerifyDepsPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  auto V = LKMMVerifier();

  return V.run(M, AM);
}

} // namespace llvm
