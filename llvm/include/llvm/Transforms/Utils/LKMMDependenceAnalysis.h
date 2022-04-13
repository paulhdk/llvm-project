//===- llvm/Transforms/LKMMDependenceAnalysis.h - LKMM Deps -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains all declarations / definitions required for LKMM
/// dependence analysis. Implementations live in LKMMDependenceAnalysis.cpp.
///
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Casting.h"
#include <list>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifndef LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEANALYSIS_H
#define LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEANALYSIS_H

namespace llvm {
namespace {
// FIXME Is there a more elegant way of dealing with duplicate IDs (preferably
//  getting eliminating the problem all together)?

// The IDReMap type alias represents the map of IDs to sets of alias IDs which
// verification contexts use for remapping duplicate IDs. Duplicate IDs appear
// when an annotated instruction is duplicated as part of optimizations.
using IDReMap =
    std::unordered_map<std::string, std::unordered_set<std::string>>;

// Represents a map of IDs to (potential) dependency halfs.
template <typename T> using DepHalfMap = std::unordered_map<std::string, T>;

class VerDepHalf;
class VerAddrDepBeg;
class VerAddrDepEnd;
class VerCtrlDepBeg;
class VerCtrlDepEnd;
} // namespace

//===----------------------------------------------------------------------===//
// The Annotation Pass
//===----------------------------------------------------------------------===//

class LKMMAnnotateDepsPass : public PassInfoMixin<LKMMAnnotateDepsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

//===----------------------------------------------------------------------===//
// The Verification Pass
//===----------------------------------------------------------------------===//

class LKMMVerifyDepsPass : public PassInfoMixin<LKMMVerifyDepsPass> {
public:
  LKMMVerifyDepsPass();

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  // Contains all unverified address dependency beginning annotations.
  std::shared_ptr<DepHalfMap<VerAddrDepBeg>> BrokenADBs;

  // Contains all unverified address dependency ending annotations.
  std::shared_ptr<DepHalfMap<VerAddrDepEnd>> BrokenADEs;

  // Contains all unverified control dependency beginning annotations.
  std::shared_ptr<DepHalfMap<VerCtrlDepBeg>> BrokenCDBs;

  // Contains all unverified control dependency ending annotations.
  std::shared_ptr<DepHalfMap<VerCtrlDepEnd>> BrokenCDEs;

  std::shared_ptr<IDReMap> RemappedIDs;

  std::shared_ptr<std::unordered_set<std::string>> VerifiedIDs;

  std::unordered_set<std::string> PrintedBrokenIDs;

  std::unordered_set<Module *> PrintedModules;

  /// Prints broken dependencies.
  void printBrokenDeps();

  void printBrokenDep(VerDepHalf &Beg, VerDepHalf &End, const std::string &ID);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CUSTOMMEMDEP_H