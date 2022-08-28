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

#include "llvm/IR/PassManager.h"

#ifndef LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEANALYSIS_H
#define LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEANALYSIS_H

namespace llvm {

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
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CUSTOMMEMDEP_H
