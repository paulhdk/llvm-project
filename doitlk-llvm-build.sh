#!/bin/sh
#

cd build

cmake -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lldb" \
  -DLLVM_TARGETS_TO_BUILD="AArch64;ARM;X86" \
  -DLLVM_TARGET_ARCH="AArch64" \
  -DLLVM_DEFAULT_TARGET_TRIPLE="aarch64-unknown-linux-gnu" \
  -DCMAKE_BUILD_TYPE="RELEASE" \
  -DLLVM_ENABLE_ASSERTIONS="ON" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS="ON" \
  -DLLVM_CCACHE_BUILD="ON" \
  ../llvm

ninja $1
