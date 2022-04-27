#!/bin/sh
#

cd build

cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE="RELEASE" -DCMAKE_EXPORT_COMPILE_COMMANDS="ON" -DLLVM_TARGETS_TO_BUILD="AArch64;ARM" -DLLVM_CCACHE_BUILD="ON" ../llvm

cmake --build . --target $1
