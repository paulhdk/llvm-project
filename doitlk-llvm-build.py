#!/usr/bin/env python3

import sys
import subprocess
import os

CMAKE_CONFIG_FLAGS = ["-DLLVM_ENABLE_PROJECTS=\"clang;clang-tools-extra;lldb\"",
                      "-DLLVM_TARGETS_TO_BUILD=\"AArch64;ARM;X86\"",
                      "-DLLVM_TARGET_ARCH=\"AArch64\"",
                      "-DLLVM_DEFAULT_TARGET_TRIPLE=\"aarch64-unknown-linux-gnu\"",
                      "-DCMAKE_BUILD_TYPE=\"DEBUG\"",
                      "-DLLVM_ENABLE_ASSERTIONS=\"ON\"",
                      "-DCMAKE_EXPORT_COMPILE_COMMANDS=\"ON\"",
                      "-DLLVM_CCACHE_BUILD=\"ON\""
                      ]


def configure_llvm():
    os.chdir("build")
    CmdStr = "cmake -G Ninja " + " ".join(map(str, CMAKE_CONFIG_FLAGS)) + " ../llvm",
    subprocess.run(CmdStr, shell=True, check=True)


def build_llvm(target):
    subprocess.run(["ninja", target], check=True)


if __name__ == "__main__":
    match sys.argv[1]:
        case "config":
            configure_llvm()
        case "build":
            build_llvm(sys.argv[2])
