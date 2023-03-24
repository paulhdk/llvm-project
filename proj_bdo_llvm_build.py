#!/usr/bin/env python3

import sys
import subprocess

CMAKE_CONFIG_FLAGS = [
    "-DLLVM_ENABLE_PROJECTS='clang;clang-tools-extra;lldb;lld'",
    "-DLLVM_TARGETS_TO_BUILD='AArch64;ARM'",
    "-DLLVM_TARGET_ARCH='AArch64'",
    "-DLLVM_DEFAULT_TARGET_TRIPLE='aarch64-unknown-linux-gnu'",
    "-DCMAKE_BUILD_TYPE='DEBUG'", "-DLLVM_ENABLE_ASSERTIONS='ON'",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS='ON'", "-DLLVM_CCACHE_BUILD='ON'",
    "-DLLVM_ENABLE_LLD='ON'", "-DCMAKE_C_COMPILER='clang'",
    "-DCMAKE_CXX_COMPILER='clang++'", "-DLLVM_ENABLE_ZLIB='ON'"]

def run(args, stderr=None, stdout=None, shell=False, executable=None):
    print("[ " + " ".join(args) + " ]\n")

    subprocess.run(args=args, stderr=stderr, stdout=stdout,
                   shell=shell, executable=executable)


def configure_llvm():
    run(["cmake", "-G", "Ninja", "-S", "./llvm", "-B", "./build"] + CMAKE_CONFIG_FLAGS)


def build_llvm(target):
    run(["cmake", "--build", "./build", "--target", target, "-j", "128"])


if __name__ == "__main__":
    match sys.argv[1]:
        case "config":
            configure_llvm()
        case "build":
            build_llvm(sys.argv[2])
