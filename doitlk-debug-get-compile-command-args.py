#!/usr/bin/env python3

import sys
import re
import os


def get_compile_command_args(CompileCmdsPath):
    with open(CompileCmdsPath) as f:
        CompileCmdStr = f.readline()

        R = re.search(r'(?<=:= clang )[\s\S]*', CompileCmdStr)

        if not R:
            exit(-1)

        CompileCmdArgs = R.group()
        CompileCmdArgs = CompileCmdArgs.rstrip()
        return CompileCmdArgs.rstrip('\n')


def writeArgsToFile(CompileCmdsPath):
    with open(".compile_args", "w+") as f:
        CompileArgs = get_compile_command_args(CompileCmdsPath)
        f.write(CompileArgs)


if __name__ == "__main__":
    arg = sys.argv[1]
    match arg:
        case "writeArgsToFile":
            writeArgsToFile(sys.argv[2])
        case "getArgs":
            with open(".compile_args") as f:
                print(f.read())
        case "clean":
            os.remove(".compile_args")
        case other:
            exit("Invalid argument provided")
