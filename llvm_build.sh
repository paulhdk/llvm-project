#!/bin/bash

if [ "$1" = "config" ] || [ "$1" = "configbuild" ]; then
	if [ ! -d "build" ]; then
		mkdir build
	fi

	CMAKE_CONFIG_FLAGS=("-DLLVM_ENABLE_PROJECTS='clang'"
		"-DCMAKE_BUILD_TYPE=RelWithDebInfo"
		"-DLLVM_TARGETS_TO_BUILD='AArch64'"
		"-DLLVM_TARGET_ARCH='AArch64'"
		"-DLLVM_DEFAULT_TARGET_TRIPLE='aarch64-unknown-linux-gnu'"
		"-DLLVM_ENABLE_ASSERTIONS='ON'"
		"-DCMAKE_EXPORT_COMPILE_COMMANDS='ON'"
		"-DLLVM_CCACHE_BUILD='ON'"
		"-DLLVM_USE_LINKER='ld'"
		"-DCMAKE_C_COMPILER='clang'"
		"-DCMAKE_CXX_COMPILER='clang++'"
	)

	cmake -G Ninja -S ./llvm -B ./build "${CMAKE_CONFIG_FLAGS[@]}"
	# ninja -C build "${CMAKE_CONFIG_FLAGS[@]}"
fi

cmake --build ./build --target "$2"
