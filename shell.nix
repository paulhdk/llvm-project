# source https://nixos.wiki/wiki/LLVM
with import <nixpkgs> { };
(mkShell.override {
  stdenv = llvmPackages_14.stdenv;
}) {
  nativeBuildInputs = [
    bashInteractive
    ccache
    gdb
    cmake
    clang-tools
    ninja
    graphviz
    python310
    python310Packages.html5lib
    python310Packages.pyyaml
    python310Packages.pygments
  ];

  disableHardening = true;
}
