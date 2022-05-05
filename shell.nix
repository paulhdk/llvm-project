# source https://nixos.wiki/wiki/LLVM
with import <nixpkgs> {};
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
    python39
    python39Packages.html5lib
    python39Packages.pyyaml
    python39Packages.pygments
  ];

  disableHardening = true;
}
