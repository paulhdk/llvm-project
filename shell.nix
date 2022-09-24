# source https://nixos.wiki/wiki/LLVM
with import <nixpkgs> { };
let
  myPython = pkgs.python310.withPackages (ps: with ps; [
    autopep8
    html5lib
    pyyaml
    pygments
  ]);
in
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
    myPython
    llvmPackages_14.lld
    llvmPackages_14.lldb
  ];

  disableHardening = true;
  shellHook = ''
    export PYTHONPATH='lldb -P'
  '';

}
