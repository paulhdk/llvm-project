# source https://nixos.wiki/wiki/LLVM
{ pkgs ? import (fetchTarball "https://github.com/PBHDK/nixpkgs/archive/2eac0c6990719e80fd56d89b1f48e7a5bf59b843.tar.gz") { }
}:
let
  myPython = pkgs.python310.withPackages (ps: with ps; [
    autopep8
    html5lib
    pyyaml
    pygments
  ]);
in
pkgs.llvmPackages_latest.stdenv.mkDerivation {
  name = "llvm-debug-env";
  nativeBuildInputs = [
    pkgs.bashInteractive
    pkgs.ccache
    pkgs.gdb
    pkgs.cmake
    pkgs.ninja
    pkgs.graphviz
    myPython
    pkgs.llvmPackages_latest.lld
    pkgs.llvmPackages_latest.lldb
    pkgs.pkg-config
  ];
  buildInputs = [ pkgs.zlib ];
  hardeningDisable = [ "all" ];
  shellHook = ''
    export PYTHONPATH='lldb -P'
  '';
  PATH_TO_CLANG = "${pkgs.llvmPackages_latest.stdenv.cc}/bin/clang++";
  # FIXME why is this not included?
  NIX_LDFLAGS = "-rpath ${pkgs.zlib}/lib";
}
