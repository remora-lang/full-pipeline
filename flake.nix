{
  description = "Combined development shell for Remora, MLIR backend, and Futhark";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    remora.url = "github:remora-lang/remora";
    mlir-backend.url = "github:remora-lang/mlir-backend";
    futhark.url = "github:diku-dk/futhark";
  };

  outputs = { self, nixpkgs, remora, mlir-backend, futhark, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system: f system);

    in {
      packages = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };

          toolchain = pkgs.buildEnv {
            name = "remora-toolchain";
            paths = [
              remora.packages.${system}.default
              mlir-backend.packages.${system}.default
              futhark.packages.${system}.default
              pkgs.llvmPackages_22.mlir
              pkgs.llvmPackages_22.llvm
              pkgs.llvmPackages_22.clang
            ];
          };
        in {
          default = toolchain;
        } // nixpkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          docker = pkgs.dockerTools.buildLayeredImage {
            name = "remora-toolchain";
            tag = "latest";
            contents = [
              toolchain
              pkgs.bashInteractive
              pkgs.coreutils
              pkgs.which
              pkgs.gnugrep
              pkgs.gnused
              pkgs.findutils
              pkgs.dockerTools.caCertificates
              pkgs.dockerTools.fakeNss
              pkgs.dockerTools.usrBinEnv
            ];
            extraCommands = ''
              mkdir -p tmp
              chmod 1777 tmp
            '';
            config = {
              Env = [
                "PATH=/bin"
                "HOME=/root"
              ];
              WorkingDir = "/root";
              Cmd = [ "${pkgs.bashInteractive}/bin/bash" ];
            };
          };
        });
    };
}
