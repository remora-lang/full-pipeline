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
        in {
          default = pkgs.buildEnv {
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
        });
    };
}
