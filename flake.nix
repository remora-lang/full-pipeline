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

          # MLIR's GPU pipeline turns gpu.launch_func into calls to
          # mgpuModuleLoad, mgpuLaunchKernel and friends, which live in
          # libmlir_rocm_runtime.so.  nixpkgs builds MLIR with
          # MLIR_ENABLE_ROCM_RUNNER=OFF, so that library is absent and anything
          # targeting an AMD GPU fails to link.  It is one self-contained source
          # file, so rather than rebuilding all of MLIR we compile just that file
          # out of the very same source revision nixpkgs used for the rest.
          mlir-rocm-runtime = pkgs.stdenv.mkDerivation {
            pname = "mlir-rocm-runtime";
            inherit (pkgs.llvmPackages_22.mlir) version src;

            nativeBuildInputs = [ pkgs.llvmPackages_22.clang ];
            buildInputs = [ pkgs.rocmPackages.clr ];

            buildPhase = ''
              runHook preBuild
              clang++ -fPIC -shared -std=c++17 -O2 \
                -D__HIP_PLATFORM_AMD__ \
                -I${pkgs.llvmPackages_22.mlir.dev}/include \
                -I${pkgs.llvmPackages_22.llvm.dev}/include \
                -I${pkgs.rocmPackages.clr}/include \
                -Wno-return-type-c-linkage \
                -Wno-nested-anon-types \
                -Wno-gnu-anonymous-struct \
                mlir/lib/ExecutionEngine/RocmRuntimeWrappers.cpp \
                -L${pkgs.rocmPackages.clr}/lib -lamdhip64 \
                -o libmlir_rocm_runtime.so
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              install -Dm555 libmlir_rocm_runtime.so \
                $out/lib/libmlir_rocm_runtime.so
              runHook postInstall
            '';
          };

          # --gpu-module-to-binary shells out to ld.lld to link the GPU code
          # object, and insists on finding it at $ROCM_PATH/llvm/bin/ld.lld --
          # having it merely on $PATH is not enough.  nixpkgs scatters ROCm over
          # separate store paths, so we assemble the directory layout MLIR
          # expects.  It lands at <toolchain>/rocm, which is where remora2rocm
          # points ROCM_PATH.  The device bitcode is only consulted once a
          # kernel calls a math function, but costs nothing to include.
          #
          # HIP itself is linked in alongside, so that scaffolding which has to
          # allocate GPU-visible memory can just say -lamdhip64.  Only the
          # libraries are exposed, not clr's bin/, which would collide with the
          # LLVM tools already in the environment.
          rocm-toolkit = pkgs.runCommand "rocm-toolkit-layout" { } ''
            mkdir -p $out/rocm/llvm/bin $out/lib
            ln -s ${pkgs.rocmPackages.llvm.lld}/bin/ld.lld \
              $out/rocm/llvm/bin/ld.lld
            ln -s ${pkgs.rocmPackages.rocm-device-libs}/amdgcn $out/rocm/amdgcn
            ln -s ${pkgs.rocmPackages.clr}/lib/libamdhip64.so* $out/lib/
          '';

          # ROCm is only packaged for x86_64-linux in nixpkgs.
          rocmSupported = system == "x86_64-linux";

          basePaths = [
            remora.packages.${system}.default
            mlir-backend.packages.${system}.default
            futhark.packages.${system}.default
            pkgs.llvmPackages_22.mlir
            pkgs.llvmPackages_22.llvm
            pkgs.llvmPackages_22.clang
          ];

          toolchain = pkgs.buildEnv {
            name = "remora-toolchain";
            paths = basePaths;
          };

          # Everything the CPU toolchain has, plus the pieces demo_rocm needs.
          # Kept as its own output rather than folded into the default: it drags
          # in all of ROCm, which is a lot to download for anyone who only wants
          # to compile for the CPU.  Enter it with `nix shell .#rocm`.
          rocmToolchain = pkgs.buildEnv {
            name = "remora-toolchain-rocm";
            paths = basePaths ++ [
              mlir-rocm-runtime
              rocm-toolkit
            ];
          };
        in {
          default = toolchain;
        } // nixpkgs.lib.optionalAttrs rocmSupported {
          rocm = rocmToolchain;
        } // nixpkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
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
