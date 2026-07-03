# Remora pipeline proof-of-concept

This repository is intended as a proof-of-concept of full compilation pipelines
that use the software developed as part of the RAPHAEL project (*Remora: Array
Programming on Heterogeneous Architectures via E-Graphs for LLVM*). The project
contains various components with exotic dependencies, which may occasionally be
incompatible with each other. This repository uses [Nix
Flakes](https://determinate.systems/blog/nix-flakes-explained/) to pull together
known-compatible components in a reproducible way. Eventually, we will also use
this setup to produce Docker images that can be used without having to interact
with Nix.

## Usage

If you have Nix installed and [Flakes enabled](https://nixos.wiki/wiki/Flakes)
(they are an "experimental feature"), then you can enter a shell with all
components available on your `$PATH` by standing in the repository and running:

```shell
$ nix shell
```

Then run the following to compile a Remora program `foo.remora` to `foo.mlir`.

TODO: I cannot figure out how to get from the MLIR to LLVM code and then an
executable.

## Example programs

Remora programs known to work with this pipeline can be found in `examples/`.

## Maintenance

The intent is that the component versions are updated from time to time when new
versions are made that work with each other.
