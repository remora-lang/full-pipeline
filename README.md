# Remora pipeline proof-of-concept

This repository is intended as a proof-of-concept of full compilation pipelines
that use the software developed as part of the RAPHAEL project (*Remora: Array
Programming on Heterogeneous Architectures via E-Graphs for LLVM*). The project
contains various components with exotic dependencies, which may occasionally be
incompatible with each other. This repository uses [Nix
Flakes](https://determinate.systems/blog/nix-flakes-explained/) to pull together
known-compatible components in a reproducible way. The same setup also produces
a Docker image.

## Usage

If you have Nix installed and [Flakes enabled](https://nixos.wiki/wiki/Flakes)
(they are an "experimental feature"), then you can enter a shell with all
components available on your `$PATH` by standing in the repository and running:

```shell
$ nix shell
```

Then run the following to compile a Remora program `foo.remora`:

```shell
$ ./remora2exe foo.remora
```

This produces a (CPU) library `build/libfoo.a` and a generated header
`build/foo.h`, which you can link into a C program of your own. If there is a
scaffolding program `foo_scaffold.c` next to `foo.remora`, it is linked against
the library as well, giving you a `build/foo` you can run.

The header is generated from the compiled entry points, so it names them and
their shapes without anything being written by hand:

```c
struct remora_context *ctx = remora_context_new();
struct remora_i64_1d *out = NULL;
if (remora_entry_main(ctx, &out) != 0) {
  fprintf(stderr, "%s\n", remora_context_get_error(ctx));
}
```

`../demo_cuda` and `../demo_rocm` produce the same interface for GPU targets,
so scaffolding can be moved between them unchanged.

## Docker

The toolchain is also published as a Docker image:

```shell
$ docker run -it ghcr.io/remora-lang/remora-toolchain
```

That drops you into a shell in `/remora`, which holds a copy of this
repository, so `./remora2exe examples/basic0.remora` and `./test.sh` work there
just as they do above.

CI builds and pushes the image on every push to `main`. To build it yourself
instead:

```shell
$ nix build .#docker
$ docker load < result
```

## Example programs

Remora programs known to work with this pipeline can be found in `examples/`.

## Maintenance

The intent is that the component versions are updated from time to time when new
versions are made that work with each other.
