# Running Remora on an AMD GPU

`remora2rocm` compiles a Remora program to an AMD GPU, using MLIR's own GPU
pipeline throughout. It is the GPU counterpart to `../remora2exe`, but is
completely independent. It would be nice if someone did a similar demonstration
for CUDA.

The following commands assume you are in the `demo_rocm` directory (the one
containing this README).

Note the `.#rocm` below: ROCm lives in a flake output of its own, so that the
default environment is smaller.

```shell
$ nix shell .#rocm
$ ./remora2rocm ../examples/inc2.remora
built .../demo_rocm/build/inc2 (kernels for gfx1100, tile 256)
$ ./build/inc2
[2, 3, 4, ..., 1024, 1025]
```

`CHIP` (default `gfx1100`) and `TILE` (default `256`) are environment variables.

The important property is that nothing about the kernel is written by hand. Its
name, launch geometry, and argument layout are all emitted by the compiler, so
the same script works for any program rather than one specific program.

## Memory weirdness

`memref.alloc` lowers to `malloc`, and a discrete AMD GPU cannot read host
malloc memory. Passing `use-generic-functions` to `--finalize-memref-to-llvm`
redirects allocation through `_mlir_memref_to_llvm_alloc`, which
`mlir_rocm_alloc.c` implements with pinned host memory. That file is
program-independent.

## Known limitations

**The tile size must divide the iteration count.** If it does not, tiling
produces an `affine.min` bound, `--convert-parallel-loops-to-gpu` quietly
declines to convert the loop, and we get a correct CPU binary with no kernel in
it at all.

**`TILE` above 256 fails at launch.** `--rocdl-attach-target` stamps the code
object with the ROCDL default `max_flat_workgroup_size = 256`, so a 512-thread
launch returns `hipErrorLaunchFailure`. The program then prints zeroes and exits
0, which is worse than crashing. `TILE=256` matching the default is luck, not
design.

**Programs whose data is compile-time constant fault.** Bufferization turns
`arith.constant dense<...>` into a `memref.global`, which lives in the host
binary's read-only data and is not GPU-accessible; only `memref.alloc` goes
through the pinned allocator. This is why `../examples/basic0`, `basic2`,
`basic4` and `basic5` (whose inputs are all literals) fault as soon as a tile
size small enough to actually produce a kernel is used. At the default `TILE`
they silently fall back to CPU and print the right answer, which is a confusing
way to be wrong. `inc2` is unaffected because its input comes from the
scaffolding. Fixing this properly means copying such globals into device-visible
memory during lowering.
