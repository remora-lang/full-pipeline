# Running Remora on an NVIDIA GPU

`remora2cuda` compiles a Remora program to an NVIDIA GPU, using MLIR's own GPU
pipeline throughout. It is the GPU counterpart to `../remora2exe`, but is
completely independent. It is also the CUDA counterpart to `../demo_rocm`, which
it follows step for step; the two differ chiefly in which target gets attached
to the kernels and which runtime the host half ends up calling.

The following commands assume you are in the `demo_cuda` directory (the one
containing this README).

Note the `.#cuda` below: CUDA lives in a flake output of its own, so that the
default environment is smaller.

```shell
$ nix shell .#cuda
$ ./remora2cuda ../examples/inc2.remora
built .../demo_cuda/build/inc2 (kernels for sm_75, tile 256)
$ ./build/inc2
[2, 3, 4, ..., 1024, 1025]
```

`CHIP` (default `sm_75`) and `TILE` (default `256`) are environment variables.

The important property is that nothing about the kernel is written by hand. Its
name, launch geometry, and argument layout are all emitted by the compiler, so
the same script works for any program rather than one specific program.

## Memory weirdness

`memref.alloc` lowers to `malloc`, and a discrete NVIDIA GPU cannot read host
malloc memory. Passing `use-generic-functions` to `--finalize-memref-to-llvm`
redirects allocation through `_mlir_memref_to_llvm_alloc`, which
`mlir_cuda_alloc.c` implements with pinned host memory. That file is
program-independent. Take it away and the kernel dereferences an address that
means nothing on the device, which surfaces as `CUDA_ERROR_ILLEGAL_ADDRESS` at
the next stream synchronisation.

## Shutdown weirdness

MLIR unloads the GPU module from a static destructor, and those run after the
`atexit` handler the CUDA driver installs when it is first initialised. By then
the driver has shut itself down, so the unload comes back
`CUDA_ERROR_DEINITIALIZED` -- which `cuGetErrorName` cannot name that late
either -- and a program that has just computed a perfectly correct answer
signs off with

```
'cuModuleUnload(module)' failed with '<unknown>'
```

The allocator is not to blame; one written against the driver API rather than
`libcudart` behaves the same way. The scaffolding therefore leaves through
`_Exit`, which skips the destructor and the message along with it. Scaffolding
that does not -- the CPU scaffolding, which programs without a
`_gpu_scaffold.c` of their own fall back to -- still prints it.

## Known limitations

**The tile size must divide the iteration count.** If it does not, tiling
produces an `affine.min` bound, `--convert-parallel-loops-to-gpu` quietly
declines to convert the loop, and we get a correct CPU binary with no kernel in
it at all.

**`TILE` above the device's block-size limit fails at launch.** A tile becomes
the block size, so exceeding the limit makes `cuLaunchKernel` return
`CUDA_ERROR_INVALID_VALUE`. The program then prints zeroes and exits 0, which is
worse than crashing. The limit is a property of the card rather than anything
the pipeline picks -- `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`, which reads
1024 on the Quadro T1000 this was tried on -- so unlike the 256 that
`../demo_rocm` is stuck with, it is not something `--nvvm-attach-target` stamps
onto the code object. Every tile up to that limit really does work; `TILE=512`
and `TILE=1024` are both fine here. Reaching the limit takes a program of more
than 1024 elements anyway, since the tile has to divide the iteration count as
well.

**Programs whose data is compile-time constant fault.** Bufferization turns
`arith.constant dense<...>` into a `memref.global`, which lives in the host
binary's read-only data and is not GPU-accessible; only `memref.alloc` goes
through the pinned allocator. Such a program dies the same way one linked
against plain `malloc` does, with `CUDA_ERROR_ILLEGAL_ADDRESS` and a row of
zeroes. This is why `../examples/basic0`, `basic2`, `basic4` and `basic5`
(whose inputs are all literals) fault as soon as a tile size small enough to
actually produce a kernel is used. At the default `TILE` they silently fall
back to CPU and print the right answer, which is a confusing way to be wrong.
`inc2` is unaffected because its input comes from the scaffolding. Fixing this
properly means copying such globals into device-visible memory during lowering.
