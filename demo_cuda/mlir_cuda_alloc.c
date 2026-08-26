// GPU-accessible implementation of MLIR's memref allocation hook.
//
// memref.alloc normally becomes a plain malloc, which a discrete NVIDIA GPU
// cannot read: the address means nothing on the device, so the kernel faults as
// soon as it touches the buffer.  Passing use-generic-functions to
// --finalize-memref-to-llvm makes MLIR call these two hooks instead, so we can
// hand back pinned host memory the GPU can reach.
//
// Nothing here is specific to any one Remora program.

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Declared by hand so this builds without CUDA headers around; we only need
// libcudart at link time.
typedef int cudaError_t;
extern cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags);
extern cudaError_t cudaFreeHost(void *ptr);
extern const char *cudaGetErrorString(cudaError_t error);

// cudaHostAllocMapped: also map it into the device address space.  Unified
// addressing then makes the device pointer the same as the host pointer, so
// MLIR's memref descriptor works on the GPU unchanged.
#define HOST_ALLOC_MAPPED 0x02

void *_mlir_memref_to_llvm_alloc(size_t size) {
  void *ptr = NULL;
  cudaError_t err = cudaHostAlloc(&ptr, size, HOST_ALLOC_MAPPED);
  if (err != 0) {
    fprintf(stderr, "cudaHostAlloc of %zu bytes failed: %s\n", size,
            cudaGetErrorString(err));
    exit(1);
  }
  return ptr;
}

void _mlir_memref_to_llvm_free(void *ptr) { cudaFreeHost(ptr); }

// Scaffolding has to put its input somewhere the GPU can see too, so here is
// the same allocator under a name that does not look compiler-internal.
void *remora_gpu_alloc(size_t size) { return _mlir_memref_to_llvm_alloc(size); }
