// GPU-accessible memory, and the hooks that make MLIR use it.
//
// memref.alloc normally becomes a plain malloc, which a discrete NVIDIA GPU
// cannot read. Passing use-generic-functions to --finalize-memref-to-llvm makes
// MLIR call _mlir_memref_to_llvm_alloc/_free instead, so we can hand back
// pinned host memory the GPU can reach.
//
// The generated bindings use the same pool, so a caller's array is as reachable
// as one the compiled code allocates itself. Nothing here is program-specific.

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "remora_internal.h"

// Declared by hand so this builds without CUDA headers around; we only need
// libcudart at link time.
typedef int cudaError_t;
extern cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags);
extern cudaError_t cudaFreeHost(void *ptr);
extern const char *cudaGetErrorString(cudaError_t error);

// cudaHostAllocMapped: also map it into the device address space. Unified
// addressing then makes the device pointer the same as the host pointer, so
// MLIR's memref descriptor works on the GPU unchanged.
#define HOST_ALLOC_MAPPED 0x02

void *remora_buffer_alloc(size_t size) {
  void *ptr = NULL;
  cudaError_t err = cudaHostAlloc(&ptr, size, HOST_ALLOC_MAPPED);
  if (err != 0) {
    fprintf(stderr, "cudaHostAlloc of %zu bytes failed: %s\n", size,
            cudaGetErrorString(err));
    return NULL;
  }
  return ptr;
}

void remora_buffer_free(void *ptr) { cudaFreeHost(ptr); }

// The compiled code cannot report a failed allocation -- memref.alloc has no
// error result -- so this is the one place left that must give up on the spot.
// Bindings call remora_buffer_alloc directly and report NULL via the context.
void *_mlir_memref_to_llvm_alloc(size_t size) {
  void *ptr = remora_buffer_alloc(size);
  if (ptr == NULL) {
    fprintf(stderr, "remora: out of GPU-visible memory inside compiled code\n");
    exit(1);
  }
  return ptr;
}

void _mlir_memref_to_llvm_free(void *ptr) { remora_buffer_free(ptr); }
