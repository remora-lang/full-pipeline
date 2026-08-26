// GPU-accessible memory, and the hooks that make MLIR use it.
//
// memref.alloc normally lowers to a plain malloc, which a discrete AMD GPU
// cannot read: gfx1100 has no XNACK, so the kernel page-faults on it. Passing
// use-generic-functions to --finalize-memref-to-llvm makes MLIR call
// _mlir_memref_to_llvm_alloc/_free instead, so we can hand back pinned host
// memory the GPU can reach.
//
// The generated bindings use the same pool, so a caller's array is as reachable
// as one the compiled code allocates itself. Nothing here is program-specific.

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "remora_internal.h"

// Declared by hand so this file builds without HIP headers on the include path;
// only libamdhip64 is needed, at link time.
typedef int hipError_t;
extern hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags);
extern hipError_t hipHostFree(void *ptr);
extern const char *hipGetErrorString(hipError_t error);

void *remora_buffer_alloc(size_t size) {
  void *ptr = NULL;
  hipError_t err = hipHostMalloc(&ptr, size, 0);
  if (err != 0) {
    fprintf(stderr, "hipHostMalloc of %zu bytes failed: %s\n", size,
            hipGetErrorString(err));
    return NULL;
  }
  return ptr;
}

void remora_buffer_free(void *ptr) { hipHostFree(ptr); }

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
