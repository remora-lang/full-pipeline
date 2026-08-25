// GPU-accessible implementation of MLIR's memref allocation hook.
//
// memref.alloc normally lowers to a plain malloc, and a discrete AMD GPU cannot
// read host malloc memory: gfx1100 has no XNACK, so the kernel takes a page
// fault the moment it touches the buffer.  Passing use-generic-functions to
// --finalize-memref-to-llvm makes MLIR emit calls to these two hooks instead of
// malloc/free, letting us hand back pinned host memory that the GPU can reach.
//
// Nothing here is specific to any one Remora program.

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Declared by hand so this file builds without HIP headers on the include path;
// only libamdhip64 is needed, at link time.
typedef int hipError_t;
extern hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags);
extern hipError_t hipHostFree(void *ptr);
extern const char *hipGetErrorString(hipError_t error);

void *_mlir_memref_to_llvm_alloc(size_t size) {
  void *ptr = NULL;
  hipError_t err = hipHostMalloc(&ptr, size, 0);
  if (err != 0) {
    fprintf(stderr, "hipHostMalloc of %zu bytes failed: %s\n", size,
            hipGetErrorString(err));
    exit(1);
  }
  return ptr;
}

void _mlir_memref_to_llvm_free(void *ptr) { hipHostFree(ptr); }

// Scaffolding for a program that takes an argument has to put its input
// somewhere the GPU can see too, so we expose the same allocator under a name
// that does not look like a compiler-internal hook.
void *remora_gpu_alloc(size_t size) { return _mlir_memref_to_llvm_alloc(size); }
