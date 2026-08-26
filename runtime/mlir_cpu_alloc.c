// Buffers for the CPU target.
//
// The counterpart to demo_cuda/mlir_cuda_alloc.c and demo_rocm's, and much
// duller: the compiled code and the caller share one address space, so plain
// malloc is already memory the target can read.
//
// No _mlir_memref_to_llvm_* hooks either: ../remora2exe does not pass
// use-generic-functions to --finalize-memref-to-llvm, so memref.alloc lowers to
// malloc directly, which is what remora_buffer_free expects.

#include <stdlib.h>

#include "remora_internal.h"

void *remora_buffer_alloc(size_t size) { return malloc(size); }

void remora_buffer_free(void *ptr) { free(ptr); }
