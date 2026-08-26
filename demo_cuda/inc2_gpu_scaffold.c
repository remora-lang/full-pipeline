// GPU scaffolding for ../examples/inc2.remora.
//
// Same as ../examples/inc2_scaffold.c apart from two things.  The CPU version's
// static array is unreadable from a discrete NVIDIA GPU, so the input comes
// from the pinned allocator in mlir_cuda_alloc.c instead.  And main leaves via
// _Exit, because MLIR unloads the GPU module from a static destructor, which
// runs after the CUDA driver has already shut itself down and so complains
// about a failure we can do nothing about.

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

typedef struct {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef1024xI64;

// The C-compatible entry point from -llvm-request-c-wrappers. This one takes an
// argument, so the result descriptor comes first and every memref argument is
// passed as a *pointer* to its descriptor.
extern void _mlir_ciface_entry_inc2(MemRef1024xI64 *result,
                                    MemRef1024xI64 *in);

// From mlir_cuda_alloc.c: pinned host memory, reachable from the GPU.
extern void *remora_gpu_alloc(size_t size);

int main(void) {
  int64_t *input = remora_gpu_alloc(N * sizeof(int64_t));
  for (int64_t i = 0; i < N; i++) {
    input[i] = i;
  }
  MemRef1024xI64 in = {
      .allocated = input,
      .aligned = input,
      .offset = 0,
      .sizes = {N},
      .strides = {1},
  };

  MemRef1024xI64 result;
  _mlir_ciface_entry_inc2(&result, &in);

  printf("[");
  for (int64_t i = 0; i < result.sizes[0]; i++) {
    if (i != 0) {
      printf(", ");
    }
    printf("%" PRId64, result.aligned[i * result.strides[0]]);
  }
  printf("]\n");
  fflush(stdout);
  _Exit(0);
}
