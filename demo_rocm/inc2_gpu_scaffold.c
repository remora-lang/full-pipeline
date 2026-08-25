// GPU scaffolding for ../examples/inc2.remora.
//
// Compare this with ../examples/inc2_scaffold.c: the entry point, the memref
// descriptor and the printing are identical, and the only change is where the
// input buffer comes from.  The CPU version can use a static array, but a
// discrete AMD GPU cannot read one, so the input is allocated with the same
// pinned-memory allocator that mlir_rocm_alloc.c installs behind memref.alloc.

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#define N 1024

typedef struct {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef1024xI64;

// The C-compatible entry point emitted by -llvm-request-c-wrappers. Unlike the
// nullary entry points, this one takes an argument: the result descriptor comes
// first, and every memref argument is passed as a *pointer* to its descriptor.
extern void _mlir_ciface_entry_inc2(MemRef1024xI64 *result,
                                    MemRef1024xI64 *in);

// From mlir_rocm_alloc.c: pinned host memory, reachable from the GPU.
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
  return 0;
}
