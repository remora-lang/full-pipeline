// driver.c
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

static int64_t input[N];

int main(void) {
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
