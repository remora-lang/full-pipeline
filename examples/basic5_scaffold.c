// driver.c
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef2xI64;

// The C-compatible entry point emitted by -llvm-request-c-wrappers; the raw
// entry_main returns the memref struct by value, which is not the C ABI.
extern void _mlir_ciface_entry_main(MemRef2xI64 *result);

int main(void) {
  MemRef2xI64 result;
  _mlir_ciface_entry_main(&result);
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
