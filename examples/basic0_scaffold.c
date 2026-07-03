// driver.c
#include <stdio.h>
#include <stdint.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef2xI32;

extern void entry_main(MemRef2xI32 *result);

int main(void) {
  MemRef2xI32 result;
  entry_main(&result);
  printf("[");
  for (int64_t i = 0; i < result.sizes[0]; i++) {
    if (i != 0) {
      printf(", ");
    }
    printf("%d", result.aligned[i]);
  }
  printf("]\n");
  return 0;
}
