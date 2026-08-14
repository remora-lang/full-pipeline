// driver.c
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

extern int64_t _mlir_ciface_entry_main(void);

int main(void) {
  int64_t result = _mlir_ciface_entry_main();
  printf("%" PRId64 "\n", result);
  return 0;
}
