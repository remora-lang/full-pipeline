#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "basic1.h"

int main(void) {
  struct remora_context *ctx = remora_context_new();
  if (ctx == NULL) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }

  int64_t result;
  if (remora_entry_main(ctx, &result) != 0) {
    fprintf(stderr, "%s\n", remora_context_get_error(ctx));
    return 1;
  }
  printf("%" PRId64 "\n", result);

  remora_context_free(ctx);
  return 0;
}
