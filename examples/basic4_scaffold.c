#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "basic4.h"

int main(void) {
  struct remora_context *ctx = remora_context_new();
  if (ctx == NULL) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }

  struct remora_i64_1d *out = NULL;
  if (remora_entry_main(ctx, &out) != 0) {
    fprintf(stderr, "%s\n", remora_context_get_error(ctx));
    return 1;
  }

  const int64_t *shape = remora_shape_i64_1d(ctx, out);
  int64_t *values = malloc(shape[0] * sizeof *values);
  if (values == NULL) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }
  remora_values_i64_1d(ctx, out, values);

  printf("[");
  for (int64_t i = 0; i < shape[0]; i++) {
    if (i != 0) {
      printf(", ");
    }
    printf("%" PRId64, values[i]);
  }
  printf("]\n");

  free(values);
  remora_free_i64_1d(ctx, out);
  remora_context_free(ctx);
  return 0;
}
