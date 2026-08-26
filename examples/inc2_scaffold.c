#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "inc2.h"

#define N 1024

int main(void) {
  struct remora_context *ctx = remora_context_new();
  int64_t *input = malloc(N * sizeof *input);
  if (ctx == NULL || input == NULL) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }
  for (int64_t i = 0; i < N; i++) {
    input[i] = i;
  }

  struct remora_i64_1d *in = remora_new_i64_1d(ctx, input, N);
  free(input);
  if (in == NULL) {
    fprintf(stderr, "%s\n", remora_context_get_error(ctx));
    return 1;
  }

  struct remora_i64_1d *out = NULL;
  if (remora_entry_inc2(ctx, &out, in) != 0) {
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
  remora_free_i64_1d(ctx, in);
  remora_context_free(ctx);
  return 0;
}
