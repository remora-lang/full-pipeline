// The program-independent half of a Remora library: the context, and the array
// operations behind it. ../remora-cbindings.janet generates the typed facade
// that forwards to these.

#include "remora_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Holds the last error, and nothing else yet. It exists because a library must
// not exit() on a bad argument, and because it is where a device handle or
// allocator will go once arrays stop being pinned host memory.
struct remora_context {
  char error[256];
  int has_error;
};

struct remora_context *remora_context_new(void) {
  return calloc(1, sizeof(struct remora_context));
}

void remora_context_free(struct remora_context *ctx) { free(ctx); }

const char *remora_context_get_error(const struct remora_context *ctx) {
  if (ctx == NULL || !ctx->has_error) {
    return NULL;
  }
  return ctx->error;
}

int remora_set_error(struct remora_context *ctx, const char *fmt, ...) {
  if (ctx != NULL) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx->error, sizeof ctx->error, fmt, ap);
    va_end(ap);
    ctx->has_error = 1;
  }
  return -1;
}

static size_t desc_size(int rank) {
  return sizeof(struct remora_desc) + (size_t)2 * (size_t)rank * sizeof(int64_t);
}

struct remora_desc *remora_array_new(struct remora_context *ctx,
                                     const char *what, const void *data,
                                     int rank, const int64_t *dims,
                                     size_t elem_size) {
  int64_t n = 1;
  for (int i = 0; i < rank; i++) {
    if (dims[i] < 0) {
      remora_set_error(ctx, "%s: dimension %d is negative", what, i);
      return NULL;
    }
    n *= dims[i];
  }

  struct remora_desc *arr = malloc(desc_size(rank));
  if (arr == NULL) {
    remora_set_error(ctx, "%s: out of memory", what);
    return NULL;
  }

  // An empty array still needs a distinct pointer to free, and neither
  // remora_buffer_alloc(0) nor memcpy from a possibly-null data is defined.
  size_t bytes = (size_t)n * elem_size;
  void *buf = remora_buffer_alloc(bytes == 0 ? 1 : bytes);
  if (buf == NULL) {
    free(arr);
    remora_set_error(ctx, "%s: could not get %lld bytes of GPU-visible memory",
                     what, (long long)bytes);
    return NULL;
  }
  if (bytes > 0) {
    memcpy(buf, data, bytes);
  }

  arr->allocated = arr->aligned = buf;
  arr->offset = 0;
  int64_t *sizes = arr->shape, *strides = arr->shape + rank;
  for (int i = 0; i < rank; i++) {
    sizes[i] = dims[i];
  }
  strides[rank - 1] = 1; // the rank is at least 1
  for (int i = rank - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * sizes[i + 1];
  }
  return arr;
}

void remora_array_free(struct remora_desc *arr) {
  if (arr != NULL) {
    remora_buffer_free(arr->allocated);
    free(arr);
  }
}

void remora_array_values(void *dst, const struct remora_desc *arr, int rank,
                         size_t elem_size) {
  const int64_t *sizes = arr->shape, *strides = arr->shape + rank;
  const char *src = (const char *)arr->aligned + (size_t)arr->offset * elem_size;
  char *out = dst;

  int64_t total = 1;
  for (int i = 0; i < rank; i++) {
    total *= sizes[i];
  }
  if (total == 0) {
    return;
  }

  // Row-major strides mean the array is contiguous, so it moves in one go.
  // Every result the pipeline produces today takes this path.
  int dense = 1;
  int64_t expected = 1;
  for (int i = rank - 1; i >= 0; i--) {
    if (strides[i] != expected) {
      dense = 0;
      break;
    }
    expected *= sizes[i];
  }
  if (dense) {
    memcpy(out, src, (size_t)total * elem_size);
    return;
  }

  // Otherwise walk the index space with an odometer and use the strides.
  int64_t index[REMORA_MAX_RANK] = {0};
  for (int64_t n = 0; n < total; n++) {
    int64_t off = 0;
    for (int i = 0; i < rank; i++) {
      off += index[i] * strides[i];
    }
    memcpy(out + (size_t)n * elem_size, src + (size_t)off * elem_size,
           elem_size);
    for (int i = rank - 1; i >= 0; i--) {
      if (++index[i] < sizes[i]) {
        break;
      }
      index[i] = 0;
    }
  }
}
