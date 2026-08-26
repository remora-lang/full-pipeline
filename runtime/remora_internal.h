// Helpers shared by the runtime and the generated bindings. Not a public
// header: a program linking against a Remora library includes NAME.h instead.

#ifndef REMORA_INTERNAL_H
#define REMORA_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

struct remora_context;

// Record an error and return -1, so a failing binding can just
// `return remora_set_error(ctx, ...)`.
int remora_set_error(struct remora_context *ctx, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

// Memory the target can read: plain malloc on the CPU, pinned host memory on a
// GPU. Implemented in mlir_cpu_alloc.c, demo_cuda/mlir_cuda_alloc.c and
// demo_rocm/mlir_rocm_alloc.c. Returns NULL on failure, having said why on
// stderr.
void *remora_buffer_alloc(size_t size);
void remora_buffer_free(void *ptr);

#define REMORA_MAX_RANK 8

// A memref descriptor of any rank: the generated remora_<elem>_<rank>d structs
// with sizes and strides run together. Generated code static-asserts the match.
struct remora_desc {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t shape[]; // sizes[rank], then strides[rank]
};

// A fresh row-major array of `dims` holding a copy of `data`. NULL with the
// reason in ctx, prefixed by `what`, if a dimension is negative or memory runs
// out.
struct remora_desc *remora_array_new(struct remora_context *ctx,
                                     const char *what, const void *data,
                                     int rank, const int64_t *dims,
                                     size_t elem_size);
void remora_array_free(struct remora_desc *arr);

// Copy an array into dense row-major memory owned by the caller.
void remora_array_values(void *dst, const struct remora_desc *arr, int rank,
                         size_t elem_size);

#endif
