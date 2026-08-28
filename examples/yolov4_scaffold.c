/* Scaffolding for examples/yolov4.remora: a full YOLOv4 forward pass.
 *
 * The entry point takes the 608x608 image followed by the {bias, scale, mean,
 * variance, weights} arrays of each of the 110 convolutions, in cfg-index
 * order -- the three detection heads have no batch norm, so they contribute
 * only {bias, weights}.  That is exactly darknet's own storage order in a
 * .weights file (parser.c:1845-1849), so after skipping darknet's header the
 * remainder of the file is the concatenation of parameters 1..541 in order,
 * with nothing in between.  All this program does is chop it up again and hand each piece
 * to the compiled Remora program as a memref.
 */

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_PARAMS 542
#define MAX_RANK 4

#define DEFAULT_WEIGHTS "yolov4.weights"
#define DEFAULT_IMAGE "input.bin"
#define DEFAULT_OUTPUT "yolov4_out.bin"

/* The shape of the detection table returned by the entry point: one row per
   (anchor, cell) across the three YOLO heads, each row (x, y, w, h, obj) plus
   80 class scores. */
#define OUT_ROWS 22743
#define OUT_COLS 85

typedef struct {
  const char *name;
  int rank;
  int64_t dims[MAX_RANK];
} Param;

/* Parameter 0 is the image; the rest come from the .weights file. */
static const Param params[NUM_PARAMS] = {
    {"image", 3, {3, 608, 608, 0}},
    {"bias-000", 1, {32, 0, 0, 0}},
    {"scale-000", 1, {32, 0, 0, 0}},
    {"mean-000", 1, {32, 0, 0, 0}},
    {"var-000", 1, {32, 0, 0, 0}},
    {"w-000", 4, {32, 3, 3, 3}},
    {"bias-001", 1, {64, 0, 0, 0}},
    {"scale-001", 1, {64, 0, 0, 0}},
    {"mean-001", 1, {64, 0, 0, 0}},
    {"var-001", 1, {64, 0, 0, 0}},
    {"w-001", 4, {64, 32, 3, 3}},
    {"bias-002", 1, {64, 0, 0, 0}},
    {"scale-002", 1, {64, 0, 0, 0}},
    {"mean-002", 1, {64, 0, 0, 0}},
    {"var-002", 1, {64, 0, 0, 0}},
    {"w-002", 4, {64, 64, 1, 1}},
    {"bias-004", 1, {64, 0, 0, 0}},
    {"scale-004", 1, {64, 0, 0, 0}},
    {"mean-004", 1, {64, 0, 0, 0}},
    {"var-004", 1, {64, 0, 0, 0}},
    {"w-004", 4, {64, 64, 1, 1}},
    {"bias-005", 1, {32, 0, 0, 0}},
    {"scale-005", 1, {32, 0, 0, 0}},
    {"mean-005", 1, {32, 0, 0, 0}},
    {"var-005", 1, {32, 0, 0, 0}},
    {"w-005", 4, {32, 64, 1, 1}},
    {"bias-006", 1, {64, 0, 0, 0}},
    {"scale-006", 1, {64, 0, 0, 0}},
    {"mean-006", 1, {64, 0, 0, 0}},
    {"var-006", 1, {64, 0, 0, 0}},
    {"w-006", 4, {64, 32, 3, 3}},
    {"bias-008", 1, {64, 0, 0, 0}},
    {"scale-008", 1, {64, 0, 0, 0}},
    {"mean-008", 1, {64, 0, 0, 0}},
    {"var-008", 1, {64, 0, 0, 0}},
    {"w-008", 4, {64, 64, 1, 1}},
    {"bias-010", 1, {64, 0, 0, 0}},
    {"scale-010", 1, {64, 0, 0, 0}},
    {"mean-010", 1, {64, 0, 0, 0}},
    {"var-010", 1, {64, 0, 0, 0}},
    {"w-010", 4, {64, 128, 1, 1}},
    {"bias-011", 1, {128, 0, 0, 0}},
    {"scale-011", 1, {128, 0, 0, 0}},
    {"mean-011", 1, {128, 0, 0, 0}},
    {"var-011", 1, {128, 0, 0, 0}},
    {"w-011", 4, {128, 64, 3, 3}},
    {"bias-012", 1, {64, 0, 0, 0}},
    {"scale-012", 1, {64, 0, 0, 0}},
    {"mean-012", 1, {64, 0, 0, 0}},
    {"var-012", 1, {64, 0, 0, 0}},
    {"w-012", 4, {64, 128, 1, 1}},
    {"bias-014", 1, {64, 0, 0, 0}},
    {"scale-014", 1, {64, 0, 0, 0}},
    {"mean-014", 1, {64, 0, 0, 0}},
    {"var-014", 1, {64, 0, 0, 0}},
    {"w-014", 4, {64, 128, 1, 1}},
    {"bias-015", 1, {64, 0, 0, 0}},
    {"scale-015", 1, {64, 0, 0, 0}},
    {"mean-015", 1, {64, 0, 0, 0}},
    {"var-015", 1, {64, 0, 0, 0}},
    {"w-015", 4, {64, 64, 1, 1}},
    {"bias-016", 1, {64, 0, 0, 0}},
    {"scale-016", 1, {64, 0, 0, 0}},
    {"mean-016", 1, {64, 0, 0, 0}},
    {"var-016", 1, {64, 0, 0, 0}},
    {"w-016", 4, {64, 64, 3, 3}},
    {"bias-018", 1, {64, 0, 0, 0}},
    {"scale-018", 1, {64, 0, 0, 0}},
    {"mean-018", 1, {64, 0, 0, 0}},
    {"var-018", 1, {64, 0, 0, 0}},
    {"w-018", 4, {64, 64, 1, 1}},
    {"bias-019", 1, {64, 0, 0, 0}},
    {"scale-019", 1, {64, 0, 0, 0}},
    {"mean-019", 1, {64, 0, 0, 0}},
    {"var-019", 1, {64, 0, 0, 0}},
    {"w-019", 4, {64, 64, 3, 3}},
    {"bias-021", 1, {64, 0, 0, 0}},
    {"scale-021", 1, {64, 0, 0, 0}},
    {"mean-021", 1, {64, 0, 0, 0}},
    {"var-021", 1, {64, 0, 0, 0}},
    {"w-021", 4, {64, 64, 1, 1}},
    {"bias-023", 1, {128, 0, 0, 0}},
    {"scale-023", 1, {128, 0, 0, 0}},
    {"mean-023", 1, {128, 0, 0, 0}},
    {"var-023", 1, {128, 0, 0, 0}},
    {"w-023", 4, {128, 128, 1, 1}},
    {"bias-024", 1, {256, 0, 0, 0}},
    {"scale-024", 1, {256, 0, 0, 0}},
    {"mean-024", 1, {256, 0, 0, 0}},
    {"var-024", 1, {256, 0, 0, 0}},
    {"w-024", 4, {256, 128, 3, 3}},
    {"bias-025", 1, {128, 0, 0, 0}},
    {"scale-025", 1, {128, 0, 0, 0}},
    {"mean-025", 1, {128, 0, 0, 0}},
    {"var-025", 1, {128, 0, 0, 0}},
    {"w-025", 4, {128, 256, 1, 1}},
    {"bias-027", 1, {128, 0, 0, 0}},
    {"scale-027", 1, {128, 0, 0, 0}},
    {"mean-027", 1, {128, 0, 0, 0}},
    {"var-027", 1, {128, 0, 0, 0}},
    {"w-027", 4, {128, 256, 1, 1}},
    {"bias-028", 1, {128, 0, 0, 0}},
    {"scale-028", 1, {128, 0, 0, 0}},
    {"mean-028", 1, {128, 0, 0, 0}},
    {"var-028", 1, {128, 0, 0, 0}},
    {"w-028", 4, {128, 128, 1, 1}},
    {"bias-029", 1, {128, 0, 0, 0}},
    {"scale-029", 1, {128, 0, 0, 0}},
    {"mean-029", 1, {128, 0, 0, 0}},
    {"var-029", 1, {128, 0, 0, 0}},
    {"w-029", 4, {128, 128, 3, 3}},
    {"bias-031", 1, {128, 0, 0, 0}},
    {"scale-031", 1, {128, 0, 0, 0}},
    {"mean-031", 1, {128, 0, 0, 0}},
    {"var-031", 1, {128, 0, 0, 0}},
    {"w-031", 4, {128, 128, 1, 1}},
    {"bias-032", 1, {128, 0, 0, 0}},
    {"scale-032", 1, {128, 0, 0, 0}},
    {"mean-032", 1, {128, 0, 0, 0}},
    {"var-032", 1, {128, 0, 0, 0}},
    {"w-032", 4, {128, 128, 3, 3}},
    {"bias-034", 1, {128, 0, 0, 0}},
    {"scale-034", 1, {128, 0, 0, 0}},
    {"mean-034", 1, {128, 0, 0, 0}},
    {"var-034", 1, {128, 0, 0, 0}},
    {"w-034", 4, {128, 128, 1, 1}},
    {"bias-035", 1, {128, 0, 0, 0}},
    {"scale-035", 1, {128, 0, 0, 0}},
    {"mean-035", 1, {128, 0, 0, 0}},
    {"var-035", 1, {128, 0, 0, 0}},
    {"w-035", 4, {128, 128, 3, 3}},
    {"bias-037", 1, {128, 0, 0, 0}},
    {"scale-037", 1, {128, 0, 0, 0}},
    {"mean-037", 1, {128, 0, 0, 0}},
    {"var-037", 1, {128, 0, 0, 0}},
    {"w-037", 4, {128, 128, 1, 1}},
    {"bias-038", 1, {128, 0, 0, 0}},
    {"scale-038", 1, {128, 0, 0, 0}},
    {"mean-038", 1, {128, 0, 0, 0}},
    {"var-038", 1, {128, 0, 0, 0}},
    {"w-038", 4, {128, 128, 3, 3}},
    {"bias-040", 1, {128, 0, 0, 0}},
    {"scale-040", 1, {128, 0, 0, 0}},
    {"mean-040", 1, {128, 0, 0, 0}},
    {"var-040", 1, {128, 0, 0, 0}},
    {"w-040", 4, {128, 128, 1, 1}},
    {"bias-041", 1, {128, 0, 0, 0}},
    {"scale-041", 1, {128, 0, 0, 0}},
    {"mean-041", 1, {128, 0, 0, 0}},
    {"var-041", 1, {128, 0, 0, 0}},
    {"w-041", 4, {128, 128, 3, 3}},
    {"bias-043", 1, {128, 0, 0, 0}},
    {"scale-043", 1, {128, 0, 0, 0}},
    {"mean-043", 1, {128, 0, 0, 0}},
    {"var-043", 1, {128, 0, 0, 0}},
    {"w-043", 4, {128, 128, 1, 1}},
    {"bias-044", 1, {128, 0, 0, 0}},
    {"scale-044", 1, {128, 0, 0, 0}},
    {"mean-044", 1, {128, 0, 0, 0}},
    {"var-044", 1, {128, 0, 0, 0}},
    {"w-044", 4, {128, 128, 3, 3}},
    {"bias-046", 1, {128, 0, 0, 0}},
    {"scale-046", 1, {128, 0, 0, 0}},
    {"mean-046", 1, {128, 0, 0, 0}},
    {"var-046", 1, {128, 0, 0, 0}},
    {"w-046", 4, {128, 128, 1, 1}},
    {"bias-047", 1, {128, 0, 0, 0}},
    {"scale-047", 1, {128, 0, 0, 0}},
    {"mean-047", 1, {128, 0, 0, 0}},
    {"var-047", 1, {128, 0, 0, 0}},
    {"w-047", 4, {128, 128, 3, 3}},
    {"bias-049", 1, {128, 0, 0, 0}},
    {"scale-049", 1, {128, 0, 0, 0}},
    {"mean-049", 1, {128, 0, 0, 0}},
    {"var-049", 1, {128, 0, 0, 0}},
    {"w-049", 4, {128, 128, 1, 1}},
    {"bias-050", 1, {128, 0, 0, 0}},
    {"scale-050", 1, {128, 0, 0, 0}},
    {"mean-050", 1, {128, 0, 0, 0}},
    {"var-050", 1, {128, 0, 0, 0}},
    {"w-050", 4, {128, 128, 3, 3}},
    {"bias-052", 1, {128, 0, 0, 0}},
    {"scale-052", 1, {128, 0, 0, 0}},
    {"mean-052", 1, {128, 0, 0, 0}},
    {"var-052", 1, {128, 0, 0, 0}},
    {"w-052", 4, {128, 128, 1, 1}},
    {"bias-054", 1, {256, 0, 0, 0}},
    {"scale-054", 1, {256, 0, 0, 0}},
    {"mean-054", 1, {256, 0, 0, 0}},
    {"var-054", 1, {256, 0, 0, 0}},
    {"w-054", 4, {256, 256, 1, 1}},
    {"bias-055", 1, {512, 0, 0, 0}},
    {"scale-055", 1, {512, 0, 0, 0}},
    {"mean-055", 1, {512, 0, 0, 0}},
    {"var-055", 1, {512, 0, 0, 0}},
    {"w-055", 4, {512, 256, 3, 3}},
    {"bias-056", 1, {256, 0, 0, 0}},
    {"scale-056", 1, {256, 0, 0, 0}},
    {"mean-056", 1, {256, 0, 0, 0}},
    {"var-056", 1, {256, 0, 0, 0}},
    {"w-056", 4, {256, 512, 1, 1}},
    {"bias-058", 1, {256, 0, 0, 0}},
    {"scale-058", 1, {256, 0, 0, 0}},
    {"mean-058", 1, {256, 0, 0, 0}},
    {"var-058", 1, {256, 0, 0, 0}},
    {"w-058", 4, {256, 512, 1, 1}},
    {"bias-059", 1, {256, 0, 0, 0}},
    {"scale-059", 1, {256, 0, 0, 0}},
    {"mean-059", 1, {256, 0, 0, 0}},
    {"var-059", 1, {256, 0, 0, 0}},
    {"w-059", 4, {256, 256, 1, 1}},
    {"bias-060", 1, {256, 0, 0, 0}},
    {"scale-060", 1, {256, 0, 0, 0}},
    {"mean-060", 1, {256, 0, 0, 0}},
    {"var-060", 1, {256, 0, 0, 0}},
    {"w-060", 4, {256, 256, 3, 3}},
    {"bias-062", 1, {256, 0, 0, 0}},
    {"scale-062", 1, {256, 0, 0, 0}},
    {"mean-062", 1, {256, 0, 0, 0}},
    {"var-062", 1, {256, 0, 0, 0}},
    {"w-062", 4, {256, 256, 1, 1}},
    {"bias-063", 1, {256, 0, 0, 0}},
    {"scale-063", 1, {256, 0, 0, 0}},
    {"mean-063", 1, {256, 0, 0, 0}},
    {"var-063", 1, {256, 0, 0, 0}},
    {"w-063", 4, {256, 256, 3, 3}},
    {"bias-065", 1, {256, 0, 0, 0}},
    {"scale-065", 1, {256, 0, 0, 0}},
    {"mean-065", 1, {256, 0, 0, 0}},
    {"var-065", 1, {256, 0, 0, 0}},
    {"w-065", 4, {256, 256, 1, 1}},
    {"bias-066", 1, {256, 0, 0, 0}},
    {"scale-066", 1, {256, 0, 0, 0}},
    {"mean-066", 1, {256, 0, 0, 0}},
    {"var-066", 1, {256, 0, 0, 0}},
    {"w-066", 4, {256, 256, 3, 3}},
    {"bias-068", 1, {256, 0, 0, 0}},
    {"scale-068", 1, {256, 0, 0, 0}},
    {"mean-068", 1, {256, 0, 0, 0}},
    {"var-068", 1, {256, 0, 0, 0}},
    {"w-068", 4, {256, 256, 1, 1}},
    {"bias-069", 1, {256, 0, 0, 0}},
    {"scale-069", 1, {256, 0, 0, 0}},
    {"mean-069", 1, {256, 0, 0, 0}},
    {"var-069", 1, {256, 0, 0, 0}},
    {"w-069", 4, {256, 256, 3, 3}},
    {"bias-071", 1, {256, 0, 0, 0}},
    {"scale-071", 1, {256, 0, 0, 0}},
    {"mean-071", 1, {256, 0, 0, 0}},
    {"var-071", 1, {256, 0, 0, 0}},
    {"w-071", 4, {256, 256, 1, 1}},
    {"bias-072", 1, {256, 0, 0, 0}},
    {"scale-072", 1, {256, 0, 0, 0}},
    {"mean-072", 1, {256, 0, 0, 0}},
    {"var-072", 1, {256, 0, 0, 0}},
    {"w-072", 4, {256, 256, 3, 3}},
    {"bias-074", 1, {256, 0, 0, 0}},
    {"scale-074", 1, {256, 0, 0, 0}},
    {"mean-074", 1, {256, 0, 0, 0}},
    {"var-074", 1, {256, 0, 0, 0}},
    {"w-074", 4, {256, 256, 1, 1}},
    {"bias-075", 1, {256, 0, 0, 0}},
    {"scale-075", 1, {256, 0, 0, 0}},
    {"mean-075", 1, {256, 0, 0, 0}},
    {"var-075", 1, {256, 0, 0, 0}},
    {"w-075", 4, {256, 256, 3, 3}},
    {"bias-077", 1, {256, 0, 0, 0}},
    {"scale-077", 1, {256, 0, 0, 0}},
    {"mean-077", 1, {256, 0, 0, 0}},
    {"var-077", 1, {256, 0, 0, 0}},
    {"w-077", 4, {256, 256, 1, 1}},
    {"bias-078", 1, {256, 0, 0, 0}},
    {"scale-078", 1, {256, 0, 0, 0}},
    {"mean-078", 1, {256, 0, 0, 0}},
    {"var-078", 1, {256, 0, 0, 0}},
    {"w-078", 4, {256, 256, 3, 3}},
    {"bias-080", 1, {256, 0, 0, 0}},
    {"scale-080", 1, {256, 0, 0, 0}},
    {"mean-080", 1, {256, 0, 0, 0}},
    {"var-080", 1, {256, 0, 0, 0}},
    {"w-080", 4, {256, 256, 1, 1}},
    {"bias-081", 1, {256, 0, 0, 0}},
    {"scale-081", 1, {256, 0, 0, 0}},
    {"mean-081", 1, {256, 0, 0, 0}},
    {"var-081", 1, {256, 0, 0, 0}},
    {"w-081", 4, {256, 256, 3, 3}},
    {"bias-083", 1, {256, 0, 0, 0}},
    {"scale-083", 1, {256, 0, 0, 0}},
    {"mean-083", 1, {256, 0, 0, 0}},
    {"var-083", 1, {256, 0, 0, 0}},
    {"w-083", 4, {256, 256, 1, 1}},
    {"bias-085", 1, {512, 0, 0, 0}},
    {"scale-085", 1, {512, 0, 0, 0}},
    {"mean-085", 1, {512, 0, 0, 0}},
    {"var-085", 1, {512, 0, 0, 0}},
    {"w-085", 4, {512, 512, 1, 1}},
    {"bias-086", 1, {1024, 0, 0, 0}},
    {"scale-086", 1, {1024, 0, 0, 0}},
    {"mean-086", 1, {1024, 0, 0, 0}},
    {"var-086", 1, {1024, 0, 0, 0}},
    {"w-086", 4, {1024, 512, 3, 3}},
    {"bias-087", 1, {512, 0, 0, 0}},
    {"scale-087", 1, {512, 0, 0, 0}},
    {"mean-087", 1, {512, 0, 0, 0}},
    {"var-087", 1, {512, 0, 0, 0}},
    {"w-087", 4, {512, 1024, 1, 1}},
    {"bias-089", 1, {512, 0, 0, 0}},
    {"scale-089", 1, {512, 0, 0, 0}},
    {"mean-089", 1, {512, 0, 0, 0}},
    {"var-089", 1, {512, 0, 0, 0}},
    {"w-089", 4, {512, 1024, 1, 1}},
    {"bias-090", 1, {512, 0, 0, 0}},
    {"scale-090", 1, {512, 0, 0, 0}},
    {"mean-090", 1, {512, 0, 0, 0}},
    {"var-090", 1, {512, 0, 0, 0}},
    {"w-090", 4, {512, 512, 1, 1}},
    {"bias-091", 1, {512, 0, 0, 0}},
    {"scale-091", 1, {512, 0, 0, 0}},
    {"mean-091", 1, {512, 0, 0, 0}},
    {"var-091", 1, {512, 0, 0, 0}},
    {"w-091", 4, {512, 512, 3, 3}},
    {"bias-093", 1, {512, 0, 0, 0}},
    {"scale-093", 1, {512, 0, 0, 0}},
    {"mean-093", 1, {512, 0, 0, 0}},
    {"var-093", 1, {512, 0, 0, 0}},
    {"w-093", 4, {512, 512, 1, 1}},
    {"bias-094", 1, {512, 0, 0, 0}},
    {"scale-094", 1, {512, 0, 0, 0}},
    {"mean-094", 1, {512, 0, 0, 0}},
    {"var-094", 1, {512, 0, 0, 0}},
    {"w-094", 4, {512, 512, 3, 3}},
    {"bias-096", 1, {512, 0, 0, 0}},
    {"scale-096", 1, {512, 0, 0, 0}},
    {"mean-096", 1, {512, 0, 0, 0}},
    {"var-096", 1, {512, 0, 0, 0}},
    {"w-096", 4, {512, 512, 1, 1}},
    {"bias-097", 1, {512, 0, 0, 0}},
    {"scale-097", 1, {512, 0, 0, 0}},
    {"mean-097", 1, {512, 0, 0, 0}},
    {"var-097", 1, {512, 0, 0, 0}},
    {"w-097", 4, {512, 512, 3, 3}},
    {"bias-099", 1, {512, 0, 0, 0}},
    {"scale-099", 1, {512, 0, 0, 0}},
    {"mean-099", 1, {512, 0, 0, 0}},
    {"var-099", 1, {512, 0, 0, 0}},
    {"w-099", 4, {512, 512, 1, 1}},
    {"bias-100", 1, {512, 0, 0, 0}},
    {"scale-100", 1, {512, 0, 0, 0}},
    {"mean-100", 1, {512, 0, 0, 0}},
    {"var-100", 1, {512, 0, 0, 0}},
    {"w-100", 4, {512, 512, 3, 3}},
    {"bias-102", 1, {512, 0, 0, 0}},
    {"scale-102", 1, {512, 0, 0, 0}},
    {"mean-102", 1, {512, 0, 0, 0}},
    {"var-102", 1, {512, 0, 0, 0}},
    {"w-102", 4, {512, 512, 1, 1}},
    {"bias-104", 1, {1024, 0, 0, 0}},
    {"scale-104", 1, {1024, 0, 0, 0}},
    {"mean-104", 1, {1024, 0, 0, 0}},
    {"var-104", 1, {1024, 0, 0, 0}},
    {"w-104", 4, {1024, 1024, 1, 1}},
    {"bias-105", 1, {512, 0, 0, 0}},
    {"scale-105", 1, {512, 0, 0, 0}},
    {"mean-105", 1, {512, 0, 0, 0}},
    {"var-105", 1, {512, 0, 0, 0}},
    {"w-105", 4, {512, 1024, 1, 1}},
    {"bias-106", 1, {1024, 0, 0, 0}},
    {"scale-106", 1, {1024, 0, 0, 0}},
    {"mean-106", 1, {1024, 0, 0, 0}},
    {"var-106", 1, {1024, 0, 0, 0}},
    {"w-106", 4, {1024, 512, 3, 3}},
    {"bias-107", 1, {512, 0, 0, 0}},
    {"scale-107", 1, {512, 0, 0, 0}},
    {"mean-107", 1, {512, 0, 0, 0}},
    {"var-107", 1, {512, 0, 0, 0}},
    {"w-107", 4, {512, 1024, 1, 1}},
    {"bias-114", 1, {512, 0, 0, 0}},
    {"scale-114", 1, {512, 0, 0, 0}},
    {"mean-114", 1, {512, 0, 0, 0}},
    {"var-114", 1, {512, 0, 0, 0}},
    {"w-114", 4, {512, 2048, 1, 1}},
    {"bias-115", 1, {1024, 0, 0, 0}},
    {"scale-115", 1, {1024, 0, 0, 0}},
    {"mean-115", 1, {1024, 0, 0, 0}},
    {"var-115", 1, {1024, 0, 0, 0}},
    {"w-115", 4, {1024, 512, 3, 3}},
    {"bias-116", 1, {512, 0, 0, 0}},
    {"scale-116", 1, {512, 0, 0, 0}},
    {"mean-116", 1, {512, 0, 0, 0}},
    {"var-116", 1, {512, 0, 0, 0}},
    {"w-116", 4, {512, 1024, 1, 1}},
    {"bias-117", 1, {256, 0, 0, 0}},
    {"scale-117", 1, {256, 0, 0, 0}},
    {"mean-117", 1, {256, 0, 0, 0}},
    {"var-117", 1, {256, 0, 0, 0}},
    {"w-117", 4, {256, 512, 1, 1}},
    {"bias-120", 1, {256, 0, 0, 0}},
    {"scale-120", 1, {256, 0, 0, 0}},
    {"mean-120", 1, {256, 0, 0, 0}},
    {"var-120", 1, {256, 0, 0, 0}},
    {"w-120", 4, {256, 512, 1, 1}},
    {"bias-122", 1, {256, 0, 0, 0}},
    {"scale-122", 1, {256, 0, 0, 0}},
    {"mean-122", 1, {256, 0, 0, 0}},
    {"var-122", 1, {256, 0, 0, 0}},
    {"w-122", 4, {256, 512, 1, 1}},
    {"bias-123", 1, {512, 0, 0, 0}},
    {"scale-123", 1, {512, 0, 0, 0}},
    {"mean-123", 1, {512, 0, 0, 0}},
    {"var-123", 1, {512, 0, 0, 0}},
    {"w-123", 4, {512, 256, 3, 3}},
    {"bias-124", 1, {256, 0, 0, 0}},
    {"scale-124", 1, {256, 0, 0, 0}},
    {"mean-124", 1, {256, 0, 0, 0}},
    {"var-124", 1, {256, 0, 0, 0}},
    {"w-124", 4, {256, 512, 1, 1}},
    {"bias-125", 1, {512, 0, 0, 0}},
    {"scale-125", 1, {512, 0, 0, 0}},
    {"mean-125", 1, {512, 0, 0, 0}},
    {"var-125", 1, {512, 0, 0, 0}},
    {"w-125", 4, {512, 256, 3, 3}},
    {"bias-126", 1, {256, 0, 0, 0}},
    {"scale-126", 1, {256, 0, 0, 0}},
    {"mean-126", 1, {256, 0, 0, 0}},
    {"var-126", 1, {256, 0, 0, 0}},
    {"w-126", 4, {256, 512, 1, 1}},
    {"bias-127", 1, {128, 0, 0, 0}},
    {"scale-127", 1, {128, 0, 0, 0}},
    {"mean-127", 1, {128, 0, 0, 0}},
    {"var-127", 1, {128, 0, 0, 0}},
    {"w-127", 4, {128, 256, 1, 1}},
    {"bias-130", 1, {128, 0, 0, 0}},
    {"scale-130", 1, {128, 0, 0, 0}},
    {"mean-130", 1, {128, 0, 0, 0}},
    {"var-130", 1, {128, 0, 0, 0}},
    {"w-130", 4, {128, 256, 1, 1}},
    {"bias-132", 1, {128, 0, 0, 0}},
    {"scale-132", 1, {128, 0, 0, 0}},
    {"mean-132", 1, {128, 0, 0, 0}},
    {"var-132", 1, {128, 0, 0, 0}},
    {"w-132", 4, {128, 256, 1, 1}},
    {"bias-133", 1, {256, 0, 0, 0}},
    {"scale-133", 1, {256, 0, 0, 0}},
    {"mean-133", 1, {256, 0, 0, 0}},
    {"var-133", 1, {256, 0, 0, 0}},
    {"w-133", 4, {256, 128, 3, 3}},
    {"bias-134", 1, {128, 0, 0, 0}},
    {"scale-134", 1, {128, 0, 0, 0}},
    {"mean-134", 1, {128, 0, 0, 0}},
    {"var-134", 1, {128, 0, 0, 0}},
    {"w-134", 4, {128, 256, 1, 1}},
    {"bias-135", 1, {256, 0, 0, 0}},
    {"scale-135", 1, {256, 0, 0, 0}},
    {"mean-135", 1, {256, 0, 0, 0}},
    {"var-135", 1, {256, 0, 0, 0}},
    {"w-135", 4, {256, 128, 3, 3}},
    {"bias-136", 1, {128, 0, 0, 0}},
    {"scale-136", 1, {128, 0, 0, 0}},
    {"mean-136", 1, {128, 0, 0, 0}},
    {"var-136", 1, {128, 0, 0, 0}},
    {"w-136", 4, {128, 256, 1, 1}},
    {"bias-137", 1, {256, 0, 0, 0}},
    {"scale-137", 1, {256, 0, 0, 0}},
    {"mean-137", 1, {256, 0, 0, 0}},
    {"var-137", 1, {256, 0, 0, 0}},
    {"w-137", 4, {256, 128, 3, 3}},
    {"bias-138", 1, {255, 0, 0, 0}},
    {"w-138", 4, {255, 256, 1, 1}},
    {"bias-141", 1, {256, 0, 0, 0}},
    {"scale-141", 1, {256, 0, 0, 0}},
    {"mean-141", 1, {256, 0, 0, 0}},
    {"var-141", 1, {256, 0, 0, 0}},
    {"w-141", 4, {256, 128, 3, 3}},
    {"bias-143", 1, {256, 0, 0, 0}},
    {"scale-143", 1, {256, 0, 0, 0}},
    {"mean-143", 1, {256, 0, 0, 0}},
    {"var-143", 1, {256, 0, 0, 0}},
    {"w-143", 4, {256, 512, 1, 1}},
    {"bias-144", 1, {512, 0, 0, 0}},
    {"scale-144", 1, {512, 0, 0, 0}},
    {"mean-144", 1, {512, 0, 0, 0}},
    {"var-144", 1, {512, 0, 0, 0}},
    {"w-144", 4, {512, 256, 3, 3}},
    {"bias-145", 1, {256, 0, 0, 0}},
    {"scale-145", 1, {256, 0, 0, 0}},
    {"mean-145", 1, {256, 0, 0, 0}},
    {"var-145", 1, {256, 0, 0, 0}},
    {"w-145", 4, {256, 512, 1, 1}},
    {"bias-146", 1, {512, 0, 0, 0}},
    {"scale-146", 1, {512, 0, 0, 0}},
    {"mean-146", 1, {512, 0, 0, 0}},
    {"var-146", 1, {512, 0, 0, 0}},
    {"w-146", 4, {512, 256, 3, 3}},
    {"bias-147", 1, {256, 0, 0, 0}},
    {"scale-147", 1, {256, 0, 0, 0}},
    {"mean-147", 1, {256, 0, 0, 0}},
    {"var-147", 1, {256, 0, 0, 0}},
    {"w-147", 4, {256, 512, 1, 1}},
    {"bias-148", 1, {512, 0, 0, 0}},
    {"scale-148", 1, {512, 0, 0, 0}},
    {"mean-148", 1, {512, 0, 0, 0}},
    {"var-148", 1, {512, 0, 0, 0}},
    {"w-148", 4, {512, 256, 3, 3}},
    {"bias-149", 1, {255, 0, 0, 0}},
    {"w-149", 4, {255, 512, 1, 1}},
    {"bias-152", 1, {512, 0, 0, 0}},
    {"scale-152", 1, {512, 0, 0, 0}},
    {"mean-152", 1, {512, 0, 0, 0}},
    {"var-152", 1, {512, 0, 0, 0}},
    {"w-152", 4, {512, 256, 3, 3}},
    {"bias-154", 1, {512, 0, 0, 0}},
    {"scale-154", 1, {512, 0, 0, 0}},
    {"mean-154", 1, {512, 0, 0, 0}},
    {"var-154", 1, {512, 0, 0, 0}},
    {"w-154", 4, {512, 1024, 1, 1}},
    {"bias-155", 1, {1024, 0, 0, 0}},
    {"scale-155", 1, {1024, 0, 0, 0}},
    {"mean-155", 1, {1024, 0, 0, 0}},
    {"var-155", 1, {1024, 0, 0, 0}},
    {"w-155", 4, {1024, 512, 3, 3}},
    {"bias-156", 1, {512, 0, 0, 0}},
    {"scale-156", 1, {512, 0, 0, 0}},
    {"mean-156", 1, {512, 0, 0, 0}},
    {"var-156", 1, {512, 0, 0, 0}},
    {"w-156", 4, {512, 1024, 1, 1}},
    {"bias-157", 1, {1024, 0, 0, 0}},
    {"scale-157", 1, {1024, 0, 0, 0}},
    {"mean-157", 1, {1024, 0, 0, 0}},
    {"var-157", 1, {1024, 0, 0, 0}},
    {"w-157", 4, {1024, 512, 3, 3}},
    {"bias-158", 1, {512, 0, 0, 0}},
    {"scale-158", 1, {512, 0, 0, 0}},
    {"mean-158", 1, {512, 0, 0, 0}},
    {"var-158", 1, {512, 0, 0, 0}},
    {"w-158", 4, {512, 1024, 1, 1}},
    {"bias-159", 1, {1024, 0, 0, 0}},
    {"scale-159", 1, {1024, 0, 0, 0}},
    {"mean-159", 1, {1024, 0, 0, 0}},
    {"var-159", 1, {1024, 0, 0, 0}},
    {"w-159", 4, {1024, 512, 3, 3}},
    {"bias-160", 1, {255, 0, 0, 0}},
    {"w-160", 4, {255, 1024, 1, 1}},
};

/* The entry point takes one pointer per parameter, so its argument list has to
   be spelled out.  Rather than write it twice, enumerate the parameters once
   and let the preprocessor expand the enumeration into a declaration and into
   a call. */
#define ENTRY_PARAMS(X) \
  X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) \
  X(14) X(15) X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23) X(24) X(25) \
  X(26) X(27) X(28) X(29) X(30) X(31) X(32) X(33) X(34) X(35) X(36) X(37) \
  X(38) X(39) X(40) X(41) X(42) X(43) X(44) X(45) X(46) X(47) X(48) X(49) \
  X(50) X(51) X(52) X(53) X(54) X(55) X(56) X(57) X(58) X(59) X(60) X(61) \
  X(62) X(63) X(64) X(65) X(66) X(67) X(68) X(69) X(70) X(71) X(72) X(73) \
  X(74) X(75) X(76) X(77) X(78) X(79) X(80) X(81) X(82) X(83) X(84) X(85) \
  X(86) X(87) X(88) X(89) X(90) X(91) X(92) X(93) X(94) X(95) X(96) X(97) \
  X(98) X(99) X(100) X(101) X(102) X(103) X(104) X(105) X(106) X(107) X(108) \
  X(109) X(110) X(111) X(112) X(113) X(114) X(115) X(116) X(117) X(118) \
  X(119) X(120) X(121) X(122) X(123) X(124) X(125) X(126) X(127) X(128) \
  X(129) X(130) X(131) X(132) X(133) X(134) X(135) X(136) X(137) X(138) \
  X(139) X(140) X(141) X(142) X(143) X(144) X(145) X(146) X(147) X(148) \
  X(149) X(150) X(151) X(152) X(153) X(154) X(155) X(156) X(157) X(158) \
  X(159) X(160) X(161) X(162) X(163) X(164) X(165) X(166) X(167) X(168) \
  X(169) X(170) X(171) X(172) X(173) X(174) X(175) X(176) X(177) X(178) \
  X(179) X(180) X(181) X(182) X(183) X(184) X(185) X(186) X(187) X(188) \
  X(189) X(190) X(191) X(192) X(193) X(194) X(195) X(196) X(197) X(198) \
  X(199) X(200) X(201) X(202) X(203) X(204) X(205) X(206) X(207) X(208) \
  X(209) X(210) X(211) X(212) X(213) X(214) X(215) X(216) X(217) X(218) \
  X(219) X(220) X(221) X(222) X(223) X(224) X(225) X(226) X(227) X(228) \
  X(229) X(230) X(231) X(232) X(233) X(234) X(235) X(236) X(237) X(238) \
  X(239) X(240) X(241) X(242) X(243) X(244) X(245) X(246) X(247) X(248) \
  X(249) X(250) X(251) X(252) X(253) X(254) X(255) X(256) X(257) X(258) \
  X(259) X(260) X(261) X(262) X(263) X(264) X(265) X(266) X(267) X(268) \
  X(269) X(270) X(271) X(272) X(273) X(274) X(275) X(276) X(277) X(278) \
  X(279) X(280) X(281) X(282) X(283) X(284) X(285) X(286) X(287) X(288) \
  X(289) X(290) X(291) X(292) X(293) X(294) X(295) X(296) X(297) X(298) \
  X(299) X(300) X(301) X(302) X(303) X(304) X(305) X(306) X(307) X(308) \
  X(309) X(310) X(311) X(312) X(313) X(314) X(315) X(316) X(317) X(318) \
  X(319) X(320) X(321) X(322) X(323) X(324) X(325) X(326) X(327) X(328) \
  X(329) X(330) X(331) X(332) X(333) X(334) X(335) X(336) X(337) X(338) \
  X(339) X(340) X(341) X(342) X(343) X(344) X(345) X(346) X(347) X(348) \
  X(349) X(350) X(351) X(352) X(353) X(354) X(355) X(356) X(357) X(358) \
  X(359) X(360) X(361) X(362) X(363) X(364) X(365) X(366) X(367) X(368) \
  X(369) X(370) X(371) X(372) X(373) X(374) X(375) X(376) X(377) X(378) \
  X(379) X(380) X(381) X(382) X(383) X(384) X(385) X(386) X(387) X(388) \
  X(389) X(390) X(391) X(392) X(393) X(394) X(395) X(396) X(397) X(398) \
  X(399) X(400) X(401) X(402) X(403) X(404) X(405) X(406) X(407) X(408) \
  X(409) X(410) X(411) X(412) X(413) X(414) X(415) X(416) X(417) X(418) \
  X(419) X(420) X(421) X(422) X(423) X(424) X(425) X(426) X(427) X(428) \
  X(429) X(430) X(431) X(432) X(433) X(434) X(435) X(436) X(437) X(438) \
  X(439) X(440) X(441) X(442) X(443) X(444) X(445) X(446) X(447) X(448) \
  X(449) X(450) X(451) X(452) X(453) X(454) X(455) X(456) X(457) X(458) \
  X(459) X(460) X(461) X(462) X(463) X(464) X(465) X(466) X(467) X(468) \
  X(469) X(470) X(471) X(472) X(473) X(474) X(475) X(476) X(477) X(478) \
  X(479) X(480) X(481) X(482) X(483) X(484) X(485) X(486) X(487) X(488) \
  X(489) X(490) X(491) X(492) X(493) X(494) X(495) X(496) X(497) X(498) \
  X(499) X(500) X(501) X(502) X(503) X(504) X(505) X(506) X(507) X(508) \
  X(509) X(510) X(511) X(512) X(513) X(514) X(515) X(516) X(517) X(518) \
  X(519) X(520) X(521) X(522) X(523) X(524) X(525) X(526) X(527) X(528) \
  X(529) X(530) X(531) X(532) X(533) X(534) X(535) X(536) X(537) X(538) \
  X(539) X(540) X(541)

/* A rank-2 memref descriptor, as the compiled code returns it. */
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2D;

/* The C-compatible wrapper emitted by --llvm-request-c-wrappers.  It takes the
   result descriptor by pointer, then one pointer to a memref descriptor per
   parameter. */
#define DECLARE_PARAM(i) , void *p##i
extern void _mlir_ciface_entry_main(MemRef2D *result ENTRY_PARAMS(DECLARE_PARAM));
#undef DECLARE_PARAM

static void die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));

static void die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "yolov4: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}

static int64_t num_elems(const Param *p) {
  int64_t n = 1;
  for (int i = 0; i < p->rank; i++) {
    n *= p->dims[i];
  }
  return n;
}

static void *xmalloc(size_t n) {
  void *p = malloc(n);
  if (p == NULL) {
    die("out of memory (wanted %zu bytes)", n);
  }
  return p;
}

/* A rank-N memref descriptor is {allocated, aligned, offset, sizes[N],
   strides[N]}, and every field is eight bytes wide, so the whole thing is just
   an array of 3+2N words.  Building it that way lets one function serve all
   the ranks that occur in the parameter list. */
static void *make_desc(float *data, const Param *p) {
  int64_t *d = xmalloc(sizeof(int64_t) * (size_t)(3 + 2 * p->rank));
  d[0] = (int64_t)(intptr_t)data; /* allocated */
  d[1] = (int64_t)(intptr_t)data; /* aligned */
  d[2] = 0;                       /* offset */
  int64_t stride = 1;
  for (int i = p->rank - 1; i >= 0; i--) {
    d[3 + i] = p->dims[i];
    d[3 + p->rank + i] = stride;
    stride *= p->dims[i];
  }
  return d;
}

static FILE *open_read(const char *path, int64_t *size) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    die("cannot open %s: %s", path, strerror(errno));
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    die("cannot seek %s: %s", path, strerror(errno));
  }
  long end = ftell(f);
  if (end < 0) {
    die("cannot tell %s: %s", path, strerror(errno));
  }
  rewind(f);
  *size = end;
  return f;
}

static void read_param(FILE *f, const char *path, const Param *p, float *dst) {
  int64_t n = num_elems(p);
  if (fread(dst, sizeof(float), (size_t)n, f) != (size_t)n) {
    die("%s: unexpected end of file while reading %s", path, p->name);
  }
}

/* darknet writes major, minor and revision as int32, then the number of images
   seen during training: int64 from version 2 onwards, int32 before that
   (parser.c:1793-1802). */
static void skip_weights_header(FILE *f, const char *path, int64_t *header_size) {
  int32_t version[3];
  if (fread(version, sizeof(int32_t), 3, f) != 3) {
    die("%s: too short to be a darknet .weights file", path);
  }
  int64_t seen = (version[0] * 10 + version[1]) >= 2 ? 8 : 4;
  if (fseek(f, seen, SEEK_CUR) != 0) {
    die("%s: too short to be a darknet .weights file", path);
  }
  *header_size = 3 * 4 + seen;
}

static void usage(const char *prog, int status) {
  FILE *f = status == 0 ? stdout : stderr;
  fprintf(f, "usage: %s [-w WEIGHTS] [-i IMAGE] [-o OUTPUT] [-f]\n", prog);
  fprintf(f, "  -w WEIGHTS  darknet .weights file (default %s)\n", DEFAULT_WEIGHTS);
  fprintf(f, "  -i IMAGE    input image: %" PRId64 " raw f32 in CHW order (default %s)\n",
          num_elems(&params[0]), DEFAULT_IMAGE);
  fprintf(f, "  -o OUTPUT   where to write the %dx%d f32 detections, or - for\n",
          OUT_ROWS, OUT_COLS);
  fprintf(f, "              stdout (default %s)\n", DEFAULT_OUTPUT);
  fprintf(f, "  -f          write the output in Futhark's binary format\n");
  exit(status);
}

/* The header Futhark's binary data format puts in front of a 2D f32 array, so
   that the output can be fed straight to Futhark-based comparison tools. */
static void write_futhark_header(FILE *f) {
  unsigned char header[3 + 4 + 2 * 8] = {'b', 2, 2, ' ', 'f', '3', '2'};
  uint64_t dims[2] = {OUT_ROWS, OUT_COLS};
  for (int i = 0; i < 2; i++) {
    for (int b = 0; b < 8; b++) {
      header[7 + i * 8 + b] = (unsigned char)(dims[i] >> (8 * b));
    }
  }
  if (fwrite(header, 1, sizeof header, f) != sizeof header) {
    die("failed to write output header: %s", strerror(errno));
  }
}

int main(int argc, char **argv) {
  const char *weights_path = DEFAULT_WEIGHTS;
  const char *image_path = DEFAULT_IMAGE;
  const char *output_path = DEFAULT_OUTPUT;
  int futhark_format = 0;

  for (int i = 1; i < argc; i++) {
    const char *opt = argv[i];
    if (strcmp(opt, "-h") == 0 || strcmp(opt, "--help") == 0) {
      usage(argv[0], 0);
    } else if (strcmp(opt, "-f") == 0) {
      futhark_format = 1;
    } else if (i + 1 == argc) {
      usage(argv[0], 1);
    } else if (strcmp(opt, "-w") == 0) {
      weights_path = argv[++i];
    } else if (strcmp(opt, "-i") == 0) {
      image_path = argv[++i];
    } else if (strcmp(opt, "-o") == 0) {
      output_path = argv[++i];
    } else {
      usage(argv[0], 1);
    }
  }

  float *data[NUM_PARAMS];
  void *descs[NUM_PARAMS];

  /* The image. */
  int64_t image_size;
  FILE *image_file = open_read(image_path, &image_size);
  int64_t image_elems = num_elems(&params[0]);
  if (image_size != image_elems * (int64_t)sizeof(float)) {
    die("%s: is %" PRId64 " bytes, expected %" PRId64
        " (%" PRId64 " f32 in CHW order)",
        image_path, image_size, image_elems * (int64_t)sizeof(float), image_elems);
  }
  data[0] = xmalloc((size_t)image_size);
  read_param(image_file, image_path, &params[0], data[0]);
  fclose(image_file);

  /* The weights, which are simply the remaining parameters back to back. */
  int64_t weights_size, header_size;
  FILE *weights_file = open_read(weights_path, &weights_size);
  skip_weights_header(weights_file, weights_path, &header_size);

  int64_t expected = 0;
  for (int i = 1; i < NUM_PARAMS; i++) {
    expected += num_elems(&params[i]) * (int64_t)sizeof(float);
  }
  if (weights_size - header_size != expected) {
    die("%s: holds %" PRId64 " bytes of weights, expected %" PRId64
        " -- is this the yolov4.weights for this network?",
        weights_path, weights_size - header_size, expected);
  }

  for (int i = 1; i < NUM_PARAMS; i++) {
    data[i] = xmalloc((size_t)num_elems(&params[i]) * sizeof(float));
    read_param(weights_file, weights_path, &params[i], data[i]);
  }
  fclose(weights_file);

  for (int i = 0; i < NUM_PARAMS; i++) {
    descs[i] = make_desc(data[i], &params[i]);
  }

  fprintf(stderr, "yolov4: loaded %d parameter arrays; running inference\n", NUM_PARAMS);

  MemRef2D result;
#define PASS_PARAM(i) , descs[i]
  _mlir_ciface_entry_main(&result ENTRY_PARAMS(PASS_PARAM));
#undef PASS_PARAM

  /* We asked for identity layout maps when bufferizing, so the result should be
     a plain row-major array; check rather than silently write garbage. */
  if (result.sizes[0] != OUT_ROWS || result.sizes[1] != OUT_COLS ||
      result.offset != 0 || result.strides[0] != OUT_COLS || result.strides[1] != 1) {
    die("unexpected result layout: %" PRId64 "x%" PRId64
        " offset %" PRId64 " strides %" PRId64 ",%" PRId64,
        result.sizes[0], result.sizes[1], result.offset,
        result.strides[0], result.strides[1]);
  }

  FILE *out;
  if (strcmp(output_path, "-") == 0) {
    out = stdout;
  } else if ((out = fopen(output_path, "wb")) == NULL) {
    die("cannot open %s for writing: %s", output_path, strerror(errno));
  }
  if (futhark_format) {
    write_futhark_header(out);
  }
  size_t n = (size_t)OUT_ROWS * OUT_COLS;
  if (fwrite(result.aligned, sizeof(float), n, out) != n) {
    die("failed to write %s: %s", output_path, strerror(errno));
  }
  if (out != stdout && fclose(out) != 0) {
    die("failed to write %s: %s", output_path, strerror(errno));
  }

  fprintf(stderr, "yolov4: wrote %dx%d detections to %s\n",
          OUT_ROWS, OUT_COLS, output_path);

  free(result.allocated);
  for (int i = 0; i < NUM_PARAMS; i++) {
    free(data[i]);
    free(descs[i]);
  }
  return 0;
}
