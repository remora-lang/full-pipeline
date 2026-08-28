#!/usr/bin/env janet
#
# Generate a C API for a Remora program from the func.func signatures in the
# MLIR that mlir-backend emits:
#
#   NAME.h            what a C program using the library includes
#   NAME_bindings.c   the implementation, compiled into the library
#
# The API follows Futhark's:
#
#   remora_context_new / remora_context_free / remora_context_get_error
#   remora_new_<elem>_<rank>d / remora_free_ / remora_values_ / remora_shape_
#   remora_entry_<name>
#
# Arrays are opaque handles, so a caller never sees what the target can read. A
# handle holds whatever remora_buffer_alloc gives out: malloc on the CPU, pinned
# host memory on a GPU. The generated array operations forward to
# runtime/remora_runtime.c, which does the work at any rank.

# MLIR element type -> C type. The name on the left also goes into the array
# type names, exactly as it does in Futhark (futhark_f32_1d).
(def ctypes
  {"i8" "int8_t" "i16" "int16_t" "i32" "int32_t" "i64" "int64_t"
   "f32" "float" "f64" "double"})

(def max-rank 8) # REMORA_MAX_RANK in runtime/remora_internal.h

(defn fail [fmt & args]
  (error (string/format fmt ;args)))

(defn basename [path] (last (string/split "/" path)))

# A type is {:elem "i64" :dims [4 8]}, or {:elem "i64"} for a scalar.
(defn array-type? [t] (truthy? (t :dims)))
(defn ctype [t] (ctypes (t :elem)))
(defn rank [t] (length (t :dims)))
(defn array-name [t] (string (t :elem) "_" (rank t) "d"))

### Reading the module ###############################################

(def signatures
  "Every entry point as [name [argument-type ...] result-type], as written."
  (peg/compile
    ~{:main (any (+ (group :entry) 1))

      :ident (some (+ :w (set "_.")))

      # Balanced brackets, so an attribute dictionary can be skipped whole.
      # "->" is not a closing bracket.
      :group (+ (* "(" (any :item) ")") (* "<" (any :item) ">")
                (* "[" (any :item) "]") (* "{" (any :item) "}"))
      :item (+ "->" :group (if-not (set "()<>[]{}") 1))

      # A type ends at a space, comma or brace. Anything bracketed is part of it.
      :type (<- (some (+ :group (if-not (+ :s (set "()<>[]{},")) 1))))

      # "%name : type {attributes}". A declaration may omit the name.
      :arg (* :s* (? (* "%" :ident :s* ":" :s*)) :type :s* (? :group) :s*)

      :entry (* "func.func" :s+ (? (* :w+ :s+)) "@" (<- (* "entry_" :ident))
                :s* "(" (group (? (* :arg (any (* "," :arg))))) ")" :s*
                (+ (* "->" :s* :type) (constant "")))}))

# Peel the dimensions off the front rather than splitting on every 'x', so that
# an element type ending in one -- 'index' -- still comes out whole.
(def tensor
  (peg/compile
    ~(* "tensor<" (group (any (* (<- (some (+ :d "?"))) "x")))
        (<- (some (if-not ">" 1))) ">" -1)))

(defn parse-type [text what]
  (if-not (string/has-prefix? "tensor<" text)
    (do (unless (ctypes text) (fail "%s has unsupported type '%s'" what text))
        {:elem text})
    (let [m (peg/match tensor text)]
      (unless m (fail "%s has unsupported type '%s'" what text))
      (def [dims elem] m)
      (unless (ctypes elem)
        (fail "%s has unsupported element type '%s'" what elem))
      (when (empty? dims)
        (fail "%s is rank-0, which has no C representation here" what))
      (when (> (length dims) max-rank)
        (fail "%s has rank %d, over REMORA_MAX_RANK" what (length dims)))
      (when (some |(string/find "?" $) dims)
        (fail "%s has a non-constant dimension in '%s'; the C API needs static shapes"
              what text))
      {:elem elem :dims (tuple ;(map scan-number dims))})))

(defn entry-point [[name args ret]]
  (when (empty? ret)
    (fail "entry point %s returns nothing, which the C API does not support" name))
  (when (string/has-prefix? "(" ret)
    (fail "entry point %s returns several values, which the C API does not support yet"
          name))
  {:name name
   :ret (parse-type ret (string/format "the result of entry point %s" name))
   :args (seq [[i arg] :pairs args]
           (def what (string/format "argument %d of entry point %s" i name))
           (def t (parse-type arg what))
           (unless (array-type? t)
             (fail "%s is not an array, which the C API does not support yet" what))
           t)})

### Emitting #########################################################

(defn fill [template &keys subs]
  "Replace each $key in `template`. Longest first, so that $dim cannot eat the
  front of a $dimlist."
  (var out template)
  (each k (sorted (keys subs) |(> (length $0) (length $1)))
    (set out (string/replace-all (string "$" k) (string (subs k)) out)))
  out)

(defn file [& blocks]
  (string (string/join blocks "\n\n") "\n"))

(defn signature [head params]
  (string head "(\n" (string/join (map |(string "    " $) params) ",\n") ")"))

(defn public-params [e]
  "The context, then the result, then the arguments."
  (def {:ret ret :args args} e)
  ["struct remora_context *ctx"
   (if (array-type? ret)
     (string "struct remora_" (array-name ret) " **out0")
     (string (ctype ret) " *out0"))
   ;(seq [[i a] :pairs args]
      (string "const struct remora_" (array-name a) " *in" i))])

(defn dim-params [t]
  (string ;(seq [i :range [0 (rank t)]] (string ", int64_t dim" i))))

(defn dim-args [t]
  (string/join (seq [i :range [0 (rank t)]] (string "dim" i)) ", "))

### Header ###########################################################

(def context-decls
  ``
  // Create one, hand it to everything, free it last.
  struct remora_context;
  struct remora_context *remora_context_new(void);
  void remora_context_free(struct remora_context *ctx);

  // The last error, or NULL. Valid until the next failing call.
  const char *remora_context_get_error(const struct remora_context *ctx);``)

(def array-decls
  ``
  // $ctype arrays of rank $rank. remora_new_ copies data in, so the pointer can
  // be anything. remora_values_ copies back out into dense row-major memory you
  // own and have made large enough; remora_shape_ says how large.
  struct remora_$name;
  struct remora_$name *remora_new_$name(
      struct remora_context *ctx, const $ctype *data$dimparams);
  int remora_free_$name(struct remora_context *ctx, struct remora_$name *arr);
  int remora_values_$name(
      struct remora_context *ctx, const struct remora_$name *arr, $ctype *data);
  const int64_t *remora_shape_$name(
      struct remora_context *ctx, const struct remora_$name *arr);``)

(def entry-intro
  ``
  // Entry points. Each returns 0, or -1 with the reason in
  // remora_context_get_error. An array result is yours to free.``)

(defn guard-name [path]
  (string "REMORA_"
          (string/ascii-upper
            (string (peg/replace-all ~(if-not :w 1) "_" (basename path))))
          "_"))

(defn header [mlir-name guard arrays entries]
  (file
    (string "// Generated by remora-cbindings.janet from " mlir-name ". Do not edit.")
    (string "#ifndef " guard "\n#define " guard)
    "#include <stdint.h>"
    "#ifdef __cplusplus\nextern \"C\" {\n#endif"
    context-decls
    ;(seq [t :in arrays]
       (fill array-decls :name (array-name t) :ctype (ctype t) :rank (rank t)
             :dimparams (dim-params t)))
    (string/join
      [entry-intro
       ;(seq [e :in entries]
          (string (signature (string "int remora_" (e :name))
                             (public-params e)) ";"))]
      "\n")
    "#ifdef __cplusplus\n}\n#endif"
    "#endif"))

### Bindings #########################################################

(def array-defs
  ``
  // The handle is the memref descriptor itself, so it passes straight to the
  // entry point. Layout must match what --llvm-request-c-wrappers expects.
  struct remora_$name {
    $ctype *allocated;
    $ctype *aligned;
    int64_t offset;
    int64_t sizes[$rank];
    int64_t strides[$rank];
  };
  _Static_assert(sizeof(struct remora_$name) ==
                     sizeof(struct remora_desc) + 2 * $rank * sizeof(int64_t),
                 "struct remora_$name is not a rank-$rank memref descriptor");

  struct remora_$name *remora_new_$name(
      struct remora_context *ctx, const $ctype *data$dimparams) {
    const int64_t dims[] = {$dimlist};
    return (struct remora_$name *)remora_array_new(
        ctx, "remora_new_$name", data, $rank, dims, sizeof($ctype));
  }

  int remora_free_$name(struct remora_context *ctx, struct remora_$name *arr) {
    (void)ctx;
    remora_array_free((struct remora_desc *)arr);
    return 0;
  }

  int remora_values_$name(
      struct remora_context *ctx, const struct remora_$name *arr, $ctype *data) {
    (void)ctx;
    remora_array_values(data, (const struct remora_desc *)arr, $rank,
                        sizeof($ctype));
    return 0;
  }

  const int64_t *remora_shape_$name(
      struct remora_context *ctx, const struct remora_$name *arr) {
    (void)ctx;
    return arr->sizes;
  }``)

# Shapes are baked into the compiled code, so a wrong extent would read off the
# end of the buffer rather than fail.
(def extent-check
  ``

    if (in$arg->sizes[$dim] != $extent) {
      return remora_set_error(ctx,
                              "remora_$name: argument $arg has extent %lld in dimension $dim, expected $extent",
                              (long long)in$arg->sizes[$dim]);
    }``)

# A result sharing a buffer with an argument would be freed twice. Nothing
# emitted today does this; returning an argument would.
(def alias-check
  ``

    if (result.allocated == in$arg->allocated) {
      return remora_set_error(ctx,
                              "remora_$name: result aliases argument $arg, which the C API does not support");
    }``)

(def array-entry
  ``
  $checks
    struct remora_$ret result;
    _mlir_ciface_$name($call);$aliases
    struct remora_$ret *handle = malloc(sizeof *handle);
    if (handle == NULL) {
      remora_buffer_free(result.allocated);
      return remora_set_error(ctx, "remora_$name: out of memory");
    }
    *handle = result;
    *out0 = handle;
    return 0;
  }``)

(def scalar-entry
  ``

    (void)ctx;$checks
    *out0 = _mlir_ciface_$name($call);
    return 0;
  }``)

(defn entry-def [e]
  "The extern declaration of an entry point, and the wrapper around it."
  (def {:name name :ret ret :args args} e)
  (def returns-array (array-type? ret))
  (def ins (seq [[i a] :pairs args]
             (string "struct remora_" (array-name a) " *in" i)))

  # --llvm-request-c-wrappers passes every memref as a pointer to its
  # descriptor, and returns an array result through a descriptor prepended to
  # the arguments. A scalar result comes back the ordinary C way instead.
  (def extern
    (signature
      (string "extern " (if returns-array "void" (ctype ret))
              " _mlir_ciface_" name)
      (cond
        returns-array [(string "struct remora_" (array-name ret) " *out0") ;ins]
        (empty? ins) ["void"]
        ins)))

  (def call
    (string/join [;(if returns-array ["&result"] [])
                  ;(seq [[i a] :pairs args]
                     (string "(struct remora_" (array-name a) " *)in" i))]
                 ", "))
  (def checks
    (string ;(seq [[i a] :pairs args [d n] :pairs (a :dims)]
               (fill extent-check :name name :arg i :dim d :extent n))))

  (string extern ";\n\n"
          (signature (string "int remora_" name) (public-params e)) " {"
          (if returns-array
            (fill array-entry :name name :ret (array-name ret) :call call
                  :checks checks
                  :aliases (string ;(seq [i :range [0 (length args)]]
                                      (fill alias-check :name name :arg i))))
            (fill scalar-entry :name name :call call :checks checks))))

(defn bindings [mlir-name header-name arrays entries]
  (file
    (string "// Generated by remora-cbindings.janet from " mlir-name ". Do not edit.")
    (string "#include \"" header-name "\"\n#include \"remora_internal.h\"")
    "#include <stdlib.h>"
    ;(seq [t :in arrays]
       (fill array-defs :name (array-name t) :ctype (ctype t) :rank (rank t)
             :dimparams (dim-params t) :dimlist (dim-args t)))
    ;(map entry-def entries)))

### Driver ###########################################################

(defn generate [mlir header-file bindings-file]
  (def entries (map entry-point (peg/match signatures (slurp mlir))))
  (when (empty? entries) (fail "no entry points found in %s" mlir))

  # Every array type the entry points mention, in the order first seen. Keyed
  # by name rather than by value: the generated struct and its operations use
  # only the element type and the rank, so two shapes of the same rank -- a
  # tensor<32xf32> and a tensor<64xf32> -- share one remora_f32_1d.
  (def arrays @[])
  (def seen @{})
  (each t (filter array-type? (mapcat |[;($ :args) ($ :ret)] entries))
    (def name (array-name t))
    (unless (seen name)
      (put seen name true)
      (array/push arrays t)))

  # Both files or neither: a half-written pair would confuse the next build.
  (def h (header (basename mlir) (guard-name header-file) arrays entries))
  (def b (bindings (basename mlir) (basename header-file) arrays entries))
  (spit header-file h)
  (spit bindings-file b))

(defn main [prog & args]
  (try
    (do
      (unless (= 3 (length args))
        (fail "usage: %s MLIR HEADER BINDINGS" (basename prog)))
      (generate ;args))
    ([err] (eprintf "%s: %s" (basename prog) err) (os/exit 1))))
