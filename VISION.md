# Development Vision for the Remora Compilation Pipeline Prototype

This document describes the plan for the development of the Remora compilation
pipeline. The ultimate goal is to compile Remora programs containing a mixture
of data ("SIMD") and task ("MIMD") parallelism to run on heterogeneous hardware.
In the short term, the goal is to compile Remora programs with data parallelism
to run on GPUs.

Our view of a Remora program is that it consists of a tree of task parallel
operations with data parallel operations at the leaves. That is, a task parallel
operation can contain data parallelism, but a data parallel operation cannot
contain task parallelism. The motivation behind this view is that it allows us
to compile the data parallel leaves with compilation technologies that cannot
handle the more irregular task parallelism.

The overall pipeline will look as follows:

1. [The Remora compiler](https://github.com/remora-lang/remora) implements type
   checking and other Remora-specific transformations, including identifying
   SIMD and MIMD parallelism. The output of this stage is two programs:

   * A Futhark program (in Futhark's SOACS IR) that contains entry points
     corresponding to all (pure) SIMD parallelism in the remora program.

   * A program in some other language that contains the MIMD parallelism, and
     contains calls to the Futhark program.

   In the short term, this second program is empty, meaning we only handle SIMD
   Remora programs that can be compiled entirely to Futhark. In the longer term,
   it doesn't matter particularly what language this second program is in, as we
   do not foresee compiling it in any particular way; e.g. making it a Scheme
   program with some reasonably flexible scheduler for task parallelism (and
   ability to call C code over an FFI) would perhaps be useful.

   Internally, this stage proceeds roughly as:

   ```
      typechecking
   -> monomorphization
   -> defunctionaliztion
   -> SIMD/MIMD analysis
   -> SIMD leaf transformation
   -> Futhark code generation
   ```

   A few of these components (or, at least, parts of them) remain open research
   problems:

   * Monomorphization. Instrumenting monomorphization to maximally specialize
     shapes (useful to compile to specialized kernels for, e.g., matrix
     multiplication of matrices of a known size) may involve doing a rank
     analysis over shapes and---where the rank is statically
     determinable---decomposing it into dimension variables to further
     specialize.

   * SIMD/MIMD analysis. Labels nodes of an AST as SIMD or MIMD
     computations. This work is in-progress on a separate
     [toy implementation](https://github.com/remora-lang/SIMD-MIMD), which is done on
     a CPS Remora-like toy IR. Open questions remain on whether we'll adapt a
     similar IR for the compiler proper and also how to adapt the work for the
     richer Remora language compared to the more constrained language in the toy
     implementation.

   * SIMD leaf transformation. Given an AST with nodes labeled as SIMD or MIMD
     from the analysis detailed above, how can we transform it to push all SIMD
     nodes into the leaves to get the tree structure described above?

2. [The Futhark compiler](https://github.com/diku-dk/futhark) will perform
   various optimisations and compile the Futhark SOACS IR to the *GPU* IR. The
   two main optimisations performed is loop fusion and *flattening* - the latter
   restructures the potentially nested parallelism of the input program to flat
   parallelism. Despite the name, the GPU IR is not particularly GPU-specific;
   it is named thus because it produces parallelism in a form that is
   straightforward to GPU code.

3. [The MLIR backend](https://github.com/remora-lang/mlir-backend) translates
   the optimised IR produced by the Futhark compiler to MLIR - this should be a
   reasonably mechanical operation, as we should target MLIR dialects that are a
   close fit for Futhark's GPU IR.

4. [MLIR](https://mlir.llvm.org/) then compiles the MLIR to (presumably) LLVM
   code, or directly to code for other target platforms.


For the prototype of the pipeline, we do not envision any modifications to MLIR
itself. All other components will be modified. The Futhark compiler will
hopefully only need light modification, except for one big part: the ability to
pass "black box primitives" (e.g. matmul) through the compiler such that we can
directly target corresponding MLIR ops.
