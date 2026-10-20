---
id: L1_tir/vivo_cpu/reference_gemm_structures
hardware: vivo_cpu (Snapdragon 8 Gen 3 heterogeneous AArch64; frozen generic target)
layer: L1 (TIR loop structure — tiling/parallelization/unroll decisions)
evidence_basis: [arm_compute_library_upstream, kleidiai_upstream, xnnpack_upstream, apache_tvm_upstream]
note: mechanism vocabulary only; this page contains no benchmark-specific schedule.
      Physical optional ISA support does not widen the frozen compiler target.
---

# Reference GEMM mechanisms on generic AArch64

Apply these mechanisms to an already generated GEMM PrimFunc (or an equivalent
GEMM-shaped PrimFunc), or its declared closed 2–3-call local fusion under
`leka.l1_fuse_of`. The latter may join a compatible epilogue while co-designing
TIR; larger graph, algorithm, fixed-state, or external-interface changes go to L0.

Arm Compute Library, KleidiAI, and XNNPACK upstream kernels show recurring
implementation families: register-blocked output tiles, packed/contiguous
panels, multiple independent accumulators, and partitioning of independent
output tiles. Apache TVM's scheduling work likewise treats tile sizes, loop
order, vectorization, unrolling, and parallel decomposition as search choices.
These are structural references—not numeric schedules for this handset.

## Candidate mechanisms

- **Register block an output tile.** Keep a small independent M-by-N tile in
  local accumulators so each loaded reduction value can be reused. Derive tile
  shape from the exact dtype, generated LLVM, spill behavior, and paired device
  timing; do not copy a micro-kernel's numeric shape from another target.
- **Use multiple independent accumulators.** A long scalar/vector dependency
  chain may limit instruction-level parallelism. A tolerance-valid tree or
  multi-accumulator reduction is legal when the task's independent correctness
  gate passes; the optimizer must still check register pressure and spills.
- **Make the vector coordinate contiguous.** Vectorize only a proven unit-
  stride, divisible/aligned inner range and retain a correct tail otherwise.
  The generic AArch64 target—not observed silicon extensions—decides which
  instructions LLVM may emit.
- **Block reuse and test it.** Unless cache capacity is available from current runtime evidence,
  panel sizes are hypotheses to sweep. Do not label a tile L1/L2-resident
  without live evidence.
- **Partition independent output work.** The formal worker verifies an
  eight-thread TVM pool. Parallelize independent outer output/batch tiles, not
  a shared reduction axis. Eight threads do not imply homogeneous cores,
  stable affinity, or equal frequency.
- **Route packing upstream.** Changing static parameter layout or performing a
  declared one-time transpose/pack belongs to L0's parameter plan. Dynamic
  packing remains timed. L1 may only exploit layout already present in the
  canonical PrimFunc.

## Sizing fact, not a cache claim

A `256 x 256` f32 matrix contains `65,536` elements and is exactly `262,144`
bytes (`256 KiB`). Three such matrices total `768 KiB`, before scratch space.
Byte counts alone do not prove residency in any cache level; compare them
with current cache information and measure the resulting implementation.

Promotion still requires parser/target compilation, independent correctness,
fresh eight-thread runtime-policy validation, and paired physical-device
timing. A copied upstream numeric tile or optional-ISA kernel is not evidence.

## Upstream mechanism sources

- [Arm Compute Library AArch64 GEMM kernel families](https://github.com/ARM-software/ComputeLibrary/tree/main/src/core/NEON/kernels/arm_gemm/kernels)
- [Arm KleidiAI matmul micro-kernel families](https://github.com/ARM-software/kleidiai/tree/main/kai/ukernels/matmul)
- [Google XNNPACK f32 GEMM micro-kernel families](https://github.com/google/XNNPACK/tree/master/src/f32-gemm)
- [Apache TVM MetaSchedule / TensorIR RFC](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md)
