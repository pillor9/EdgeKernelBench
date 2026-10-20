---
id: vivo_cpu/reference_families
hardware: vivo_cpu (V2403A, SM8650, AArch64 Android CPU)
tags: [reference-family, shape-regime, mechanism-contract]
layers: [L1]
evidence_basis: [vendor_doc, tvm_source, inferred]
version_sensitivity: high -- rebuild and remeasure after TVM, LLVM target, Android runtime, or affinity-policy change
---

# Vivo CPU: parameterized reference families

The current LLVM target is generic AArch64 Android. A source mechanism is eligible only when it lowers under
that frozen target; observed silicon features do not authorize a target flag
[leka-vivo-contract] [tvm-compatibility].

This page spans diagnosis across layers. L1 implementation authority includes
existing-body scheduling plus one closed 2–3-call canonical fusion region
jointly implemented in TIR under `leka.l1_fuse_of`. External interfaces and all
unrelated graph/ABIs stay frozen. Larger fusion/split, high-level algorithm-family
changes, and static-parameter packing/layout are L0-only advice.

Before selection define symbolic independent work `S`, reduction work `R`,
groups `B`, graph passes, stride/tails, static parameters, reuse distance,
working-set bytes, parallel ownership, and the configured runtime policy. Treat
cache capacity as unknown unless present in the live capability card.

## Elementwise and broadcast families

| family | eligible regime | mechanism | compatibility/status |
|---|---|---|---|
| E-contiguous | long stride-one traversal | parallelize a race-free outer range; vectorize a legal inner chunk; scalar tail | exact target; L1-local |
| E-fused | adjacent maps/broadcasts | keep intermediates in scalar/vector expressions inside an existing PrimFunc or declared closed 2–3-call fusion | L1 bounded local-fusion contract; larger regions go to L0 |
| E-small | little independent work | serial/vector traversal to avoid parallel scheduling cost | portable mechanism; L1-local |

Do not equate `T.vectorized` with SIMD success: inspect generated LLVM/assembly
or instruction counters, and derive width from dtype plus frozen target.

Refutation: instruction count/vector code improves but paired latency does not,
or parallel overhead/migrations dominate.

## Reduction families

- **R-output-parallel:** parallelize independent outputs; each worker owns the
  complete reduction and uses multiple accumulators to shorten dependency
  chains.
- **R-vector-serial:** for scarce outputs, keep the reduction in one worker
  with vector chunks plus a scalar tail before introducing thread partials.
- **R-thread-partials:** only when `R` is large enough to repay partial buffers
  and combine; pad worker slots to avoid false sharing and preserve one owner
  for the final store.

Correctness includes identity, tails, accumulation dtype, permitted
reassociation, and exceptional values.

Refutation: stall rate or dependency depth falls without latency improvement,
or partial-buffer traffic/synchronization grows.

## Matrix-product families

- **M-cache-panel:** partition symbolic `[M,N,K]` so reused panels have a
  testable cache-residency hypothesis; keep a register micro-tile and a
  contiguous vector coordinate.
- **M-scarce-axis:** parallelize the abundant batch/output axis and keep the
  scarce tile small enough to supply independent work to the selected policy.
- **M-batched-small:** parallelize groups while each small product uses a
  compact serial/vector microkernel.
- **M-static-pack (L0 only):** repack a declared static parameter outside timed execution
  and record the index bijection; dynamic packing remains timed.

Pinned TVM CPU GEMV provides loop-structure vocabulary only. It is neither a
measured SM8650 optimum nor a callable solution [tvm-dlight-cpu-gemv].

Refutation: the panel exceeds the actual cache regime, micro-tiles spill to the
stack, packing is not amortized, or worker placement becomes unstable.

## Convolution families

- **C-pointwise:** map positions/channels to a product family.
- **C-direct:** workers own output tiles; reuse a small input window and
  vectorize a contiguous output/channel coordinate.
- **C-channelwise:** direct traversal plus a light epilogue inside an existing
  PrimFunc or a declared closed 2–3-call fusion; larger regions go to L0.
  Avoid a large layout intermediate.
- **C-packed/product (L0 only when it changes packing/algorithm):** use only
  when static weights or repeated downstream reuse repay packing.
- **C-transform (L0 only):** escape-only with whole-graph and numerical validation.

Refutation: the aggregate cache-miss lead worsens, IPC/paired time regresses,
vectorization is lost, or the transform merely moves cost to another timed
PrimFunc. The portable counter set does not expose backend-stall cycles.

## Cross-layer fusion and layout routing

- **L1 bounded / L0 broader:** fuse maps or a light epilogue under the closed
  2–3-call contract to remove a memory round trip; larger regions go to L0.
- **L0 only:** split when fusion blocks vectorization, inflates code footprint, or repeats a
  producer.
- **L0 only:** static transpose/pad/interleave is compatible only for static parameters;
  dynamic relayout must repay a complete timed pass.
- **L1 only within an existing PrimFunc:** parallel chunks must have exclusive output ownership and avoid adjacent
  workers repeatedly writing the same cache line.

Refutation: the selected PrimFunc improves but whole-graph time does not, or
PMU evidence contradicts the expected instruction/cache effect.

## Proposal packet and validation

Record family, symbolic regime, explicit L0-or-L1 ownership, source/target
hardware, TVM/LLVM identity, compatibility, reproducibility state, expected
instruction/cache/parallel effect, numerical/tail/race obligations, runtime
policy, rollback parent, and Refutation. Promotion requires syntax validation,
target build, correctness, and paired device timing.
