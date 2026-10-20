---
id: jetson_gpu/reference_families
hardware: jetson_gpu (Jetson AGX Orin, CUDA sm_87)
tags: [reference-family, shape-regime, mechanism-contract]
layers: [L1]
evidence_basis: [vendor_doc, tvm_source, inferred]
version_sensitivity: high -- every mechanism must be rebuilt and remeasured after TVM, CUDA, driver, or target-contract change
---

# Jetson GPU: parameterized reference families

These are reference-only mechanism cards, not optimized kernels. Upstream TVM
DLight at the pinned commit is a language/mechanism reference; the candidate
contract forbids calling the framework scheduler as the solution
[tvm-compatibility] [tvm-dlight-gpu-matmul].

This page spans diagnosis across layers. L1 implementation authority includes
existing-body scheduling plus one closed 2–3-call canonical fusion region
jointly implemented in TIR under `leka.l1_fuse_of`. External interfaces and all
unrelated graph/ABIs stay frozen. Larger fusion/split/recomputation, high-level
algorithm-family changes, and static-parameter packing/layout are L0-only advice.

Before selection define symbolic output work `S`, reduction work `R`, batch or
group count `B`, materialized graph stages, byte traffic, reuse distance,
contiguous axes, alignment/tails, and whether each parameter is static. Classify
parallelism against the live SM count and resources, never against a borrowed
tile [leka-jetson-contract].

## Elementwise and broadcast families

| family | eligible regime | mechanism | compatibility/status |
|---|---|---|---|
| E-map | abundant independent contiguous outputs | lanes own adjacent elements; hoist broadcast invariants; predicate tails | portable CUDA mechanism; L1-local [nvidia-cuda-best-practices] |
| E-fused | adjacent maps or a light producer/consumer with materialized traffic | retain intermediates in registers inside an existing PrimFunc or declared closed 2–3-call fusion | L1 bounded local-fusion contract; larger regions go to L0 |
| E-batched | many small independent maps | flatten group and position within an existing launch without cross-group synchronization | portable mechanism; L1-local |

Use wider memory operations only after proving alignment, contiguity, dtype
legality, and tail safety in generated CUDA. Scalar lane accesses can already
coalesce; vector width is a candidate parameter, not a hardware fact
[nvidia-cuda-best-practices].

Refutation: launch count or transferred bytes changes as predicted but paired
latency does not improve, or fusion reduces resident work through register use.

## Reduction families

- **R-output-parallel:** many outputs and short/moderate `R`; one lane or a
  small lane group owns each output, with multiple independent accumulators to
  shorten dependency depth.
- **R-block:** scarce outputs and long `R`; one block owns an output fragment,
  lanes form register partials, and a uniform shared-memory tree combines
  them. Every thread reaches every barrier.
- **R-multistage:** only when one block cannot expose enough parallelism and
  the extra partial buffer plus launch is amortized.

These match the structure categories in pinned TVM reduction schedules but do
not import their constants [tvm-dlight-gpu-reduction]. Correctness must cover
identity, empty/tail behavior, accumulator dtype, permitted reassociation, and
NaN/Inf behavior.

Refutation: the dependency or occupancy signal improves without a paired
latency improvement; then extra synchronization/traffic or launch cost is the
limiter.

## Matrix-product families

For symbolic `[M,N,K]` products:

- **M-tiled:** abundant output tiles; cooperative contiguous panel loads,
  optional shared staging only with demonstrated reuse, register micro-tiles,
  and predicated boundary stores.
- **M-scarce-axis:** keep the scarce output dimension small enough to expose
  blocks; stage only the reused operand and stream the other contiguously.
- **M-batched-small:** flatten batch/group and tile coordinates to share a
  launch while keeping each product's ownership independent.
- **M-static-pack (L0 only):** transpose/pad/interleave a truly static parameter outside
  timed execution when it turns the hot coordinate contiguous; record the
  exact index bijection.

The pinned DLight matmul/GEMV files support these as portable mechanism
references, not as Orin measurements or permitted runtime calls
[tvm-dlight-gpu-matmul] [tvm-dlight-gpu-gemv].

Refutation: measured traffic falls but occupancy/residency collapses; a packed
layout is timed or reused too little; or a scarce-axis mapping leaves fewer
independent blocks than the untiled family.

## Convolution families

- **C-pointwise:** map spatial positions and channels to a product family.
- **C-direct:** a block owns an output tile and reuses a small input footprint;
  stage a halo only when multiple outputs consume it.
- **C-channelwise:** direct contiguous spatial/channel traversal with a light
  epilogue inside an existing PrimFunc or a declared closed 2–3-call fusion;
  larger regions go to L0. Avoid a shared copy consumed once.
- **C-transform (L0 only):** eligible only under escape when transform/workspace cost and
  numerical equivalence are evaluated for the whole graph.

Refutation: reuse is lower than assumed, halo traffic grows, or the transform
cost is merely shifted outside the inspected PrimFunc rather than outside the
timed contract.

## Cross-layer fusion and layout routing

- **L1 bounded / L0 broader:** map-to-map and light reduction epilogues may fuse
  under the closed 2–3-call contract when this removes dispatch/intermediate
  traffic without excessive duplicated work; larger graph-region design goes to L0.
- **L0 only:** a producer with several consumers may be recomputed only when the removed
  traffic is worth the arithmetic and register pressure.
- **L0 only:** static layout packing is compatible only for declared static parameters;
  dynamic relayout must repay a timed read/write pass.
- **L1 only when already one PrimFunc:** tiled transpose uses coalesced global access and a shared layout that avoids
  bank conflicts; derive resources from the live contract
  [tvm-dlight-gpu-transpose] [nvidia-cuda-best-practices].

Refutation: the whole graph does not improve even if the selected PrimFunc
does, or fusion changes a synchronization/aliasing boundary without proof.

## Proposal packet and validation

Record: family key; symbolic regime; explicit L0-or-L1 ownership; source hardware;
target hardware; TVM commit; `exact` or `portable-mechanism` compatibility;
current reproducibility state; expected launch/byte/instruction/resource
change; semantic and synchronization obligations; rollback parent; and one
measured refutation. Promote only through syntax validation, target build,
correctness, and paired device measurement.
