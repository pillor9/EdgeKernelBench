---
id: rk3588_gpu/reference_families
hardware: rk3588_gpu (RK3588S, Mali-G610, OpenCL)
tags: [reference-family, shape-regime, mechanism-contract]
layers: [L1]
evidence_basis: [vendor_doc, tvm_source, inferred]
version_sensitivity: high -- rebuild and remeasure after TVM, OpenCL driver, kernel, or target-contract change
---

# RK3588 GPU: parameterized reference families

Arm guidance and pinned TVM source provide candidate mechanisms. Compile and
measure each implementation with the active OpenCL driver [arm-opencl-guide] [tvm-compatibility]. Do not transfer
CUDA warps/shared-memory assumptions or an Adreno schedule.

This page spans diagnosis across layers. L1 implementation authority includes
existing-body scheduling plus one closed 2–3-call canonical fusion region
jointly implemented in TIR under `leka.l1_fuse_of`. External interfaces and all
unrelated graph/ABIs stay frozen. Larger fusion/split, high-level algorithm-family
changes, and static-parameter packing/layout are L0-only advice.

Before selection define symbolic output work `S`, reduction work `R`, groups
`B`, graph stages, byte traffic, reuse, stride/alignment/tails, static
parameters, and legal live workgroup/local-memory limits
[leka-rk3588-contract].

## Elementwise and broadcast families

| family | eligible regime | mechanism | compatibility/status |
|---|---|---|---|
| E-map | abundant contiguous outputs | adjacent work-items own adjacent data; hoist invariants; predicate tails | portable Mali OpenCL; L1-local [arm-opencl-guide] |
| E-fused | adjacent light maps with materialized traffic | retain scalar intermediates inside an existing PrimFunc or declared closed 2–3-call fusion | L1 bounded local-fusion contract; larger regions go to L0 |
| E-grouped | many small independent maps already in one region | flatten group and position within the existing NDRange without cross-group dependence | portable mechanism; L1-local |

Vector width follows the live OpenCL C types, alignment, tail, and measured
code generation. The G610 product name does not prove one universally fastest
width. Local storage is useful only for cross-work-item reuse; on Mali's
unified-memory design, copying a value used once can add work
[arm-opencl-guide].

Refutation: event timing does not fall after the predicted launch or traffic
change, or Mali Offline Compiler shows spills/occupancy loss.

## Reduction families

- **R-output-parallel:** many outputs; one work-item or small cooperative group
  owns an output and uses independent accumulators.
- **R-workgroup:** scarce outputs and long `R`; work-items form private
  partials, use local storage for a uniform tree, and all reach every barrier.
- **R-multistage:** only when the extra partial buffer and dispatch are repaid
  by otherwise unavailable parallelism.

Pinned TVM reduction code supplies structure vocabulary, not G610 constants or
permission to invoke DLight [tvm-dlight-gpu-reduction].

Refutation: the static pipeline estimate improves but event time does not, or
dynamic counters show the assumed arithmetic/memory pressure was wrong.

## Matrix-product families

- **M-tiled:** abundant `[M,N]` work; workgroups own output tiles, private
  accumulators reuse operands, and local panels are introduced only with
  demonstrated reuse.
- **M-scarce-axis:** preserve enough workgroups by shrinking the scarce axis;
  stage only the repeatedly consumed operand and stream contiguous data.
- **M-batched-small:** flatten batch/group and tile coordinates while keeping
  synchronization workgroup-local.
- **M-static-pack (L0 only):** deterministically repack a declared static parameter
  outside timed execution and document its index relation.

The non-Adreno/default path of pinned TVM matmul and GEMV is a
`portable-mechanism` reference for this Linux-hosted Mali target, not an exact
schedule [tvm-dlight-gpu-matmul] [tvm-dlight-gpu-gemv].

Refutation: local traffic/barriers outweigh reuse, private registers spill, or
the layout copy enters the timed path.

## Convolution families

- **C-pointwise:** express positions/channels as a product family.
- **C-direct:** reuse a small input footprint across an output tile without
  materializing im2col.
- **C-channelwise:** direct traversal plus a light epilogue inside an existing
  PrimFunc or a declared closed 2–3-call fusion; larger regions go to L0.
  Local staging consumed once is rejected.
- **C-packed/product (L0 only when it changes packing/algorithm):** use only
  when static packing or downstream reuse repays the layout; dynamic
  transforms count in whole-graph time.
- **C-transform (L0 only):** escape-only, with numerical and workspace obligations.

Refutation: measured reuse is absent, event time rises despite a smaller inner
kernel, or the working set becomes bandwidth-limited elsewhere in the graph.

## Cross-layer fusion and layout routing

- **L1 bounded / L0 broader:** fuse map chains and light epilogues under the
  closed 2–3-call contract when it removes intermediate traffic and dispatch
  without excessive duplicated work; larger graph-region design goes to L0.
- **L0 only:** split when fusion increases private-register pressure, spills, or repeats a
  producer for several consumers.
- **L1 only when already one PrimFunc:** for transpose/layout, coalesce global accesses and use a local tile only if
  reuse/access regularity repays barriers [tvm-dlight-gpu-transpose]
  [arm-opencl-guide].
- **L0 only:** static pack is compatible only with declared static parameters; dynamic
  relayout is a timed candidate stage.

Refutation: a single kernel improves but whole-graph event time does not, or a
counter/trace window shows added memory/system pressure.

## Proposal packet and validation

Record family, shape predicate, explicit L0-or-L1 ownership, source/target
hardware, TVM/driver identity, compatibility, reproducibility state, expected
event/counter/static-code change, correctness/barrier obligations, rollback,
and Refutation. Never promote a static Mali Offline Compiler estimate as a
device speed claim [arm-malioc].
