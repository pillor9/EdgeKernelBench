# RK3588 GPU profiling recipe

This page describes the profiling digest emitted by LEKA.

Every action below is a routed recommendation to the layer that owns the
rewrite; the hardware-sense agent never edits an optimization artifact.
Route schedule/loop/memory-scope work and declared bounded local fusion of
one closed 2–3-call canonical region to L1; route broader graph changes to L0.
Route body-only emitted OpenCL changes to L2. Local-fusion suggestions must
respect `leka.l1_fuse_of`, unchanged external interfaces, and private intermediates.

## Digest contract

- `schema_version` is `leka.profile_digest.v2`.
- `primfunc_profile` contains same-measurement TVM VM/OpenCL per-operation
  timing rows. Runtime names, counts, duration and share are exact; static
  PrimFunc identity remains `unknown` unless a verified symbol map is present.
- `collector` follows `leka.collector_status.v1`. Production
  defaults to mode `off`, status `disabled_by_run_config`,
  `exact_bundle=false`; the TVM profile still remains available.
- Opt-in `devfreq` and `hwcpipe` modes sample a board/GPU-global workload
  window. They are useful environment context but currently remain
  `exact_bundle=false`: another process may contribute, and no complete
  candidate-to-counter attribution is claimed.

Consequently current RK deep counters must not be emitted as layer evidence.
Clock/utilization drift can be recorded as a hypothesis or an infrastructure
note, while all optimization decisions are validated by paired T1.

## Routing rules

1. Many short OpenCL rows suggest L1 bounded local fusion when one closed
   2–3-call region qualifies; broader fusion/kernel-boundary changes go to L0.
2. One dominant operation is a `hot_kernel` lead: workgroup, tiling, reduction,
   memory-scope and vector structure belong to L1; generated OpenCL body forms
   belong to L2.
3. Static serial-reduction extent is a reason to test an L1 tree/partition
   schedule, not proof that it is faster.
4. A layout/access-order concern that cannot be fixed inside the existing
   PrimFunc is routed to L0.

The recipe may be upgraded only after device validation proves exclusive
capture, full launch coverage and a hash-bound PrimFunc/symbol mapping. Until
then missing or global counters are explicitly unknown.
