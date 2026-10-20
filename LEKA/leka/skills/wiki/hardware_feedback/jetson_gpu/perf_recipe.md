# Jetson GPU profiling recipe

This page describes the profiling digest emitted by LEKA.

Every action below is a routed recommendation to the layer that owns the
rewrite; the hardware-sense agent never edits an optimization artifact.
Route schedule/loop/memory-scope work and declared bounded local fusion of
one closed 2–3-call canonical region to L1; route broader graph changes to L0.
Route body-only emitted CUDA changes to L2. Local-fusion suggestions must
respect `leka.l1_fuse_of`, unchanged external interfaces, and private intermediates.

## Digest contract

- `schema_version` is `leka.profile_digest.v2`.
- `primfunc_profile` is the TVM VM per-operation timing from the same frozen
  stack measurement. Its rows preserve runtime names, call counts, duration
  and share. Until a verified compiled-symbol map exists,
  `static_primfunc_id` is `unknown`.
- `collector` follows `leka.collector_status.v1`. Legal status
  values are `collected`, `unavailable`, and `disabled_by_run_config`.
  `exact_bundle=true` means the deep data is hash-bound to the exact candidate
  stack; absence or false is never interpreted as zero.
- Jetson's deep mode is `ncu`. It replays the exact frozen module/binding in an
  isolated trusted job and publishes NCU kernel rows only after CSV parsing and
  binding checks succeed. Its scope is `per_kernel_replay`, so it explains the
  same candidate but is not the timing authority for the original T1 sample.

The normalized NCU fields currently used by attribution are
`sm_throughput_pct`, `l2_throughput_pct`, `occupancy_pct`, and
`stall_long_scoreboard_pct`. Missing metrics remain unknown.

## Routing rules

1. Many short runtime rows suggest L1 bounded local fusion when one closed
   2–3-call region qualifies; broader kernel-boundary changes go to L0.
2. A large observed runtime row is a `hot_kernel` lead for L1/L2; NCU counters
   help choose a schedule/resource or emitted-source experiment. Kernel-to-VM
   mapping and uninstrumented wall-time coverage remain separate questions.
3. High L2 throughput with lower SM throughput supports a memory/coalescing
   lead. Loop/memory-scope structure belongs to L1; body-level load forms belong
   to L2.
4. Low occupancy or long-scoreboard pressure supports a resource/latency lead
   at L1, with a source-level follow-up at L2 only when launch structure is
   already suitable.
5. High SM throughput supports a compute/instruction lead. Low SM and L2
   throughput together may support an instruction-dependency hypothesis.

Long-scoreboard measures waits for L1TEX dependencies, including local, global,
surface and texture operations. The waiting consumer is not necessarily the
producer to optimize. This alone does not identify global/DRAM latency, and a
low L2 percent-of-peak does not rule out other memory limits. Throughput metrics
are relative to their own peak rates, not percentages of idle time; compare
stall classes before calling one dominant. Inspect source and launch facts to
choose one controlled test, keeping the unmeasured mechanism explicit.
[NVIDIA Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#statistical-sampler)

Only rows/counters copied verbatim from a collected exact-bundle digest may be
placed in `EVIDENCE.json`. Pairwise T1 remains the sole speed authority.
