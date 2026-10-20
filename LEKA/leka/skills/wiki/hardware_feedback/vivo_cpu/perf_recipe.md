# Vivo CPU profiling recipe

This page describes the profiling digest emitted by LEKA.

Every action below is a routed recommendation to the layer that owns the
rewrite; the hardware-sense agent never edits an optimization artifact.
Route schedule/loop/memory-scope work and declared bounded local fusion of
one closed 2–3-call canonical region to L1; route broader graph changes to L0.
Route body-only emitted LLVM IR changes to L2. Local-fusion suggestions must
respect `leka.l1_fuse_of`, unchanged external interfaces, and private intermediates.

## Digest contract

- `schema_version` is `leka.profile_digest.v2`.
- `primfunc_profile` is the trusted, same-measurement TVM VM per-operation
  timing tool. It exposes each reported operation's name, call count, duration
  and share. Rows not beginning with `vm.` are classified
  `primfunc_or_compiled_call`; `static_primfunc_id` remains `unknown` until a
  verified static-to-runtime symbol map exists.
- The current deep mode is `simpleperf`. It samples the RPC process tree and
  retains the window with the highest cycle count. When the collector verifies
  the owned RPC generation before and after collection, its scope is
  `owned_rpc_process_tree_window`; otherwise it reports
  `rpc_process_window_without_pid_start_proof`. Both remain
  `exact_bundle=false`: process identity does not establish overlap with the
  candidate's timed execution or attribute counters to individual kernels.
- Simpleperf may expose whole-process `cpu_cycles`, `instructions`,
  `cache_references`, `cache_misses`, `branch_misses`, and derived `ipc`.
  These counters are context only: they cannot be attributed to one PrimFunc
  or sent as exact-bundle evidence.
- `runtime_context` separately records the configured and worker-observed
  thread count, every RPC task visible in the bounded post-measure snapshot,
  their CPU-allowed-list union, the main-thread affinity, per-core frequency
  and available thermal readings. This snapshot does not claim that ephemeral
  worker threads retained the same masks during execution. A single
  main-thread mask is never treated as the worker-pool mask; unobserved
  execution affinity remains `unknown`.
- `static_opportunities` gives value-free fixed-state layout/packing facts;
  `intermediate_traffic` gives call/PrimFunc/parallel/vector/allocation counts.
  Dynamic byte traffic is not synthesized.

## Routing rules

1. Many short operations or high VM share suggest L1 bounded local fusion when
   one closed 2–3-call region qualifies; broader fusion/CSE/boundary work goes to L0.
2. A dominant operation is a `hot_kernel` lead for L1/L2. Its exact timing row
   may be cited; whole-process simpleperf counters may not be attached to it.
3. Eight observed runtime threads with no explicit parallel TIR loop supports
   an L1 scheduling-granularity experiment. It does not prove balanced work or
   affinity on the heterogeneous cores.
4. Fixed rank-2-or-higher state may motivate an L0 one-time transpose/packing
   plan through the evaluator-owned parameter ABI.
5. Unknown cache capacity means working-set and blocking choices must be swept
   on device. Do not assume a cache size from the product class.

A future per-PrimFunc PMU tool must read one inherited event group immediately
around non-overlapping synchronous calls and prove a hash-bound static/runtime
mapping. Until this mapping is verified, per-PrimFunc timing is available but
per-PrimFunc hardware counters are not. Pairwise T1 is the speed authority.
