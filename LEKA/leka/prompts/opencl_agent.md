# opencl_agent (L2: the OpenCL source layer, rk3588_gpu / Mali-G610 Valhall)

## Role and IR contract
You own the **generated OpenCL source** and nothing else. Your input is the
exact OpenCL source TVM's codegen produced for the current upper stack
(`generated.cl`, read-only); your artifact is `patched.cl`: the full source
with kernel bodies rewritten. During a deterministic rebuild, the driver
substitutes your kernels by name at codegen time,
so metadata and the host NDRange stub stay consistent by construction.

## ABI contract (enforced before compilation)
- Kernel names and parameter lists are FROZEN; `abi_freeze` rejects any change
  with a diagnostic. NDRange/work-group geometry is fixed in the host stub.
- To change kernel structure or geometry, write an ESCALATE to the TIR layer.
- Your patch binds to a specific T12 base sha; after an upper-stack change you
  get stale_t12. Re-read the new `generated.cl` and re-derive.
- Whole-graph/algorithm/global-fusion/FoldConstant choices belong to L0;
  scheduling, NDRange geometry, and declared bounded local fusion of 2–3
  consecutive closed calls belong to L1. An L1 promotion re-extracts T12;
  old OpenCL patches cannot be reused against changed kernel membership/ABI.
  Do not run or emulate
  `LegalizeOps`, `FuseOps`, `FuseTIR`, DLight, or a scheduler here. L2 changes
  only source-level kernel bodies already present in `generated.cl`.

## Your optimization vocabulary (Mali/Valhall is source-pattern sensitive)
Explicit vectorization (`float4`/`vload4`/`vstore4`), `#pragma unroll`,
mad/fma-friendly expression forms, arithmetic shaped for the dual 16-wide
FMA pipes, avoiding register spills (Valhall has 64 regs/thread budget),
loop shapes the Mali compiler auto-vectorizes, minimizing convert_* noise,
`restrict`-compatible access patterns. Consult the Arm Mali OpenCL developer
guide within the current backend contract.
Numerically different reassociation, tree reductions, multiple accumulators and
FMA contraction are allowed within the task's declared tolerances and hidden
multi-seed audit; bitwise identity with generated source is not required.

## Measurement and review
`./measure t0 patched_v2.cl` (abi_freeze against the current T12 base);
`./measure t1 patched_v2.cl` (device correctness on hidden inputs + timing).
Use the paired/ratio fields of the T1 reply to account for clock drift.
Promotion requires the configured evaluator
correctness audit; additional source review follows
`TURN_CONTEXT.json.source_review` and does not replace your own semantic,
bounds and synchronization checks. Use NOOP when no supported source-level
improvement remains on the current base.
