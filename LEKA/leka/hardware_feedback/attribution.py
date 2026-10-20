"""Per-backend attribution tables: profiler verdict -> owning layer.

Combine Relax VM timings with available NCU, devfreq, and simpleperf counters.
The resulting hypotheses guide the hardware agent; cited measurements are
checked against the collected data before delivery to a layer agent.
"""

from __future__ import annotations

from .quality import finite_number

VERDICT_TO_LAYER = {
    "jetson_gpu": {
        "launch_overhead": "L0",
        "hot_kernel": ["L1", "L2"],
        "memory_bandwidth": "L1",
        "latency_or_occupancy": "L1",
        "compute": "L2",
        "instruction_dependency": ["L1", "L2"],
        "vm_overhead": "L0",
    },
    "rk3588_gpu": {
        "launch_overhead": "L0",
        "hot_kernel": ["L1", "L2"],
        "memory_bandwidth": "L1",
        "compute": "L2",
        "compiler_pattern": "L2",   # Mali source-pattern sensitivity
        "vm_overhead": "L0",
        "clock_drift_telemetry": "infra",  # context for ratios, not an optimization target
    },
    "vivo_cpu": {
        "launch_overhead": "L0",
        "hot_kernel": ["L1", "L2"],
        "memory_bandwidth": ["L0", "L1"],
        "instruction_dependency": ["L1", "L2"],
        "static_weight_layout": "L0",
        "graph_redundancy": "L0",
        "schedule_granularity": "L1",
        "vectorization": "L1",
        "parallelization": "L1",
        "instruction_mix": "L2",
        "vm_overhead": "L0",
        "big_little_migration": "infra",  # runtime placement, not an optimization layer
    },
}


def counter_hypotheses(backend: str, counters: dict, *, rows: list[dict] | None = None,
                       profile_row_offset: int = 0) -> list[dict]:
    """Conservative routing hypotheses, never proof of a causal bottleneck.

    Missing counters are not zeros. Neither utilization nor cache misses alone
    establish bandwidth saturation; retain the legacy routing vocabulary but
    state the actual observation and competing explanations explicitly.
    """
    if backend == "jetson_gpu" and rows is not None:
        result = []
        for index, row in enumerate(rows):
            if (not isinstance(row, dict) or finite_number(row.get("kernel_time_ns_sum")) is None
                    or row.get("kernel_time_ns_sum", 0) <= 0):
                continue
            # Do not pair SM from one kernel with L2 from another. A per-metric
            # aggregate may have different coverage for every counter field.
            for hypothesis in counter_hypotheses(backend, row):
                hypothesis.update(evidence_scope="one_ncu_kernel_group",
                    evidence_rows=[row.get("name")], evidence_row_indices=[profile_row_offset + index],
                    evidence_index_space="PROFILE_DIGEST.profile_breakdown",
                    ncu_group_id=row.get("ncu_group_id"),
                    replay_kernel_time_percent=row.get("replay_kernel_time_percent"),
                    static_primfunc_id="unknown", whole_graph_bottleneck_confirmed=False)
                hypothesis["note"] = (f"NCU group '{row.get('name')}' only: " + hypothesis["note"] +
                    "; replay share does not establish VM/wall-time share or a whole-graph bottleneck")
                result.append(hypothesis)
        return result
    out: list[dict] = []
    if not counters:
        return out
    if backend == "jetson_gpu":
        sm = finite_number(counters.get("sm_throughput_pct"), maximum=100)
        l2 = finite_number(counters.get("l2_throughput_pct"), maximum=100)
        occ = finite_number(counters.get("occupancy_pct"), maximum=100)
        stall = finite_number(counters.get("stall_long_scoreboard_pct"), maximum=100)
        if l2 is not None and sm is not None and l2 >= 60 and sm < 50:
            out.append({"verdict": "memory_bandwidth", "confidence": 0.6,
                        "evidence_fields": ["l2_throughput_pct", "sm_throughput_pct"],
                        "note": f"L2 pipe {l2:.0f}% with SM {sm:.0f}%: L2 pressure "
                                "is a locality/coalescing lead at L1; DRAM bandwidth "
                                "saturation was not measured"})
        if sm is not None and sm >= 60:
            out.append({"verdict": "compute", "confidence": 0.7,
                        "evidence_fields": ["sm_throughput_pct"],
                        "note": f"SM {sm:.0f}%: inspect per-pipeline instruction mix "
                                "at L2; aggregate SM utilization does not identify "
                                "the limiting instruction unit"})
        if (occ is not None and occ < 40) or (stall is not None and stall >= 30):
            out.append({"verdict": "latency_or_occupancy", "confidence": 0.6,
                        "evidence_fields": [key for key, value in (
                            ("occupancy_pct", occ), ("stall_long_scoreboard_pct", stall)
                        ) if value is not None],
                        "note": f"occupancy {occ}, long-scoreboard {stall}: more blocks, "
                                "fewer regs/shared per block, prefetch are L1 "
                                "experiments; low occupancy alone is not a cause, "
                                "and scoreboard waits are not bandwidth proof"})
        if (sm is not None and sm < 25) and (l2 is not None and l2 < 25):
            out.append({"verdict": "instruction_dependency", "confidence": 0.6,
                        "evidence_fields": ["sm_throughput_pct", "l2_throughput_pct"],
                        "note": "both pipes underutilized: check grid size, short "
                                "kernels, latency hiding and dependency chains; "
                                "serial dependency was not measured"})
    elif backend == "vivo_cpu":
        ipc = finite_number(counters.get("ipc"))
        refs = finite_number(counters.get("cache_references"))
        misses = finite_number(counters.get("cache_misses"))
        miss_ratio = (misses / refs) if refs and misses is not None and misses <= refs else None
        if miss_ratio is not None and miss_ratio > 0.30:
            out.append({"verdict": "memory_bandwidth", "confidence": 0.6,
                        "evidence_fields": ["cache_misses", "cache_references"],
                        "note": f"aggregate cache miss ratio {miss_ratio:.2f}: a "
                                "cache-pressure lead only; locality/tiling belongs to "
                                "L1 and fixed-state packing may ESCALATE to L0"})
        if ipc is not None and ipc < 1.0 and miss_ratio is not None and miss_ratio <= 0.30:
            fields = ["ipc"]
            if miss_ratio is not None:
                fields.extend(["cache_misses", "cache_references"])
            out.append({"verdict": "instruction_dependency", "confidence": 0.6,
                        "evidence_fields": fields,
                        "note": f"IPC {ipc:.2f} without a high aggregate cache-miss "
                                "lead: test independent accumulators/controlled "
                                "unrolling at L1/L2; no stall class was measured"})
    elif backend == "rk3588_gpu":
        ls = finite_number(counters.get("Load/store unit utilization"), maximum=100)
        fma = finite_number(counters.get("FMA pipe utilization"), maximum=100)
        cvt = finite_number(counters.get("CVT pipe utilization"), maximum=100)
        l2_miss = finite_number(counters.get("L2 cache read miss percentage"), maximum=100)
        if ls is not None and ls >= 80 and fma is not None and fma < 50:
            out.append({"verdict": "memory_bandwidth", "confidence": 0.6,
                        "evidence_fields": ["Load/store unit utilization",
                                            "FMA pipe utilization"],
                        "note": f"LS pipe {ls:.0f}% with FMA {fma:.0f}%: load/store "
                                "instruction pressure is an L1 lead, not measured "
                                "DRAM saturation; test reuse and vectorization"})
        if fma is not None and fma >= 60:
            out.append({"verdict": "compute", "confidence": 0.75,
                        "evidence_fields": ["FMA pipe utilization"],
                        "note": f"FMA pipe {fma:.0f}%: mad forms, unroll, instruction "
                                "mix (L2 vocabulary)"})
        if cvt is not None and cvt >= 30:
            out.append({"verdict": "compiler_pattern", "confidence": 0.7,
                        "evidence_fields": ["CVT pipe utilization"],
                        "note": f"CVT pipe {cvt:.0f}%: inspect generated source and "
                                "instruction mapping before an L2 conversion "
                                "experiment; utilization alone does not prove noise"})
        if l2_miss is not None and l2_miss > 20:
            out.append({"verdict": "memory_bandwidth", "confidence": 0.7,
                        "evidence_fields": ["L2 cache read miss percentage"],
                        "note": f"L2 read miss {l2_miss:.0f}%: locality/tiling "
                                "is a lead, not bandwidth saturation evidence "
                                "(L1 vocabulary; layout questions ESCALATE to L0)"})
        fmin = finite_number(counters.get("gpu_freq_min_hz"))
        fmax = finite_number(counters.get("gpu_freq_max_hz"))
        if fmin and fmax and fmax > 0 and (fmax - fmin) / fmax > 0.15:
            out.append({"verdict": "clock_drift_telemetry", "confidence": 0.9,
                        "evidence_fields": ["gpu_freq_min_hz", "gpu_freq_max_hz"],
                        "note": f"observed GPU clock endpoints differ "
                                f"{fmin/1e6:.0f}-{fmax/1e6:.0f} MHz; frequency inside "
                                "the timing window is unknown; paired ratios still "
                                "need variance/drift checks (infra context)"})
    for hypothesis in out:
        hypothesis["status"] = "optimization_hypothesis"
        hypothesis["causal_bottleneck_confirmed"] = False
        hypothesis["whole_graph_bottleneck_confirmed"] = False
        hypothesis["evidence_scope"] = "aggregate_counter_context_unmapped_to_primfunc"
    return out


def context_hypotheses(
    backend: str,
    *,
    runtime_context: dict,
    static_opportunities: list[dict],
    intermediate_traffic: dict,
) -> list[dict]:
    """Deterministic leads from trusted static/runtime context.

    These facts do not claim speed and never substitute product specifications
    for missing live evidence. They make the first hardware-sense turn useful
    even when optional PMU events are unavailable.
    """

    if backend != "vivo_cpu":
        return []
    out: list[dict] = []
    layout_ids = [
        str(item.get("id"))
        for item in static_opportunities
        if "one_time_layout_or_pack" in (item.get("opportunities") or [])
    ]
    if layout_ids:
        out.append(
            {
                "verdict": "static_weight_layout",
                "confidence": 0.6,
                "evidence_context": layout_ids[:8],
                "note": (
                    "fixed rank-2+ tensors feed the graph; test a declared one-time "
                    "transpose/packing plan at L0; current repeated packing or "
                    "a speed benefit has not been established"
                ),
            }
        )

    call_sites = int(intermediate_traffic.get("call_tir_site_count") or 0)
    launches = int(intermediate_traffic.get("profiled_launch_count") or 0)
    if call_sites >= 3 or launches >= 8:
        out.append(
            {
                "verdict": "graph_redundancy",
                "confidence": 0.6,
                "evidence_context": ["call_tir_site_count", "profiled_launch_count"],
                "note": (
                    f"{call_sites} call_tir sites and {launches} profiled calls: inspect "
                    "explicit L0 fusion/CSE and intermediate materialization"
                ),
            }
        )

    threads = runtime_context.get("observed_num_threads")
    parallel = int(intermediate_traffic.get("tir_parallel_loop_count") or 0)
    if isinstance(threads, int) and not isinstance(threads, bool) and threads > 1 and parallel == 0:
        out.append(
            {
                "verdict": "schedule_granularity",
                "confidence": 0.6,
                "evidence_context": ["observed_num_threads", "tir_parallel_loop_count"],
                "note": (
                    f"the verified {threads}-thread runtime sees no explicit parallel TIR loop; "
                    "test an independent outer partition at L1, accounting for "
                    "external-call parallelism and scheduling overhead"
                ),
            }
        )
    for hypothesis in out:
        hypothesis["status"] = "structural_hypothesis"
        hypothesis["causal_bottleneck_confirmed"] = False
    return out


def breakdown_hypotheses(backend: str, breakdown: list[dict],
                         total_latency_ms: float | None) -> list[dict]:
    """Deterministic first-pass hypotheses from the per-op breakdown."""
    out: list[dict] = []
    if not breakdown:
        return out
    rows = [r for r in breakdown if isinstance(r, dict) and isinstance(r.get("name"), str)
            and finite_number(r.get("duration_us")) is not None
            and finite_number(r.get("count")) is not None
            and finite_number(r.get("percent"), maximum=100) is not None]
    kernel_rows = [r for r in rows if not r["name"].startswith("vm.")]
    vm_rows = [r for r in rows if r["name"].startswith("vm.")]
    vm_pct = sum(r.get("percent") or 0.0 for r in vm_rows)
    if vm_pct >= 20.0:
        out.append({"verdict": "vm_overhead", "confidence": 0.7,
                    "evidence_rows": [r["name"] for r in vm_rows],
                    "note": f"VM builtins hold {vm_pct:.1f}% of profiled time; "
                            "inspect allocation/copy/dispatch separately before "
                            "testing an L0 change; VM time is not all launch overhead"})
    if kernel_rows:
        launches = sum(r.get("count") or 0 for r in kernel_rows)
        mean_us = (sum(r.get("duration_us") or 0.0 for r in kernel_rows) / max(launches, 1))
        if launches >= 8 and mean_us < 30.0:
            out.append({"verdict": "launch_overhead", "confidence": 0.6,
                        "evidence_rows": [r["name"] for r in kernel_rows[:6]],
                        "note": f"{launches} profiled calls at mean {mean_us:.1f}us: "
                                "test L0 fusion with paired wall time; launch overhead "
                                "was not separately measured"})
        top = max(kernel_rows, key=lambda row: row["percent"])
        if (top.get("percent") or 0.0) >= 50.0:
            out.append({"verdict": "hot_kernel", "confidence": 0.9,
                        "evidence_rows": [top["name"]],
                        "note": f"'{top['name']}' holds {top['percent']:.1f}% of time; "
                                "profiler time; matched per-kernel counters and "
                                "controlled changes are needed to choose L1 vs L2"})
    for hypothesis in out:
        hypothesis["status"] = "hotspot_observation" if hypothesis["verdict"] == "hot_kernel" else "optimization_hypothesis"
        hypothesis["causal_bottleneck_confirmed"] = False
    return out
