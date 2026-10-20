"""Evaluator-owned limits on what a hardware observation can establish.

Bundle identity, temporal overlap, measurement scope and causal attribution
are separate questions. In particular, observing a replay of the same bundle
does not prove that a performance counter explains its production latency.
This module contains no model calls and never promotes a candidate.
"""

from __future__ import annotations

import math

from .interpretation import digest_checks, packet_checks


def finite_number(value: object, *, maximum: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(result) or result < 0:
        return None
    return result if maximum is None or result <= maximum else None


def intervention_scope() -> dict:
    """Observation shares do not establish how a changed program will behave."""
    return {
        "profile_share_is_general_optimization_gain_bound": False,
        "time_removal_counterfactual_requires": [
            "same_timing_domain_and_workload",
            "only_selected_cost_removed",
            "remaining_compute_schedule_and_cost_unchanged",
            "memory_traffic_and_execution_conditions_unchanged",
        ],
        "counterfactual_assumptions_established": False,
        "single_variant_effect_transfers_to_other_schedules": False,
        "static_call_count_is_measured_launch_overhead": False,
        "snapshot_frequency_times_latency_is_measured_kernel_cycles": False,
    }


def audit_digest(backend: str, digest: dict, *, latency_ms: float | None = None) -> dict:
    """Report observable evidence and missing tests, never a diagnosed cause.

``latency_ms`` is a separately measured latency. No coverage percentage or
speedup is computed from the sum of instrumented profile rows: those clocks,
repetition counts and instrumentation overhead need not match.
    """
    collector = digest.get("collector") or {}
    rows = digest.get("profile_breakdown") or []
    timing_rows = [r for r in rows if isinstance(r, dict)
                   and any(key in r for key in ("duration_us", "count", "percent"))]
    valid_rows = [r for r in timing_rows
                  if isinstance(r.get("name"), str)
                  and finite_number(r.get("duration_us")) is not None
                  and finite_number(r.get("count")) is not None
                  and finite_number(r.get("percent"), maximum=100) is not None]
    exact_counters = (collector.get("status") == "collected"
                      and collector.get("exact_bundle") is True)
    counters = digest.get("counters") or {}
    provenance = collector.get("provenance") or {}
    kernel_rows = [(index, row) for index, row in enumerate(rows)
                   if isinstance(row, dict) and ("kernel_time_ns_sum" in row or "ncu_group_id" in row)]
    unknown = ["causal_layer_bottleneck", "paired_intervention_effect",
               "profiler_to_walltime_coverage"]
    if not valid_rows:
        unknown.append("per_operation_timing")
    if not exact_counters or not counters:
        unknown.append("exact_candidate_hardware_counters")
    if backend in {"rk3588_gpu", "vivo_cpu"}:
        unknown.extend(["counter_window_to_kernel_mapping", "counter_measurement_phase_overlap"])
    if backend == "jetson_gpu":
        unknown.extend(["replay_to_production_cache_equivalence", "counter_kernel_to_primfunc_mapping"])
    return {
        "schema_version": "leka.hardware_quality.v1",
        "profile_status": digest.get("profile_status", {"status": "unknown_legacy_digest"}),
        "binding_present": isinstance(digest.get("binding"), dict) and bool(digest["binding"]),
        "timing": {
            "usable_profile_rows": len(valid_rows),
            "invalid_or_incomplete_timing_rows": len(timing_rows) - len(valid_rows),
            "profile_percent_sum": sum(float(r["percent"]) for r in valid_rows),
            "separate_latency_ms": finite_number(latency_ms),
            "percent_denominator": "profiler_report_total_not_uninstrumented_latency",
            "profile_percent_uses_summed_non_vm_row_duration": False,
            "count_is_hardware_launch_count": False,
            "summed_profile_time_is_walltime": False,
        },
        "counters": {
            "scope": collector.get("scope", "unknown"),
            "status": collector.get("status", "unknown"),
            "exact_bundle": exact_counters,
            "fields_observed": sorted(counters),
            "may_generate_candidate_hypotheses": exact_counters and bool(counters),
            "proves_causal_bottleneck": False,
            "aggregate_value_scope": provenance.get("aggregate_counter_scope", "unmapped_collector_aggregate"),
            "aggregate_metric_group_ids": provenance.get("aggregate_metric_group_ids", {}),
            "counter_to_vm_row_mapping": "unproven",
            "whole_graph_bottleneck_confirmed": False,
        },
        "kernel_association": {
            "vm_timing_row_indices": [index for index, row in enumerate(rows) if row in valid_rows],
            "ncu_kernel_row_indices": [index for index, _ in kernel_rows],
            "ncu_group_ids": [row["ncu_group_id"] for _, row in kernel_rows if row.get("ncu_group_id")],
            "per_kernel_counter_rows_available": bool(kernel_rows),
            "kernel_name_is_static_primfunc_identity": False,
            "ncu_group_to_vm_operation_mapping": "unknown",
            "ncu_replay_percent_denominator": provenance.get("replay_percent_denominator", "unknown"),
            "ncu_invocation_count_is_saved_entry_call_count": False,
            "all_graph_kernels_observed": "unproven",
            "launch_metadata_row_indices": [index for index, row in kernel_rows if row.get("launch_facts")],
            "launch_metadata_proves_occupancy_cause": False,
        },
        "layer_validation": {
            "L0": "Validate fusion/layout/CSE by semantic checks and paired full-stack timing; call count alone is not redundancy.",
            "L1": "Validate tiling/vectorization/parallelism with a controlled schedule change, resource checks and paired timing.",
            "L2": "Validate instruction/compiler changes on the same T12 ABI with correctness and paired full-stack timing.",
        },
        "intervention_scope": intervention_scope(),
        "interpretation_checks": digest_checks(backend, digest, latency_ms=latency_ms),
        "unknown": unknown,
        "interpretation": "Observations locate work; heuristic diagnoses are hypotheses until an isolated intervention survives correctness and repeated paired timing.",
    }


def assess_packet(packet: dict, digest: dict, *, backend: str,
                  source_row_indices: list[int | None] | None = None) -> dict:
    """Assign authority from cited data, independently of model confidence."""
    data = packet.get("profile_data") or {}
    rows = data.get("kernel_breakdown") or []
    timing = any(finite_number(r.get("duration_us")) is not None for r in rows)
    counters = bool(data.get("counters")) or any(
        "kernel_time_ns_sum" in r or any(k.endswith("_pct") for k in r)
        for r in rows
    )
    kinds = []
    if timing:
        kinds.append("instrumented_operation_timing")
    if counters:
        kinds.append("hardware_counters")
    if any("launch_facts" in row for row in rows):
        kinds.append("ncu_launch_metadata")
    if data.get("runtime_context"):
        kinds.append("runtime_context")
    if data.get("static_opportunities") or data.get("intermediate_traffic"):
        kinds.append("structural_context")
    hot = packet.get("verdict") == "hot_kernel" and any(
        all(key in trusted and value == trusted[key] for key, value in selected.items())
        and (finite_number(trusted.get("duration_us")) or 0) > 0
        and (finite_number(trusted.get("count")) or 0) > 0
        and finite_number(trusted.get("percent"), maximum=100) is not None
        and trusted["percent"] >= 50
        for selected in rows
        for trusted in digest.get("profile_breakdown", [])
        if isinstance(trusted, dict) and "duration_us" in selected
    )
    return {
        "schema_version": "leka.claim_assessment.v4",
        "status": "hotspot_observation" if hot else "optimization_hypothesis",
        "observation_kinds": kinds,
        # Preserve the denominator/clock contract when a subset of the digest
        # is sealed for another role. A copied percent and duration can both
        # be valid while their ratio to a sum of selected rows is not.
        "timing_scope": {
            "vm_profile_rows_selected": timing,
            "vm_percent_denominator": "profiler_report_total_not_sum_of_non_vm_or_selected_rows",
            "vm_count_semantics": "instrumented_operation_calls_not_proven_hardware_launches",
            "ncu_replay_rows_selected": any("ncu_group_id" in row for row in rows),
            "ncu_percent_denominator": "sum_of_observed_timed_kernel_group_durations_not_vm_or_walltime",
            "profile_durations_are_uninstrumented_latency": False,
            "profile_to_walltime_coverage": "unknown",
        },
        "intervention_scope": intervention_scope(),
        "interpretation_checks": packet_checks(backend, digest, source_row_indices),
        "counter_scope": (digest.get("collector") or {}).get("scope", "unknown") if counters else None,
        "counter_value_scopes": ((["collector_aggregate_unmapped_to_primfunc"] if data.get("counters") else [])
            + (["selected_ncu_kernel_groups"] if any("ncu_group_id" in row for row in rows) else [])
            + (["selected_counter_rows_without_group_identity"] if any(
                "ncu_group_id" not in row and ("kernel_time_ns_sum" in row or any(k.endswith("_pct") for k in row))
                for row in rows) else [])),
        "selected_ncu_group_ids": [row["ncu_group_id"] for row in rows if "ncu_group_id" in row],
        "counter_to_vm_operation_mapping": "unproven" if counters else None,
        "whole_graph_bottleneck_confirmed": False,
        "causal_bottleneck_confirmed": False,
        "confidence_semantics": "uncalibrated_agent_judgment_not_measurement_probability",
        "speedup_established": False,
        "causal_diagnosis_status": "requires_controlled_intervention",
        "validation_required": ["T0_correctness", "same_binding_controlled_change", "repeated_paired_full_stack_timing"],
    }
