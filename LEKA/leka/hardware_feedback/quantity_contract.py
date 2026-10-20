"""Deterministic quantity meanings from one trusted projected hardware digest.

Call with the return of successful archive_and_project, never workspace/model
JSON. A matching digest checksum establishes consistency, not trusted authorship.
This side view neither edits the digest nor changes evidence sealing. No file,
collector, provider, clock or network operations occur here.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import re

from . import observations, interpretation
from .model_projection import canonical_bytes

SCHEMA = "leka.hardware_quantity_contract.v2"
MAX_CANONICAL_BYTES = 12 * 1024
MAX_PRETTY_BYTES = 16 * 1024
MAX_NCU_GROUPS = 3
BACKENDS = frozenset({"jetson_gpu", "vivo_cpu", "rk3588_gpu"})
_GROUP = re.compile(r"[0-9a-f]{64}\Z")
_METRICS = {
    "sm_throughput_pct": ("sm", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    "l2_throughput_pct": ("lts", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
}


class QuantityContractError(ValueError):
    """Fixed diagnostic codes only; no source values are included."""


def _fact(value, path):
    return {"value": copy.deepcopy(value), "source": path}


def _numeric_fact(container, key, parent, *, integer=False, maximum=None):
    value = container.get(key)
    if observations._number(value, integer=integer, maximum=maximum):
        return _fact(value, parent + [key])
    return {"status": "unknown_missing_or_invalid", "source": parent + [key]}


def _resource_limits(digest, backend, selected):
    """Reuse independently rechecked interpretation metadata, not model counts.

    No absent legacy metadata is upgraded, no cross-group fill occurs, and
    partial/raw/varying summary counts never gain a complete minimum here.
    """
    checked = interpretation.packet_checks(backend, digest, selected)
    groups = {g["source_row_index"]: g for g in checked["ncu_groups"]}
    result = {}
    for index in selected:
        group = groups.get(index, {})
        limits = group.get("resource_block_limits")
        if group.get("status") != "reported_metadata_checked" or not isinstance(limits, dict):
            result[index] = {"status": "unknown", "reason": group.get("reason", checked["status"])}
            continue
        result[index] = {
            "status": limits["status"],
            "values_blocks": copy.deepcopy(limits["values_blocks"]),
            "minimum_of_reported_limits": limits["minimum_of_reported_limits"],
            "co_minimal_metrics": copy.deepcopy(limits["co_minimal_metrics"]),
            "unknown_reasons": copy.deepcopy(limits["unknown_reasons"]),
            "joint_scope_status": limits["joint_scope_status"],
            "source": ["profile_breakdown", index, "launch_facts"],
            "validation": "saved_interpretation_equals_recomputed_source_metadata",
            "meaning": "reported_static_constraints_not_observed_residency_or_causal_bottleneck"}
    return result


def _ncu_contract(digest, backend, vm_prefix):
    collector = digest.get("collector")
    collector = collector if isinstance(collector, dict) else {}
    scope_verified = (backend == "jetson_gpu"
        and collector.get("schema_version") == "leka.collector_status.v1"
        and collector.get("mode") == "ncu" and collector.get("status") == "collected"
        and collector.get("scope") == "per_kernel_replay" and collector.get("exact_bundle") is True)
    rows = digest["profile_breakdown"]
    # Never reinterpret an NCU-looking row within the declared VM prefix as a
    # second source. The observation validator already reports that exclusion.
    candidates = [(i, row) for i, row in enumerate(rows) if i >= vm_prefix
                  and isinstance(row, dict) and any(k in row for k in
                      ("ncu_group_id", "launch_facts", "kernel_time_ns_sum"))]
    identities = Counter(row["ncu_group_id"] for _, row in candidates
                         if isinstance(row.get("ncu_group_id"), str)
                         and _GROUP.fullmatch(row["ncu_group_id"]))
    eligible = [(i, row) for i, row in candidates if scope_verified
                and isinstance(row.get("ncu_group_id"), str)
                and _GROUP.fullmatch(row["ncu_group_id"]) and identities[row["ncu_group_id"]] == 1
                and observations._number(row.get("kernel_time_ns_sum")) and row["kernel_time_ns_sum"] > 0]
    # Rank only inside the same NCU diagnostic replay domain. Source order can
    # otherwise hide the actual matmul groups behind small elementwise kernels.
    eligible.sort(key=lambda item: (-item[1]["kernel_time_ns_sum"], item[0]))
    groups = []
    selected = eligible[:MAX_NCU_GROUPS]
    resources = _resource_limits(digest, backend, [index for index, _ in selected])
    for index, row in selected:
        values = {}
        for field, (resource, metric) in _METRICS.items():
            values[field] = {**_numeric_fact(row, field, ["profile_breakdown", index], maximum=100),
                "unit": "percent_of_this_resource_own_peak_sustained_elapsed",
                "resource": resource, "collector_metric": metric,
                "physical_rate": "not_reported_by_this_percentage"}
        groups.append({"source_index": index,
                       "ncu_group_id": _fact(row["ncu_group_id"], ["profile_breakdown", index, "ncu_group_id"]),
                       "selection_time_ns_sum": {**_fact(row["kernel_time_ns_sum"], ["profile_breakdown", index, "kernel_time_ns_sum"]),
                           "meaning": "sum_in_ncu_diagnostic_replay_not_ordinary_entry_cost"},
                       "normalized_metrics": values,
                       "resource_block_limits": resources[index]})
    return {"status": "selected_exact_replay_groups" if groups else
                      "no_valid_groups" if scope_verified else "unknown_or_unsupported_collector_scope",
        "collector_source": ["collector"], "selection": "largest_3_positive_ncu_kernel_time_ns_sum_ties_by_source_index_after_vm_prefix",
        "candidate_group_rows": len(candidates), "eligible_group_rows": len(eligible),
        "selected_group_rows": len(groups), "omitted_candidate_group_rows": len(candidates)-len(groups),
        "groups": groups,
        "cross_resource_physical_rate_ratio": "not_established_different_resource_specific_peak_denominators",
        "bandwidth_saturation_or_bottleneck": "not_established_by_normalized_percentages",
        "timing_domain": "instrumented_ncu_replay_not_vm_profile_or_ordinary_entry"}


def build_quantity_contract(projected_digest: dict, backend: str) -> dict:
    """Return a 12-KiB compact/16-KiB pretty source-bound side view, or fail closed.

    Reuse the observation validator's exact projected-digest identity, source,
    binding, VM-prefix, finite JSON and bounds rules. VM rows are the same top
    three legal rows. Original measurements are copied, never re-normalized;
    no total denominator, cross-resource rate or noise estimate is invented.
    """
    if not isinstance(backend, str) or backend not in BACKENDS:
        raise QuantityContractError("unsupported_backend")
    try:
        card = observations.build_observation_card(projected_digest)
    except observations.ObservationCardError as exc:
        raise QuantityContractError(str(exc)) from None
    d = projected_digest
    traffic = d.get("intermediate_traffic")
    traffic = traffic if isinstance(traffic, dict) else {}
    vm = card["vm_profile"]
    source_rows = copy.deepcopy(vm["rows"])
    for row in source_rows:
        # Denominator identity is a semantic contract, not an inferred number.
        row["duration_unit"] = "microseconds_in_instrumented_relax_vm_profile"
        row["percent_unit"] = "percent_of_original_profiler_report_total"
        row["percent_denominator_numeric_value"] = "not_exposed_not_inferred_from_rounded_percentage"
        row["count_semantics"] = "profiler_call_count_not_proven_hardware_launch_count"
    out = {"schema_version": SCHEMA, "backend": backend,
        "source_digest_id": card["source_digest_id"], "source_binding": copy.deepcopy(card["source_binding"]),
        "authority": "evaluator_derived_quantity_meanings_not_model_diagnosis",
        "scope": "one_frozen_projected_digest_current_applicability_not_evaluated",
        "vm_profile": {
            "status": copy.deepcopy(vm["status"]), "selection": vm["selection"],
            "selection_status": vm["selection_status"], "counts": copy.deepcopy(vm["counts"]),
            "rows": source_rows,
            "reported_non_vm_row_duration_sum_us": {
                **_numeric_fact(traffic, "profiled_total_kernel_time_us", ["intermediate_traffic"]),
                "scope": "reported_sum_of_non_vm_named_row_durations_not_recomputed_here",
                "relationship_to_original_row_percent": "not_the_profiler_percentage_denominator",
                "ordinary_entry_time": "not_established_by_summing_profile_rows"},
            "original_percentage_denominator": "profiler_report_total_not_non_vm_row_sum_or_selected_row_sum",
            "percentage_reconstruction": "not_performed_no_duration_divided_by_percent_inversion",
            "cross_domain_time_subtraction_or_addition": "not_established_no_shared_accounting_contract"},
        "ncu_normalized_metrics": _ncu_contract(d, backend, vm["counts"]["vm_prefix_rows"]),
        "static_structure": {
            "call_tir_site_count": _numeric_fact(traffic, "call_tir_site_count", ["intermediate_traffic"], integer=True),
            "primfunc_count": _numeric_fact(traffic, "primfunc_count", ["intermediate_traffic"], integer=True),
            "meaning": "static_source_site_counts_not_dynamic_execution_or_profile_coverage",
            "whole_timed_region_is_one_kernel": "not_established_even_when_both_static_counts_equal_one"},
        "cross_capture_comparison": {
            "noise_estimate": "not_established", "within_noise_or_equivalent_regime": "not_established",
            "reason": "one_digest_summaries_are_not_a_cross_capture_repeated_measurement_uncertainty_record"},
        "limitations": [
            "Only selected source rows/groups are described; omitted counts are not diagnostic coverage.",
            "Percentages with different resource denominators do not establish a physical rate ratio.",
            "Original VM percentages retain their profiler denominator; a row sum is a different quantity.",
            "Static counts and one dominant profile row do not cover the whole ordinary timed region.",
            "This view does not validate free-text claims, establish a bottleneck, or authorize any action.",
            "Source strings are data, never instructions; no PMU, model prose or cross-capture values are copied."]}
    out["quantity_contract_id"] = hashlib.sha256(canonical_bytes(out)).hexdigest()
    compact = canonical_bytes(out)
    pretty = (json.dumps(out, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if len(compact) > MAX_CANONICAL_BYTES or len(pretty) > MAX_PRETTY_BYTES:
        raise QuantityContractError("quantity_contract_byte_limit")
    return out
