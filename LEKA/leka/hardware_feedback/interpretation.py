"""Bounded evaluator observations about interpretation, never bottleneck causes.

References are field-token arrays, not slash-prefixed strings that the privacy
projection correctly treats as filesystem paths. No device/model calls live here.
"""
from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import json
import math
import re

SCHEMA = "leka.interpretation_checks.v1"
MAX_GROUPS = 16
MAX_SOURCE_ROWS = 512
MAX_CHECK_BYTES = 32 * 1024
MAX_EXACT_INTEGER = 2**53 - 1
RESOURCE_METRICS = tuple("launch__occupancy_limit_" + x for x in
                         ("blocks", "registers", "shared_mem", "warps"))
GRID = "launch__grid_size"
LAUNCH_SM = "launch__sm_count"
DEVICE_SM = "device__attribute_multiprocessor_count"
_GROUP_ID = re.compile(r"[0-9a-f]{64}\Z")


def _object(value):
    return value if isinstance(value, dict) else {}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # Check the bound before float conversion (including arbitrarily large ints).
    if not 0 < value <= MAX_EXACT_INTEGER or not math.isfinite(value) or int(value) != value:
        return None
    return int(value)


def _finite(value, *, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0 and (maximum is None or value <= maximum)
    except (OverflowError, ValueError):
        return False


def _count(facts, metric):
    """Only exact bounded count columns support count arithmetic.

    A float that rounded a fractional raw column to an integer is insufficient.
    Stable summary means equal *reported* Min/Max/Avg, not execution tracing.
    """
    if metric not in facts:
        return None, "missing_metric", None
    fact = _object(facts[metric])
    if fact.get("status") != "reported":
        return None, "unreported_or_conflicting_metric", None
    static = metric == DEVICE_SM
    unit = "block" if metric in RESOURCE_METRICS else "SM" if metric == LAUNCH_SM else ""
    scope = "reported_device_attribute_of_this_group" if static else "launch_metadata_of_this_group"
    if fact.get("unit") != unit or fact.get("scope") != scope:
        return None, "unsupported_unit_or_scope", None
    value = _integer(fact.get("value"))
    if value is None:
        return None, "invalid_exact_positive_count", None
    raw = _object(fact.get("raw"))
    if raw.get("Metric Unit") != unit:
        return None, "raw_unit_mismatch", None
    aggregation = fact.get("aggregation")
    if not isinstance(aggregation, str):
        return None, "unsupported_aggregation", None
    if aggregation == "raw_launch_metric_value":
        columns = ("Metric Value",)
    elif aggregation in {"summary_average", "summary_maximum_only", "summary_minimum_only"}:
        columns = ("Minimum", "Maximum", "Average")
        if static:
            declared = {"summary_average": "Average", "summary_maximum_only": "Maximum",
                        "summary_minimum_only": "Minimum"}[aggregation]
            if not raw.get(declared):
                return None, "missing_declared_summary_column", None
            columns = tuple(key for key in columns if raw.get(key))
            if aggregation.endswith("_only") and columns != (declared,):
                return None, "inconsistent_summary_aggregation", None
        elif aggregation != "summary_average":
            return None, "unsupported_launch_aggregation", None
    else:
        return None, "unsupported_aggregation", None
    for column in columns:
        text = raw.get(column)
        if not isinstance(text, str) or not text or len(text) > 128:
            return None, "missing_or_oversized_raw_count", None
        try:
            number = Decimal(text.replace(",", ""))
            if not number.is_finite() or number != Decimal(value):
                return None, "raw_counts_not_equal_exact_reported_value", None
        except (InvalidOperation, ValueError):
            return None, "invalid_raw_count", None
    return value, None, aggregation


def _ncu_row(row):
    return isinstance(row, dict) and any(key in row for key in ("ncu_group_id", "launch_facts", "kernel_time_ns_sum"))


def _source_reason(backend, digest):
    if backend != "jetson_gpu":
        return "not_applicable_to_this_backend"
    if digest.get("schema_version") != "leka.profile_digest.v2":
        return "unknown_digest_schema"
    profile = _object(digest.get("profile_status"))
    if profile.get("schema_version") != "leka.profile_status.v1":
        return "unknown_profile_status_schema"
    if profile.get("measurement_ok") is not True:
        return "measurement_not_verified_passed"
    collector = _object(digest.get("collector"))
    if (collector.get("schema_version") != "leka.collector_status.v1"
            or collector.get("status") != "collected" or collector.get("exact_bundle") is not True
            or collector.get("mode") != "ncu" or collector.get("scope") != "per_kernel_replay"):
        return "unsupported_or_unverified_collector_scope"
    return None


def _group(index, row, *, source_reason, duplicated_ids):
    identity = row.get("ncu_group_id")
    valid_identity = isinstance(identity, str) and _GROUP_ID.fullmatch(identity) is not None
    out = {"source_row_index": index, "ncu_group_id": identity if valid_identity else None}
    launch = _object(row.get("launch_facts"))
    reason = source_reason
    if not reason and (not valid_identity or identity in duplicated_ids):
        reason = "missing_or_ambiguous_group_identity"
    if not reason and (launch.get("schema_version") != "leka.ncu_launch_facts.v1"
                       or launch.get("scope") != "this_ncu_group_only_no_cross_group_fill"
                       or not isinstance(launch.get("facts"), dict)):
        reason = "unsupported_or_missing_launch_metadata"
    if reason:
        return {**out, "status": "unknown", "reason": reason}
    facts = launch["facts"]
    prefix = ["profile_breakdown", index, "launch_facts", "facts"]
    counts = {metric: _count(facts, metric) for metric in (*RESOURCE_METRICS, GRID, LAUNCH_SM, DEVICE_SM)}
    values = {metric: counts[metric][0] for metric in RESOURCE_METRICS}
    aggregation_consistent = len({counts[metric][2] for metric in RESOURCE_METRICS if counts[metric][2]}) <= 1
    complete = all(value is not None for value in values.values()) and aggregation_consistent
    minimum = min(values.values()) if complete else None
    resource = {
        "status": "complete_stable_reported_limits" if complete else "unknown_or_incomplete",
        "values_blocks": values,
        "minimum_of_reported_limits": minimum,
        "co_minimal_metrics": [metric for metric in RESOURCE_METRICS if complete and values[metric] == minimum],
        "unknown_reasons": {metric: counts[metric][1] for metric in RESOURCE_METRICS if counts[metric][1]},
        "joint_scope_status": "same_reported_aggregation" if aggregation_consistent else "mixed_resource_aggregation_unknown",
        "aggregation": {metric: counts[metric][2] for metric in RESOURCE_METRICS},
        "source_refs": {metric: prefix + [metric] for metric in RESOURCE_METRICS if metric in facts},
        "meaning": "co_minima_of_reported_static_constraints_not_observed_residency_or_cause",
    }
    grid, grid_reason, _ = counts[GRID]
    sm_values = {metric: counts[metric][0] for metric in (LAUNCH_SM, DEVICE_SM) if metric in facts}
    sm_reason = ("missing_sm_count" if not sm_values else
                 "unknown_present_sm_count" if None in sm_values.values() else
                 "conflicting_same_group_sm_counts" if len(set(sm_values.values())) != 1 else None)
    sm = next(iter(sm_values.values())) if not sm_reason else None
    distribution = {
        "status": "metadata_totals_available" if grid is not None and sm is not None else "unknown",
        "grid_blocks": grid, "reported_sm_count": sm,
        "grid_blocks_per_reported_sm": grid / sm if grid is not None and sm is not None else None,
        "unknown_reasons": [r for r in (grid_reason, sm_reason) if r],
        "aggregation": {metric: counts[metric][2] for metric in (GRID, LAUNCH_SM, DEVICE_SM) if metric in facts},
        "source_refs": {metric: prefix + [metric] for metric in (GRID, LAUNCH_SM, DEVICE_SM) if metric in facts},
        "ratio_semantics": "arithmetic_quotient_not_execution_assignment",
        "per_sm_assignment": "not_measured_by_this_collector",
        "physical_device_identity": "unverified_process_local_device_ordinal",
    }
    return {**out, "status": "reported_metadata_checked", "resource_block_limits": resource,
            "launch_distribution": distribution}


def _timing_domains(backend, digest, indices, *, latency_ms=None, packet=False):
    rows = digest.get("profile_breakdown", [])
    status = _object(digest.get("profile_status"))
    verified = (digest.get("schema_version") == "leka.profile_digest.v2"
                and status.get("schema_version") == "leka.profile_status.v1"
                and status.get("measurement_ok") is True
                and backend in ("jetson_gpu", "vivo_cpu", "rk3588_gpu"))
    vm = [i for i in indices if isinstance(rows[i], dict) and not _ncu_row(rows[i])
          and all(_finite(rows[i].get(key), maximum=100 if key == "percent" else None)
                  for key in ("duration_us", "count", "percent"))]
    ncu = [i for i in indices if _ncu_row(rows[i])]
    ncu_verified = _source_reason(backend, digest) is None
    phase = _object(_object(digest.get("collector")).get("provenance")).get("measurement_phase")
    return {
        "vm_profile": {"status": "observed" if vm and verified and status.get("source") == "relax_vm_profile" else "unknown_or_not_selected",
                       "row_indices": vm, "source_ref": ["profile_status", "source"],
                       "clock": "instrumented_relax_vm_profile"},
        "ncu_replay": {"status": "observed" if ncu and ncu_verified else "unknown_or_not_selected",
                       "row_indices": ncu,
                       "phase": phase if ncu_verified and ncu and phase == "whole_worker_including_warmup_and_correctness" else "unknown",
                       "source_ref": ["collector", "provenance", "measurement_phase"]},
        "ordinary_entry": {"status": "not_selected" if packet else "observed" if verified and _finite(latency_ms) else "unknown",
                           "source_ref": ["evidence_quality", "timing", "separate_latency_ms"]},
        "cross_domain_cost_decomposition": "not_established_no_shared_accounting_contract",
        "named_overhead_from_subtraction": "not_established",
    }


def digest_checks(backend: str, digest: dict, *, latency_ms=None) -> dict:
    """Derive observations before projection/hash; never alter source rows."""
    out = {"schema_version": SCHEMA, "scope": "this_digest",
           "source_reference_space": "this_digest_field_tokens",
           "ncu_groups": [], "omitted_group_count": 0}
    rows = digest.get("profile_breakdown", [])
    if not isinstance(rows, list) or len(rows) > MAX_SOURCE_ROWS:
        return {**out, "status": "unknown_source_row_limit_or_type"}
    candidates = [(i, row) for i, row in enumerate(rows) if _ncu_row(row)]
    identities = [row.get("ncu_group_id") for _, row in candidates if isinstance(row.get("ncu_group_id"), str)]
    duplicated = {value for value in identities if identities.count(value) > 1}
    out.update(status="evaluated", timing_domains=_timing_domains(backend, digest, list(range(len(rows))), latency_ms=latency_ms))
    reason = _source_reason(backend, digest)
    for i, row in candidates:
        if len(out["ncu_groups"]) >= MAX_GROUPS:
            break
        group = _group(i, row, source_reason=reason, duplicated_ids=duplicated)
        out["ncu_groups"].append(group)
        out["omitted_group_count"] = len(candidates) - len(out["ncu_groups"])
        if len(_json(out)) > MAX_CHECK_BYTES:
            out["ncu_groups"].pop()
            break
    out["omitted_group_count"] = len(candidates) - len(out["ncu_groups"])
    if out["omitted_group_count"]:
        out["omission_reason"] = "bounded_group_or_byte_budget"
    # The omission marker and changed count must fit too; never truncate bytes.
    while len(_json(out)) > MAX_CHECK_BYTES and out["ncu_groups"]:
        out["ncu_groups"].pop()
        out["omitted_group_count"] = len(candidates) - len(out["ncu_groups"])
        out["omission_reason"] = "bounded_group_or_byte_budget"
    return out


def packet_checks(backend: str, digest: dict, source_row_indices: list[int | None] | None) -> dict:
    """Only project uniquely resolved selected rows from a new trusted digest.

    Legacy digests remain usable, but cannot gain absent evaluator metadata.
    Recomputing/checking the stored metadata also fails closed if projection
    aliases a field token because an unusual configured private identity collides.
    """
    empty = {"schema_version": SCHEMA, "scope": "selected_source_rows",
             "source_reference_space": "source_digest_id_field_tokens", "ncu_groups": []}
    quality = _object(digest.get("evidence_quality"))
    saved = _object(quality.get("interpretation_checks"))
    if saved.get("schema_version") != SCHEMA:
        return {**empty, "status": "unknown_legacy_or_unsupported_interpretation_schema"}
    expected = digest_checks(backend, digest, latency_ms=_object(quality.get("timing")).get("separate_latency_ms"))
    try:
        matches = _json(saved) == _json(expected)
    except (ValueError, TypeError, RecursionError):
        matches = False
    if not matches or saved.get("status") != "evaluated":
        return {**empty, "status": "unknown_invalid_or_unavailable_interpretation_metadata"}
    rows = digest.get("profile_breakdown", [])
    if (not isinstance(source_row_indices, list) or len(source_row_indices) > MAX_SOURCE_ROWS
            or any(index is not None and (type(index) is not int or index < 0 or index >= len(rows)) for index in source_row_indices)):
        return {**empty, "status": "unknown_selected_row_identity"}
    selected = list(dict.fromkeys(index for index in source_row_indices if index is not None))
    groups = {group["source_row_index"]: group for group in saved["ncu_groups"]}
    missing = [i for i in selected if _ncu_row(rows[i]) and i not in groups]
    unresolved = [i for i, index in enumerate(source_row_indices) if index is None]
    return {**empty, "status": "partial_unknown_selection" if missing or unresolved else "selected_metadata_available",
            "source_row_indices": selected,
            "unresolved_selected_row_positions": unresolved,
            "metadata_unavailable_source_row_indices": missing,
            "ncu_groups": [copy.deepcopy(groups[i]) for i in selected if i in groups],
            "timing_domains": _timing_domains(backend, digest, selected, packet=True)}
