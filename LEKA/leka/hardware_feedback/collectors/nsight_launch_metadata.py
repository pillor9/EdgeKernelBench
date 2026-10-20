"""Bounded, per-NCU-group metadata; no occupancy or bottleneck inference.

Keep CSV column names, raw values and units. Device attributes are attached
only to the group that actually reported them, never propagated by ordinal.
An Average, a Maximum-only static record and a raw launch value are distinct.
"""
from __future__ import annotations

import math

SCHEMA = "leka.ncu_launch_facts.v1"
# This list is intentionally bounded. Unknown metrics remain in the raw log.
METRIC_UNITS = {
    "launch__grid_size": {""},
    "launch__block_size": {""},
    "launch__registers_per_thread": {"register/thread"},
    "launch__shared_mem_per_block": {"byte/block"},
    "launch__shared_mem_per_block_allocated": {"byte/block"},
    "launch__shared_mem_per_block_static": {"byte/block"},
    "launch__shared_mem_per_block_dynamic": {"byte/block"},
    "launch__shared_mem_per_block_driver": {"byte/block"},
    "launch__occupancy_limit_blocks": {"block"},
    "launch__occupancy_limit_registers": {"block"},
    "launch__occupancy_limit_shared_mem": {"block"},
    "launch__occupancy_limit_warps": {"block"},
    "launch__waves_per_multiprocessor": {""},
    "launch__sm_count": {"SM"},
    "device__attribute_multiprocessor_count": {""},
    "device__attribute_warp_size": {""},
}


def _number(text):
    try:
        value = float(text.replace(",", ""))
    except (ValueError, AttributeError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _observation(metric, record, *, summary):
    columns = ("Minimum", "Maximum", "Average") if summary else ("Metric Value",)
    raw = {key: str(record.get(key) or "").strip() for key in (*columns, "Metric Unit")}
    # Do not let one malicious CSV cell become an unbounded model field.
    if any(len(value) > 128 for value in raw.values()):
        return {"status": "unknown", "reason": "raw_field_exceeded_limit", "value": None}
    static = metric.startswith("device__attribute_")
    fact = {"status": "unknown", "value": None, "unit": raw["Metric Unit"], "raw": raw,
        "scope": "reported_device_attribute_of_this_group" if static else "launch_metadata_of_this_group",
        "unit_status": "unspecified_in_csv" if not raw["Metric Unit"] else "as_reported_in_csv"}
    if raw["Metric Unit"] not in METRIC_UNITS[metric]:
        return {**fact, "reason": "unsupported_unit"}
    present = {key: _number(raw[key]) for key in columns if raw[key]}
    if any(value is None for value in present.values()):
        return {**fact, "reason": "invalid_nonnegative_finite_numeric_value"}
    if summary:
        if "Minimum" in present and "Maximum" in present and present["Minimum"] > present["Maximum"]:
            return {**fact, "reason": "inconsistent_summary_range"}
        if "Average" in present:
            value, aggregation = present["Average"], "summary_average"
            if (("Minimum" in present and value < present["Minimum"])
                    or ("Maximum" in present and value > present["Maximum"])):
                return {**fact, "reason": "inconsistent_summary_range"}
        elif static and set(present) == {"Maximum"}:
            # Preserve the exact column meaning; never call it a mean.
            value, aggregation = present["Maximum"], "summary_maximum_only"
        elif static and set(present) == {"Minimum"}:
            value, aggregation = present["Minimum"], "summary_minimum_only"
        else:
            return {**fact, "reason": "missing_supported_value_column"}
        if static and len(set(present.values())) > 1:
            return {**fact, "reason": "conflicting_static_attribute_summary"}
    elif "Metric Value" in present:
        value, aggregation = present["Metric Value"], "raw_launch_metric_value"
    else:
        return {**fact, "reason": "missing_supported_value_column"}
    if static and not value.is_integer():
        return {**fact, "reason": "nonintegral_static_device_attribute"}
    return {**fact, "status": "reported", "value": value, "aggregation": aggregation}


def record_fact(row, metric, record, *, summary):
    """Conflicting duplicate facts become unknown without replacing evidence."""
    facts = row.setdefault("_launch_facts", {})
    observation = _observation(metric, record, summary=summary)
    if metric not in facts:
        facts[metric] = observation
    elif facts[metric] != observation:
        prior = facts[metric]
        if prior.get("reason") != "conflicting_duplicate_metric":
            facts[metric] = {"status": "unknown", "value": None,
                "reason": "conflicting_duplicate_metric",
                "observations": [prior, observation], "conflicting_rows": 2}
        else:
            prior["conflicting_rows"] += 1


def finish_facts(row):
    facts = row.pop("_launch_facts", {})
    row["launch_facts"] = {"schema_version": SCHEMA,
        "scope": "this_ncu_group_only_no_cross_group_fill",
        "physical_device_identity": "unverified_process_local_device_ordinal",
        "missing_metrics": [name for name in METRIC_UNITS if name not in facts],
        "facts": facts, "causal_bottleneck_confirmed": False}
