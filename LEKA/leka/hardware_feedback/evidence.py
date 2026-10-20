"""Evidence packets: attribution and attached profiler data, subset-verified.

The hw_sense agent SELECTS rows/fields from the trusted digest; it may not
alter a single value.  verify_subset() checks every numeric leaf in the
packet's profile_data against the digest before the driver delivers it.
Free-text fields travel wrapped as untrusted data.
"""

from __future__ import annotations

import json
import copy
import hashlib
import math

from .attribution import VERDICT_TO_LAYER
from .quality import assess_packet

MAX_PACKET_BYTES = 16 * 1024
MAX_FREE_TEXT_CHARS = 600
FREE_TEXT_FIELDS = ("rationale", "suggested_direction")
DATA_FIELDS = {"kernel_breakdown", "counters", "runtime_context", "intermediate_traffic", "static_opportunities"}
AUTHORITY_FIELDS = {"source_binding", "source_digest_id", "evidence_id", "claim_assessment", "source_selection"}


def output_contract(backend: str, trusted_digest: dict | None = None) -> dict:
    """One shared source of truth for the prompt, schema and rejection reply."""
    mapping = VERDICT_TO_LAYER[backend]
    contract = {
        "schema_version": "leka.hw_evidence_contract.v6",
        "backend": backend,
        "verdict_to_layers": {
            verdict: ([layers] if isinstance(layers, str) else list(layers))
            for verdict, layers in mapping.items() if layers != "infra"
        },
        "required_fields": ["target_layer", "verdict", "confidence", "profile_data"],
        "confidence_range": [0.6, 1.0],
        "free_text_limits": {
            "fields": list(FREE_TEXT_FIELDS),
            "maximum_characters_per_field": MAX_FREE_TEXT_CHARS,
            "character_count": "Unicode code points before evaluator wrapping",
            "over_limit_behavior": "reject_packet_without_truncation; existing bounded repair only",
        },
        "preferred_row_selection": {
            "field": "profile_data.kernel_breakdown_indices",
            "source": "PROFILE_DIGEST.json.profile_breakdown",
            "index_base": 0, "maximum_items": 16, "unique_integers_only": True,
            "example": {"target_layer": "L1", "verdict": "hot_kernel", "confidence": 0.7,
                        "profile_data": {"kernel_breakdown_indices": [0]}},
        },
        "preferred_counter_selection": {
            "field": "profile_data.counter_names",
            "source": "PROFILE_DIGEST.json.counters",
            "maximum_items": 16,
            "unique_exact_names_only": True,
            "requires_collected_exact_bundle": True,
            "value_authority": "evaluator expands exact frozen values; do not transcribe decimals",
        },
        "rules": [
            "verdict is an exact key, not a status such as attributed/confirmed/refuted",
            "target_layer must be one of that verdict's allowed layers",
            "cite at least one nonempty verbatim measured/context/structural data item",
            "prefer zero-based kernel_breakdown_indices to copying decimal values; evaluator expands exact rows from this frozen digest",
            "when counter packets are enabled, prefer counter_names selected from allowed_counter_fields; evaluator supplies exact values; existing inline counters remain strictly checked",
            "counter_names refers to the digest-level counters and preserves their recorded aggregate scope; for a specific kernel/group select its exact breakdown row instead; never relabel an aggregate as that kernel's counter",
            "missing counters are unknown; code-derived bottlenecks are hypotheses",
            "confidence is uncalibrated judgment; no packet establishes a causal bottleneck or speedup",
            "only documented profile_data sections are allowed; an item must include values, not only a name",
            "packets must fit the 16 KiB limit; select fewer source rows instead of truncating evidence",
            "rationale and suggested_direction must each fit 600 characters; state the observation, uncertainty and next test concisely together; overlong text is rejected, never silently shortened",
            "no justified recommendation is valid: write an empty array, not invented evidence",
        ],
    }
    if trusted_digest is not None:
        collector = trusted_digest.get("collector") or {}
        counters = trusted_digest.get("counters") or {}
        exact = collector.get("status") == "collected" and collector.get("exact_bundle") is True
        rows = trusted_digest.get("profile_breakdown") or []
        allowed_rows, context_rows = [], []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            deep_row = any(key in row for key in ("kernel_time_ns_sum", "ncu_group_id", "launch_facts")) or any(key.endswith("_pct") for key in row)
            (context_rows if deep_row and not exact else allowed_rows).append(index)
        contract["current_digest_capabilities"] = {
            "source_digest_id": trusted_digest.get("digest_id"),
            "source_binding": copy.deepcopy(trusted_digest.get("binding")),
            "collector_status": collector.get("status", "unknown"),
            "collector_scope": collector.get("scope", "unknown"),
            "collector_exact_bundle": collector.get("exact_bundle") is True,
            "counter_packets_enabled": exact,
            "allowed_counter_fields": sorted(counters) if exact else [],
            "context_only_counter_fields": [] if exact else sorted(counters),
            "allowed_kernel_breakdown_indices": allowed_rows,
            "context_only_kernel_breakdown_indices": context_rows,
            "profile_status": copy.deepcopy(trusted_digest.get("profile_status")),
        }
        contract["rules"].append(
            "For THIS digest prefer counter_names from allowed_counter_fields; inline counters, if supplied, must match verbatim and are never corrected by a selector; causal diagnoses remain hypotheses."
            if exact else
            "For THIS digest counter packets are DISABLED: omit counter_names and write profile_data.counters={} or omit it. Process/global counters are diagnostic context only; do not cite them through rows or relabel them as runtime_context. Use allowed timing-row indices or verified structural/runtime fields for packets; discuss counter-only leads in NOTES.md or peer suggestions with their unproven scope."
        )
    return contract


def _expand_row_selection(packet: dict, trusted_digest: dict) -> tuple[dict, list[int] | None]:
    """Resolve references only against the evaluator-supplied frozen digest.

    Legacy inline rows remain checked verbatim, including when mixed with
    indices. Expansion never overwrites or hides an invalid model-written row.
    """
    data = packet.get("profile_data")
    if not isinstance(data, dict) or "kernel_breakdown_indices" not in data:
        return packet, None
    indices = data["kernel_breakdown_indices"]
    if (not isinstance(indices, list) or len(indices) > 16
            or any(type(index) is not int for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("kernel_breakdown_indices must be at most 16 unique integer indices (not booleans)")
    source = trusted_digest.get("profile_breakdown") or []
    if not isinstance(source, list) or any(index < 0 or index >= len(source) for index in indices):
        raise ValueError("kernel_breakdown_indices contains an out-of-range index for this frozen digest")
    rows = data.get("kernel_breakdown", [])
    if not isinstance(rows, list):
        raise ValueError("kernel_breakdown must be an array of objects")
    if any(not isinstance(source[index], dict) for index in indices):
        raise ValueError("kernel_breakdown_indices selected a malformed trusted row")
    expanded = copy.deepcopy(packet)
    expanded["profile_data"].pop("kernel_breakdown_indices")
    expanded["profile_data"]["kernel_breakdown"] = copy.deepcopy(rows + [source[index] for index in indices])
    return expanded, list(indices)


def _expand_counter_selection(packet: dict, trusted_digest: dict) -> tuple[dict, list[str] | None]:
    """Expand named counters without repairing or overwriting inline values."""
    data = packet.get("profile_data")
    if not isinstance(data, dict) or "counter_names" not in data:
        return packet, None
    names = data["counter_names"]
    if (not isinstance(names, list) or len(names) > 16
            or any(type(name) is not str or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("counter_names must be at most 16 unique nonempty exact field names")
    source = trusted_digest.get("counters") or {}
    if not isinstance(source, dict) or any(name not in source for name in names):
        raise ValueError("counter_names contains a field absent from this frozen digest")
    inline = data.get("counters", {})
    if not isinstance(inline, dict):
        raise ValueError("counters must be an object")
    expanded = copy.deepcopy(packet)
    expanded["profile_data"].pop("counter_names")
    counters = copy.deepcopy(inline)
    for name in names:
        # A bad inline value must survive expansion and fail verify_subset.
        # A selector is a reference, never permission to silently fix evidence.
        if name not in counters:
            counters[name] = copy.deepcopy(source[name])
    expanded["profile_data"]["counters"] = counters
    return expanded, list(names)


def _rows_index(digest_rows: list[dict]) -> dict[str, list[dict]]:
    """Digest rows may legitimately repeat a name (e.g. one row per VM builtin
    call site) — index every row under its name."""
    index: dict[str, list[dict]] = {}
    for r in digest_rows or []:
        index.setdefault(str(r.get("name")), []).append(r)
    return index


def _same_value(selected: object, trusted: object) -> bool:
    """Strict JSON equality without Python's True == 1 or NaN loopholes."""
    if isinstance(selected, bool) or isinstance(trusted, bool):
        return type(selected) is type(trusted) and selected == trusted
    if isinstance(selected, (int, float)) and isinstance(trusted, (int, float)):
        try:
            return math.isfinite(selected) and math.isfinite(trusted) and selected == trusted
        except (ValueError, OverflowError):
            return False
    if type(selected) is not type(trusted):
        return False
    if isinstance(selected, dict):
        return selected.keys() == trusted.keys() and all(_same_value(v, trusted[k]) for k, v in selected.items())
    if isinstance(selected, list):
        return len(selected) == len(trusted) and all(_same_value(a, b) for a, b in zip(selected, trusted))
    return selected == trusted


def _row_matches(row: dict, trusted: dict) -> str | None:
    """None when every included field matches; else the first offending field."""
    for key, val in row.items():
        if key == "name":
            continue
        if key not in trusted:
            return key
        if not _same_value(val, trusted[key]):
            return key
    return None


def _resolved_row_sources(packet: dict, digest: dict, selected_indices: list[int] | None) -> list[int | None]:
    """Resolve verified inline subsets and actual appended selectors separately.

    A model-supplied source_selection is never authority. Legacy partial NCU
    rows need their group identity as well as a unique strict field match.
    """
    rows = packet.get("profile_data", {}).get("kernel_breakdown", [])
    source = digest.get("profile_breakdown", [])
    appended = selected_indices or []
    inline_count = len(rows) - len(appended)
    resolved = []
    for position, row in enumerate(rows):
        if position >= inline_count:
            index = appended[position - inline_count]
            resolved.append(index if _same_value(row, source[index]) else None)
            continue
        candidates = [i for i, trusted in enumerate(source)
                      if isinstance(trusted, dict) and trusted.get("name") == row.get("name")
                      and _row_matches(row, trusted) is None]
        index = candidates[0] if len(candidates) == 1 else None
        if index is not None and "ncu_group_id" in source[index] and "ncu_group_id" not in row:
            index = None
        resolved.append(index)
    return resolved


def verify_subset(packet: dict, trusted_digest: dict) -> dict:
    """Returns {ok} or {ok: False, error}.  Each profile_data row must be a
    verbatim field-subset of SOME trusted digest row with the same name."""
    if not isinstance(packet, dict) or not isinstance(trusted_digest, dict):
        return {"ok": False, "error": "packet and trusted digest must be objects"}
    try:
        packet, _ = _expand_row_selection(packet, trusted_digest)
        packet, _ = _expand_counter_selection(packet, trusted_digest)
    except ValueError as error:
        return {"ok": False, "error": str(error)}
    data = packet.get("profile_data", {})
    if not isinstance(data, dict):
        return {"ok": False, "error": "profile_data must be an object"}
    if set(data) - DATA_FIELDS:
        return {"ok": False, "error": "profile_data contains unknown sections"}
    rows = data.get("kernel_breakdown", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return {"ok": False, "error": "kernel_breakdown must be an array of objects"}
    index = _rows_index(trusted_digest.get("profile_breakdown") or [])
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or not name or len(row) < 2:
            return {"ok": False, "error": "each kernel row needs a name and measured fields"}
        candidates = index.get(name)
        if not candidates:
            return {"ok": False, "error": f"row '{name}' not present in trusted digest"}
        if all(_row_matches(row, trusted) is not None for trusted in candidates):
            return {"ok": False,
                    "error": f"row '{name}' does not match any trusted digest row "
                             "with that name (field values differ)"}
    counters = data.get("counters", {})
    trusted_counters = trusted_digest.get("counters") or {}
    if not isinstance(counters, dict) or not isinstance(trusted_counters, dict):
        return {"ok": False, "error": "counters must be an object"}
    for key, val in counters.items():
        if key not in trusted_counters:
            return {"ok": False, "error": f"counter '{key}' not in trusted digest"}
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not _same_value(val, trusted_counters[key]):
            return {"ok": False, "error": f"counter '{key}' value mismatch"}
    for section in ("runtime_context", "intermediate_traffic"):
        selected = data.get(section, {})
        trusted = trusted_digest.get(section) or {}
        if not isinstance(selected, dict) or not isinstance(trusted, dict):
            return {"ok": False, "error": f"{section} must be an object"}
        for key, value in selected.items():
            if key not in trusted:
                return {
                    "ok": False,
                    "error": f"{section} field '{key}' not in trusted digest",
                }
            if not _same_value(value, trusted[key]):
                return {
                    "ok": False,
                    "error": f"{section} field '{key}' value mismatch",
                }
    opportunities = data.get("static_opportunities", [])
    trusted_opportunities = trusted_digest.get("static_opportunities") or []
    if not isinstance(opportunities, list) or not isinstance(trusted_opportunities, list):
        return {"ok": False, "error": "static_opportunities must be an array"}
    for item in opportunities:
        if not isinstance(item, dict) or not any(_same_value(item, trusted) for trusted in trusted_opportunities):
            return {
                "ok": False,
                "error": "static opportunity is not a verbatim trusted digest item",
            }
    return {"ok": True}


def seal_packet(packet: dict, trusted_digest: dict, *, backend: str) -> dict:
    """Validate, bound, and wrap a packet for delivery; raises on violation."""
    if not isinstance(packet, dict):
        raise TypeError("evidence packet must be an object")
    try:
        encoded = json.dumps(packet, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("evidence must be finite JSON data") from error
    if len(encoded.encode("utf-8")) > MAX_PACKET_BYTES:
        raise ValueError("evidence exceeds 16 KiB; select fewer source items")
    if set(packet) - ({"target_layer", "verdict", "confidence", "profile_data"} | set(FREE_TEXT_FIELDS) | AUTHORITY_FIELDS):
        raise ValueError("evidence contains unknown top-level fields")
    for field in FREE_TEXT_FIELDS:
        if field in packet and not isinstance(packet[field], str):
            raise ValueError(f"evidence {field} must be text")
        if field in packet and len(packet[field]) > MAX_FREE_TEXT_CHARS:
            raise ValueError(
                f"evidence {field} exceeds {MAX_FREE_TEXT_CHARS} characters; "
                "shorten the whole field while preserving the observation, uncertainty and next test; "
                "no text was truncated or delivered"
            )
    packet, selected_indices = _expand_row_selection(packet, trusted_digest)
    packet, selected_counters = _expand_counter_selection(packet, trusted_digest)
    if backend not in VERDICT_TO_LAYER:
        raise ValueError("unknown evidence backend")
    target = packet.get("target_layer")
    if not isinstance(target, str) or target not in {"L0", "L1", "L2"}:
        raise ValueError("evidence target_layer must be L0, L1, or L2")
    verdict = packet.get("verdict")
    mapping = VERDICT_TO_LAYER[backend]
    if not isinstance(verdict, str) or verdict not in mapping:
        raise ValueError("evidence verdict is not defined for this backend")
    allowed = mapping[verdict]
    allowed_layers = {allowed} if isinstance(allowed, str) else set(allowed)
    if "infra" in allowed_layers or target not in allowed_layers:
        raise ValueError("evidence verdict is incompatible with target_layer")
    confidence = packet.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.6 <= float(confidence) <= 1.0
    ):
        raise ValueError("evidence confidence must be finite and in [0.6, 1.0]")
    profile_data = packet.get("profile_data", {})
    if not isinstance(profile_data, dict):
        raise ValueError("evidence profile_data must be an object")
    if not any(profile_data.get(k) for k in (
        "kernel_breakdown", "counters", "runtime_context", "static_opportunities",
        "intermediate_traffic",
    )):
        raise ValueError("evidence needs nonempty source data; use [] for no evidence")
    check = verify_subset(packet, trusted_digest)
    selected_rows = profile_data.get("kernel_breakdown", [])
    deep_rows = any(
        isinstance(row, dict) and (any(key in row for key in ("kernel_time_ns_sum", "ncu_group_id", "launch_facts")) or any(key.endswith("_pct") for key in row))
        for row in selected_rows
    ) if isinstance(selected_rows, list) else False
    if profile_data.get("counters") or deep_rows:
        collector = trusted_digest.get("collector") or {}
        if (
            not isinstance(collector, dict)
            or collector.get("status") != "collected"
            or collector.get("exact_bundle") is not True
        ):
            raise ValueError(
                "deep counters require a collected exact-bundle source"
            )
    if not check["ok"]:
        raise ValueError(f"evidence subset verification failed: {check['error']}")
    sealed = copy.deepcopy(packet)
    sealed.pop("source_selection", None)
    if selected_indices is not None:
        sealed["source_selection"] = {"kernel_breakdown_indices": selected_indices,
                                      "index_base": 0, "source": "profile_breakdown"}
    if selected_counters is not None:
        sealed.setdefault("source_selection", {}).update(
            counter_names=selected_counters, counter_source="counters")
    # Never let a model provide or relabel the observation's authority.
    sealed["source_binding"] = copy.deepcopy(trusted_digest.get("binding"))
    sealed["source_digest_id"] = trusted_digest.get("digest_id") or hashlib.sha256(
        json.dumps(trusted_digest, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    sealed["claim_assessment"] = assess_packet(
        packet, trusted_digest, backend=backend,
        source_row_indices=_resolved_row_sources(packet, trusted_digest, selected_indices))
    for field in FREE_TEXT_FIELDS:
        if field in sealed and sealed[field]:
            value = sealed[field].replace("UNTRUSTED_BEGIN", "[untrusted-begin]").replace("UNTRUSTED_END", "[untrusted-end]")
            sealed[field] = f"UNTRUSTED_BEGIN {value} UNTRUSTED_END"
    # The identifier covers the final delivered representation. Do not hash
    # before wrapping text or silently drop evidence after computing the hash.
    sealed.pop("evidence_id", None)
    # Reserve the final fixed-width hash before computing it. Evaluator-owned
    # interpretation metadata is subject to exactly the same final byte cap.
    if len(json.dumps({**sealed, "evidence_id": "0" * 64}, ensure_ascii=False,
                      allow_nan=False).encode("utf-8")) > MAX_PACKET_BYTES:
        raise ValueError("sealed evidence exceeds 16 KiB; select fewer source items")
    sealed["evidence_id"] = hashlib.sha256(
        json.dumps(sealed, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return sealed
