"""Validate correctness tolerances and inputs from the original task.

The trusted task loader creates the record; generated BuildResult metadata
cannot set it. Older bundles can be inspected, but device execution requires
this record. This module has no compiler or device dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Mapping


SCHEMA = "pipeline.correctness_contract.v1"
PROFILE_ID = "ordinary_0_1_2_v1"
CONTRACT_KEY = "correctness_contract"
CONTRACT_SHA_KEY = "correctness_contract_sha256"
_KEYS = {"schema_version", "profile_id", "task_source_sha256", "atol", "rtol",
         "input_seeds", "input_distribution", "tolerance_authority", "tolerance_origin"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _tolerance(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("task tolerance must be a finite non-negative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("task tolerance must be a finite non-negative real number")
    return result


def validate_correctness_contract(
    value: Any, *, expected_task_source_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _KEYS:
        raise ValueError("original-task correctness contract is missing or malformed")
    digest = value.get("task_source_sha256")
    if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
        raise ValueError("correctness contract task source hash is invalid")
    if expected_task_source_sha256 is not None and digest != expected_task_source_sha256:
        raise ValueError("correctness contract task source hash differs")
    if (value.get("schema_version") != SCHEMA or value.get("profile_id") != PROFILE_ID
            or value.get("input_distribution") != "task_original_unscaled"
            or value.get("tolerance_authority") != "task_source"
            or value.get("input_seeds") != [0, 1, 2]
            or any(type(seed) is not int for seed in value.get("input_seeds", []))):
        raise ValueError("correctness contract formal input/authority policy differs")
    origins = value.get("tolerance_origin")
    if not isinstance(origins, Mapping) or set(origins) != {"atol", "rtol"}:
        raise ValueError("correctness contract tolerance origin is invalid")
    for name in ("atol", "rtol"):
        if origins[name] not in {f"module_{name.upper()}", f"CASE_METADATA.{name}", "default_1e-4"}:
            raise ValueError("correctness contract tolerance origin is invalid")
        if origins[name] == "default_1e-4" and _tolerance(value[name]) != 1e-4:
            raise ValueError("correctness contract default tolerance differs")
    return {**dict(value), "atol": _tolerance(value["atol"]),
            "rtol": _tolerance(value["rtol"]), "input_seeds": [0, 1, 2],
            "tolerance_origin": dict(origins)}


def correctness_contract_sha256(value: Any) -> str:
    contract = validate_correctness_contract(value)
    return hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def contract_from_task_module(module: Any, task_source_sha256: str) -> dict[str, Any]:
    metadata = getattr(module, "CASE_METADATA", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    tolerances: dict[str, float] = {}
    origins: dict[str, str] = {}
    for name in ("atol", "rtol"):
        if hasattr(module, name.upper()):
            value, origin = getattr(module, name.upper()), f"module_{name.upper()}"
        elif name in metadata:
            value, origin = metadata[name], f"CASE_METADATA.{name}"
        else:
            value, origin = 1e-4, "default_1e-4"
        tolerances[name], origins[name] = _tolerance(value), origin
    return validate_correctness_contract({
        "schema_version": SCHEMA, "profile_id": PROFILE_ID,
        "task_source_sha256": task_source_sha256, **tolerances,
        "input_seeds": [0, 1, 2], "input_distribution": "task_original_unscaled",
        "tolerance_authority": "task_source", "tolerance_origin": origins,
    })


def contract_from_metadata(metadata: Any, *, required: bool = True) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        raise ValueError("correctness metadata must be an object")
    if CONTRACT_KEY not in metadata and CONTRACT_SHA_KEY not in metadata and not required:
        return None  # Historical read-only artifact inspection; never native execution.
    task_sha = metadata.get("task_source_sha256")
    if not isinstance(task_sha, str) or _SHA.fullmatch(task_sha) is None:
        raise ValueError("correctness metadata lacks the trusted task source hash")
    contract = validate_correctness_contract(metadata.get(CONTRACT_KEY), expected_task_source_sha256=task_sha)
    if metadata.get(CONTRACT_SHA_KEY) != correctness_contract_sha256(contract):
        raise ValueError("correctness contract hash differs")
    return contract


def correctness_result_matches_contract(stage: Any, contract: Any, *, require_baseline: bool = False) -> bool:
    """Recheck result policy and per-leaf effective bounds before qualification."""
    try:
        expected = validate_correctness_contract(contract)
        if (not isinstance(stage, Mapping)
                or stage.get("status") != "passed"
                or stage.get("profile_id") != PROFILE_ID
                or stage.get("formal_seeds") != [0, 1, 2]
                or stage.get("seed_count") != 3
                or stage.get(CONTRACT_KEY) != expected
                or stage.get(CONTRACT_SHA_KEY) != correctness_contract_sha256(expected)):
            return False
        seeds = stage.get("seeds")
        if (not isinstance(seeds, list) or len(seeds) != 3
                or [seed.get("seed") for seed in seeds if isinstance(seed, Mapping)] != [0, 1, 2]):
            return False
        for seed in seeds:
            if type(seed.get("seed")) is not int:
                return False
            for role in ("candidate", "baseline"):
                report = seed.get(role)
                if report is None:
                    if role == "candidate" or require_baseline:
                        return False
                    continue
                leaves = report.get("leaves")
                if report.get("passed") is not True or not isinstance(leaves, list) or not leaves:
                    return False
                for leaf in leaves:
                    dtype = str(leaf.get("expected_dtype", ""))
                    bounds = (expected["atol"], expected["rtol"]) if dtype.startswith(("float", "bfloat")) else (0.0, 0.0)
                    if (not dtype or leaf.get("actual_dtype") != dtype
                            or leaf.get("passed") is not True or type(leaf.get("mismatch_count")) is not int
                            or leaf.get("mismatch_count") != 0
                            or not isinstance(leaf.get("expected_shape"), (list, tuple))
                            or leaf.get("actual_shape") != leaf.get("expected_shape")
                            or isinstance(leaf.get("atol"), bool) or isinstance(leaf.get("rtol"), bool)
                            or (leaf.get("atol"), leaf.get("rtol")) != bounds):
                        return False
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def ordinary_input_authority(contents: Any) -> tuple[str, str]:
    """Bind a verified bundle's *actual* ordinary inputs to source and bounds."""
    metadata = contents.metadata
    contract = contract_from_metadata(metadata)
    identities = metadata.get("correctness_input_identity_sha256_by_seed")
    if (metadata.get("correctness_nonce_sha256") is not None
            or metadata.get("correctness_input_authority") not in (None, "canonical_seeded")
            or not isinstance(identities, Mapping) or set(identities) != {"0", "1", "2"}
            or any(not isinstance(value, str) or _SHA.fullmatch(value) is None
                   for value in identities.values())
            or contents.manifest.get("correctness_seeds") != [0, 1, 2]
            or type(contents.manifest.get("parameter_seed")) is not int):
        raise ValueError("ordinary audit requires trusted unrandomized 0/1/2 input identities")
    kind = "trusted_canonical_seed_task_loader"
    payload = {"kind": kind, "task_source_sha256": contract["task_source_sha256"],
               "parameter_seed": contents.manifest["parameter_seed"],
               "correctness_contract_sha256": correctness_contract_sha256(contract),
               "correctness_input_identity_sha256_by_seed": dict(identities)}
    return kind, hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                          separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def mutation_tolerance_matches_contract(sentinel: Any, contract: Any) -> bool:
    """A diagnostic sentinel may neither replace nor halve the task's bounds."""
    try:
        expected = validate_correctness_contract(contract)
        tolerance = sentinel.get("tolerance", {})
        if (tolerance.get("audit_factor") != 1.0
                or isinstance(tolerance.get("audit_factor"), bool)
                or tolerance.get("canonical_atol") != expected["atol"]
                or tolerance.get("canonical_rtol") != expected["rtol"]):
            return False
        leaves = tolerance.get("per_leaf_canonical")
        if not isinstance(leaves, list) or not leaves:
            return False
        for leaf in leaves:
            dtype = leaf.get("dtype", "")
            bounds = (expected["atol"], expected["rtol"]) if dtype.startswith(("float", "bfloat")) else (0.0, 0.0)
            if (not dtype or (leaf.get("canonical_atol"), leaf.get("canonical_rtol")) != bounds
                    or (leaf.get("audit_atol"), leaf.get("audit_rtol")) != bounds):
                return False
        # Bounded attempts carry individual, hash-bound copies of the policy.
        attempts = sentinel.get("bounded_mutation", {}).get("attempts", [])
        return isinstance(attempts, list) and all(
            mutation_tolerance_matches_contract(item, expected) for item in attempts)
    except (ValueError, TypeError, AttributeError):
        return False
