"""Declarative, evaluator-owned preparation of fixed PyTorch state.

The L0 agent attaches a JSON plan to ``main`` without seeing parameter values.
The evaluator prepares fixed state before timing and checks the resulting
program against PyTorch. Dynamic inputs cannot enter the preparation plan.
Version 1 supports layout transforms; version 2 adds a static expression DAG.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


ATTR_NAME = "leka_parameter_plan"
# Preserve the v1 identifier for existing layout-only plans.
SCHEMA_VERSION = "leka.parameter_plan.v1"
EXPRESSION_SCHEMA_VERSION = "leka.parameter_plan.v2"
MAX_BINDINGS = 256
MAX_TRANSFORMS_PER_BINDING = 8
MAX_NODES = 512
MAX_NODE_INPUTS = 64
MAX_NODE_ID_LENGTH = 64
MAX_RANK = 16
MAX_ELEMENTS = 1 << 30
# These limits bound materialized fixed state, not dynamic inputs. Per-array,
# total-size, and relative amplification limits prevent padding or duplication
# from consuming excessive evaluator memory.
MAX_SINGLE_OUTPUT_BYTES = 1 << 30
MAX_TOTAL_OUTPUT_BYTES = 2 << 30
MAX_STATE_AMPLIFICATION = 4
MAX_STATE_OVERHEAD_BYTES = 64 << 20
ALLOWED_CAST_DTYPES = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
    }
)


class ParameterPlanError(ValueError):
    """A candidate's fixed-state preparation plan is invalid."""


def schema_document() -> dict[str, Any]:
    """Return the immutable agent-facing plan schema and timing contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "supported_schema_versions": [SCHEMA_VERSION, EXPRESSION_SCHEMA_VERSION],
        "recommended_schema_version": EXPRESSION_SCHEMA_VERSION,
        "relax_main_attribute": ATTR_NAME,
        "encoding": "canonical JSON string stored in R.func_attr",
        "timing_boundary": {
            "parameter_preparation": "outside_saved_relax_entry_closure",
            "device_transfer": "outside_saved_relax_entry_closure",
            "relax_main": "timed",
        },
        "plan": {
            "schema_version": SCHEMA_VERSION,
            "bindings": [
                {
                    "source": "a relax_name from PYTORCH_SEMANTICS.json",
                    "transforms": [
                        {"op": "transpose", "axes": [1, 0]},
                        {"op": "reshape", "shape": [1, -1]},
                        {"op": "pad", "widths": [[0, 0], [0, 8]], "value": 0},
                        {
                            "op": "slice",
                            "starts": [0, 0],
                            "stops": [None, None],
                            "steps": [1, 1],
                        },
                        {"op": "cast", "dtype": "float16"},
                        {"op": "contiguous"},
                    ],
                }
            ],
        },
        "expression_plan": {
            "schema_version": EXPRESSION_SCHEMA_VERSION,
            "nodes_are_topologically_ordered": True,
            "nodes": [
                {"id": "weight", "op": "source", "source": "p_conv_weight"},
                {"id": "gamma", "op": "source", "source": "p_bn_weight"},
                {"id": "variance", "op": "source", "source": "b_bn_running_var"},
                {"id": "epsilon", "op": "scalar", "value": 1e-05, "dtype": "float32"},
                {"id": "variance_epsilon", "op": "add", "inputs": ["variance", "epsilon"]},
                {"id": "stddev", "op": "sqrt", "input": "variance_epsilon"},
                {"id": "scale", "op": "div", "inputs": ["gamma", "stddev"]},
                {"id": "scale_nchw", "op": "reshape", "input": "scale", "shape": [-1, 1, 1, 1]},
                {"id": "folded_weight", "op": "mul", "inputs": ["weight", "scale_nchw"]},
            ],
            "bindings": [{"value": "folded_weight"}],
            "operators": {
                "leaves": ["source", "scalar"],
                "elementwise": ["add", "sub", "mul", "div", "sqrt"],
                "combine": ["concat"],
                "layout": [
                    "transpose",
                    "reshape",
                    "pad",
                    "slice",
                    "cast",
                    "contiguous",
                ],
            },
        },
        "semantics": [
            "bindings correspond positionally to main parameters after num_input",
            "a source may be duplicated and an unused source may be omitted",
            "an absent attribute means identity bindings in the original state order",
            "v2 nodes are shape-static and may reference only earlier node ids",
            "v2 has no comparisons, branches, reductions, indexing by values, or value-derived shapes",
            "v2 intermediate values remain inside the trusted evaluator and are never reported",
            "all transformed shapes/dtypes are checked before lowering",
            "all candidates must pass hidden PyTorch correctness and fresh-seed audit",
        ],
        "limits": {
            "max_bindings": MAX_BINDINGS,
            "max_transforms_per_binding": MAX_TRANSFORMS_PER_BINDING,
            "max_nodes": MAX_NODES,
            "max_node_inputs": MAX_NODE_INPUTS,
            "max_node_id_length": MAX_NODE_ID_LENGTH,
            "max_rank": MAX_RANK,
            "max_elements_per_binding": MAX_ELEMENTS,
            "max_single_output_bytes": MAX_SINGLE_OUTPUT_BYTES,
            "max_total_output_bytes": MAX_TOTAL_OUTPUT_BYTES,
            "max_total_dag_value_bytes": MAX_TOTAL_OUTPUT_BYTES,
            "max_state_amplification": MAX_STATE_AMPLIFICATION,
            "max_dag_state_amplification": MAX_STATE_AMPLIFICATION,
            "max_state_overhead_bytes": MAX_STATE_OVERHEAD_BYTES,
            "allowed_cast_dtypes": sorted(ALLOWED_CAST_DTYPES),
        },
    }


def default_plan(source_state_names: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "bindings": [
            {"source": str(name), "transforms": []} for name in source_state_names
        ],
    }


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterPlanError(f"{label} must be an integer")
    return int(value)


def _optional_integer(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label=label)


def _number(value: object, *, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterPlanError(f"{label} must be a finite number")
    if not math.isfinite(float(value)):
        raise ParameterPlanError(f"{label} must be a finite number")
    return value


_PLAIN_DTYPE = re.compile(
    r"^(?P<kind>bfloat|float|int|uint|complex)(?P<bits>[0-9]+)(?:x(?P<lanes>[0-9]+))?$"
)


def _dtype_itemsize(dtype: str) -> int:
    """Return bytes per tensor element without inspecting tensor contents."""

    name = str(dtype)
    if name == "bool":
        return 1
    match = _PLAIN_DTYPE.fullmatch(name)
    if match is None:
        raise ParameterPlanError(
            f"parameter resource limit exceeded: unsupported_dtype; dtype={name}"
        )
    bits = int(match.group("bits"))
    lanes = int(match.group("lanes") or "1")
    if bits <= 0 or lanes <= 0 or bits % 8:
        raise ParameterPlanError(
            f"parameter resource limit exceeded: unsupported_dtype; dtype={name}"
        )
    return (bits // 8) * lanes


def _tensor_nbytes(
    shape: Sequence[int],
    dtype: str,
    *,
    enforce_output_limit: bool,
) -> int:
    """Compute a scalar/zero-size-safe byte count from public tensor specs."""

    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) > MAX_RANK or any(dim < 0 for dim in normalized_shape):
        raise ParameterPlanError(
            "parameter resource limit exceeded: invalid_shape; "
            f"shape={normalized_shape}, dtype={dtype}"
        )
    elements = math.prod(normalized_shape)
    if enforce_output_limit and elements > MAX_ELEMENTS:
        raise ParameterPlanError(
            "parameter resource limit exceeded: element_count; "
            f"shape={normalized_shape}, dtype={dtype}"
        )
    nbytes = elements * _dtype_itemsize(str(dtype))
    if enforce_output_limit and nbytes > MAX_SINGLE_OUTPUT_BYTES:
        raise ParameterPlanError(
            "parameter resource limit exceeded: single_output_bytes; "
            f"shape={normalized_shape}, dtype={dtype}"
        )
    return nbytes


def _enforce_aggregate_bytes(*, output_bytes: int, source_bytes: int) -> None:
    if output_bytes > MAX_TOTAL_OUTPUT_BYTES:
        raise ParameterPlanError(
            "parameter resource limit exceeded: total_output_bytes"
        )
    relative_limit = (
        source_bytes * MAX_STATE_AMPLIFICATION + MAX_STATE_OVERHEAD_BYTES
    )
    if output_bytes > relative_limit:
        raise ParameterPlanError(
            "parameter resource limit exceeded: state_amplification"
        )


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ParameterPlanError(
            f"{label} keys differ: expected {sorted(expected)}, got {sorted(value)}"
        )


def _normalize_transform(raw: object, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ParameterPlanError(f"{label} must be an object")
    op = raw.get("op")
    if op == "transpose":
        _exact_keys(raw, {"op", "axes"}, label=label)
        axes = raw["axes"]
        if not isinstance(axes, list):
            raise ParameterPlanError(f"{label}.axes must be a list")
        return {
            "op": op,
            "axes": [_integer(item, label=f"{label}.axes") for item in axes],
        }
    if op == "reshape":
        _exact_keys(raw, {"op", "shape"}, label=label)
        shape = raw["shape"]
        if not isinstance(shape, list):
            raise ParameterPlanError(f"{label}.shape must be a list")
        normalized = [_integer(item, label=f"{label}.shape") for item in shape]
        if any(dim < -1 for dim in normalized) or normalized.count(-1) > 1:
            raise ParameterPlanError(f"{label}.shape has an invalid inferred dimension")
        return {"op": op, "shape": normalized}
    if op == "pad":
        _exact_keys(raw, {"op", "widths", "value"}, label=label)
        widths = raw["widths"]
        if not isinstance(widths, list):
            raise ParameterPlanError(f"{label}.widths must be a list")
        normalized_widths: list[list[int]] = []
        for index, pair in enumerate(widths):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ParameterPlanError(f"{label}.widths[{index}] must be [before, after]")
            before = _integer(pair[0], label=f"{label}.widths[{index}][0]")
            after = _integer(pair[1], label=f"{label}.widths[{index}][1]")
            if before < 0 or after < 0:
                raise ParameterPlanError(f"{label}.widths must be non-negative")
            normalized_widths.append([before, after])
        return {
            "op": op,
            "widths": normalized_widths,
            "value": _number(raw["value"], label=f"{label}.value"),
        }
    if op == "slice":
        _exact_keys(raw, {"op", "starts", "stops", "steps"}, label=label)
        fields: dict[str, list[int | None]] = {}
        for field in ("starts", "stops", "steps"):
            values = raw[field]
            if not isinstance(values, list):
                raise ParameterPlanError(f"{label}.{field} must be a list")
            fields[field] = [
                _optional_integer(item, label=f"{label}.{field}") for item in values
            ]
        if len({len(fields[name]) for name in fields}) != 1:
            raise ParameterPlanError(f"{label} slice arrays must have equal length")
        if any(step == 0 for step in fields["steps"]):
            raise ParameterPlanError(f"{label}.steps cannot contain zero")
        return {"op": op, **fields}
    if op == "cast":
        _exact_keys(raw, {"op", "dtype"}, label=label)
        dtype = raw["dtype"]
        if not isinstance(dtype, str) or dtype not in ALLOWED_CAST_DTYPES:
            raise ParameterPlanError(f"{label}.dtype is not allowed")
        return {"op": op, "dtype": dtype}
    if op == "contiguous":
        _exact_keys(raw, {"op"}, label=label)
        return {"op": op}
    raise ParameterPlanError(f"{label}.op is unsupported: {op!r}")


_NODE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_BINARY_OPS = frozenset({"add", "sub", "mul", "div"})
_UNARY_VALUE_OPS = frozenset({"sqrt"})
_LAYOUT_OPS = frozenset(
    {"transpose", "reshape", "pad", "slice", "cast", "contiguous"}
)
_FLOAT_MAX = {
    "float16": 65504.0,
    "float32": 3.4028234663852886e38,
    "float64": 1.7976931348623157e308,
}


def _normalized_node_id(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > MAX_NODE_ID_LENGTH
        or _NODE_ID.fullmatch(value) is None
    ):
        raise ParameterPlanError(
            f"{label} must match [A-Za-z][A-Za-z0-9_]{{0,{MAX_NODE_ID_LENGTH - 1}}}"
        )
    return value


def _normalized_scalar(value: object, dtype: object, *, label: str) -> int | float:
    if not isinstance(dtype, str) or dtype not in ALLOWED_CAST_DTYPES:
        raise ParameterPlanError(f"{label}.dtype is not allowed")
    number = _number(value, label=f"{label}.value")
    if dtype == "bool":
        if not isinstance(number, int) or number not in (0, 1):
            raise ParameterPlanError(
                f"{label}.value is not representable; dtype={dtype}"
            )
        return int(number)
    match = _PLAIN_DTYPE.fullmatch(dtype)
    if match is None:
        raise ParameterPlanError(f"{label}.dtype is not allowed")
    kind = match.group("kind")
    bits = int(match.group("bits"))
    if kind in {"int", "uint"}:
        if not isinstance(number, int):
            raise ParameterPlanError(
                f"{label}.value is not representable; dtype={dtype}"
            )
        minimum = 0 if kind == "uint" else -(1 << (bits - 1))
        maximum = (1 << bits) - 1 if kind == "uint" else (1 << (bits - 1)) - 1
        if not minimum <= number <= maximum:
            raise ParameterPlanError(
                f"{label}.value is not representable; dtype={dtype}"
            )
        return int(number)
    if kind == "float":
        normalized = float(number)
        if abs(normalized) > _FLOAT_MAX[dtype]:
            raise ParameterPlanError(
                f"{label}.value is not representable; dtype={dtype}"
            )
        return normalized
    raise ParameterPlanError(f"{label}.dtype is not allowed")


def _normalized_references(
    values: object,
    *,
    label: str,
    known_node_ids: set[str],
    exact_count: int | None = None,
) -> list[str]:
    if not isinstance(values, list):
        raise ParameterPlanError(f"{label} must be a list")
    if exact_count is not None and len(values) != exact_count:
        raise ParameterPlanError(f"{label} must contain exactly {exact_count} ids")
    if exact_count is None and not 1 <= len(values) <= MAX_NODE_INPUTS:
        raise ParameterPlanError(
            f"{label} must contain between 1 and {MAX_NODE_INPUTS} ids"
        )
    result: list[str] = []
    for value in values:
        reference = _normalized_node_id(value, label=label)
        if reference not in known_node_ids:
            raise ParameterPlanError(
                f"{label} must reference only earlier v2 node ids"
            )
        result.append(reference)
    return result


def _normalize_v2_node(
    raw: object,
    *,
    label: str,
    source_names: set[str],
    known_node_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ParameterPlanError(f"{label} must be an object")
    node_id = _normalized_node_id(raw.get("id"), label=f"{label}.id")
    if node_id in known_node_ids:
        raise ParameterPlanError(f"{label}.id must be unique")
    op = raw.get("op")
    if op == "source":
        _exact_keys(raw, {"id", "op", "source"}, label=label)
        source = raw["source"]
        if not isinstance(source, str) or source not in source_names:
            raise ParameterPlanError(f"{label}.source does not name trusted fixed state")
        return {"id": node_id, "op": op, "source": source}
    if op == "scalar":
        _exact_keys(raw, {"id", "op", "value", "dtype"}, label=label)
        dtype = raw["dtype"]
        return {
            "id": node_id,
            "op": op,
            "value": _normalized_scalar(raw["value"], dtype, label=label),
            "dtype": dtype,
        }
    if op in _BINARY_OPS:
        _exact_keys(raw, {"id", "op", "inputs"}, label=label)
        return {
            "id": node_id,
            "op": op,
            "inputs": _normalized_references(
                raw["inputs"],
                label=f"{label}.inputs",
                known_node_ids=known_node_ids,
                exact_count=2,
            ),
        }
    if op in _UNARY_VALUE_OPS:
        _exact_keys(raw, {"id", "op", "input"}, label=label)
        reference = _normalized_node_id(raw["input"], label=f"{label}.input")
        if reference not in known_node_ids:
            raise ParameterPlanError(
                f"{label}.input must reference an earlier v2 node id"
            )
        return {"id": node_id, "op": op, "input": reference}
    if op == "concat":
        _exact_keys(raw, {"id", "op", "inputs", "axis"}, label=label)
        return {
            "id": node_id,
            "op": op,
            "inputs": _normalized_references(
                raw["inputs"],
                label=f"{label}.inputs",
                known_node_ids=known_node_ids,
            ),
            "axis": _integer(raw["axis"], label=f"{label}.axis"),
        }
    if op in _LAYOUT_OPS:
        reference = _normalized_node_id(raw.get("input"), label=f"{label}.input")
        if reference not in known_node_ids:
            raise ParameterPlanError(
                f"{label}.input must reference an earlier v2 node id"
            )
        transform = _normalize_transform(
            {key: value for key, value in raw.items() if key not in {"id", "input"}},
            label=label,
        )
        return {"id": node_id, "input": reference, **transform}
    raise ParameterPlanError(f"{label}.op is unsupported: {op!r}")


def _v2_node_references(node: Mapping[str, Any]) -> tuple[str, ...]:
    op = node["op"]
    if op in _BINARY_OPS or op == "concat":
        return tuple(node["inputs"])
    if op in _UNARY_VALUE_OPS or op in _LAYOUT_OPS:
        return (str(node["input"]),)
    return ()


def _normalize_v1_plan(
    raw: Mapping[str, Any], sources: tuple[str, ...]
) -> dict[str, Any]:
    _exact_keys(raw, {"schema_version", "bindings"}, label="parameter plan")
    bindings = raw["bindings"]
    if not isinstance(bindings, list) or len(bindings) > MAX_BINDINGS:
        raise ParameterPlanError(
            f"parameter plan bindings must be a list of at most {MAX_BINDINGS}"
        )
    normalized_bindings: list[dict[str, Any]] = []
    source_set = set(sources)
    for index, binding in enumerate(bindings):
        label = f"bindings[{index}]"
        if not isinstance(binding, Mapping):
            raise ParameterPlanError(f"{label} must be an object")
        _exact_keys(binding, {"source", "transforms"}, label=label)
        source = binding["source"]
        if not isinstance(source, str) or source not in source_set:
            raise ParameterPlanError(f"{label}.source does not name trusted fixed state")
        transforms = binding["transforms"]
        if not isinstance(transforms, list) or len(transforms) > MAX_TRANSFORMS_PER_BINDING:
            raise ParameterPlanError(
                f"{label}.transforms must contain at most {MAX_TRANSFORMS_PER_BINDING} items"
            )
        normalized_bindings.append(
            {
                "source": source,
                "transforms": [
                    _normalize_transform(item, label=f"{label}.transforms[{op_index}]")
                    for op_index, item in enumerate(transforms)
                ],
            }
        )
    return {"schema_version": SCHEMA_VERSION, "bindings": normalized_bindings}


def _normalize_v2_plan(
    raw: Mapping[str, Any], sources: tuple[str, ...]
) -> dict[str, Any]:
    _exact_keys(raw, {"schema_version", "nodes", "bindings"}, label="parameter plan")
    nodes = raw["nodes"]
    if not isinstance(nodes, list) or len(nodes) > MAX_NODES:
        raise ParameterPlanError(
            f"parameter plan nodes must be a list of at most {MAX_NODES}"
        )
    normalized_nodes: list[dict[str, Any]] = []
    known_node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        normalized = _normalize_v2_node(
            node,
            label=f"nodes[{index}]",
            source_names=set(sources),
            known_node_ids=known_node_ids,
        )
        normalized_nodes.append(normalized)
        known_node_ids.add(normalized["id"])

    bindings = raw["bindings"]
    if not isinstance(bindings, list) or len(bindings) > MAX_BINDINGS:
        raise ParameterPlanError(
            f"parameter plan bindings must be a list of at most {MAX_BINDINGS}"
        )
    normalized_bindings: list[dict[str, str]] = []
    for index, binding in enumerate(bindings):
        label = f"bindings[{index}]"
        if not isinstance(binding, Mapping):
            raise ParameterPlanError(f"{label} must be an object")
        _exact_keys(binding, {"value"}, label=label)
        value = _normalized_node_id(binding["value"], label=f"{label}.value")
        if value not in known_node_ids:
            raise ParameterPlanError(f"{label}.value does not name a v2 node")
        normalized_bindings.append({"value": value})

    nodes_by_id = {node["id"]: node for node in normalized_nodes}
    reachable = {binding["value"] for binding in normalized_bindings}
    pending = list(reachable)
    while pending:
        node_id = pending.pop()
        for reference in _v2_node_references(nodes_by_id[node_id]):
            if reference not in reachable:
                reachable.add(reference)
                pending.append(reference)
    if reachable != known_node_ids:
        raise ParameterPlanError("v2 parameter plan contains unreachable nodes")
    return {
        "schema_version": EXPRESSION_SCHEMA_VERSION,
        "nodes": normalized_nodes,
        "bindings": normalized_bindings,
    }


def normalize_plan(
    raw: object | None, source_state_names: Sequence[str]
) -> dict[str, Any]:
    """Validate and canonicalize one plan without touching tensor values."""

    sources = tuple(str(name) for name in source_state_names)
    if len(set(sources)) != len(sources) or any(not name for name in sources):
        raise ParameterPlanError("source state names must be unique non-empty strings")
    if raw is None:
        return default_plan(sources)
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > 64 * 1024:
            raise ParameterPlanError("parameter plan exceeds 64 KiB")
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ParameterPlanError("parameter plan attribute is not valid JSON") from error
    if not isinstance(raw, Mapping):
        raise ParameterPlanError("parameter plan must be an object")
    version = raw.get("schema_version")
    if version == SCHEMA_VERSION:
        return _normalize_v1_plan(raw, sources)
    if version == EXPRESSION_SCHEMA_VERSION:
        return _normalize_v2_plan(raw, sources)
    raise ParameterPlanError("unsupported parameter plan schema_version")


def plan_from_main(main: object, source_state_names: Sequence[str]) -> dict[str, Any]:
    """Read and normalize the plan attached to a Relax ``main`` function."""

    attrs = getattr(main, "attrs", None)
    raw = None
    if attrs is not None:
        try:
            if ATTR_NAME in attrs:
                raw = str(attrs[ATTR_NAME])
        except (TypeError, KeyError):
            raw = None
    return normalize_plan(raw, source_state_names)


def transformed_spec(
    shape: Sequence[int], dtype: str, transforms: Sequence[Mapping[str, Any]]
) -> tuple[tuple[int, ...], str]:
    """Compute a transform chain's static output spec without allocating it."""

    current = tuple(int(dim) for dim in shape)
    current_dtype = str(dtype)
    if len(current) > MAX_RANK or any(dim < 0 for dim in current):
        raise ParameterPlanError("source tensor has an unsupported shape")
    for transform in transforms:
        op = transform["op"]
        if op == "transpose":
            axes = tuple(transform["axes"])
            if len(axes) != len(current) or set(axes) != set(range(len(current))):
                raise ParameterPlanError("transpose axes are not a permutation of the rank")
            current = tuple(current[axis] for axis in axes)
        elif op == "reshape":
            target = list(transform["shape"])
            source_elements = math.prod(current)
            known = math.prod(dim for dim in target if dim != -1)
            if -1 in target:
                if known == 0 or source_elements % known:
                    raise ParameterPlanError("reshape cannot infer an integral dimension")
                target[target.index(-1)] = source_elements // known
            if math.prod(target) != source_elements:
                raise ParameterPlanError("reshape changes the tensor element count")
            current = tuple(target)
        elif op == "pad":
            widths = transform["widths"]
            if len(widths) != len(current):
                raise ParameterPlanError("pad widths rank differs from the tensor rank")
            current = tuple(
                dim + int(pair[0]) + int(pair[1]) for dim, pair in zip(current, widths)
            )
        elif op == "slice":
            if len(transform["starts"]) != len(current):
                raise ParameterPlanError("slice arrays rank differs from the tensor rank")
            current = tuple(
                len(
                    range(
                        *slice(start, stop, step).indices(dim)
                    )
                )
                for dim, start, stop, step in zip(
                    current,
                    transform["starts"],
                    transform["stops"],
                    transform["steps"],
                )
            )
        elif op == "cast":
            current_dtype = str(transform["dtype"])
        elif op == "contiguous":
            pass
        else:  # normalize_plan makes this unreachable.
            raise ParameterPlanError(f"unsupported transform: {op}")
        # Check every intermediate spec before NumPy executes the corresponding
        # operation.  This matters for pad/cast and for reshape operations that
        # may need a copy after a non-contiguous transpose.
        _tensor_nbytes(current, current_dtype, enforce_output_limit=True)
    # An identity binding has no loop iteration above, but is still one output
    # in the materialized ABI and must obey the same single-output limits.
    _tensor_nbytes(current, current_dtype, enforce_output_limit=True)
    return current, current_dtype


def _broadcast_shape(*shapes: Sequence[int]) -> tuple[int, ...]:
    """Infer NumPy-style broadcasting using public, static shapes only."""

    result: list[int] = []
    max_rank = max((len(shape) for shape in shapes), default=0)
    for offset in range(1, max_rank + 1):
        dimension = 1
        for shape in shapes:
            candidate = int(shape[-offset]) if offset <= len(shape) else 1
            if candidate == 1:
                continue
            if dimension == 1:
                dimension = candidate
            elif candidate != dimension:
                raise ParameterPlanError(
                    "v2 parameter expression has incompatible broadcast shapes"
                )
        result.append(dimension)
    return tuple(reversed(result))


def _arithmetic_spec(
    op: str,
    left: tuple[tuple[int, ...], str],
    right: tuple[tuple[int, ...], str],
) -> tuple[tuple[int, ...], str]:
    left_shape, left_dtype = left
    right_shape, right_dtype = right
    if left_dtype != right_dtype:
        raise ParameterPlanError(
            "v2 parameter expression arithmetic dtypes must match; "
            f"dtype={left_dtype}/{right_dtype}"
        )
    match = _PLAIN_DTYPE.fullmatch(left_dtype)
    kind = None if match is None else match.group("kind")
    if kind not in {"float", "int", "uint", "complex"}:
        raise ParameterPlanError(
            f"v2 parameter expression arithmetic dtype is unsupported; dtype={left_dtype}"
        )
    if op == "div" and kind not in {"float", "complex"}:
        raise ParameterPlanError(
            f"v2 parameter expression div requires floating dtype; dtype={left_dtype}"
        )
    return _broadcast_shape(left_shape, right_shape), left_dtype


def _concat_spec(
    inputs: Sequence[tuple[tuple[int, ...], str]], axis: int
) -> tuple[tuple[int, ...], str]:
    first_shape, first_dtype = inputs[0]
    rank = len(first_shape)
    if rank == 0:
        raise ParameterPlanError("v2 parameter expression cannot concat rank-zero tensors")
    normalized_axis = axis + rank if axis < 0 else axis
    if normalized_axis < 0 or normalized_axis >= rank:
        raise ParameterPlanError("v2 parameter expression concat axis is out of range")
    output = list(first_shape)
    output[normalized_axis] = 0
    for shape, dtype in inputs:
        if dtype != first_dtype:
            raise ParameterPlanError(
                "v2 parameter expression concat dtypes must match; "
                f"dtype={first_dtype}/{dtype}"
            )
        if len(shape) != rank or any(
            shape[index] != first_shape[index]
            for index in range(rank)
            if index != normalized_axis
        ):
            raise ParameterPlanError(
                "v2 parameter expression concat shapes are incompatible"
            )
        output[normalized_axis] += shape[normalized_axis]
    return tuple(output), first_dtype


def _enforce_dag_bytes(*, dag_bytes: int, source_bytes: int) -> None:
    if dag_bytes > MAX_TOTAL_OUTPUT_BYTES:
        raise ParameterPlanError("parameter resource limit exceeded: dag_total_bytes")
    relative_limit = (
        source_bytes * MAX_STATE_AMPLIFICATION + MAX_STATE_OVERHEAD_BYTES
    )
    if dag_bytes > relative_limit:
        raise ParameterPlanError(
            "parameter resource limit exceeded: dag_state_amplification"
        )


def _v2_static_specs(
    normalized: Mapping[str, Any],
    source_specs: Mapping[str, tuple[tuple[int, ...], str]],
    *,
    source_bytes: int,
) -> tuple[
    dict[str, tuple[tuple[int, ...], str]],
    tuple[tuple[tuple[int, ...], str], ...],
]:
    node_specs: dict[str, tuple[tuple[int, ...], str]] = {}
    dag_bytes = 0
    for node in normalized["nodes"]:
        op = node["op"]
        if op == "source":
            spec = source_specs[node["source"]]
        elif op == "scalar":
            spec = ((), str(node["dtype"]))
        elif op in _BINARY_OPS:
            spec = _arithmetic_spec(
                op,
                node_specs[node["inputs"][0]],
                node_specs[node["inputs"][1]],
            )
        elif op == "sqrt":
            input_shape, input_dtype = node_specs[node["input"]]
            match = _PLAIN_DTYPE.fullmatch(input_dtype)
            kind = None if match is None else match.group("kind")
            if kind not in {"float", "complex"}:
                raise ParameterPlanError(
                    "v2 parameter expression sqrt requires floating dtype; "
                    f"dtype={input_dtype}"
                )
            spec = (input_shape, input_dtype)
        elif op == "concat":
            spec = _concat_spec(
                [node_specs[reference] for reference in node["inputs"]],
                int(node["axis"]),
            )
        elif op in _LAYOUT_OPS:
            input_shape, input_dtype = node_specs[node["input"]]
            spec = transformed_spec(input_shape, input_dtype, [node])
        else:  # normalize_plan makes this unreachable.
            raise ParameterPlanError(f"unsupported v2 parameter expression op: {op}")
        node_specs[node["id"]] = spec
        node_bytes = _tensor_nbytes(*spec, enforce_output_limit=True)
        if op != "source":
            dag_bytes += node_bytes
            _enforce_dag_bytes(dag_bytes=dag_bytes, source_bytes=source_bytes)

    output_specs: list[tuple[tuple[int, ...], str]] = []
    output_bytes = 0
    for binding in normalized["bindings"]:
        spec = node_specs[binding["value"]]
        output_bytes += _tensor_nbytes(*spec, enforce_output_limit=True)
        _enforce_aggregate_bytes(
            output_bytes=output_bytes,
            source_bytes=source_bytes,
        )
        output_specs.append(spec)
    return node_specs, tuple(output_specs)


def _validated_resource_plan(
    plan: Mapping[str, Any],
    source_specs: Mapping[str, tuple[Sequence[int], str]],
) -> tuple[
    dict[str, Any],
    tuple[tuple[tuple[int, ...], str], ...],
    int,
    dict[str, tuple[tuple[int, ...], str]],
]:
    """Validate aggregate output resources from shape/dtype facts only."""

    source_names = tuple(str(name) for name in source_specs)
    normalized = normalize_plan(plan, source_names)
    normalized_sources: dict[str, tuple[tuple[int, ...], str]] = {}
    source_bytes = 0
    for name in source_names:
        try:
            shape, dtype = source_specs[name]
        except (KeyError, TypeError, ValueError) as error:
            raise ParameterPlanError(
                "parameter resource limit exceeded: invalid_source_spec"
            ) from error
        normalized_shape = tuple(int(dim) for dim in shape)
        normalized_dtype = str(dtype)
        source_bytes += _tensor_nbytes(
            normalized_shape,
            normalized_dtype,
            enforce_output_limit=False,
        )
        normalized_sources[name] = (normalized_shape, normalized_dtype)

    if normalized["schema_version"] == EXPRESSION_SCHEMA_VERSION:
        node_specs, output_specs = _v2_static_specs(
            normalized,
            normalized_sources,
            source_bytes=source_bytes,
        )
        return normalized, output_specs, source_bytes, node_specs

    output_specs: list[tuple[tuple[int, ...], str]] = []
    output_bytes = 0
    for binding in normalized["bindings"]:
        source_shape, source_dtype = normalized_sources[binding["source"]]
        output_shape, output_dtype = transformed_spec(
            source_shape,
            source_dtype,
            binding["transforms"],
        )
        output_bytes += _tensor_nbytes(
            output_shape,
            output_dtype,
            enforce_output_limit=True,
        )
        _enforce_aggregate_bytes(
            output_bytes=output_bytes,
            source_bytes=source_bytes,
        )
        output_specs.append((output_shape, output_dtype))
    return normalized, tuple(output_specs), source_bytes, {}


def validate_plan_resources(
    plan: Mapping[str, Any],
    source_specs: Mapping[str, tuple[Sequence[int], str]],
) -> tuple[tuple[tuple[int, ...], str], ...]:
    """Return output specs after enforcing all fixed-state resource limits.

    This function is suitable for T0 because it consumes only public
    shape/dtype metadata.  It deliberately returns no tensor values or
    value-derived statistics.
    """

    _normalized, output_specs, _source_bytes, _node_specs = _validated_resource_plan(
        plan,
        source_specs,
    )
    return output_specs


def apply_transforms(array: Any, transforms: Sequence[Mapping[str, Any]]):
    """Apply a previously normalized plan to one trusted NumPy array."""

    import numpy as np

    value = np.asarray(array)
    expected_shape, expected_dtype = transformed_spec(
        value.shape, value.dtype.name, transforms
    )
    for transform in transforms:
        op = transform["op"]
        if op == "transpose":
            value = value.transpose(tuple(transform["axes"]))
        elif op == "reshape":
            value = value.reshape(tuple(transform["shape"]))
        elif op == "pad":
            value = np.pad(
                value,
                tuple(tuple(pair) for pair in transform["widths"]),
                mode="constant",
                constant_values=transform["value"],
            )
        elif op == "slice":
            value = value[
                tuple(
                    slice(start, stop, step)
                    for start, stop, step in zip(
                        transform["starts"], transform["stops"], transform["steps"]
                    )
                )
            ]
        elif op == "cast":
            value = value.astype(transform["dtype"], copy=False)
        elif op == "contiguous":
            value = _as_contiguous_preserving_rank(value)
    value = _as_contiguous_preserving_rank(value)
    if tuple(value.shape) != expected_shape or value.dtype.name != expected_dtype:
        raise ParameterPlanError("NumPy parameter preparation disagrees with static plan")
    return value


def _as_contiguous_preserving_rank(value: Any):
    import numpy as np

    array_value = np.asarray(value)
    if array_value.ndim == 0:
        # np.ascontiguousarray promotes a scalar to shape (1,), which would
        # silently disagree with Relax's rank-zero TensorStructInfo.
        return np.array(array_value, copy=True, order="C").reshape(())
    return np.ascontiguousarray(array_value)


def _require_materialized_spec(
    value: Any,
    expected: tuple[tuple[int, ...], str],
    *,
    category: str,
) -> None:
    actual_shape = tuple(value.shape)
    actual_dtype = value.dtype.name
    if (actual_shape, actual_dtype) != expected:
        raise ParameterPlanError(
            f"parameter materialization failed: {category}; "
            f"shape={actual_shape}, dtype={actual_dtype}"
        )


def _materialize_v2_values(
    normalized: Mapping[str, Any],
    state_arrays: Mapping[str, Any],
    expected_node_specs: Mapping[str, tuple[tuple[int, ...], str]],
    *,
    source_bytes: int,
) -> dict[str, Any]:
    """Evaluate a normalized v2 DAG without exposing any intermediate value."""

    import numpy as np

    values: dict[str, Any] = {}
    dag_bytes = 0
    for node in normalized["nodes"]:
        op = node["op"]
        try:
            if op == "source":
                value = state_arrays[node["source"]]
            elif op == "scalar":
                value = np.asarray(node["value"], dtype=node["dtype"]).reshape(())
            elif op in _BINARY_OPS:
                left = values[node["inputs"][0]]
                right = values[node["inputs"][1]]
                with np.errstate(all="ignore"):
                    if op == "add":
                        value = np.add(left, right)
                    elif op == "sub":
                        value = np.subtract(left, right)
                    elif op == "mul":
                        value = np.multiply(left, right)
                    else:
                        value = np.divide(left, right)
            elif op == "sqrt":
                with np.errstate(all="ignore"):
                    value = np.sqrt(values[node["input"]])
            elif op == "concat":
                value = np.concatenate(
                    [values[reference] for reference in node["inputs"]],
                    axis=int(node["axis"]),
                )
            elif op in _LAYOUT_OPS:
                value = apply_transforms(values[node["input"]], [node])
            else:  # normalize_plan makes this unreachable.
                raise ParameterPlanError("unsupported v2 parameter expression op")
        except ParameterPlanError:
            raise
        except (ArithmeticError, MemoryError, TypeError, ValueError) as error:
            # The exception is deliberately replaced rather than interpolated:
            # NumPy error text can include hidden values for malformed inputs.
            raise ParameterPlanError(
                f"parameter materialization failed: expression_{op}"
            ) from error
        if op != "source":
            value = _as_contiguous_preserving_rank(value)
        expected = expected_node_specs[node["id"]]
        _require_materialized_spec(value, expected, category=f"expression_{op}")
        values[node["id"]] = value
        if op != "source":
            dag_bytes += _tensor_nbytes(*expected, enforce_output_limit=True)
            _enforce_dag_bytes(dag_bytes=dag_bytes, source_bytes=source_bytes)
    return values


def materialize_arguments(
    *,
    inputs: Mapping[str, Any],
    dynamic_input_names: Sequence[str],
    source_state_names: Sequence[str],
    plan: Mapping[str, Any],
) -> list[Any]:
    """Build the positional candidate ABI from trusted original task arrays."""

    import numpy as np

    sources = tuple(str(name) for name in source_state_names)
    state_arrays: dict[str, Any] = {}
    for name in sources:
        if name not in inputs:
            raise ParameterPlanError(f"missing fixed state {name!r}")
        state_arrays[name] = np.asarray(inputs[name])
    source_specs = {
        name: (tuple(value.shape), value.dtype.name)
        for name, value in state_arrays.items()
    }
    # Re-run the complete static gate immediately before materialization.  No
    # transformed array has been allocated at this point.
    normalized, expected_specs, source_bytes, expected_node_specs = _validated_resource_plan(
        plan,
        source_specs,
    )
    arguments = []
    for name in dynamic_input_names:
        if name not in inputs:
            raise ParameterPlanError(f"missing dynamic input {name!r}")
        arguments.append(inputs[name])

    if normalized["schema_version"] == EXPRESSION_SCHEMA_VERSION:
        values = _materialize_v2_values(
            normalized,
            state_arrays,
            expected_node_specs,
            source_bytes=source_bytes,
        )
        output_bytes = 0
        for binding, expected in zip(normalized["bindings"], expected_specs):
            value = _as_contiguous_preserving_rank(values[binding["value"]])
            _require_materialized_spec(value, expected, category="binding")
            output_bytes += _tensor_nbytes(*expected, enforce_output_limit=True)
            _enforce_aggregate_bytes(
                output_bytes=output_bytes,
                source_bytes=source_bytes,
            )
            arguments.append(value)
        return arguments

    output_bytes = 0
    for binding, (expected_shape, expected_dtype) in zip(
        normalized["bindings"], expected_specs
    ):
        source = binding["source"]
        value = apply_transforms(state_arrays[source], binding["transforms"])
        actual_shape = tuple(value.shape)
        actual_dtype = value.dtype.name
        if actual_shape != expected_shape or actual_dtype != expected_dtype:
            raise ParameterPlanError(
                "materialized parameter differs from its static spec; "
                f"shape={actual_shape}, dtype={actual_dtype}"
            )
        output_bytes += _tensor_nbytes(
            actual_shape,
            actual_dtype,
            enforce_output_limit=True,
        )
        _enforce_aggregate_bytes(
            output_bytes=output_bytes,
            source_bytes=source_bytes,
        )
        arguments.append(value)
    return arguments


def canonical_plan_json(plan: Mapping[str, Any]) -> str:
    return json.dumps(
        plan, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


__all__ = [
    "ATTR_NAME",
    "EXPRESSION_SCHEMA_VERSION",
    "MAX_NODES",
    "MAX_SINGLE_OUTPUT_BYTES",
    "MAX_STATE_AMPLIFICATION",
    "MAX_STATE_OVERHEAD_BYTES",
    "MAX_TOTAL_OUTPUT_BYTES",
    "SCHEMA_VERSION",
    "ParameterPlanError",
    "apply_transforms",
    "canonical_plan_json",
    "default_plan",
    "materialize_arguments",
    "normalize_plan",
    "plan_from_main",
    "schema_document",
    "transformed_spec",
    "validate_plan_resources",
]
