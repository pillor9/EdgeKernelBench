"""Deterministic, value-free PyTorch semantics projected for the L0 agent.

The projection deliberately contains *structure*, never model state or example
tensor contents.  It is built from the already trusted ``nn.Module`` and the
pre/post-decomposition ``ExportedProgram`` objects used by :mod:`pytorch_task`.
No task source, ``get_inputs`` implementation, stack trace, tensor statistic,
or RNG seed is copied into the agent-visible contract.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import enum
import hashlib
import inspect
import json
import math
from typing import Any


SCHEMA_VERSION = "leka.pytorch_semantics.v1"
MAX_GRAPH_NODES = 4096
MAX_STRING_CHARS = 240


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return the unique JSON representation used for semantic identities."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def semantics_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def defensive_json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Detach callers from the trusted in-memory projection."""

    return json.loads(canonical_json(value))


class _Projector:
    def __init__(self, *, task_module_prefix: str) -> None:
        self.task_module_prefix = task_module_prefix
        self.symbols: dict[str, str] = {}
        self.complete = True

    def text(self, value: str) -> str:
        # Control characters are valid JSON but add no operator semantics and can
        # turn a data file into a poor instruction surface for an LLM.
        cleaned = "".join(character if character >= " " else " " for character in value)
        return cleaned[:MAX_STRING_CHARS]

    def type_name(self, value: Any) -> str:
        cls = value if inspect.isclass(value) else type(value)
        module = str(getattr(cls, "__module__", ""))
        qualname = str(getattr(cls, "__qualname__", getattr(cls, "__name__", "unknown")))
        if module.startswith(self.task_module_prefix):
            module = "<task>"
        return self.text(f"{module}.{qualname}" if module else qualname)

    def dimension(self, value: Any) -> int | str:
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
        raw = str(value)
        if raw not in self.symbols:
            self.symbols[raw] = f"sym{len(self.symbols)}"
        return self.symbols[raw]

    def tensor_spec(self, value: Any) -> dict[str, Any]:
        shape = getattr(value, "shape", ())
        dtype = str(getattr(value, "dtype", "unknown")).removeprefix("torch.")
        spec: dict[str, Any] = {
            "kind": "tensor",
            "shape": [self.dimension(item) for item in tuple(shape)],
            "dtype": self.text(dtype),
        }
        stride = getattr(value, "stride", None)
        if callable(stride):
            try:
                stride = stride()
            except (RuntimeError, TypeError):
                stride = None
        if isinstance(stride, Sequence) and not isinstance(stride, (str, bytes, bytearray)):
            spec["stride"] = [self.dimension(item) for item in tuple(stride)]
        dimensions = spec["shape"]
        if all(isinstance(item, int) for item in dimensions):
            numel = math.prod(dimensions)
            element_size = getattr(value, "element_size", None)
            if callable(element_size):
                try:
                    bytes_per_element = int(element_size())
                except (RuntimeError, TypeError, ValueError):
                    bytes_per_element = None
            else:
                bytes_per_element = None
            spec["numel"] = int(numel)
            if bytes_per_element is not None:
                spec["nbytes"] = int(numel) * bytes_per_element
        return spec

    def argument(self, value: Any, *, depth: int = 0) -> Any:
        import torch

        if depth > 12:
            self.complete = False
            return {"kind": "depth_bounded", "type": self.type_name(value)}
        if value is None or isinstance(value, (bool, int, str)):
            return self.text(value) if isinstance(value, str) else value
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return {"kind": "infinity", "sign": 1 if value > 0 else -1}
        if isinstance(value, torch.fx.Node):
            return {"node": self.text(value.name)}
        if isinstance(value, torch.Tensor):
            # This branch intentionally does not read, hash, summarize, or copy
            # tensor contents.
            return self.tensor_spec(value)
        if isinstance(value, torch.Size):
            return [self.dimension(item) for item in value]
        sym_types = tuple(
            candidate
            for candidate in (
                getattr(torch, "SymInt", None),
                getattr(torch, "SymFloat", None),
                getattr(torch, "SymBool", None),
            )
            if isinstance(candidate, type)
        )
        if sym_types and isinstance(value, sym_types):
            return {"kind": self.type_name(value), "symbol": self.dimension(value)}
        if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
            return self.text(str(value).removeprefix("torch."))
        if isinstance(value, slice):
            return {
                "kind": "slice",
                "start": self.argument(value.start, depth=depth + 1),
                "stop": self.argument(value.stop, depth=depth + 1),
                "step": self.argument(value.step, depth=depth + 1),
            }
        if isinstance(value, enum.Enum):
            return self.text(f"{self.type_name(value)}.{value.name}")
        if isinstance(value, Mapping):
            return {
                self.text(str(key)): self.argument(value[key], depth=depth + 1)
                for key in sorted(value, key=lambda item: str(item))
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self.argument(item, depth=depth + 1) for item in value]
        self.complete = False
        return {"kind": "unsupported", "type": self.type_name(value)}

    def output_spec(self, value: Any, *, depth: int = 0) -> Any:
        import torch

        if depth > 12:
            self.complete = False
            return {"kind": "depth_bounded"}
        if isinstance(value, torch.Tensor) or (
            hasattr(value, "shape") and hasattr(value, "dtype")
        ):
            return self.tensor_spec(value)
        if isinstance(value, tuple):
            return {
                "kind": "tuple",
                "items": [self.output_spec(item, depth=depth + 1) for item in value],
            }
        if isinstance(value, list):
            return {
                "kind": "list",
                "items": [self.output_spec(item, depth=depth + 1) for item in value],
            }
        if value is None:
            return {"kind": "none"}
        if isinstance(value, (bool, int, float, str)):
            return {"kind": "literal", "literal": self.argument(value)}
        return self.argument(value, depth=depth + 1)

    def target_name(self, target: Any) -> str:
        if isinstance(target, str):
            return self.text(target)
        schema = getattr(target, "_schema", None)
        if schema is not None:
            name = str(getattr(schema, "name", "")).replace("::", ".")
            overload = str(getattr(schema, "overload_name", ""))
            if name:
                return self.text(name + (f".{overload}" if overload else ".default"))
        module = str(getattr(target, "__module__", ""))
        qualname = str(getattr(target, "__qualname__", getattr(target, "__name__", "")))
        if module or qualname:
            return self.text(f"{module}.{qualname}".strip("."))
        self.complete = False
        return self.type_name(target)

    def node_arguments(self, node: Any) -> dict[str, Any]:
        schema = getattr(node.target, "_schema", None)
        schema_arguments = list(getattr(schema, "arguments", ())) if schema is not None else []
        positional: dict[str, Any] = {}
        for index, argument in enumerate(node.args):
            name = (
                str(getattr(schema_arguments[index], "name", f"arg{index}"))
                if index < len(schema_arguments)
                else f"arg{index}"
            )
            positional[self.text(name)] = self.argument(argument)
        return {
            "positional": positional,
            "keywords": {
                self.text(str(key)): self.argument(node.kwargs[key])
                for key in sorted(node.kwargs)
            },
        }

    def module_stack(self, node: Any) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        stack = node.meta.get("nn_module_stack") or {}
        if not isinstance(stack, Mapping):
            return result
        for raw in stack.values():
            if not isinstance(raw, Sequence) or len(raw) < 2:
                continue
            path, module_type = raw[0], raw[1]
            path = str(path)
            if path == "L['self']":
                path = ""
            elif path.startswith("L['self']."):
                path = path[len("L['self'].") :]
            type_name = module_type if isinstance(module_type, str) else self.type_name(module_type)
            if str(type_name).startswith(self.task_module_prefix):
                type_name = "<task>." + str(type_name).rsplit(".", 1)[-1]
            result.append({"path": self.text(path), "type": self.text(str(type_name))})
        return result

    def graph(self, exported_program: Any, *, stage: str) -> dict[str, Any]:
        nodes = list(exported_program.graph.nodes)
        kept = nodes[:MAX_GRAPH_NODES]
        graph_nodes: list[dict[str, Any]] = []
        histogram: Counter[str] = Counter()
        was_complete = self.complete
        for node in kept:
            operation = self.target_name(node.target)
            histogram[operation] += 1
            meta = node.meta.get("val", node.meta.get("tensor_meta"))
            graph_nodes.append(
                {
                    "id": self.text(node.name),
                    "kind": self.text(node.op),
                    "op": operation,
                    "arguments": self.node_arguments(node),
                    "module_stack": self.module_stack(node),
                    "output": self.output_spec(meta) if meta is not None else None,
                    "users": sorted(self.text(user.name) for user in node.users),
                }
            )
        if len(kept) != len(nodes):
            self.complete = False
        payload = {
            "stage": stage,
            "dialect": "functional_aten",
            "node_count": len(nodes),
            "omitted_node_count": len(nodes) - len(kept),
            "complete": was_complete and self.complete and len(kept) == len(nodes),
            "nodes": graph_nodes,
            "op_histogram": dict(sorted(histogram.items())),
            "range_constraints": _range_constraints(exported_program, self),
        }
        payload["sha256"] = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return payload


def _range_constraints(exported_program: Any, projector: _Projector) -> list[dict[str, Any]]:
    constraints = getattr(exported_program, "range_constraints", {}) or {}
    result = []
    for symbol, bounds in sorted(constraints.items(), key=lambda item: str(item[0])):
        lower = getattr(bounds, "lower", None)
        upper = getattr(bounds, "upper", None)
        result.append(
            {
                "symbol": projector.dimension(symbol),
                "lower": projector.argument(lower),
                "upper": projector.argument(upper),
            }
        )
    return result


def _module_attribute_names(module: Any) -> tuple[str, ...]:
    import torch.nn as nn

    names: list[str] = []
    if isinstance(
        module,
        (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d),
    ):
        names += [
            "in_channels", "out_channels", "kernel_size", "stride", "padding",
            "dilation", "groups", "padding_mode", "output_padding", "transposed",
        ]
    if isinstance(module, nn.Linear):
        names += ["in_features", "out_features"]
    if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
        names += [
            "num_embeddings", "embedding_dim", "padding_idx", "max_norm", "norm_type",
            "scale_grad_by_freq", "sparse", "mode", "include_last_offset",
        ]
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        names += ["num_features", "eps", "momentum", "affine", "track_running_stats"]
    if isinstance(module, nn.LayerNorm):
        names += ["normalized_shape", "eps", "elementwise_affine"]
    if isinstance(module, nn.GroupNorm):
        names += ["num_groups", "num_channels", "eps", "affine"]
    pooling_types = tuple(
        candidate
        for candidate in (
            nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
            nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
        )
        if isinstance(candidate, type)
    )
    if isinstance(module, pooling_types):
        names += [
            "kernel_size", "stride", "padding", "dilation", "ceil_mode",
            "count_include_pad", "divisor_override", "return_indices",
        ]
    adaptive_types = tuple(
        candidate
        for candidate in (
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
            nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
        )
        if isinstance(candidate, type)
    )
    if isinstance(module, adaptive_types):
        names += ["output_size", "return_indices"]
    if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
        names += ["p", "inplace"]
    activation_names = {
        nn.ReLU: ("inplace",),
        nn.LeakyReLU: ("negative_slope", "inplace"),
        nn.GELU: ("approximate",),
        nn.ELU: ("alpha", "inplace"),
        nn.Softmax: ("dim",),
        nn.LogSoftmax: ("dim",),
        nn.Hardtanh: ("min_val", "max_val", "inplace"),
    }
    for module_type, attributes in activation_names.items():
        if isinstance(module, module_type):
            names += list(attributes)
    if isinstance(module, nn.Flatten):
        names += ["start_dim", "end_dim"]
    if isinstance(module, nn.Upsample):
        names += ["size", "scale_factor", "mode", "align_corners", "recompute_scale_factor"]
    if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
        names += [
            "input_size", "hidden_size", "num_layers", "bias", "batch_first",
            "dropout", "bidirectional", "proj_size", "nonlinearity",
        ]
    if isinstance(module, nn.MultiheadAttention):
        names += [
            "embed_dim", "num_heads", "dropout", "batch_first", "head_dim",
            "kdim", "vdim", "add_zero_attn",
        ]
    # Keep first occurrence so subclasses matching multiple categories remain stable.
    return tuple(dict.fromkeys(names))


def _module_tree(model: Any, projector: _Projector) -> list[dict[str, Any]]:
    try:
        modules = list(model.named_modules(remove_duplicate=False))
    except TypeError:  # older supported PyTorch releases
        modules = list(model.named_modules())
    first_path: dict[int, str] = {}
    result: list[dict[str, Any]] = []
    for path, module in modules:
        identity = id(module)
        alias_of = first_path.get(identity)
        first_path.setdefault(identity, path)
        attributes = {}
        raw_attributes = vars(module)
        for name in _module_attribute_names(module):
            if name in raw_attributes:
                attributes[name] = projector.argument(raw_attributes[name])
        parameters = vars(module).get("_parameters", {}) or {}
        buffers = vars(module).get("_buffers", {}) or {}
        if "bias" in parameters:
            attributes["bias_present"] = parameters["bias"] is not None
        result.append(
            {
                "path": projector.text(path),
                "parent": projector.text(path.rsplit(".", 1)[0]) if path else None,
                "type": projector.type_name(module),
                "alias_of": projector.text(alias_of) if alias_of is not None else None,
                "attributes": attributes,
                "direct_parameters": sorted(projector.text(str(name)) for name in parameters),
                "direct_buffers": sorted(projector.text(str(name)) for name in buffers),
            }
        )
    return result


def _init_arguments(model: Any, init_args: Sequence[Any], projector: _Projector) -> list[dict[str, Any]]:
    names: list[str] = []
    try:
        signature = inspect.signature(type(model).__init__)
        bound = signature.bind_partial(None, *init_args)
        names = [name for name in bound.arguments if name != "self"]
    except (TypeError, ValueError):
        names = []
    return [
        {
            "name": projector.text(names[index] if index < len(names) else f"arg{index}"),
            "argument": projector.argument(argument),
        }
        for index, argument in enumerate(init_args)
    ]


def _input_tensor(exported_program: Any, spec: Any, projector: _Projector) -> Any | None:
    kind = str(getattr(spec.kind, "name", spec.kind)).lower()
    target = spec.target
    if kind in {"parameter", "buffer"} and isinstance(target, str):
        if target in exported_program.state_dict:
            return exported_program.state_dict[target]
        constants = getattr(exported_program, "constants", {}) or {}
        return constants.get(target)
    if kind == "constant_tensor" and isinstance(target, str):
        constants = getattr(exported_program, "constants", None)
        if constants is None:
            constants = getattr(exported_program, "tensor_constants", {})
        return (constants or {}).get(target)
    graph_name = str(getattr(spec.arg, "name", ""))
    for node in exported_program.graph.nodes:
        if node.op == "placeholder" and node.name == graph_name:
            return node.meta.get("val", node.meta.get("tensor_meta"))
    return None


def _semantic_role(kind: str, source_name: str | None) -> str:
    local_name = source_name.rsplit(".", 1)[-1] if source_name else ""
    if local_name in {"weight", "bias", "running_mean", "running_var", "num_batches_tracked"}:
        return local_name
    return kind


def _abi_inputs(
    exported_program: Any,
    *,
    relax_names: Sequence[str],
    projector: _Projector,
) -> list[dict[str, Any]]:
    relax_position = {name: index for index, name in enumerate(relax_names)}
    result = []
    for position, spec in enumerate(exported_program.graph_signature.input_specs):
        graph_name = projector.text(str(getattr(spec.arg, "name", f"input{position}")))
        kind = projector.text(str(getattr(spec.kind, "name", spec.kind)).lower())
        source_name = projector.text(spec.target) if isinstance(spec.target, str) else None
        tensor = _input_tensor(exported_program, spec, projector)
        tensor_info = projector.tensor_spec(tensor) if tensor is not None and hasattr(tensor, "shape") else None
        local_name = source_name.rsplit(".", 1)[-1] if source_name else None
        module_path = source_name.rsplit(".", 1)[0] if source_name and "." in source_name else ""
        entry = {
            "export_position": position,
            "graph_name": graph_name,
            "kind": kind,
            "source_name": source_name,
            "module_path": projector.text(module_path) if source_name else None,
            "local_name": projector.text(local_name) if local_name else None,
            "semantic_role": _semantic_role(kind, source_name),
            "persistent": getattr(spec, "persistent", None),
            "relax_name": graph_name if graph_name in relax_position else None,
            "relax_position": relax_position.get(graph_name),
            "relax_binding": "runtime_input" if graph_name in relax_position else "bound_by_importer",
            "tensor": tensor_info,
            "requires_grad": bool(getattr(tensor, "requires_grad", False)) if tensor is not None else None,
            "contents": "hidden" if tensor_info is not None else None,
        }
        argument_type = projector.type_name(spec.arg)
        entry["argument_type"] = argument_type
        literal = getattr(spec.arg, "value", None)
        if tensor_info is None and argument_type.endswith("ConstantArgument"):
            entry["literal"] = projector.argument(literal)
        result.append(entry)
    return result


def _abi_outputs(
    exported_program: Any,
    *,
    public_output_specs: Sequence[Mapping[str, Any]],
    projector: _Projector,
) -> list[dict[str, Any]]:
    user_index = 0
    result = []
    for position, spec in enumerate(exported_program.graph_signature.output_specs):
        kind = projector.text(str(getattr(spec.kind, "name", spec.kind)).lower())
        entry: dict[str, Any] = {
            "export_position": position,
            "graph_name": projector.text(str(getattr(spec.arg, "name", f"output{position}"))),
            "kind": kind,
            "target": projector.text(spec.target) if isinstance(spec.target, str) else None,
        }
        if kind == "user_output" and user_index < len(public_output_specs):
            entry["public_output"] = projector.argument(dict(public_output_specs[user_index]))
            user_index += 1
        result.append(entry)
    return result


def build_pytorch_semantics(
    *,
    task_id: str,
    task_source_sha256: str,
    model: Any,
    init_args: Sequence[Any],
    pre_decomposition_exported_program: Any,
    post_decomposition_exported_program: Any,
    relax_names: Sequence[str],
    num_dynamic: int,
    public_output_specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a closed, deterministic, value-free semantic projection."""

    import torch

    task_prefix = str(type(model).__module__).rsplit(".", 1)[0]
    projector = _Projector(task_module_prefix=task_prefix)
    pre_graph = projector.graph(pre_decomposition_exported_program, stage="pre_decomposition")
    post_graph = projector.graph(post_decomposition_exported_program, stage="post_decomposition")
    inputs = _abi_inputs(
        post_decomposition_exported_program,
        relax_names=relax_names,
        projector=projector,
    )
    relax_runtime_names = [
        entry["graph_name"]
        for entry in sorted(
            (entry for entry in inputs if entry["relax_name"]),
            key=lambda entry: int(entry["relax_position"]),
        )
    ]
    if list(relax_names) != relax_runtime_names:
        raise ValueError(
            "PyTorch export inputs do not match the imported Relax ABI: "
            f"{relax_runtime_names!r} != {list(relax_names)!r}"
        )
    projection = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "task_id": str(task_id)[:120],
            "task_source_sha256": str(task_source_sha256),
        },
        "capture": {
            "execution_mode": "eval",
            "torch_version": str(torch.__version__)[:80],
            "strict_export": False,
            "shape_policy": "static_example_shapes",
            "semantic_graph": "pre_decomposition_functional_aten",
            "import_graph": "post_run_decompositions_functional_aten",
        },
        "model": {
            "root_type": projector.type_name(model),
            "init_arguments": _init_arguments(model, init_args, projector),
            "modules": _module_tree(model, projector),
        },
        "abi": {
            "entry": "main",
            "num_dynamic_inputs": int(num_dynamic),
            "relax_input_order": list(relax_names),
            "state_policy": "fixed_hidden_runtime_inputs",
            "inputs": inputs,
            "outputs": _abi_outputs(
                post_decomposition_exported_program,
                public_output_specs=public_output_specs,
                projector=projector,
            ),
        },
        "graphs": {
            "pre_decomposition": pre_graph,
            "post_decomposition": post_graph,
        },
        "semantic_obligations": [
            "preserve the exact Relax main input order, shape, dtype, and output tree",
            "preserve eval-mode PyTorch mathematics for every hidden correctness input",
            "treat every parameter, buffer, and tensor constant marked hidden as an opaque runtime quantity",
            "preserve every runtime state input even when its original PyTorch role was buffer or constant_tensor",
            "do not specialize to example tensor contents or move dynamic-input work outside main",
        ],
        "bounds": {
            "max_graph_nodes_per_stage": MAX_GRAPH_NODES,
            "max_string_chars": MAX_STRING_CHARS,
            "complete": bool(projector.complete and pre_graph["complete"] and post_graph["complete"]),
        },
    }
    # Canonicalization is also the final JSON-type and finite-number check.
    return defensive_json_copy(projection)


__all__ = [
    "SCHEMA_VERSION",
    "build_pytorch_semantics",
    "canonical_json",
    "defensive_json_copy",
    "semantics_sha256",
]
