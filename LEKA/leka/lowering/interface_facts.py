"""Bounded, non-executing interface projections of evaluator-owned sources.

This describes static source interfaces, never future lowering, instruction
selection, legality of a proposed edit, or measured hardware behavior.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re

from .abi_contract import signatures

SCHEMA = "leka.interface_facts.v1"
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_FUNCTIONS = 64
MAX_PARAMETERS = 32
MAX_AXES = 16
MAX_FACTS_BYTES = 64 * 1024


def _integer(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value if -(2 ** 63) <= node.value < 2 ** 63 else None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "T"
            and node.func.attr in {"int32", "int64", "uint32", "uint64"}
            and len(node.args) == 1 and not node.keywords):
        return _integer(node.args[0])
    return None


def _integers(node):
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) > MAX_AXES:
        return None
    values = [_integer(value) for value in node.elts]
    return values if all(value is not None for value in values) else None


def _buffer(annotation):
    if not (isinstance(annotation, ast.Call) and isinstance(annotation.func, ast.Attribute)
            and isinstance(annotation.func.value, ast.Name) and annotation.func.value.id == "T"
            and annotation.func.attr == "Buffer"):
        return {"status": "unknown", "reason": "not_a_supported_T_Buffer_annotation"}
    kwargs = {item.arg: item.value for item in annotation.keywords}
    shape = _integers(annotation.args[0] if annotation.args else kwargs.get("shape"))
    dtype_node = annotation.args[1] if len(annotation.args) > 1 else kwargs.get("dtype")
    dtype = dtype_node.value if isinstance(dtype_node, ast.Constant) and isinstance(dtype_node.value, str) else None
    # Optional positional parameters have version-dependent meanings. Do not
    # relabel them as an implicit compact layout.
    supported_options = len(annotation.args) <= 2 and None not in kwargs
    axis_node = kwargs.get("axis_separators")
    ordinary_axes = axis_node is None or (isinstance(axis_node, (ast.Tuple, ast.List)) and not axis_node.elts)
    buffer_type = kwargs.get("buffer_type")
    ordinary_type = buffer_type is None or (isinstance(buffer_type, ast.Constant) and buffer_type.value == "default")
    ordinary_axes = ordinary_axes and ordinary_type
    strides, layout = None, "unknown"
    if supported_options and "strides" in kwargs:
        strides = _integers(kwargs["strides"])
        if strides is not None and shape is not None and len(strides) == len(shape) and ordinary_axes:
            layout = "explicit_static_strides"
        else:
            strides = None
    elif supported_options and ordinary_axes and shape is not None and all(dim >= 0 for dim in shape):
        strides, stride = [], 1
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        layout = "implicit_compact_T_Buffer"
    offset = _integer(kwargs["elem_offset"]) if "elem_offset" in kwargs else (0 if supported_options else None)
    return {"status": "static_source_annotation", "dtype": dtype,
            "shape": shape, "layout": layout,
            "axes": ([{"axis": index, "extent": extent,
                       "stride_elements": strides[index] if strides is not None else None}
                      for index, extent in enumerate(shape)] if shape is not None else None),
            "elem_offset_elements": offset,
            "declared_data_alignment_bytes": _integer(kwargs.get("data_alignment")),
            "unknown_fields": [name for name, value in (("shape", shape), ("dtype", dtype),
                                                        ("strides", strides), ("elem_offset", offset)) if value is None]}


def tir_interfaces(source: str) -> dict:
    """Recognize a conservative TVM Script subset without importing the module."""
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return {"status": "unknown", "reason": "source_did_not_parse"}
    # This exact alias is the shipped canonical TVM Script spelling. Other
    # spellings remain unknown instead of evaluating imports or globals.
    alias_declarations = [(node, alias) for node in ast.walk(module) if isinstance(node, ast.ImportFrom)
                          for alias in node.names if node.module == "tvm.script"
                          and alias.name == "tir" and alias.asname == "T"]
    if len(alias_declarations) != 1 or alias_declarations[0][0] not in module.body:
        return {"status": "unknown", "reason": "unsupported_tir_alias"}
    for node in ast.walk(module):
        if (isinstance(node, ast.Name) and node.id == "T" and isinstance(node.ctx, (ast.Store, ast.Del))
                or isinstance(node, ast.arg) and node.arg == "T"
                or isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "T"
                or isinstance(node, ast.ExceptHandler) and node.name == "T"):
            return {"status": "unknown", "reason": "tir_alias_is_shadowed"}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    return {"status": "unknown", "reason": "wildcard_import_makes_alias_unknown"}
                bound_name = alias.asname or alias.name.split(".")[0]
                if bound_name == "T" and (node, alias) != alias_declarations[0]:
                    return {"status": "unknown", "reason": "tir_alias_is_shadowed"}
    functions = []
    module_classes = [node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Module"]
    if len(module_classes) > 1 or any(isinstance(node, ast.Name) and node.id == "Module"
                                    and isinstance(node.ctx, (ast.Store, ast.Del)) for node in ast.walk(module)):
        return {"status": "unknown", "reason": "module_definition_is_ambiguous"}
    candidates = module_classes[0].body if len(module_classes) == 1 else module.body
    nodes = [node for node in candidates if isinstance(node, ast.FunctionDef)
             and any(isinstance(decorator.func if isinstance(decorator, ast.Call) else decorator, ast.Attribute)
                     and isinstance((decorator.func if isinstance(decorator, ast.Call) else decorator).value, ast.Name)
                     and (decorator.func if isinstance(decorator, ast.Call) else decorator).value.id == "T"
                     and (decorator.func if isinstance(decorator, ast.Call) else decorator).attr == "prim_func"
                     for decorator in node.decorator_list)]
    if not nodes:
        return {"status": "unknown", "reason": "no_supported_primfunc_definitions"}
    for node in nodes[:MAX_FUNCTIONS]:
        parameters = [*node.args.posonlyargs, *node.args.args]
        functions.append({"source_function_name": node.name, "source_line": node.lineno,
                          "parameter_count": len(parameters),
                          "parameters": [{"index": index, "name": parameter.arg,
                                          "buffer": _buffer(parameter.annotation)}
                                         for index, parameter in enumerate(parameters[:MAX_PARAMETERS])],
                          "parameters_truncated": len(parameters) > MAX_PARAMETERS})
    return {"status": "static_source_interface", "primfunc_definition_count": len(nodes),
            "functions": functions, "functions_truncated": len(nodes) > MAX_FUNCTIONS,
            "interpretation": "definition_count_is_not_runtime_kernel_launch_count"}


def _split_parameters(text):
    depth, start, parts = 0, 0, []
    for index, char in enumerate(text):
        if char in "([{<":
            depth += 1
        elif char in ")]}>":
            depth -= 1
            if depth < 0:
                return None
        elif char == "," and not depth:
            parts.append(text[start:index].strip())
            start = index + 1
    if depth:
        return None
    parts.append(text[start:].strip())
    return [] if parts == [""] or parts == ["void"] else parts


def native_interfaces(language: str, source: str) -> dict:
    """Expose exactly the existing freeze checker's supported signature surface."""
    if language == "llvmir":
        declaration_lines = [line for line in source.splitlines()
                             if re.match(r"^\s*define\s", line) and "{" in line]
    elif language in {"cuda", "opencl"}:
        # Compiler-emitted flat declarations are enough for the projection.
        # Comments, string snippets, prototypes and multiline forms must not
        # acquire the status of a native function definition.
        clean = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
        declaration_lines = [line.split("//", 1)[0] for line in clean.splitlines()
                             if re.match(r'^\s*(?:extern\s+"C"\s+)?(?:__global__|__kernel)\s', line)
                             and "{" in line]
    else:
        return {"status": "unknown", "reason": "unsupported_native_language"}
    try:
        found = signatures(language, "\n".join(declaration_lines))
    except (KeyError, TypeError):
        return {"status": "unknown", "reason": "unsupported_native_language"}
    if not found:
        return {"status": "unknown", "reason": "no_supported_native_signatures"}
    functions = []
    for symbol, raw in list(found.items())[:MAX_FUNCTIONS]:
        parts = _split_parameters(raw)
        # The legacy freeze regex stops at the first ')'. Nested attributes
        # such as byval(...) must not be presented as a verified full ABI.
        if parts is None or len(raw) > 4096:
            functions.append({"symbol": symbol, "status": "unknown",
                              "reason": "signature_outside_supported_flat_surface"})
            continue
        functions.append({"symbol": symbol, "status": "frozen_parameter_surface",
                          "parameter_count": len(parts),
                          "parameters": [{"index": index, "declaration": parameter}
                                         for index, parameter in enumerate(parts[:MAX_PARAMETERS])],
                          "parameters_truncated": len(parts) > MAX_PARAMETERS})
    return {"status": "static_source_interface", "language": language,
            "supported_definition_count": len(found), "functions": functions,
            "functions_truncated": len(found) > MAX_FUNCTIONS,
            "interpretation": "parameter_surface_only_not_full_ABI_or_instruction_semantics"}


def build_interface_facts(*, layer: str, backend: str, binding_payload: dict,
                          sources: dict[str, str | None], native_language: str | None = None) -> dict:
    artifacts = []
    for filename, source in sources.items():
        if not isinstance(source, str):
            artifacts.append({"artifact": filename, "status": "unknown", "reason": "source_unavailable"})
            continue
        encoded = source.encode("utf-8")
        item = {"artifact": filename, "sha256": hashlib.sha256(encoded).hexdigest()}
        slot = {"canonical_tir.py": "T01", "scheduled_tir.py": "L1"}.get(filename, "T12")
        expected = (binding_payload.get("projected_inputs") or {}).get(slot)
        item["projected_slot"] = slot
        item["matches_projected_input"] = item["sha256"] == expected if expected is not None else None
        if layer == "L2" and filename.endswith(".py"):
            item["visibility"] = "upstream_L1_interface_summary_only_source_body_not_projected"
        if item["matches_projected_input"] is not True:
            item.update(status="unknown", reason="source_binding_missing_or_mismatched")
        elif len(encoded) > MAX_SOURCE_BYTES:
            item.update(status="unknown", reason="source_exceeds_bounded_parser_limit")
        else:
            item.update(tir_interfaces(source) if filename.endswith(".py") else native_interfaces(native_language, source))
        artifacts.append(item)
    facts = {"schema_version": SCHEMA, "backend": backend, "recipient_layer": layer,
             "binding": json.loads(json.dumps(binding_payload.get("binding"))),
             "projected_inputs": json.loads(json.dumps(binding_payload.get("projected_inputs"))),
             "artifacts": artifacts,
             "authority": "evaluator_static_source_projection_not_candidate_approval",
             "validation_tiers": {"T0": "structural_check", "T1": "device_correctness_and_paired_candidate_incumbent_timing"},
             "downstream_regeneration_owner": "evaluator",
             "unknown_without_separate_evidence": ["future_scratch_placement", "register_promotion",
                                                   "runtime_kernel_launch_count", "performance_gain"],
             "limits": "Unparsed fields are unknown. Match source hashes and binding before use. Source annotations and frozen parameter text do not prove a proposed edit legal or a hardware bottleneck causal."}
    if len(json.dumps(facts).encode()) > MAX_FACTS_BYTES:
        facts["artifacts"] = [{"artifact": item["artifact"], "sha256": item.get("sha256"),
                               "status": "unknown", "reason": "fact_projection_exceeds_size_limit"}
                              for item in artifacts]
    facts["facts_id"] = hashlib.sha256(json.dumps(facts, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return facts
