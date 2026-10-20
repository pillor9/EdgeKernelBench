"""Validate generated IRModules and entry signatures with TVM."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from pipeline.interfaces import BuildResult, BuildTask, JSONValue, TensorSpec

from .errors import FaultRecord, FaultStage, PipelineFault, diagnostic_digest


CodegenKind = Literal[
    "tvm_native_codegen", "tvm_external_library_call", "embedded_backend_source"
]

TRUSTED_REPLAY_BACKENDS = frozenset(
    {"vivo_cpu", "vivo_gpu", "jetson_gpu", "rk3588_gpu"}
)
GPU_REPLAY_BACKENDS = frozenset({"vivo_gpu", "jetson_gpu", "rk3588_gpu"})


def validate_trusted_replay_structure(
    mod: Any,
    *,
    backend: str,
) -> dict[str, JSONValue]:
    """Allow Relax-only baseline views and require bindings on GPU PrimFuncs."""

    import tvm

    if backend not in TRUSTED_REPLAY_BACKENDS:
        raise ValueError(f"unsupported trusted replay backend: {backend!r}")
    if not isinstance(mod, tvm.IRModule):
        raise TypeError("trusted replay must be a tvm.IRModule")

    primfuncs: list[dict[str, JSONValue]] = []
    for global_var, function in mod.functions_items():
        if not isinstance(function, tvm.tir.PrimFunc):
            continue
        tags: list[str] = []

        def visit(node: Any) -> None:
            if isinstance(node, tvm.tir.For) and node.thread_binding is not None:
                tags.append(str(node.thread_binding.thread_tag))

        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
        if backend in GPU_REPLAY_BACKENDS:
            has_block = any(tag.startswith("blockIdx") for tag in tags)
            has_thread = any(tag.startswith("threadIdx") for tag in tags)
            if not has_block or not has_thread:
                raise ValueError(
                    f"trusted {backend} replay left an unbound GPU PrimFunc: "
                    f"{global_var.name_hint}, bindings={tags}"
                )
        primfuncs.append(
            {
                "primfunc": str(global_var.name_hint),
                "thread_bindings": tags,
                "tir_is_scheduled": bool(
                    function.attrs is not None
                    and function.attrs.get("tir.is_scheduled") is not None
                    and int(function.attrs["tir.is_scheduled"])
                ),
            }
        )

    relax_only = not primfuncs
    return {
        "backend": backend,
        "classification": "relax_only" if relax_only else "tir_backed",
        "relax_only_no_tunable_task": relax_only,
        "tir_primfunc_count": len(primfuncs),
        "gpu_binding_required": backend in GPU_REPLAY_BACKENDS and not relax_only,
        "primfuncs": primfuncs,
    }


@dataclass(frozen=True, slots=True)
class TVMValidationReport:
    entry_name: str
    argument_names: tuple[str, ...]
    module_json: str
    module_script: str
    module_json_sha256: str
    module_script_sha256: str
    structural_hash: str
    roundtrip_structural_hash: str
    relax_well_formed: bool
    tir_function_count: int
    codegen_kind: CodegenKind
    warnings: tuple[str, ...]
    script_roundtrip_checked: bool = True
    trusted_replay_structure: Mapping[str, JSONValue] | None = None

    def to_manifest(self) -> dict[str, JSONValue]:
        return {
            "entry_name": self.entry_name,
            "argument_names": list(self.argument_names),
            "module_json_sha256": self.module_json_sha256,
            "module_script_sha256": self.module_script_sha256,
            "structural_hash": self.structural_hash,
            "roundtrip_structural_hash": self.roundtrip_structural_hash,
            "relax_well_formed": self.relax_well_formed,
            "tir_function_count": self.tir_function_count,
            "codegen_kind": self.codegen_kind,
            "warnings": list(self.warnings),
            "script_roundtrip_checked": self.script_roundtrip_checked,
            "compute_structure": (
                "relax_only" if self.tir_function_count == 0 else "tir_backed"
            ),
            "trusted_replay_structure": (
                None
                if self.trusted_replay_structure is None
                else dict(self.trusted_replay_structure)
            ),
        }


def _fault(stage: FaultStage, code: str, error: BaseException) -> PipelineFault:
    diagnostic = str(error) or type(error).__name__
    return PipelineFault(
        FaultRecord(
            stage,
            code,
            diagnostic,
            exception_type=type(error).__name__,
            diagnostic_sha256=diagnostic_digest(diagnostic),
        ),
        diagnostic=diagnostic,
    )


def _fixed_tensor_spec(struct_info: Any, *, name: str) -> TensorSpec:
    import tvm

    if not isinstance(struct_info, tvm.relax.TensorStructInfo):
        raise TypeError(f"entry argument {name!r} is not a TensorStructInfo")
    shape_expr = struct_info.shape
    if not isinstance(shape_expr, tvm.relax.ShapeExpr):
        raise TypeError(f"entry argument {name!r} does not have a fixed ShapeExpr")
    shape: list[int] = []
    for dim in shape_expr.values:
        if not isinstance(dim, tvm.tir.IntImm):
            raise TypeError(f"entry argument {name!r} has a symbolic dimension")
        shape.append(int(dim))
    dtype = str(struct_info.dtype)
    if dtype in {"", "void"}:
        raise TypeError(f"entry argument {name!r} has no concrete dtype")
    return TensorSpec(tuple(shape), dtype, name)


def _output_tensor_specs(struct_info: Any, *, prefix: str = "output") -> tuple[TensorSpec, ...]:
    import tvm

    if isinstance(struct_info, tvm.relax.TensorStructInfo):
        return (_fixed_tensor_spec(struct_info, name=prefix),)
    if isinstance(struct_info, tvm.relax.TupleStructInfo):
        return tuple(
            spec
            for index, field in enumerate(struct_info.fields)
            for spec in _output_tensor_specs(field, prefix=f"{prefix}_{index}")
        )
    raise TypeError("Relax entry output must be a tensor or tensor tuple")


def _assert_same_spec(expected: TensorSpec, actual: TensorSpec, *, label: str) -> None:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError(
            f"{label} differs: expected shape={expected.shape}, dtype={expected.dtype}; "
            f"entry has shape={actual.shape}, dtype={actual.dtype}"
        )


def _task_input_names(task: BuildTask) -> tuple[str, ...]:
    names = getattr(task, "input_names", None)
    if names is not None:
        return tuple(names)
    result = tuple(spec.name for spec in task.input_specs)
    if any(name is None for name in result):
        raise ValueError("BuildTask without input_names requires named input_specs")
    return tuple(str(name) for name in result)


def _classify_codegen(mod: Any) -> CodegenKind:
    import tvm

    attrs = mod.attrs
    if attrs is not None and any(
        str(key) in {"external_mods", "device_modules", "embedded_source"} for key in attrs.keys()
    ):
        return "embedded_backend_source"
    has_external_call = False

    def visit(node: Any) -> None:
        nonlocal has_external_call
        if isinstance(node, tvm.tir.Call):
            op_name = getattr(node.op, "name", "")
            if op_name in {"tir.call_extern", "tir.call_packed", "tir.call_cpacked"}:
                has_external_call = True

    for _, function in mod.functions_items():
        if isinstance(function, tvm.tir.PrimFunc):
            tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    return "tvm_external_library_call" if has_external_call else "tvm_native_codegen"


def _validate_abi(result: BuildResult, task: BuildTask, entry: Any) -> None:
    import tvm

    expected_input_names = _task_input_names(task)
    if result.input_names != expected_input_names:
        raise ValueError(
            f"BuildResult.input_names {result.input_names!r} do not match task inputs "
            f"{expected_input_names!r}"
        )
    if len(entry.params) != len(result.argument_names):
        raise ValueError(
            f"Relax entry expects {len(entry.params)} arguments; BuildResult provides "
            f"{len(result.argument_names)}"
        )
    # Runtime binding uses argument order, not Var name hints.
    if entry.attrs is not None and "num_input" in entry.attrs:
        num_input = int(entry.attrs["num_input"])
        if num_input != len(result.input_names):
            raise ValueError(
                f"Relax num_input={num_input} differs from BuildResult dynamic input count "
                f"{len(result.input_names)}"
            )
    expected_specs = tuple(task.input_specs) + tuple(
        TensorSpec.from_array(param.value, name=param.name) for param in result.params
    )
    for name, expected, parameter in zip(result.argument_names, expected_specs, entry.params):
        actual = _fixed_tensor_spec(parameter.struct_info, name=name)
        _assert_same_spec(expected, actual, label=f"entry argument {name!r}")
    actual_outputs = _output_tensor_specs(entry.ret_struct_info)
    expected_outputs = task.output_spec.tensor_leaves()
    if len(actual_outputs) != len(expected_outputs):
        raise ValueError(
            f"Relax entry returns {len(actual_outputs)} tensors; task expects "
            f"{len(expected_outputs)}"
        )
    for index, (expected, actual) in enumerate(zip(expected_outputs, actual_outputs)):
        _assert_same_spec(expected, actual, label=f"entry output {index}")


def validate_build_result(
    result: BuildResult,
    task: BuildTask,
    *,
    require_script_roundtrip: bool = True,
    trusted_replay_backend: str | None = None,
) -> TVMValidationReport:
    """Check types, ABI, structure, and serialization without rewriting the IR.

    Trusted baselines may skip TVMScript parse-back when printing changes
    extent dtypes. JSON round-trip validation remains mandatory; generated
    candidates must retain the default TVMScript check.
    """

    import tvm

    if not isinstance(result, BuildResult):
        raise TypeError("result must be BuildResult")
    if not isinstance(result.mod, tvm.IRModule):
        error = TypeError("BuildResult.mod must be tvm.IRModule")
        raise _fault(FaultStage.TVM_IR_CONSTRUCTION, "not_ir_module", error)
    try:
        entry = result.mod[result.entry_name]
        if not isinstance(entry, tvm.relax.Function):
            raise TypeError(f"entry {result.entry_name!r} is not a Relax Function")
    except BaseException as error:
        raise _fault(FaultStage.ENTRY_ABI, "missing_or_invalid_entry", error) from error
    try:
        _validate_abi(result, task, entry)
    except BaseException as error:
        raise _fault(FaultStage.ENTRY_ABI, "abi_mismatch", error) from error

    try:
        relax_ok = bool(tvm.relax.analysis.well_formed(result.mod, check_struct_info=True))
        if not relax_ok:
            raise ValueError("tvm.relax.analysis.well_formed returned false")
        tir_count = 0
        for _, function in result.mod.functions_items():
            if isinstance(function, tvm.tir.PrimFunc):
                tir_count += 1
                tvm.tir.analysis.verify_well_formed(function, assert_mode=True)
        module_json = tvm.ir.save_json(result.mod)
        json_roundtrip = tvm.ir.load_json(module_json)
        tvm.ir.assert_structural_equal(result.mod, json_roundtrip, map_free_vars=False)
        module_script = result.mod.script(show_meta=True)
        if require_script_roundtrip:
            script_roundtrip = tvm.script.from_source(module_script)
            tvm.ir.assert_structural_equal(
                result.mod, script_roundtrip, map_free_vars=False
            )
    except BaseException as error:
        raise _fault(FaultStage.TVM_WELL_FORMED, "official_validation_failed", error) from error

    replay_structure = None
    if trusted_replay_backend is not None:
        try:
            replay_structure = validate_trusted_replay_structure(
                result.mod, backend=trusted_replay_backend
            )
        except BaseException as error:
            raise _fault(
                FaultStage.TVM_WELL_FORMED,
                "trusted_replay_structure_invalid",
                error,
            ) from error

    warnings: list[str] = []
    if not require_script_roundtrip:
        warnings.append(
            "TVMScript parse-back comparison skipped for trusted replay module"
        )
    actual_names = tuple(parameter.name_hint for parameter in entry.params)
    if actual_names != result.argument_names:
        warnings.append(
            "Relax entry Var name hints differ from BuildResult names; positional ABI is used"
        )
    if tir_count == 0:
        warnings.append(
            "entry contains no TIR PrimFunc; legal Relax-only/view programs remain eligible"
        )
    codegen_kind = _classify_codegen(result.mod)
    if codegen_kind != "tvm_native_codegen":
        warnings.append(
            f"codegen category is {codegen_kind}; keep separate from native-TIR aggregate"
        )
    return TVMValidationReport(
        entry_name=result.entry_name,
        argument_names=result.argument_names,
        module_json=module_json,
        module_script=module_script,
        module_json_sha256=hashlib.sha256(module_json.encode("utf-8")).hexdigest(),
        module_script_sha256=hashlib.sha256(module_script.encode("utf-8")).hexdigest(),
        structural_hash=str(tvm.ir.structural_hash(result.mod)),
        roundtrip_structural_hash=str(tvm.ir.structural_hash(json_roundtrip)),
        relax_well_formed=relax_ok,
        tir_function_count=tir_count,
        codegen_kind=codegen_kind,
        warnings=tuple(warnings),
        script_roundtrip_checked=require_script_roundtrip,
        trusted_replay_structure=replay_structure,
    )


__all__ = [
    "CodegenKind",
    "GPU_REPLAY_BACKENDS",
    "TRUSTED_REPLAY_BACKENDS",
    "TVMValidationReport",
    "validate_build_result",
    "validate_trusted_replay_structure",
]
