"""Subprocess body for T01.

stdin carries the untrusted L0 candidate, the trusted naive L0 root whose
``main`` function defines the task ABI, and the target.  The candidate is
checked *before* LegalizeOps: L0 may replace the whole Relax graph and add
helper Relax functions, but it may not embed TIR across T01 or
change the positional runtime ABI.  Fusion boundaries are candidate-authored:
only private Relax helpers carrying ``Primitive=True`` are realized as fused
TIR regions.  T01 never runs automatic FuseOps, constant folding, or a target
schedule.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

from ..candidate_source import validate_candidate_source
from ..parameter_plan import (
    ParameterPlanError,
    plan_from_main,
    validate_plan_resources,
)


class L0ContractError(ValueError):
    """An agent-authored L0 artifact violates the pre-lowering contract."""


def _load_relax_module(l0_source: str, workdir: Path, *, tag: str):
    """Import the L0 artifact from a real file (TVMScript inspects sys.modules)."""
    import importlib.util

    validate_candidate_source(l0_source)
    path = workdir / f"{tag}.py"
    path.write_text(l0_source)
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {tag}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    try:
        spec.loader.exec_module(module)  # source-validated worker import
    finally:
        sys.modules.pop(tag, None)
    build = getattr(module, "build_relax", None)
    if not callable(build):
        raise RuntimeError("L0 artifact must define build_relax() -> tvm.IRModule")
    return build()


def _num_input(function, *, label: str) -> int | None:
    attrs = function.attrs
    if attrs is None or "num_input" not in attrs:
        return None
    try:
        value = int(attrs["num_input"])
    except (TypeError, ValueError) as error:
        raise L0ContractError(f"{label} main num_input must be an integer") from error
    if value < 0 or value > len(function.params):
        raise L0ContractError(
            f"{label} main num_input={value} is outside [0, {len(function.params)}]"
        )
    return value


def _call_tir_ops(mod, relax) -> list[str]:
    """Return explicit call_tir-family ops present in the raw Relax module."""
    found: set[str] = set()

    def visit(expr) -> None:
        if not isinstance(expr, relax.Call):
            return
        name = getattr(expr.op, "name", None)
        if isinstance(name, str) and name.startswith("relax.call_tir"):
            found.add(name)

    for function in mod.functions.values():
        if isinstance(function, relax.Function):
            relax.analysis.post_order_visit(function.body, visit)
    return sorted(found)


def _require_pure_relax(mod, *, label: str, tvm, relax):
    if not isinstance(mod, tvm.IRModule):
        raise L0ContractError(
            f"{label} build_relax() must return tvm.IRModule, got {type(mod).__name__}"
        )
    try:
        main = mod["main"]
    except (KeyError, ValueError, tvm.error.TVMError) as error:
        raise L0ContractError(f"{label} module must define Relax entry 'main'") from error
    if not isinstance(main, relax.Function):
        raise L0ContractError(f"{label} entry 'main' must be a Relax Function")

    call_tir_ops = _call_tir_ops(mod, relax)
    if call_tir_ops:
        raise L0ContractError(
            f"{label} contains forbidden hand-written TIR calls: {', '.join(call_tir_ops)}"
        )

    non_relax = [
        f"{global_var.name_hint}:{type(function).__name__}"
        for global_var, function in mod.functions.items()
        if not isinstance(function, relax.Function)
    ]
    if non_relax:
        raise L0ContractError(
            f"{label} may contain only Relax Functions; found {', '.join(sorted(non_relax))}"
        )
    if not relax.analysis.well_formed(mod):
        raise L0ContractError(f"{label} Relax IRModule is not well-formed")
    return main


def _primitive_attr(function, *, label: str) -> bool:
    attrs = function.attrs
    if attrs is None or "Primitive" not in attrs:
        return False
    try:
        value = int(attrs["Primitive"])
    except (TypeError, ValueError) as error:
        raise L0ContractError(f"{label} Primitive attribute must be boolean") from error
    if value not in (0, 1):
        raise L0ContractError(f"{label} Primitive attribute must be boolean")
    return value == 1


def _explicit_primitive_regions(mod, *, tvm, relax) -> tuple[str, ...]:
    """Validate and snapshot candidate-authored fusion regions."""

    regions: list[str] = []
    for global_var, function in mod.functions.items():
        if not isinstance(function, relax.Function):
            continue
        name = global_var.name_hint
        if not _primitive_attr(function, label=f"candidate function {name!r}"):
            continue
        if name == "main":
            raise L0ContractError(
                "candidate entry 'main' cannot be Primitive; use a private helper"
            )
        attrs = function.attrs
        if attrs is not None and "global_symbol" in attrs:
            raise L0ContractError(
                f"candidate Primitive region {name!r} must be a private Relax helper"
            )
        if attrs is not None and any(key in attrs for key in ("Codegen", "Composite")):
            raise L0ContractError(
                f"candidate Primitive region {name!r} cannot request external codegen"
            )
        regions.append(name)
    return tuple(sorted(regions))


def _require_primitive_regions_realized(
    mod, regions: tuple[str, ...], *, tvm, relax
) -> None:
    """Ensure FuseTIR consumed every live explicit region and created no Relax residue."""

    functions_by_name = {
        global_var.name_hint: function for global_var, function in mod.functions.items()
    }
    for name in regions:
        function = functions_by_name.get(name)
        # An unreachable private region may be removed by FuseTIR's internal
        # DCE.  Every region that remains must have become a PrimFunc.
        if function is not None and not isinstance(function, tvm.tir.PrimFunc):
            raise L0ContractError(
                f"candidate Primitive region {name!r} was not realized as TIR"
            )
    for name, function in functions_by_name.items():
        if isinstance(function, relax.Function) and _primitive_attr(
            function, label=f"lowered function {name!r}"
        ):
            raise L0ContractError(
                f"lowered module retains Primitive Relax function {name!r}"
            )


def _tensor_spec(struct_info, *, label: str, tvm, relax) -> tuple[tuple[int, ...], str]:
    if not isinstance(struct_info, relax.TensorStructInfo):
        raise L0ContractError(f"{label} must be a Tensor StructInfo")
    shape = struct_info.shape
    if not isinstance(shape, relax.ShapeExpr):
        raise L0ContractError(f"{label} must have a static ShapeExpr")
    dims: list[int] = []
    for dim in shape.values:
        if not isinstance(dim, tvm.tir.IntImm):
            raise L0ContractError(f"{label} must have a fully static shape")
        dims.append(int(dim))
    return tuple(dims), str(struct_info.dtype)


def _require_compatible_main_abi(candidate, expected, *, tvm, relax) -> dict:
    """Validate dynamic/output ABI and the declarative fixed-state plan.

    A closed JSON plan describes fixed-parameter preparation before the timed
    entry. Dynamic inputs and outputs remain frozen, while state
    arguments after ``num_input`` may be reordered, duplicated, omitted, or
    transformed into a different static shape/dtype.
    """

    candidate_num_input = _num_input(candidate, label="candidate")
    expected_num_input = _num_input(expected, label="trusted expected")
    if candidate_num_input != expected_num_input:
        raise L0ContractError(
            "candidate main num_input differs from the trusted task ABI: "
            f"{candidate_num_input!r} != {expected_num_input!r}"
        )
    if expected_num_input is None:
        raise L0ContractError("trusted expected main must declare num_input")
    if len(candidate.params) < expected_num_input:
        raise L0ContractError("candidate main omits a trusted dynamic input")

    for index, (candidate_param, expected_param) in enumerate(
        zip(
            candidate.params[:expected_num_input],
            expected.params[:expected_num_input],
        )
    ):
        if not tvm.ir.structural_equal(
            candidate_param.struct_info,
            expected_param.struct_info,
            map_free_vars=True,
        ):
            raise L0ContractError(
                f"candidate main positional parameter {index} StructInfo differs "
                "from the trusted task ABI"
            )

    if not tvm.ir.structural_equal(
        candidate.ret_struct_info,
        expected.ret_struct_info,
        map_free_vars=True,
    ):
        raise L0ContractError(
            "candidate main return StructInfo differs from the trusted task ABI"
        )

    # Compare the dynamic/output signature once as well, so a candidate cannot
    # preserve standalone symbolic shapes while changing cross-argument shape
    # relationships.  Fixed-state relationships are instead defined by the
    # validated transform plan below.
    candidate_signature = relax.TupleStructInfo(
        [
            param.struct_info
            for param in candidate.params[:expected_num_input]
        ]
        + [candidate.ret_struct_info]
    )
    expected_signature = relax.TupleStructInfo(
        [
            param.struct_info
            for param in expected.params[:expected_num_input]
        ]
        + [expected.ret_struct_info]
    )
    if not tvm.ir.structural_equal(
        candidate_signature, expected_signature, map_free_vars=True
    ):
        raise L0ContractError(
            "candidate main aggregate positional StructInfo differs from the trusted task ABI"
        )

    expected_state = expected.params[expected_num_input:]
    source_names = tuple(param.name_hint for param in expected_state)
    source_specs = {
        name: _tensor_spec(
            parameter.struct_info,
            label=f"trusted state {name!r}",
            tvm=tvm,
            relax=relax,
        )
        for name, parameter in zip(source_names, expected_state)
    }
    try:
        plan = plan_from_main(candidate, source_names)
        planned_specs = validate_plan_resources(plan, source_specs)
    except ParameterPlanError as error:
        raise L0ContractError(f"candidate fixed-state parameter plan is invalid: {error}") from error
    candidate_state = candidate.params[expected_num_input:]
    if len(candidate_state) != len(plan["bindings"]):
        raise L0ContractError(
            "candidate fixed-state parameter count differs from leka_parameter_plan: "
            f"{len(candidate_state)} != {len(plan['bindings'])}"
        )
    for index, (parameter, (expected_shape, expected_dtype)) in enumerate(
        zip(candidate_state, planned_specs)
    ):
        actual_shape, actual_dtype = _tensor_spec(
            parameter.struct_info,
            label=f"candidate fixed-state parameter {index}",
            tvm=tvm,
            relax=relax,
        )
        if actual_shape != expected_shape or actual_dtype != expected_dtype:
            raise L0ContractError(
                f"candidate fixed-state parameter {index} StructInfo {actual_shape}/"
                f"{actual_dtype} differs from planned {expected_shape}/{expected_dtype}"
            )
    return plan


_SCRIPT_HEADER = (
    "import tvm\n"
    "from tvm.script import ir as I\n"
    "from tvm.script import tir as T\n"
    "from tvm.script import relax as R\n\n"
)


def _executable_script(mod) -> str:
    """mod.script() emits commented imports; prepend real ones so the file execs."""
    return _SCRIPT_HEADER + mod.script()


def main() -> None:
    request = json.loads(sys.stdin.read())
    stage = "expected_l0"
    try:
        import tvm
        from tvm import relax

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            expected = _load_relax_module(
                request["expected_l0_source"], workdir, tag="leka_expected_l0"
            )
            expected_main = _require_pure_relax(
                expected, label="trusted expected L0", tvm=tvm, relax=relax
            )
            stage = "exec_l0"
            mod = _load_relax_module(
                request["l0_source"], workdir, tag="leka_candidate_l0"
            )
            stage = "l0_contract"
            candidate_main = _require_pure_relax(
                mod, label="candidate L0", tvm=tvm, relax=relax
            )
            _require_compatible_main_abi(
                candidate_main, expected_main, tvm=tvm, relax=relax
            )
            primitive_regions = _explicit_primitive_regions(
                mod, tvm=tvm, relax=relax
            )
        stage = "relax_legalize"
        canonical = relax.transform.LegalizeOps()(mod)
        stage = "explicit_primitive_fuse"
        canonical = relax.transform.FuseTIR()(canonical)
        stage = "relax_dce"
        canonical = relax.transform.DeadCodeElimination(
            entry_functions=["main"]
        )(canonical)
        stage = "explicit_primitive_postcondition"
        _require_primitive_regions_realized(
            canonical,
            primitive_regions,
            tvm=tvm,
            relax=relax,
        )
        canonical_script = _executable_script(canonical)
        # The legacy field aliases canonical TIR; scheduling belongs to L1.
        print(
            json.dumps(
                {
                    "ok": True,
                    "canonical": canonical_script,
                    "scheduled_default": canonical_script,
                }
            )
        )
    except Exception:
        print(
            json.dumps(
                {"ok": False, "fault_stage": stage, "error": traceback.format_exc()}
            )
        )


if __name__ == "__main__":
    main()
