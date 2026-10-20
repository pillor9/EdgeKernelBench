"""Isolated worker for the conservative L1 executable seed."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import traceback
from pathlib import Path

from ..candidate_source import validate_candidate_source


_SCRIPT_HEADER = (
    "import tvm\n"
    "from tvm.script import ir as I\n"
    "from tvm.script import tir as T\n"
    "from tvm.script import relax as R\n\n"
)


def _load_module(source: str, workdir: Path):
    validate_candidate_source(source)
    path = workdir / "canonical_t01.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("leka_l1_seed_input", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical T01 module")
    imported = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = imported
    try:
        spec.loader.exec_module(imported)
    finally:
        sys.modules.pop(spec.name, None)
    return imported.Module


def _bind_loopless_scalar_seeds(mod, target):
    """Give scalar-output GPU seeds one thread when DLight has no loop to bind.

    This is an execution schedule, not a graph rewrite or candidate repair.
    Existing thread schedules and tensor kernels are left untouched.
    """
    import tvm
    from tvm import tir

    if target.kind.name not in ("cuda", "opencl"):
        return mod
    result = tvm.IRModule(mod.functions, attrs=mod.attrs)
    for gv, func in mod.functions.items():
        if not isinstance(func, tir.PrimFunc) or not func.buffer_map:
            continue
        blocked = []
        stores = []

        def inspect(node):
            if isinstance(node, tir.BufferStore):
                stores.append(node)
                if len(node.buffer.shape) != 0:
                    blocked.append(True)
            if isinstance(node, tir.For):
                blocked.append(True)
            if isinstance(node, tir.AttrStmt) and node.attr_key in (
                "thread_extent", "virtual_thread"
            ):
                blocked.append(True)

        tir.stmt_functor.post_order_visit(func.body, inspect)
        if blocked or not stores:
            continue
        thread = tir.IterVar(
            tvm.ir.Range(0, 1), tir.Var("scalar_thread", "int32"),
            tir.IterVar.ThreadIndex, "threadIdx.x",
        )
        body = tir.AttrStmt(thread, "thread_extent", 1, func.body)
        result[gv] = func.with_body(body).with_attr("tir.is_scheduled", True)
    return result


def main() -> None:
    request = json.loads(sys.stdin.read())
    stage = "l1_seed_load"
    try:
        import tvm
        from tvm import dlight as dl

        canonical_source = request["canonical_source"]
        target = tvm.target.Target(request["target"])
        if target.kind.name == "llvm":
            print(
                json.dumps(
                    {
                        "ok": True,
                        "source": canonical_source,
                        "scheduler": "canonical_cpu",
                    }
                )
            )
            return
        with tempfile.TemporaryDirectory() as tmp:
            mod = _load_module(canonical_source, Path(tmp))
        stage = "l1_seed_schedule"
        with target:
            scheduled = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
            scheduled = _bind_loopless_scalar_seeds(scheduled, target)
        stage = "l1_seed_serialize"
        print(
            json.dumps(
                {
                    "ok": True,
                    "source": _SCRIPT_HEADER + scheduled.script(),
                    "scheduler": "dlight_gpu_fallback",
                }
            )
        )
    except Exception:
        print(
            json.dumps(
                {
                    "ok": False,
                    "fault_stage": stage,
                    "error": traceback.format_exc(),
                }
            )
        )


if __name__ == "__main__":
    main()
