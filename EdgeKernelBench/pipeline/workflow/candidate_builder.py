"""Execute generated ``build(task)`` in a bounded worker process."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
import linecache
import multiprocessing as mp
import os
import tempfile
import time
import traceback
import types
import sys
from typing import Any, Mapping

import numpy as np

from .tensor_utils import as_c_contiguous

from pipeline.interfaces import (
    BuildResult,
    BuildTask,
    BuildTaskSnapshot,
    NamedArray,
    RuntimePolicy,
    TensorSpec,
    TreeSpec,
)

from .errors import FaultRecord, FaultStage, PipelineFault, diagnostic_digest
from .response_parser import sha256_text
from .candidate_safety import (
    SafetyReport,
    SandboxLimits,
    SandboxPolicy,
    apply_process_resource_limits,
    require_safe_source,
    restricted_builtins,
)


@dataclass(frozen=True, slots=True)
class BuildExecution:
    result: BuildResult
    safety_report: SafetyReport
    source_sha256: str
    duration_seconds: float
    stdout: str
    stderr: str
    worker_start_method: str


def _array_payload(array: np.ndarray) -> dict[str, Any]:
    value = as_c_contiguous(array)
    if value.dtype.hasobject:
        raise TypeError("object arrays cannot cross the build boundary")
    return {
        "dtype": value.dtype.str,
        "shape": tuple(int(dim) for dim in value.shape),
        "data": value.tobytes(order="C"),
    }


def _array_from_payload(payload: Mapping[str, Any]) -> np.ndarray:
    dtype = np.dtype(payload["dtype"])
    shape = tuple(int(dim) for dim in payload["shape"])
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    data = payload["data"]
    if not isinstance(data, bytes) or len(data) != expected:
        raise ValueError("invalid array payload length")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy(order="C")


def _tree_from_dict(payload: Mapping[str, Any]) -> TreeSpec:
    kind = str(payload["kind"])
    if kind == "tensor":
        tensor = payload["tensor"]
        return TreeSpec.tensor_leaf(
            TensorSpec(
                tuple(int(dim) for dim in tensor["shape"]),
                str(tensor["dtype"]),
                tensor.get("name"),
            )
        )
    return TreeSpec(
        kind=kind,  # type: ignore[arg-type]
        children=tuple(_tree_from_dict(child) for child in payload.get("children", ())),
        keys=tuple(str(key) for key in payload.get("keys", ())),
    )


def task_to_worker_payload(task: BuildTask) -> dict[str, Any]:
    """Create a value-only payload; hidden correctness data is never included."""

    input_names = tuple(
        getattr(
            task,
            "input_names",
            tuple(spec.name or f"input_{i}" for i, spec in enumerate(task.input_specs)),
        )
    )
    return {
        "source_text": task.source_text,
        "init_args": tuple(task.init_args),
        "input_specs": [spec.to_dict() for spec in task.input_specs],
        "output_spec": task.output_spec.to_dict(),
        "input_names": input_names,
        "task_id": str(getattr(task, "task_id", "anonymous_task")),
        "parameters": [
            (name, _array_payload(task.parameter(name))) for name in task.parameter_names()
        ],
        "buffers": [(name, _array_payload(task.buffer(name))) for name in task.buffer_names()],
        "execution_mode": str(task.execution_mode),
    }


def task_from_worker_payload(payload: Mapping[str, Any]) -> BuildTaskSnapshot:
    input_specs = tuple(
        TensorSpec(
            tuple(int(dim) for dim in spec["shape"]),
            str(spec["dtype"]),
            spec.get("name"),
        )
        for spec in payload["input_specs"]
    )
    return BuildTaskSnapshot(
        source_text=str(payload["source_text"]),
        init_args=tuple(payload["init_args"]),
        input_specs=input_specs,
        output_spec=_tree_from_dict(payload["output_spec"]),
        input_names=tuple(str(name) for name in payload["input_names"]),
        task_id=str(payload["task_id"]),
        _parameters={
            name: _array_from_payload(array) for name, array in payload["parameters"]
        },
        _buffers={name: _array_from_payload(array) for name, array in payload["buffers"]},
        execution_mode=str(payload.get("execution_mode", "eval")),
    )


def _candidate_line(error: BaseException) -> int | None:
    """Return a trusted candidate frame without parsing exception prose."""

    if isinstance(error, SyntaxError):
        filename = error.filename
        lineno = error.lineno
        if (
            isinstance(filename, str)
            and os.path.basename(filename) == "candidate_case.py"
            and isinstance(lineno, int)
            and not isinstance(lineno, bool)
            and lineno > 0
        ):
            return lineno
    frames = traceback.extract_tb(error.__traceback__)
    lines = [
        int(frame.lineno)
        for frame in frames
        if os.path.basename(frame.filename) == "candidate_case.py" and frame.lineno > 0
    ]
    return lines[-1] if lines else None


def _result_payload(result: BuildResult, limits: SandboxLimits) -> dict[str, Any]:
    import tvm

    if not isinstance(result.mod, tvm.IRModule):
        raise TypeError("BuildResult.mod must be tvm.IRModule")
    total_param_bytes = sum(item.value.nbytes for item in result.params)
    if total_param_bytes > limits.max_param_bytes:
        raise ValueError("BuildResult params exceed sandbox byte limit")
    module_json = tvm.ir.save_json(result.mod)
    if len(module_json.encode("utf-8")) > limits.max_artifact_bytes:
        raise ValueError("serialized IRModule exceeds sandbox artifact limit")
    return {
        "module_json": module_json,
        "params": [(item.name, _array_payload(item.value)) for item in result.params],
        "input_names": result.input_names,
        "entry_name": result.entry_name,
        "runtime_policy": (
            None if result.runtime_policy is None else result.runtime_policy.profile_id
        ),
        "notes": dict(result.notes),
    }


def _result_from_payload(payload: Mapping[str, Any]) -> BuildResult:
    import tvm

    return BuildResult(
        mod=tvm.ir.load_json(payload["module_json"]),
        params=tuple(
            NamedArray(name, _array_from_payload(array)) for name, array in payload["params"]
        ),
        input_names=tuple(payload["input_names"]),
        entry_name=str(payload["entry_name"]),
        runtime_policy=(
            None
            if payload["runtime_policy"] is None
            else RuntimePolicy(str(payload["runtime_policy"]))
        ),
        notes=dict(payload["notes"]),
    )


def _trim_output(value: str, limit: int) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value
    suffix = b"\n...[sandbox output truncated]"
    return (encoded[: max(0, limit - len(suffix))] + suffix).decode(
        "utf-8", errors="replace"
    )


def _sanitize_environment() -> None:
    retained = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "PATH",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
            "TVM_HOME",
        }
    }
    os.environ.clear()
    os.environ.update(retained)
    os.environ["PYTHONHASHSEED"] = "0"


def _worker_main(
    connection: Any,
    source: str,
    task_payload: Mapping[str, Any],
    policy: SandboxPolicy,
    limits: SandboxLimits,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    stage = FaultStage.TVM_IR_CONSTRUCTION
    try:
        _sanitize_environment()
        apply_process_resource_limits(limits)
        task = task_from_worker_payload(task_payload)
        with tempfile.TemporaryDirectory(prefix="pipeline-build-") as workdir:
            os.chdir(workdir)
            filename = "candidate_case.py"
            linecache.cache[filename] = (
                len(source),
                None,
                source.splitlines(True),
                filename,
            )
            module_name = "pipeline_candidate"
            module = types.ModuleType(module_name)
            module.__file__ = filename
            module.__package__ = None
            namespace = module.__dict__
            namespace["__builtins__"] = restricted_builtins(policy)
            sys.modules[module_name] = module
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = compile(source, filename, "exec", dont_inherit=True)
                    exec(code, namespace, namespace)
                    stage = FaultStage.BUILD_EXECUTION
                    build = namespace.get("build")
                    if not callable(build):
                        raise TypeError("candidate did not define callable build(task)")
                    result = build(task)
                    if not isinstance(result, BuildResult):
                        raise TypeError(
                            "build(task) must return pipeline.interfaces.BuildResult"
                        )
                    stage = FaultStage.TVM_IR_CONSTRUCTION
                    serialized = _result_payload(result, limits)
            finally:
                sys.modules.pop(module_name, None)
        connection.send(
            {
                "ok": True,
                "result": serialized,
                "stdout": _trim_output(stdout.getvalue(), limits.max_output_bytes),
                "stderr": _trim_output(stderr.getvalue(), limits.max_output_bytes),
            }
        )
    except BaseException as error:  # generated-code failure must become structured data
        diagnostic = traceback.format_exc()
        candidate_line = _candidate_line(error)
        try:
            connection.send(
                {
                    "ok": False,
                    "stage": stage.value,
                    "code": "candidate_exception",
                    "message": str(error) or type(error).__name__,
                    "exception_type": type(error).__name__,
                    "candidate_line": candidate_line,
                    "diagnostic": _trim_output(diagnostic, limits.max_output_bytes),
                    "stdout": _trim_output(stdout.getvalue(), limits.max_output_bytes),
                    "stderr": _trim_output(stderr.getvalue(), limits.max_output_bytes),
                }
            )
        except BaseException:
            pass
    finally:
        connection.close()


def run_build_isolated(
    source: str,
    task: BuildTask,
    *,
    policy: SandboxPolicy | None = None,
    limits: SandboxLimits | None = None,
    start_method: str = "spawn",
) -> BuildExecution:
    active_policy = policy or SandboxPolicy()
    active_limits = limits or SandboxLimits()
    safety = require_safe_source(source, policy=active_policy, limits=active_limits)
    task_payload = task_to_worker_payload(task)
    context = mp.get_context(start_method)
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_main,
        args=(child_connection, source, task_payload, active_policy, active_limits),
        name="pipeline-build",
        daemon=True,
    )
    started = time.monotonic()
    process.start()
    child_connection.close()
    payload: Mapping[str, Any]
    try:
        if not parent_connection.poll(active_limits.timeout_seconds):
            process.terminate()
            process.join(timeout=2.0)
            raise PipelineFault(
                FaultRecord(
                    FaultStage.BUILD_EXECUTION,
                    "timeout",
                    f"build worker exceeded {active_limits.timeout_seconds:g} seconds",
                )
            )
        payload = parent_connection.recv()
    except EOFError as error:
        raise PipelineFault(
            FaultRecord(
                FaultStage.BUILD_EXECUTION,
                "worker_crash",
                f"build worker exited without a result (exitcode={process.exitcode})",
            )
        ) from error
    finally:
        parent_connection.close()
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
    duration = time.monotonic() - started
    if not payload.get("ok"):
        diagnostic = str(payload.get("diagnostic", ""))
        record = FaultRecord(
            stage=FaultStage(payload["stage"]),
            code=str(payload.get("code", "candidate_exception")),
            message=str(payload.get("message", "candidate build failed")),
            exception_type=str(payload.get("exception_type", "Exception")),
            diagnostic_sha256=diagnostic_digest(diagnostic),
            details={
                "stdout_sha256": sha256_text(str(payload.get("stdout", ""))),
                "stderr_sha256": sha256_text(str(payload.get("stderr", ""))),
                "candidate_line": (
                    int(payload["candidate_line"])
                    if isinstance(payload.get("candidate_line"), int)
                    and not isinstance(payload.get("candidate_line"), bool)
                    and int(payload["candidate_line"]) > 0
                    else None
                ),
            },
        )
        raise PipelineFault(record, diagnostic=diagnostic)
    result = _result_from_payload(payload["result"])
    return BuildExecution(
        result=result,
        safety_report=safety,
        source_sha256=sha256_text(source),
        duration_seconds=duration,
        stdout=str(payload.get("stdout", "")),
        stderr=str(payload.get("stderr", "")),
        worker_start_method=start_method,
    )


__all__ = [
    "BuildExecution",
    "run_build_isolated",
    "task_from_worker_payload",
    "task_to_worker_payload",
]
