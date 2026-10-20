"""Trusted adapter from a KernelBench-style PyTorch task to an LEKA task.

The source task remains the semantic authority.  Model parameters, buffers,
and exported tensor constants are initialized under one driver-owned parameter
seed, lifted to Relax inputs, then detached from the printed TVMScript.  Their
values are therefore absent from agent workspaces but are supplied as fixed
hidden runtime inputs for every measurement.  Only the task's ordinary
``forward`` inputs vary by hidden seed.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import sys
from types import ModuleType
from typing import Any, Iterator, Sequence

import numpy as np

from .arrays import as_c_contiguous

from .pytorch_semantics import (
    SCHEMA_VERSION as PYTORCH_SEMANTICS_SCHEMA_VERSION,
    build_pytorch_semantics,
    defensive_json_copy,
    semantics_sha256,
)


_SCRIPT_HEADER = (
    "import tvm\n"
    "from tvm.script import ir as I\n"
    "from tvm.script import relax as R\n"
    # Relax attributes may print typed scalar literals such as T.float32.
    "from tvm.script import tir as T\n\n"
)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


@contextmanager
def _suppress_task_reseeding(module: ModuleType, *, external_seed: int) -> Iterator[None]:
    """Keep ``get_inputs`` subordinate to the evaluator-selected seed.

    Many benchmark files call ``torch.manual_seed(<literal>)`` inside
    ``get_inputs``.  Temporarily replacing the common global seed entry points
    keeps hidden draws distinct without modifying the benchmark source.
    """

    import torch

    def no_op(*_args: Any, **_kwargs: Any) -> None:
        return None

    def torch_manual_seed(*_args: Any, **_kwargs: Any) -> Any:
        return torch.default_generator

    targets: list[tuple[Any, str, Any]] = [
        (random, "seed", no_op),
        (np.random, "seed", no_op),
        (torch, "manual_seed", torch_manual_seed),
    ]
    torch_random = getattr(torch, "random", None)
    if torch_random is not None:
        targets.extend(
            (
                (torch_random, "manual_seed", torch_manual_seed),
                (torch_random, "seed", lambda *_a, **_k: int(external_seed)),
            )
        )
    torch_cuda = getattr(torch, "cuda", None)
    if torch_cuda is not None:
        targets.extend(
            (
                (torch_cuda, "manual_seed", no_op),
                (torch_cuda, "manual_seed_all", no_op),
            )
        )

    def callable_key(value: Any) -> tuple[Any, ...]:
        function = getattr(value, "__func__", None)
        owner = getattr(value, "__self__", None)
        if function is not None:
            return ("bound", id(owner), id(function))
        return ("identity", id(value))

    restores: list[tuple[Any, str, Any]] = []
    replacements: dict[tuple[Any, ...], Any] = {}
    for owner, name, replacement in targets:
        original = getattr(owner, name, None)
        if not callable(original):
            continue
        restores.append((owner, name, original))
        replacements[callable_key(original)] = replacement
        setattr(owner, name, replacement)

    module_restores: list[tuple[str, Any]] = []
    for name, value in tuple(vars(module).items()):
        if not callable(value):
            continue
        replacement = replacements.get(callable_key(value))
        if replacement is not None:
            module_restores.append((name, value))
            setattr(module, name, replacement)
    try:
        yield
    finally:
        for name, original in reversed(module_restores):
            setattr(module, name, original)
        for owner, name, original in reversed(restores):
            setattr(owner, name, original)


def _load_module(path: Path) -> ModuleType:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    name = f"leka_trusted_pytorch_task_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load PyTorch task: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    for required in ("Model", "get_init_inputs", "get_inputs"):
        if not hasattr(module, required):
            raise AttributeError(f"PyTorch task is missing {required}")
    return module


def _numpy(value: Any) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        return as_c_contiguous(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return as_c_contiguous(value)
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        return as_c_contiguous(numpy_method())
    raise TypeError(f"expected a tensor value, got {type(value).__name__}")


def _flatten_outputs(value: Any, *, prefix: str = "output") -> list[tuple[str, np.ndarray]]:
    import torch

    if isinstance(value, (torch.Tensor, np.ndarray)):
        return [(prefix, _numpy(value))]
    if isinstance(value, (tuple, list)):
        leaves: list[tuple[str, np.ndarray]] = []
        for index, child in enumerate(value):
            leaves.extend(_flatten_outputs(child, prefix=f"{prefix}_{index}"))
        return leaves
    if isinstance(value, Mapping):
        leaves = []
        for key, child in value.items():
            leaves.extend(_flatten_outputs(child, prefix=f"{prefix}_{key}"))
        return leaves
    raise TypeError(f"unsupported PyTorch output container: {type(value).__name__}")


def _tensor_spec(name: str, value: np.ndarray, *, kind: str) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(dimension) for dimension in value.shape],
        "dtype": str(value.dtype),
        "kind": kind,
    }


def _fixed_state_commitment(
    names: Sequence[str], values: Sequence[np.ndarray]
) -> str:
    """Commit to the exact hidden state ABI and bytes without exposing either."""

    if len(names) != len(values):
        raise ValueError("fixed state commitment requires aligned names and tensors")
    digest = hashlib.sha256(b"leka.fixed_state.v1\0")

    def add_field(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    for name, raw_value in zip(names, values):
        value = as_c_contiguous(raw_value)
        if value.dtype.hasobject:
            raise TypeError("object tensors cannot participate in fixed state commitment")
        add_field(str(name).encode("utf-8"))
        add_field(value.dtype.str.encode("ascii"))
        add_field(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        add_field(value.tobytes(order="C"))
    return digest.hexdigest()


@contextmanager
def _export_tensor_state_as_parameters(exported_program: Any) -> Iterator[None]:
    """Make every exported tensor state item survive the TVM importer as input.

    TVM's exported-program importer keeps PyTorch parameters as inputs when
    ``keep_params_as_input=True``, but still binds buffers and lifted tensor
    constants into the Relax module.  A printed ``show_meta=True`` module then
    exposes those tensor bytes.  During import only, classify every BUFFER and
    CONSTANT_TENSOR spec as a parameter and make its tensor available through
    ``state_dict``.  The importer consequently places all of them in its
    detachable ``params`` attribute.  Restore the ExportedProgram exactly
    afterwards so the semantic projection retains the true PyTorch roles.

    The dictionary snapshots below are shallow: tensor objects are never copied,
    serialized, hashed, or inspected here.
    """

    from torch.export.graph_signature import InputKind

    specs = list(exported_program.graph_signature.input_specs)
    original_spec_fields = [
        (spec, spec.kind, getattr(spec, "persistent", None)) for spec in specs
    ]
    state_dict = exported_program.state_dict
    constants = exported_program.constants
    original_state_dict = dict(state_dict)
    original_constants = dict(constants)
    try:
        for spec in specs:
            if spec.kind not in {InputKind.BUFFER, InputKind.CONSTANT_TENSOR}:
                continue
            target = spec.target
            if not isinstance(target, str):
                raise ValueError("exported tensor state lacks a stable target name")
            if target in original_state_dict:
                tensor = original_state_dict[target]
            elif target in original_constants:
                tensor = original_constants[target]
            else:
                raise ValueError(f"exported tensor state is missing its value: {target}")
            state_dict[target] = tensor
            constants.pop(target, None)
            spec.kind = InputKind.PARAMETER
            spec.persistent = None
        yield
    finally:
        state_dict.clear()
        state_dict.update(original_state_dict)
        constants.clear()
        constants.update(original_constants)
        for spec, kind, persistent in original_spec_fields:
            spec.kind = kind
            spec.persistent = persistent


class PyTorchTask:
    """One fixed-parameter PyTorch benchmark exposed through the LEKA task API."""

    def __init__(
        self,
        path: str | Path,
        *,
        parameter_seed: int = 0,
        task_id: str | None = None,
        atol: float | None = None,
        rtol: float | None = None,
    ) -> None:
        import torch
        from tvm import relax
        from tvm.relax.frontend.torch import from_exported_program

        source = Path(path).expanduser().resolve(strict=True)
        if source.is_symlink() or not source.is_file():
            raise ValueError("PyTorch task must be a regular non-symlink file")
        self.source_path = source
        source_bytes = source.read_bytes()
        self.source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        self._source_text = source_bytes.decode("utf-8")
        self.module = _load_module(source)
        self.parameter_seed = int(parameter_seed)
        self.TASK_ID = task_id or source.stem

        _seed_everything(self.parameter_seed)
        with _suppress_task_reseeding(
            self.module, external_seed=self.parameter_seed
        ):
            init_args = self.module.get_init_inputs()
        if not isinstance(init_args, (tuple, list)):
            raise TypeError("get_init_inputs() must return a tuple or list")
        self.init_args = tuple(init_args)
        _seed_everything(self.parameter_seed)
        self.model = self.module.Model(*self.init_args).eval()

        example_values = self._draw_dynamic_inputs(0)
        with torch.no_grad():
            pre_decomposition_exported = torch.export.export(
                self.model,
                tuple(example_values),
                strict=False,
            )
            exported = pre_decomposition_exported.run_decompositions()
        with _export_tensor_state_as_parameters(exported):
            imported = from_exported_program(
                exported,
                keep_params_as_input=True,
                unwrap_unit_return_tuple=True,
                run_ep_decomposition=False,
            )
        imported, detached = relax.frontend.detach_params(imported)
        from .importer_normalization import normalize_imported_module

        imported = normalize_imported_module(imported)
        main = imported["main"]
        attrs = main.attrs
        num_dynamic = (
            int(attrs["num_input"])
            if attrs is not None and "num_input" in attrs
            else len(example_values)
        )
        if num_dynamic != len(example_values):
            raise ValueError(
                f"Relax num_input={num_dynamic} differs from get_inputs count "
                f"{len(example_values)}"
            )
        relax_names = tuple(parameter.name_hint for parameter in main.params)
        if len(relax_names) < num_dynamic:
            raise ValueError("Relax main has fewer parameters than dynamic inputs")
        state_values = tuple(_numpy(value) for value in detached.get("main", ()))
        state_names = relax_names[num_dynamic:]
        if len(state_names) != len(state_values):
            raise ValueError(
                "detached parameter count differs from the fixed Relax state ABI: "
                f"{len(state_values)} != {len(state_names)}"
            )

        self.dynamic_input_names = relax_names[:num_dynamic]
        self.state_input_names = state_names
        self._state_values = state_values
        self._fixed_state_commitment_sha256 = _fixed_state_commitment(
            self.state_input_names,
            self._state_values,
        )
        self._example_dynamic = tuple(_numpy(value) for value in example_values)
        self._relax_module = imported
        self._relax_source = (
            _SCRIPT_HEADER
            + imported.script(show_meta=True)
            + "\n\ndef build_relax():\n    return Module\n"
        )

        first_reference = self._run_reference(self._example_dynamic)
        output_leaves = _flatten_outputs(first_reference)
        self.output_names = tuple(name for name, _value in output_leaves)
        self._output_specs = tuple(
            _tensor_spec(name, value, kind="output")
            for name, value in output_leaves
        )

        metadata = getattr(self.module, "CASE_METADATA", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        self.case_metadata = dict(metadata)
        self.ATOL = float(
            atol
            if atol is not None
            else getattr(self.module, "ATOL", metadata.get("atol", 1e-4))
        )
        self.RTOL = float(
            rtol
            if rtol is not None
            else getattr(self.module, "RTOL", metadata.get("rtol", 1e-4))
        )
        if not np.isfinite([self.ATOL, self.RTOL]).all() or min(self.ATOL, self.RTOL) < 0:
            raise ValueError("correctness tolerances must be finite and non-negative")
        self._pytorch_semantics = build_pytorch_semantics(
            task_id=self.TASK_ID,
            task_source_sha256=self.source_sha256,
            model=self.model,
            init_args=self.init_args,
            pre_decomposition_exported_program=pre_decomposition_exported,
            post_decomposition_exported_program=exported,
            relax_names=relax_names,
            num_dynamic=num_dynamic,
            public_output_specs=self._output_specs,
        )
        self._pytorch_semantics_sha256 = semantics_sha256(self._pytorch_semantics)

    def _draw_dynamic_inputs(self, seed: int) -> tuple[Any, ...]:
        _seed_everything(int(seed))
        with _suppress_task_reseeding(self.module, external_seed=int(seed)):
            values = self.module.get_inputs()
        if not isinstance(values, (tuple, list)):
            raise TypeError("get_inputs() must return a tuple or list")
        signature = inspect.signature(self.model.forward)
        try:
            signature.bind(*values)
        except TypeError as error:
            raise ValueError(
                "get_inputs() values do not satisfy Model.forward's positional ABI"
            ) from error
        return tuple(values)

    def _run_reference(self, dynamic_arrays: Sequence[np.ndarray]) -> Any:
        import torch

        tensors = tuple(torch.from_numpy(as_c_contiguous(value)) for value in dynamic_arrays)
        with torch.no_grad():
            return self.model(*tensors)

    def naive_relax_source(self) -> str:
        return self._relax_source

    def pytorch_source(self) -> str:
        """Return the public benchmark task source, never inputs or references.

        L0 uses this program as semantic context. The driver owns model state
        values, dynamic input draws, and oracle execution.
        """

        return self._source_text


    def pytorch_semantics(self) -> dict[str, Any]:
        """Return a detached, value-free PyTorch semantic projection."""

        return defensive_json_copy(self._pytorch_semantics)

    def pytorch_semantics_sha256(self) -> str:
        return self._pytorch_semantics_sha256

    def make_inputs(self, seed: int) -> dict[str, np.ndarray]:
        dynamic = tuple(_numpy(value) for value in self._draw_dynamic_inputs(seed))
        for name, expected, actual in zip(
            self.dynamic_input_names, self._example_dynamic, dynamic
        ):
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                raise ValueError(f"dynamic input spec changed for {name}")
        return {
            **{
                name: np.array(value, copy=True, order="C")
                for name, value in zip(self.dynamic_input_names, dynamic)
            },
            **{
                name: np.array(value, copy=True, order="C")
                for name, value in zip(self.state_input_names, self._state_values)
            },
        }

    def reference(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        dynamic = tuple(inputs[name] for name in self.dynamic_input_names)
        output = _flatten_outputs(self._run_reference(dynamic))
        names = tuple(name for name, _value in output)
        if names != self.output_names:
            raise ValueError("PyTorch output tree changed across hidden inputs")
        return {
            name: np.array(value, copy=True, order="C") for name, value in output
        }

    def public_facts(self) -> dict[str, Any]:
        dynamic_specs = tuple(
            _tensor_spec(name, value, kind="dynamic_input")
            for name, value in zip(self.dynamic_input_names, self._example_dynamic)
        )
        state_specs = tuple(
            _tensor_spec(name, value, kind="fixed_hidden_state")
            for name, value in zip(self.state_input_names, self._state_values)
        )
        return {
            "schema_version": "leka.pytorch_task.v2",
            "task_id": self.TASK_ID,
            "task_source_sha256": self.source_sha256,
            "entry": "main",
            "input_specs": [*dynamic_specs, *state_specs],
            "dynamic_input_specs": list(dynamic_specs),
            "state_specs": list(state_specs),
            "state_policy": "fixed_hidden_runtime_inputs",
            "output_specs": [dict(spec) for spec in self._output_specs],
            "atol": self.ATOL,
            "rtol": self.RTOL,
            "case_metadata": self.case_metadata,
            "pytorch_semantics": {
                "schema_version": PYTORCH_SEMANTICS_SCHEMA_VERSION,
                "sha256": self._pytorch_semantics_sha256,
            },
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "leka.pytorch_task_manifest.v3",
            "task_id": self.TASK_ID,
            "task_source_sha256": self.source_sha256,
            "dynamic_inputs": list(self.dynamic_input_names),
            "fixed_state_inputs": list(self.state_input_names),
            "fixed_state_commitment_sha256": self._fixed_state_commitment_sha256,
            "outputs": list(self.output_names),
            "relax_source_sha256": hashlib.sha256(
                self._relax_source.encode("utf-8")
            ).hexdigest(),
            "pytorch_semantics_schema_version": PYTORCH_SEMANTICS_SCHEMA_VERSION,
            "pytorch_semantics_sha256": self._pytorch_semantics_sha256,
        }


def load_pytorch_task(
    path: str | Path,
    *,
    parameter_seed: int = 0,
    task_id: str | None = None,
    atol: float | None = None,
    rtol: float | None = None,
) -> PyTorchTask:
    return PyTorchTask(
        path,
        parameter_seed=parameter_seed,
        task_id=task_id,
        atol=atol,
        rtol=rtol,
    )


__all__ = ["PyTorchTask", "load_pytorch_task"]
