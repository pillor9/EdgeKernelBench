"""Trusted PyTorch task loading separated from the generated-code worker."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import sys
import traceback
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np

from .tensor_utils import as_c_contiguous

from pipeline.interfaces import BuildTaskSnapshot, NamedArray, TensorSpec, TreeSpec

from .candidate_builder import task_from_worker_payload, task_to_worker_payload
from .numerical_policy import contract_from_task_module, validate_correctness_contract
from .errors import FaultOwner, FaultRecord, FaultStage, PipelineFault, diagnostic_digest


# Draw correctness inputs from the task's original distribution.
PLAIN_CORRECTNESS_SEEDS: tuple[int, ...] = (0, 1, 2)
CANONICAL_CORRECTNESS_SEEDS: tuple[int, ...] = PLAIN_CORRECTNESS_SEEDS
INPUT_IDENTITY_SCHEMA = "pipeline.correctness_inputs.v1"
CORRECTNESS_NONCE_SCHEMA = "pipeline.correctness_nonce.v1"


def _draw_seed(
    *, parameter_seed: int, correctness_seed: int, correctness_nonce: str | None
) -> int:
    """Map a stable case label to an evaluator-owned hidden RNG draw."""

    base = correctness_seed
    if correctness_nonce is None:
        return base
    payload = (
        f"{CORRECTNESS_NONCE_SCHEMA}\0{correctness_nonce}\0"
        f"{parameter_seed}\0{base}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


@dataclass(frozen=True, slots=True)
class TaskLoadConfig:
    task_id: str | None = None
    parameter_seeds: tuple[int, ...] = (0, 1)
    correctness_seeds: tuple[int, ...] = CANONICAL_CORRECTNESS_SEEDS
    correctness_nonce: str | None = None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        for name in ("parameter_seeds", "correctness_seeds"):
            seeds = getattr(self, name)
            if (
                not seeds
                or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != len(seeds)
            ):
                raise ValueError(f"{name} must contain unique non-negative integers")
            object.__setattr__(self, name, tuple(seeds))
        if self.correctness_nonce is not None and (
            not isinstance(self.correctness_nonce, str)
            or len(self.correctness_nonce) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.correctness_nonce
            )
        ):
            raise ValueError(
                "correctness_nonce must be null or 64 lowercase hex characters"
            )
        if (
            isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class HiddenCase:
    seed: int
    inputs: tuple[NamedArray, ...]
    reference: tuple[NamedArray, ...]


@dataclass(frozen=True, slots=True)
class LoadedTask:
    parameter_seed: int
    build_task: BuildTaskSnapshot
    hidden_cases: tuple[HiddenCase, ...]
    task_source_sha256: str
    correctness_contract: Mapping[str, Any] | None = None

    def hidden_case(self, seed: int) -> HiddenCase:
        for case in self.hidden_cases:
            if case.seed == seed:
                return case
        raise KeyError(f"unknown correctness seed: {seed}")


@dataclass(frozen=True, slots=True)
class MultiSeedLoadedTask:
    variants: tuple[LoadedTask, ...]
    task_source_sha256: str

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError("MultiSeedLoadedTask requires at least one parameter seed")
        if len({variant.parameter_seed for variant in self.variants}) != len(self.variants):
            raise ValueError("parameter seeds must be unique")
        if any(
            variant.task_source_sha256 != self.task_source_sha256 for variant in self.variants
        ):
            raise ValueError("all parameter variants must originate from the same task source")

    @property
    def primary(self) -> LoadedTask:
        return self.variants[0]

    @property
    def build_task(self) -> BuildTaskSnapshot:
        """Representative prompt task; evaluation must iterate ``variants``."""

        return self.primary.build_task

    def variant(self, parameter_seed: int) -> LoadedTask:
        for variant in self.variants:
            if variant.parameter_seed == parameter_seed:
                return variant
        raise KeyError(f"unknown parameter seed: {parameter_seed}")


def _load_module(path: Path) -> ModuleType:
    source_digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    module_name = f"pipeline_trusted_task_{source_digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    import torch

    torch.manual_seed(seed)


@contextmanager
def _suppress_internal_reseeding(
    module: ModuleType, *, torch: ModuleType, external_seed: int
):
    """Preserve the loader's seed while the task's input factory runs.

    Temporarily disable global reseeding calls and their imported aliases.
    Separate RNG instances remain unchanged; input hashes detect repeated draws.
    """

    def _none(*_args: Any, **_kwargs: Any) -> None:
        return None

    def _torch_manual_seed(*_args: Any, **_kwargs: Any) -> Any:
        return torch.default_generator

    def _torch_seed(*_args: Any, **_kwargs: Any) -> int:
        return int(external_seed)

    targets: list[tuple[Any, str, Any]] = [
        (random, "seed", _none),
        (np.random, "seed", _none),
        (torch, "manual_seed", _torch_manual_seed),
    ]
    torch_random = getattr(torch, "random", None)
    if torch_random is not None:
        targets.extend(
            (
                (torch_random, "manual_seed", _torch_manual_seed),
                (torch_random, "seed", _torch_seed),
            )
        )
    torch_cuda = getattr(torch, "cuda", None)
    if torch_cuda is not None:
        targets.extend(
            (
                (torch_cuda, "manual_seed", _none),
                (torch_cuda, "manual_seed_all", _none),
            )
        )

    owner_restores: list[tuple[Any, str, Any]] = []
    def _callable_key(value: Any) -> tuple[str, int, int] | tuple[str, int]:
        bound_function = getattr(value, "__func__", None)
        bound_owner = getattr(value, "__self__", None)
        if bound_function is not None:
            return ("bound", id(bound_owner), id(bound_function))
        return ("identity", id(value))

    replacements: dict[tuple[Any, ...], Any] = {}
    for owner, name, replacement in targets:
        original = getattr(owner, name, None)
        if not callable(original):
            continue
        owner_restores.append((owner, name, original))
        replacements[_callable_key(original)] = replacement
        setattr(owner, name, replacement)

    module_restores: list[tuple[str, Any]] = []
    for name, value in tuple(vars(module).items()):
        if not callable(value):
            continue
        replacement = replacements.get(_callable_key(value))
        if replacement is None:
            continue
        module_restores.append((name, value))
        setattr(module, name, replacement)
    try:
        yield
    finally:
        for name, original in reversed(module_restores):
            setattr(module, name, original)
        for owner, name, original in reversed(owner_restores):
            setattr(owner, name, original)


def _to_numpy(value: Any) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return as_c_contiguous(tensor.numpy())
    if isinstance(value, np.ndarray):
        return as_c_contiguous(value)
    raise TypeError(f"task values must be torch.Tensor or numpy.ndarray, got {type(value).__name__}")


def input_identity_sha256(inputs: Sequence[NamedArray]) -> str:
    """Return one canonical, order-sensitive identity for an input set."""

    digest = hashlib.sha256()
    digest.update(INPUT_IDENTITY_SCHEMA.encode("ascii") + b"\0")
    for index, item in enumerate(inputs):
        if not isinstance(item, NamedArray):
            raise TypeError("correctness inputs must contain NamedArray values")
        value = as_c_contiguous(item.value)
        header = json.dumps(
            {
                "index": index,
                "name": item.name,
                "dtype": value.dtype.str,
                "shape": [int(dimension) for dimension in value.shape],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        payload = value.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def correctness_input_identities(
    hidden_cases: Sequence[HiddenCase],
) -> dict[int, str]:
    """Map seeds to identities and reject any repeated canonical input set."""

    identities: dict[int, str] = {}
    seed_by_identity: dict[str, int] = {}
    for case in hidden_cases:
        if case.seed in identities:
            raise ValueError(f"duplicate correctness seed in hidden cases: {case.seed}")
        identity = input_identity_sha256(case.inputs)
        if not case.inputs:
            identities[case.seed] = identity
            continue
        previous = seed_by_identity.get(identity)
        all_discrete = all(
            np.issubdtype(item.value.dtype, np.integer)
            or np.issubdtype(item.value.dtype, np.bool_)
            for item in case.inputs
        )
        if previous is not None and not all_discrete:
            raise ValueError(
                "canonical correctness seeds produced identical inputs: "
                f"{previous} and {case.seed} ({identity}); every canonical "
                "correctness case must have a distinct input identity"
            )
        identities[case.seed] = identity
        seed_by_identity[identity] = case.seed
    return identities


def _input_values_identity_sha256(values: Sequence[Any]) -> str:
    """Hash raw task inputs with stable positional names before ABI naming."""

    return input_identity_sha256(
        tuple(
            NamedArray(f"input_{index}", _to_numpy(value))
            for index, value in enumerate(values)
        )
    )


def _tree_arrays(value: Any, *, prefix: str) -> tuple[TreeSpec, tuple[NamedArray, ...]]:
    if hasattr(value, "detach") or isinstance(value, np.ndarray):
        array = _to_numpy(value)
        item = NamedArray(prefix, array)
        return TreeSpec.tensor_leaf(TensorSpec.from_array(array, name=prefix)), (item,)
    if isinstance(value, tuple):
        children: list[TreeSpec] = []
        arrays: list[NamedArray] = []
        for index, child in enumerate(value):
            child_spec, child_arrays = _tree_arrays(child, prefix=f"{prefix}_{index}")
            children.append(child_spec)
            arrays.extend(child_arrays)
        return TreeSpec("tuple", children=tuple(children)), tuple(arrays)
    if isinstance(value, list):
        children = []
        arrays = []
        for index, child in enumerate(value):
            child_spec, child_arrays = _tree_arrays(child, prefix=f"{prefix}_{index}")
            children.append(child_spec)
            arrays.extend(child_arrays)
        return TreeSpec("list", children=tuple(children)), tuple(arrays)
    if isinstance(value, Mapping):
        keys = tuple(str(key) for key in value)
        if len(keys) != len(set(keys)):
            raise ValueError("output mapping keys collapse to duplicate strings")
        children = []
        arrays = []
        for key, child in zip(keys, value.values()):
            child_spec, child_arrays = _tree_arrays(child, prefix=f"{prefix}_{key}")
            children.append(child_spec)
            arrays.extend(child_arrays)
        return TreeSpec("dict", children=tuple(children), keys=keys), tuple(arrays)
    raise TypeError(f"unsupported tensor tree node: {type(value).__name__}")


def _input_names(model: Any, count: int) -> tuple[str, ...]:
    signature = inspect.signature(model.forward)
    parameters = tuple(signature.parameters.values())
    if any(
        item.kind
        not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        for item in parameters
    ):
        raise TypeError("Model.forward must use fixed positional tensor inputs")
    if len(parameters) != count:
        raise ValueError(
            f"Model.forward expects {len(parameters)} inputs but get_inputs returned {count}"
        )
    return tuple(item.name for item in parameters)


def _validate_same_specs(
    expected: Sequence[TensorSpec], actual: Sequence[TensorSpec], *, label: str
) -> None:
    left = tuple((item.shape, item.dtype) for item in expected)
    right = tuple((item.shape, item.dtype) for item in actual)
    if left != right:
        raise ValueError(f"{label} shape/dtype changes across correctness seeds")


def _all_finite(values: object) -> bool:
    import numpy as _np

    if isinstance(values, _np.ndarray):
        if not _np.issubdtype(values.dtype, _np.floating):
            return True
        return bool(_np.isfinite(values).all())
    if isinstance(values, (tuple, list)):
        return all(_all_finite(item) for item in values)
    return True


def _all_inputs_discrete(input_values: Sequence[Any], *, torch: ModuleType) -> bool:
    return bool(input_values) and all(
        torch.is_tensor(value)
        and not torch.is_floating_point(value)
        and not torch.is_complex(value)
        for value in input_values
    )


def _draw_correctness_case(
    module: ModuleType,
    model: Any,
    seed: int,
    *,
    torch: ModuleType,
    draw_seed: int | None = None,
    forbidden_input_identities: frozenset[str] = frozenset(),
) -> tuple[tuple, Any, float]:
    """Return one ordinary, unscaled correctness draw and eager reference."""

    base_seed = seed if draw_seed is None else draw_seed
    _seed_everything(base_seed)
    with _suppress_internal_reseeding(
        module, torch=torch, external_seed=base_seed
    ):
        drawn = module.get_inputs()
    if not isinstance(drawn, (tuple, list)):
        raise TypeError("get_inputs() must return a tuple or list of tensors")
    drawn = tuple(drawn)
    if not drawn:
        # Constant graphs have no dynamic inputs; still check each reference.
        with torch.no_grad():
            output_value = model()
        _, arrays = _tree_arrays(output_value, prefix="output")
        if not _all_finite(tuple(item.value for item in arrays)):
            raise ValueError(f"correctness seed {seed} has a nonfinite no-input reference")
        return drawn, output_value, 1.0
    identity = _input_values_identity_sha256(drawn)
    if (
        identity in forbidden_input_identities
        and not _all_inputs_discrete(drawn, torch=torch)
    ):
        raise ValueError(
            f"canonical correctness seed {seed} repeats an earlier input identity"
        )
    with torch.no_grad():
        return drawn, model(*drawn), 1.0


def _load_parameter_variant(
    module: ModuleType,
    *,
    source: str,
    source_sha256: str,
    task_id: str,
    parameter_seed: int,
    correctness_seeds: tuple[int, ...],
    correctness_nonce: str | None,
) -> LoadedTask:
    import torch

    _seed_everything(parameter_seed)
    with _suppress_internal_reseeding(
        module, torch=torch, external_seed=parameter_seed
    ):
        init_args_value = module.get_init_inputs()
        if not isinstance(init_args_value, (tuple, list)):
            raise TypeError("get_init_inputs() must return a tuple or list")
        init_args = tuple(init_args_value)
        model = module.Model(*init_args)
    model.eval()
    parameters = {
        name: _to_numpy(value) for name, value in model.named_parameters(recurse=True)
    }
    buffers = {name: _to_numpy(value) for name, value in model.named_buffers(recurse=True)}

    hidden_cases: list[HiddenCase] = []
    first_input_specs: tuple[TensorSpec, ...] | None = None
    first_output_spec: TreeSpec | None = None
    names: tuple[str, ...] | None = None
    seen_raw_input_identities: dict[str, int] = {}
    for seed in correctness_seeds:
        input_values, output_value, _stress_scale = _draw_correctness_case(
            module,
            model,
            seed,
            torch=torch,
            draw_seed=(
                None
                if correctness_nonce is None
                else _draw_seed(
                    parameter_seed=parameter_seed,
                    correctness_seed=seed,
                    correctness_nonce=correctness_nonce,
                )
            ),
            forbidden_input_identities=frozenset(seen_raw_input_identities),
        )
        if input_values:
            raw_identity = _input_values_identity_sha256(input_values)
            previous_seed = seen_raw_input_identities.get(raw_identity)
            if (
                previous_seed is not None
                and not _all_inputs_discrete(input_values, torch=torch)
            ):
                raise ValueError(
                    "canonical correctness seeds produced identical raw inputs: "
                    f"{previous_seed} and {seed} ({raw_identity})"
                )
            seen_raw_input_identities[raw_identity] = seed
        if names is None:
            names = _input_names(model, len(input_values))
        input_arrays = tuple(_to_numpy(value) for value in input_values)
        input_specs = tuple(
            TensorSpec.from_array(value, name=name) for name, value in zip(names, input_arrays)
        )
        output_spec, reference = _tree_arrays(output_value, prefix="output")
        if first_input_specs is None:
            first_input_specs = input_specs
            first_output_spec = output_spec
        else:
            _validate_same_specs(first_input_specs, input_specs, label="input")
            assert first_output_spec is not None
            _validate_same_specs(
                first_output_spec.tensor_leaves(),
                output_spec.tensor_leaves(),
                label="output",
            )
            if first_output_spec != output_spec:
                raise ValueError("output container structure changes across correctness seeds")
        hidden_cases.append(
            HiddenCase(
                seed=seed,
                inputs=tuple(
                    NamedArray(name, value) for name, value in zip(names, input_arrays)
                ),
                reference=reference,
            )
        )
    correctness_input_identities(hidden_cases)
    assert first_input_specs is not None and first_output_spec is not None and names is not None
    task = BuildTaskSnapshot(
        source_text=source,
        init_args=init_args,
        input_specs=first_input_specs,
        output_spec=first_output_spec,
        input_names=names,
        task_id=task_id,
        _parameters=parameters,
        _buffers=buffers,
        execution_mode="eval",
    )
    return LoadedTask(
        parameter_seed=parameter_seed,
        build_task=task,
        hidden_cases=tuple(hidden_cases),
        task_source_sha256=source_sha256,
        correctness_contract=(
            contract_from_task_module(module, source_sha256)
            if correctness_nonce is None and correctness_seeds == CANONICAL_CORRECTNESS_SEEDS
            else None  # Custom seeds do not use the canonical correctness profile.
        ),
    )


def load_task(
    path: str | Path, config: TaskLoadConfig | None = None
) -> MultiSeedLoadedTask:
    active = config or TaskLoadConfig()
    task_path = Path(path).resolve(strict=True)
    source = task_path.read_text(encoding="utf-8")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    module = _load_module(task_path)
    for required in ("Model", "get_init_inputs", "get_inputs"):
        if not hasattr(module, required):
            raise AttributeError(f"task source is missing {required}")
    task_id = active.task_id or task_path.stem
    variants = tuple(
        _load_parameter_variant(
            module,
            source=source,
            source_sha256=source_sha256,
            task_id=task_id,
            parameter_seed=parameter_seed,
            correctness_seeds=active.correctness_seeds,
            correctness_nonce=active.correctness_nonce,
        )
        for parameter_seed in active.parameter_seeds
    )
    return MultiSeedLoadedTask(variants=variants, task_source_sha256=source_sha256)


def _array_payload(item: NamedArray) -> dict[str, Any]:
    return {
        "name": item.name,
        "dtype": item.value.dtype.str,
        "shape": item.value.shape,
        "data": item.value.tobytes(order="C"),
    }


def _array_from_payload(payload: Mapping[str, Any]) -> NamedArray:
    dtype = np.dtype(payload["dtype"])
    shape = tuple(int(dim) for dim in payload["shape"])
    array = np.frombuffer(payload["data"], dtype=dtype).reshape(shape).copy()
    return NamedArray(str(payload["name"]), array)


def _loader_worker(connection: Any, path: str, config: TaskLoadConfig) -> None:
    try:
        loaded = load_task(path, config)
        connection.send(
            {
                "ok": True,
                "variants": [
                    {
                        "parameter_seed": variant.parameter_seed,
                        "correctness_contract": variant.correctness_contract,
                        "task": task_to_worker_payload(variant.build_task),
                        "hidden_cases": [
                            {
                                "seed": case.seed,
                                "inputs": [_array_payload(item) for item in case.inputs],
                                "reference": [
                                    _array_payload(item) for item in case.reference
                                ],
                            }
                            for case in variant.hidden_cases
                        ],
                    }
                    for variant in loaded.variants
                ],
                "task_source_sha256": loaded.task_source_sha256,
            }
        )
    except BaseException as error:
        diagnostic = traceback.format_exc()
        try:
            connection.send(
                {
                    "ok": False,
                    "message": str(error) or type(error).__name__,
                    "exception_type": type(error).__name__,
                    "diagnostic": diagnostic,
                }
            )
        except BaseException:
            pass
    finally:
        connection.close()


def load_task_isolated(
    path: str | Path,
    config: TaskLoadConfig | None = None,
    *,
    start_method: str = "spawn",
) -> MultiSeedLoadedTask:
    active = config or TaskLoadConfig()
    context = mp.get_context(start_method)
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_loader_worker,
        args=(child_connection, str(Path(path).resolve()), active),
        name="pipeline-task-loader",
        daemon=True,
    )
    process.start()
    child_connection.close()
    try:
        if not parent_connection.poll(active.timeout_seconds):
            process.terminate()
            process.join(timeout=2.0)
            raise PipelineFault(
                FaultRecord(
                    FaultStage.BUILD_EXECUTION,
                    "task_loader_timeout",
                    f"trusted task loader exceeded {active.timeout_seconds:g} seconds",
                    owner=FaultOwner.PIPELINE,
                    details={"phase": "task_loader"},
                )
            )
        payload = parent_connection.recv()
    except EOFError as error:
        raise PipelineFault(
            FaultRecord(
                FaultStage.BUILD_EXECUTION,
                "task_loader_crash",
                f"trusted task loader exited without data (exitcode={process.exitcode})",
                owner=FaultOwner.PIPELINE,
                details={"phase": "task_loader"},
            )
        ) from error
    finally:
        parent_connection.close()
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
    if not payload["ok"]:
        diagnostic = str(payload["diagnostic"])
        raise PipelineFault(
            FaultRecord(
                FaultStage.BUILD_EXECUTION,
                "task_loader_error",
                str(payload["message"]),
                owner=FaultOwner.PIPELINE,
                exception_type=str(payload["exception_type"]),
                diagnostic_sha256=diagnostic_digest(diagnostic),
                details={"phase": "task_loader"},
            ),
            diagnostic=diagnostic,
        )
    source_sha256 = str(payload["task_source_sha256"])
    return MultiSeedLoadedTask(
        variants=tuple(
            LoadedTask(
                parameter_seed=int(variant["parameter_seed"]),
                build_task=task_from_worker_payload(variant["task"]),
                hidden_cases=tuple(
                    HiddenCase(
                        seed=int(case["seed"]),
                        inputs=tuple(
                            _array_from_payload(item) for item in case["inputs"]
                        ),
                        reference=tuple(
                            _array_from_payload(item) for item in case["reference"]
                        ),
                    )
                    for case in variant["hidden_cases"]
                ),
                task_source_sha256=source_sha256,
                correctness_contract=validate_correctness_contract(
                    variant.get("correctness_contract"),
                    expected_task_source_sha256=source_sha256,
                ) if variant.get("correctness_contract") is not None else None,
            )
            for variant in payload["variants"]
        ),
        task_source_sha256=source_sha256,
    )


__all__ = [
    "CANONICAL_CORRECTNESS_SEEDS",
    "HiddenCase",
    "INPUT_IDENTITY_SCHEMA",
    "LoadedTask",
    "MultiSeedLoadedTask",
    "PLAIN_CORRECTNESS_SEEDS",
    "TaskLoadConfig",
    "correctness_input_identities",
    "input_identity_sha256",
    "load_task",
    "load_task_isolated",
]
