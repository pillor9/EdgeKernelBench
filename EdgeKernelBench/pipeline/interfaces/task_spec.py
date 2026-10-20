"""Read-only task view supplied to a generated ``build(task)`` function."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from .tensor_spec import JSONValue, TensorSpec, TreeSpec, _json_value


def _frozen_array(value: np.ndarray) -> np.ndarray:
    array = np.array(value, copy=True, order="C")
    if array.dtype.hasobject:
        raise TypeError("BuildTask arrays cannot use object dtype")
    array.setflags(write=False)
    return array


@runtime_checkable
class BuildTask(Protocol):
    """Task interface available to a generated ``build(task)`` function."""

    source_text: str
    task_id: str
    init_args: tuple[Any, ...]
    input_names: tuple[str, ...]
    input_specs: tuple[TensorSpec, ...]
    output_spec: TreeSpec
    execution_mode: str

    def parameter_names(self) -> tuple[str, ...]: ...

    def buffer_names(self) -> tuple[str, ...]: ...

    def parameter(self, name: str) -> np.ndarray: ...

    def buffer(self, name: str) -> np.ndarray: ...

    def parameter_specs(self) -> tuple[TensorSpec, ...]: ...

    def buffer_specs(self) -> tuple[TensorSpec, ...]: ...

    def metadata(self) -> dict[str, JSONValue]: ...


@dataclass(frozen=True, slots=True)
class BuildTaskSnapshot:
    """Serializable task snapshot that excludes hidden evaluation inputs and outputs."""

    source_text: str
    init_args: tuple[Any, ...]
    input_specs: tuple[TensorSpec, ...]
    output_spec: TreeSpec
    input_names: tuple[str, ...]
    task_id: str
    _parameters: Mapping[str, np.ndarray] = field(repr=False)
    _buffers: Mapping[str, np.ndarray] = field(repr=False)
    execution_mode: str = "eval"

    def __post_init__(self) -> None:
        if not isinstance(self.source_text, str) or not self.source_text.strip():
            raise ValueError("BuildTask source_text must be non-empty")
        if not isinstance(self.task_id, str) or not self.task_id or self.task_id.strip() != self.task_id:
            raise ValueError("BuildTask task_id must be a non-empty trimmed string")
        if self.execution_mode != "eval":
            raise ValueError("BuildTask execution_mode must be 'eval'")
        if not isinstance(self.init_args, tuple):
            object.__setattr__(self, "init_args", tuple(self.init_args))
        if not isinstance(self.input_specs, tuple):
            object.__setattr__(self, "input_specs", tuple(self.input_specs))
        if not isinstance(self.input_names, tuple):
            object.__setattr__(self, "input_names", tuple(self.input_names))
        if len(self.input_names) != len(self.input_specs):
            raise ValueError("BuildTask input_names and input_specs lengths differ")
        if len(set(self.input_names)) != len(self.input_names) or any(
            not isinstance(name, str) or not name or name.strip() != name
            for name in self.input_names
        ):
            raise ValueError("BuildTask input_names must be unique non-empty strings")
        parameters = {str(name): _frozen_array(value) for name, value in self._parameters.items()}
        buffers = {str(name): _frozen_array(value) for name, value in self._buffers.items()}
        if any(not name or name.strip() != name for name in (*parameters, *buffers)):
            raise ValueError("parameter and buffer names must be non-empty and trimmed")
        overlap = set(parameters) & set(buffers)
        if overlap:
            raise ValueError(f"parameter/buffer names overlap: {sorted(overlap)}")
        object.__setattr__(self, "_parameters", MappingProxyType(parameters))
        object.__setattr__(self, "_buffers", MappingProxyType(buffers))

    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self._parameters)

    def buffer_names(self) -> tuple[str, ...]:
        return tuple(self._buffers)

    def parameter(self, name: str) -> np.ndarray:
        if name not in self._parameters:
            raise KeyError(f"unknown parameter: {name}")
        return _frozen_array(self._parameters[name])

    def buffer(self, name: str) -> np.ndarray:
        if name not in self._buffers:
            raise KeyError(f"unknown buffer: {name}")
        return _frozen_array(self._buffers[name])

    def parameter_specs(self) -> tuple[TensorSpec, ...]:
        return tuple(TensorSpec.from_array(value, name=name) for name, value in self._parameters.items())

    def buffer_specs(self) -> tuple[TensorSpec, ...]:
        return tuple(TensorSpec.from_array(value, name=name) for name, value in self._buffers.items())

    def metadata(self) -> dict[str, JSONValue]:
        return {
            "task_id": self.task_id,
            "init_args": _json_value(self.init_args),
            "input_names": list(self.input_names),
            "input_specs": [spec.to_dict() for spec in self.input_specs],
            "parameters": [spec.to_dict() for spec in self.parameter_specs()],
            "buffers": [spec.to_dict() for spec in self.buffer_specs()],
            "output_spec": self.output_spec.to_dict(),
            "execution_mode": self.execution_mode,
        }


__all__ = ["BuildTask", "BuildTaskSnapshot"]
