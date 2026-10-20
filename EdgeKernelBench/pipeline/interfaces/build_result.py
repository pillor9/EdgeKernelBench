"""Generated IR, parameters, and entry-point metadata for TVM deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .tensor_spec import JSONValue, _json_value


def _valid_name(name: str, *, field_name: str) -> str:
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    return name


@dataclass(frozen=True, slots=True)
class NamedArray:
    name: str
    value: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        _valid_name(self.name, field_name="NamedArray.name")
        if not isinstance(self.value, np.ndarray):
            raise TypeError("NamedArray.value must be a numpy.ndarray")
        if self.value.dtype.hasobject:
            raise TypeError("NamedArray.value cannot use object dtype")
        value = np.array(self.value, copy=True, order="C")
        value.setflags(write=False)
        object.__setattr__(self, "value", value)

    def manifest(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "shape": [int(dim) for dim in self.value.shape],
            "dtype": self.value.dtype.name,
            "nbytes": int(self.value.nbytes),
        }


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    profile_id: str

    def __post_init__(self) -> None:
        _valid_name(self.profile_id, field_name="RuntimePolicy.profile_id")

    def manifest(self) -> dict[str, JSONValue]:
        return {"profile_id": self.profile_id}


@dataclass(frozen=True, slots=True)
class BuildResult:
    mod: Any
    params: tuple[NamedArray, ...]
    input_names: tuple[str, ...]
    entry_name: str = "main"
    runtime_policy: RuntimePolicy | None = None
    notes: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.params, tuple):
            object.__setattr__(self, "params", tuple(self.params))
        if not isinstance(self.input_names, tuple):
            object.__setattr__(self, "input_names", tuple(self.input_names))
        if any(not isinstance(item, NamedArray) for item in self.params):
            raise TypeError("BuildResult.params must contain only NamedArray values")
        param_names = tuple(item.name for item in self.params)
        if len(set(param_names)) != len(param_names):
            raise ValueError("BuildResult parameter names must be unique")
        for name in self.input_names:
            _valid_name(name, field_name="BuildResult input name")
        if len(set(self.input_names)) != len(self.input_names):
            raise ValueError("BuildResult input names must be unique")
        overlap = set(param_names) & set(self.input_names)
        if overlap:
            raise ValueError(f"BuildResult input/parameter names overlap: {sorted(overlap)}")
        _valid_name(self.entry_name, field_name="BuildResult.entry_name")
        if self.runtime_policy is not None and not isinstance(self.runtime_policy, RuntimePolicy):
            raise TypeError("BuildResult.runtime_policy must be RuntimePolicy or None")
        normalized_notes = _json_value(self.notes)
        assert isinstance(normalized_notes, dict)
        object.__setattr__(self, "notes", MappingProxyType(normalized_notes))

    @property
    def argument_names(self) -> tuple[str, ...]:
        return self.input_names + tuple(item.name for item in self.params)

    def abi_manifest(self) -> dict[str, JSONValue]:
        return {
            "entry_name": self.entry_name,
            "input_names": list(self.input_names),
            "params": [item.manifest() for item in self.params],
            "argument_names": list(self.argument_names),
            "runtime_policy": (
                None if self.runtime_policy is None else self.runtime_policy.manifest()
            ),
            "notes": dict(self.notes),
        }


__all__ = ["BuildResult", "NamedArray", "RuntimePolicy"]
