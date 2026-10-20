"""Small, backend-neutral tensor and output-tree contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np


JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
TreeKind: TypeAlias = Literal["tensor", "tuple", "list", "dict"]


def _json_value(value: Any, *, path: str = "$") -> JSONValue:
    """Return a detached JSON value or fail with an exact field path."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"non-finite JSON number at {path}")
        return float(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"JSON object key at {path} must be a non-empty string")
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    raise TypeError(f"value at {path} is not JSON-serializable: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """One fixed-shape tensor visible in the model-facing task contract."""

    shape: tuple[int, ...]
    dtype: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple):
            object.__setattr__(self, "shape", tuple(self.shape))
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in self.shape):
            raise ValueError("TensorSpec dimensions must be non-negative integers")
        try:
            normalized_dtype = np.dtype(self.dtype).name
        except TypeError as error:
            raise ValueError(f"unsupported NumPy dtype: {self.dtype!r}") from error
        object.__setattr__(self, "dtype", normalized_dtype)
        if self.name is not None and (
            not isinstance(self.name, str) or not self.name or self.name.strip() != self.name
        ):
            raise ValueError("TensorSpec.name must be a non-empty trimmed string")

    @classmethod
    def from_array(cls, value: np.ndarray, *, name: str | None = None) -> "TensorSpec":
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("object arrays are not valid tensor values")
        return cls(tuple(int(dim) for dim in array.shape), array.dtype.name, name)

    def to_dict(self) -> dict[str, JSONValue]:
        result: dict[str, JSONValue] = {
            "shape": [int(dim) for dim in self.shape],
            "dtype": self.dtype,
        }
        if self.name is not None:
            result["name"] = self.name
        return result


@dataclass(frozen=True, slots=True)
class TreeSpec:
    """Tensor-only output pytree shape without any hidden output values."""

    kind: TreeKind
    tensor: TensorSpec | None = None
    children: tuple["TreeSpec", ...] = ()
    keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"tensor", "tuple", "list", "dict"}:
            raise ValueError(f"unsupported TreeSpec kind: {self.kind!r}")
        if not isinstance(self.children, tuple):
            object.__setattr__(self, "children", tuple(self.children))
        if not isinstance(self.keys, tuple):
            object.__setattr__(self, "keys", tuple(self.keys))
        if self.kind == "tensor":
            if self.tensor is None or self.children or self.keys:
                raise ValueError("tensor TreeSpec requires tensor and no children/keys")
            return
        if self.tensor is not None:
            raise ValueError("container TreeSpec cannot carry a tensor")
        if self.kind == "dict":
            if len(self.keys) != len(self.children):
                raise ValueError("dict TreeSpec requires one key per child")
            if len(set(self.keys)) != len(self.keys) or any(not key for key in self.keys):
                raise ValueError("dict TreeSpec keys must be unique and non-empty")
        elif self.keys:
            raise ValueError("only dict TreeSpec may carry keys")

    @classmethod
    def tensor_leaf(cls, spec: TensorSpec) -> "TreeSpec":
        return cls(kind="tensor", tensor=spec)

    def tensor_leaves(self) -> tuple[TensorSpec, ...]:
        if self.kind == "tensor":
            assert self.tensor is not None
            return (self.tensor,)
        return tuple(leaf for child in self.children for leaf in child.tensor_leaves())

    def to_dict(self) -> dict[str, JSONValue]:
        if self.kind == "tensor":
            assert self.tensor is not None
            return {"kind": "tensor", "tensor": self.tensor.to_dict()}
        result: dict[str, JSONValue] = {
            "kind": self.kind,
            "children": [child.to_dict() for child in self.children],
        }
        if self.kind == "dict":
            result["keys"] = list(self.keys)
        return result


__all__ = ["JSONScalar", "JSONValue", "TensorSpec", "TreeKind", "TreeSpec", "_json_value"]
