"""Stable model-facing API for the evaluation pipeline."""

from .build_result import BuildResult, NamedArray, RuntimePolicy
from .task_spec import BuildTask, BuildTaskSnapshot
from .tensor_spec import JSONScalar, JSONValue, TensorSpec, TreeSpec

__all__ = [
    "BuildResult",
    "BuildTask",
    "BuildTaskSnapshot",
    "JSONScalar",
    "JSONValue",
    "NamedArray",
    "RuntimePolicy",
    "TensorSpec",
    "TreeSpec",
]
