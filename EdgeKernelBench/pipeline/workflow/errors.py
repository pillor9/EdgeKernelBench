"""Stable, backend-neutral failure attribution for the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from typing import Any, Mapping

from pipeline.interfaces.tensor_spec import JSONValue, _json_value


class FaultStage(str, Enum):
    PROVIDER_TRANSPORT = "provider_transport"
    RESPONSE_SOURCE = "response_source"
    PYTHON_SYNTAX = "python_syntax"
    SANDBOX_SECURITY = "sandbox_security"
    BUILD_EXECUTION = "build_execution"
    TVM_IR_CONSTRUCTION = "tvm_ir_construction"
    TVM_WELL_FORMED = "tvm_well_formed"
    ENTRY_ABI = "entry_abi"
    TVM_COMPILE = "tvm_compile"
    BACKEND_DRIVER_COMPILE = "backend_driver_compile"
    DEVICE_LOAD = "device_load"
    DEVICE_INVOKE = "device_invoke"
    CORRECTNESS = "correctness"
    TIMING_ENVIRONMENT = "timing_environment"
    TIMING_STABILITY = "timing_stability"
    # Controller stages for exceptions without a more specific fault.
    HOST_BUILD = "host_build"
    BASELINE_REPLAY = "baseline_replay"
    DEVICE_EXECUTION = "device_execution"


class FaultOwner(str, Enum):
    MODEL = "model"
    PROVIDER = "provider"
    PIPELINE = "pipeline"
    TVM = "tvm"
    BACKEND = "backend"
    DEVICE_ENVIRONMENT = "device_environment"
    INDETERMINATE = "indeterminate"
    BASELINE = "baseline"


_DEFAULT_OWNER = {
    FaultStage.PROVIDER_TRANSPORT: FaultOwner.PROVIDER,
    FaultStage.RESPONSE_SOURCE: FaultOwner.MODEL,
    FaultStage.PYTHON_SYNTAX: FaultOwner.MODEL,
    FaultStage.SANDBOX_SECURITY: FaultOwner.MODEL,
    FaultStage.BUILD_EXECUTION: FaultOwner.MODEL,
    FaultStage.TVM_IR_CONSTRUCTION: FaultOwner.MODEL,
    FaultStage.TVM_WELL_FORMED: FaultOwner.MODEL,
    FaultStage.ENTRY_ABI: FaultOwner.MODEL,
    FaultStage.TVM_COMPILE: FaultOwner.TVM,
    FaultStage.BACKEND_DRIVER_COMPILE: FaultOwner.BACKEND,
    FaultStage.DEVICE_LOAD: FaultOwner.DEVICE_ENVIRONMENT,
    FaultStage.DEVICE_INVOKE: FaultOwner.INDETERMINATE,
    FaultStage.CORRECTNESS: FaultOwner.MODEL,
    FaultStage.TIMING_ENVIRONMENT: FaultOwner.DEVICE_ENVIRONMENT,
    FaultStage.TIMING_STABILITY: FaultOwner.DEVICE_ENVIRONMENT,
    FaultStage.HOST_BUILD: FaultOwner.MODEL,
    FaultStage.BASELINE_REPLAY: FaultOwner.BASELINE,
    FaultStage.DEVICE_EXECUTION: FaultOwner.INDETERMINATE,
}


def diagnostic_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass(frozen=True, slots=True)
class FaultRecord:
    stage: FaultStage
    code: str
    message: str
    owner: FaultOwner | None = None
    exception_type: str | None = None
    diagnostic_sha256: str | None = None
    details: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.stage, FaultStage):
            object.__setattr__(self, "stage", FaultStage(self.stage))
        if not self.code or not self.code.replace("_", "").isalnum():
            raise ValueError("fault code must be a non-empty alphanumeric snake-case token")
        if not self.message or not self.message.strip():
            raise ValueError("fault message must be non-empty")
        if self.owner is None:
            object.__setattr__(self, "owner", _DEFAULT_OWNER[self.stage])
        elif not isinstance(self.owner, FaultOwner):
            object.__setattr__(self, "owner", FaultOwner(self.owner))
        if self.diagnostic_sha256 is not None and (
            len(self.diagnostic_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.diagnostic_sha256)
        ):
            raise ValueError("diagnostic_sha256 must be a lowercase SHA-256 digest")
        normalized = _json_value(self.details)
        assert isinstance(normalized, dict)
        object.__setattr__(self, "details", normalized)

    @classmethod
    def from_exception(
        cls,
        stage: FaultStage,
        code: str,
        error: BaseException,
        *,
        diagnostic: str | None = None,
        owner: FaultOwner | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> "FaultRecord":
        return cls(
            stage=stage,
            code=code,
            message=str(error) or type(error).__name__,
            owner=owner,
            exception_type=type(error).__name__,
            diagnostic_sha256=(None if diagnostic is None else diagnostic_digest(diagnostic)),
            details={} if details is None else details,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "stage": self.stage.value,
            "code": self.code,
            "message": self.message,
            "owner": self.owner.value if self.owner is not None else "indeterminate",
            "exception_type": self.exception_type,
            "diagnostic_sha256": self.diagnostic_sha256,
            "details": dict(self.details),
        }


class PipelineFault(RuntimeError):
    """Exception carrying a structured fault without discarding the diagnostic."""

    def __init__(self, record: FaultRecord, *, diagnostic: str | None = None) -> None:
        super().__init__(f"{record.stage.value}/{record.code}: {record.message}")
        self.record = record
        self.diagnostic = diagnostic


__all__ = [
    "FaultOwner",
    "FaultRecord",
    "FaultStage",
    "PipelineFault",
    "diagnostic_digest",
]
