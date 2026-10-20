"""Baseline provenance and replay interfaces; no tuning occurs in this module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable

from pipeline.interfaces import JSONValue
from pipeline.interfaces.tensor_spec import _json_value

from .paired_timing import TimingDomain, require_same_timing_domain


class BaselineClass(str, Enum):
    METASCHEDULE_EXACT = "metaschedule_exact"
    METASCHEDULE_TRANSFER = "metaschedule_transfer"
    DLIGHT = "dlight"
    ZERO_TUNE = "zero_tune"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class BaselineDescriptor:
    baseline_id: str
    source_kind: str
    record_target: str | None
    current_target: str
    source_artifact_sha256: str
    historical_latency_ms: float | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("baseline_id", "source_kind", "current_target", "source_artifact_sha256"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"BaselineDescriptor.{name} must be non-empty")
        normalized = _json_value(self.metadata)
        assert isinstance(normalized, dict)
        object.__setattr__(self, "metadata", normalized)

    @property
    def classification(self) -> BaselineClass:
        kind = self.source_kind.lower()
        if kind == "metaschedule":
            return (
                BaselineClass.METASCHEDULE_EXACT
                if self.record_target == self.current_target
                else BaselineClass.METASCHEDULE_TRANSFER
            )
        if kind == "dlight":
            return BaselineClass.DLIGHT
        if kind in {"zero_tune", "zero-tune"}:
            return BaselineClass.ZERO_TUNE
        return BaselineClass.OTHER


@dataclass(frozen=True, slots=True)
class BaselineReplayArtifact:
    descriptor: BaselineDescriptor
    timing_domain: TimingDomain
    compiled_module_sha256: str
    compiler_diagnostic_sha256: str
    replayed_in_current_environment: bool
    formal_baseline_eligible: bool
    ineligibility_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.formal_baseline_eligible and (
            not self.replayed_in_current_environment or self.ineligibility_reasons
        ):
            raise ValueError("formal baseline must be current-environment replay with no reasons")

    def require_pairable(self, candidate_domain: TimingDomain) -> None:
        if not self.formal_baseline_eligible:
            raise ValueError(
                "baseline is not formal-pair eligible: "
                + "; ".join(self.ineligibility_reasons or ("unqualified baseline",))
            )
        require_same_timing_domain(candidate_domain, self.timing_domain)


@runtime_checkable
class BaselineReplayAdapter(Protocol):
    def replay(
        self, descriptor: BaselineDescriptor, *, timing_domain: TimingDomain
    ) -> BaselineReplayArtifact: ...


__all__ = [
    "BaselineClass",
    "BaselineDescriptor",
    "BaselineReplayAdapter",
    "BaselineReplayArtifact",
]
