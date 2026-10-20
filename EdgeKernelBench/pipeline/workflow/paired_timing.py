"""Backend-neutral AB/BA paired-timing coordinator and raw sample schema."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Mapping, Protocol, Sequence, Literal, runtime_checkable

from pipeline.interfaces import JSONValue
from pipeline.interfaces.tensor_spec import _json_value

from .errors import FaultRecord, FaultStage, PipelineFault
from .stability import coefficient_of_variation, robust_coefficient_of_variation


@dataclass(frozen=True, slots=True)
class TimingDomain:
    backend: str
    device_identity_sha256: str
    target: str
    runtime_policy_id: str | None
    compiler_identity_sha256: str
    runtime_identity_sha256: str
    session_id: str
    measurement_boundary: str = "relax_entry_closure"

    def __post_init__(self) -> None:
        for name in (
            "backend",
            "device_identity_sha256",
            "target",
            "compiler_identity_sha256",
            "runtime_identity_sha256",
            "session_id",
            "measurement_boundary",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"TimingDomain.{name} must be non-empty")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "backend": self.backend,
            "device_identity_sha256": self.device_identity_sha256,
            "target": self.target,
            "runtime_policy_id": self.runtime_policy_id,
            "compiler_identity_sha256": self.compiler_identity_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "session_id": self.session_id,
            "measurement_boundary": self.measurement_boundary,
        }


@dataclass(frozen=True, slots=True)
class TimingEnvironment:
    eligible: bool
    checks: Mapping[str, JSONValue] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized = _json_value(self.checks)
        assert isinstance(normalized, dict)
        object.__setattr__(self, "checks", normalized)
        if self.eligible and self.reasons:
            raise ValueError("eligible timing environment cannot carry failure reasons")
        if not self.eligible and not self.reasons:
            raise ValueError("ineligible timing environment requires a reason")


@dataclass(frozen=True, slots=True)
class TimingConfig:
    warmup_minimum_invocations: int = 1
    minimum_warmup_duration_ms_per_side: float = 0.0
    warmup_duration_basis: str = "matched_scored_measurement_budget"
    evaluator_target_window_ms: float = 100.0
    evaluator_maximum_number: int = 1_000
    outer_pairs: int = 21
    discard_first_outer_pair: bool = True
    max_rcv: float = 0.05
    first_order: Literal["AB", "BA"] = "AB"

    def __post_init__(self) -> None:
        for name in ("warmup_minimum_invocations", "evaluator_maximum_number", "outer_pairs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.discard_first_outer_pair) is not bool:
            raise ValueError("discard_first_outer_pair must be a boolean")
        if self.scored_pair_count < 2:
            raise ValueError("timing requires at least two scored pairs")
        if (
            isinstance(self.minimum_warmup_duration_ms_per_side, bool)
            or isinstance(self.evaluator_target_window_ms, bool)
            or not math.isfinite(self.minimum_warmup_duration_ms_per_side)
            or self.minimum_warmup_duration_ms_per_side < 0.0
            or not math.isfinite(self.evaluator_target_window_ms)
            or self.evaluator_target_window_ms <= 0.0
        ):
            raise ValueError("timing durations are outside valid bounds")
        if self.warmup_duration_basis != "matched_scored_measurement_budget":
            raise ValueError("warmup must match the scored measurement budget")
        if isinstance(self.max_rcv, bool) or not math.isfinite(self.max_rcv) or not 0 < self.max_rcv <= 0.05:
            raise ValueError("max_rcv must be in (0, 0.05]")
        if self.first_order not in {"AB", "BA"}:
            raise ValueError("first_order must be AB or BA")

    @property
    def scored_pair_count(self) -> int:
        return self.outer_pairs - (1 if self.discard_first_outer_pair else 0)


@dataclass(frozen=True, slots=True)
class WarmupEvidence:
    invocations: int
    elapsed_ms: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.invocations, bool)
            or not isinstance(self.invocations, int)
            or self.invocations < 1
        ):
            raise ValueError("warmup invocations must be positive")
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms <= 0.0:
            raise ValueError("warmup elapsed_ms must be finite and positive")


@runtime_checkable
class MeasurementSource(Protocol):
    """Backend adapter interface for one synchronized saved-entry evaluator."""

    def time_evaluator(self, number: int) -> float: ...

    def warmup_to_target(
        self,
        *,
        evaluator_number: int,
        target_duration_ms: float,
        minimum_invocations: int,
    ) -> WarmupEvidence: ...


@dataclass(frozen=True, slots=True)
class TimingSample:
    pair_index: int
    order: Literal["AB", "BA"]
    leg: Literal["candidate", "baseline"]
    repetition_index: int
    latency_ms: float
    discarded: bool
    sequence_index: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.latency_ms) or self.latency_ms <= 0:
            raise ValueError("latency_ms must be finite and positive")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "pair_index": self.pair_index,
            "order": self.order,
            "leg": self.leg,
            "repetition_index": self.repetition_index,
            "latency_ms": self.latency_ms,
            "discarded": self.discarded,
            "sequence_index": self.sequence_index,
        }


@dataclass(frozen=True, slots=True)
class PairedTimingResult:
    domain: TimingDomain
    environment: TimingEnvironment
    config: TimingConfig
    candidate_probe_ms: float
    baseline_probe_ms: float
    candidate_evaluator_number: int
    baseline_evaluator_number: int
    candidate_warmup_target_ms: float
    baseline_warmup_target_ms: float
    candidate_warmup: WarmupEvidence
    baseline_warmup: WarmupEvidence
    samples: tuple[TimingSample, ...]
    candidate_median_ms: float
    baseline_median_ms: float
    speedup: float
    candidate_cv: float
    baseline_cv: float
    paired_ratio_cv: float
    candidate_rcv: float
    baseline_rcv: float
    paired_ratio_rcv: float
    stable: bool
    formal_result_eligible: bool

    def to_manifest(self) -> dict[str, JSONValue]:
        return {
            "domain": self.domain.to_dict(),
            "environment": {
                "eligible": self.environment.eligible,
                "checks": dict(self.environment.checks),
                "reasons": list(self.environment.reasons),
            },
            "config": {
                "warmup_minimum_invocations": (
                    self.config.warmup_minimum_invocations
                ),
                "minimum_warmup_duration_ms_per_side": (
                    self.config.minimum_warmup_duration_ms_per_side
                ),
                "warmup_duration_basis": self.config.warmup_duration_basis,
                "evaluator_target_window_ms": (
                    self.config.evaluator_target_window_ms
                ),
                "evaluator_maximum_number": self.config.evaluator_maximum_number,
                "outer_pairs": self.config.outer_pairs,
                "scored_pair_count": self.config.scored_pair_count,
                "discard_first_outer_pair": self.config.discard_first_outer_pair,
                "max_rcv": self.config.max_rcv,
                "first_order": self.config.first_order,
            },
            "non_scored_probe": {
                "candidate_ms": self.candidate_probe_ms,
                "baseline_ms": self.baseline_probe_ms,
            },
            "evaluator_numbers": {
                "candidate": self.candidate_evaluator_number,
                "baseline": self.baseline_evaluator_number,
            },
            "warmup": {
                "duration_basis": self.config.warmup_duration_basis,
                "candidate": {
                    "target_duration_ms": self.candidate_warmup_target_ms,
                    "invocations": self.candidate_warmup.invocations,
                    "elapsed_ms": self.candidate_warmup.elapsed_ms,
                },
                "baseline": {
                    "target_duration_ms": self.baseline_warmup_target_ms,
                    "invocations": self.baseline_warmup.invocations,
                    "elapsed_ms": self.baseline_warmup.elapsed_ms,
                },
            },
            "samples": [sample.to_dict() for sample in self.samples],
            "candidate_median_ms": self.candidate_median_ms,
            "baseline_median_ms": self.baseline_median_ms,
            "speedup": self.speedup,
            "candidate_cv": self.candidate_cv,
            "baseline_cv": self.baseline_cv,
            "paired_ratio_cv": self.paired_ratio_cv,
            "candidate_rcv": self.candidate_rcv,
            "baseline_rcv": self.baseline_rcv,
            "paired_ratio_rcv": self.paired_ratio_rcv,
            "stable": self.stable,
            "formal_result_eligible": self.formal_result_eligible,
        }


def require_same_timing_domain(candidate: TimingDomain, baseline: TimingDomain) -> None:
    if candidate != baseline:
        left = candidate.to_dict()
        right = baseline.to_dict()
        differing = sorted(key for key in left if left[key] != right[key])
        raise PipelineFault(
            FaultRecord(
                FaultStage.TIMING_ENVIRONMENT,
                "timing_domain_mismatch",
                f"candidate/baseline timing domains differ: {', '.join(differing)}",
                details={"differing_fields": differing},
            )
        )


def _latency_ms(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a numeric latency")
    latency = float(value)
    if not math.isfinite(latency) or latency <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return latency


def _validate_warmup(
    evidence: WarmupEvidence,
    *,
    evaluator_number: int,
    target_duration_ms: float,
    minimum_invocations: int,
    label: str,
) -> None:
    if (
        evidence.invocations < minimum_invocations
        or evidence.invocations % evaluator_number != 0
        or evidence.elapsed_ms < target_duration_ms
    ):
        raise ValueError(f"{label} warmup did not satisfy its scored-budget target")


def run_paired_timing(
    candidate: MeasurementSource,
    baseline: MeasurementSource,
    *,
    candidate_domain: TimingDomain,
    baseline_domain: TimingDomain,
    environment: TimingEnvironment,
    config: TimingConfig | None = None,
) -> PairedTimingResult:
    active = config or TimingConfig()
    require_same_timing_domain(candidate_domain, baseline_domain)
    if not environment.eligible:
        raise PipelineFault(
            FaultRecord(
                FaultStage.TIMING_ENVIRONMENT,
                "environment_ineligible",
                "; ".join(environment.reasons),
                details={"checks": dict(environment.checks)},
            )
        )
    candidate_probe_ms = _latency_ms(
        candidate.time_evaluator(1), label="candidate probe"
    )
    baseline_probe_ms = _latency_ms(
        baseline.time_evaluator(1), label="baseline probe"
    )
    candidate_number = min(
        active.evaluator_maximum_number,
        max(1, math.ceil(active.evaluator_target_window_ms / candidate_probe_ms)),
    )
    baseline_number = min(
        active.evaluator_maximum_number,
        max(1, math.ceil(active.evaluator_target_window_ms / baseline_probe_ms)),
    )
    candidate_warmup_target_ms = max(
        active.minimum_warmup_duration_ms_per_side,
        candidate_probe_ms * candidate_number * active.scored_pair_count,
    )
    baseline_warmup_target_ms = max(
        active.minimum_warmup_duration_ms_per_side,
        baseline_probe_ms * baseline_number * active.scored_pair_count,
    )
    candidate_warmup = candidate.warmup_to_target(
        evaluator_number=candidate_number,
        target_duration_ms=candidate_warmup_target_ms,
        minimum_invocations=active.warmup_minimum_invocations,
    )
    baseline_warmup = baseline.warmup_to_target(
        evaluator_number=baseline_number,
        target_duration_ms=baseline_warmup_target_ms,
        minimum_invocations=active.warmup_minimum_invocations,
    )
    _validate_warmup(
        candidate_warmup,
        evaluator_number=candidate_number,
        target_duration_ms=candidate_warmup_target_ms,
        minimum_invocations=active.warmup_minimum_invocations,
        label="candidate",
    )
    _validate_warmup(
        baseline_warmup,
        evaluator_number=baseline_number,
        target_duration_ms=baseline_warmup_target_ms,
        minimum_invocations=active.warmup_minimum_invocations,
        label="baseline",
    )
    samples: list[TimingSample] = []
    sequence_index = 0
    for pair_index in range(active.outer_pairs):
        starts_ab = active.first_order == "AB"
        order: Literal["AB", "BA"] = (
            "AB" if ((pair_index % 2 == 0) == starts_ab) else "BA"
        )
        legs = (
            (("candidate", candidate), ("baseline", baseline))
            if order == "AB"
            else (("baseline", baseline), ("candidate", candidate))
        )
        for leg, source in legs:
            evaluator_number = (
                candidate_number if leg == "candidate" else baseline_number
            )
            value = _latency_ms(
                source.time_evaluator(evaluator_number),
                label=f"{leg} scored measurement",
            )
            samples.append(
                TimingSample(
                    pair_index=pair_index,
                    order=order,
                    leg=leg,  # type: ignore[arg-type]
                    repetition_index=0,
                    latency_ms=value,
                    discarded=active.discard_first_outer_pair and pair_index == 0,
                    sequence_index=sequence_index,
                )
            )
            sequence_index += 1
    kept_candidate = [
        sample.latency_ms
        for sample in samples
        if sample.leg == "candidate" and not sample.discarded
    ]
    kept_baseline = [
        sample.latency_ms
        for sample in samples
        if sample.leg == "baseline" and not sample.discarded
    ]
    candidate_median = statistics.median(kept_candidate)
    baseline_median = statistics.median(kept_baseline)
    kept_pairs = [
        sample
        for sample in samples
        if sample.leg == "candidate" and not sample.discarded
    ]
    paired_ratios = [
        baseline_sample.latency_ms / candidate_sample.latency_ms
        for candidate_sample, baseline_sample in zip(
            kept_pairs,
            [
                sample
                for sample in samples
                if sample.leg == "baseline" and not sample.discarded
            ],
            strict=True,
        )
    ]
    candidate_cv = coefficient_of_variation(kept_candidate)
    baseline_cv = coefficient_of_variation(kept_baseline)
    paired_ratio_cv = coefficient_of_variation(paired_ratios)
    candidate_rcv = robust_coefficient_of_variation(kept_candidate)
    baseline_rcv = robust_coefficient_of_variation(kept_baseline)
    paired_ratio_rcv = robust_coefficient_of_variation(paired_ratios)
    stable = all(value <= active.max_rcv for value in (
        candidate_rcv, baseline_rcv, paired_ratio_rcv
    ))
    return PairedTimingResult(
        domain=candidate_domain,
        environment=environment,
        config=active,
        candidate_probe_ms=candidate_probe_ms,
        baseline_probe_ms=baseline_probe_ms,
        candidate_evaluator_number=candidate_number,
        baseline_evaluator_number=baseline_number,
        candidate_warmup_target_ms=candidate_warmup_target_ms,
        baseline_warmup_target_ms=baseline_warmup_target_ms,
        candidate_warmup=candidate_warmup,
        baseline_warmup=baseline_warmup,
        samples=tuple(samples),
        candidate_median_ms=candidate_median,
        baseline_median_ms=baseline_median,
        speedup=statistics.median(paired_ratios),
        candidate_cv=candidate_cv,
        baseline_cv=baseline_cv,
        paired_ratio_cv=paired_ratio_cv,
        candidate_rcv=candidate_rcv,
        baseline_rcv=baseline_rcv,
        paired_ratio_rcv=paired_ratio_rcv,
        stable=stable,
        formal_result_eligible=stable,
    )


def require_stable_timing(result: PairedTimingResult) -> None:
    candidate = [s.latency_ms for s in result.samples if s.leg == "candidate" and not s.discarded]
    baseline = [s.latency_ms for s in result.samples if s.leg == "baseline" and not s.discarded]
    ratios = [b / c for c, b in zip(candidate, baseline)] if len(candidate) == len(baseline) else []
    dispersions = {
        "candidate_rcv": robust_coefficient_of_variation(candidate),
        "baseline_rcv": robust_coefficient_of_variation(baseline),
        "paired_ratio_rcv": robust_coefficient_of_variation(ratios),
    }
    if not all(value <= result.config.max_rcv for value in dispersions.values()):
        raise PipelineFault(FaultRecord(
            FaultStage.TIMING_STABILITY,
            "unstable_samples",
            f"timing requires all three rCV values <= {result.config.max_rcv:g}",
            details={**dispersions, "max_rcv": result.config.max_rcv},
        ))


__all__ = [
    "MeasurementSource",
    "PairedTimingResult",
    "TimingConfig",
    "TimingDomain",
    "TimingEnvironment",
    "TimingSample",
    "WarmupEvidence",
    "require_same_timing_domain",
    "require_stable_timing",
    "run_paired_timing",
]
