"""Robust dispersion for finite, positive paired timing samples."""

from __future__ import annotations

import math
import statistics
from typing import Sequence

ROBUST_CV_LIMIT = 0.05
MINIMUM_SAMPLES = 2


def _valid(values: Sequence[float]) -> bool:
    return len(values) >= MINIMUM_SAMPLES and all(
        not isinstance(value, bool) and isinstance(value, (int, float))
        and math.isfinite(value) and value > 0 for value in values
    )


def coefficient_of_variation(values: Sequence[float]) -> float:
    if not _valid(values):
        return float("inf")
    scale = max(values)
    normalized = [value / scale for value in values]
    return statistics.pstdev(normalized) / statistics.fmean(normalized)


def robust_coefficient_of_variation(values: Sequence[float]) -> float:
    if not _valid(values):
        return float("inf")
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (
        ordered[middle - 1] + (ordered[middle] - ordered[middle - 1]) / 2
    )
    deviations = sorted(abs(value - median) for value in ordered)
    mad = deviations[middle] if len(ordered) % 2 else (
        deviations[middle - 1] + (deviations[middle] - deviations[middle - 1]) / 2
    )
    return 1.4826 * (mad / median)


def ratio_stability_eligible(
    values: Sequence[float], *, robust_limit: float = ROBUST_CV_LIMIT,
) -> tuple[bool, float, float]:
    if not math.isfinite(robust_limit) or not 0 < robust_limit <= ROBUST_CV_LIMIT:
        raise ValueError("robust_limit must be in (0, 0.05]")
    ordinary = coefficient_of_variation(values)
    robust = robust_coefficient_of_variation(values)
    return robust <= robust_limit, ordinary, robust


__all__ = ["ROBUST_CV_LIMIT", "MINIMUM_SAMPLES", "coefficient_of_variation",
           "ratio_stability_eligible", "robust_coefficient_of_variation"]
