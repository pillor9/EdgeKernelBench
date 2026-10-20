"""Validate raw timing and enforce the robust dispersion criterion."""

import math
import statistics


def _positive(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _nonnegative(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _median(values):
    """Finite positive timing values need not have a representable sum.

    Log-ratios also use this helper; their finite range bounds the subtraction.
    """
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    lower, upper = ordered[middle - 1:middle + 1]
    return lower + (upper - lower) / 2


ROBUST_CV_LIMIT = 0.05
MINIMUM_SAMPLES = 2


def robust_cv(values):
    """Return 1.4826 * MAD / median, rejecting invalid or insufficient data."""
    if len(values) < MINIMUM_SAMPLES or any(not _positive(value) for value in values):
        return float("inf")
    median = _median(values)
    return 1.4826 * (_median(abs(value - median) for value in values) / median)


def paired_stability(candidate, baseline):
    """Assess both latency series and paired baseline/candidate speedups."""
    valid = (len(candidate) == len(baseline) and len(candidate) >= MINIMUM_SAMPLES
             and all(_positive(value) for value in (*candidate, *baseline)))
    speedups = [b / c for c, b in zip(candidate, baseline)] if valid else []
    values = {
        "candidate_rcv": robust_cv(candidate),
        "baseline_rcv": robust_cv(baseline),
        "paired_speedup_rcv": robust_cv(speedups),
    }
    stable = valid and all(value <= ROBUST_CV_LIMIT for value in values.values())
    return {"stable": stable, "limit": ROBUST_CV_LIMIT, "comparison": "less_than_or_equal",
            "minimum_samples": MINIMUM_SAMPLES,
            **{key: value if math.isfinite(value) else None for key, value in values.items()}}


def _single_arm_diagnostics(batches):
    """Describe a validated ordered sample series without changing admission.

    Log differences avoid dividing extreme positive finite latencies. These
    within-session summaries cannot identify DVFS, affinity, or any other cause.
    """
    threshold = math.log(1.1)
    log_std = statistics.pstdev(math.log(value) for value in batches)
    diagnostics = {
        "schema_version": "leka.single_arm_timing_diagnostics.v1",
        "scored_samples": len(batches),
        "log_latency_std": log_std,
        "policy": {
            "log_latency_std_warning_threshold": threshold,
            "temporal_factor_warning_threshold": 1.1,
            "meaning": "descriptive_10pct_heuristics_not_significance_or_promotion_veto",
            "causal_attribution": "not_established; no_DVFS_or_affinity_cause_inferred",
        },
        "temporal_halves": {
            "status": "not_assessed",
            "reason": "fewer_than_4_samples",
            "minimum_samples": 4,
        },
    }
    warnings = []
    if log_std > threshold:
        warnings.append("single_arm_dispersion_exceeds_10pct_heuristic")
    if len(batches) >= 4:
        half = len(batches) // 2
        first, last = _median(batches[:half]), _median(batches[-half:])
        # Positive means the earlier half is slower. Reversing time negates
        # this value but preserves its absolute value and symmetric factor.
        signed_log = math.log(first) - math.log(last)
        effect = abs(signed_log)
        try:
            factor = math.exp(effect)
        except OverflowError:
            factor = None
        diagnostics["temporal_halves"] = {
            "status": "available",
            "minimum_samples": 4,
            "samples_per_half": half,
            "order_source": "latency_batches_ms_as_recorded",
            "odd_middle_policy": "exclude_middle_sample_from_temporal_halves_only",
            "excluded_middle_index": half if len(batches) % 2 else None,
            "first_half_median_ms": first,
            "last_half_median_ms": last,
            "signed_log_first_over_last": signed_log,
            "absolute_log_factor": effect,
            "symmetric_factor": factor,
            "symmetric_factor_overflow": factor is None,
        }
        if effect > threshold:
            warnings.append("single_arm_temporal_shift_exceeds_10pct_heuristic")
    return diagnostics, warnings


def assess_timing(raw: dict, *, paired: bool) -> dict:
    report = {
        "schema_version": "leka.timing_quality.v3",
        "valid": True,
        "stable": False,
        "warnings": [],
        "statistical_claim": "descriptive_single_session_not_independent_replication",
        "promotion_floor_is_significance_test": False,
        "validation_scope": "within_session_raw_timing_consistency",
        "cross_context_consistency": "not_assessed",
        "execution_environment_stability": "not_established_by_timing_consistency",
        "variability_diagnostic_policy": {
            "order_factor_warning_threshold": 1.1,
            "log_ratio_std_warning_threshold": math.log(1.1),
            "meaning": "descriptive_10pct_heuristics_not_significance_or_promotion_veto",
        },
    }
    errors = []
    if not _positive(raw.get("latency_ms")):
        errors.append("latency_must_be_finite_positive")
    batches = raw.get("latency_batches_ms")
    if batches is not None and (not isinstance(batches, list) or not batches or
                                any(not _positive(x) for x in batches)):
        errors.append("invalid_latency_batches")
    protocol = raw.get("timing_protocol")
    if not isinstance(protocol, dict):
        protocol = {}
    if not paired and not errors:
        if batches is None:
            errors.append("latency_batches_missing")
        else:
            if not math.isclose(raw["latency_ms"], _median(batches), rel_tol=1e-7, abs_tol=5.1e-10):
                errors.append("latency_summary_does_not_match_raw_batches")
            declared = protocol.get("scored_pair_count")
            if declared is not None and (type(declared) is not int or declared != len(batches)):
                errors.append("sample_count_does_not_match_protocol")
            if len(batches) < MINIMUM_SAMPLES:
                errors.append("insufficient_latency_samples")
            if not errors:
                dispersion = robust_cv(batches)
                report["stability"] = {"candidate_rcv": dispersion if math.isfinite(dispersion) else None,
                                       "limit": ROBUST_CV_LIMIT, "comparison": "less_than_or_equal"}
                report["stable"] = dispersion <= ROBUST_CV_LIMIT
                diagnostics, warnings = _single_arm_diagnostics(batches)
                report["single_arm_diagnostics"] = diagnostics
                report["warnings"].extend(warnings)
    if paired:
        vs = raw.get("vs_base")
        if not isinstance(vs, dict) or not _positive(vs.get("median_ratio")):
            errors.append("invalid_paired_ratio")
        samples = raw.get("timing_samples")
        if samples is None:
            errors.append("paired_samples_missing")
        elif not isinstance(samples, list) or not samples:
            errors.append("invalid_timing_samples")
        else:
            scored = []
            for s in samples:
                if not isinstance(s, dict) or type(s.get("discarded")) is not bool:
                    errors.append("invalid_sample_record")
                    break
                c, r, ratio = (s.get(k) for k in ("candidate_ms", "reference_ms", "ratio"))
                if not all(_positive(x) for x in (c, r, ratio)) or not math.isclose(c / r, ratio, rel_tol=1e-7):
                    errors.append("sample_ratio_does_not_match_latencies")
                    break
                if s.get("order") not in ("AB", "BA"):
                    errors.append("invalid_pair_order")
                    break
                if not s["discarded"]:
                    scored.append(s)
            if not errors and len(scored) < MINIMUM_SAMPLES:
                errors.append("insufficient_scored_pairs")
            if not errors and scored:
                report["stability"] = paired_stability(
                    [sample["candidate_ms"] for sample in scored],
                    [sample["reference_ms"] for sample in scored],
                )
                report["stable"] = report["stability"]["stable"]
                declared = protocol.get("scored_pair_count")
                discarded = protocol.get("discard_prefix_pair_count")
                if declared is not None and (type(declared) is not int or declared != len(scored)):
                    errors.append("sample_count_does_not_match_protocol")
                if discarded is not None and (type(discarded) is not int or
                        discarded != len(samples) - len(scored)):
                    errors.append("discard_count_does_not_match_protocol")
                ratios = [s["ratio"] for s in scored]
                median = _median(ratios)
                if (type(vs.get("pairs")) is not int or vs["pairs"] != len(scored)
                        or not math.isclose(vs["median_ratio"], median, rel_tol=1e-7)):
                    errors.append("paired_summary_does_not_match_raw_samples")
                if not math.isclose(raw["latency_ms"], _median(s["candidate_ms"] for s in scored), rel_tol=1e-7):
                    errors.append("latency_summary_does_not_match_raw_samples")
                reference_median = _median(s["reference_ms"] for s in scored)
                if (not _positive(vs.get("base_latency_ms")) or
                        not math.isclose(vs["base_latency_ms"], reference_median, rel_tol=1e-7)):
                    errors.append("reference_summary_does_not_match_raw_samples")
                if batches is not None and (len(batches) != len(scored) or any(
                        not math.isclose(value, sample["candidate_ms"], rel_tol=1e-7, abs_tol=5.1e-10)
                        for value, sample in zip(batches, scored))):
                    errors.append("latency_batches_do_not_match_raw_samples")
                cv = vs.get("ratio_cv")
                expected_cv = statistics.pstdev(ratios) / statistics.mean(ratios)
                log_ratios = [math.log(r) for r in ratios]
                log_std = statistics.pstdev(log_ratios)
                report.update(paired_ratio_cv_recomputed=expected_cv,
                              paired_log_ratio_std=log_std)
                if log_std > math.log(1.1):
                    report["warnings"].append("paired_ratio_dispersion_exceeds_10pct_heuristic")
                if (not _nonnegative(cv)
                        or not math.isclose(cv, expected_cv, rel_tol=1e-7, abs_tol=1e-12)):
                    errors.append("dispersion_summary_does_not_match_raw_samples")
                by_order = {o: [s["ratio"] for s in scored if s["order"] == o] for o in ("AB", "BA")}
                medians = {o: _median(v) for o, v in by_order.items() if v}
                report.update(scored_pairs=len(scored), order_median_ratios=medians,
                              paired_ratio_min=min(ratios), paired_ratio_max=max(ratios),
                              fraction_pairs_faster=sum(r < 1 for r in ratios) / len(ratios))
                if len(medians) != 2:
                    report["warnings"].append("missing_counterbalanced_order")
                else:
                    delta = abs(medians["AB"] - medians["BA"])
                    report["order_effect_ratio"] = delta
                    # A raw ratio delta can hide a large relative order effect
                    # when both ratios are far below 1. Use logs so exchanging
                    # candidate/reference keeps this diagnostic unchanged.
                    order_logs = {o: _median(math.log(r) for r in values)
                                  for o, values in by_order.items()}
                    order_log_effect = abs(order_logs["AB"] - order_logs["BA"])
                    try:
                        order_factor = math.exp(order_log_effect)
                    except OverflowError:
                        order_factor = None
                    report.update(order_effect_log_ratio=order_log_effect,
                                  order_effect_factor=order_factor,
                                  order_effect_factor_overflow=order_factor is None)
                    if order_log_effect > math.log(1.1):
                        report["warnings"].append("pair_order_scale_effect_exceeds_10pct_heuristic")
                    if (medians["AB"] < 1) != (medians["BA"] < 1):
                        report["warnings"].append("speedup_direction_changes_with_pair_order")
                    if delta > max(0.005, abs(1 - median)):
                        report["warnings"].append("order_effect_exceeds_reported_gain")
                if len(scored) < 10:
                    report["warnings"].append("few_scored_pairs")
            elif not errors:
                errors.append("no_scored_pairs")
    report["errors"] = errors
    report["valid"] = not errors
    report["stable"] = report["stable"] and report["valid"]
    return report
