"""Verify the paper's inclusive rCV threshold and raw-sample admission."""

import copy
import math
import unittest
from unittest.mock import patch

from leka.evaluation.timing_quality import assess_timing, paired_stability


def paired_record():
    samples = [
        {"pair_index": index, "order": "AB" if index % 2 == 0 else "BA",
         "candidate_ms": 1.0, "reference_ms": 2.0, "ratio": 0.5,
         "discarded": index == 0}
        for index in range(5)
    ]
    return {
        "latency_ms": 1.0,
        "latency_batches_ms": [1.0] * 4,
        "timing_samples": samples,
        "timing_protocol": {"scored_pair_count": 4, "discard_prefix_pair_count": 1},
        "vs_base": {"median_ratio": 0.5, "ratio_cv": 0.0, "pairs": 4,
                    "base_latency_ms": 2.0},
    }


class TimingPolicyTests(unittest.TestCase):
    def test_five_percent_is_inclusive_in_both_modes(self):
        for dispersion, accepted in ((0.05, True), (math.nextafter(0.05, math.inf), False)):
            for paired in (False, True):
                with self.subTest(dispersion=dispersion, paired=paired), patch(
                    "leka.evaluation.timing_quality.robust_cv", return_value=dispersion,
                ):
                    raw = paired_record()
                    if not paired:
                        raw.pop("vs_base")
                        raw.pop("timing_samples")
                    report = assess_timing(raw, paired=paired)
                    self.assertTrue(report["valid"])
                    self.assertEqual(report["stable"], accepted)
                    self.assertEqual(report["stability"]["comparison"], "less_than_or_equal")

    def test_each_latency_and_speedup_sequence_must_be_stable(self):
        above = math.nextafter(0.05, math.inf)
        for values in ((above, 0, 0), (0, above, 0), (0, 0, above)):
            with self.subTest(values=values), patch(
                "leka.evaluation.timing_quality.robust_cv", side_effect=values,
            ):
                self.assertFalse(paired_stability([1.0, 1.0], [2.0, 2.0])["stable"])

    def test_invalid_raw_measurements_never_qualify(self):
        for values in ([], [1.0], [0.0, 1.0], [-1.0, 1.0],
                       [float("nan"), 1.0], [float("inf"), 1.0], [True, 1.0]):
            with self.subTest(values=values):
                self.assertFalse(paired_stability(values, [2.0] * len(values))["stable"])

    def test_reported_summaries_cannot_override_raw_samples(self):
        mutations = (
            lambda raw: raw["timing_samples"][1].update(candidate_ms=float("nan")),
            lambda raw: raw["timing_samples"][1].update(candidate_ms=0.0),
            lambda raw: raw["timing_samples"][1].update(ratio=0.6),
            lambda raw: raw["vs_base"].update(median_ratio=0.4),
            lambda raw: raw.update(latency_ms=0.9),
            lambda raw: raw["timing_protocol"].update(scored_pair_count=10),
        )
        for mutate in mutations:
            raw = copy.deepcopy(paired_record())
            mutate(raw)
            raw["stability"] = {"stable": True}
            with self.subTest(mutation=mutate):
                report = assess_timing(raw, paired=True)
                self.assertFalse(report["valid"])
                self.assertFalse(report["stable"])


if __name__ == "__main__":
    unittest.main()
