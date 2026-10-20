"""Check worker output comparisons without loading TVM or contacting a device."""

import unittest
import warnings

import numpy as np

from leka.evaluation import android_worker, device_worker


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class CorrectnessOutputTests(unittest.TestCase):
    def compare(self, worker, actual, expected, *, atol=1e-4, rtol=1e-4):
        reports, passed = worker.summarize_correctness(
            Tensor(actual), ["output"], {"output": np.asarray(expected)},
            atol=atol, rtol=rtol,
        )
        return reports["output"], passed

    def test_discrete_outputs_require_exact_values(self):
        examples = [
            (np.array([100001], dtype=np.int64), np.array([100000], dtype=np.int64), 1),
            (np.array([2**63 + 1], dtype=np.uint64), np.array([2**63], dtype=np.uint64), 1),
            (np.array([255], dtype=np.uint8), np.array([0], dtype=np.uint8), 255),
            (np.array([-128], dtype=np.int8), np.array([127], dtype=np.int8), 255),
            (np.array([False]), np.array([True]), 1),
        ]
        for worker in (android_worker, device_worker):
            for actual, expected, error in examples:
                with self.subTest(worker=worker.__name__, dtype=actual.dtype, actual=actual):
                    report, passed = self.compare(worker, actual, expected)
                    self.assertFalse(passed)
                    self.assertFalse(report["pass"])
                    self.assertEqual(report["max_abs_error"], error)

    def test_equal_boolean_and_integer_outputs_pass(self):
        for worker in (android_worker, device_worker):
            for value in (np.array([True, False]), np.array([2**63 - 1], dtype=np.int64),
                          np.array([], dtype=np.bool_), np.array(3, dtype=np.int32)):
                with self.subTest(worker=worker.__name__, value=value):
                    report, passed = self.compare(worker, value, value)
                    self.assertTrue(passed)
                    self.assertEqual(report["max_abs_error"], 0)

    def test_scalar_signed_extrema_do_not_warn(self):
        minimum = np.array(-(2**63), dtype=np.int64)
        maximum = np.array(2**63 - 1, dtype=np.int64)
        for worker in (android_worker, device_worker):
            for actual, expected in ((minimum, maximum), (maximum, minimum)):
                with self.subTest(worker=worker.__name__, actual=actual):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        report, passed = self.compare(worker, actual, expected)
                    self.assertFalse(passed)
                    self.assertEqual(report["max_abs_error"], float(2**64 - 1))

    def test_float_tolerances_are_preserved(self):
        for worker in (android_worker, device_worker):
            for actual, passed in ((1.00005, True), (1.001, False)):
                with self.subTest(worker=worker.__name__, actual=actual):
                    _, result = self.compare(
                        worker, np.array([actual]), np.array([1.0]), atol=0, rtol=1e-4,
                    )
                    self.assertEqual(result, passed)

    def test_nonfinite_outputs_fail_even_when_equal(self):
        for worker in (android_worker, device_worker):
            for actual, expected in ((np.nan, np.nan), (np.inf, np.inf),
                                     (-np.inf, -np.inf), (np.inf, 1.0), (1.0, np.inf)):
                with self.subTest(worker=worker.__name__, actual=actual, expected=expected):
                    _, passed = self.compare(worker, np.array([actual]), np.array([expected]))
                    self.assertFalse(passed)

    def test_shape_and_dtype_must_match_without_broadcasting(self):
        for worker in (android_worker, device_worker):
            for actual, expected in (
                (np.ones((1, 2)), np.ones(2)),
                (np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float64)),
                (np.array(True), np.array([True])),
            ):
                with self.subTest(worker=worker.__name__, actual=actual, expected=expected):
                    _, passed = self.compare(worker, actual, expected)
                    self.assertFalse(passed)

    def test_nested_output_leaf_order_and_count(self):
        for worker in (android_worker, device_worker):
            with self.subTest(worker=worker.__name__):
                value = (Tensor(np.array([1], dtype=np.int64)),
                         [Tensor(np.array([True]))])
                expected = {"first": np.array([1], dtype=np.int64),
                            "second": np.array([True])}
                reports, passed = worker.summarize_correctness(
                    value, ["first", "second"], expected, atol=1e-4, rtol=1e-4,
                )
                self.assertTrue(passed)
                self.assertEqual(list(reports), ["first", "second"])
                with self.assertRaisesRegex(RuntimeError, "output leaves"):
                    worker.summarize_correctness(
                        value, ["first"], expected, atol=1e-4, rtol=1e-4,
                    )


if __name__ == "__main__":
    unittest.main()
