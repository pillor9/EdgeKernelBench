"""Rank-preserving input and output conversion tests."""

import unittest

import numpy as np
import torch

from leka.tasks.arrays import as_c_contiguous
from leka.tasks.pytorch_task import PyTorchTask, _fixed_state_commitment


class TaskArrayTests(unittest.TestCase):
    def test_scalar_conversion_preserves_rank_and_dtype(self):
        value = np.array(2.5, dtype=np.float32)
        converted = as_c_contiguous(value)
        self.assertEqual(converted.shape, ())
        self.assertEqual(converted.dtype, value.dtype)
        self.assertTrue(converted.flags.c_contiguous)
        np.testing.assert_array_equal(converted, value)

    def test_strided_conversion_preserves_shape_and_values(self):
        value = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
        converted = as_c_contiguous(value)
        self.assertEqual(converted.shape, (3, 2))
        self.assertTrue(converted.flags.c_contiguous)
        np.testing.assert_array_equal(converted, value)

    def test_reference_preserves_scalar_input_and_output(self):
        class ScalarModel(torch.nn.Module):
            def forward(self, value):
                if value.ndim != 0:
                    raise ValueError("expected a scalar input")
                return value + 1

        task = PyTorchTask.__new__(PyTorchTask)
        task.model = ScalarModel()
        task.dynamic_input_names = ("value",)
        task.output_names = ("output",)
        result = task.reference({"value": np.array(2.5, dtype=np.float32)})
        self.assertEqual(result["output"].shape, ())
        self.assertEqual(result["output"].dtype, np.dtype("float32"))
        self.assertEqual(result["output"].item(), 3.5)

    def test_fixed_state_commitment_distinguishes_scalar_and_vector(self):
        scalar = np.array(2.5, dtype=np.float32)
        vector = np.array([2.5], dtype=np.float32)
        self.assertNotEqual(
            _fixed_state_commitment(("state",), (scalar,)),
            _fixed_state_commitment(("state",), (vector,)),
        )


if __name__ == "__main__":
    unittest.main()
