"""Reference semantics for task arguments beyond the built-in profiles."""

import ast
import importlib.util
import itertools
import math
from pathlib import Path
import unittest

if importlib.util.find_spec("torch"):
    import torch
else:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class OneHotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "tasks/level1/096_onehot.py.tpl"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
        ]
        namespace = {}
        exec(compile(tree, str(path), "exec"), namespace)
        cls.model_class = namespace["Model"]

    def check_axes(self, on_value, off_value):
        depth = 4
        for shape in ((), (3,), (2, 3), (2, 1, 3)):
            indices = (torch.arange(math.prod(shape)) % depth).reshape(shape)
            output_rank = indices.ndim + 1
            for axis in range(-output_rank, output_rank):
                with self.subTest(shape=shape, axis=axis, on_value=on_value, off_value=off_value):
                    position = axis % output_rank
                    expected_shape = list(shape)
                    expected_shape.insert(position, depth)
                    expected = torch.full(expected_shape, off_value)
                    for coordinate in itertools.product(*(range(size) for size in shape)):
                        output_coordinate = list(coordinate)
                        output_coordinate.insert(position, int(indices[coordinate]))
                        expected[tuple(output_coordinate)] = on_value
                    actual = self.model_class(
                        depth, on_value=on_value, off_value=off_value, axis=axis,
                    )(indices)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_positive_and_negative_axes(self):
        self.check_axes(on_value=1, off_value=0)

    def test_custom_on_and_off_values(self):
        self.check_axes(on_value=7, off_value=-2)
        self.check_axes(on_value=0.75, off_value=-0.25)

    def test_out_of_range_axes(self):
        for shape in ((), (3,), (2, 3), (2, 1, 3)):
            indices = torch.zeros(shape, dtype=torch.int64)
            output_rank = indices.ndim + 1
            for axis in (-output_rank - 1, output_rank):
                with self.subTest(shape=shape, axis=axis), self.assertRaises(IndexError):
                    self.model_class(4, axis=axis)(indices)


if __name__ == "__main__":
    unittest.main()
