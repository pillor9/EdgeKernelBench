"""Test task rendering without importing PyTorch or using a device."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from materialize import ROOT, metadata, profile_fields, render, scenario_profiles, task_index


class MaterializeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.template = "width = __EKB_PARAM_width__\nshape = __EKB_INPUT_shape__\n"
        self.profile = {"parameters": {"width": 4}, "input_constants": {"shape": [2, 4]}}

    def test_renders_literals(self) -> None:
        self.assertEqual(render(self.template, self.profile), "width = 4\nshape = [2, 4]\n")

    def test_lists_unique_fields(self) -> None:
        self.assertEqual(
            profile_fields(self.template + self.template),
            {"parameters": ["width"], "input_constants": ["shape"]},
        )

    def test_rejects_missing_and_extra_fields(self) -> None:
        for parameters in ({}, {"width": 4, "unused": 1}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                render(self.template, {**self.profile, "parameters": parameters})

    def test_rejects_unknown_section(self) -> None:
        with self.assertRaises(ValueError):
            render(self.template, {**self.profile, "unknown": {}})

    def test_checks_expression_syntax_without_evaluation(self) -> None:
        profile = {**self.profile, "parameters": {"width": {"python_expr": "2 * 4"}}}
        assignment = ast.parse(render(self.template, profile)).body[0]
        self.assertEqual(ast.unparse(assignment.value), "2 * 4")
        profile["parameters"]["width"]["python_expr"] = "2 *"
        with self.assertRaises(SyntaxError):
            render(self.template, profile)

    def test_expression_preserves_precedence_in_context(self) -> None:
        profile = {"input_constants": {"value": {"python_expr": "1 + 2"}}}
        for expression, expected in (
            ("-__EKB_INPUT_value__", -3),
            ("4 * __EKB_INPUT_value__", 12),
        ):
            with self.subTest(expression=expression):
                rendered = render(f"result = {expression}\n", profile)
                namespace = {}
                exec(rendered, {}, namespace)
                self.assertEqual(namespace["result"], expected)

    def test_rejects_unknown_scenario(self) -> None:
        with self.assertRaises(ValueError):
            scenario_profiles("unknown")

    def test_all_profiles_render_and_match_paper_counts(self) -> None:
        tasks = task_index()
        self.assertEqual(len(tasks), 331)
        expected = {
            "mobile_personal": {1: 91, 2: 93, 3: 20},
            "embodied_robotics": {1: 98, 2: 95, 3: 25},
            "smart_aiot": {1: 98, 2: 105, 3: 25},
        }
        for scenario, counts in expected.items():
            observed = {1: 0, 2: 0, 3: 0}
            for task_id, profile in scenario_profiles(scenario).items():
                template = tasks[task_id].read_text(encoding="utf-8")
                observed[metadata(template)["level"]] += 1
                ast.parse(render(template, profile))
            self.assertEqual(observed, counts)

    def test_cli_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "task.py"
            command = [
                sys.executable, str(ROOT / "tools" / "materialize.py"),
                "--task", "ekb_l1_001_abs", "--scenario", "mobile_personal",
                "--output", str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            original = output.read_bytes()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
