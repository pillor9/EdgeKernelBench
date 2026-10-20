"""Host-side checks for task isolation, validation, and paired timing."""

from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.interfaces import BuildResult, BuildTaskSnapshot, NamedArray, TensorSpec, TreeSpec
from pipeline.workflow.candidate_safety import SandboxLimits, inspect_source
from pipeline.workflow.errors import PipelineFault
from pipeline.workflow.ir_validation import TVMValidationReport, _validate_abi
from pipeline.workflow.job_bundle import JobBundleReader, JobBundleWriter
from pipeline.workflow.numerical_policy import contract_from_task_module
from pipeline.workflow.paired_timing import (
    TimingConfig, TimingDomain, TimingEnvironment, WarmupEvidence,
    require_stable_timing, run_paired_timing,
)
from pipeline.workflow.prompt_builder import PromptBuilder, ShotExample, public_task_payload
from pipeline.workflow.response_parser import normalize_response
from pipeline.workflow.stability import ratio_stability_eligible
from pipeline.workflow.task_loader import HiddenCase, TaskLoadConfig, _load_parameter_variant


def task_snapshot() -> BuildTaskSnapshot:
    spec = TensorSpec((2,), "float32", "x")
    return BuildTaskSnapshot(
        source_text="class Model: pass\n", init_args=(), input_specs=(spec,),
        output_spec=TreeSpec.tensor_leaf(TensorSpec((2,), "float32", "output")),
        input_names=("x",), task_id="example", _parameters={}, _buffers={},
    )


class ConstantTimer:
    def __init__(self, latency_ms: float) -> None:
        self.latency_ms = latency_ms

    def time_evaluator(self, number: int) -> float:
        return self.latency_ms

    def warmup_to_target(self, *, evaluator_number, target_duration_ms, minimum_invocations):
        count = max(evaluator_number, minimum_invocations)
        count = ((count + evaluator_number - 1) // evaluator_number) * evaluator_number
        return WarmupEvidence(count, target_duration_ms)


def paired_result():
    domain = TimingDomain("cpu", "device", "llvm", None, "compiler", "runtime", "session")
    return run_paired_timing(
        ConstantTimer(1.0), ConstantTimer(2.0), candidate_domain=domain,
        baseline_domain=domain, environment=TimingEnvironment(True),
        config=TimingConfig(outer_pairs=5),
    )


class PipelineTests(unittest.TestCase):
    def test_public_prompt_contains_specs_not_parameter_values(self):
        task = replace(task_snapshot(), _parameters={"weight": np.array([17.0, 29.0])})
        public = public_task_payload(task)
        self.assertEqual(public["parameters"][0]["shape"], [2])
        self.assertNotIn("value", public["parameters"][0])
        self.assertNotIn("hidden_cases", public)

    def test_prompt_uses_one_demonstration_and_target(self):
        task = task_snapshot()
        shot = ShotExample("example", "vivo_cpu", task, "def build(task):\n    return task\n")
        package = PromptBuilder("Implement the supplied computation.").build(
            task, backend="vivo_cpu", hardware_profile={}, build_environment={},
            timing_contract={}, shots=(shot,),
        )
        self.assertEqual([item.role for item in package.messages], ["system", "user", "assistant", "user"])
        self.assertEqual(package.shot_ids, ("example",))

    def test_wrong_backend_demonstration_is_rejected(self):
        task = task_snapshot()
        shot = ShotExample("example", "jetson_gpu", task, "def build(task):\n    return task\n")
        with self.assertRaises(ValueError):
            PromptBuilder("Implement the computation.").build(
                task, backend="vivo_cpu", hardware_profile={}, build_environment={},
                timing_contract={}, shots=(shot,),
            )

    def test_restricted_prompt_uses_json_demonstration(self):
        task = task_snapshot()
        source = {"tvm_module": "@I.ir_module\nclass Module:\n    pass\n"}
        for wrapper in (lambda text: text, lambda text: "```json\n" + text + "\n```"):
            with self.subTest(wrapper=wrapper):
                shot = ShotExample(
                    "example", "vivo_cpu", task, wrapper(json.dumps(source)),
                    mode="restricted_irmodule",
                )
                package = PromptBuilder("Implement the computation.").build(
                    task, backend="vivo_cpu", hardware_profile={}, build_environment={},
                    timing_contract={}, shots=(shot,), mode="restricted_irmodule",
                )
                self.assertEqual(json.loads(package.messages[2].content), source)
                self.assertIn("single field tvm_module", package.messages[1].content)

    def test_restricted_prompt_rejects_python_demonstration(self):
        task = task_snapshot()
        source = "def build(task):\n    return task\n"
        with self.assertRaises(ValueError):
            ShotExample("example", "vivo_cpu", task, source, mode="restricted_irmodule")
        shot = ShotExample("example", "vivo_cpu", task, source)
        with self.assertRaisesRegex(ValueError, "output mode"):
            PromptBuilder("Implement the computation.").build(
                task, backend="vivo_cpu", hardware_profile={}, build_environment={},
                timing_contract={}, shots=(shot,), mode="restricted_irmodule",
            )

    def test_restricted_demonstration_requires_exact_json_contract(self):
        for source in ('{}', '{"tvm_module": ""}', '{"tvm_module": null}',
                       '{"tvm_module": "a", "tvm_module": "b"}',
                       '{"tvm_module": "a", "extra": 1}', '{"tvm_module": NaN}'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                ShotExample("example", "vivo_cpu", task_snapshot(), source,
                            mode="restricted_irmodule")

    def test_python_fence_preserves_complete_source(self):
        source = "def build(task):\n    return task\n"
        self.assertEqual(normalize_response("```python\n" + source + "```").source, source)

    def test_multiple_candidates_are_rejected(self):
        with self.assertRaises(PipelineFault):
            normalize_response("```python\nx = 1\n```\n```python\nx = 2\n```")

    def test_restricted_host_and_scheduler_capabilities(self):
        for operation in ("import os", "import numpy as np\nnp.load('input.npy')",
                          "import tvm\nf = tvm.tir.Schedule"):
            with self.subTest(operation=operation):
                self.assertFalse(inspect_source(operation + "\ndef build(task):\n    return task\n").accepted)

    def test_valid_array_preparation_is_allowed(self):
        source = "import numpy as np\ndef build(task):\n    return np.zeros((2,), dtype='float32')\n"
        self.assertTrue(inspect_source(source).accepted)

    def test_entry_argument_count_is_exact(self):
        result = BuildResult(None, (), ("x",))
        for count in (0, 2):
            with self.subTest(count=count), patch.dict("sys.modules", {"tvm": SimpleNamespace()}):
                with self.assertRaisesRegex(ValueError, "arguments"):
                    _validate_abi(result, task_snapshot(), SimpleNamespace(params=[None] * count))

    def test_invalid_seed_types_are_rejected(self):
        for seeds in ((True,), (1.5,), (-1,), (1, 1)):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                TaskLoadConfig(correctness_seeds=seeds)

    def test_seed_lists_are_normalized_to_tuples(self):
        self.assertEqual(TaskLoadConfig(correctness_seeds=[0, 1, 2]).correctness_seeds, (0, 1, 2))

    def test_invalid_timeouts_are_rejected(self):
        for value in (float("nan"), float("inf"), 0, True):
            for cls in (TaskLoadConfig, SandboxLimits):
                with self.subTest(value=value, cls=cls), self.assertRaises(ValueError):
                    cls(timeout_seconds=value)

    def test_timing_requires_integer_counts_and_two_scored_pairs(self):
        for kwargs in ({"outer_pairs": 2}, {"outer_pairs": 3.5},
                       {"warmup_minimum_invocations": True}, {"evaluator_maximum_number": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TimingConfig(**kwargs)

    def test_paired_speedup_and_balanced_order(self):
        result = paired_result()
        self.assertEqual(result.speedup, 2.0)
        self.assertTrue(result.stable)
        scored = [sample for sample in result.samples if sample.leg == "candidate" and not sample.discarded]
        self.assertEqual([sample.order for sample in scored], ["BA", "AB", "BA", "AB"])
        require_stable_timing(result)

    def test_stability_includes_the_five_percent_boundary(self):
        for dispersion, accepted in ((0.05, True), (math.nextafter(0.05, math.inf), False)):
            with self.subTest(dispersion=dispersion), patch(
                "pipeline.workflow.paired_timing.robust_coefficient_of_variation",
                return_value=dispersion,
            ):
                result = paired_result()
                self.assertEqual(result.stable, accepted)
                if accepted:
                    require_stable_timing(result)
                else:
                    with self.assertRaises(PipelineFault):
                        require_stable_timing(result)

    def test_ratio_stability_includes_boundary_and_rejects_invalid_samples(self):
        for dispersion, accepted in ((0.05, True), (math.nextafter(0.05, math.inf), False)):
            with self.subTest(dispersion=dispersion), patch(
                "pipeline.workflow.stability.robust_coefficient_of_variation",
                return_value=dispersion,
            ):
                eligible, _, _ = ratio_stability_eligible([1.0, 1.0])
                self.assertEqual(eligible, accepted)
        for values in ([], [1.0], [0.0, 1.0], [-1.0, 1.0],
                       [float("nan"), 1.0], [float("inf"), 1.0], [True, 1.0]):
            with self.subTest(values=values):
                self.assertFalse(ratio_stability_eligible(values)[0])

    def test_stability_rechecks_samples_instead_of_trusting_summary(self):
        result = paired_result()
        changed = tuple(
            replace(sample, latency_ms=2.0)
            if sample.leg == "candidate" and not sample.discarded and sample.pair_index % 2
            else sample
            for sample in result.samples
        )
        result = replace(result, samples=changed)
        self.assertTrue(result.stable)
        with self.assertRaises(PipelineFault):
            require_stable_timing(result)

    def test_each_latency_series_must_be_stable(self):
        for values in ([0.06, 0.0, 0.0], [0.0, 0.06, 0.0], [0.0, 0.0, 0.06]):
            with self.subTest(values=values), patch(
                "pipeline.workflow.paired_timing.robust_coefficient_of_variation", side_effect=values,
            ):
                self.assertFalse(paired_result().stable)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_nested_output_structure_cannot_change_between_seeds(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return (x, [x])

        module = SimpleNamespace(Model=Model, get_init_inputs=lambda: ())
        x = torch.ones(2)
        draws = [((x,), (x, [x]), 1.0), ((x * 2,), (x, (x,)), 1.0)]
        with patch("pipeline.workflow.task_loader._draw_correctness_case", side_effect=draws):
            with self.assertRaisesRegex(ValueError, "container structure"):
                _load_parameter_variant(
                    module, source="class Model: pass", source_sha256="0" * 64,
                    task_id="example", parameter_seed=0, correctness_seeds=(0, 1),
                    correctness_nonce=None,
                )

    def test_bundle_roundtrip_and_tamper_detection(self):
        source_hash = "0" * 64
        contract = contract_from_task_module(SimpleNamespace(), source_hash)
        module_json = "{}"
        digest = hashlib.sha256(module_json.encode()).hexdigest()
        validation = TVMValidationReport(
            "main", ("x",), module_json, "", digest, "", "0", "0", True, 0,
            "tvm_native_codegen", (),
        )
        cases = tuple(HiddenCase(
            seed, (NamedArray("x", np.array([seed], dtype="float32")),),
            (NamedArray("output", np.array([seed], dtype="float32")),),
        ) for seed in (0, 1, 2))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "job"
            bundle = JobBundleWriter(path).write(
                case_source="def build(task):\n    return task\n", validation=validation,
                build_result=BuildResult(None, (), ("x",)), hidden_cases=cases,
                parameter_seed=0, correctness_contract=contract,
            )
            contents = JobBundleReader(path, expected_manifest_sha256=bundle.manifest_sha256).verify()
            self.assertEqual([case.seed for case in contents.hidden_cases], [0, 1, 2])
            (path / "module.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                JobBundleReader(path).verify()


if __name__ == "__main__":
    unittest.main()
