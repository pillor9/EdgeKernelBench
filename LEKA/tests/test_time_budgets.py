"""Check CLI budget validation and startup deadlines without device access."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from leka.operation_control import OperationCancelled, OperationControl
from leka.orchestration import optimization_loop
from leka.orchestration.optimization_loop import validate_time_budgets


class _ReachedAgentSetup(Exception):
    pass


class TimeBudgetTests(unittest.TestCase):
    def test_nonfinite_time_budgets_are_rejected(self):
        for option in ("wall_minutes", "turn_minutes", "finalize_minutes"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(option=option, value=value):
                    budgets = dict(wall_minutes=60, turn_minutes=8, finalize_minutes=7)
                    budgets[option] = value
                    with self.assertRaisesRegex(ValueError, option.replace("_", "-") + " must be finite"):
                        validate_time_budgets(**budgets, scheduling="async")

    def test_valid_budgets_remain_accepted(self):
        for scheduling in ("batch", "async"):
            with self.subTest(scheduling=scheduling):
                validate_time_budgets(wall_minutes=60, turn_minutes=8,
                                      finalize_minutes=7, scheduling=scheduling)

    @contextmanager
    def startup(self, *, finalize_only=False):
        clock = SimpleNamespace(now=1000.0)
        task = Mock()
        task.manifest.return_value = {}
        service = Mock()
        service.correctness_policy.return_value = {}
        pipe = Mock()
        pipe.operation_control = OperationControl()
        pipe.begin_operation_phase.side_effect = pipe.operation_control.reset
        pipe.bootstrap.side_effect = pipe.operation_control.checkpoint
        baseline = SimpleNamespace(ok=True, public={"latency_ms": 1.0})
        pipe.measure_default.return_value = baseline
        pipe.initialize_global_best.return_value = {
            "recovered": True, "selected_latency_ms": 1.0, "snapshot_id": "test",
        }
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            argv = [
                "leka run", "--run-dir", directory,
                "--task", str(Path(directory) / "task.py"),
                "--model", "test-model", "--wall-minutes", "8",
                "--finalize-minutes", "7",
            ]
            if finalize_only:
                argv.append("--finalize-only")
            stack.enter_context(patch.object(optimization_loop.sys, "argv", argv))
            stack.enter_context(patch.object(optimization_loop.time, "monotonic",
                                             side_effect=lambda: clock.now))
            load = stack.enter_context(patch.object(optimization_loop, "load_pytorch_task",
                                                    return_value=task))
            make_service = stack.enter_context(patch.object(optimization_loop, "MeasureService",
                                                            return_value=service))
            make_pipe = stack.enter_context(patch.object(optimization_loop, "Pipeline",
                                                         return_value=pipe))
            stack.enter_context(patch.object(optimization_loop, "bind_correctness_policy"))
            stack.enter_context(patch.object(optimization_loop, "require_baseline_latency",
                                             return_value=1.0))
            agent_setup = stack.enter_context(patch.object(
                optimization_loop, "layer_agents", side_effect=_ReachedAgentSetup))
            finalize = stack.enter_context(patch.object(optimization_loop, "finalize_and_report"))
            stack.enter_context(patch("builtins.print"))
            yield SimpleNamespace(
                clock=clock, task=task, load=load, make_service=make_service,
                make_pipe=make_pipe, pipe=pipe, baseline=baseline,
                agent_setup=agent_setup, finalize=finalize,
            )

    def test_task_loading_expiry_stops_before_service_setup(self):
        with self.startup() as state:
            def load(*args, **kwargs):
                state.clock.now = 1061.0
                return state.task
            state.load.side_effect = load
            with self.assertRaises(OperationCancelled):
                optimization_loop.main()
            state.make_service.assert_not_called()
            state.make_pipe.assert_not_called()

    def test_pipeline_setup_expiry_stops_before_bootstrap(self):
        with self.startup() as state:
            def make_pipe(*args, **kwargs):
                state.clock.now = 1061.0
                return state.pipe
            state.make_pipe.side_effect = make_pipe
            with self.assertRaises(OperationCancelled):
                optimization_loop.main()
            state.pipe.bootstrap.assert_not_called()
            state.pipe.measure_default.assert_not_called()

    def test_bootstrap_expiry_stops_before_baseline(self):
        with self.startup() as state:
            state.pipe.bootstrap.side_effect = lambda: setattr(state.clock, "now", 1061.0)
            with self.assertRaises(OperationCancelled):
                optimization_loop.main()
            state.pipe.measure_default.assert_not_called()
            state.pipe.initialize_global_best.assert_not_called()

    def test_baseline_expiry_stops_before_best_selection(self):
        with self.startup() as state:
            def measure():
                state.clock.now = 1061.0
                return state.baseline
            state.pipe.measure_default.side_effect = measure
            with self.assertRaises(OperationCancelled):
                optimization_loop.main()
            state.pipe.initialize_global_best.assert_not_called()
            state.agent_setup.assert_not_called()

    def test_startup_consumes_the_original_work_budget(self):
        with self.startup() as state:
            def bootstrap():
                self.assertEqual(state.pipe.operation_control.clamp_timeout(600), 60.0)
                state.clock.now += 20.0
            def measure():
                self.assertEqual(state.pipe.operation_control.clamp_timeout(600), 40.0)
                state.clock.now += 10.0
                return state.baseline
            state.pipe.bootstrap.side_effect = bootstrap
            state.pipe.measure_default.side_effect = measure
            with self.assertRaises(_ReachedAgentSetup):
                optimization_loop.main()
            self.assertEqual(state.pipe.operation_control.clamp_timeout(600), 30.0)
            state.pipe.begin_operation_phase.assert_called_once_with(deadline_monotonic=1060.0)

    def test_best_selection_expiry_stops_before_agents(self):
        with self.startup() as state:
            def select(baseline):
                state.clock.now = 1061.0
                return {"recovered": True, "selected_latency_ms": 1.0, "snapshot_id": "test"}
            state.pipe.initialize_global_best.side_effect = select
            with self.assertRaises(OperationCancelled):
                optimization_loop.main()
            state.agent_setup.assert_not_called()

    def test_finalize_only_startup_uses_the_final_deadline(self):
        with self.startup(finalize_only=True) as state:
            def load(*args, **kwargs):
                state.clock.now = 1070.0
                return state.task
            def bootstrap():
                self.assertEqual(state.pipe.operation_control.clamp_timeout(600), 400.0)
            state.load.side_effect = load
            state.pipe.bootstrap.side_effect = bootstrap
            optimization_loop.main()
            self.assertTrue(all(call.kwargs["deadline_monotonic"] == 1470.0
                                for call in state.pipe.begin_operation_phase.call_args_list))
            state.finalize.assert_called_once()
            state.agent_setup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
