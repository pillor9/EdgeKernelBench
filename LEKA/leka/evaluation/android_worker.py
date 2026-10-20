"""vivo_cpu execution: host cross-compile (+ optional .ll reinjection) + RPC run.

Runs on the host inside the trusted driver (subprocess via run_vivo_job for
isolation from agent-authored module code).

Reinjection route: the Executable.mod tree for a Relax module is
relax.VMExecutable(root, binary-serializable) + one llvm import carrying the .ll.
Patch flow: verify sha(.ll)==t12_sha -> llc -filetype=obj -> root.clear_imports()
-> export_library(fcompile=ndk.create_shared, addons=[patched.o]).

``mode=audit`` executes the task-oracle correctness path once and returns
before warmup, timing, or profiling.
"""

import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from leka.evaluation.timing_quality import paired_stability
from leka.candidate_source import validate_candidate_source
from leka.parameter_plan import materialize_arguments, plan_from_main
from leka.evaluation.diagnostics import bounded_error_excerpt

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_ll(text: str) -> str:
    """TVM's LLVM codegen embeds process pointers in TBAA metadata names
    (e.g. !"0x7629a5b00.w32.b0"), so raw .ll differs across processes.
    Rename each distinct address to a stable id in first-appearance order;
    distinct addresses stay distinct, so aliasing information is preserved."""
    import re as _re

    mapping: dict[str, str] = {}

    def repl(m):
        addr = m.group(0)
        if addr not in mapping:
            mapping[addr] = f"0xLEKA{len(mapping)}"
        return mapping[addr]

    return _re.sub(r"0x[0-9a-f]{6,}", repl, text)


def flatten_outputs(value):
    """Flatten TVM tensor/Array/tuple outputs in deterministic tree order."""
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        return [value]
    if isinstance(value, (tuple, list)):
        return [leaf for child in value for leaf in flatten_outputs(child)]
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        try:
            return [
                leaf
                for index in range(len(value))
                for leaf in flatten_outputs(value[index])
            ]
        except (IndexError, TypeError):
            pass
    raise TypeError(f"unsupported VM output container: {type(value).__name__}")


def summarize_correctness(value, output_names, expected, *, rtol, atol):
    """Compare one VM result tree with the task oracle without broadcasting."""
    import numpy as np

    outs = flatten_outputs(value)
    if len(outs) != len(output_names):
        raise RuntimeError(
            f"VM returned {len(outs)} output leaves; task declares {len(output_names)}"
        )
    correctness = {}
    all_ok = True
    for name, val in zip(output_names, outs):
        got = val.numpy()
        want = expected[name]
        same_shape = got.shape == want.shape
        same_dtype = got.dtype == want.dtype
        ok = False
        max_abs_error = float("inf")
        if same_shape and same_dtype:
            if np.issubdtype(want.dtype, np.bool_):
                differences = np.logical_xor(got, want)
                ok = not bool(np.any(differences))
                max_abs_error = float(np.any(differences))
            elif np.issubdtype(want.dtype, np.integer):
                ok = bool(np.array_equal(got, want))
                # The unsigned difference is exact even across signed extrema.
                high = np.maximum(got, want).astype(np.uint64)
                low = np.minimum(got, want).astype(np.uint64)
                with np.errstate(over="ignore"):
                    differences = high - low
                max_abs_error = float(np.max(differences)) if got.size else 0.0
            elif np.issubdtype(want.dtype, np.inexact):
                finite = bool(np.isfinite(got).all() and np.isfinite(want).all())
                if finite:
                    ok = bool(np.allclose(got, want, rtol=rtol, atol=atol))
                    error_dtype = np.result_type(want.dtype, np.float64)
                    with np.errstate(over="ignore", invalid="ignore"):
                        differences = np.abs(got.astype(error_dtype) - want.astype(error_dtype))
                    max_abs_error = float(np.max(differences)) if got.size else 0.0
        correctness[name] = {
            "pass": ok,
            "max_abs_error": max_abs_error,
            "shape": list(got.shape),
            "dtype": str(got.dtype),
            "expected_dtype": str(want.dtype),
        }
        all_ok = all_ok and ok
    return correctness, all_ok


def _timing_policy(job: dict) -> dict:
    """Return the normalized FinalBenchmark saved-closure timing policy."""
    cfg = job.get("backend_cfg") or {}
    return {
        "protocol": cfg.get("timing_protocol", "saved_relax_entry_ab_ba_v1"),
        "scored_pairs": max(1, int(cfg.get("timing_scored_pair_count", 20))),
        "target_window_ms": max(
            0.001, float(cfg.get("timing_evaluator_target_window_ms", 100))
        ),
        "maximum_number": max(
            1, int(cfg.get("timing_evaluator_maximum_number", 1000))
        ),
        "minimum_warmup_invocations": max(
            int(job.get("warmup", 1)),
            int(cfg.get("timing_minimum_warmup_invocations", 1)),
        ),
    }


def _saved_timer(vm, name: str, dev):
    """Cache RPC-side evaluators; returned samples exclude host/RPC latency."""
    evaluators = {}

    def sample(number: int) -> float:
        if number < 1:
            raise ValueError("saved-closure evaluator number must be positive")
        timer = evaluators.get(number)
        if timer is None:
            timer = vm.time_evaluator(
                name, dev, number=number, repeat=1, min_repeat_ms=0
            )
            evaluators[number] = timer
        dev.sync()
        values = tuple(float(value) for value in timer().results)
        dev.sync()
        if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0:
            raise RuntimeError(f"invalid saved-closure timing result: {values!r}")
        return values[0] * 1000.0

    return sample


def _adaptive_number(probe_ms: float, policy: dict) -> int:
    return min(
        policy["maximum_number"],
        max(1, int(math.ceil(policy["target_window_ms"] / probe_ms))),
    )


def _warm_saved_closure(vm, name: str, dev, *, evaluator_number: int,
                        probe_ms: float, policy: dict) -> dict:
    target_ms = max(
        probe_ms * evaluator_number * policy["scored_pairs"],
        policy["target_window_ms"],
    )
    minimum = policy["minimum_warmup_invocations"]
    warm_number = max(minimum, evaluator_number)
    started = time.monotonic()
    timer = vm.time_evaluator(
        name,
        dev,
        number=warm_number,
        repeat=1,
        min_repeat_ms=max(0, int(math.ceil(target_ms))),
    )
    values = tuple(float(value) for value in timer().results)
    dev.sync()
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError(f"invalid saved-closure warmup result: {values!r}")
    return {
        "minimum_invocations": minimum,
        "evaluator_number": evaluator_number,
        "target_duration_ms": target_ms,
        "host_elapsed_ms": (time.monotonic() - started) * 1000.0,
        "returned_device_per_invocation_ms": [value * 1000.0 for value in values],
        "initial_evaluator_number": warm_number,
        "final_evaluator_number": None,
        "total_invocations": None,
        "invocation_count_status": "adaptive_runtime_count_not_exposed",
    }


def finish(job_path: Path, result: dict, *, message: dict | None = None) -> None:
    (job_path / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(message or {
        "ok": result.get("ok", False),
        "fault_stage": result.get("fault_stage"),
    }))


def _llvm_import(root):
    for c in root.imports:
        if c.kind == "llvm":
            return c
    raise RuntimeError("no llvm module in executable import tree")


def _refresh_rpc(cfg: dict) -> None:
    threads = int(cfg.get("runtime_num_threads", 8))
    env = dict(os.environ, VIVO_SERIAL=cfg["vivo_serial"],
               VIVO_RPC_PORT=str(cfg["rpc_port"]), VIVO_NUM_THREADS=str(threads),
               LEKA_VIVO_STATE_DIR=cfg.get(
                   "rpc_state_dir", str(Path(tempfile.gettempdir()) / "LEKA" / "vivo_rpc")
               ),
               LEKA_VIVO_DEVICE_LIB_DIR=cfg.get("device_lib_dir", "/data/local/tmp/vivo_tvm"))
    subprocess.run(("bash", str(REPO_ROOT / cfg["rpc_refresh_script"])), cwd=str(REPO_ROOT),
                   env=env, check=True, capture_output=True, text=True, timeout=180)


def _runtime_affinity_mode(cfg: dict) -> int:
    """Missing fields in saved jobs retain their historical mode 0 behavior."""
    mode = cfg.get("runtime_affinity_mode", 0)
    if type(mode) is not int or mode not in (0, 1):
        raise ValueError("runtime_affinity_mode must be integer 0 (legacy) or 1 (kBig)")
    return mode


def _configure_runtime_policy(remote, cfg: dict) -> dict:
    mode = _runtime_affinity_mode(cfg)
    expected_threads = int(cfg.get("runtime_num_threads", 8))
    remote.get_function("runtime.config_threadpool")(mode, expected_threads)
    observed_threads = int(remote.get_function("runtime.NumThreads")())
    if observed_threads != expected_threads:
        raise RuntimeError(
            "Vivo runtime thread policy mismatch: "
            f"expected {expected_threads}, observed {observed_threads}"
        )
    # Derive the execution-request label from the actual mode, so a legacy
    # override cannot retain a copied mode-1 label from a new backend config.
    suffix = "v1" if mode == 1 else "legacy_v1"
    return {
        "profile_id": f"vivo_cpu_{expected_threads}threads_affinity_mode{mode}_{suffix}",
        "configured_profile_id": cfg.get("runtime_policy_id"),
        "num_threads": observed_threads,
        "requested_num_threads": expected_threads,
        "observed_num_threads": observed_threads,
        "requested_affinity_mode": mode,
        "affinity_mode_source": (
            "backend_cfg.runtime_affinity_mode" if "runtime_affinity_mode" in cfg
            else "legacy_saved_job_missing_field_default_0"
        ),
        "affinity_validation": "not_observed_by_this_worker",
        "actual_execution_cpus": None,
        "profile_id_scope": "requested_configuration_not_execution_proof",
        "mode_semantics": (
            "local_TVM_AffinityMode.kBig" if mode == 1
            else "legacy_mode_0_not_a_named_local_TVM_AffinityMode"
        ),
        "mode_semantics_authority": "inspected_local_TVM_source",
        "target_runtime_abi_validation": "not_verified_by_this_worker",
        "limits": [
            "NumThreads proves pool size, not per-thread masks or CPU utilization.",
            "Local kBig semantics with an explicit thread count do not prove big-only execution.",
            "Target runtime binary ABI and phase-specific affinity are not verified here.",
            "This mode setting does not establish a cause or cure for timing-context dependence.",
        ],
    }


def load_ir_module(path: Path, tag: str, module_format: str = "tvmscript"):
    import tvm

    if module_format == "tvm_ir_json":
        module = tvm.ir.load_json(path.read_text())
        if not isinstance(module, tvm.IRModule):
            raise TypeError("trusted IR JSON did not decode to tvm.IRModule")
        return module
    if module_format != "tvmscript":
        raise ValueError(f"unsupported module format: {module_format}")
    validate_candidate_source(path.read_text())
    spec = importlib.util.spec_from_file_location(tag, path)
    mod_py = importlib.util.module_from_spec(spec)
    sys.modules[tag] = mod_py
    spec.loader.exec_module(mod_py)
    return mod_py.Module


def build_so(job_path: Path, job: dict, *, module_rel: str, patched_rel, t12_sha,
             tag: str, so_name: str, module_format: str = "tvmscript") -> tuple[Path, str, object]:
    """parse -> compile -> (optional .ll patch: llc + link-time object swap) -> .so"""
    import tvm
    from tvm.contrib import ndk

    module_path = job_path / module_rel
    ir_module = load_ir_module(module_path, tag, module_format)
    target = tvm.target.Target(job["target"])
    ex = tvm.compile(ir_module, target=target)
    ll_text = canonicalize_ll(_llvm_import(ex.mod).inspect_source())
    so_path = job_path / so_name
    if patched_rel:
        if t12_sha and sha(ll_text) != t12_sha:
            raise RuntimeError("stale_t12: generated .ll no longer matches the "
                               "T12 base this patch was written against")
        obj = job_path / f"{so_name}.patched.o"
        subprocess.run(("llc", "-mtriple=aarch64-linux-android", "-relocation-model=pic",
                        "-filetype=obj", str(job_path / patched_rel), "-o", str(obj)),
                       check=True, capture_output=True, text=True, timeout=120)

        # Swap the llvm-derived object for patched.o at link time; the
        # packed-imports object (devc*) and everything else stay intact.
        def swapping_fcompile(output, objects, **kwargs):
            new_objects, swapped = [], False
            for entry_obj in objects:
                base = Path(str(entry_obj)).name
                if not swapped and base.endswith(".o") and not base.startswith("devc"):
                    new_objects.append(str(obj))
                    swapped = True
                else:
                    new_objects.append(entry_obj)
            if not swapped:
                raise RuntimeError(f"no llvm object found to swap in {objects}")
            ndk.create_shared(output, new_objects, **kwargs)

        for attr in ("output_format", "object_format", "get_target_triple"):
            if hasattr(ndk.create_shared, attr):
                setattr(swapping_fcompile, attr, getattr(ndk.create_shared, attr))
        ex.export_library(str(so_path), fcompile=swapping_fcompile)
    else:
        ex.export_library(str(so_path), fcompile=ndk.create_shared)
    return so_path, ll_text, ir_module


def main(job_dir: str) -> None:
    job_path = Path(job_dir)
    job = json.loads((job_path / "job.json").read_text())
    cfg = job["backend_cfg"]
    result: dict = {"ok": False, "fault_stage": "init"}
    try:
        # Reject invalid modes before compilation, RPC refresh, or device access.
        _runtime_affinity_mode(cfg)
        if not cfg.get("vivo_serial"):
            raise ValueError("set VIVO_SERIAL to the device identifier shown by adb devices")
        if not cfg.get("ndk_clang"):
            raise ValueError("set TVM_NDK_CC to the Android NDK aarch64 compiler")
        if not Path(cfg["ndk_clang"]).is_file():
            raise ValueError("TVM_NDK_CC must identify an existing Android NDK compiler")
        import numpy as np
        import tvm
        from tvm import relax as _relax
        from tvm import rpc

        os.environ["TVM_NDK_CC"] = cfg["ndk_clang"]

        if job.get("mode") == "extract":
            result["fault_stage"] = "compile"
            import tvm as _tvm
            module_path = job_path / job["module_path"]
            ir_module = load_ir_module(
                module_path,
                "leka_candidate_module",
                job.get("module_format", "tvmscript"),
            )
            ex = _tvm.compile(ir_module, target=_tvm.target.Target(job["target"]))
            ll_text = canonicalize_ll(_llvm_import(ex.mod).inspect_source())
            result["generated_sha"] = sha(ll_text)
            result["generated_source"] = ll_text
            result["ok"] = True
            result["fault_stage"] = None
            (job_path / "result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({"ok": True, "mode": "extract"}))
            return

        result["fault_stage"] = "compile"
        so_path, ll_text, candidate_ir = build_so(
            job_path, job, module_rel=job["module_path"],
            patched_rel=job.get("patched_path"),
            t12_sha=job.get("t12_sha"),
            tag="leka_candidate_module", so_name="candidate.so",
            module_format=job.get("module_format", "tvmscript")
        )
        result["generated_sha"] = sha(ll_text)
        ref_so = None
        if job.get("reference_module_path"):
            result["fault_stage"] = "compile_reference"
            ref_so, _, reference_ir = build_so(
                job_path, job, module_rel=job["reference_module_path"],
                patched_rel=job.get("reference_patched_path"),
                t12_sha=job.get("reference_t12_sha"),
                tag="leka_reference_module", so_name="reference.so",
                module_format=job.get("reference_module_format", "tvmscript")
            )

        result["fault_stage"] = "rpc"
        _refresh_rpc(cfg)
        remote = rpc.connect("127.0.0.1", cfg["rpc_port"], session_timeout=600)
        result["runtime_policy"] = _configure_runtime_policy(remote, cfg)
        stamp = int(time.time())
        remote.upload(str(so_path), target=f"leka_{stamp}_cand.so")
        rmod = remote.load_module(f"leka_{stamp}_cand.so")
        dev = remote.cpu(0)
        ref_entry = None

        result["fault_stage"] = "device_setup"
        inputs = dict(np.load(job_path / job["inputs_npz"]))
        expected = dict(np.load(job_path / job["expected_npz"]))
        source_state_names = job.get("state_input_names", [])
        dynamic_input_names = job.get("dynamic_input_names", job["input_names"])
        candidate_plan = plan_from_main(
            candidate_ir[job["entry"]], source_state_names
        )
        candidate_arrays = materialize_arguments(
            inputs=inputs,
            dynamic_input_names=dynamic_input_names,
            source_state_names=source_state_names,
            plan=candidate_plan,
        )
        vm = _relax.VirtualMachine(rmod, dev)
        entry = vm[job["entry"]]
        args = [tvm.runtime.tensor(value, device=dev) for value in candidate_arrays]
        result["parameter_plan"] = candidate_plan
        if ref_so is not None:
            remote.upload(str(ref_so), target=f"leka_{stamp}_ref.so")
            ref_mod = remote.load_module(f"leka_{stamp}_ref.so")
            ref_vm = _relax.VirtualMachine(ref_mod, dev)
            ref_entry = ref_vm[job["entry"]]
            reference_plan = plan_from_main(
                reference_ir[job["entry"]], source_state_names
            )
            reference_arrays = materialize_arguments(
                inputs=inputs,
                dynamic_input_names=dynamic_input_names,
                source_state_names=source_state_names,
                plan=reference_plan,
            )
            ref_args = [
                tvm.runtime.tensor(value, device=dev) for value in reference_arrays
            ]

        result["fault_stage"] = "run_correctness"
        # Across TVM RPC, invoking a VM entry as a packed function transports
        # tensor arguments as legacy DLTensor pointers.  Stateful VM input
        # binding performs the required remote ffi.Tensor conversion and is
        # the production Vivo worker's established ABI boundary.
        vm.set_input(job["entry"], *args)
        vm.invoke_stateful(job["entry"])
        out = vm.get_outputs(job["entry"])
        correctness, all_ok = summarize_correctness(
            out,
            job["output_names"],
            expected,
            rtol=job["rtol"],
            atol=job["atol"],
        )
        result["correctness"] = correctness
        result["correct"] = all_ok

        if ref_entry is not None:
            result["fault_stage"] = "reference_correctness"
            ref_vm.set_input(job["entry"], *ref_args)
            ref_vm.invoke_stateful(job["entry"])
            ref_out = ref_vm.get_outputs(job["entry"])
            reference_correctness, reference_ok = summarize_correctness(
                ref_out,
                job["output_names"],
                expected,
                rtol=job["rtol"],
                atol=job["atol"],
            )
            result["reference_correctness"] = reference_correctness
            result["reference_correct"] = reference_ok
            if not reference_ok:
                result["error"] = "paired reference failed task-oracle correctness"
                finish(job_path, result)
                return

        if not all_ok:
            result["fault_stage"] = "run_correctness"
            result["error"] = "candidate failed task-oracle correctness"
            finish(job_path, result)
            return

        if job.get("mode") == "audit":
            result["ok"] = True
            result["fault_stage"] = None
            finish(job_path, result, message={"ok": True, "mode": "audit"})
            return

        result["fault_stage"] = "timing"
        policy = _timing_policy(job)
        candidate_name = "leka_candidate_bench"
        vm.save_function(
            job["entry"], candidate_name, *args, include_return=False
        )
        candidate_timer = _saved_timer(vm, candidate_name, dev)
        candidate_probe_ms = candidate_timer(1)
        candidate_number = _adaptive_number(candidate_probe_ms, policy)
        candidate_warmup = _warm_saved_closure(
            vm,
            candidate_name,
            dev,
            evaluator_number=candidate_number,
            probe_ms=candidate_probe_ms,
            policy=policy,
        )
        result["timing_protocol"] = {
            "name": policy["protocol"],
            "measurement_boundary": "saved_relax_entry_closure",
            "clock_location": "device_time_evaluator",
            "python_dispatch_timed": False,
            "transport_timed": False,
            "parameter_preparation_timed": False,
            "discard_prefix_pair_count": 1 if ref_entry is not None else 0,
            "scored_pair_count": policy["scored_pairs"],
            "candidate_probe_ms": candidate_probe_ms,
            "candidate_evaluator_number": candidate_number,
            "candidate_warmup": candidate_warmup,
        }

        if ref_entry is not None:
            reference_name = "leka_reference_bench"
            ref_vm.save_function(
                job["entry"], reference_name, *ref_args, include_return=False
            )
            reference_timer = _saved_timer(ref_vm, reference_name, dev)
            reference_probe_ms = reference_timer(1)
            reference_number = _adaptive_number(reference_probe_ms, policy)
            reference_warmup = _warm_saved_closure(
                ref_vm,
                reference_name,
                dev,
                evaluator_number=reference_number,
                probe_ms=reference_probe_ms,
                policy=policy,
            )
            result["timing_protocol"].update({
                "reference_probe_ms": reference_probe_ms,
                "reference_evaluator_number": reference_number,
                "reference_warmup": reference_warmup,
                "pair_order": "deterministic_balanced_AB_BA",
            })
            cand_ms, ref_ms, ratios, samples = [], [], [], []
            for i in range(policy["scored_pairs"] + 1):
                order = "AB" if i % 2 == 0 else "BA"
                if order == "AB":
                    cc = candidate_timer(candidate_number)
                    rr = reference_timer(reference_number)
                else:
                    rr = reference_timer(reference_number)
                    cc = candidate_timer(candidate_number)
                discarded = i == 0
                samples.append({
                    "pair_index": i,
                    "order": order,
                    "candidate_ms": cc,
                    "reference_ms": rr,
                    "ratio": cc / rr,
                    "discarded": discarded,
                })
                if discarded:
                    continue
                ref_ms.append(rr)
                cand_ms.append(cc)
                ratios.append(cc / rr)
            mean_ratio = sum(ratios) / len(ratios)
            result["latency_ms"] = statistics.median(cand_ms)
            result["latency_batches_ms"] = [round(b, 9) for b in cand_ms]
            result["timing_samples"] = samples
            result["stability"] = paired_stability(cand_ms, ref_ms)
            result["vs_base"] = {
                "median_ratio": statistics.median(ratios),
                "ratio_cv": (statistics.pstdev(ratios) / mean_ratio) if mean_ratio > 0 else 0.0,
                "pairs": len(ratios),
                "base_latency_ms": statistics.median(ref_ms),
            }
        else:
            batches = [
                candidate_timer(candidate_number)
                for _ in range(policy["scored_pairs"])
            ]
            result["latency_ms"] = statistics.median(batches)
            result["latency_batches_ms"] = [round(b, 9) for b in batches]

        if job.get("profile"):
            result["fault_stage"] = "profile"
            try:
                import re as _re
                vmp = _relax.VirtualMachine(rmod, dev, profile=True)
                pargs = [
                    tvm.runtime.tensor(value, device=dev)
                    for value in candidate_arrays
                ]
                report = vmp.profile(job["entry"], *pargs)
                num = _re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

                def value(cell):
                    m = num.search(str(cell))
                    return float(m.group(0)) if m else None

                rows = []
                for call in report.calls:
                    row = {"name": str(call.get("Name", ""))}
                    for key, out_key in (("Duration (us)", "duration_us"),
                                         ("Count", "count"), ("Percent", "percent")):
                        if key in call:
                            row[out_key] = value(call[key])
                    rows.append(row)
                rows.sort(key=lambda r: -(r.get("duration_us") or 0.0))
                result["profile_breakdown"] = rows[:64]
            except Exception as err:  # profiling is best-effort evidence, never a failure
                result["profile_error"] = str(err)[:400]
        result["ok"] = True
        result["fault_stage"] = None
    except subprocess.CalledProcessError as err:
        error = err.stderr or err.stdout or str(err)
        result["error"], truncated = bounded_error_excerpt(error, max_bytes=6000, max_lines=80)
        result["error_capture"] = {"policy": "error_anchor_context_v1", "truncated": truncated,
                                   "source_bytes": len(error.encode("utf-8"))}
    except Exception:
        result["error"] = traceback.format_exc()[-8000:]
    finish(job_path, result)


if __name__ == "__main__":
    main(sys.argv[1])
