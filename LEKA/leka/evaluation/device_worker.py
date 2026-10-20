"""Execute CUDA or OpenCL jobs and write result.json on the target device.

job.json core: {module_path, entry, device: cuda|opencl, target,
                mode: measure|audit|extract,
                input_names, output_names, atol, rtol, warmup, repeat,
                patched_path|null, t12_sha|null, profile: bool,
                reference_module_path|null, reference_patched_path|null,
                reference_t12_sha|null}

L2 reinjection: a patched file is a full device-source
translation unit.  cuda: the postproc callback receives the whole generated
code — verify sha256 == t12_sha, then return the patched text.  opencl: the
callback runs per kernel — substitute the same-name kernel block; original
per-kernel sources are accumulated and their concatenation hashed for the
staleness check.

Timing: when a reference module is supplied, candidate and reference run as
interleaved pairs: per-pair latencies and candidate/reference ratios,
median ratio, and ratio CV are reported (first pair discarded).  Without a
reference, the median-of-batches fallback applies.

mode=extract compiles WITHOUT patches and returns the generated device source
(the driver stores it as the T12 slot).
mode=audit compiles and executes correctness once, but performs no warmup,
timing, or profiling.
"""

import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

RK_TIMING_SEGMENT_SAMPLES = 10

try:  # copied beside this script on the remote device
    from timing_quality import paired_stability
    from candidate_source import validate_candidate_source
    from parameter_plan import materialize_arguments, plan_from_main
except ImportError:  # package import used by local tests/tooling
    from leka.evaluation.timing_quality import paired_stability
    from leka.candidate_source import validate_candidate_source
    from leka.parameter_plan import materialize_arguments, plan_from_main


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_sources(root, kind: str) -> str:
    out = []

    def visit(m):
        if m.kind == kind:
            try:
                out.append(m.inspect_source())
            except Exception:
                pass
        for c in m.imports:
            visit(c)

    visit(root)
    return "\n".join(out)


_OPENCL_KERNEL_SIGNATURE_RE = re.compile(
    r"__kernel\s+void\s+(\w+)\s*\([^)]*\)"
)


def _c_brace_end(source: str, opening: int) -> int:
    """Return one-past a balanced C/OpenCL body, ignoring comments/strings."""

    if opening >= len(source) or source[opening] != "{":
        raise ValueError("OpenCL definition has no opening brace")
    depth = 0
    index = opening
    state = "code"
    quote = ""
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and following == "/":
                state = "code"
                index += 1
        elif state == "string":
            if char == "\\":
                index += 1
            elif char == quote:
                state = "code"
        elif char == "/" and following == "/":
            state = "line_comment"
            index += 1
        elif char == "/" and following == "*":
            state = "block_comment"
            index += 1
        elif char in {'"', "'"}:
            state = "string"
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                break
        index += 1
    raise ValueError("OpenCL kernel definition has unbalanced braces")


def _opencl_kernel_definitions(source: str) -> dict[str, tuple[int, int, str]]:
    """Return exact definition ranges, ignoring repeated forward declarations."""

    definitions: dict[str, tuple[int, int, str]] = {}
    for match in _OPENCL_KERNEL_SIGNATURE_RE.finditer(source):
        opening = source.find("{", match.end())
        terminator = source.find(";", match.end())
        if opening < 0 or (terminator >= 0 and terminator < opening):
            continue
        # TVM may place a work-group attribute between the parameter list and
        # opening brace.  A semicolon before that brace still unambiguously
        # identifies a declaration; otherwise retain the full suffix.
        end = _c_brace_end(source, opening)
        name = match.group(1)
        if name in definitions:
            raise ValueError(f"duplicate OpenCL kernel definition: {name}")
        definitions[name] = (match.start(), end, source[match.start():end])
    return definitions


def split_opencl_kernels(source: str) -> dict[str, str]:
    """Map kernel name to its exact definition, excluding shared prototypes."""

    return {
        name: definition
        for name, (_start, _end, definition) in _opencl_kernel_definitions(source).items()
    }


def rewrite_opencl_kernels(fsource: str, patched_blocks: dict[str, str]) -> str:
    """Replace definitions in one TVM callback unit while preserving its preamble.

    TVM's callback input commonly begins with declarations for every kernel.
    Selecting the first ``__kernel`` token therefore picks a prototype (often
    the same first kernel in every unit) and caused every runtime module to be
    rebuilt with the wrong definition.  Definition-range matching prevents
    that CL_INVALID_KERNEL_NAME failure.
    """

    replacements = []
    for name, (start, end, _definition) in _opencl_kernel_definitions(fsource).items():
        patched = patched_blocks.get(name)
        if patched is not None:
            replacements.append((start, end, patched))
    rewritten = fsource
    for start, end, patched in sorted(replacements, reverse=True):
        rewritten = rewritten[:start] + patched + rewritten[end:]
    return rewritten


def load_ir_module(path: Path, tag: str, module_format: str = "tvmscript"):
    if module_format == "tvm_ir_json":
        import tvm

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
    """Return the backend's normalized FinalBenchmark timing policy."""
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


def _make_vm(relax, executable, device, *, device_kind: str, profile: bool = False):
    # Match the independent RK replay worker: Mali retains pooled OpenCL
    # temporaries across VM calls until the 16 GiB board is exhausted.
    # Both members of a pair (and profiling) must use the same allocator.
    options = {"profile": True} if profile else {}
    if device_kind == "opencl":
        options["memory_cfg"] = "naive"
    return relax.VirtualMachine(executable, device, **options)


def _guard_mali_memory() -> dict:
    """Check host and system memory outside timed calls, before another dispatch.

    Mali driver allocations need not appear in this process's RSS. Keeping a
    system-memory reserve complements the per-process guard; neither is a
    substitute for bounded OpenCL context lifetimes.
    """
    snapshot = {}
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                snapshot["worker_rss_kib"] = int(line.split()[1])
                if snapshot["worker_rss_kib"] > 10 * 1024 * 1024:
                    raise MemoryError(f"Mali worker RSS guard: {snapshot['worker_rss_kib']} KiB exceeds 10 GiB; "
                                      "result is infrastructure-incomplete, not correctness-failed")
                break
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values = {}
        for line in meminfo.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] in ("MemTotal:", "MemAvailable:") and len(fields) >= 2:
                values[fields[0][:-1]] = int(fields[1])
        if "MemAvailable" in values:
            reserve = max(1024 * 1024, values.get("MemTotal", 0) // 10)
            snapshot.update(system_available_kib=values["MemAvailable"], reserve_kib=reserve)
            if values["MemAvailable"] < reserve:
                raise MemoryError(f"Mali system memory guard: MemAvailable={values['MemAvailable']} KiB "
                                  f"below reserve={reserve} KiB; infrastructure-incomplete")
    return snapshot


def _profile_candidate(result, relax, executable, device, job, candidate_arrays, tvm):
    """Collect one VM profile; RK invokes this in its own short-lived process."""
    result["fault_stage"] = "profile"
    try:
        if job["device"] == "opencl":
            result["profile_memory_before"] = _guard_mali_memory()
        vmp = _make_vm(relax, executable, device, device_kind=job["device"], profile=True)
        pargs = [tvm.runtime.tensor(value, device=device) for value in candidate_arrays]
        report = vmp.profile(job["entry"], *pargs)
        num = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
        def value(cell):
            match = num.search(str(cell))
            return float(match.group(0)) if match else None
        rows = []
        for call in report.calls:
            row = {"name": str(call.get("Name", ""))}
            for key, out_key in (("Duration (us)", "duration_us"), ("Count", "count"), ("Percent", "percent")):
                if key in call:
                    row[out_key] = value(call[key])
            rows.append(row)
        rows.sort(key=lambda row: -(row.get("duration_us") or 0.0))
        result["profile_breakdown"] = rows[:64]
        if job["device"] == "opencl":
            result["profile_memory_after"] = _guard_mali_memory()
    except MemoryError:
        raise  # resource exhaustion is an infrastructure failure, not optional missing telemetry
    except Exception as error:
        result["profile_error"] = str(error)[:400]


def _warm_saved_closure(vm, name: str, dev, *, evaluator_number: int,
                        probe_ms: float, policy: dict) -> dict:
    target_ms = max(
        probe_ms * evaluator_number * policy["scored_pairs"],
        policy["target_window_ms"],
    )
    minimum = policy["minimum_warmup_invocations"]
    if policy["maximum_number"] == 1:
        # min_repeat_ms lets TVM silently increase number beyond the Mali
        # one-dispatch cap. Synchronize individual calls instead.
        timer = _saved_timer(vm, name, dev)
        started = time.monotonic()
        invocations = 0
        # Match the replay worker's fixed Mali warmup; duration-chasing also
        # retains driver event resources on g24p0.
        while invocations < minimum:
            _guard_mali_memory()
            timer(1)
            invocations += 1
        return {
            "minimum_invocations": minimum,
            "evaluator_number": 1,
            "invocations": invocations,
            "target_duration_ms": 0.0,
            "driver_safe_fixed_warmup": True,
            "host_elapsed_ms": (time.monotonic() - started) * 1000.0,
        }
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
    }


def finish(job_path: Path, result: dict, *, message: dict | None = None,
           result_name: str = "result.json") -> None:
    (job_path / result_name).write_text(json.dumps(result, indent=2))
    print(json.dumps(message or {
        "ok": result.get("ok", False),
        "fault_stage": result.get("fault_stage"),
    }))


def build_executable(
    job_path: Path,
    job: dict,
    module_rel: str,
    patched_rel,
    t12_sha,
    tag: str,
    module_format: str = "tvmscript",
):
    """Compile one module with (optionally) one L2 patch applied via postproc."""
    import tvm
    import tvm_ffi

    device_kind = job["device"]
    ir_mod = load_ir_module(job_path / module_rel, tag, module_format)
    patched_text = (job_path / patched_rel).read_text() if patched_rel else None
    state = {"seen_generated": []}

    if device_kind == "cuda":
        def postproc(code, target):
            state["seen_generated"].append(code)
            if patched_text is None:
                return code
            if t12_sha and sha(code) != t12_sha:
                raise RuntimeError("stale_t12: generated CUDA no longer matches the "
                                   "T12 base this patch was written against")
            return patched_text
        tvm_ffi.register_global_func("tvm_callback_cuda_postproc", postproc, override=True)
    else:
        patched_blocks = split_opencl_kernels(patched_text) if patched_text else {}

        def postproc(fsource, target):
            state["seen_generated"].append(fsource)
            if not patched_blocks:
                return fsource
            return rewrite_opencl_kernels(fsource, patched_blocks)
        tvm_ffi.register_global_func("tvm_callback_opencl_postproc", postproc, override=True)

    target = tvm.target.Target(job["target"])
    ex = tvm.compile(ir_mod, target=target)
    generated_concat = "\n".join(state["seen_generated"])
    if (device_kind == "opencl" and patched_text is not None and t12_sha
            and sha(generated_concat) != t12_sha):
        raise RuntimeError("stale_t12: generated OpenCL no longer matches the "
                           "T12 base this patch was written against")
    return ex, generated_concat, ir_mod


def main(job_dir: str, *, segment_pairs: int | None = None,
         result_name: str = "result.json", skip_profile: bool = False,
         profile_only: bool = False) -> None:
    job_path = Path(job_dir)
    job = json.loads((job_path / "job.json").read_text())
    if profile_only:
        if job.get("device") != "opencl" or job.get("mode") != "measure" or not job.get("profile"):
            raise ValueError("profile-only child requires an RK profile measurement job")
        for name in ("reference_module_path", "reference_patched_path", "reference_t12_sha"):
            job.pop(name, None)
    if skip_profile:
        job["profile"] = False
    if segment_pairs is not None:
        job["backend_cfg"] = dict(job.get("backend_cfg") or {})
        job["backend_cfg"]["timing_scored_pair_count"] = segment_pairs
    result: dict = {"ok": False, "fault_stage": "init"}
    try:
        if job.get("device") == "opencl":
            result["memory_before"] = _guard_mali_memory()
        import numpy as np
        import tvm
        if job["device"] == "cuda":
            import tvm.contrib.nvcc  # registers tvm_callback_cuda_compile
        from tvm import relax as _relax

        result["fault_stage"] = "compile"
        ex, generated, candidate_ir = build_executable(
            job_path, job, job["module_path"],
            job.get("patched_path"), job.get("t12_sha"),
            "leka_candidate_module", job.get("module_format", "tvmscript")
        )
        result["generated_sha"] = sha(generated)

        if job.get("mode") == "extract":
            result["generated_source"] = (generated if generated
                                          else collect_sources(ex.mod, job["device"]))
            result["ok"] = True
            result["fault_stage"] = None
            (job_path / result_name).write_text(json.dumps(result, indent=2))
            print(json.dumps({"ok": True, "mode": "extract"}))
            return

        ref_entry = None
        if job.get("reference_module_path"):
            result["fault_stage"] = "compile_reference"
            ref_ex, _, reference_ir = build_executable(
                job_path, job, job["reference_module_path"],
                job.get("reference_patched_path"),
                job.get("reference_t12_sha"),
                "leka_reference_module",
                job.get("reference_module_format", "tvmscript")
            )

        result["fault_stage"] = "device_setup"
        dev = tvm.cuda(0) if job["device"] == "cuda" else tvm.opencl(0)
        if not bool(dev.exist):
            raise RuntimeError(f"{job['device']} device 0 unavailable")
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
        vm = _make_vm(_relax, ex, dev, device_kind=job["device"])
        result["vm_allocator"] = "naive" if job["device"] == "opencl" else "pooled"
        entry = vm[job["entry"]]
        args = [tvm.runtime.tensor(value, device=dev) for value in candidate_arrays]
        result["parameter_plan"] = candidate_plan
        if job.get("reference_module_path"):
            reference_plan = plan_from_main(
                reference_ir[job["entry"]], source_state_names
            )
            reference_arrays = materialize_arguments(
                inputs=inputs,
                dynamic_input_names=dynamic_input_names,
                source_state_names=source_state_names,
                plan=reference_plan,
            )
            ref_vm = _make_vm(_relax, ref_ex, dev, device_kind=job["device"])
            ref_entry = ref_vm[job["entry"]]
            ref_args = [
                tvm.runtime.tensor(value, device=dev) for value in reference_arrays
            ]

        result["fault_stage"] = "run_correctness"
        if job["device"] == "opencl":
            _guard_mali_memory()
        out = entry(*args)
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
            ref_out = ref_entry(*ref_args)
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
                finish(job_path, result, result_name=result_name)
                return

        if not all_ok:
            result["fault_stage"] = "run_correctness"
            result["error"] = "candidate failed task-oracle correctness"
            finish(job_path, result, result_name=result_name)
            return

        if job.get("mode") == "audit":
            result["ok"] = True
            result["fault_stage"] = None
            finish(job_path, result, message={"ok": True, "mode": "audit"}, result_name=result_name)
            return

        if profile_only:
            _profile_candidate(result, _relax, ex, dev, job, candidate_arrays, tvm)
            result.update(ok=True, fault_stage=None,
                          profile_execution={"scope": "fresh_process_single_vm_profile", "used_for_speedup": False})
            finish(job_path, result, result_name=result_name)
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
                if job["device"] == "opencl":
                    _guard_mali_memory()
                order = "AB" if i % 2 == 0 else "BA"
                if order == "AB":
                    c = candidate_timer(candidate_number)
                    r = reference_timer(reference_number)
                else:
                    r = reference_timer(reference_number)
                    c = candidate_timer(candidate_number)
                discarded = i == 0
                samples.append({
                    "pair_index": i,
                    "order": order,
                    "candidate_ms": c,
                    "reference_ms": r,
                    "ratio": c / r,
                    "discarded": discarded,
                })
                if discarded:
                    continue
                ref_ms.append(r)
                cand_ms.append(c)
                ratios.append(c / r)
            median_ratio = statistics.median(ratios)
            mean_ratio = sum(ratios) / len(ratios)
            ratio_cv = (statistics.pstdev(ratios) / mean_ratio) if mean_ratio > 0 else 0.0
            result["latency_ms"] = statistics.median(cand_ms)
            result["latency_batches_ms"] = [round(b, 9) for b in cand_ms]
            result["timing_samples"] = samples
            result["stability"] = paired_stability(cand_ms, ref_ms)
            result["vs_base"] = {
                "median_ratio": median_ratio,
                "ratio_cv": ratio_cv,
                "pairs": len(ratios),
                "base_latency_ms": statistics.median(ref_ms),
            }
        else:
            batches = []
            for _ in range(policy["scored_pairs"]):
                if job["device"] == "opencl":
                    _guard_mali_memory()
                batches.append(candidate_timer(candidate_number))
            result["latency_ms"] = statistics.median(batches)
            result["latency_batches_ms"] = [round(b, 9) for b in batches]

        if job.get("profile"):
            _profile_candidate(result, _relax, ex, dev, job, candidate_arrays, tvm)
        if job["device"] == "opencl":
            result["memory_after"] = _guard_mali_memory()

        result["ok"] = True
        result["fault_stage"] = None
    except MemoryError:
        result["fault_stage"] = "device_memory_guard"
        result["error"] = traceback.format_exc()[-8000:]
    except Exception:
        result["error"] = traceback.format_exc()[-8000:]
    finish(job_path, result, result_name=result_name)


def merge_opencl_segments(results: list[dict], expected_pairs: int) -> dict:
    """Keep every scored pair, its actual order, and all discarded prefixes.

    Only the context lifetime is bounded: both sides of EVERY pair still run
    in the same child process. Never treat partial sessions as a valid result.
    """
    if not results:
        raise ValueError("no RK timing segments")
    first = results[0]
    if 'reference_correct' not in first:
        batches = []
        for item in results:
            if item.get('ok') is not True or item.get('correct') is not True or 'reference_correct' in item:
                raise ValueError('RK single-arm segment failed or changed mode')
            for key in ('generated_sha', 'parameter_plan', 'vm_allocator'):
                if item.get(key) != first.get(key):
                    raise ValueError('RK single-arm segment identity changed')
            values = item.get('latency_batches_ms', [])
            if len(values) != item.get('timing_protocol', {}).get('scored_pair_count'):
                raise ValueError('RK single-arm segment count differs')
            if not values or not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError('RK single-arm segment invalid latency')
            batches.extend(values)
        if len(batches) != expected_pairs:
            raise ValueError('RK single-arm aggregate count differs')
        return {**first, 'latency_ms': statistics.median(batches), 'latency_batches_ms': batches,
                'timing_protocol': {**first['timing_protocol'],
                    'session_scope': 'rk3588_mali_fresh_process_segments_v1',
                    'session_count': len(results), 'scored_pair_count': len(batches),
                    'per_session': [item['timing_protocol'] for item in results]},
                'segment_evidence': results}
    samples = []
    for index, item in enumerate(results):
        if not (item.get("ok") is True and item.get("correct") is True
                and item.get("reference_correct") is True):
            raise ValueError(f"RK segment {index} failed correctness/execution")
        for key in ("generated_sha", "parameter_plan", "vm_allocator"):
            if item.get(key) != first.get(key):
                raise ValueError(f"RK segment {index} changed {key}")
        records = item.get("timing_samples", [])
        count = item.get("timing_protocol", {}).get("scored_pair_count")
        if (not records or count is None or len(records) != count + 1
                or sum(record.get("discarded") is True for record in records) != 1):
            raise ValueError("RK segment sample counts differ")
        for local_index, record in enumerate(records):
            if (record.get("pair_index") != local_index
                    or record.get("discarded") is not (local_index == 0)
                    or record.get("order") != ("AB" if local_index % 2 == 0 else "BA")):
                raise ValueError("RK segment pair order differs")
            c, r = float(record["candidate_ms"]), float(record["reference_ms"])
            if not all(math.isfinite(x) and x > 0 for x in (c, r)):
                raise ValueError("RK segment has invalid latency")
            samples.append({**record, "segment_index": index,
                            "segment_pair_index": local_index,
                            "pair_index": len(samples), "ratio": c / r})
    scored = [record for record in samples if not record["discarded"]]
    if len(scored) != expected_pairs:
        raise ValueError("RK aggregate scored pair count differs")
    candidates = [record["candidate_ms"] for record in scored]
    references = [record["reference_ms"] for record in scored]
    ratios = [record["ratio"] for record in scored]
    mean = statistics.mean(ratios)
    return {
        **first, "timing_samples": samples,
        "stability": paired_stability(candidates, references),
        "latency_ms": statistics.median(candidates),
        "latency_batches_ms": candidates,
        "vs_base": {"median_ratio": statistics.median(ratios),
                    "ratio_cv": statistics.pstdev(ratios) / mean,
                    "pairs": len(scored), "base_latency_ms": statistics.median(references)},
        "timing_protocol": {
            **first["timing_protocol"],
            "session_scope": "rk3588_mali_fresh_process_segments_v1",
            "same_process_per_pair": True, "single_process_all_pairs": False,
            "session_count": len(results), "scored_pair_count": len(scored),
            "discard_prefix_pair_count": len(results),
            "per_session": [item["timing_protocol"] for item in results],
        },
        "segment_evidence": results,
    }


def run_opencl_segments(job_dir: str, job: dict) -> None:
    # OpenCL/Mali retains event-associated buffers until the context exits.
    # Use fresh processes for bounded segments to limit accumulated buffers
    # during long paired measurements.
    job_path = Path(job_dir)
    total = _timing_policy(job)["scored_pairs"]
    results = []
    deadline = time.monotonic() + 550
    try:
        profile_result = None
        if job.get("profile"):
            # Run telemetry before timing, in a fresh process with no timing
            # warmup/100-sample loop. Never keep its OpenCL context alive into
            # the scored segments or use instrumented latency as speedup.
            profile_name = "rk_profile.json"
            profile_proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), str(job_path),
                 "--profile-only", "--result-name", profile_name],
                capture_output=True, text=True,
                timeout=max(0.01, deadline - time.monotonic()),
            )
            if profile_proc.returncode != 0 or not (job_path / profile_name).is_file():
                raise RuntimeError(f"RK profile worker returncode={profile_proc.returncode}; {profile_proc.stderr[-1500:]}")
            profile_result = json.loads((job_path / profile_name).read_text())
            if not (profile_result.get("ok") is True and profile_result.get("correct") is True):
                finish(job_path, {**profile_result, "ok": False, "failed_profile_child": True})
                return
        for index, offset in enumerate(range(0, total, RK_TIMING_SEGMENT_SAMPLES)):
            count = min(RK_TIMING_SEGMENT_SAMPLES, total - offset)
            name = f"rk_segment_{index:03d}.json"
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), str(job_path),
                 "--segment-pairs", str(count), "--skip-profile", "--result-name", name],
                capture_output=True, text=True,
                timeout=max(0.01, deadline - time.monotonic()),
            )
            if proc.returncode != 0 or not (job_path / name).is_file():
                raise RuntimeError(f"RK segment {index} worker returncode={proc.returncode}; "
                                   f"{proc.stderr[-1500:]}")
            item = json.loads((job_path / name).read_text())
            if not (item.get("ok") is True and item.get("correct") is True
                    and (not job.get('reference_module_path') or item.get("reference_correct") is True)):
                # Preserve real correctness failures verbatim; they must not
                # become recoverable infrastructure timeouts in the resolver.
                finish(job_path, {**item, "ok": False, "failed_segment_index": index,
                                  "completed_segments": results})
                return
            # Validate each complete segment before dispatching another one.
            merge_opencl_segments([item], count)
            results.append(item)
        merged = merge_opencl_segments(results, total)
        if profile_result is not None:
            for key in ("generated_sha", "parameter_plan", "vm_allocator"):
                if profile_result.get(key) != merged.get(key):
                    raise ValueError(f"RK profile child changed {key}")
            for key in ("profile_breakdown", "profile_error", "profile_execution",
                        "profile_memory_before", "profile_memory_after"):
                if key in profile_result:
                    merged[key] = profile_result[key]
            merged["profile_child_evidence"] = profile_result
        finish(job_path, merged)
    except Exception:
        finish(job_path, {"ok": False, "fault_stage": "segmented_device_execution",
                          "error": traceback.format_exc()[-4000:],
                          "completed_segments": results})


def needs_opencl_segments(job: dict) -> bool:
    return (job.get("device") == "opencl" and job.get("mode") == "measure"
            and (bool(job.get("profile"))
                 or _timing_policy(job)["scored_pairs"] > RK_TIMING_SEGMENT_SAMPLES))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir")
    parser.add_argument("--segment-pairs", type=int)
    parser.add_argument("--result-name", default="result.json")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    cli = parser.parse_args()
    job = json.loads((Path(cli.job_dir) / "job.json").read_text())
    if cli.segment_pairs is None and not cli.profile_only and needs_opencl_segments(job):
        run_opencl_segments(cli.job_dir, job)
    else:
        main(cli.job_dir, segment_pairs=cli.segment_pairs, result_name=cli.result_name,
             skip_profile=cli.skip_profile, profile_only=cli.profile_only)
