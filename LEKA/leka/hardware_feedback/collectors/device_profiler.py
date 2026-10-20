"""Deep per-backend collectors feeding the trusted digest's `counters` section.

All three observe real jobs. Only NCU's scoped worker replay is bound to the
candidate bundle; process/global windows remain runtime context with unknown
phase overlap. Every collector returns {"counters": {...}, "rows": [...],
"source": "<collector id>", "notes": [...]}; failures return {"error": ...}
and are recorded, never fabricated.

- jetson_gpu: Nsight Compute wraps the on-device worker process; a bounded
  metric set is parsed from NCU's CSV log.  NCU serializes kernels and adds
  overhead: it generates bottleneck hypotheses, never speedup evidence.
- rk3588_gpu: opt-in HWCpipe, or low-risk devfreq snapshots immediately before
  and after the job — frequency context for the paired-ratio numbers.
- vivo_cpu: simpleperf whole-process PMU counters over the tvm_rpc process
  during the job window.  Counters are never apportioned across PrimFuncs.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import io
import json
import math
import os
import re
import select
import shlex
import subprocess
import threading
import time
from pathlib import PurePosixPath

from ...backends import BACKENDS
from ..quality import finite_number
from .android_process_identity import read_rpc_owner, scoped_stat_command, parse_generation_proof, runtime_snapshot_command, parse_runtime_snapshot_proof, parse_cpu_list
from .profile_artifacts import WindowArtifacts, archive_runtime_snapshot, POST_SNAPSHOT_ARTIFACT_SCHEMA, POST_SNAPSHOT_PARSER_SCHEMA
from . import nsight_launch_metadata as ncu_launch_facts
from ...operation_control import (
    OperationCancelled,
    OperationControl,
    run_cancellable,
)

# Nsight Compute runs with the target's scoped sudo permission. Its Python
# interpreter must have TVM available independently of a user's login shell.
NCU_BIN = os.environ.get("LEKA_NCU_BIN", "/usr/local/cuda/bin/ncu")
JETSON_PYTHON = os.environ.get("LEKA_JETSON_PYTHON", "python3")
JETSON_PYTHONUSERBASE = os.environ.get("LEKA_JETSON_PYTHONUSERBASE")
JETSON_DEVICE_ENV = (
    "PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE=1",
    *(
        ("PYTHONPATH=" + os.environ["LEKA_JETSON_PYTHONPATH"],)
        if os.environ.get("LEKA_JETSON_PYTHONPATH") else ()
    ),
)
NCU_METRICS = ",".join((
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "gpu__time_duration.sum",
))
_FIELD_OF_METRIC = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "stall_long_scoreboard_pct",
    "gpu__time_duration.sum": "kernel_time_ns_sum",
}


def _jetson_environment_setup(python: str, user_base: str | None) -> str:
    """Reproduce the normal SSH interpreter's user-site hooks under sudo.

    PYTHONPATH alone does not process .pth files. Editable TVM-FFI installs
    need those hooks to find core.so outside the source package directory.
    Resolve interpreter/user base before sudo; never repurpose HOME or rely
    on root's unrelated site-packages. An explicit deployment override wins.
    """
    resolve_python = shlex.quote("import sys; print(sys.executable)")
    resolve_base = shlex.quote("import site; print(site.getuserbase() if site.ENABLE_USER_SITE else '')")
    setup = (f"profile_python=$(PYTHONDONTWRITEBYTECODE=1 {shlex.quote(python)} -c {resolve_python}) || exit 1; "
             "case \"$profile_python\" in /*) ;; *) exit 1;; esac; ")
    if user_base is None:
        setup += f"profile_user_base=$(PYTHONDONTWRITEBYTECODE=1 \"$profile_python\" -c {resolve_base}) || exit 1; "
    else:
        setup += f"profile_user_base={shlex.quote(user_base)}; "
    return setup + ("if test -n \"$profile_user_base\"; then "
                    "profile_user_site_arg=\"PYTHONUSERBASE=$profile_user_base\"; "
                    "else profile_user_site_arg=PYTHONNOUSERSITE=1; fi; ")


def _validated_remote_job(remote_root: str, job_name: str) -> tuple[str, str]:
    """Return a cleanup-safe remote root/job pair.

    The collector removes only the unique job it created.  Validate both
    components before ever constructing the remote ``rm`` command.
    """

    root = PurePosixPath(str(remote_root))
    if not root.is_absolute() or root == PurePosixPath("/") or ".." in root.parts:
        raise ValueError("NCU remote root must be a specific absolute directory")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", job_name) is None:
        raise ValueError("NCU job name contains unsafe characters")
    remote = root / job_name
    if remote.parent != root:
        raise ValueError("NCU remote job escaped its configured root")
    return str(root), str(remote)


def _cleanup_jetson_ncu_job(remote: str, ssh_mux_opts: tuple[str, ...]) -> str | None:
    """Bounded cleanup that still runs after the shared token is cancelled.

    The remote wrapper records only PIDs whose command lines contain the exact
    unique job path.  This cleanup first asks those scoped processes to exit,
    verifies that they did, and then removes only that job directory.  It uses
    a short independent timeout because the exploration token may already be
    cancelled; failure is returned so the caller can fail closed.
    """

    quoted_remote = shlex.quote(remote)
    cleanup_cmd = (
        f"remote={quoted_remote}; alive=0; "
        "for pid_file in \"$remote/.ncu_shell.pid\" \"$remote/.ncu_child.pid\"; do "
        "  if test -r \"$pid_file\"; then "
        "    pid=$(cat \"$pid_file\" 2>/dev/null || true); "
        "    case \"$pid\" in ''|*[!0-9]*) continue;; esac; "
        "    if test -r \"/proc/$pid/cmdline\" && "
        "       tr '\\000' ' ' <\"/proc/$pid/cmdline\" | grep -F -- \"$remote\" >/dev/null; then "
        "      kill -TERM \"$pid\" 2>/dev/null || true; "
        "    fi; "
        "  fi; "
        "done; "
        "for delay in 1 2 3 4 5 6 7 8 9 10; do "
        "  alive=0; "
        "  for pid_file in \"$remote/.ncu_shell.pid\" \"$remote/.ncu_child.pid\"; do "
        "    if test -r \"$pid_file\"; then "
        "      pid=$(cat \"$pid_file\" 2>/dev/null || true); "
        "      case \"$pid\" in ''|*[!0-9]*) continue;; esac; "
        "      if test -r \"/proc/$pid/cmdline\" && "
        "         tr '\\000' ' ' <\"/proc/$pid/cmdline\" | grep -F -- \"$remote\" >/dev/null; then "
        "        alive=1; "
        "      fi; "
        "    fi; "
        "  done; "
        "  test \"$alive\" -eq 0 && break; sleep 0.2; "
        "done; "
        "rm -rf -- \"$remote\"; test \"$alive\" -eq 0 && test ! -e \"$remote\""
    )
    try:
        cleanup = run_cancellable(
            ("ssh", *ssh_mux_opts, BACKENDS["jetson_gpu"]["ssh_alias"], cleanup_cmd),
            control=None,
            capture_output=True,
            text=True,
            timeout=12,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"bounded NCU cleanup failed: {error}"
    if cleanup.returncode != 0:
        return "bounded NCU cleanup could not verify process and directory removal"
    return None


def collect_jetson_ncu(
    service,
    module_source: str,
    binding: dict,
    patch_source: str | None = None,
    t12_sha: str | None = None,
    *,
    operation_control: OperationControl | None = None,
) -> dict:
    """Run one small measure job under NCU on the jetson; parse the CSV log."""
    if operation_control is None:
        operation_control = getattr(service, "operation_control", None)
    if operation_control is not None:
        operation_control.checkpoint()
    job_dir = service._base_job(module_source, "measure", binding,
                                patch_source, t12_sha, False, None)
    raw_artifacts = {"schema_version": "leka.ncu_job_artifacts.v1",
                     "job_dir": str(job_dir.absolute()), "files": {}}
    _bound_ncu_job(job_dir)
    from ...evaluation.evaluation_service import (
        CANDIDATE_SOURCE_GUARD, DEVICE_WORKER, PARAMETER_PLAN_HELPER,
        TIMING_QUALITY_HELPER,
    )
    from ...evaluation.evaluation_service import SSH_MUX_OPTS
    (job_dir / "device_worker.py").write_text(DEVICE_WORKER.read_text())
    (job_dir / "candidate_source.py").write_text(CANDIDATE_SOURCE_GUARD.read_text())
    # Include every helper imported by the isolated device worker.
    (job_dir / "parameter_plan.py").write_text(PARAMETER_PLAN_HELPER.read_text())
    (job_dir / "timing_quality.py").write_text(TIMING_QUALITY_HELPER.read_text())
    remote_root, remote = _validated_remote_job(
        service.runner.remote_root, job_dir.name
    )
    remote_owned = False
    try:
        setup = run_cancellable(
            ("ssh", *SSH_MUX_OPTS, BACKENDS["jetson_gpu"]["ssh_alias"],
             f"mkdir -p {shlex.quote(remote_root)} && "
             f"test ! -e {shlex.quote(remote)}"),
            control=operation_control,
            capture_output=True,
            text=True,
            timeout=60,
        )
        setup.check_returncode()
        remote_owned = True
        push = run_cancellable(
            ("rsync", "-a", "-e", "ssh " + " ".join(SSH_MUX_OPTS),
             str(job_dir) + "/", f"{BACKENDS['jetson_gpu']['ssh_alias']}:{remote}/"),
            control=operation_control,
            capture_output=True,
            text=True,
            timeout=300,
        )
        push.check_returncode()
        env_args = " ".join(shlex.quote(e) for e in JETSON_DEVICE_ENV)
        quoted_remote = shlex.quote(remote)
        worker = shlex.quote(f"{remote}/device_worker.py")
        wrapper = (
            _jetson_environment_setup(JETSON_PYTHON, JETSON_PYTHONUSERBASE)
            +
            f"cd {quoted_remote} || exit 1; "
            "echo $$ > .ncu_shell.pid; ncu_pid=''; "
            "on_signal() { test -z \"$ncu_pid\" || kill -TERM \"$ncu_pid\" "
            "2>/dev/null || true; exit 143; }; "
            "trap on_signal HUP INT TERM; "
            f"sudo -n {shlex.quote(NCU_BIN)} --config-file off --csv --page raw "
            "--print-units base --print-kernel-base function "
            "--print-summary per-kernel --replay-mode kernel "
            f"--metrics {NCU_METRICS} "
            "--cache-control all --clock-control none --check-exit-code yes "
            "--target-processes all "
            f"/usr/bin/env {env_args} \"$profile_user_site_arg\" \"$profile_python\" {worker} {quoted_remote} & "
            "ncu_pid=$!; echo \"$ncu_pid\" > .ncu_child.pid; wait \"$ncu_pid\"; "
            "status=$?; rm -f .ncu_shell.pid .ncu_child.pid; "
            "trap - HUP INT TERM; exit \"$status\""
        )
        proc = run_cancellable(
            ("ssh", *SSH_MUX_OPTS, BACKENDS["jetson_gpu"]["ssh_alias"], wrapper),
            control=operation_control,
            capture_output=True,
            text=True,
            timeout=900,
        )
        for name, value in (("ncu_stdout.log", proc.stdout), ("ncu_stderr.log", proc.stderr)):
            raw_artifacts["files"][name] = _save_ncu_artifact(job_dir / name, value or "")
        if operation_control is not None:
            operation_control.checkpoint()
        # device_worker deliberately exits zero after writing structured
        # failure results. Successful NCU transport alone cannot establish
        # that the instrumented candidate passed its correctness oracle. Read
        # this even when no CSV was produced so setup failure remains visible.
        result_read = run_cancellable(
            ("ssh", *SSH_MUX_OPTS, BACKENDS["jetson_gpu"]["ssh_alias"],
             f"head -c 1048577 {shlex.quote(remote + '/result.json')}"),
            control=operation_control, capture_output=True, text=True, timeout=15,
        )
        raw_artifacts["files"]["ncu_worker_result.json"] = _save_ncu_artifact(
            job_dir / "ncu_worker_result.json", result_read.stdout or "")
        if proc.returncode != 0:
            return _ncu_failure(f"ncu exited with status {proc.returncode}", proc, result_read.stdout, raw_artifacts)
        if result_read.returncode != 0:
            return _ncu_failure("NCU worker result could not be read", proc, result_read.stdout, raw_artifacts)
        verified = _verify_ncu_worker_result(result_read.stdout)
        if "error" in verified:
            return _ncu_failure(verified["error"], proc, result_read.stdout, raw_artifacts)
        report = _ncu_report_stream(proc.stdout, proc.stderr)
        if report is None:
            return _ncu_failure("ncu produced no unique recognized csv stream", proc, result_read.stdout, raw_artifacts)
        parsed = _parse_ncu_csv(report)
        if "error" in parsed:
            return _ncu_failure(parsed["error"], proc, result_read.stdout, raw_artifacts)
        parsed["provenance"].update(verified)
        parsed["provenance"]["raw_artifacts"] = raw_artifacts
        return parsed
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as err:
        return {"error": str(err)[:800], "provenance": {"raw_artifacts": raw_artifacts}}
    finally:
        if remote_owned:
            cleanup_error = _cleanup_jetson_ncu_job(remote, SSH_MUX_OPTS)
            if cleanup_error is not None:
                raise RuntimeError(cleanup_error)


def _save_ncu_artifact(path, text: str) -> dict:
    payload = text.encode("utf-8")
    # The per-job directory is evaluator-owned and the names are fixed here.
    # O_EXCL prevents accidentally replacing any prior/symlinked artifact.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    return {"path": str(path.absolute()), "saved_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "encoding": "utf8_from_subprocess_text"}


def _ncu_failure(reason: str, proc, worker_text: str, raw_artifacts: dict | None = None) -> dict:
    from ...evaluation.diagnostics import sanitize_report_text
    details = [reason, f"stdout_csv_header={_has_ncu_csv_header(proc.stdout or '')}",
               f"stderr_csv_header={_has_ncu_csv_header(proc.stderr or '')}"]
    errors = [line for stream in (proc.stdout or "", proc.stderr or "")
              for line in stream.splitlines() if "==ERROR==" in line][:4]
    if errors:
        details.append(sanitize_report_text("; ".join(errors), max_chars=600))
    try:
        worker = json.loads(worker_text)
        if isinstance(worker, dict):
            details.append("worker_fault_stage=" + sanitize_report_text(worker.get("fault_stage"), max_chars=100))
            if worker.get("error"):
                details.append("worker_error=" + sanitize_report_text(worker["error"], max_chars=600))
    except (ValueError, RecursionError):
        pass
    result = {"error": "; ".join(details)}
    if raw_artifacts is not None:
        result["provenance"] = {"raw_artifacts": raw_artifacts}
    return result


def _bound_ncu_job(job_dir) -> None:
    """Bound instrumented dispatches without changing normal timing policy.

    The evaluator takes its timing settings from backend_cfg. Set the profiler
    job to one dispatch per evaluation and disable duration-based warmup and
    min_repeat_ms expansion. Normal measurement jobs keep their timing policy.
    """
    path = job_dir / "job.json"
    job = json.loads(path.read_text())
    cfg = dict(job.get("backend_cfg") or {})
    cfg.update({
        "timing_protocol": "ncu_bounded_diagnostic_only_v1",
        "timing_scored_pair_count": 2,
        "timing_evaluator_target_window_ms": 0.001,
        "timing_evaluator_maximum_number": 1,
        "timing_minimum_warmup_invocations": 1,
    })
    job.update(backend_cfg=cfg, warmup=1, repeat=1, profile=False)
    path.write_text(json.dumps(job, indent=2, allow_nan=False))


def _verify_ncu_worker_result(text: str) -> dict:
    if len(text.encode("utf-8")) > 1024 * 1024:
        return {"error": "NCU worker result exceeded size bound"}
    try:
        result = json.loads(text)
    except (ValueError, RecursionError):
        return {"error": "NCU worker result was not valid JSON"}
    if not isinstance(result, dict) or result.get("ok") is not True or result.get("correct") is not True:
        return {"error": "NCU worker did not prove successful execution and correctness"}
    policy = result.get("timing_protocol") or {}
    if (not isinstance(policy, dict)
            or policy.get("name") != "ncu_bounded_diagnostic_only_v1"
            or policy.get("candidate_evaluator_number") != 1
            or policy.get("scored_pair_count") != 2):
        return {"error": "NCU worker did not use the bounded diagnostic timing policy"}
    return {"worker_correctness_verified": True,
            "worker_result_sha256": hashlib.sha256(text.encode()).hexdigest()}


def _has_ncu_csv_header(text: str) -> bool:
    raw_required = {
        "ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"
    }
    summary_required = {
        "Process ID", "Kernel Name", "Invocations", "Metric Name",
        "Metric Unit", "Minimum", "Maximum", "Average",
    }
    for line in text.splitlines():
        if not line or line.lstrip().startswith("=="):
            continue
        try:
            row = next(csv.reader((line,)))
        except (csv.Error, StopIteration):
            continue
        fields = {str(value).lstrip("\ufeff").strip() for value in row}
        if raw_required.issubset(fields) or summary_required.issubset(fields):
            return True
    return False


def _ncu_report_stream(stdout: str, stderr: str) -> str | None:
    """Select exactly one NCU table stream without merging or duplicating it."""

    stdout_has = _has_ncu_csv_header(stdout)
    stderr_has = _has_ncu_csv_header(stderr)
    if stdout_has == stderr_has:
        return None
    return stdout if stdout_has else stderr


def _parse_ncu_csv(text: str, *, include_launch_facts: bool = True) -> dict:
    """Parse both NCU raw-launch and physical Jetson per-kernel CSV shapes.

    NCU 2024.3 on the lab Jetson emits a summary row per
    ``(kernel-group, metric)`` when ``--print-summary per-kernel`` is used.
    Older fixtures expose one raw row per launch.  Keep the two semantics
    explicit: summary values use ``Average`` and ``Invocations``; raw values
    use ``Metric Value``.  Percentage aggregates are duration weighted.
    """

    if not isinstance(text, str) or len(text.encode("utf-8")) > 8 * 1024 * 1024:
        return {"error": "ncu csv is not bounded text"}
    lines = [
        line for line in text.splitlines()
        if not line.lstrip().startswith("==")
    ]
    raw_required = {
        "ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"
    }
    summary_required = {
        "Process ID", "Kernel Name", "Invocations", "Metric Name",
        "Metric Unit", "Minimum", "Maximum", "Average",
    }
    start = None
    fields: set[str] = set()
    for index, line in enumerate(lines):
        try:
            parsed = next(csv.reader((line,)))
        except (csv.Error, StopIteration):
            continue
        candidate = {str(value).lstrip("\ufeff").strip() for value in parsed}
        if raw_required.issubset(candidate) or summary_required.issubset(candidate):
            start, fields = index, candidate
            break
    if start is None:
        return {"error": "no recognized csv header in ncu log"}
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    if reader.fieldnames is not None:
        reader.fieldnames = [
            str(name).lstrip("\ufeff").strip() for name in reader.fieldnames
        ]

    def number(raw: object) -> float | None:
        try:
            value = float(str(raw or "").replace(",", "").strip())
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    def duration_ns(value: float, unit: str) -> float | None:
        scale = {
            "ns": 1.0,
            "nsecond": 1.0,
            "nseconds": 1.0,
            "us": 1_000.0,
            "usecond": 1_000.0,
            "useconds": 1_000.0,
            "ms": 1_000_000.0,
            "msecond": 1_000_000.0,
            "mseconds": 1_000_000.0,
            "s": 1_000_000_000.0,
            "second": 1_000_000_000.0,
            "seconds": 1_000_000_000.0,
        }.get(unit.strip().lower())
        return value * scale if scale is not None else None

    grouped: dict[tuple[str, ...], dict] = {}
    summary = summary_required.issubset(fields)
    dropped_metric_rows = {"missing_kernel_name": 0, "invalid_invocations": 0, "invalid_value": 0}
    for record in reader:
        metric = str(record.get("Metric Name") or "").strip()
        field = _FIELD_OF_METRIC.get(metric)
        is_launch_fact = include_launch_facts and metric in ncu_launch_facts.METRIC_UNITS
        if field is None and not is_launch_fact:
            continue
        kernel = str(record.get("Kernel Name") or "").strip()
        if not kernel:
            dropped_metric_rows["missing_kernel_name"] += 1
            continue
        if summary:
            identity_fields = (
                "Process ID", "Host Name", "Device", "Context", "Stream", "CC", "Block Size",
                "Grid Size", "Kernel Name",
            )
            count_value = number(record.get("Invocations"))
            value = number(record.get("Average"))
            if (
                count_value is None or not count_value.is_integer()
                or count_value <= 0
            ):
                dropped_metric_rows["invalid_invocations"] += 1
                continue
            count = int(count_value)
        else:
            identity_fields = (
                "Process ID", "Device", "Context", "Stream", "ID", "Kernel Name",
            )
            count = 1
            value = number(record.get("Metric Value"))
        identity = {key: str(record.get(key) or "").strip() for key in identity_fields}
        group = tuple(identity.values())
        row = grouped.setdefault(group, {"name": kernel, "invocations": count,
            "ncu_group_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            "ncu_group_identity": identity,
            "row_scope": "ncu_kernel_summary" if summary else "ncu_kernel_launch",
            "static_primfunc_id": "unknown"})
        if row["invocations"] != count:
            return {"error": "ncu invocation count changed within one kernel group"}
        if is_launch_fact:
            ncu_launch_facts.record_fact(row, metric, record, summary=summary)
            continue
        if value is None:
            dropped_metric_rows["invalid_value"] += 1
            continue
        if field == "kernel_time_ns_sum":
            converted = duration_ns(value, str(record.get("Metric Unit") or ""))
            if converted is None:
                return {"error": "ncu duration metric used an unsupported unit"}
            if not math.isfinite(converted) or converted < 0 or not math.isfinite(converted * count):
                return {"error": "ncu converted duration was not finite/nonnegative"}
            if field in row and row[field] != converted * count:
                return {"error": "ncu duration changed within one kernel group"}
            row["mean_kernel_time_ns"] = converted
            row[field] = converted * count
            if summary:
                minimum = number(record.get("Minimum"))
                maximum = number(record.get("Maximum"))
                if minimum is not None and maximum is not None:
                    if not 0 <= minimum <= value <= maximum:
                        return {"error": "ncu duration summary range was inconsistent"}
                    row["min_kernel_time_ns"] = duration_ns(
                        minimum, str(record.get("Metric Unit") or "")
                    )
                    row["max_kernel_time_ns"] = duration_ns(
                        maximum, str(record.get("Metric Unit") or "")
                    )
        else:
            if str(record.get("Metric Unit") or "").strip().lower() not in {"%", "percent"}:
                return {"error": "ncu percentage metric used an unsupported unit"}
            if field in row and row[field] != value:
                return {"error": "ncu metric changed within one kernel group"}
            row[field] = value
        if value < 0:
            return {"error": "ncu metric was negative"}
    rows = list(grouped.values())
    if not rows or not any("kernel_time_ns_sum" in row for row in rows):
        return {"error": "ncu csv parsed but no timed known metrics were found"}

    counters: dict[str, float | int] = {}
    total_time = sum(float(row.get("kernel_time_ns_sum") or 0.0) for row in rows)
    if not math.isfinite(total_time) or total_time <= 0:
        return {"error": "ncu report has no positive finite kernel duration"}
    launch_count = sum(int(row.get("invocations") or 0) for row in rows)
    for row in rows:
        if include_launch_facts:
            ncu_launch_facts.finish_facts(row)
        if "kernel_time_ns_sum" in row:
            row["replay_kernel_time_percent"] = 100 * row["kernel_time_ns_sum"] / total_time
    coverage: dict[str, float] = {}
    metric_groups: dict[str, list[str]] = {}
    for field in set(_FIELD_OF_METRIC.values()) - {"kernel_time_ns_sum"}:
        samples = [
            (float(row[field]), float(row["kernel_time_ns_sum"]))
            for row in rows
            if field in row and float(row.get("kernel_time_ns_sum") or 0.0) > 0
        ]
        weight = sum(item[1] for item in samples)
        coverage[field] = weight / total_time
        metric_groups[field] = [row["ncu_group_id"] for row in rows
            if field in row and float(row.get("kernel_time_ns_sum") or 0) > 0]
        if weight > 0:
            counters[field] = sum(value * duration for value, duration in samples) / weight
    counters["kernel_launches_profiled"] = launch_count
    counters["total_kernel_time_ns"] = total_time
    return {
        "source": "jetson_ncu",
        "counters": counters,
        "rows": rows,
        "provenance": {
            **({"launch_facts_schema": ncu_launch_facts.SCHEMA} if include_launch_facts else {}),
            "scope": "per_kernel_replay",
            "measurement_phase": "whole_worker_including_warmup_and_correctness",
            "normal_execution_timing": False,
            "aggregation": "duration_weighted_kernel_summary_approximation" if summary else "duration_weighted_launch_metrics",
            "metric_timed_coverage": coverage,
            "aggregate_counter_scope": "across_timed_kernel_groups_with_per_metric_coverage",
            "aggregate_metric_group_ids": metric_groups,
            "kernel_group_count": len(rows),
            "timed_kernel_group_count": sum("kernel_time_ns_sum" in row for row in rows),
            "untimed_kernel_group_count": sum("kernel_time_ns_sum" not in row for row in rows),
            "dropped_metric_rows": dropped_metric_rows,
            "replay_percent_denominator": "sum_of_observed_timed_kernel_group_durations_not_vm_or_walltime",
            "invocation_count_scope": "whole_diagnostic_worker_not_one_saved_entry_call",
            "kernel_to_primfunc_mapping": "unknown",
            "cache_control": "all",
            "clock_control": "none",
            "timing_policy": "ncu_bounded_diagnostic_only_v1",
            "scored_samples": 2,
            "maximum_evaluator_number": 1,
        },
        "notes": [
            "NCU exact-bundle kernel replay serializes and instruments execution; "
            "it supplies bottleneck hypotheses and is never wall-time speed evidence",
            "Summary averages weighted by average duration approximate launch-level "
            "weighting; absent metrics reduce coverage and must not be read as zeros",
        ],
    }


MALI_SAMPLER = os.environ.get("LEKA_MALI_SAMPLER", "mali-sampler")
MALI_ACTIVITY = "GPU active cycles"
# Compute-relevant Valhall counters; exact descriptive names verified via
# `mali-sampler list` on the physical board (253 supported on G610).
MALI_COUNTERS = (MALI_ACTIVITY,
                 "Execution core utilization",
                 "FMA pipe utilization",
                 "CVT pipe utilization",
                 "Load/store unit utilization",
                 "L2 cache read miss percentage")


def _join_sampler(
    thread: threading.Thread,
    stop: threading.Event,
    operation_control: OperationControl | None,
    *,
    timeout_s: float = 6.0,
) -> None:
    """Stop and join a sampler; a lingering sampler cancels the whole phase."""

    stop.set()
    thread.join(timeout=timeout_s)
    if thread.is_alive() and operation_control is not None:
        operation_control.cancel("hardware sampler did not stop")
        thread.join(timeout=5)
    if thread.is_alive():
        raise RuntimeError("hardware sampler did not stop; operation fails closed")


def _sampler_cancelled_normally(
    stop: threading.Event, operation_control: OperationControl | None
) -> bool:
    return stop.is_set() and not (
        operation_control is not None and operation_control.is_cancelled()
    )


def _stop_streaming_process(
    proc: subprocess.Popen,
    *,
    graceful_timeout_s: float = 0.0,
    timeout_s: float = 5.0,
) -> None:
    # A finite hardware-counter sampler should normally reach stop_sampling()
    # itself.  Give it that opportunity before closing the SSH transport; an
    # abrupt transport teardown may strand vendor-driver counter state.
    if proc.poll() is None and graceful_timeout_s > 0:
        try:
            proc.wait(timeout=graceful_timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass
    if proc.poll() is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    if proc.poll() is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("hardware sampler process did not exit") from error


def _drain_bounded_stderr(stream, chunks: list[str], *, limit: int = 16_384,
                          diagnostics: dict | None = None) -> None:
    """Drain to EOF without blocking the sampler on a full stderr pipe.

    Retain only a bounded prefix (at most 64 KiB in UTF-8), but keep reading
    excess output until the finite subprocess exits. The owning thread joins
    this reader after stopping/waiting for the subprocess.
    """
    retained = 0
    try:
        while True:
            chunk = stream.read(2048)
            if not chunk:
                return
            if diagnostics is not None:
                diagnostics["original_bytes"] = diagnostics.get("original_bytes", 0) + len(chunk.encode("utf-8"))
                diagnostics["truncated"] = diagnostics.get("truncated", False) or retained + len(chunk) > limit
            if retained < limit:
                fragment = chunk[:limit - retained]
                chunks.append(fragment)
                retained += len(fragment)
    except (OSError, ValueError):
        return  # process status remains the authority; retained text is best effort


def _mali_process_failure(returncode: int | None, stderr_text: str) -> str | None:
    if returncode in (0, None):
        return None
    from ...evaluation.diagnostics import sanitize_report_text
    category = "transport" if returncode == 255 else "sampler"
    detail = sanitize_report_text(stderr_text, max_chars=600)
    return f"Mali {category} exited with status {returncode}" + (f": {detail}" if detail else "")


def collect_rk3588_hwcpipe_during(
    runner_job_fn,
    *args,
    counters=MALI_COUNTERS,
    interval_ms: int = 200,
    sample_count: int = 50,
    operation_control: OperationControl | None = None,
    artifact_dir=None,
    **kwargs,
) -> dict:
    """Sample Mali HW counters (libGPUCounters/HWCPipe) on the board while a
    measure job runs; keep the busiest window (max GPU active cycles).
    Values are per-interval deltas; utilization counters are percentages."""
    samples: list[dict] = []
    stop = threading.Event()
    job_running = threading.Event()
    sampler_errors: list[OperationCancelled] = []
    sampler_failures: list[str] = []
    stderr_chunks: list[str] = []
    stderr_diagnostics: dict = {}
    from ...evaluation.evaluation_service import SSH_MUX_OPTS

    if operation_control is not None:
        operation_control.checkpoint()
    if interval_ms <= 0 or sample_count <= 0:
        raise ValueError("Mali sampler interval and count must be positive")

    artifacts = (WindowArtifacts(artifact_dir, collector="rk3588_hwcpipe",
                 parser_schema="leka.hwcpipe_jsonl.v1") if artifact_dir is not None else None)
    artifact_summary = None
    best_activity = 0.0
    selected_artifact_index = None

    remote_cmd = (f"{shlex.quote(MALI_SAMPLER)} sample {interval_ms} {sample_count} "
                  + " ".join(shlex.quote(c) for c in counters))
    try:
        proc = subprocess.Popen(
            ("ssh", *SSH_MUX_OPTS, BACKENDS["rk3588_gpu"]["ssh_alias"], remote_cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        if artifacts is not None:
            artifact_summary = artifacts.finish(exact_bundle=False,
                transport_start_exception=type(error).__name__)
        result = runner_job_fn(*args, **kwargs)
        if operation_control is not None:
            operation_control.checkpoint()
        return {
            "error": f"unable to start Mali sampler transport: {error}",
            "provenance": {"exact_bundle": False, "artifacts": artifact_summary},
            "job_result": result,
        }

    def sampler():
        nonlocal best_activity, selected_artifact_index
        try:
            assert proc.stdout is not None
            while not stop.is_set():
                if operation_control is not None:
                    operation_control.checkpoint()
                ready, _, _ = select.select((proc.stdout,), (), (), 0.2)
                if not ready:
                    if proc.poll() is not None:
                        return
                    continue
                line = proc.stdout.readline(1_048_577)
                if not line:
                    if proc.poll() is not None:
                        return
                    continue
                running = job_running.is_set()
                sample = None
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    pass
                cleaned = ({key: value for key, value in sample.items()
                            if key in counters and finite_number(value) is not None}
                           if isinstance(sample, dict) and running else {})
                activity = cleaned.get(MALI_ACTIVITY, 0.0)
                if artifacts is not None:
                    window_index = artifacts.record(line, "", {
                        "received_unix_ns": time.time_ns(), "received_monotonic_ns": time.monotonic_ns(),
                        "job_running_at_receive": running, "valid_json_object": isinstance(sample, dict),
                        "parsed_counters": cleaned, "interval_requested_ms": interval_ms,
                        "stream_fragment_may_be_incomplete": len(line) >= 1_048_577 and not line.endswith("\n")},
                        selected=activity > best_activity)
                    if activity > best_activity:
                        selected_artifact_index = window_index
                best_activity = max(best_activity, activity)
                if cleaned:
                    samples.append(cleaned)
        except OperationCancelled as error:
            if not _sampler_cancelled_normally(stop, operation_control):
                sampler_errors.append(error)
        except (OSError, ValueError, TypeError) as error:
            sampler_failures.append(type(error).__name__)

    thread = threading.Thread(target=sampler, daemon=True)
    stderr_stream = getattr(proc, "stderr", None)
    stderr_thread = (threading.Thread(target=_drain_bounded_stderr,
                     args=(stderr_stream, stderr_chunks), kwargs={"diagnostics": stderr_diagnostics}, daemon=True)
                     if stderr_stream is not None else None)
    thread.start()
    if stderr_thread is not None:
        stderr_thread.start()
    try:
        job_running.set()
        result = runner_job_fn(*args, **kwargs)
    finally:
        job_running.clear()
        # Allow the finite sampler (10 seconds by default) to close
        # libGPUCounters before ending its SSH connection.
        remaining_window_s = interval_ms * sample_count / 1000.0
        _stop_streaming_process(
            proc,
            graceful_timeout_s=min(remaining_window_s + 2.0, 15.0),
        )
        stop.set()
        _join_sampler(thread, stop, operation_control)
        if stderr_thread is not None:
            _join_sampler(stderr_thread, stop, operation_control)
        if artifacts is not None:
            artifact_summary = artifacts.finish(process_stderr="".join(stderr_chunks),
                process_stderr_capture=stderr_diagnostics, returncode=proc.poll(), exact_bundle=False,
                requested_counters=list(counters),
                selection="maximum_gpu_active_cycles_biased_sample",
                sampler_failures=sampler_failures, sampler_cancelled=bool(sampler_errors))
    if sampler_errors:
        raise sampler_errors[0]
    if operation_control is not None:
        operation_control.checkpoint()
    process_failure = _mali_process_failure(proc.poll(), "".join(stderr_chunks))
    if process_failure is not None:
        return {"error": process_failure, "job_result": result,
                "provenance": {"exact_bundle": False, "artifacts": artifact_summary}}
    if sampler_failures:
        return {"error": "Mali sampler failed: " + ", ".join(sampler_failures), "job_result": result,
                "provenance": {"exact_bundle": False, "artifacts": artifact_summary}}
    active = [s for s in samples if s.get(MALI_ACTIVITY, 0) > 0]
    if not active:
        return {"error": f"no active Mali windows captured ({len(samples)} samples)",
                "provenance": {"exact_bundle": False, "artifacts": artifact_summary},
                "job_result": result}
    busy = max(active, key=lambda s: s.get(MALI_ACTIVITY, 0))
    counters_out = {k: v for k, v in busy.items() if k != "t_ms"}
    counters_out["active_windows"] = len(active)
    return {"source": "rk3588_hwcpipe", "counters": counters_out, "rows": [],
            "provenance": {
                "scope": "gpu_global_window", "exact_bundle": False,
                "measurement_phase_overlap": "unproven",
                "selection": "maximum_gpu_active_cycles_biased_sample",
                "sampling_window_ms": interval_ms, "sample_limit": sample_count,
                "captured_samples": len(samples), "active_samples": len(active),
                "excludes_post_job_received_samples": True,
                "other_gpu_work_excluded": False,
                "selected_window_index": selected_artifact_index,
                "artifacts": artifact_summary,
            },
            "notes": [f"busiest of {len(active)} active {interval_ms}ms windows; "
                      "per-interval deltas from libGPUCounters on /dev/mali0"],
            "job_result": result}


def collect_rk3588_devfreq(
    runner_job_fn,
    *args,
    operation_control: OperationControl | None = None,
    **kwargs,
) -> dict:
    """Read Mali devfreq once before and once after a measure job.

    Two snapshots limit SSH traffic when the device is unavailable. They
    provide frequency context without attributing readings to individual
    kernels.
    """
    freqs: list[int] = []
    read_errors: list[str] = []
    from ...evaluation.evaluation_service import SSH_MUX_OPTS

    def snapshot(label: str) -> None:
        if operation_control is not None:
            operation_control.checkpoint()
        try:
            proc = run_cancellable(
                (
                    "ssh",
                    *SSH_MUX_OPTS,
                    BACKENDS["rk3588_gpu"]["ssh_alias"],
                    "cat /sys/class/devfreq/fb000000.gpu/cur_freq",
                ),
                control=operation_control,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as error:
            read_errors.append(f"{label} frequency snapshot failed: {type(error).__name__}")
            return
        if proc.returncode != 0:
            read_errors.append(f"{label} frequency snapshot failed")
            return
        try:
            frequency = int(proc.stdout.strip())
            if frequency <= 0:
                raise ValueError("nonpositive frequency")
            freqs.append(frequency)
        except ValueError:
            read_errors.append(f"{label} frequency snapshot was not an integer")

    snapshot("before")
    result = runner_job_fn(*args, **kwargs)
    trusted = getattr(result, "trusted", {})
    raw = trusted.get("raw", {}) if isinstance(trusted, dict) else {}
    if not (isinstance(raw, dict) and raw.get("fault_stage") == "transport"):
        snapshot("after")
    else:
        read_errors.append("after frequency snapshot skipped after transport failure")
    telemetry = {}
    if freqs:
        telemetry = {"gpu_freq_min_hz": min(freqs), "gpu_freq_max_hz": max(freqs),
                     "gpu_freq_mean_hz": sum(freqs) / len(freqs),
                     "gpu_freq_samples": len(freqs)}
    wrapped = {"source": "rk3588_devfreq", "counters": telemetry, "rows": [],
               "provenance": {
                   "scope": "before_after_job_frequency_snapshots", "exact_bundle": False,
                   "measurement_phase_overlap": "unproven",
                   "min_max_are_window_extrema": False,
                   "snapshot_count": len(freqs),
               },
               "notes": ["before/after snapshots only: frequency context for paired "
                         "ratios, not kernel attribution"],
               "job_result": result}
    if read_errors:
        wrapped["error"] = "; ".join(read_errors)
    return wrapped


VIVO_EVENTS = "cpu-cycles,instructions,cache-references,cache-misses,branch-misses"


def collect_vivo_runtime_snapshot(
    serial: str,
    *,
    operation_control: OperationControl | None = None,
    expected_owner: dict | None = None,
    require_expected_owner: bool = False,
    artifact_dir=None,
) -> dict:
    """Collect a bounded post-measure placement/frequency/thermal snapshot.

    Process context is accepted only after owned-root and task generation
    proofs. Device-global sysfs observations have separate provenance; neither
    source establishes what ran during the timed computation.
    """

    if type(require_expected_owner) is not bool:
        raise ValueError("require_expected_owner must be boolean")
    owner = (expected_owner if expected_owner is not None or require_expected_owner
             else read_rpc_owner(serial))
    owner = copy.deepcopy(owner)
    owner_source = ("runner_fresh_owner_gate" if require_expected_owner else
                    "explicit_owner" if expected_owner else "configured_owned_receipt")
    script = runtime_snapshot_command(owner)
    started_unix_ns, started_monotonic_ns = time.time_ns(), time.monotonic_ns()
    stdout, stderr, returncode, transport_error = "", "", None, None
    try:
        proc = run_cancellable(
            ("adb", "-s", serial, "shell", script),
            control=operation_control, capture_output=True, text=True, timeout=15,
        )
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except OperationCancelled:
        raise  # Optional archival must never replace or suppress cancellation.
    except Exception as error:
        transport_error = type(error).__name__
    ended_unix_ns, ended_monotonic_ns = time.time_ns(), time.monotonic_ns()
    if operation_control is not None:
        operation_control.checkpoint()
    artifact_receipt = None
    if artifact_dir is not None:
        try:
            artifact_receipt = archive_runtime_snapshot(artifact_dir, stdout=stdout, stderr=stderr,
                metadata={"schema_version": POST_SNAPSHOT_PARSER_SCHEMA,
                    "expected_rpc_owner": owner, "owner_source": owner_source,
                    "started_unix_ns": started_unix_ns, "ended_unix_ns": ended_unix_ns,
                    "started_monotonic_ns": started_monotonic_ns, "ended_monotonic_ns": ended_monotonic_ns,
                    "returncode": returncode, "output_available": transport_error is None,
                    "transport_error": transport_error,
                    "scope": "post_measure_owned_rpc_root_tasks_and_separate_global_sysfs",
                    "exact_bundle": False, "timed_phase_overlap": "unproven"})
        except OperationCancelled:
            raise
        except Exception as error:
            # Optional local evidence I/O must not discard a measured result,
            # change identity acceptance, or reflect paths/raw text to agents.
            artifact_receipt = {"schema_version": POST_SNAPSHOT_ARTIFACT_SCHEMA,
                "status": "unavailable", "error_type": type(error).__name__}
    if operation_control is not None:
        operation_control.checkpoint()
    snapshot = parse_vivo_runtime_snapshot_output(stdout, returncode=returncode,
        expected_owner=owner, owner_source=owner_source, transport_error=transport_error)
    if artifact_receipt is not None:
        snapshot["post_snapshot_artifacts"] = artifact_receipt
    return snapshot


def parse_vivo_runtime_snapshot_output(
    stdout: str, *, returncode: int | None, expected_owner: dict | None,
    owner_source: str, transport_error: str | None = None,
) -> dict:
    """Pure parser shared by collection and offline post-snapshot replay."""
    owner = copy.deepcopy(expected_owner)
    snapshot = {
        "status": "unavailable", "cpus_allowed_list": "unknown",
        "affinity_observation_scope": "post_measure_owned_rpc_root_tasks_only",
        "measured_execution_affinity_proven": False,
        "execution_worker_affinity": "unknown",
        "main_thread_cpus_allowed_list": "unknown",
        "thread_cpus_allowed_list": {}, "rpc_task_count": "unknown",
        "process_identity": {
            "status": "unverified" if owner else "owned_receipt_unavailable",
            "expected_rpc_owner": owner, "root_generation_verified": False,
            "owner_source": owner_source,
            "measurement_execution_identity_proven": False,
        },
        "cpu_frequency_khz": {}, "thermal_readings_raw": {},
        "global_telemetry": {"scope": "post_measure_device_global_sysfs",
            "process_bound": False, "timed_phase_overlap": "unproven",
            "frequency_status": "unknown", "thermal_status": "unknown",
            "thermal_units": "raw_driver_units_not_assumed_celsius"},
        "unknown": ["affinity", "frequency", "thermal"],
    }
    if transport_error is not None:
        snapshot["error"] = f"runtime snapshot failed: {transport_error}"
        return snapshot
    if returncode != 0:
        snapshot["error"] = "runtime snapshot command failed"
        return snapshot
    if len(stdout.encode("utf-8")) > 256 * 1024 or len(stdout.splitlines()) > 300:
        snapshot["error"] = "runtime snapshot exceeded bounded report size"
        return snapshot

    proof = parse_runtime_snapshot_proof(stdout, owner)
    if proof is not None:
        snapshot["process_identity"].update(proof, status="verified")
        for field in ("main_thread_cpus_allowed_list", "thread_cpus_allowed_list", "rpc_task_count"):
            snapshot[field] = proof[field]
    elif owner is not None:
        snapshot["process_identity"]["status"] = "generation_or_task_proof_failed"
    frequencies: dict[str, int] = {}
    thermal: dict[str, dict[str, object]] = {}
    for raw_line in stdout.splitlines():
        key, separator, value = raw_line.strip().partition("=")
        if not separator:
            continue
        if re.fullmatch(r"cpu[0-7]_freq_khz", key):
            try:
                frequency = int(value)
                if frequency > 0:
                    frequencies[key.removesuffix("_freq_khz")] = frequency
            except ValueError:
                pass
        elif re.fullmatch(r"thermal(?:[0-9]|1[0-5])", key):
            kind, thermal_separator, reading = value.rpartition(":")
            if not thermal_separator:
                continue
            try:
                thermal[key] = {"type": kind or "unknown", "value": int(reading)}
            except ValueError:
                continue
    def compact(cpus: set[int]) -> str:
        if not cpus:
            return "unknown"
        values = sorted(cpus)
        ranges: list[str] = []
        start = previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
                continue
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = value
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        return ",".join(ranges)

    union: set[int] = set()
    for allowed in snapshot["thread_cpus_allowed_list"].values():
        union.update(parse_cpu_list(allowed) or set())
    effective_affinity = compact(union)
    unknown = []
    if effective_affinity == "unknown":
        unknown.append("affinity")
    if len(frequencies) != 8:
        unknown.append("frequency")
    if not thermal:
        unknown.append("thermal")
    snapshot.update(status=("available" if not unknown else
                    "partial" if proof or frequencies or thermal else "unavailable"),
                    cpus_allowed_list=effective_affinity, cpu_frequency_khz=frequencies,
                    thermal_readings_raw=thermal, unknown=unknown)
    snapshot["global_telemetry"].update(
        frequency_status="available" if len(frequencies) == 8 else "partial" if frequencies else "unknown",
        thermal_status="observed_subset" if thermal else "unknown")
    return snapshot


def _simpleperf_window(
    serial: str,
    duration_s: float,
    *,
    operation_control: OperationControl | None = None,
    stop_event: threading.Event | None = None,
    diagnostics: dict[str, object] | None = None,
    capture: dict | None = None,
    expected_owner: dict | None = None,
) -> dict:
    if diagnostics is not None:
        diagnostics.pop("process_identity", None)
    # A refresh-aware caller supplies a newly published owned generation.
    # The shell still verifies its boot/start ticks/command before and after
    # collection; publication alone is not a readiness or execution proof.
    owner = expected_owner if expected_owner is not None else read_rpc_owner(serial)
    if capture is not None:
        capture["expected_rpc_owner"] = owner
        capture["stage"] = "pid_probe" if owner is None else "simpleperf_stat"
        capture["output_available"] = False
    if owner is not None:
        command = scoped_stat_command(owner, duration_s=duration_s, events=VIVO_EVENTS)
        pid = [owner["pid"]]
    else:
        pid_probe = run_cancellable(
            ("adb", "-s", serial, "shell", "pidof tvm_rpc"),
            control=operation_control,
            stop_event=stop_event,
            capture_output=True,
            text=True,
            timeout=15,
        )
        pid = pid_probe.stdout.strip().split()
        if capture is not None:
            capture.update(stdout=pid_probe.stdout, stderr=pid_probe.stderr,
                           returncode=pid_probe.returncode, output_available=True,
                           output_stage="pid_probe")
        if pid_probe.returncode != 0 or len(pid) != 1 or not pid[0].isdigit():
            if diagnostics is not None:
                diagnostics["pid_probe_failures"] = int(
                    diagnostics.get("pid_probe_failures", 0)
                ) + 1
                diagnostics["last_pid_probe_returncode"] = pid_probe.returncode
                diagnostics["pid_selection"] = "missing_or_ambiguous"
            return {}
        command = f"simpleperf stat -p {pid[0]} --duration {duration_s} -e {VIVO_EVENTS} --csv"
    if capture is not None:
        capture["stage"] = "simpleperf_stat"
        # A later transport cancellation must not label the successful PID
        # probe's stdout as an empty/completed simpleperf report.
        capture.update(stdout="", stderr="", output_available=False)
        capture.pop("returncode", None)
        capture.pop("output_stage", None)
    proc = run_cancellable(
        ("adb", "-s", serial, "shell", command),
        control=operation_control,
        stop_event=stop_event,
        capture_output=True,
        text=True,
        timeout=duration_s + 30,
    )
    if capture is not None:
        capture.update(stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode,
                       output_available=True, output_stage="simpleperf_stat")
    if proc.returncode != 0:
        if diagnostics is not None:
            diagnostics["simpleperf_failures"] = int(
                diagnostics.get("simpleperf_failures", 0)
            ) + 1
            diagnostics["last_simpleperf_returncode"] = proc.returncode
        return {}
    # Android simpleperf writes statistics to stderr in some releases even
    # when --csv is selected; other builds use stdout.  Parse both bounded
    # streams, and locate the event column instead of assuming it is always
    # column 1 (some versions insert an empty/count-unit column).
    report = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    if len(report.encode("utf-8")) > 1024 * 1024:
        if diagnostics is not None:
            diagnostics["report_overflow"] = True
        return {}
    if owner is not None:
        proof = parse_generation_proof(report, owner)
        if proof is None:
            if diagnostics is not None:
                diagnostics["generation_proof_failed"] = True
            return {}
        if diagnostics is not None:
            diagnostics["process_identity"] = proof
    try:
        counters = _parse_simpleperf_report(report)
    except ValueError:
        if diagnostics is not None:
            diagnostics["ambiguous_event_rows"] = True
        return {}
    if diagnostics is not None:
        diagnostics["successful_windows"] = int(
            diagnostics.get("successful_windows", 0)
        ) + 1
        if not counters.get("cpu_cycles"):
            diagnostics["zero_cycle_windows"] = int(
                diagnostics.get("zero_cycle_windows", 0)
            ) + 1
        diagnostics["parsed_events"] = sorted(counters)
        diagnostics["pid_observed"] = pid[0]
        diagnostics["multiplex_coverage"] = "unknown"
    return counters


def _parse_simpleperf_report(report: str) -> dict:
    """Replay the event parser independently of transport and PID validation.

    Callers must also validate the saved return code, generation proof and
    truncation flags before treating replayed values as usable evidence.
    """
    if len(report.encode("utf-8")) > 1024 * 1024:
        raise ValueError("simpleperf report exceeds parsing limit")
    counters = {}
    admitted_events = set(VIVO_EVENTS.split(","))
    for parts in csv.reader(io.StringIO(report)):
        cells = [cell.strip() for cell in parts]
        event_index = next(
            (index for index, cell in enumerate(cells[:4]) if cell in admitted_events),
            None,
        )
        if event_index is None:
            continue
        count = None
        for cell in reversed(cells[:event_index]):
            try:
                count = float(cell.replace(",", ""))
            except ValueError:
                continue
            if math.isfinite(count) and count >= 0:
                break
            count = None
        if count is not None:
            event = cells[event_index].replace("-", "_")
            if event in counters and counters[event] != count:
                raise ValueError("ambiguous simpleperf event rows")
            counters[event] = count
    return counters


def collect_vivo_simpleperf_during(
    runner_job_fn,
    *args,
    serial: str,
    window_s: float = 0.2,
    operation_control: OperationControl | None = None,
    artifact_dir=None,
    require_fresh_rpc_owner: bool = False,
    fresh_owner_timeout_s: float = 180.0,
    **kwargs,
) -> dict:
    """Sample repeated simpleperf windows and retain the busiest one.

    RPC startup can delay execution, so a single fixed-delay sample may
    capture an idle server instead of the measurement phase. Callers that
    refresh RPC must request a fresh-owner gate, preventing sampling of a
    prior job's listener while the new worker compiles/refreshes. The gate
    only reads host receipts; it never starts another measurement.
    """
    windows: list[dict] = []
    window_diagnostics: dict[str, object] = {}
    stop = threading.Event()
    sampler_errors: list[OperationCancelled] = []
    sampler_failures: list[str] = []

    if not math.isfinite(window_s) or not 0.01 <= window_s <= 60:
        raise ValueError("simpleperf window must be finite and between 0.01 and 60 seconds")
    if type(require_fresh_rpc_owner) is not bool:
        raise ValueError("require_fresh_rpc_owner must be boolean")
    if (type(fresh_owner_timeout_s) not in (int, float)
            or not math.isfinite(fresh_owner_timeout_s)
            or not 0 < fresh_owner_timeout_s <= 180):
        raise ValueError("fresh owner timeout must be finite and in (0, 180] seconds")

    if operation_control is not None:
        operation_control.checkpoint()

    artifacts = (WindowArtifacts(artifact_dir, collector="vivo_simpleperf",
                 parser_schema="leka.simpleperf_csv.v1") if artifact_dir is not None else None)
    artifact_summary = None
    best_cycles = 0.0
    initial_owner = read_rpc_owner(serial) if require_fresh_rpc_owner else None
    gate_started = time.monotonic()
    gate = {
        "required": require_fresh_rpc_owner,
        "status": "waiting_for_new_owned_receipt" if require_fresh_rpc_owner else "not_requested",
        "initial_owner": initial_owner,
        "new_generations_observed": [],
        "timeout_s": fresh_owner_timeout_s,
        "publication_proves_readiness": False,
        "measurement_phase_overlap": "unproven",
    }
    normal_tail_cancellations = 0

    def current_fresh_owner():
        """Host-only wait; each window reacquires after any launch retry."""
        while not stop.is_set():
            if operation_control is not None:
                operation_control.checkpoint()
            owner = read_rpc_owner(serial)
            if owner is not None and owner != initial_owner:
                if owner not in gate["new_generations_observed"]:
                    if len(gate["new_generations_observed"]) >= 8:
                        gate["status"] = "generation_limit_exceeded"
                        return None
                    gate["new_generations_observed"].append(copy.deepcopy(owner))
                gate.setdefault("first_fresh_receipt_after_ms", (time.monotonic() - gate_started) * 1000)
                gate["status"] = "fresh_receipt_observed_not_readiness_proof"
                return owner
            if time.monotonic() - gate_started >= fresh_owner_timeout_s:
                gate["status"] = "fresh_owner_wait_timeout"
                return None
            stop.wait(0.05)
        if not gate["new_generations_observed"]:
            gate["status"] = "runner_finished_before_fresh_owner"
        return None

    def sampler():
        nonlocal best_cycles, normal_tail_cancellations
        try:
            while not stop.is_set():
                fresh_owner = current_fresh_owner() if require_fresh_rpc_owner else None
                if require_fresh_rpc_owner and fresh_owner is None:
                    break
                capture = {"started_unix_ns": time.time_ns(),
                           "started_monotonic_ns": time.monotonic_ns()}
                counters = {}
                window_index = None
                try:
                    counters = _simpleperf_window(
                        serial,
                        window_s,
                        operation_control=operation_control,
                        stop_event=stop,
                        diagnostics=window_diagnostics,
                        capture=capture,
                        **({"expected_owner": fresh_owner} if require_fresh_rpc_owner else {}),
                    )
                except BaseException as error:
                    capture["exception_type"] = type(error).__name__
                    raise
                finally:
                    capture.update(ended_unix_ns=time.time_ns(),
                                   ended_monotonic_ns=time.monotonic_ns())
                    if artifacts is not None:
                        stdout, stderr = capture.pop("stdout", ""), capture.pop("stderr", "")
                        window_index = artifacts.record(stdout, stderr,
                            {**capture, "duration_requested_s": window_s,
                             "accepted_event_window": bool(counters),
                             "parsed_counters": counters, "diagnostics": copy.deepcopy(window_diagnostics)},
                            selected=counters.get("cpu_cycles", 0.0) > best_cycles)
                    best_cycles = max(best_cycles, counters.get("cpu_cycles", 0.0))
                if counters:
                    windows.append({"counters": counters,
                                    "artifact_window_index": window_index,
                                    "diagnostics": copy.deepcopy(window_diagnostics)})
                else:
                    # A missing/restarting RPC server must not become an ADB
                    # subprocess storm while the compiler is still working.
                    stop.wait(0.05)
        except OperationCancelled as error:
            if not _sampler_cancelled_normally(stop, operation_control):
                sampler_errors.append(error)
            else:
                normal_tail_cancellations += 1
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            sampler_failures.append(type(error).__name__)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    try:
        result = runner_job_fn(*args, **kwargs)
    finally:
        try:
            _join_sampler(thread, stop, operation_control)
        finally:
            if not thread.is_alive() and gate["status"] == "waiting_for_new_owned_receipt":
                gate["status"] = "runner_finished_before_fresh_owner"
            if artifacts is not None and not thread.is_alive():
                artifact_summary = artifacts.finish(selection="maximum_cpu_cycles_biased_sample",
                    exact_bundle=False, sampling_window_s=window_s,
                    requested_events=VIVO_EVENTS.split(","),
                    sampler_failures=sampler_failures,
                    sampler_cancelled=bool(sampler_errors),
                    normal_tail_cancellations=normal_tail_cancellations,
                    fresh_owner_gate=gate)
    if sampler_errors:
        raise sampler_errors[0]
    if operation_control is not None:
        operation_control.checkpoint()
    runtime_context = collect_vivo_runtime_snapshot(
        serial, operation_control=operation_control,
        artifact_dir=artifact_dir,
        expected_owner=(gate["new_generations_observed"][-1]
                        if require_fresh_rpc_owner and gate["new_generations_observed"] else None),
        require_expected_owner=require_fresh_rpc_owner,
    )
    if sampler_failures:
        return {"error": "simpleperf sampler failed: " + ", ".join(sampler_failures),
                "provenance": {"exact_bundle": False, "artifacts": artifact_summary,
                               "fresh_owner_gate": gate, "normal_tail_cancellations": normal_tail_cancellations},
                "runtime_context": runtime_context, "job_result": result}
    selected = max(windows, key=lambda w: w["counters"].get("cpu_cycles", 0.0), default=None)
    busy = selected["counters"] if selected else None
    selected_diagnostics = selected["diagnostics"] if selected else {}
    if not busy or not busy.get("cpu_cycles"):
        return {"error": (
                    f"no busy simpleperf window captured ({len(windows)} windows); "
                    f"fresh_owner_gate={gate['status']}; "
                    f"normal_tail_cancellations={normal_tail_cancellations}; "
                    f"bounded_attempts={json.dumps(window_diagnostics, sort_keys=True)}"
                ),
                "provenance": {"exact_bundle": False, "artifacts": artifact_summary,
                               "fresh_owner_gate": gate, "normal_tail_cancellations": normal_tail_cancellations},
                "runtime_context": runtime_context,
                "job_result": result}
    if "instructions" in busy and busy.get("cpu_cycles"):
        busy["ipc"] = busy["instructions"] / busy["cpu_cycles"]
    runtime_context["counter_window_process_context"] = selected_diagnostics.get("process_identity") or {
        "process_coverage": "unknown", "execution_worker_affinity": "unknown"
    }
    return {"source": "vivo_simpleperf", "counters": busy, "rows": [],
            "provenance": {
                "scope": ("owned_rpc_process_tree_window" if selected_diagnostics.get("process_identity")
                          else "rpc_process_window_without_pid_start_proof"),
                "exact_bundle": False,
                "measurement_phase_overlap": "unproven",
                "selection": "maximum_cpu_cycles_biased_sample",
                "sampling_window_s": window_s, "captured_windows": len(windows),
                "multiplex_coverage": "unknown",
                "pid_start_identity": ("verified_owned_root_generation" if selected_diagnostics.get("process_identity")
                                       else "unproven"),
                "selected_window_process_identity": selected_diagnostics.get("process_identity"),
                "selected_window_index": selected.get("artifact_window_index"),
                "artifacts": artifact_summary,
                "fresh_owner_gate": gate,
                "normal_tail_cancellations": normal_tail_cancellations,
                "diagnostics": window_diagnostics,
            },
            "runtime_context": runtime_context,
            "notes": [f"busiest of {len(windows)} whole-process PMU windows over the "
                      "RPC server: never apportioned across PrimFuncs"],
            "job_result": result}
