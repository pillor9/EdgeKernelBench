"""Experimental finite-epoch simpleperf collector; not wired into default runs.

The parser uses an explicit CSV dialect tested with synthetic fixtures.
Device-format validation is required before enabling live interval capture.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import threading
import time

from ...backends import BACKENDS
from ...evaluation.device_lease import device_lease
from ...operation_control import OperationCancelled, OperationControl, run_cancellable
from .profile_artifacts import WindowArtifacts
from .android_process_identity import read_rpc_owner, scoped_stat_command, parse_generation_proof

EVENTS = ("cpu-cycles", "instructions", "cache-references", "cache-misses", "branch-misses")
PARSER_SCHEMA = "leka.simpleperf_interval_repeated_sections.experimental_v0"
MAX_RAW_BYTES = 1_048_576


def _bounds(duration_s, interval_ms):
    if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s) or not .2 <= duration_s <= 10
            or type(interval_ms) is not int or not 50 <= interval_ms <= 2000
            or duration_s * 1000 < interval_ms):
        raise ValueError("interval epoch requires .2..10 seconds and 50..2000 ms intervals")


def interval_command(owner, *, duration_s=5.0, interval_ms=200):
    """Reuse the exact owned-root/descendant proof; change only stat flags."""
    _bounds(duration_s, interval_ms)
    command = scoped_stat_command(owner, duration_s=duration_s, events=",".join(EVENTS))
    original = 'simpleperf stat -p "$targets" --duration "$duration" -e "$events" --csv'
    if command.count(original) != 1:
        raise RuntimeError("owned stat template changed; interval command needs review")
    return command.replace(original, original + f" --interval {interval_ms} --interval-only-values")


def _number(value, *, positive=False):
    try:
        result = float(value.replace(",", ""))
    except (ValueError, AttributeError) as error:
        raise ValueError("unsupported interval numeric field") from error
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError("invalid interval numeric field")
    return result


def parse_epoch(stdout, stderr, *, owner, returncode, duration_s, interval_ms, elapsed_mode):
    """Parse a candidate dialect only after whole-epoch generation validation.

    elapsed_mode is explicitly cumulative_seconds or interval_seconds; it is
    never inferred from approximate values. Counts are already interval deltas
    because the command requested --interval-only-values. The caller must first
    validate this dialect against an actual fixture from the installed tool.
    """
    _bounds(duration_s, interval_ms)
    if elapsed_mode not in {"cumulative_seconds", "interval_seconds"}:
        raise ValueError("explicit elapsed_mode required; device dialect is unverified")
    if type(returncode) is not int or returncode != 0:
        raise ValueError("nonzero epoch exit invalidates every interval")
    if sum(len(s.encode("utf-8")) for s in (stdout, stderr)) > MAX_RAW_BYTES:
        raise ValueError("interval epoch raw exceeded parsing bound")
    proof = parse_generation_proof(stdout + "\n" + stderr, owner)
    if proof is None:
        raise ValueError("epoch generation proof missing or invalid")
    streams = [(name, text) for name, text in (("stdout", stdout), ("stderr", stderr))
               if "Performance counter statistics," in text]
    if len(streams) != 1:
        raise ValueError("no unique supported interval CSV stream")
    stream_name, text = streams[0]
    frames = []
    pending = None
    offset = 0
    start = None
    elapsed = 0.0
    frame_limit = min(201, math.ceil(duration_s * 1000 / interval_ms) + 1)
    for line in text.splitlines(keepends=True):
        begin = offset
        offset += len(line.encode("utf-8"))
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("LEKA_PMU_"):
            continue
        cells = [item.strip() for item in next(csv.reader([line]))]
        if cells and cells[0] == "Performance counter statistics":
            if any(cells[1:]):
                raise ValueError("unsupported interval section header")
            if pending is not None:
                raise ValueError("new interval header before prior section terminator")
            pending, start = {}, begin
            continue
        if pending is None:
            raise ValueError("unexpected data outside interval section")
        if len(cells) >= 3 and cells[0] == "Total test time":
            if cells[2] != "seconds" or set(pending) != {e.replace("-", "_") for e in EVENTS}:
                raise ValueError("incomplete event set or unsupported interval time unit")
            stamp = _number(cells[1], positive=True)
            span = stamp - elapsed if elapsed_mode == "cumulative_seconds" else stamp
            if span <= 0 or span > interval_ms / 1000 * 1.5 + .02:
                raise ValueError("interval time reversed, missing, or outside expected cadence")
            elapsed = stamp if elapsed_mode == "cumulative_seconds" else elapsed + stamp
            if elapsed > duration_s + interval_ms / 1000 * 1.5 + .02:
                raise ValueError("interval report extends beyond bounded epoch")
            frames.append({"interval_index": len(frames), "counters": pending,
                "actual_duration_s": span, "reported_time_s": stamp,
                "elapsed_mode": elapsed_mode, "partial_interval": span < interval_ms / 1000 * .9,
                "raw_stream": stream_name, "raw_begin_byte": start, "raw_end_byte": offset})
            if len(frames) > frame_limit:
                raise ValueError("too many intervals for bounded epoch")
            pending, start = None, None
            continue
        if len(cells) < 2 or cells[1] not in EVENTS:
            raise ValueError("unsupported interval event row")
        key = cells[1].replace("-", "_")
        if key in pending:
            raise ValueError("duplicate interval event")
        pending[key] = _number(cells[0])
    if not frames:
        raise ValueError("no complete interval sections")
    return {"parser_schema": PARSER_SCHEMA, "intervals": frames,
            "incomplete_terminal_section": pending is not None, "process_identity": proof,
            "counts_are_interval_deltas": True, "exact_bundle": False,
            "measurement_phase_overlap": "unproven", "multiplex_coverage": "unknown",
            "requested_duration_s": duration_s, "reported_elapsed_s": elapsed,
            "unreported_tail_s": max(0.0, duration_s - elapsed),
            "coverage_scope": "reported_complete_sections_not_full_requested_epoch"}


def capture_owned_epoch(serial, *, duration_s=5.0, interval_ms=200, operation_control=None, capture=None):
    """One finite read-only stat epoch; caller owns the physical-device lease.

    It never restarts the RPC server and never falls back to process-name
    attachment. No per-job stop_event is passed into the subprocess: an already
    finite epoch may finish and obtain its proof after the runner ends. Global
    cancellation/deadline still aborts the command and invalidates its evidence.
    """
    _bounds(duration_s, interval_ms)
    owner = read_rpc_owner(serial)
    captured = capture if capture is not None else {}
    captured.update({"expected_rpc_owner": owner, "started_unix_ns": time.time_ns(),
                "started_monotonic_ns": time.monotonic_ns(), "stdout": "", "stderr": "",
                "output_available": False, "interval_ms": interval_ms, "duration_s": duration_s,
                "requested_events": list(EVENTS), "requested_count_mode": "interval_only_values"})
    try:
        if owner is None:
            captured["error"] = "owned RPC receipt missing; no attachment attempted"
            return captured
        proc = run_cancellable(("adb", "-s", serial, "shell", interval_command(
            owner, duration_s=duration_s, interval_ms=interval_ms)), control=operation_control,
            capture_output=True, text=True, timeout=duration_s + 15)
        captured.update(stdout=proc.stdout or "", stderr=proc.stderr or "",
                        returncode=proc.returncode, output_available=True)
        captured["process_identity"] = parse_generation_proof(captured["stdout"] + "\n" + captured["stderr"], owner)
        captured["generation_verified"] = proc.returncode == 0 and captured["process_identity"] is not None
        return captured
    finally:
        captured.update(ended_unix_ns=time.time_ns(), ended_monotonic_ns=time.monotonic_ns())


def collect_vivo_interval_during(runner_job_fn, *args, serial, artifact_dir,
        duration_s=5.0, interval_ms=200, max_epochs=4, max_reattachments=2,
        elapsed_mode=None, dialect_fixture_sha256=None, operation_control=None, **kwargs):
    """Experimental wrapper, deliberately absent from normal mode selection.

    elapsed_mode=None captures fixtures without emitting counters. To exercise
    the parser, a caller must explicitly supply the reviewed fixture hash and
    its elapsed-time dialect. That hash identifies the configured fixture; it
    does not prove device compatibility by itself. All epochs remain nonexact.
    """
    import re
    _bounds(duration_s, interval_ms)
    if type(max_epochs) is not int or not 1 <= max_epochs <= 16 or type(max_reattachments) is not int or not 0 <= max_reattachments <= 2:
        raise ValueError("experimental epoch/reattachment limits invalid")
    if elapsed_mode is not None and (elapsed_mode not in {"cumulative_seconds", "interval_seconds"}
            or not isinstance(dialect_fixture_sha256, str) or re.fullmatch("[a-f0-9]{64}", dialect_fixture_sha256) is None):
        raise ValueError("explicit reviewed dialect and fixture SHA-256 required")
    artifacts = WindowArtifacts(artifact_dir, collector="vivo_simpleperf_interval_experimental",
        parser_schema=PARSER_SCHEMA, max_windows=max(2, max_epochs + 1), max_stream_bytes=MAX_RAW_BYTES)
    stop = threading.Event()
    failures = []
    selected = None
    busy = 0.0
    global_cancellations = []
    def sample_epochs():
        nonlocal selected, busy
        reattachments = 0
        for epoch_index in range(max_epochs):
            if stop.is_set():
                break
            capture = {"epoch_index": epoch_index, "output_available": False}
            candidate = None
            try:
                capture_owned_epoch(serial, duration_s=duration_s, interval_ms=interval_ms,
                    operation_control=operation_control, capture=capture)
                if not capture.get("generation_verified"):
                    capture["parse_status"] = "unusable_epoch_identity_or_transport"
                    reattachments += 1
                elif elapsed_mode is None:
                    capture["parse_status"] = "fixture_only_dialect_unverified"
                else:
                    parsed = parse_epoch(capture["stdout"], capture["stderr"], owner=capture["expected_rpc_owner"],
                        returncode=capture["returncode"], duration_s=duration_s, interval_ms=interval_ms,
                        elapsed_mode=elapsed_mode)
                    capture.update(parse_status="parsed_experimental_dialect", parsed=parsed,
                                   dialect_fixture_sha256=dialect_fixture_sha256)
                    candidate = max(parsed["intervals"], key=lambda x:x["counters"].get("cpu_cycles", 0))
            except OperationCancelled as error:
                capture["exception_type"] = type(error).__name__
                global_cancellations.append(error)
            except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
                capture["exception_type"] = type(error).__name__
                failures.append(type(error).__name__)
            finally:
                stdout, stderr = capture.pop("stdout", ""), capture.pop("stderr", "")
                stronger = candidate is not None and candidate["counters"].get("cpu_cycles", 0) > busy
                artifacts.record(stdout, stderr, capture, selected=stronger)
                if stronger:
                    busy = candidate["counters"]["cpu_cycles"]
                    selected = {"epoch_index": epoch_index, **candidate,
                                "process_identity": capture["process_identity"]}
            if global_cancellations or failures or reattachments > max_reattachments:
                break
            if not capture.get("generation_verified"):
                stop.wait(.25)
    def sampler():
        try:
            sample_epochs()
        except Exception as error:
            # Includes archive write failures and unexpected parser/transport
            # exceptions; never silently lose a failed background sampler.
            failures.append(type(error).__name__)
    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    try:
        result = runner_job_fn(*args, **kwargs)
    finally:
        stop.set()
        thread.join(duration_s + 17)
        if thread.is_alive():
            if operation_control is not None:
                operation_control.cancel("interval sampler failed to finish finite epoch")
            thread.join(3)
        if thread.is_alive():
            raise RuntimeError("interval sampler did not stop; no trusted result released")
        receipt = artifacts.finish(exact_bundle=False, requested_events=list(EVENTS),
            configured_elapsed_mode=elapsed_mode, dialect_fixture_sha256=dialect_fixture_sha256,
            default_collector_unchanged=True, scope="owned_rpc_process_tree_interval_epoch")
    if global_cancellations:
        raise global_cancellations[0]
    provenance = {"exact_bundle": False, "scope": "owned_rpc_process_tree_interval_epoch",
        "measurement_phase_overlap": "unproven", "multiplex_coverage": "unknown",
        "parser_schema": PARSER_SCHEMA, "artifacts": receipt, "failures": failures,
        "dialect_fixture_sha256": dialect_fixture_sha256, "selected_interval": selected}
    if selected is None or failures:
        return {"error": "no validated positive interval; fixture-only or collection unavailable",
                "provenance": provenance, "job_result": result}
    counters = dict(selected["counters"])
    counters["ipc"] = counters["instructions"] / counters["cpu_cycles"]
    return {"source": "vivo_simpleperf_interval_experimental", "counters": counters,
            "rows": [], "provenance": provenance, "job_result": result}


def main():
    parser = argparse.ArgumentParser(description="Capture one finite owned-RPC interval fixture; no workload/model request")
    parser.add_argument("--serial", default=BACKENDS["vivo_cpu"]["vivo_serial"])
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--interval-ms", type=int, default=200)
    args = parser.parse_args()
    if not args.serial:
        parser.error("explicit configured Vivo serial required")
    try:
        _bounds(args.duration, args.interval_ms)
    except ValueError as error:
        parser.error(str(error))
    control = OperationControl()
    control.reset(deadline_monotonic=time.monotonic() + args.duration + 25)
    artifacts = WindowArtifacts(args.artifact_dir.absolute(), collector="vivo_simpleperf_interval_experimental",
        parser_schema=PARSER_SCHEMA, max_windows=2, max_stream_bytes=MAX_RAW_BYTES)
    capture = {"stdout": "", "stderr": "", "output_available": False}
    # Fail quickly if another LEKA workload owns the device. This command does
    # not refresh the RPC server; a missing owner receipt means no attachment.
    try:
        with device_lease("adb:" + args.serial).hold(control, timeout_s=1):
            capture_owned_epoch(args.serial, duration_s=args.duration,
                interval_ms=args.interval_ms, operation_control=control, capture=capture)
    except (OperationCancelled, OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
        capture["exception_type"] = type(error).__name__
        capture["error"] = "fixture transport or ownership unavailable; see private artifact metadata"
    stdout, stderr = capture.pop("stdout"), capture.pop("stderr")
    artifacts.record(stdout, stderr, {**capture, "parse_status": "fixture_only_dialect_unverified"})
    receipt = artifacts.finish(exact_bundle=False, fixture_only=True, default_collector_unchanged=True)
    print(json.dumps({"artifacts": receipt, "generation_verified": capture.get("generation_verified", False),
                      "returncode": capture.get("returncode"), "error": capture.get("error"),
                      "parser_not_enabled": True}, indent=2))
    return 0 if capture.get("generation_verified") else 2


if __name__ == "__main__":
    raise SystemExit(main())
