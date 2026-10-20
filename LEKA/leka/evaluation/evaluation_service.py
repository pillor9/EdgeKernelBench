"""Check candidate structure on the host and correctness and timing on devices.

CUDA and OpenCL jobs run through SSH workers; Android CPU jobs use a host
worker and TVM RPC. Agents receive validation results without hidden inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..backends import BACKENDS
from ..candidate_source import CandidateSourceError, validate_candidate_source
from ..lowering import abi_contract as abi_freeze
from ..lowering.initial_schedule import run_l1_seed
from ..lowering.relax_to_tir import run_t01
from ..operation_control import (
    OperationCancelled,
    OperationControl,
    run_cancellable,
)
from ..sandboxing import candidate_environment, sandboxed_candidate_argv
from ..ssh_transport import SSH_DEVICE_UNAVAILABLE_MARKERS, network_unavailable_marker
from . import diagnostics
from .transport_evidence import boot_id_from_text, collect_transport_evidence
from .device_lease import device_lease
from .timing_quality import assess_timing

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE_WORKER = Path(__file__).resolve().parent / "device_worker.py"
VIVO_WORKER = Path(__file__).resolve().parent / "android_worker.py"
CANDIDATE_SOURCE_GUARD = Path(__file__).resolve().parents[1] / "candidate_source.py"
TIMING_QUALITY_HELPER = Path(__file__).resolve().parent / "timing_quality.py"
PARAMETER_PLAN_HELPER = Path(__file__).resolve().parents[1] / "parameter_plan.py"
MAX_TRUSTED_IR_JSON_BYTES = 64 * 1024 * 1024
# Both profiles require three timing-free promotion checks. Ordinary
# mode recreates seeds 0/1/2, including T1's seed 0; generic hidden mode instead
# draws three fresh populations distinct from its timed T1 input.
PROMOTION_AUDIT_SEED_COUNT = 3
ORDINARY_CORRECTNESS_PROFILE = "ordinary_0_1_2_v1"
FRESH_HIDDEN_CORRECTNESS_PROFILE = "fresh_hidden_v1"
CORRECTNESS_PROFILES = (ORDINARY_CORRECTNESS_PROFILE, FRESH_HIDDEN_CORRECTNESS_PROFILE)


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class MeasureResult:
    ok: bool
    tier: str
    layer: str
    public: dict = field(default_factory=dict)   # what agents may see
    trusted: dict = field(default_factory=dict)  # driver-only detail


@dataclass(frozen=True)
class TrustedIRModule:
    """Lossless driver-owned IR transport that never executes Python source.

    Agent requests can only supply strings, which always pass through the
    candidate AST guard. Trusted callers may instead hold an already
    verified ``tvm.IRModule``.  This explicit capability sends its lossless
    JSON to workers for direct ``tvm.ir.load_json`` decoding, without a
    Python/base64 loader that would weaken the agent-source boundary.
    """

    json_text: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.json_text, str):
            raise TypeError("trusted IR JSON must be text")
        encoded = self.json_text.encode("utf-8")
        if not encoded or len(encoded) > MAX_TRUSTED_IR_JSON_BYTES:
            raise ValueError("trusted IR JSON is empty or exceeds its size limit")
        try:
            value = json.loads(self.json_text)
        except json.JSONDecodeError as error:
            raise ValueError("trusted IR serialization is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("trusted IR serialization must be a JSON object")
        object.__setattr__(self, "sha256", hashlib.sha256(encoded).hexdigest())

    @classmethod
    def from_module(cls, module) -> "TrustedIRModule":
        import tvm

        if not isinstance(module, tvm.IRModule):
            raise TypeError("trusted module capability requires tvm.IRModule")
        return cls(tvm.ir.save_json(module))


# Connection multiplexing: one master connection per device, reused by every
# measure/collector/scp call. This reduces connection latency and avoids
# exceeding sshd MaxStartups when measurement and profiling overlap.
SSH_MUX_OPTS = ("-o", "BatchMode=yes",
                "-o", "ControlMaster=auto",
                "-o", "ControlPath=~/.ssh/leka-mux-%r@%h:%p",
                "-o", "ControlPersist=600",
                "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=3",
                "-o", "ConnectTimeout=15")


class SSHRunner:
    """ssh/rsync transport; ships device_worker.py with each no-clobber job dir.
    Transport failures are RESULTS (fault_stage=transport), never exceptions."""

    def __init__(self, alias: str, *, path_prefix: str = "", remote_root: str = "/tmp/leka_jobs"):
        self.alias = alias
        self.path_prefix = path_prefix
        self.remote_root = remote_root
        self.operation_control: OperationControl | None = None

        # A confirmed network-level device loss is terminal for this runner.
        # Keeping the breaker open prevents every layer and collector from
        # creating a new ssh/rsync storm while the board is hung or powered off.
        self._device_unavailable = threading.Event()
        self._device_unavailable_lock = threading.Lock()
        self._device_unavailable_reason = ""

    def set_operation_control(self, control: OperationControl) -> None:
        self.operation_control = control

    @property
    def device_unavailable(self) -> bool:
        return self._device_unavailable.is_set()

    @property
    def device_unavailable_reason(self) -> str:
        return self._device_unavailable_reason

    @staticmethod
    def _unavailable_marker(proc) -> str | None:
        return network_unavailable_marker(proc)

    def _open_device_breaker(self, marker: str) -> None:
        with self._device_unavailable_lock:
            if self._device_unavailable.is_set():
                return
            self._device_unavailable_reason = f"ssh reported: {marker}"
            self._device_unavailable.set()

    def _breaker_result(self, argv: tuple) -> subprocess.CompletedProcess:
        reason = self._device_unavailable_reason or "device unavailable"
        return subprocess.CompletedProcess(
            argv,
            255,
            "",
            f"{self.alias} transport circuit breaker is open ({reason})",
        )

    def _run(self, argv: tuple, timeout_s: int, retries: int = 2):
        if self.device_unavailable:
            return self._breaker_result(argv)
        last = None
        for attempt in range(retries + 1):
            try:
                proc = run_cancellable(
                    argv,
                    control=self.operation_control,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                if proc.returncode == 0:
                    return proc
                last = proc
                marker = self._unavailable_marker(proc)
                if marker is not None:
                    self._open_device_breaker(marker)
                    break
            except subprocess.TimeoutExpired as err:
                last = err
            if attempt < retries:
                if self.operation_control is None:
                    time.sleep(2 * (attempt + 1))
                else:
                    self.operation_control.wait(2 * (attempt + 1))
        return last

    def run_job(self, job_dir: Path, *, timeout_s: int = 300) -> dict:
        if self.device_unavailable:
            return {
                "ok": False,
                "fault_stage": "transport",
                "error": (
                    f"{self.alias} unavailable; transport circuit breaker is open "
                    f"({self.device_unavailable_reason})"
                ),
            }
        (job_dir / "device_worker.py").write_text(DEVICE_WORKER.read_text())
        (job_dir / "candidate_source.py").write_text(CANDIDATE_SOURCE_GUARD.read_text())
        (job_dir / "parameter_plan.py").write_text(PARAMETER_PLAN_HELPER.read_text())
        (job_dir / "timing_quality.py").write_text(TIMING_QUALITY_HELPER.read_text())
        remote = f"{self.remote_root}/{job_dir.name}"

        before_boot_id = None

        def record_failure(result):
            evidence = collect_transport_evidence(
                self.alias, before_boot_id=before_boot_id,
                control=self.operation_control,
            )
            result["transport_evidence"] = evidence
            # This is trusted host evidence, separate from candidate result.json.
            try:
                (job_dir / "transport_failure.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
                )
            except OSError:
                result["transport_evidence_persisted"] = False
            return result

        def fail(step, proc):
            detail = ""
            if isinstance(proc, subprocess.TimeoutExpired):
                detail = f"{step} timed out"
            elif proc is not None:
                detail = (proc.stderr or proc.stdout or "")[-2000:]
            return record_failure({"ok": False, "fault_stage": "transport",
                    "error": f"{step} failed: {detail}"})

        setup = self._run(("ssh", *SSH_MUX_OPTS, self.alias,
                           f"mkdir -p {shlex.quote(self.remote_root)} && "
                           f"test ! -e {shlex.quote(remote)} && "
                           "cat /proc/sys/kernel/random/boot_id"), 60)
        if not (hasattr(setup, "returncode") and setup.returncode == 0):
            return fail("job setup", setup)
        before_boot_id = boot_id_from_text(setup.stdout)
        (job_dir / "transport_start.json").write_text(json.dumps({
            "device_boot_id": before_boot_id,
            "scope": "before_upload_and_worker_execution",
        }, indent=2) + "\n")
        push = self._run(("rsync", "-a", "-e", "ssh " + " ".join(SSH_MUX_OPTS),
                          str(job_dir) + "/", f"{self.alias}:{remote}/"), 300)
        if not (hasattr(push, "returncode") and push.returncode == 0):
            return fail("job upload", push)
        proc = self._run(("ssh", *SSH_MUX_OPTS, self.alias,
                          f"cd {shlex.quote(remote)} && {self.path_prefix}"
                          f"python3 device_worker.py ."), timeout_s, retries=0)
        if self.device_unavailable:
            return fail("job execution", proc)
        pull = self._run(("scp", *SSH_MUX_OPTS, "-q",
                          f"{self.alias}:{remote}/result.json",
                          str(job_dir / "result.json")), 60)
        if not (hasattr(pull, "returncode") and pull.returncode == 0):
            worker_out = ""
            if hasattr(proc, "stderr"):
                worker_out = (proc.stderr or proc.stdout or "")[-2000:]
            worker_rc = getattr(proc, "returncode", None)
            pull_rc = getattr(pull, "returncode", None)
            pull_error = str(getattr(pull, "stderr", "") or "")[-2000:]
            return record_failure({"ok": False, "fault_stage": "transport",
                    "worker_returncode": worker_rc,
                    "worker_timed_out": isinstance(proc, subprocess.TimeoutExpired),
                    "result_pull_returncode": pull_rc,
                    "error": (f"result pull failed; worker returncode: {worker_rc}; "
                              f"pull returncode: {pull_rc}; pull error: {pull_error}; "
                              f"worker output: {worker_out}")})
        result = json.loads((job_dir / "result.json").read_text())
        if result.get("ok"):  # keep failed job dirs on the device for debugging
            self._run(("ssh", *SSH_MUX_OPTS, self.alias,
                       f"rm -rf {shlex.quote(remote)}"), 30, retries=0)
        return result


class VivoHostRunner:
    """Host-side worker subprocess (cross compile + RPC); isolation seam for
    agent-authored module code, same as the T01 worker."""

    def __init__(self, tvm_python: str):
        self.tvm_python = tvm_python
        self.operation_control: OperationControl | None = None

    def set_operation_control(self, control: OperationControl) -> None:
        self.operation_control = control

    def run_job(self, job_dir: Path, *, timeout_s: int = 600) -> dict:
        try:
            return self._run_job(job_dir, timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            return {"ok": False, "fault_stage": "transport",
                    "error": f"vivo worker timed out after {timeout_s}s"}

    def _run_job(self, job_dir: Path, *, timeout_s: int) -> dict:
        proc = run_cancellable(
            (self.tvm_python, str(VIVO_WORKER), str(job_dir)),
            control=self.operation_control,
            capture_output=True, text=True, timeout=timeout_s, cwd=str(REPO_ROOT),
            env=candidate_environment(
                pythonpath=str(REPO_ROOT),
                # The caller's PATH must contain the matching LLVM 18 llc.
                path=os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                home=str(Path.home()),
            ))
        result_path = job_dir / "result.json"
        if not result_path.exists():
            return {"ok": False, "fault_stage": "vivo_worker",
                    "error": (proc.stderr or proc.stdout)[-4000:]}
        return json.loads(result_path.read_text())


# Compatibility alias for callers of the original helper.
class JetsonRunner(SSHRunner):
    def __init__(self, remote_root: str = "/tmp/leka_jobs", ssh_alias: str = "jetson"):
        super().__init__(ssh_alias, path_prefix="PATH=/usr/local/cuda/bin:$PATH ",
                         remote_root=remote_root)


class MeasureService:
    def __init__(self, *, task, backend: str = "jetson_gpu", target: str | None = None,
                 runner=None, tvm_python: str = sys.executable, warmup: int = 5,
                 repeat: int = 50, jobs_root: Path | None = None,
                 correctness_profile: str = FRESH_HIDDEN_CORRECTNESS_PROFILE):
        if correctness_profile not in CORRECTNESS_PROFILES:
            raise ValueError("unsupported correctness profile")
        self.correctness_profile = correctness_profile
        self.task = task
        self._sealed_tolerances = (
            self._read_task_tolerances()
            if correctness_profile == ORDINARY_CORRECTNESS_PROFILE else None
        )
        self.backend = backend
        self.cfg = BACKENDS[backend]
        self.target = target or self.cfg["target"]
        self.tvm_python = tvm_python
        if runner is not None:
            self.runner = runner
        elif self.cfg["runner"] == "ssh":
            self.runner = SSHRunner(self.cfg["ssh_alias"],
                                    path_prefix=self.cfg.get("worker_path_prefix", ""))
        else:
            self.runner = VivoHostRunner(tvm_python)
        self.warmup = warmup
        self.repeat = repeat
        self.jobs_root = jobs_root or Path(tempfile.gettempdir()) / "leka_jobs_host"
        # Ordinary mode must not draw even a discarded random seed.  Generic
        # hidden-input callers retain their previous explicitly distinct policy.
        self._seed = 0 if correctness_profile == ORDINARY_CORRECTNESS_PROFILE else self._random_seed()
        self.operation_control: OperationControl | None = None

    def device_lease(self, *, timeout_s=600):
        """Reserve the full measurement/profiling window on this backend."""
        cfg = self.cfg
        identity = (
            "adb:" + str(cfg.get("vivo_serial", ""))
            if cfg["runner"] == "vivo_rpc"
            else "ssh:" + str(cfg.get("ssh_alias", self.backend))
        )
        return device_lease(identity).hold(self.operation_control, timeout_s=timeout_s)

    def _run_device_job(self, job_dir: Path, *, timeout_s: int = 300) -> dict:
        with self.device_lease():
            return self.runner.run_job(job_dir, timeout_s=timeout_s)

    @staticmethod
    def _random_seed() -> int:
        value = secrets.randbits(31)
        while value == 900:
            value = secrets.randbits(31)
        return value

    def _read_task_tolerances(self) -> tuple[float, float]:
        facts = self.task.public_facts()
        values = (facts["atol"], facts["rtol"])
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("task correctness tolerances must be finite nonnegative numbers")
        return tuple(float(value) for value in values)

    def correctness_policy(self) -> dict:
        """Driver-owned safe policy metadata; contains no input values."""
        tolerances = self._read_task_tolerances()
        if self._sealed_tolerances is not None and tolerances != self._sealed_tolerances:
            raise ValueError("ordinary task correctness tolerances changed after service creation")
        ordinary = self.correctness_profile == ORDINARY_CORRECTNESS_PROFILE
        return {
            "schema_version": "leka.correctness_policy.v1",
            "profile_id": self.correctness_profile,
            "input_distribution": "task_original_unscaled",
            "input_scaling_applied": False,
            "tolerance_authority": "task_public_facts",
            "atol": tolerances[0], "rtol": tolerances[1],
            "t1_input_seeds": [0] if ordinary else None,
            "t1_input_count": 1,
            "audit_input_seeds": [0, 1, 2] if ordinary else None,
            "audit_input_count": PROMOTION_AUDIT_SEED_COUNT,
            "audit_reuses_t1_seed": ordinary,
            "audit_recreates_inputs": True,
            "paired_arms_share_input_artifact": True,
            "t1_single_seed_is_not_formal_audit": True,
        }

    def set_operation_control(self, control: OperationControl) -> None:
        """Bind every trusted blocking seam to the driver's phase control."""

        self.operation_control = control
        setter = getattr(self.runner, "set_operation_control", None)
        if callable(setter):
            setter(control)

    def _checkpoint(self) -> None:
        if self.operation_control is not None:
            self.operation_control.checkpoint()

    # -- T0: host structural screens ----------------------------------------
    def t0(self, layer: str, artifact_text: str, canonical_text: str | None = None,
           t12_text: str | None = None) -> MeasureResult:
        self._checkpoint()
        if layer == "L0":
            r = run_t01(
                artifact_text,
                self.target,
                expected_l0_source=self.task.naive_relax_source(),
                python=self.tvm_python,
                operation_control=self.operation_control,
            )
            if not r.ok:
                diag = diagnostics.sanitize(
                    r.error, r.fault_stage, lowest_edited="L0"
                )
                return MeasureResult(
                    ok=False,
                    tier="t0",
                    layer=layer,
                    public={"pass": False, "diagnostic": diag},
                    trusted={
                        "canonical": r.canonical,
                        "scheduled_default": "",
                        "raw_error": r.error,
                        "fault_stage": r.fault_stage,
                    },
                )

            # T01 ends at canonical TIR.  Build the executable fallback in an
            # explicitly separate L1 stage so it cannot own Relax fusion or
            # constant-folding decisions.  The usual L1 structural contract
            # then proves that only PrimFunc implementations changed.
            seed = run_l1_seed(
                r.canonical,
                self.target,
                python=self.tvm_python,
                operation_control=self.operation_control,
            )
            if not seed.ok:
                diag = diagnostics.sanitize(
                    seed.error, seed.fault_stage, lowest_edited="L1"
                )
                return MeasureResult(
                    ok=False,
                    tier="t0",
                    layer=layer,
                    public={"pass": False, "diagnostic": diag},
                    trusted={
                        "canonical": r.canonical,
                        "scheduled_default": "",
                        "raw_error": seed.error,
                        "fault_stage": seed.fault_stage,
                    },
                )
            seed_contract = self._parse_check(seed.source, r.canonical)
            if not seed_contract.get("ok"):
                fault_stage = "l1_seed_contract"
                error = str(seed_contract.get("error", "invalid L1 seed"))
                diag = diagnostics.sanitize(
                    error, fault_stage, lowest_edited="L1"
                )
                return MeasureResult(
                    ok=False,
                    tier="t0",
                    layer=layer,
                    public={"pass": False, "diagnostic": diag},
                    trusted={
                        "canonical": r.canonical,
                        "scheduled_default": "",
                        "raw_error": error,
                        "fault_stage": fault_stage,
                    },
                )
            return MeasureResult(
                ok=True,
                tier="t0",
                layer=layer,
                public={"pass": True, "diagnostic": None},
                trusted={
                    "canonical": r.canonical,
                    # Compatibility name used throughout the driver.  This is
                    # an L1 bootstrap/fallback, never the canonical T01 output.
                    "scheduled_default": seed.source,
                    "l1_seed_scheduler": seed.scheduler,
                    "raw_error": "",
                    "fault_stage": "",
                },
            )
        if layer == "L1":
            # Agent-authored L1 may jointly contract a small declared local
            # region and schedule its TIR.  Bootstrap seeds above remain strict.
            check = self._parse_check(artifact_text, canonical_text,
                                      allow_local_fusion=True)
            diag = None if check["ok"] else diagnostics.sanitize(
                check.get("error"), check.get("fault_stage"), lowest_edited="L1")
            return MeasureResult(ok=check["ok"], tier="t0", layer=layer,
                                 public={"pass": check["ok"], "diagnostic": diag,
                                         "contract": "l1_local_joint_v1",
                                         "local_rewrite": check.get("local_rewrite")},
                                 trusted=check)
        if layer == "L2":
            if not t12_text:
                diagnostic = diagnostics.sanitize(
                    "T12 slot empty: driver has not extracted the generated source yet",
                    "device_setup",
                    lowest_edited="L2",
                )
                return MeasureResult(ok=False, tier="t0", layer=layer,
                                     public={"pass": False,
                                             "diagnostic": diagnostic})
            check = abi_freeze.check(self.cfg["l2_lang"], t12_text, artifact_text)
            diag = None if check["ok"] else diagnostics.sanitize(
                check.get("error"), "load_patch", lowest_edited="L2")
            return MeasureResult(ok=check["ok"], tier="t0", layer=layer,
                                 public={"pass": check["ok"], "diagnostic": diag,
                                         "kernels": check.get("kernels")}, trusted=check)
        raise ValueError(f"unknown layer: {layer}")

    def _parse_check(self, artifact_text: str, canonical_text: str | None, *,
                     allow_local_fusion: bool = False) -> dict:
        """Check L1 in the isolated TVM process; seed generation stays strict."""
        try:
            validate_candidate_source(artifact_text)
            if canonical_text:
                validate_candidate_source(canonical_text)
        except CandidateSourceError as error:
            return {"ok": False, "fault_stage": "parse", "error": str(error)}
        script = (
            "import importlib.util, json, sys, tempfile\n"
            "import tvm\n"
            "from pathlib import Path\n"
            "from leka.candidate_source import validate_candidate_source\n"
            "from leka.lowering.local_tir_fusion import check_l1_contract\n"
            "def load(tag, text):\n"
            "    validate_candidate_source(text)\n"
            "    d = tempfile.mkdtemp(); p = Path(d)/f'{tag}.py'; p.write_text(text)\n"
            "    s = importlib.util.spec_from_file_location(tag, p)\n"
            "    m = importlib.util.module_from_spec(s); sys.modules[tag]=m; s.loader.exec_module(m)\n"
            "    return m.Module\n"
            "req = json.loads(sys.stdin.read())\n"
            "cand = load('leka_cand', req['artifact'])\n"
            "canon = load('leka_canon', req['canonical']) if req.get('canonical') else None\n"
            "out = check_l1_contract(cand, canon, allow_local_fusion=req['allow_local_fusion'])\n"
            "print(json.dumps(out))\n"
        )
        proc = run_cancellable(
            sandboxed_candidate_argv((self.tvm_python, "-c", script)),
            control=self.operation_control,
            input=json.dumps({"artifact": artifact_text, "canonical": canonical_text,
                              "allow_local_fusion": allow_local_fusion}),
            capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
            env=candidate_environment(
                pythonpath=str(REPO_ROOT),
                path="/usr/bin:/bin",
            ))
        if proc.returncode != 0:
            return {"ok": False, "fault_stage": "parse",
                    "error": (proc.stderr or proc.stdout)[-4000:]}
        return json.loads(proc.stdout)

    # -- job assembly ----------------------------------------------------------
    def _job_dir(self, mode: str) -> Path:
        job_id = f"{self.task.TASK_ID}-{mode}-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True)
        return job_dir

    def _base_job(self, final_module_source: str | TrustedIRModule, mode: str, binding: dict,
                  patch_source: str | None, t12_sha: str | None, profile: bool,
                  reference: dict | None = None, *, input_seed: int | None = None,
                  paired_replay: bool = False) -> Path:
        self._checkpoint()
        if paired_replay and (mode != "measure" or reference is None or
                              self.correctness_profile != ORDINARY_CORRECTNESS_PROFILE):
            raise ValueError("paired replay requires ordinary paired measurement")

        def validate_module(value: object) -> None:
            if isinstance(value, TrustedIRModule):
                return
            if not isinstance(value, str):
                raise TypeError("module must be candidate TVMScript or trusted IR JSON")
            validate_candidate_source(value)

        validate_module(final_module_source)
        if reference is not None:
            validate_module(reference["module_source"])
        facts = self.task.public_facts()
        policy = self.correctness_policy()
        if mode in {"measure", "audit"}:
            if input_seed is None:
                if mode == "audit":
                    raise ValueError("audit jobs require a driver-owned input seed")
                input_seed = self._seed
            if self.correctness_profile == ORDINARY_CORRECTNESS_PROFILE:
                allowed = (0,) if mode == "measure" and not paired_replay else (0, 1, 2)
                if type(input_seed) is not int or input_seed not in allowed:
                    raise ValueError("nonordinary input seed rejected before input materialization")
            elif input_seed == 900:
                raise ValueError("forbidden input seed rejected before input materialization")
        job_dir = self._job_dir(mode)

        def write_module(stem: str, value: str | TrustedIRModule) -> tuple[str, str]:
            if isinstance(value, TrustedIRModule):
                name = f"{stem}.json"
                (job_dir / name).write_text(value.json_text, encoding="utf-8")
                return name, "tvm_ir_json"
            name = f"{stem}.py"
            (job_dir / name).write_text(value, encoding="utf-8")
            return name, "tvmscript"

        module_path, module_format = write_module(
            "candidate_module", final_module_source
        )
        if patch_source is not None:
            (job_dir / f"patched.{self.cfg['l2_ext']}").write_text(patch_source)
        ref_fields = {"reference_module_path": None, "reference_patched_path": None,
                      "reference_t12_sha": None, "reference_module_format": None}
        if reference is not None:
            reference_path, reference_format = write_module(
                "reference_module", reference["module_source"]
            )
            ref_fields["reference_module_path"] = reference_path
            ref_fields["reference_module_format"] = reference_format
            if reference.get("patch_source") is not None:
                ref_name = f"reference_patched.{self.cfg['l2_ext']}"
                (job_dir / ref_name).write_text(reference["patch_source"])
                ref_fields["reference_patched_path"] = ref_name
            ref_fields["reference_t12_sha"] = reference.get("t12_sha")
        job = {
            "module_path": module_path,
            "module_format": module_format,
            "patched_path": f"patched.{self.cfg['l2_ext']}" if patch_source is not None else None,
            "t12_sha": t12_sha,
            **ref_fields,
            "entry": facts["entry"],
            "input_names": [s["name"] for s in facts["input_specs"]],
            "dynamic_input_names": [
                s["name"] for s in facts.get("dynamic_input_specs", facts["input_specs"])
            ],
            "state_input_names": [s["name"] for s in facts.get("state_specs", [])],
            "output_names": [s["name"] for s in facts["output_specs"]],
            "atol": facts["atol"], "rtol": facts["rtol"],
            "warmup": self.warmup, "repeat": self.repeat,
            "target": self.target, "device": self.cfg["device"],
            "mode": mode, "profile": profile, "binding": binding,
            "profile_id": self.correctness_profile,
            "correctness_policy": policy,
            "backend_cfg": {k: v for k, v in self.cfg.items()},
        }
        if mode in {"measure", "audit"}:
            inputs = self.task.make_inputs(input_seed)
            np.savez(job_dir / "inputs.npz", **inputs)
            np.savez(job_dir / "expected.npz", **self.task.reference(inputs))
            self._checkpoint()
            job["inputs_npz"] = "inputs.npz"
            job["expected_npz"] = "expected.npz"
            provenance = {
                "schema_version": "leka.correctness_provenance.v1",
                "policy": policy, "mode": mode,
                "input_seed": input_seed if self.correctness_profile == ORDINARY_CORRECTNESS_PROFILE else None,
                "module_sha256": _artifact_sha256(job_dir / module_path),
                "inputs_sha256": _artifact_sha256(job_dir / "inputs.npz"),
                "expected_sha256": _artifact_sha256(job_dir / "expected.npz"),
                "paired_arms_share_input_artifact": reference is not None,
                "reference_module_sha256": (
                    _artifact_sha256(job_dir / ref_fields["reference_module_path"])
                    if reference is not None else None
                ),
            }
            provenance_text = json.dumps(provenance, indent=2, allow_nan=False)
            (job_dir / "correctness_provenance.json").write_text(provenance_text)
            job["correctness_provenance_path"] = "correctness_provenance.json"
            job["correctness_provenance_sha256"] = hashlib.sha256(provenance_text.encode()).hexdigest()
        (job_dir / "job.json").write_text(json.dumps(job, indent=2))
        return job_dir

    # -- T12 extraction ---------------------------------------------------------
    def extract(self, final_module_source: str | TrustedIRModule, binding: dict) -> MeasureResult:
        self._checkpoint()
        job_dir = self._base_job(final_module_source, "extract", binding, None, None, False, None)
        raw = self._run_device_job(job_dir)
        self._checkpoint()
        ok = bool(raw.get("ok")) and bool(raw.get("generated_source"))
        diag = None if ok else diagnostics.sanitize(raw.get("error"), raw.get("fault_stage"),
                                                    lowest_edited="L1")
        return MeasureResult(ok=ok, tier="extract", layer="T12",
                             public={"pass": ok, "diagnostic": diag},
                             trusted={"raw": raw, "generated_source": raw.get("generated_source"),
                                      "generated_sha": raw.get("generated_sha")})

    # -- T1: device correctness + timing -------------------------------------
    def t1(self, final_module_source: str | TrustedIRModule, *, layer: str, binding: dict,
           patch_source: str | None = None, t12_sha: str | None = None,
           profile: bool = False, reference: dict | None = None) -> MeasureResult:
        return self._timed_measure(final_module_source, layer=layer, binding=binding,
            patch_source=patch_source, t12_sha=t12_sha, profile=profile, reference=reference)

    def paired_replay(self, final_module_source: str | TrustedIRModule, *, input_seed: int,
                      layer: str, binding: dict, reference: dict,
                      patch_source: str | None = None, t12_sha: str | None = None) -> MeasureResult:
        """Trusted post-search paired timing on one explicitly named ordinary seed.

        This is not exposed as an optimizer tool and does not change T1's seed 0.
        The caller must aggregate all three seeds before claiming formal validity.
        """
        if type(input_seed) is not int or input_seed not in (0, 1, 2):
            raise ValueError("nonordinary paired replay seed")
        return self._timed_measure(final_module_source, layer=layer, binding=binding,
            patch_source=patch_source, t12_sha=t12_sha, profile=False, reference=reference,
            replay_seed=input_seed)

    def _timed_measure(self, final_module_source: str | TrustedIRModule, *, layer: str,
                       binding: dict, patch_source: str | None, t12_sha: str | None,
                       profile: bool, reference: dict | None,
                       replay_seed: int | None = None) -> MeasureResult:
        self._checkpoint()
        job_dir = self._base_job(final_module_source, "measure", binding,
                                 patch_source, t12_sha, profile, reference,
                                 input_seed=replay_seed, paired_replay=replay_seed is not None)
        raw = self._run_device_job(job_dir, timeout_s=600)
        self._checkpoint()
        reference_ok = (
            reference is None or raw.get("reference_correct") is True
        )
        quality = assess_timing(raw, paired=reference is not None)
        ok = (raw.get("ok") is True and raw.get("correct") is True
              and reference_ok and quality["valid"] and quality["stable"])
        diag = None
        if not ok:
            timing_error = (("invalid timing evidence: " + ", ".join(quality["errors"]))
                            if not quality["valid"] else
                            "timing requires rCV <= 0.05" if not quality["stable"] else None)
            diag = diagnostics.sanitize(raw.get("error") or timing_error,
                                        raw.get("fault_stage") or ("timing" if timing_error else None),
                                        lowest_edited=layer)
        public = {
            "pass": ok,
            "correct": raw.get("correct"),
            "latency_ms": raw.get("latency_ms"),
            "latency_batches_ms": raw.get("latency_batches_ms"),
            "vs_base": raw.get("vs_base"),
            "timing_protocol": raw.get("timing_protocol"),
            "timing_samples": raw.get("timing_samples"),
            "measurement_quality": quality,
            "runtime_policy": raw.get("runtime_policy"),
            "parameter_plan": raw.get("parameter_plan"),
            "correctness_summary": raw.get("correctness"),
            "reference_correct": (
                raw.get("reference_correct") if reference is not None else None
            ),
            "reference_correctness_summary": (
                raw.get("reference_correctness") if reference is not None else None
            ),
            "diagnostic": diag,
            "binding": binding,
        }
        trusted = {"raw": raw, "job_dir": str(job_dir)}
        if self.correctness_profile == ORDINARY_CORRECTNESS_PROFILE:
            public.update({"profile_id": self.correctness_profile,
                           "correctness_policy": self.correctness_policy(),
                           "correctness_scope": ("paired_replay_single_seed" if replay_seed is not None
                                                 else "t1_single_seed_not_formal_audit")})
            trusted.update({"correctness_policy": self.correctness_policy(),
                            "correctness_provenance": str(job_dir / "correctness_provenance.json")})
        return MeasureResult(ok=ok, tier="t1", layer=layer, public=public,
                             trusted=trusted)

    def _fresh_audit_seeds(self) -> tuple[int, ...]:
        """Return the configured audit population, without changing policy.

        Ordinary mode recreates exactly 0/1/2 (including T1's seed 0).
        Generic hidden mode draws three distinct private seeds excluding T1.
        """
        if self.correctness_profile == ORDINARY_CORRECTNESS_PROFILE:
            return (0, 1, 2)
        seeds: set[int] = set()
        while len(seeds) < PROMOTION_AUDIT_SEED_COUNT:
            candidate = self._random_seed()
            if candidate != self._seed:
                seeds.add(candidate)
        return tuple(seeds)

    def audit(self, final_module_source: str | TrustedIRModule, *, layer: str,
              binding: dict, patch_source: str | None = None,
              t12_sha: str | None = None) -> MeasureResult:
        """Recreate and check the configured promotion-audit input population.

        Audit workers compile and execute the candidate but never warm up,
        time, or profile it. All runs must pass. Ordinary mode publicly names
        0/1/2, not their values; generic random seeds remain private.
        """
        runs: list[dict] = []
        seeds = self._fresh_audit_seeds()
        if self.correctness_profile == ORDINARY_CORRECTNESS_PROFILE and seeds != (0, 1, 2):
            raise ValueError("ordinary audit must cover exactly seeds 0/1/2")
        for seed in seeds:
            self._checkpoint()
            job_dir = self._base_job(
                final_module_source,
                "audit",
                binding,
                patch_source,
                t12_sha,
                False,
                None,
                input_seed=seed,
            )
            try:
                raw = self._run_device_job(job_dir, timeout_s=600)
            except OperationCancelled:
                raise
            except Exception as error:  # an incomplete audit fails closed
                raw = {
                    "ok": False,
                    "fault_stage": "transport",
                    "error": f"audit runner failed: {type(error).__name__}",
                }
            runs.append({"raw": raw, "job_dir": str(job_dir)})

        failures = [
            run["raw"] for run in runs
            if not (bool(run["raw"].get("ok")) and run["raw"].get("correct") is True)
        ]
        ok = not failures and len(runs) >= PROMOTION_AUDIT_SEED_COUNT
        diagnostic = None
        if not ok:
            failed = failures[0] if failures else {
                "fault_stage": "run_correctness",
                "error": "hidden correctness audit was incomplete",
            }
            diagnostic = diagnostics.sanitize(
                failed.get("error"), failed.get("fault_stage"), lowest_edited=layer
            )
        public = {
            "pass": ok,
            "correct": ok,
            "checks_completed": len(runs),
            "diagnostic": diagnostic,
            "binding": binding,
        }
        trusted = {"runs": runs}
        if self.correctness_profile == ORDINARY_CORRECTNESS_PROFILE:
            public.update({"profile_id": self.correctness_profile,
                           "correctness_policy": self.correctness_policy(),
                           "correctness_scope": "promotion_audit_three_inputs"})
            trusted["correctness_policy"] = self.correctness_policy()
        return MeasureResult(
            ok=ok,
            tier="audit",
            layer=layer,
            public=public,
            trusted=trusted,
        )
