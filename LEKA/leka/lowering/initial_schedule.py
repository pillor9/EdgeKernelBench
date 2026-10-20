"""Build an executable L1 starting point from unscheduled TIR.

The initial target schedule leaves PrimFuncs and L0 fusion boundaries unchanged.
It supplies the bootstrap/fallback implementation, not an agent improvement.
Subsequent L1 edits use the separate layer contract.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ..candidate_source import CandidateSourceError, validate_candidate_source
from ..operation_control import OperationControl, run_cancellable
from ..sandboxing import candidate_environment, sandboxed_candidate_argv


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class L1SeedResult:
    ok: bool
    source: str = ""
    error: str = ""
    fault_stage: str = ""
    scheduler: str = ""


def run_l1_seed(
    canonical_source: str,
    target: str,
    *,
    python: str = sys.executable,
    timeout_s: int = 120,
    operation_control: OperationControl | None = None,
) -> L1SeedResult:
    """Return an executable L1 seed without changing the canonical graph."""

    try:
        validate_candidate_source(canonical_source)
    except CandidateSourceError as error:
        return L1SeedResult(
            ok=False,
            error=f"canonical T01 source policy failure: {error}",
            fault_stage="l1_seed_policy",
        )
    proc = run_cancellable(
        sandboxed_candidate_argv(
            (python, "-m", "leka.lowering.initial_schedule_worker")
        ),
        control=operation_control,
        input=json.dumps({"canonical_source": canonical_source, "target": target}),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(REPO_ROOT),
        env=candidate_environment(
            pythonpath=str(REPO_ROOT),
            path="/usr/bin:/bin",
        ),
    )
    if proc.returncode != 0:
        return L1SeedResult(
            ok=False,
            error=(proc.stderr or proc.stdout)[-4000:],
            fault_stage="initial_schedule_worker",
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return L1SeedResult(
            ok=False,
            error=proc.stdout[-4000:],
            fault_stage="l1_seed_protocol",
        )
    if not payload.get("ok"):
        return L1SeedResult(
            ok=False,
            error=str(payload.get("error", ""))[:4000],
            fault_stage=str(payload.get("fault_stage", "l1_seed")),
        )
    return L1SeedResult(
        ok=True,
        source=str(payload["source"]),
        scheduler=str(payload.get("scheduler", "")),
    )
