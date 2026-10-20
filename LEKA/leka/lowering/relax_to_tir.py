"""T01: deterministic translation from L0 Relax to canonical, unscheduled TIR.

Agent-supplied L0 source runs in a subprocess worker (never in the driver
process).  The worker returns two module scripts:

  canonical  - relax main + unscheduled TIR functions (tir_agent's read-only input)
  scheduled  - legacy protocol alias of ``canonical``; L1 owns all scheduling

Source validation precedes worker execution; device checks establish numerical
correctness. Process separation alone does not sandbox arbitrary Python.
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
class T01Result:
    ok: bool
    canonical: str = ""
    scheduled_default: str = ""
    error: str = ""
    fault_stage: str = ""


def run_t01(
    l0_source: str,
    target: str,
    *,
    expected_l0_source: str,
    python: str = sys.executable,
    timeout_s: int = 120,
    operation_control: OperationControl | None = None,
) -> T01Result:
    """Validate and lower one L0 candidate against the trusted task ABI.

    ``expected_l0_source`` is the task adapter's naive Relax root.  Its graph
    is not imposed on the candidate; only its ``main`` positional StructInfo,
    return StructInfo, and ``num_input`` contract are frozen by the worker.
    """
    try:
        validate_candidate_source(expected_l0_source)
    except CandidateSourceError as error:
        return T01Result(
            ok=False,
            error=f"trusted expected L0 source policy failure: {error}",
            fault_stage="expected_l0_policy",
        )
    try:
        validate_candidate_source(l0_source)
    except CandidateSourceError as error:
        return T01Result(
            ok=False,
            error=f"candidate L0 source policy failure: {error}",
            fault_stage="exec_l0",
        )
    request = json.dumps(
        {
            "l0_source": l0_source,
            "expected_l0_source": expected_l0_source,
            "target": target,
        }
    )
    proc = run_cancellable(
        sandboxed_candidate_argv(
            (python, "-m", "leka.lowering.relax_to_tir_worker")
        ),
        control=operation_control,
        input=request,
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
        return T01Result(ok=False, error=(proc.stderr or proc.stdout)[-4000:], fault_stage="relax_to_tir_worker")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return T01Result(ok=False, error=proc.stdout[-4000:], fault_stage="t01_protocol")
    if not payload.get("ok"):
        # TVM tracebacks put the actionable diagnostic at the end.  Keep that
        # end when bounding worker output so L0 receives the actual failing op
        # instead of only C++ stack frames.
        return T01Result(
            ok=False,
            error=str(payload.get("error", ""))[-4000:],
            fault_stage=str(payload.get("fault_stage", "t01")),
        )
    return T01Result(
        ok=True,
        canonical=payload["canonical"],
        scheduled_default=payload["scheduled_default"],
    )
