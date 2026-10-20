"""Run layer agents and hardware feedback within a shared optimization budget.

Agents edit their own representations and request validation through ./measure.
The evaluator serializes device work, validates candidate promotion, and keeps
hardware evidence tied to the program version that produced it. Batch and
asynchronous scheduling both reserve time to validate the final selected stack.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import ExitStack, nullcontext
import hashlib
import json
import math
import re
import os
import sys
import stat
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from ..backends import BACKENDS
from ..hardware_feedback import attribution, evidence
from ..hardware_feedback.collectors.profile_artifacts import artifact_dir_for_jobs_root
from ..hardware_feedback.model_projection import archive_and_project
from ..hardware_feedback.quality import audit_digest
from ..lowering.interface_facts import build_interface_facts
from ..evaluation import diagnostics
from ..evaluation.evaluation_service import MeasureService, ORDINARY_CORRECTNESS_PROFILE, PROMOTION_AUDIT_SEED_COUNT
from ..operation_control import OperationCancelled, OperationControl
from ..parameter_plan import schema_document as parameter_plan_schema
from ..review.source_review import CodexReviewer, DisabledReviewer
from ..tasks.pytorch_task import load_pytorch_task
from .evaluation_server import MeasureServer, install_shim, validate_workspace_path
from .hardware_delivery import HardwareEvidenceDelivery, display_packet
from .observation_delivery import HardwareObservationDelivery
from .source_candidate import project_native_candidate
from .optimization_pipeline import Pipeline
from .agent_sessions import AgentSession
from .agent_turn_budget import RoundBudget
from .event_scheduler import (ResidentTasks, budget_prompt_suffix, run_event_loop,
                              historical_discussion_reply_proof, measurement_discussion_reply_context,
                              early_hardware_capture_gate)
from .shared_message_pool import (
    BROADCAST,
    MAX_TEXT_CHARS as MAX_PEER_TEXT_CHARS,
    SharedMessagePool,
    SharedMessagePoolError,
    backend_participants,
    binding_status,
    advisory_projection,
)

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
TVM_PYTHON = os.environ.get("LEKA_TVM_PYTHON", sys.executable)
QUIESCE_JOIN_MAX_S = 30.0
# ``AgentSession.stop`` may spend 20 s waiting for a graceful exit, two 5 s
# escalation waits, and two 5 s reader/watcher joins.  Finalization performs
# server quiescence and session shutdown sequentially, so the CLI admission
# check must reserve both worst-case teardown windows before the 75 s minimum
# needed to restore/audit/measure the selected stack.
SESSION_STOP_MAX_S = 40.0
FINALIZATION_TEARDOWN_MAX_S = QUIESCE_JOIN_MAX_S + SESSION_STOP_MAX_S
# Async hardware futures can still be unwinding after CLI/server shutdown.
ASYNC_WORKER_JOIN_MAX_S = QUIESCE_JOIN_MAX_S
FINALIZATION_EXIT_MARGIN_S = 10.0
MIN_FINALIZATION_START_BUDGET_S = 75.0
TERMINAL_PROVIDER_ERROR = re.compile(r"^\s*API Error:\s*(401|402|403)\b", re.I)
PROVIDER_API_ERROR = re.compile(r"^\s*API Error:\s*\d{3}\b", re.I)


class TerminalProviderError(RuntimeError):
    """A provider response that cannot recover inside the current run."""


def terminal_provider_error(result: dict | None) -> str | None:
    """Return a non-retryable provider status embedded in a result event."""

    reply = (result or {}).get("result", "")
    match = TERMINAL_PROVIDER_ERROR.match(reply if isinstance(reply, str) else "")
    return match.group(1) if match else None


def provider_turn_succeeded(result: dict | None) -> bool:
    """Whether a formal CLI result proves the model executed this turn.

    Retryable provider failures (429/5xx/overload) are formal stream events too,
    but the model never consumed the prompt semantically.  They must therefore
    leave peer advice unread for a later retry.  Minimal fake/legacy success
    events without ``subtype``/``is_error`` remain supported.
    """

    if not isinstance(result, dict):
        return False
    if result.get("is_error") is True:
        return False
    subtype = result.get("subtype")
    if subtype is not None and subtype != "success":
        return False
    reply = result.get("result", "")
    if isinstance(reply, str) and PROVIDER_API_ERROR.match(reply):
        return False
    return True


class DescendantRebaseScheduler:
    """Give invalidated descendants one bounded turn on each new upper base.

    Candidate freshness and branch lineage remain Pipeline authorities. This
    scheduler controls presentation only: after L0 changes, L1 gets one turn
    and then L2 gets one; after L1 changes, L2 gets one. A retryable provider
    failure is retried once, while a timed-out-but-still-running turn keeps its
    slot until its formal result arrives. This bounds the pause imposed on L0
    while preventing upper promotions from starving descendants for the whole
    wall budget.
    """

    max_provider_failures = 2

    def __init__(self) -> None:
        self._layers: list[str] = []
        self._provider_failures: dict[str, int] = {}

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(self._layers)

    @property
    def active(self) -> bool:
        return bool(self._layers)

    def observe_binding_change(self, before: dict, after: dict) -> str | None:
        """Replace rebase debt with descendants of the newest changed layer."""

        changed_layer = None
        descendants: list[str] = []
        if before.get("l0_sha") != after.get("l0_sha"):
            changed_layer, descendants = "L0", ["L1", "L2"]
        elif before.get("l1_sha") != after.get("l1_sha"):
            changed_layer, descendants = "L1", ["L2"]
        if changed_layer is not None:
            self._layers = descendants
            self._provider_failures.clear()
        return changed_layer

    def next(self, by_layer: dict[str, str]) -> tuple[str, str] | None:
        """Return the next available protected ``(layer, agent)`` slot."""

        while self._layers and self._layers[0] not in by_layer:
            self._provider_failures.pop(self._layers.pop(0), None)
        if not self._layers:
            return None
        layer = self._layers[0]
        return layer, by_layer[layer]

    def finish_turn(
        self,
        layer: str,
        *,
        completed: bool,
        provider_succeeded: bool,
        projection_current: bool = True,
    ) -> str:
        """Advance one protected slot, retaining pending or first-failed work."""

        if not self._layers or self._layers[0] != layer:
            return "superseded"
        if not completed:
            return "pending"
        if provider_succeeded and not projection_current:
            # Finishing an old in-flight turn is not a provider failure or a
            # rebased search opportunity. Next turn refreshes on the same
            # session; do not consume the descendant's failure allowance.
            return "refresh_required"
        if provider_succeeded:
            self._provider_failures.pop(layer, None)
            self._layers.pop(0)
            return "completed"
        failures = self._provider_failures.get(layer, 0) + 1
        self._provider_failures[layer] = failures
        if failures < self.max_provider_failures:
            return "retry"
        self._provider_failures.pop(layer, None)
        self._layers.pop(0)
        return "provider_failures_exhausted"

    def skip_exhausted(self, layer: str) -> str:
        """Release quota-blocked rebase debt without claiming a completed turn."""
        if not self._layers or self._layers[0] != layer:
            return "superseded"
        self._provider_failures.pop(layer, None)
        self._layers.pop(0)
        return "new_turn_quota_exhausted"


def validate_time_budgets(
    *, wall_minutes: float, turn_minutes: float, finalize_minutes: float,
    scheduling: str = "batch",
) -> None:
    """Reject impossible wall/finalization budgets before any device work."""

    for option, value in (("wall-minutes", wall_minutes),
                          ("turn-minutes", turn_minutes),
                          ("finalize-minutes", finalize_minutes)):
        if not math.isfinite(value):
            raise ValueError(f"--{option} must be finite")
    if wall_minutes <= 0:
        raise ValueError("--wall-minutes must be positive")
    if turn_minutes <= 0:
        raise ValueError("--turn-minutes must be positive")
    if finalize_minutes < 0 or finalize_minutes >= wall_minutes:
        raise ValueError(
            "--finalize-minutes must be non-negative and below --wall-minutes"
        )
    teardown = FINALIZATION_TEARDOWN_MAX_S + (
        ASYNC_WORKER_JOIN_MAX_S if scheduling == "async" else 0.0)
    usable_final_s = (
        finalize_minutes * 60
        - FINALIZATION_EXIT_MARGIN_S
        - teardown
    )
    if usable_final_s < MIN_FINALIZATION_START_BUDGET_S:
        minimum = (
            MIN_FINALIZATION_START_BUDGET_S + FINALIZATION_EXIT_MARGIN_S
            + teardown
        ) / 60
        raise ValueError(
            f"--finalize-minutes must be at least {minimum:.3f} so finalization "
            "retains its minimum start budget after bounded quiesce/session "
            "teardown and the exit margin"
        )


def validate_turn_limits(*, max_rounds: int | None, session_max_turns: int | None) -> None:
    """Validate optional optimization and session budgets."""
    for name, value in (("max-rounds", max_rounds), ("session-max-turns", session_max_turns)):
        if value is not None and (type(value) is not int or not 1 <= value <= 128):
            raise ValueError(f"--{name} must be between 1 and 128 when explicitly supplied")


def round_limit_reached(rounds: int, maximum: int | None) -> bool:
    return maximum is not None and rounds >= maximum


def bounded_turn_wait_seconds(*, turn_minutes: float, work_deadline: float) -> float:
    """A wait slice never grants time past the unchanged exploration deadline."""
    return min(turn_minutes * 60, max(0.0, work_deadline - time.monotonic()))


def invoke_layer_turn(session, *, prompt: str | None, turn_minutes: float, work_deadline: float):
    """Recheck the hard deadline when the worker actually invokes a session."""
    remaining = bounded_turn_wait_seconds(turn_minutes=turn_minutes, work_deadline=work_deadline)
    if prompt is None:
        return session.finish_pending(timeout_s=remaining)
    if remaining <= 0:
        return None
    return session.send(prompt, timeout_s=remaining)


def next_admissible_rebase(rebase, by_layer, sessions, budget, log):
    """Skip exhausted protected slots in one bounded pass, retaining pending work."""
    protected = rebase.next(by_layer)
    while protected is not None:
        layer, agent = protected
        if sessions[agent].has_pending() or budget.can_start(agent):
            return protected
        disposition = rebase.skip_exhausted(layer)
        log("descendant_rebase_turn_skipped", layer=layer, agent=agent,
            disposition=disposition, remaining_layers=list(rebase.layers))
        protected = rebase.next(by_layer)
    return None


def session_turn_options(maximum: int | None) -> dict[str, int]:
    return {} if maximum is None else {"max_turns": maximum}


def profile_text(
    name: str,
    backend: str,
    *,
    task_facts: dict,
    hardware_feedback: bool = True,
) -> str:
    backend_config = BACKENDS[backend]
    facts = task_facts
    public_contract = {
        "backend": backend,
        "target": backend_config["target"],
        "device": backend_config["device"],
        "runtime_num_threads": backend_config.get("runtime_num_threads"),
        "runtime_affinity_mode": backend_config.get("runtime_affinity_mode"),
        "runtime_policy_id": backend_config.get("runtime_policy_id"),
        "timing_protocol": backend_config["timing_protocol"],
        "correctness_atol": facts["atol"],
        "correctness_rtol": facts["rtol"],
        "correctness_authority": "task_source",
    }
    base = (PROMPTS / "_base_discipline.md").read_text()
    dialogue_path = PROMPTS / "_peer_dialogue.md"
    if not dialogue_path.is_file():
        # Older custom profile volumes inherit the shipped transport contract.
        dialogue_path = Path(__file__).resolve().parents[1] / "prompts" / "_peer_dialogue.md"
    dialogue = dialogue_path.read_text()
    scope = (
        "# Sealed runtime scope\n\n"
        f"- Target backend: `{backend}`. Every proposal ultimately runs on this "
        "device contract.\n"
        "- The following evaluator-owned runtime contract is authoritative for "
        "this run (null means not applicable):\n\n"
        f"```json\n{json.dumps(public_contract, sort_keys=True, indent=2)}\n```\n"
        "- Read only this backend's mounted wiki volume; never transfer numeric "
        "limits or syntax from another backend.\n\n"
    )
    if name.startswith("hw_sense"):
        recipe = f"(initial per-op breakdown criteria for {backend}; deep counters land later)\n" + \
            json.dumps(evidence.output_contract(backend), indent=2)
        hardware = (PROMPTS / "hardware_agent.md").read_text()
        return scope + hardware.replace("{backend}", backend).replace("{recipe}", recipe) + "\n\n" + dialogue
    text = scope + (PROMPTS / f"{name}.md").read_text() + "\n\n" + base + "\n\n" + dialogue
    if not hardware_feedback:
        text = text.replace("L0|L1|L2|HW|all", "L0|L1|L2|all")
        text += (
            "\n\n# Hardware-feedback ablation runtime\n"
            "No hardware-sense agent, profiler collection, per-kernel timing breakdown, "
            "hardware-counter evidence, or hardware-driven scheduling is available. "
            "Do not request these or wait for HW replies. Static hardware/ISA/compiler "
            "facts remain available in wiki/hardware. Use ordinary ./measure correctness, "
            "whole-task latency, paired gains and compiler diagnostics to optimize. "
            "Peer advice among L0/L1/L2, local fusion, base refresh, correctness rules "
            "and promotion are unchanged. The 'all' recipient means optimizer roles only.\n"
        )
    return text


def resident_roles(agents: dict[str, str], backend: str, *, hardware_feedback: bool) -> list[str]:
    """The ablated hardware role must have neither a workspace nor a provider."""
    return [*agents, *([BACKENDS[backend]["hw_sense"]] if hardware_feedback else [])]


def layer_agents(backend: str, selector: str) -> dict[str, str]:
    cfg = BACKENDS[backend]
    table = {"relax": ("relax_agent", "L0"), "tir": ("tir_agent", "L1"),
             "l2": (cfg["l2_agent"], "L2")}
    picked = {}
    for key in selector.split(","):
        key = key.strip()
        if key in table:
            name, layer = table[key]
            picked[name] = layer
    return picked


def _replace_workspace_text(path: Path, text: str) -> None:
    """Replace one direct workspace file without following prior links/hardlinks."""

    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if not path.is_file():
            raise RuntimeError(f"workspace projection path is not a file: {path}")
        # Unlink first so a hostile pre-existing hardlink cannot redirect the
        # driver's write outside the workspace.
        path.unlink()
    path.write_text(text, encoding="utf-8")


def refresh_inputs(ws: Path, agent: str, layer: str | None, pipe: Pipeline) -> None:
    # Keep projected files and their signed identity on one stack snapshot.
    # A concurrent promotion either happens before this critical section or
    # after it; in the latter case the server sees the signed snapshot as stale
    # and refuses to relabel the request.
    with pipe._lock:
        if layer == "L0":
            _replace_workspace_text(ws / "relax_mod.py", pipe.stack.read("L0"))
        elif layer == "L1":
            _write_read_only(
                ws / "canonical_tir.py", pipe.stack.read("T01")
            )
            _replace_workspace_text(
                ws / "scheduled_tir.py", pipe.stack.read("L1")
            )
        elif layer == "L2":
            pipe.ensure_t12()
            ext = pipe.stack.l2_ext
            generated = pipe.stack.read("T12")
            current = pipe.stack.read("L2")
            l2_envelope = pipe.workspace_binding_envelope(layer)
            project_native_candidate(ws, extension=ext, generated_source=generated,
                                     active_override=current, binding_payload=l2_envelope["payload"])
        if layer is not None:
            envelope = l2_envelope if layer == "L2" else pipe.workspace_binding_envelope(layer)
            _write_read_only(
                ws / "BASE_BINDING.json",
                json.dumps(
                    envelope,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
            )
            if layer in {"L1", "L2"}:
                sources = ({"canonical_tir.py": pipe.stack.read("T01"),
                            "scheduled_tir.py": pipe.stack.read("L1")}
                           if layer == "L1" else
                           {"scheduled_tir.py": pipe.stack.read("L1"),
                            f"generated.{pipe.stack.l2_ext}": pipe.stack.read("T12")})
                service = getattr(pipe, "service", None)
                facts = build_interface_facts(
                    layer=layer, backend=getattr(service, "backend", "unknown"),
                    binding_payload=envelope["payload"], sources=sources,
                    native_language=getattr(service, "cfg", {}).get("l2_lang"))
                _write_read_only(ws / "INTERFACE_FACTS.json", json.dumps(
                    facts, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n")
            if getattr(pipe, "operator_tree", None) is not None:
                _project_operator_tree(ws, layer, pipe)


WIKI_ROOT = Path(__file__).resolve().parents[1] / "skills" / "wiki"
WIKI_SOURCE_ROOT = WIKI_ROOT
_WIKI_DIR_OF_LAYER = {"L0": "L0_relax", "L1": "L1_tir", "L2": "L2_source", None: "hardware_feedback"}
_L0_SEMANTIC_FILES = (
    "PYTORCH_SEMANTICS.json",
    "PYTORCH_TASK.py",
    "IMPORTED_RELAX.py",
    "PARAMETER_PLAN_SCHEMA.json",
)
MAX_AGENT_PROTOCOL_BYTES = 16 * 1024


def mount_wiki(ws: Path, layer: str | None, backend: str) -> None:
    """Mount read-only knowledge for one layer and backend.

    PyTorch task semantics deliberately do not enter this tree: they are a
    per-task authority projected separately into the L0 workspace.  Every
    optimizer layer does, however, need to know where its final code runs, so
    the selected backend's architecture/contract pages are shared read-only
    alongside the layer-specific IR/kernel pages.
    """
    import shutil

    dest = ws / "wiki"
    if dest.exists():
        for path in sorted(dest.rglob("*"), reverse=True):
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() else 0o600)
        dest.chmod(0o700)
        shutil.rmtree(dest)
    section = WIKI_ROOT / _WIKI_DIR_OF_LAYER[layer]
    for sub in ("common", backend):
        src = section / sub
        if src.is_dir():
            shutil.copytree(src, dest / "layer" / sub, dirs_exist_ok=True)

    # Architecture and deployment-contract knowledge belongs to every layer,
    # not only to the profiler interpreter.  Keep it visibly separate from
    # layer vocabulary so a Relax agent does not mistake a warp fact for a
    # legal graph rewrite.
    hardware = WIKI_ROOT / "hardware_feedback" / backend
    if hardware.is_dir() and layer is not None:
        # Optimizers need the complete architecture/target contract, but the
        # operational collector recipe belongs only to the hardware-sense
        # role. EVIDENCE packets provide profiling results to optimizers.
        architecture = hardware / "arch_facts.md"
        if architecture.is_file():
            target = dest / "hardware" / backend
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(architecture, target / architecture.name)

    # Include the source records referenced by the mounted knowledge pages.
    source = WIKI_SOURCE_ROOT / "sources" / backend
    if source.is_dir():
        shutil.copytree(source, dest / "sources" / backend, dirs_exist_ok=True)
    tvm_compatibility = WIKI_SOURCE_ROOT / "sources" / "tvm_compatibility.json"
    if tvm_compatibility.is_file():
        (dest / "sources").mkdir(parents=True, exist_ok=True)
        shutil.copy2(tvm_compatibility, dest / "sources" / tvm_compatibility.name)

    # Include only citations used by the selected layer and backend.
    source_provenance = json.loads(
        (WIKI_SOURCE_ROOT / "PROVENANCE.json").read_text(encoding="utf-8")
    )
    cited: set[str] = set()
    citation_pattern = re.compile(r"\[([a-z][a-z0-9]*(?:-[a-z0-9]+)+)\]")
    for page in dest.rglob("*.md"):
        cited.update(citation_pattern.findall(page.read_text(encoding="utf-8")))
    registry = source_provenance.get("sources") or {}
    scoped_provenance = {
        "schema_version": "leka.scoped_wiki_provenance.v1",
        "backend": backend,
        "layer": layer if layer is not None else "hardware_sense",
        "derived_from_schema": source_provenance.get("schema_version"),
        "contains_benchmark_answers": False,
        "sources": {
            source_id: registry[source_id]
            for source_id in sorted(cited)
            if source_id in registry
        },
        "unregistered_inline_source_ids": sorted(cited - set(registry)),
    }
    _write_read_only(
        dest / "PROVENANCE.json",
        json.dumps(
            scoped_provenance,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )

    manifest = {
        "schema_version": "leka.wiki_mount.v1",
        "backend": backend,
        "layer": layer if layer is not None else "hardware_sense",
        "task_semantics_in_wiki": False,
        "layer_volume": f"layer/{backend}",
        "common_volume": "layer/common" if (dest / "layer" / "common").is_dir() else None,
        "hardware_volume": (
            f"hardware/{backend}" if layer is not None else f"layer/{backend}"
        ),
        "hardware_architecture_visible": True,
        "hardware_perf_recipe_visible": layer is None,
        "provenance": "PROVENANCE.json",
    }
    _write_read_only(
        dest / "MOUNT_MANIFEST.json",
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    # File-tool policy also denies writes to wiki/, but read-only modes make
    # accidental mutation fail closed outside a model session as well.
    for path in dest.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o555 if path.is_dir() else 0o444)
    dest.chmod(0o555)


def _write_read_only(path: Path, text: str) -> None:
    """Replace one agent-visible projection without trusting prior workspace state."""

    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if not path.is_file():
            raise RuntimeError(f"semantic projection path is not a file: {path}")
        path.chmod(0o600)
        path.unlink()
    path.write_text(text, encoding="utf-8")
    path.chmod(0o444)


def _project_hardware_stack(ws: Path, snapshot: list[tuple[str, str]], *,
                            expected_names: tuple[str, ...], binding: dict,
                            digest_id: str) -> None:
    """Project exactly the frozen stack, including absence of invalidated slots."""
    from .program_state import slot_files

    known_names = {name for ext in ("cu", "cl", "ll") for name in slot_files(ext).values()}
    names = [name for name, _ in snapshot]
    if (len(names) != len(set(names)) or not set(names) <= set(expected_names)
            or not set(expected_names) <= known_names):
        raise ValueError("invalid frozen hardware stack filenames")
    directory = ws / "stack_view"
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise RuntimeError("hardware stack view must be a real directory")
    directory.mkdir(exist_ok=True)
    directory.chmod(0o700)
    try:
        # Remove only evaluator-owned projection names. An absent L2 after an
        # upper-layer promotion must not leave the previous patch visible.
        for name in known_names - set(names):
            stale = directory / name
            if stale.is_symlink() or stale.is_file():
                stale.unlink()
            elif stale.exists():
                raise RuntimeError("hardware stack slot is not a regular file")
        for name, text in snapshot:
            _write_read_only(directory / name, text)
        manifest = {
            "schema_version": "leka.hardware_stack_view.v1", "binding": binding,
            "source_digest_id": digest_id,
            "files": {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                      for name, text in snapshot},
            "absent_files": sorted(set(expected_names) - set(names)),
            "authority": "frozen_profile_stack_not_latest_live_stack",
        }
        _write_read_only(directory / "SNAPSHOT_MANIFEST.json",
                         json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    finally:
        directory.chmod(0o555)


def _project_operator_tree(ws: Path, layer: str | None, pipe: Pipeline) -> None:
    """Publish whole-tree summaries and own-layer source without sharing trust."""

    import shutil

    dest = ws / "operator_tree"
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or not dest.is_dir():
            dest.unlink()
        else:
            for path in sorted(dest.rglob("*"), reverse=True):
                if not path.is_symlink():
                    path.chmod(0o700 if path.is_dir() else 0o600)
            dest.chmod(0o700)
            shutil.rmtree(dest)
    dest.mkdir(mode=0o700)
    projection = pipe.operator_tree.projection(layer)
    _write_read_only(
        dest / "TREE.json",
        json.dumps(
            projection,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )
    sources = dest / "own_layer_sources"
    sources.mkdir(mode=0o700)
    for node in projection["own_layer_nodes"]:
        source = pipe.operator_tree.source_for_projection(
            node["node_id"], layer=layer
        )
        if source is not None:
            _write_read_only(sources / f"{node['node_id']}.txt", source)
    sources.chmod(0o555)
    dest.chmod(0o555)


def _remove_l0_semantic_files(ws: Path) -> None:
    """Keep PyTorch semantics scoped to the L0 workspace on resumed runs."""

    for name in _L0_SEMANTIC_FILES:
        path = ws / name
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or path.is_file():
            if not path.is_symlink():
                path.chmod(0o600)
            path.unlink()
            continue
        raise RuntimeError(f"stale semantic projection path is not a file: {path}")


def _mount_l0_semantics(ws: Path, pipe: Pipeline, facts: dict) -> None:
    """Mount immutable semantic context and the importer safety root for L0 only."""

    semantics = getattr(pipe.task, "pytorch_semantics", None)
    if callable(semantics):
        payload = semantics()
        canonical = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        )
        pretty = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2,
        ) + "\n"
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        file_digest = hashlib.sha256(pretty.encode("utf-8")).hexdigest()
        declared_digest = getattr(pipe.task, "pytorch_semantics_sha256", None)
        if callable(declared_digest) and declared_digest() != digest:
            raise RuntimeError("PyTorch semantic projection digest mismatch")
        _write_read_only(ws / "PYTORCH_SEMANTICS.json", pretty)
        facts["pytorch_semantics"] = {
            "filename": "PYTORCH_SEMANTICS.json",
            "schema_version": payload.get("schema_version"),
            # ``sha256`` is retained as the backwards-compatible semantic
            # identity.  Publish both identities so the pretty on-disk file is
            # directly verifiable without guessing the canonicalization rule.
            "sha256": digest,
            "canonical_sha256": digest,
            "canonicalization": "json(sort_keys,separators=(',',':'),allow_nan=false)",
            "file_sha256": file_digest,
            "authority": "trusted_read_only_projection",
        }

    pytorch_source = getattr(pipe.task, "pytorch_source", None)
    if callable(pytorch_source):
        source = pytorch_source()
        _write_read_only(ws / "PYTORCH_TASK.py", source)
        facts["pytorch_task_source"] = {
            "filename": "PYTORCH_TASK.py",
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "authority": "public_benchmark_semantics_only",
            "contains_parameter_values": False,
            "contains_hidden_inputs": False,
        }

    plan_schema = parameter_plan_schema()
    _write_read_only(
        ws / "PARAMETER_PLAN_SCHEMA.json",
        json.dumps(
            plan_schema,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )
    facts["fixed_state_preparation"] = {
        "schema_file": "PARAMETER_PLAN_SCHEMA.json",
        "relax_main_attribute": plan_schema["relax_main_attribute"],
        "preparation_timed": False,
        "device_transfer_timed": False,
        "entry_execution_timed": True,
        "correctness_authority": "sealed_pytorch_oracle_plus_configured_promotion_audit",
    }

    naive_source = getattr(pipe.task, "naive_relax_source", None)
    if callable(naive_source):
        source = naive_source()
        _write_read_only(ws / "IMPORTED_RELAX.py", source)
        facts["imported_relax_seed"] = {
            "filename": "IMPORTED_RELAX.py",
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "role": "safe_reference_root_not_an_editing_constraint",
        }


def stage_workspace(run_dir: Path, agent: str, layer: str | None, pipe: Pipeline,
                    backend: str) -> Path:
    ws = run_dir / "agents" / agent
    if ws.is_symlink():
        raise RuntimeError("agent workspace may not be a symlink")
    ws.mkdir(parents=True, exist_ok=True)
    # This runs on fresh staging and every resume, before any trusted launcher
    # or semantic projection is created.
    validate_workspace_path(ws)
    install_shim(ws)
    if layer is None:
        _write_read_only(ws / "EVIDENCE_SCHEMA.json",
                         json.dumps(evidence.output_contract(backend), indent=2) + "\n")
    facts = dict(pipe.task.public_facts())
    policy_getter = getattr(getattr(pipe, "service", None), "correctness_policy", None)
    if callable(policy_getter):
        facts["correctness_policy"] = policy_getter()
    if layer == "L0":
        _mount_l0_semantics(ws, pipe, facts)
    else:
        _remove_l0_semantic_files(ws)
    _write_read_only(
        ws / "TASK_FACTS.json",
        json.dumps(facts, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
    )
    mount_wiki(ws, layer, backend)
    refresh_inputs(ws, agent, layer, pipe)
    return ws


def drain_inbox(pipe: Pipeline, agent: str) -> list[dict]:
    messages = []
    while True:
        m = pipe.mailbox.pop(agent)
        if m is None:
            return messages
        messages.append(m)


def retain_hardware_messages(pipe: Pipeline, agent: str, messages: list[dict]) -> list[dict]:
    """Keep HW packets across failed provider turns; acknowledge only success."""
    if not hasattr(pipe, "_hardware_inflight"):
        pipe._hardware_inflight = {}
    retained = pipe._hardware_inflight.setdefault(agent, {})
    other = []
    for message in messages:
        if message.get("kind") != "evidence":
            other.append(message)
            continue
        packet = message.get("packet") or {}
        identity = packet.get("evidence_id") or hashlib.sha256(
            json.dumps(message, sort_keys=True).encode()
        ).hexdigest()
        retained[identity] = message
    return other + list(retained.values())


def acknowledge_hardware_messages(pipe: Pipeline, agent: str) -> None:
    retained = getattr(pipe, "_hardware_inflight", {}).pop(agent, {})
    for identity in retained:
        pipe.log("evidence_turn_completed", agent=agent, evidence_id=identity,
                 meaning="included_in_successful_turn_not_proof_of_adoption")


def _turn_collaboration_checkpoint(layer: str, ws: Path, context: dict | None) -> str:
    """Put one concrete dependency check in the existing formal user turn.

    This is a prompt, not a published peer message or another provider turn.
    Only the model may fill and Write the optional bounded request/handoff.
    """
    if not isinstance(context, dict):
        return ""
    dialogue = context.get("dialogue") or {}
    target = {"L0": "L1", "L1": "L0", "L2": "L1"}.get(layer)
    if target not in dialogue.get("available_roles", {}):
        return ""
    turn_id = context.get("turn_id")
    if not isinstance(turn_id, str) or not _DRIVER_TURN_ID.fullmatch(turn_id):
        return ""
    # Derive the exact first slot instead of trusting a path from a message.
    slot = f"DIALOGUE.{turn_id}.1.json"
    if not dialogue.get("outbox_slots") or dialogue["outbox_slots"][0] != slot:
        return ""
    checks = {
        "L0": ("If a planned fusion, split or layout edit changes lowered boundaries or an interface, "
               "send L1 a planned handoff naming the current graph region, expected lowering change "
               "and scheduling assumption it may invalidate. A tensor being dynamic only rules out "
               "its fixed-state pre-pack, not graph dependencies. Ask about a constraint if its answer "
               "could change your plan."),
        "L1": ("Preserving the external ABI does not ensure your chosen canonical PrimFuncs or "
               "call region survive a graph edit. If that unresolved dependency affects your planned "
               "fusion/schedule, ask L0 about the named region and state which choice the answer changes."),
        "L2": ("A body-only patch is tied to the captured L1/T12 source; identical signatures do not "
               "make it valid under any L1 schedule. If a long rewrite relies on current tile, index, "
               "shared-buffer or launch assumptions and learning whether L1 will change them could "
               "alter your choice, ask about that named function and assumption. Do not invent an upcoming change."),
    }
    template = {
        "turn_id": turn_id, "intent": "handoff" if layer == "L0" else "request",
        "targets": [target], "topic": "interface_constraint",
        "summary": "Planned, unverified: <specific cross-layer dependency and decision>",
        "evidence": "<visible artifact/field and source binding; state unknowns>",
        "suggestion": "<one concrete question or falsifiable check for the owning peer>",
        "reply_to": None,
    }
    return (
        "[collaboration checkpoint] Read TURN_CONTEXT.json and PEER_INBOX.json first. "
        "Before lengthy code generation, check this dependency: " + checks[layer] + " "
        "Only if this affects your or the peer's concrete experiment assumptions, Write one specific message to "
        f"{ws / slot}. Replace template placeholders with visible facts; plans are unverified. "
        "No generic status or invented dependency. Skip a message when no concrete dependency needs "
        "communication. Continue independent work without waiting. "
        "Before your next measure, read the inbox once and answer a relevant question using "
        "its message_id. Source regeneration remains evaluator-owned; quotas and permissions stay unchanged."
        "\nTemplate:\n" + json.dumps(template, ensure_ascii=False)
    )


def layer_turn_text(agent: str, layer: str, pipe: Pipeline, ws: Path,
                    messages: list[dict], first: bool,
                    *, dialogue_context: dict | None = None) -> str:
    checkpoint = _turn_collaboration_checkpoint(layer, ws, dialogue_context)
    parts = [checkpoint] if checkpoint else []
    if first:
        ready = (
            "Begin work. Your workspace is ready: TASK_FACTS.json, your layer's "
            "artifacts (see your profile), operator_tree/TREE.json, and the ./measure "
            "executable. Goal: find a "
            "real, above-noise speedup on the current base and request PROMOTE."
        )
        if layer == "L0" and (ws / "PYTORCH_SEMANTICS.json").is_file():
            ready += (
                " PYTORCH_TASK.py and PYTORCH_SEMANTICS.json are the trusted public "
                "PyTorch program/structured semantics; IMPORTED_RELAX.py is only the "
                "immutable importer safety root. You may replace relax_mod.py completely "
                "with a new pure-Relax graph. PARAMETER_PLAN_SCHEMA.json describes the "
                "legal evaluator-owned fixed-state relayout/pre-pack ABI available to "
                "that graph; hidden values remain sealed and correctness is oracle-checked."
            )
        parts.append(ready)
    if layer == "L2" and (ws / "L2_CANDIDATE_ORIGIN.json").is_file():
        parts.append(
            "Your writable patched native artifact is already a complete copy of the "
            "captured active override, or generated source when no override exists. "
            "Use Edit for a focused function-body change instead of recreating the full "
            "module. L2_CANDIDATE_ORIGIN.json records only the initial copy's source/hash; "
            "it is not validation of subsequent edits. Preserve frozen ABI and run the "
            "ordinary T0/T1 gates before requesting promotion.")
    facts_first_feedback = bool(getattr(getattr(pipe, "_hardware_delivery", None),
                                        "inline_feedback", False))
    if facts_first_feedback:
        parts.append(
            "[complete hardware feedback transport] Run exactly one ./measure command "
            "per Bash tool call; use separate calls for T0, T1 and promote. Do not "
            "chain, pipe or truncate measurement output. Each response may contain "
            "one complete sealed hardware packet, but combined tool output can be "
            "persisted or truncated by the CLI. If this happens, or a response has "
            "hardware_evidence_feedback.status=path_fallback, do not repeat the "
            "measurement merely to recover hardware feedback. Read HW_EVIDENCE_INDEX.json "
            "in your workspace and then Read the relevant complete packet_file. "
            "The index/preview is not the packet. Check source_binding against your "
            "current inputs and separate profile_data/claim_assessment from unverified "
            "model judgments before using the feedback. A recovered hardware packet "
            "does not recover a hidden measurement result or prove promotion succeeded.")
    if getattr(pipe, "_hardware_observations", None) is not None:
        parts.append(
            "[early hardware observations] A measure response may include a complete "
            "hardware_observations card before the hardware agent finishes analysis. "
            "It contains a selected, source-bound VM timing view and collector scope, "
            "not a layer bottleneck diagnosis or complete hardware profile. Compare "
            "source binding to current inputs; do not use historical facts as current. "
            "Later sealed hypotheses are separate. Never repeat measurement just to "
            "fetch a card; keep the ordinary T0/T1 and promotion gates.")
    for m in messages:
        displayed_message = m
        if m["kind"] == "evidence":
            packet = m.get("packet") or {}
            source = packet.get("source_binding")
            current = pipe.stack.binding().to_dict()
            state = "current" if source == current else "historical_or_unknown"
            parts.append(
                f"[hardware evidence: {state}] Treat the diagnosis as a hypothesis. "
                "Use the attached source binding; if it differs from your current base, "
                "reassess applicability. Never interpret a missing counter as zero. "
                "Choose a useful experiment or explain in NOTES.md why no edit is warranted; "
                "correctness and paired promotion gates remain unchanged."
            )
            if facts_first_feedback:
                displayed_message = {**m, "packet": display_packet(packet)}
                parts.append(
                    "[hardware facts and hypotheses] The packet's profile_data and "
                    "claim_assessment, including interpretation_checks, precede the "
                    "model text. These are source-checked observations and deterministic "
                    "scope checks, not proof of a causal bottleneck. Rationale, suggested "
                    "direction, confidence and verdict are unverified model judgments; "
                    "target_layer assigns an experiment owner, not a proven bottleneck. "
                    "If the text contradicts the checked fields, do not use the text as "
                    "a fact. Historical counters describe only their captured source, "
                    "not your changed candidate. Unknown checks remain unknown. "
                    "A complete packet covers selected evidence only, not the full device.")
            pipe.log("evidence_prompt_included", agent=agent,
                     evidence_id=packet.get("evidence_id"), binding_status=state)
        parts.append(f"[inbox:{m['kind']}] {json.dumps(displayed_message, ensure_ascii=False)}")
        if m["kind"] in {"base_change", "tree_checkout"}:
            refresh_inputs(ws, agent, layer, pipe)
            parts.append("Your workspace input files have been refreshed to the new base.")
    binding = pipe.stack.binding().to_dict()
    active_branch_id = getattr(pipe.stack, "active_branch_id", None)
    branch_id = active_branch_id() if callable(active_branch_id) else "legacy-root"
    parts.append(
        f"[stack] epoch={binding['epoch']} branch={branch_id}. Continue this turn. "
        "Epoch is an observation revision; signed projected hashes and branch lineage "
        "decide candidate validity. Note: each turn has a "
        "fixed time budget — finish one small step (1-2 measured candidates with "
        "conclusions) and end the turn; unfinished work continues next turn.")
    parts.append(
        "If ./measure reports workspace_refresh_required or a stale branch/input, "
        "do not repeatedly submit, edit BASE_BINDING.json, or copy old candidates "
        "onto a new identity. Save your idea in NOTES.md and end this turn promptly. "
        "The driver refreshes inputs at the next turn boundary in this same "
        "conversation; rebase and remeasure then. This is not a correctness failure."
    )
    return "\n\n".join(parts)


def workspace_projection_status(pipe: Pipeline, workspace: Path, layer: str) -> tuple[bool, str]:
    """Inspect the signed projection without changing files or candidate lineage."""
    try:
        envelope = json.loads((workspace / "BASE_BINDING.json").read_text())
    except (OSError, ValueError):
        envelope = None
    return pipe.validate_workspace_binding(layer, envelope)


def prepare_layer_turn(
    pipe: Pipeline,
    pool: SharedMessagePool | None,
    *,
    agent: str,
    layer: str,
    workspace: Path,
    backend: str,
    first: bool,
) -> tuple[str, dict, tuple[str, ...], str]:
    """Atomically bind inbox, workspace refresh, prompt, and turn identity."""

    with pipe._lock:
        messages = retain_hardware_messages(pipe, agent, drain_inbox(pipe, agent))
        shared = peek_shared_advice(pipe, pool, agent)
        messages += shared
        # Inbox delivery is advisory, not proof that the resident workspace
        # still targets the active projection. A notice can have been consumed
        # during an earlier/pending turn. Repair at this turn boundary only;
        # never relabel an in-flight candidate or discard same-base edits.
        valid_projection, projection_reason = workspace_projection_status(
            pipe, workspace, layer
        )
        if not valid_projection and not any(
            message.get("kind") in {"base_change", "tree_checkout"}
            for message in messages
        ):
            messages.append({
                "kind": "base_change",
                "note": (
                    "Driver detected an outdated workspace at the new turn boundary. "
                    "Rebase onto the refreshed inputs in this same conversation; "
                    "old candidates remain tied to their original projection."
                ),
                "reason": projection_reason,
            })
        binding = pipe.stack.binding().to_dict()
        shared_message_ids = tuple(
            message["message_id"] for message in shared
        )
        _project_operator_tree(workspace, layer, pipe)
        turn_id = prepare_turn_context(
            pipe,
            pool,
            workspace=workspace,
            agent=agent,
            backend=backend,
            binding=binding,
            shared_message_ids=shared_message_ids,
        )
        prompt = layer_turn_text(
            agent,
            layer,
            pipe,
            workspace,
            messages,
            first,
            dialogue_context=_validated_turn_context(workspace, agent=agent, backend=backend),
        )
        if pipe.stack.binding().to_dict() != binding:
            raise RuntimeError("layer turn stack binding changed while prompt was staged")
    return prompt, binding, shared_message_ids, turn_id


def prepare_discussion_turn(pipe, pool, *, agent, layer, workspace, backend,
                            message_ids, allow_historical_replies=False):
    """Read/peer-only pilot. Do not drain unrelated formal hardware mail."""
    if layer != "L0" or not getattr(pipe, "turn_authority_enabled", False):
        raise ValueError("discussion preparation requires enabled L0 pilot")
    with pipe._lock:
        binding = pipe.stack.binding().to_dict()
        selected = [m for m in pool.peek(agent, limit=32, binding=binding)
                    if m["message_id"] in message_ids]
        measured_replies = []
        if any(m["topic"] == "measurement_observation" for m in selected):
            history = pool.history()
            by_id = {event["id"]: event for event in history}
            roles = {"relax_agent": "L0", "tir_agent": "L1", BACKENDS[backend]["l2_agent"]: "L2"}
            retained = []
            for message in selected:
                if message["topic"] == "measurement_observation":
                    context = measurement_discussion_reply_context(history=history,
                        event=by_id[message["message_id"]], role=agent, agents=roles, current=binding)
                    if context is None:
                        continue
                    measured_replies.append({"source_relation": context,
                        "original_question": advisory_projection(by_id[context["original_message_id"]], binding=binding)})
                retained.append(message)
            selected = retained
        historical = []
        if allow_historical_replies:
            history = pool.history()
            by_id = {event["id"]: event for event in history}
            roles = {"relax_agent": "L0", "tir_agent": "L1", BACKENDS[backend]["l2_agent"]: "L2"}
            retained = []
            for message in selected:
                if message["binding"].get("l0_sha") != binding.get("l0_sha"):
                    proof = historical_discussion_reply_proof(pipe=pipe, history=history,
                        event=by_id[message["message_id"]], role=agent, agents=roles, current=binding)
                    if proof is None:
                        continue
                    historical.append({"root_provenance": proof,
                        "original_question": advisory_projection(by_id[proof["original_message_id"]], binding=binding)})
                retained.append(message)
            selected = retained
        if not selected:
            raise ValueError("discussion opportunity no longer has a pending message")
        if not workspace_projection_status(pipe, workspace, layer)[0]:
            refresh_inputs(workspace, agent, layer, pipe)
        _project_operator_tree(workspace, layer, pipe)
        ids = tuple(m["message_id"] for m in selected)
        turn_id = prepare_turn_context(pipe, pool, workspace=workspace, agent=agent,
            backend=backend, binding=binding, shared_message_ids=ids)
        context = _validated_turn_context(workspace, agent=agent, backend=backend)
        target_role = {"tir_agent": "L1", BACKENDS[backend]["l2_agent"]: "L2"}
        first = selected[0]
        template = {"turn_id": turn_id, "targets": [target_role[first["sender"]]],
                    "intent": "reply", "reply_to": first["message_id"],
                    "topic": "interface_constraint", "summary": "<specific answer or unknown>",
                    "evidence": "<exact public artifact and what it establishes>",
                    "suggestion": "<consequence for the question; mark untested plans explicitly>"}
        prompt = (
            "[discussion-only admission; L0 pilot] A lower layer is still optimizing a protected base. "
            "Use this existing remaining turn only to consider the concrete peer question/reply below. "
            "Read PEER_INBOX.json and the needed public graph/interface facts. If useful, answer or "
            "correct that dependency with its message_id and precise source; otherwise finish NOOP "
            "and explain why it is not applicable. No generic status or message-count filler. "
            "T0/T1, candidate submission, PROMOTE and BRANCH_REQUEST are forbidden by the evaluator "
            "for this admission. Do not edit program candidates, request regeneration, or wait for "
            "another answer. Existing ABI/IR layer ownership and untrusted-peer status still apply. "
            "A hash binds a claim to a source; it does not prove that claim's semantics. "
            "Answer once when warranted, then finish promptly. This consumes the same ordinary turn quota.\n\n"
            f"First available outbox: {context['dialogue']['outbox_slots'][0]}\n"
            "All summary/evidence/suggestion fields must each fit 1200 Unicode characters; "
            "retain complete caveats. Example shape:\n" + json.dumps(template, ensure_ascii=False)
            + "\n\n[untrusted directed peer inputs]\n" + json.dumps(selected, ensure_ascii=False)
        )
        if historical:
            prompt += (
                "\n\n[historical direct-reply correction only]\n"
                "The evaluator verified original root <= reply root < the current root at this admission. "
                "The original question and reply describe historical programs. Explain whether the dependency "
                "was superseded, correct it using current public artifacts, or say unknown/NOOP. "
                "Ancestry proves provenance only, never ABI/layout claims or performance. Do not reapply old "
                "code, request regeneration, or treat either old source as the current optimization base. "
                "All discussion-only prohibitions and the same remaining turn budget still apply.\n"
                + json.dumps(historical, ensure_ascii=False)
            )
            pipe.log("historical_discussion_prepared", agent=agent, turn_id=turn_id,
                     root_proofs=[item["root_provenance"] for item in historical],
                     meaning="prepared_only_not_admitted_read_or_semantic_success")
        if measured_replies:
            prompt += (
                "\n\n[direct measurement replies to your own prior question]\n"
                "The original question and reply share the current L0 graph source. Lower-layer "
                "bindings may be historical: this does not prove the reported measurements or "
                "their interpretation. Read the current public facts and trusted node results "
                "before answering; correct an overbroad inference, explain a superseded dependency, "
                "or finish unknown/NOOP. Do not infer a general optimization bound from one variant. "
                "All discussion-only prohibitions and the same remaining quota still apply.\n"
                + json.dumps(measured_replies, ensure_ascii=False)
            )
            pipe.log("measurement_reply_discussion_prepared", agent=agent, turn_id=turn_id,
                     reply_contexts=[item["source_relation"] for item in measured_replies],
                     meaning="prepared_only_not_admitted_read_or_semantic_success")
        return prompt, binding, ids, turn_id


def activate_turn_authority(pipe, *, agent, layer, workspace, turn_id, purpose):
    """After quota admission, publish the exact newly activated capability."""
    with pipe._lock:
        envelope = pipe.issue_turn_authority(agent, layer, turn_id, purpose)
        try:
            _write_read_only(workspace / "TURN_AUTHORITY.json",
                             json.dumps(envelope, allow_nan=False, sort_keys=True, indent=2) + "\n")
        except BaseException:
            pipe.end_turn_authority(agent, envelope)
            raise
        return envelope


def _consume_hw_evidence(
    pipe: Pipeline, ws: Path, backend: str, digest: dict
) -> list[dict]:
    """Verify one completed hw turn against its immutable driver-held digest."""

    evidence_path = ws / "EVIDENCE.json"
    delivered: list[dict] = []
    rejected: list[dict] = []
    accepted: list[dict] = []
    packets = []
    try:
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise ValueError("write EVIDENCE.json as a regular JSON array; [] is a valid NOOP")
        packets = json.loads(evidence_path.read_text())
        if not isinstance(packets, list):
            raise ValueError("EVIDENCE.json must be an array")
    except (OSError, ValueError) as err:
        rejected.append({"index": None, "reason": str(err)[:300]})
        pipe.log("evidence_invalid", error=str(err)[:300])
        packets = []
    if not hasattr(pipe, "_hardware_queued_ids"):
        pipe._hardware_queued_ids = set()
    for index, packet in enumerate(packets):
        try:
            sealed = evidence.seal_packet(packet, digest, backend=backend)
        except (ValueError, TypeError, AttributeError, KeyError) as err:
            pipe.log("evidence_rejected", error=str(err)[:300])
            rejected.append({"index": index, "reason": str(err)[:300]})
            continue
        captures = getattr(pipe, "_hardware_capture_state", None)
        if captures is not None:
            # Timestamp the successful seal itself, before lock contention or
            # mailbox/projection I/O; keep the packet's canonical ID untouched.
            sealed_at = time.monotonic()
            # Capture bookkeeping has its own lock. The sealed source is
            # immutable; publishing it must not wait for another T1 job.
            captures.record_seal(digest.get("digest_id"), sealed["evidence_id"], sealed_at)
        target = sealed.get("target_layer")
        agent = {
            "L0": "relax_agent",
            "L1": "tir_agent",
            "L2": BACKENDS[backend]["l2_agent"],
        }.get(target)
        if agent:
            identity = sealed["evidence_id"]
            accepted.append({"index": index, "evidence_id": identity, "target": target})
            delivery = getattr(pipe, "_hardware_delivery", None)
            if delivery is not None:
                try:
                    # This trusted, post-seal entry is the sole publisher. The
                    # live projection is independent of formal mailbox delivery
                    # and retries even if that mailbox already holds this ID.
                    delivery.publish_sealed(agent, sealed)
                except Exception as err:
                    pipe.log("hardware_evidence_projection_error", agent=agent,
                             evidence_id=identity, error=str(err)[:300])
            if identity not in pipe._hardware_queued_ids:
                pipe.mailbox.send(agent, "evidence", {"packet": sealed})
                pipe._hardware_queued_ids.add(identity)
                delivered.append(sealed)
                pipe.log("evidence_delivered", target=target, agent=agent,
                         verdict=sealed.get("verdict"), evidence_id=identity)
    _write_read_only(ws / "EVIDENCE_FEEDBACK.json", json.dumps({
        "source_digest_id": digest.get("digest_id"),
        "accepted": accepted, "rejected": rejected,
        "contract": evidence.output_contract(backend, trusted_digest=digest),
    }, indent=2) + "\n")
    return delivered


def _early_hardware_model_checkpoint(pipe, session):
    """The optional second attempt cannot send a new prompt after exploration."""
    deadline = getattr(session, "_leka_early_capture_deadline", None)
    if deadline is not None:
        pipe.operation_control.checkpoint()
        if time.monotonic() >= deadline:
            raise OperationCancelled("early hardware exploration deadline reached")


def _consume_and_repair_hw_evidence(pipe, session, ws, backend, digest, deadline):
    """At most two corrections, same session/digest and original time slice.

    A timed-out correction remains pending under hw_turn's existing frozen-digest
    machinery. Already queued packets survive and are never delivered twice.
    """
    all_delivered = getattr(session, "_leka_hw_delivered", [])
    while True:
        all_delivered.extend(_consume_hw_evidence(pipe, ws, backend, digest))
        session._leka_hw_delivered = all_delivered
        feedback = json.loads((ws / "EVIDENCE_FEEDBACK.json").read_text())
        attempts = getattr(session, "_leka_hw_repair_attempts", 0)
        remaining = deadline - time.monotonic()
        if not feedback["rejected"] or attempts >= 2 or remaining <= 0:
            if feedback["rejected"]:
                pipe.log("evidence_repair_deferred", rejected=len(feedback["rejected"]),
                         attempts=attempts, reason="turn_budget_or_attempt_limit")
            for attr in ("_leka_hw_delivered", "_leka_hw_repair_attempts"):
                if hasattr(session, attr):
                    delattr(session, attr)
            return all_delivered, False
        _early_hardware_model_checkpoint(pipe, session)
        session._leka_hw_repair_attempts = attempts + 1
        pipe.log("evidence_repair_requested", attempt=attempts + 1,
                 rejected=len(feedback["rejected"]))
        pipe.log("hardware_model_request_admitted", request_kind="correction",
                 digest_id=digest.get("digest_id"), correction_attempt=attempts + 1,
                 turn_id=getattr(session, "_leka_pending_hw_turn_id", ""),
                 meaning="resident_user_prompt_admission; not_provider_http_count_or_bill")
        result = session.send(
            "Your hardware feedback was partially or wholly rejected. Read the trusted "
            "EVIDENCE_FEEDBACK.json, EVIDENCE_SCHEMA.json and the same QUANTITY_CONTRACT.json. Correct only the rejected "
            "packets in EVIDENCE.json on the SAME PROFILE_DIGEST.json and stack_view, "
            "in this conversation. verdict must be an exact category key, not attributed "
            "or confirmed. Do not invent evidence or force advice to every layer. "
            "Write [] if nothing is justified. No new profiling or optimization. "
            "Finish promptly within this remaining time slice.", timeout_s=remaining,
        )
        if result is None:
            return all_delivered, True
        pipe.log("evidence_repair_result", attempt=attempts + 1,
                 cost_usd=result.get("total_cost_usd"),
                 succeeded=provider_turn_succeeded(result))
        status = terminal_provider_error(result)
        if status is not None:
            raise TerminalProviderError(f"hardware agent provider returned HTTP {status}")
        if not provider_turn_succeeded(result):
            pipe.log("evidence_repair_provider_error", action="retain_feedback_for_next_turn")
            for attr in ("_leka_hw_delivered", "_leka_hw_repair_attempts"):
                if hasattr(session, attr):
                    delattr(session, attr)
            return all_delivered, False


def _static_parameter_opportunities(task) -> list[dict]:
    """Publish value-free fixed-state layout facts for hardware attribution."""

    semantic_projection = getattr(task, "pytorch_semantics", None)
    if not callable(semantic_projection):
        return []
    try:
        semantics = semantic_projection()
    except Exception:
        return []
    abi = semantics.get("abi") if isinstance(semantics, dict) else None
    inputs = abi.get("inputs") if isinstance(abi, dict) else None
    if not isinstance(inputs, list):
        return []
    out: list[dict] = []
    for item in inputs[:256]:
        if not isinstance(item, dict) or item.get("kind") == "user_input":
            continue
        tensor = item.get("tensor")
        if not isinstance(tensor, dict):
            continue
        shape = tensor.get("shape")
        if not isinstance(shape, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in shape
        ):
            continue
        opportunities = []
        if len(shape) >= 2:
            opportunities.append("one_time_layout_or_pack")
        out.append(
            {
                "id": str(item.get("relax_name") or item.get("graph_name") or "fixed"),
                "semantic_role": str(item.get("semantic_role") or "fixed_state"),
                "shape": shape,
                "dtype": str(tensor.get("dtype") or "unknown"),
                "nbytes": tensor.get("nbytes") if isinstance(tensor.get("nbytes"), int) else None,
                "opportunities": opportunities,
                "values_visible": False,
            }
        )
    return out


def _intermediate_traffic_context(
    stack_snapshot: list[tuple[str, str]], breakdown: list[dict]
) -> dict:
    """Summarize exact structural counts without pretending to know byte traffic."""

    sources = dict(stack_snapshot)
    scheduled = sources.get("scheduled_tir.py", "")
    canonical = sources.get("canonical_tir.py", "")
    module = scheduled or canonical
    kernel_rows = [
        row
        for row in breakdown
        if isinstance(row, dict) and not str(row.get("name", "")).startswith("vm.")
    ]
    return {
        "call_tir_site_count": module.count("R.call_tir"),
        "primfunc_count": module.count("@T.prim_func"),
        "tir_parallel_loop_count": module.count("T.parallel("),
        "tir_vectorized_loop_count": module.count("T.vectorized("),
        "tir_alloc_buffer_count": module.count("T.alloc_buffer("),
        "profiled_launch_count": sum(
            int(row.get("count") or 0) for row in kernel_rows
        ),
        "profiled_total_kernel_time_us": sum(
            float(row.get("duration_us") or 0.0) for row in kernel_rows
        ),
        # Exact dynamic byte traffic requires liveness/alias analysis or a
        # counter that this phone does not expose. Keep this explicitly unknown.
        "intermediate_tensor_bytes": "unknown",
    }


def _primfunc_profile(breakdown: list[dict]) -> dict:
    """Normalize TVM VM timing rows without inventing static symbol mapping."""

    rows: list[dict] = []
    for row in breakdown:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "kind": (
                    "vm_builtin" if name.startswith("vm.")
                    else "primfunc_or_compiled_call"
                ),
                "count": row.get("count"),
                "duration_us": row.get("duration_us"),
                "percent": row.get("percent"),
                "static_primfunc_id": "unknown",
            }
        )
    return {
        "schema_version": "leka.primfunc_profile.v1",
        "status": "available" if rows else "unavailable",
        "scope": "same_measurement_tvm_vm_per_operation",
        "exact_stack_binding": True,
        "counter_attribution": "none",
        "mapping": (
            "TVM runtime operation names are preserved verbatim; a static PrimFunc id "
            "is unknown unless the backend supplies an independently verified symbol map."
        ),
        "rows": rows,
    }


def profile_measurement_status(measured) -> tuple[dict, list[dict]]:
    """Separate successful T1 from available VM profiling; bound every error.

    Old diagnostic/test transports omitted ``ok``. Preserve their rows as
    explicitly unverified legacy observations; an explicit False always wins.
    """
    public = getattr(measured, "public", {})
    trusted = getattr(measured, "trusted", {})
    raw = trusted.get("raw", {}) if isinstance(trusted, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    flag = getattr(measured, "ok", None)
    if type(flag) is not bool:
        flag = public.get("pass") if isinstance(public, dict) else None
    if type(flag) is not bool:
        flag = raw.get("ok")
    flag = flag if type(flag) is bool else None
    if raw.get("correct") is False or (isinstance(public, dict) and public.get("correct") is False):
        flag = False
    received = raw.get("profile_breakdown")
    rows = received if isinstance(received, list) else []
    usable_rows = [row for row in rows if isinstance(row, dict)]
    errors = []
    profile_error = raw.get("profile_error")
    if profile_error:
        errors.append(diagnostics.sanitize(str(profile_error), "profile", lowest_edited="L2"))
    if flag is False:
        status = "discarded_after_measurement_failure"
        errors.append(diagnostics.sanitize(
            str(raw.get("error") or "Trusted T1 measurement did not pass; discard profiler evidence."),
            raw.get("fault_stage") or "timing", lowest_edited="L2"))
        usable_rows = []
    elif profile_error:
        status = "unavailable"
        usable_rows = []
    elif not usable_rows:
        status = "unavailable"
        errors.append(diagnostics.sanitize(
            "The VM profiler returned no operation rows; per-operation timing is unavailable.",
            "profile", lowest_edited="L2"))
    else:
        status = "available" if flag is True else "available_measurement_unverified"
    return {
        "schema_version": "leka.profile_status.v1",
        "source": "relax_vm_profile",
        "measurement_ok": flag,
        "measurement_status": "passed" if flag is True else "failed" if flag is False else "unknown_legacy_transport",
        "status": status,
        "received_row_count": len(rows),
        "available_row_count": len(usable_rows),
        "diagnostics": errors,
        "successful_t1_implies_available_profile": False,
    }, usable_rows



def _project_hardware_quantity_contract(pipe, ws: Path, digest: dict, backend: str) -> dict:
    """Read-only explanation of units, not evidence or an additional measurement."""
    from ..hardware_feedback.quantity_contract import build_quantity_contract
    try:
        contract = build_quantity_contract(digest, backend)
    except (ValueError, TypeError, OverflowError, RecursionError):
        # Replace any older view rather than letting a new digest inherit it.
        contract = {"schema_version": "leka.hardware_quantity_contract_unavailable.v1",
                    "status": "unavailable", "source_digest_id": None,
                    "reason": "source_or_budget_contract_not_satisfied",
                    "instruction": "Use this turn's original PROFILE_DIGEST.json and its explicit unknowns."}
    text = json.dumps(contract, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if len(text.encode("utf-8")) > 16 * 1024:
        raise ValueError("quantity contract pretty view exceeds its fixed budget")
    _write_read_only(ws / "QUANTITY_CONTRACT.json", text)
    pipe.log("hardware_quantity_contract_projected", agent=BACKENDS[backend]["hw_sense"],
             source_digest_id=contract.get("source_digest_id"),
             quantity_contract_id=contract.get("quantity_contract_id"),
             file_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
             file_bytes=len(text.encode("utf-8")), monotonic_s=time.monotonic(),
             meaning="source_derived_units_view_not_evidence_or_semantic_certification")
    return contract


def hw_turn(
    pipe: Pipeline,
    session: AgentSession,
    ws: Path,
    backend: str,
    timeout_s: float,
    *,
    rk_collector: str = "off",
    shared_pool: SharedMessagePool | None = None,
    capture_state=None,
    fresh_attempt_id: str | None = None,
    early_opportunity=None,
) -> list[dict]:
    """Collect -> reason -> verify, without overwriting a timed-out turn."""

    hw_agent = getattr(session, "name", BACKENDS[backend]["hw_sense"])
    deadline = time.monotonic() + timeout_s

    # A previous model call may have finished after its wall slice.  Its
    # digest stays in trusted driver memory; consume that result and evidence
    # before collecting or deleting anything for a later turn.
    if session.has_pending():
        digest = getattr(session, "_leka_pending_digest", None)
        if not isinstance(digest, dict):
            raise RuntimeError("pending hardware turn has no trusted digest")
        _write_read_only(ws / "EVIDENCE_SCHEMA.json", json.dumps(
            evidence.output_contract(backend, trusted_digest=digest), indent=2) + "\n")
        result = session.finish_pending(timeout_s=timeout_s)
        if result is None:
            return []
        provider_status = terminal_provider_error(result)
        if provider_status is not None:
            delattr(session, "_leka_pending_digest")
            raise TerminalProviderError(
                f"hardware agent provider returned HTTP {provider_status}"
            )
        if not provider_turn_succeeded(result):
            pipe.log(
                "hw_turn_provider_error",
                agent=hw_agent,
                resumed_pending=True,
                action="leave_peer_advice_unacknowledged",
                cost_usd=result.get("total_cost_usd"),
            )
            delattr(session, "_leka_pending_digest")
            for attribute in (
                "_leka_pending_shared_message_ids",
                "_leka_pending_hw_turn_id",
                "_leka_hw_delivered",
                "_leka_hw_repair_attempts",
            ):
                if hasattr(session, attribute):
                    delattr(session, attribute)
            return []
        shared_message_ids = tuple(
            getattr(session, "_leka_pending_shared_message_ids", ())
        )
        turn_id = str(getattr(session, "_leka_pending_hw_turn_id", ""))
        forward_help_reports(pipe, hw_agent, ws, backend)
        delivered, repair_pending = _consume_and_repair_hw_evidence(
            pipe, session, ws, backend, digest, deadline)
        if repair_pending:
            return delivered
        operator_tree = getattr(pipe, "operator_tree", None)
        if operator_tree is not None:
            with pipe._lock:
                operator_tree.record_hardware_observation(
                    digest=digest,
                    hypotheses=digest.get("initial_hypotheses") or [],
                    evidence_packets=delivered,
                )
        forward_shared_advice(
            pipe,
            shared_pool,
            agent=hw_agent,
            workspace=ws,
            backend=backend,
            binding=digest["binding"],
            turn_id=turn_id,
        )
        acknowledge_shared_advice(
            pipe,
            shared_pool,
            hw_agent,
            shared_message_ids,
            turn_id=turn_id,
            binding=digest["binding"],
        )
        delattr(session, "_leka_pending_digest")
        for attribute in (
            "_leka_pending_shared_message_ids",
            "_leka_pending_hw_turn_id",
        ):
            if hasattr(session, attribute):
                delattr(session, attribute)
        pipe.log(
            "hw_turn",
            resumed_pending=True,
            cost_usd=result.get("total_cost_usd"),
            delivered=len(delivered),
        )
        return delivered

    from ..hardware_feedback.collectors import device_profiler as deep

    # An exception/failed correction must not carry deliveries or attempt counts
    # into an unrelated freshly collected digest. Existing mailbox items remain.
    for attr in ("_leka_hw_delivered", "_leka_hw_repair_attempts"):
        if hasattr(session, attr):
            delattr(session, attr)

    # Keep the profile job, deep counters, binding, and projected stack on one
    # frozen epoch.  MeasureServer may promote concurrently, so every live
    # stack read and both Jetson jobs remain under the Pipeline's re-entrant
    # lock; workspace I/O uses the captured immutable texts after release.
    counters, deep_rows, collector_diagnostics = {}, [], []
    live_runtime_context: dict[str, object] = {}
    collector_status: dict[str, object] = {
        "schema_version": "leka.collector_status.v1",
        "mode": (
            rk_collector
            if backend == "rk3588_gpu"
            else "ncu" if backend == "jetson_gpu" else "simpleperf"
        ),
        "status": "pending",
        "exact_bundle": False,
        "scope": (
            "gpu_global_window" if backend == "rk3588_gpu"
            else "per_kernel_replay" if backend == "jetson_gpu"
            else "rpc_process_window_without_pid_start_proof"
        ),
    }

    def record_collector_error(collector: str, raw: object, *, fallback: str | None = None):
        diagnostic = diagnostics.sanitize(
            str(raw or "collector unavailable"),
            "device_setup",
            lowest_edited="L2",
        )
        diagnostic.update(
            {
                "backend": backend,
                "collector": collector,
                "retryable": True,
                "fallback": fallback,
            }
        )
        collector_diagnostics.append(diagnostic)
        pipe.log(
            "deep_collector_error",
            backend=backend,
            collector=collector,
            signature=diagnostic["error_feedback"]["signature"],
            fallback=fallback,
        )

    stack_snapshot: list[tuple[str, str]] = []
    tree_node_id = None
    # The evaluator-owned jobs root is outside all writable agent workspaces.
    # Never accept an artifact destination from the hardware agent or digest.
    trusted_jobs_root = getattr(getattr(pipe, "service", None), "jobs_root", None)
    hardware_artifact_dir = (artifact_dir_for_jobs_root(trusted_jobs_root)
                             if trusted_jobs_root is not None else None)
    lease_factory = getattr(getattr(pipe, "service", None), "device_lease", None)
    capture_seq = None
    with ExitStack() as capture_locks:
        if early_opportunity is not None:
            if not capture_locks.enter_context(early_hardware_capture_gate(pipe, early_opportunity)):
                return []
        else:
            # Ordinary first/default HW keeps the original lock/lease order
            # and timeout; only the opt-in second attempt tries zero wait.
            capture_locks.enter_context(pipe._lock)
            capture_locks.enter_context(lease_factory() if callable(lease_factory) else nullcontext())
        pipe.operation_control.checkpoint()
        if capture_state is not None:
            capture_seq = capture_state.start(fresh_attempt_id)
            capture_locks.callback(capture_state.close_capture, fresh_attempt_id, capture_seq)
        if backend == "rk3588_gpu":
            if rk_collector == "off":
                # Preserve the hardware-sense turn and TVM profile breakdown,
                # but do not touch Mali counter/debug interfaces by default.
                measured = pipe.measure_default(profile=True)
                collector_status["status"] = "disabled_by_run_config"
            elif rk_collector == "devfreq":
                wrapped = deep.collect_rk3588_devfreq(
                    pipe.measure_default,
                    profile=True,
                    operation_control=pipe.operation_control,
                )
                measured = wrapped.pop("job_result")
                counters = wrapped.get("counters") or {}
                collector_status["provenance"] = wrapped.get("provenance") or {}
                if "error" in wrapped:
                    record_collector_error("rk3588_devfreq", wrapped["error"])
                collector_status["status"] = (
                    "unavailable" if "error" in wrapped else "collected"
                )
            elif rk_collector == "hwcpipe":
                wrapped = deep.collect_rk3588_hwcpipe_during(
                    pipe.measure_default,
                    profile=True,
                    operation_control=pipe.operation_control,
                    artifact_dir=hardware_artifact_dir,
                )
                measured = wrapped.pop("job_result")
                counters = wrapped.get("counters") or {}
                collector_status["provenance"] = wrapped.get("provenance") or {}
                if "error" in wrapped:
                    # Never run a second workload as an automatic fallback.
                    # A collector failure may be the first sign that the board
                    # is unhealthy; retrying here previously amplified it.
                    record_collector_error("rk3588_hwcpipe", wrapped["error"])
                collector_status["status"] = (
                    "unavailable" if "error" in wrapped else "collected"
                )
            else:
                raise ValueError(f"unknown RK3588 collector mode: {rk_collector}")
        elif backend == "vivo_cpu":
            wrapped = deep.collect_vivo_simpleperf_during(
                pipe.measure_default,
                profile=True,
                serial=BACKENDS[backend]["vivo_serial"],
                require_fresh_rpc_owner=True,
                operation_control=pipe.operation_control,
                artifact_dir=hardware_artifact_dir,
            )
            measured = wrapped.pop("job_result")
            counters = wrapped.get("counters") or {}
            collector_status["provenance"] = wrapped.get("provenance") or {}
            live_runtime_context = wrapped.get("runtime_context") or {}
            if "error" in wrapped:
                record_collector_error("vivo_simpleperf", wrapped["error"])
            collector_status["status"] = (
                "unavailable" if "error" in wrapped else "collected"
            )
        else:
            measured = pipe.measure_default(profile=True)
            frozen_binding = pipe.stack.binding().to_dict()
            measured_binding = measured.public.get("binding")
            if measured_binding != frozen_binding:
                raise RuntimeError("profile result binding changed before NCU collection")
            measurement_status, _ = profile_measurement_status(measured)
            frozen_patch = pipe.stack.read("L2")
            ncu = ({"error": "NCU replay skipped after failed trusted T1 measurement"}
                   if measurement_status["measurement_ok"] is False else
                   deep.collect_jetson_ncu(
                       pipe.service, pipe.final_module_source(), frozen_binding,
                       patch_source=frozen_patch,
                       t12_sha=(pipe.stack.sha("T12") if frozen_patch is not None else None),
                       operation_control=pipe.operation_control))
            collector_status["provenance"] = ncu.get("provenance") or {}
            if "error" in ncu:
                record_collector_error("jetson_ncu", ncu["error"])
                collector_status["status"] = "unavailable"
            else:
                counters = ncu.get("counters") or {}
                deep_rows = ncu.get("rows") or []
                collector_status["status"] = "collected"
                collector_status["exact_bundle"] = True

        provenance = collector_status.get("provenance")
        if isinstance(provenance, dict) and isinstance(provenance.get("scope"), str):
            collector_status["scope"] = provenance["scope"]

        unavailable, unavailable_reason = device_unavailable(pipe)
        if unavailable:
            pipe.log(
                "hw_turn_aborted_device_unavailable",
                backend=backend,
                reason=unavailable_reason,
            )
            return []

        measured_binding = measured.public.get("binding")
        current_binding = pipe.stack.binding().to_dict()
        if not isinstance(measured_binding, dict) or measured_binding != current_binding:
            raise RuntimeError("hardware profile and stack snapshot bindings differ")
        digest_binding = dict(measured_binding)
        stack_view_names = tuple(pipe.stack.path(slot).name
                                 for slot in ("L0", "T01", "L1", "T12", "L2"))
        for slot in ("L0", "T01", "L1", "T12", "L2"):
            text = pipe.stack.read(slot)
            if text is not None:
                stack_snapshot.append((pipe.stack.path(slot).name, text))
        # Hardware sense receives the same whole-tree topology/measurement
        # projection as optimizer layers, but no optimizer source payload.
        if getattr(pipe, "operator_tree", None) is not None:
            tree_node_id = pipe.operator_tree.projection(None).get("active_leaf_node_id")
            _project_operator_tree(ws, None, pipe)
        pipe.operation_control.checkpoint()
        if capture_state is not None:
            captured_profile_status, _ = profile_measurement_status(measured)
            capture_state.finish(fresh_attempt_id, capture_seq, binding=digest_binding,
                slot_sha={slot: pipe.stack.sha(slot) for slot in ("L0", "T01", "L1", "T12", "L2")},
                tree_node_id=tree_node_id, profile_status=captured_profile_status)
    profile_status, breakdown = profile_measurement_status(measured)
    measurement_failed = profile_status["measurement_ok"] is False
    if measurement_failed:
        counters, deep_rows = {}, []
        collector_status["status"] = "discarded_after_measurement_failure"
        collector_status["exact_bundle"] = False
    measured_policy = measured.public.get("runtime_policy")
    configured_threads = BACKENDS[backend].get("runtime_num_threads")
    runtime_context = {
        **live_runtime_context,
        "runtime_policy_id": BACKENDS[backend].get("runtime_policy_id"),
        "configured_affinity_mode": BACKENDS[backend].get("runtime_affinity_mode"),
        "configured_num_threads": configured_threads,
        "observed_num_threads": (
            measured_policy.get("num_threads")
            if isinstance(measured_policy, dict)
            else None
        ),
        "thread_count_authority": (
            "trusted_worker_runtime.NumThreads"
            if isinstance(measured_policy, dict)
            else "unknown"
        ),
    }
    # Small test/diagnostic pipeline doubles may not carry a benchmark task;
    # absence means there is no trusted fixed-state projection to publish.
    static_opportunities = _static_parameter_opportunities(
        getattr(pipe, "task", None)
    )
    intermediate_traffic = _intermediate_traffic_context(stack_snapshot, breakdown)
    primfunc_profile = _primfunc_profile(breakdown)
    digest = {"schema_version": "leka.profile_digest.v2",
              "profile_breakdown": breakdown + deep_rows,
              "primfunc_profile": primfunc_profile,
              "counters": counters,
              "runtime_context": runtime_context,
              "static_opportunities": static_opportunities,
              "intermediate_traffic": intermediate_traffic,
              "collector": collector_status,
              "collector_diagnostics": collector_diagnostics,
              "profile_status": profile_status,
              "tree_node_id": tree_node_id,
              "binding": digest_binding}
    # Counter-derived optimization claims are promotable evidence only when
    # the collector proved that its samples belong to this exact stack.  RK
    # devfreq/HWCPipe windows and Vivo's long-lived RPC process remain useful
    # runtime context, but must not silently become per-candidate diagnoses.
    counter_hyps = (
        attribution.counter_hypotheses(backend, counters, rows=deep_rows,
                                      profile_row_offset=len(breakdown))
        if collector_status.get("status") == "collected"
        and collector_status.get("exact_bundle") is True
        else []
    )
    hyps = (attribution.breakdown_hypotheses(backend, breakdown,
                                             measured.public.get("latency_ms"))
            + counter_hyps
            + attribution.context_hypotheses(
                backend,
                runtime_context=runtime_context,
                static_opportunities=static_opportunities,
                intermediate_traffic=intermediate_traffic,
            ))
    digest["initial_hypotheses"] = hyps
    digest["evidence_quality"] = audit_digest(
        backend, digest, latency_ms=None if measurement_failed else measured.public.get("latency_ms"))
    # Preserve the full local report outside every model-readable workspace.
    # Seal the ID only after projection: schema examples and exact-row evidence
    # validation must refer to the same de-identified rows the model can read.
    # Production Pipeline always has run_dir; small offline doubles use a sibling
    # private root, never a descendant of their synthetic model workspace.
    evaluator_root = Path(getattr(pipe, "run_dir", ws.parent)).resolve()
    digest = archive_and_project(
        digest,
        archive_root=evaluator_root / "private" / "hardware_model_digests",
        workspace=ws,
        private_values=tuple(BACKENDS[backend].get(key) for key in
                             ("vivo_serial", "ssh_alias", "ssh_host", "rpc_host")),
    )
    if capture_state is not None:
        capture_state.bind_digest(fresh_attempt_id, capture_seq, digest["digest_id"])
    hyps = digest["initial_hypotheses"]
    _write_read_only(ws / "PROFILE_DIGEST.json", json.dumps(digest, indent=2) + "\n")
    _write_read_only(ws / "HYPOTHESES.json", json.dumps(hyps, indent=2) + "\n")
    _write_read_only(ws / "EVIDENCE_SCHEMA.json",
                     json.dumps(evidence.output_contract(backend, trusted_digest=digest), indent=2) + "\n")
    _project_hardware_stack(ws, stack_snapshot, expected_names=stack_view_names,
                            binding=digest_binding, digest_id=digest["digest_id"])
    evidence_path = ws / "EVIDENCE.json"
    if evidence_path.exists() or evidence_path.is_symlink():
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise RuntimeError("hardware evidence path must be a regular file")
        evidence_path.unlink()
    _project_hardware_quantity_contract(pipe, ws, digest, backend)
    if measurement_failed:
        pipe.log("hw_turn_aborted_measurement_failed", profile_status=profile_status,
                 binding=digest_binding, tree_node_id=tree_node_id,
                 action="saved_failed_digest_without_calling_hardware_model")
        return []
    observations = getattr(pipe, "_hardware_observations", None)
    if observations is not None:
        try:
            observations.publish_projected_digest(digest)
        except Exception as err:
            # Optional factual delivery cannot suppress the existing HW model.
            pipe.log("hardware_observation_publication_failed", error=str(err)[:300])
    if time.monotonic() >= deadline:
        pipe.log("hw_turn_budget_exhausted", action="skip_provider_after_collection")
        return []
    session._leka_pending_digest = digest
    shared_advice = peek_shared_advice(pipe, shared_pool, hw_agent, digest_binding)
    shared_message_ids = tuple(
        message["message_id"] for message in shared_advice
    )
    turn_id = prepare_turn_context(
        pipe,
        shared_pool,
        workspace=ws,
        agent=hw_agent,
        backend=backend,
        binding=digest_binding,
        shared_message_ids=shared_message_ids,
    )
    session._leka_pending_shared_message_ids = shared_message_ids
    session._leka_pending_hw_turn_id = turn_id
    advice_text = "".join(
        "\n\n[shared_advice] " + json.dumps(message, ensure_ascii=False)
        for message in shared_advice
    )
    try:
        _early_hardware_model_checkpoint(pipe, session)
        pipe.log("hardware_model_request_admitted", request_kind="fresh_digest",
                 digest_id=digest.get("digest_id"), correction_attempt=0,
                 turn_id=turn_id,
                 meaning="resident_user_prompt_admission; not_provider_http_count_or_bill")
        result = session.send(
            "A fresh trusted PROFILE_DIGEST.json and HYPOTHESES.json are in your workspace "
            "(stack views under stack_view/). Confirm or refute the hypotheses against the "
            "digest and write EVIDENCE.json (an array of packets per your profile). "
            "Rows must be verbatim subsets of the digest."
            " Read QUANTITY_CONTRACT.json for this digest's selected quantities, units and missing comparisons; "
            "its source_digest_id must match PROFILE_DIGEST.json. It is an explanation, not extra evidence. "
            "Read EVIDENCE_SCHEMA.json for exact categories and allowed target layers. "
            "If EVIDENCE_FEEDBACK.json exists, learn from its previous validation errors; "
            "it refers to the previous output, not evidence for this fresh digest."
            + advice_text,
            timeout_s=max(0.0, deadline - time.monotonic()),
        )
    except Exception:
        delattr(session, "_leka_pending_digest")
        raise
    if result is None:
        return []
    provider_status = terminal_provider_error(result)
    if provider_status is not None:
        delattr(session, "_leka_pending_digest")
        raise TerminalProviderError(
            f"hardware agent provider returned HTTP {provider_status}"
        )
    if not provider_turn_succeeded(result):
        pipe.log(
            "hw_turn_provider_error",
            agent=hw_agent,
            resumed_pending=False,
            action="leave_peer_advice_unacknowledged",
            cost_usd=result.get("total_cost_usd"),
        )
        delattr(session, "_leka_pending_digest")
        delattr(session, "_leka_pending_shared_message_ids")
        delattr(session, "_leka_pending_hw_turn_id")
        return []
    forward_help_reports(pipe, hw_agent, ws, backend)
    delivered, repair_pending = _consume_and_repair_hw_evidence(
        pipe, session, ws, backend, digest, deadline)
    if repair_pending:
        return delivered
    operator_tree = getattr(pipe, "operator_tree", None)
    if operator_tree is not None:
        with pipe._lock:
            operator_tree.record_hardware_observation(
                digest=digest,
                hypotheses=hyps,
                evidence_packets=delivered,
            )
    forward_shared_advice(
        pipe,
        shared_pool,
        agent=hw_agent,
        workspace=ws,
        backend=backend,
        binding=digest["binding"],
        turn_id=turn_id,
    )
    acknowledge_shared_advice(
        pipe,
        shared_pool,
        hw_agent,
        shared_message_ids,
        turn_id=turn_id,
        binding=digest["binding"],
    )
    delattr(session, "_leka_pending_digest")
    delattr(session, "_leka_pending_shared_message_ids")
    delattr(session, "_leka_pending_hw_turn_id")
    pipe.log(
        "hw_turn",
        resumed_pending=False,
        cost_usd=result.get("total_cost_usd"),
        delivered=len(delivered),
    )
    return delivered


def _archive_agent_protocol(path: Path, disposition: str) -> None:
    path.rename(path.with_name(f"{path.stem}.{disposition}.{time.time_ns()}.json"))


def _read_agent_protocol(path: Path) -> dict:
    """Read one small direct regular JSON object from an agent workspace."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("protocol must be a direct regular file")
        if metadata.st_size > MAX_AGENT_PROTOCOL_BYTES:
            raise ValueError("protocol exceeds 16 KiB")
        raw = os.read(descriptor, MAX_AGENT_PROTOCOL_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_AGENT_PROTOCOL_BYTES:
        raise ValueError("protocol exceeds 16 KiB")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"protocol must be valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("protocol JSON must be an object")
    return payload


def forward_help_reports(
    pipe: Pipeline, agent: str, ws: Path, backend: str
) -> list[str]:
    """Acknowledge a legal non-layer HELP/infra report without restarting work.

    Candidate-code failures stay in the structured measure response and are
    repaired by their owning layer.  HELP is only for evaluator-owned device,
    transport, profiler, or timing infrastructure; the driver records the
    report and returns a sealed acknowledgement rather than pretending an
    agent-authored code edit can repair infrastructure.
    """

    path = ws / "HELP.json"
    if not path.exists() and not path.is_symlink():
        return []
    try:
        payload = _read_agent_protocol(path)
        category = str(payload.get("category", ""))
        if category not in {
            "infra",
            "measurement_transport",
            "profiler_collection",
            "timing_environment",
        }:
            raise ValueError("HELP category must describe evaluator infrastructure")
        response = {
            "schema_version": "leka.help_response.v1",
            "status": "recorded",
            "owning_layer": "infra",
            "backend": backend,
            "reporter": agent,
            "category": category,
            "operation": diagnostics.sanitize_report_text(
                payload.get("operation"), max_chars=120
            ),
            "problem": diagnostics.sanitize_report_text(
                payload.get("problem"), max_chars=1200
            ),
            "evidence": diagnostics.sanitize_report_text(
                payload.get("evidence"), max_chars=800
            ),
            "requested_action": diagnostics.sanitize_report_text(
                payload.get("requested_action"), max_chars=300
            ),
            "driver_action": (
                "report recorded; no automatic restart or device mutation; "
                "continue independent work while the trusted driver owns diagnosis"
            ),
        }
        _write_read_only(
            ws / "HELP_RESPONSE.json",
            json.dumps(response, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        if agent in {"relax_agent", "tir_agent", BACKENDS[backend]["l2_agent"]}:
            pipe.mailbox.send(agent, "help_response", response)
        pipe.log(
            "help_reported",
            agent=agent,
            backend=backend,
            category=category,
            operation=response["operation"],
        )
        _archive_agent_protocol(path, "sent")
        return [f"HELP/{category} recorded"]
    except (OSError, ValueError) as error:
        pipe.log("help_invalid", agent=agent, error=str(error)[:200])
        if path.exists() or path.is_symlink():
            _archive_agent_protocol(path, "rejected")
        return ["HELP rejected"]


def forward_escalations(pipe: Pipeline, agent: str, ws: Path, backend: str,
                       *, binding: dict | None = None, turn_id: str = "") -> list[str]:
    """Deliver a workspace ESCALATE.json to the target layer's inbox, then
    archive it (the file is the protocol promised in every profile)."""
    path = ws / "ESCALATE.json"
    if not path.exists() and not path.is_symlink():
        return []
    try:
        payload = _read_agent_protocol(path)
    except (OSError, ValueError) as err:
        pipe.log("escalate_invalid", agent=agent, error=str(err)[:200])
        if path.exists() or path.is_symlink():
            _archive_agent_protocol(path, "rejected")
        return [f"escalate rejected: invalid json"]
    target = str(payload.get("target_layer", ""))
    target_agent = {"L0": "relax_agent", "L1": "tir_agent",
                    "L2": BACKENDS[backend]["l2_agent"]}.get(target)
    if not target_agent or target_agent == agent:
        pipe.log("escalate_invalid", agent=agent, target=target)
        _archive_agent_protocol(path, "rejected")
        return [f"escalate rejected: bad target_layer {target!r}"]
    try:
        prose = {field: _peer_report_text(payload.get(field), field, maximum)
                 for field, maximum in (("problem", 1200), ("evidence", 800), ("suggestion", 600))}
    except ValueError as error:
        _report_peer_protocol_rejection(pipe, agent, "ESCALATE.json", error)
        _archive_agent_protocol(path, "rejected")
        return ["escalate rejected: invalid peer text"]
    pipe.mailbox.send(target_agent, "escalate", {
        "from_agent": agent,
        **prose,
        "binding": binding if binding is not None else pipe.stack.binding().to_dict(),
        "producer_turn_id": turn_id or None,
        "authority": "untrusted_non_binding_advice",
        "instruction": "Revalidate this peer hypothesis against your current layer inputs.",
    })
    pipe.log("escalate_forwarded", from_agent=agent, target=target)
    _archive_agent_protocol(path, "sent")
    return [f"escalate forwarded to {target}"]


def peek_shared_advice(
    pipe: Pipeline, pool: SharedMessagePool | None, agent: str, binding: dict | None = None
) -> list[dict]:
    """Peek newly routed peer advice without claiming delivery yet.

    A corrupt advisory log is recorded but never allowed to stop optimization,
    measurement, or finalization.  Observation is called only immediately before
    a *new* model turn; callers deliberately skip it while a prior turn is pending.
    """

    if pool is None:
        return []
    try:
        messages = pool.peek(agent, binding=binding if binding is not None else (
            pipe.stack.binding().to_dict() if hasattr(pipe, "stack") else None))
    except (OSError, SharedMessagePoolError) as error:
        pipe.log(
            "shared_advice_observe_error",
            agent=agent,
            error=str(error)[:300],
        )
        return []
    if messages:
        pipe.log(
            "shared_advice_peeked",
            agent=agent,
            message_ids=[message["message_id"] for message in messages],
        )
    return messages


def acknowledge_shared_advice(
    pipe: Pipeline,
    pool: SharedMessagePool | None,
    agent: str,
    message_ids: tuple[str, ...],
    *,
    turn_id: str,
    binding: dict,
) -> None:
    """Acknowledge peer inputs only after the turn's formal result arrives."""

    if pool is None or not message_ids:
        return
    try:
        acknowledged = pool.acknowledge(
            agent, message_ids, turn_id=turn_id, binding=binding
        )
    except (OSError, SharedMessagePoolError) as error:
        # Keep the messages unread; a later new turn may safely receive them
        # again.  At-least-once advice is preferable to silent loss.
        pipe.log(
            "shared_advice_ack_error",
            agent=agent,
            message_ids=list(message_ids),
            turn_id=turn_id,
            binding=binding,
            error=str(error)[:300],
        )
        return
    pipe.log(
        "shared_advice_acknowledged",
        agent=agent,
        message_ids=acknowledged,
        turn_id=turn_id,
        binding=binding,
    )


_DRIVER_TURN_ID = re.compile(r"[0-9a-f]{32}\Z")
_TURN_CONTEXT_SCHEMA = "leka.turn_context.v1"
MAX_DIALOGUE_SLOTS = 4
MAX_TOOL_ADVISORY_BYTES = 8 * 1024
MAX_TOOL_ADVISORY_MESSAGES = 3


class LivePeerDialogue:
    """Cooperative peer transport during resident model/tool iterations.

    Peers own the content and directed request/reply graph. This transport only
    validates and routes bounded JSON and publishes read-only inbox snapshots. It
    never injects a second CLI user turn, calls a device, edits an IR, promotes
    a candidate, or counts a file projection as proof that the model read it.
    """

    def __init__(self, pipe: Pipeline, pool: SharedMessagePool,
                 workspaces: dict[str, Path], backend: str) -> None:
        self.pipe, self.pool = pipe, pool
        self.workspaces, self.backend = workspaces, backend
        self._lock = threading.RLock()
        self._contexts: dict[str, dict] = {}
        self._processed: dict[tuple[str, str, int], dict] = {}
        self._projected: dict[str, str] = {}
        self._last_poll = 0.0
        self._errors: set[tuple[str, str]] = set()
        self._tool_delivered: dict[str, set[str]] = {}

    def begin(self, agent: str, context: dict) -> None:
        with self._lock:
            self._contexts[agent] = context
            self._projected.pop(agent, None)
            self._processed = {key: value for key, value in self._processed.items()
                               if key[0] != agent}
        self.poll(force=True)

    def _error(self, agent: str, error: Exception) -> None:
        key = (agent, str(error)[:300])
        if key not in self._errors:
            self._errors.add(key)
            self.pipe.log("peer_dialogue_transport_error", agent=agent, error=key[1])

    def _consume(self, agent: str, context: dict) -> None:
        turn_id = context["turn_id"]
        for slot in range(1, MAX_DIALOGUE_SLOTS + 1):
            key = (agent, turn_id, slot)
            if key in self._processed:
                continue
            path = self.workspaces[agent] / f"DIALOGUE.{turn_id}.{slot}.json"
            if not path.exists() and not path.is_symlink():
                continue
            try:
                # A model Write may still be in progress. Incomplete JSON is
                # retried later, never renamed underneath the writer.
                payload = _read_agent_protocol(path)
            except (OSError, ValueError) as error:
                self._error(agent, error)
                continue
            try:
                if payload.get("turn_id") != turn_id:
                    raise ValueError("live dialogue requires the exact frozen turn_id")
                normalized = _normalized_advice_payload(
                    payload, agent=agent, backend=self.backend, turn_id=turn_id)
                intent = payload.get("intent", "observation")
                if not isinstance(intent, str):
                    raise ValueError("dialogue intent must be text")
                if intent != "observation" and normalized["recipients"] == BROADCAST:
                    raise ValueError("requests/replies/handoffs require explicit owning-layer targets")
                if intent in {"request", "handoff"} and not normalized["suggestion"]:
                    raise ValueError("requests/handoffs require a concrete suggested experiment")
                event = self.pool.publish(
                    sender=agent, producer_turn_id=f"{turn_id}.dialogue.{slot}",
                    binding=context["binding"], dialogue_intent=intent, **normalized)
                self._processed[key] = {"slot": slot, "status": "published",
                                        "message_id": event["id"]}
                self.pipe.log("peer_dialogue_published", agent=agent, turn_id=turn_id,
                              message_id=event["id"], reply_to=event["reply_to"],
                              recipients=event["recipients"], binding=event["binding"],
                              intent=intent, event_sha256=event["event_sha256"])
            except (OSError, ValueError, SharedMessagePoolError) as error:
                self._processed[key] = {"slot": slot, "status": "rejected",
                                        "reason": str(error)[:300],
                                        **(error.details if isinstance(error, PeerTextLimitError) else {})}
                self._error(agent, error)

    def _project(self, agent: str, context: dict, current: dict | None) -> None:
        messages = self.pool.peek(agent, limit=12, binding=current)
        for message in messages:
            message["recipient_turn_binding_status"] = binding_status(
                message["binding"], context["binding"])
        snapshot = {
            "schema_version": "leka.peer_inbox.v1",
            "agent": agent, "turn_id": context["turn_id"],
            "turn_binding": context["binding"], "current_stack_binding": current,
            "authority": "untrusted_non_binding_advice",
            "delivery_semantics": "projected_only_not_proof_of_read_or_adoption",
            "messages": messages,
            "outbox_status": [value for key, value in sorted(self._processed.items())
                              if key[:2] == (agent, context["turn_id"])],
        }
        text = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
        digest = hashlib.sha256(text.encode()).hexdigest()
        if self._projected.get(agent) == digest:
            return
        path = self.workspaces[agent] / "PEER_INBOX.json"
        temporary = path.with_name(f".peer-inbox.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(text)
            temporary.chmod(0o444)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._projected[agent] = digest
        self.pipe.log("peer_dialogue_projected", agent=agent, turn_id=context["turn_id"],
                      message_ids=[item["message_id"] for item in messages],
                      meaning="file_projection_only_not_read_receipt")

    def poll(self, *, force: bool = False) -> None:
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_poll < 0.25:
                return
            self._last_poll = now
            # Never delay a device measurement or invert Pipeline/dialogue
            # lock order. During measurement, freshness is explicitly unknown.
            current = None
            if self.pipe._lock.acquire(blocking=False):
                try:
                    current = self.pipe.stack.binding().to_dict()
                finally:
                    self.pipe._lock.release()
            for agent, context in self._contexts.items():
                try:
                    self._consume(agent, context)
                except (OSError, ValueError, SharedMessagePoolError) as error:
                    self._error(agent, error)
            # Route all new messages before projecting any peer, so a directed
            # reply can be seen at that peer's next tool step in the same turn.
            for agent, context in self._contexts.items():
                try:
                    self._project(agent, context, current)
                except (OSError, ValueError, SharedMessagePoolError) as error:
                    self._error(agent, error)

    def finish(self, agent: str) -> None:
        self.poll(force=True)
        with self._lock:
            self._contexts.pop(agent, None)

    def measure_response_advisory(self, agent: str, *, request_id: str,
                                  tier: str, result_binding: object) -> dict | None:
        """Prepare a bounded, non-consuming advisory at a tool return boundary.

        Called after measurement has released the Pipeline lock. A response
        write is a transport observation only; it cannot acknowledge pool
        history or establish a model read, adoption, or measurement authority.
        """
        self.poll(force=True)
        with self._lock:
            context = self._contexts.get(agent)
            if context is None:
                return None
            current = None
            if self.pipe._lock.acquire(blocking=False):
                try:
                    current = self.pipe.stack.binding().to_dict()
                finally:
                    self.pipe._lock.release()
            delivered = self._tool_delivered.get(agent, set())
            # Formal-prompt receipts must not suppress tool-boundary delivery.
            # This channel has its own cursor, and never mutates those receipts.
            events = [event for event in self.pool.history()
                      if agent in event["recipients"] and event["id"] not in delivered]
            events.sort(key=lambda e: (0 if (e.get("dialogue") or {}).get("intent")
                                      in {"request", "reply", "handoff"} else 1,
                                      e["sequence"]))
            if not events:
                return None
            envelope = {
                "schema_version": "leka.peer_tool_advisory.v1",
                "agent": agent, "turn_id": context["turn_id"],
                "request_id": request_id, "measure_tier": tier,
                "recipient_turn_binding": context["binding"],
                "measurement_result_binding": (result_binding
                                               if isinstance(result_binding, dict) else None),
                "current_stack_binding_at_projection": current,
                "authority": "untrusted_non_binding_advice",
                "delivery_semantics": "tool_response_delivery_not_read_or_adoption",
                "instruction": "Optional peer hypotheses. Bindings may differ from this measurement. "
                               "Revalidate claims; reply via your current dialogue slots if useful. "
                               "PEER_INBOX.json retains full pending messages; do not wait for replies.",
                "messages": [], "omitted_message_count": len(events),
            }
            for event in events:
                if len(envelope["messages"]) >= MAX_TOOL_ADVISORY_MESSAGES:
                    break
                # Preserve complete content whenever it fits. A large message
                # gets explicitly incomplete excerpts with intact trust markers
                # and source binding, never a silent cut through JSON or hashes.
                limits = (None, 480, 240, 120, 60, 0) if not envelope["messages"] else (None,)
                fitted = False
                for limit in limits:
                    clipped = {key: value if limit is None else value[:limit]
                               for key, value in event["content"].items()}
                    message = advisory_projection({**event, "content": clipped}, binding=current)
                    message["recipient_turn_binding_status"] = binding_status(
                        event["binding"], context["binding"])
                    message["source_event_sha256"] = event["event_sha256"]
                    message["content_complete"] = clipped == event["content"]
                    message["truncated_fields"] = [key for key in clipped
                                                   if clipped[key] != event["content"][key]]
                    envelope["messages"].append(message)
                    envelope["omitted_message_count"] = len(events) - len(envelope["messages"])
                    # Match the response writer's actual ASCII/indent encoding,
                    # including nesting; non-ASCII text must not evade the cap.
                    size = len(json.dumps({"peer_advisories": envelope},
                                          allow_nan=False, indent=2).encode("utf-8"))
                    if size <= MAX_TOOL_ADVISORY_BYTES:
                        fitted = True
                        break
                    envelope["messages"].pop()
                    envelope["omitted_message_count"] = len(events) - len(envelope["messages"])
                if not fitted:
                    break
            return envelope if envelope["messages"] else None

    def record_tool_response_delivery(self, agent: str, advisory: dict) -> None:
        """Commit only after the atomic response-file write succeeded."""
        with self._lock:
            identifiers = [message["message_id"] for message in advisory["messages"]]
            encoded = json.dumps(advisory, ensure_ascii=False, allow_nan=False,
                                 sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.pipe.log("tool_response_delivery", agent=agent,
                          turn_id=advisory["turn_id"], request_id=advisory["request_id"],
                          message_ids=identifiers, advisory_sha256=hashlib.sha256(encoded).hexdigest(),
                          meaning="response_file_written_not_model_read_or_adoption")
            self._tool_delivered.setdefault(agent, set()).update(identifiers)


def _validated_turn_context(
    workspace: Path, *, agent: str, backend: str
) -> dict | None:
    """Read the prior immutable context used to recover an interrupted claim."""

    path = workspace / "TURN_CONTEXT.json"
    if not path.exists() and not path.is_symlink():
        return None
    payload = _read_agent_protocol(path)
    if payload.get("schema_version") != _TURN_CONTEXT_SCHEMA:
        raise ValueError("TURN_CONTEXT schema differs from the driver contract")
    if payload.get("agent") != agent or payload.get("backend") != backend:
        raise ValueError("TURN_CONTEXT agent/backend identity differs")
    turn_id = payload.get("turn_id")
    if not isinstance(turn_id, str) or _DRIVER_TURN_ID.fullmatch(turn_id) is None:
        raise ValueError("TURN_CONTEXT turn_id is invalid")
    binding = payload.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("TURN_CONTEXT binding must be an object")
    epoch = binding.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("TURN_CONTEXT binding epoch is invalid")
    return payload


class PeerTextLimitError(ValueError):
    """Machine-readable whole-message rejection, never partial factual prose."""

    def __init__(self, field: str, maximum: int, observed: int, phase: str):
        self.details = {"error_code": "peer_text_limit", "field": field,
                        "max_characters": maximum, "observed_characters": observed,
                        "length_phase": phase, "character_unit": "Unicode_code_points",
                        "truncated": False}
        super().__init__(f"{field} exceeds {maximum} Unicode characters: observed {observed} "
                         f"({phase}); whole message rejected, no text truncated; "
                         "shorten the complete claim and use the next unused dialogue slot")


def _peer_report_text(value: object, field: str, maximum: int = MAX_PEER_TEXT_CHARS) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    if len(value) > maximum:
        raise PeerTextLimitError(field, maximum, len(value), "before_redaction")
    # sanitize_report_text truncates by design for diagnostics. Peer prose
    # needs complete claims/caveats, so use the shared scrubber without slicing.
    clean = diagnostics.scrub_report_text(value)
    clean = " ".join(clean.replace("\x00", "").split())
    if len(clean) > maximum:
        raise PeerTextLimitError(field, maximum, len(clean), "after_redaction")
    return clean


def _report_peer_protocol_rejection(pipe, agent, protocol, error):
    feedback = {"protocol": protocol, "status": "rejected", "reason": str(error)[:300],
                **(error.details if isinstance(error, PeerTextLimitError) else {})}
    pipe.log("peer_protocol_rejected", agent=agent, **feedback)
    mailbox = getattr(pipe, "mailbox", None)
    if mailbox is not None:
        # Legacy protocols complete at the formal turn boundary. Preserve the
        # reason for the existing next turn; never inject a new model turn.
        try:
            mailbox.send(agent, "peer_protocol_rejected", feedback)
        except OSError as failure:
            pipe.log("peer_rejection_feedback_error", agent=agent, protocol=protocol,
                     error=type(failure).__name__)


def _normalized_advice_payload(
    payload: dict, *, agent: str, backend: str, turn_id: str
) -> dict:
    """Validate the agent file and resolve logical roles without side effects."""

    # The path is atomically emptied before the turn and consumed only after
    # that turn's formal result, so the driver owns the binding. ``CURRENT``
    # avoids asking a resident model to remember/copy a random id across turns;
    # exact ids remain supported for recovery and old clients. Any other id is
    # still rejected, preserving the orphan-output guard.
    if payload.get("turn_id") not in {turn_id, "CURRENT"}:
        raise ValueError("ADVISE turn_id differs from immutable TURN_CONTEXT.json")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not 1 <= len(raw_targets) <= 4:
        raise ValueError("targets must be a non-empty list of at most four roles")
    if any(not isinstance(target, str) for target in raw_targets):
        raise ValueError("every advice target must be text")
    targets = list(dict.fromkeys(raw_targets))
    # Treat a redundant explicit list plus ``all`` as a broadcast. Models
    # commonly emit both while intending the superset; rejecting the whole
    # advisory loses useful, non-authoritative coordination for no safety gain.
    if "all" in targets:
        targets = ["all"]
    role_map = {
        "L0": "relax_agent",
        "L1": "tir_agent",
        "L2": BACKENDS[backend]["l2_agent"],
        "HW": BACKENDS[backend]["hw_sense"],
    }
    if targets == ["all"]:
        recipients: str | tuple[str, ...] = BROADCAST
    else:
        unknown = [target for target in targets if target not in role_map]
        if unknown:
            raise ValueError(f"unknown advice target: {unknown[0]}")
        recipients = tuple(
            recipient
            for recipient in dict.fromkeys(role_map[target] for target in targets)
            if recipient != agent
        )
        if not recipients:
            raise ValueError("advice must target at least one other role")
    reply_to = payload.get("reply_to")
    if reply_to is not None and not isinstance(reply_to, str):
        raise ValueError("reply_to must be a message id or null")
    return {
        "recipients": recipients,
        "topic": str(payload.get("topic", "")),
        **{field: _peer_report_text(payload.get(field), field)
           for field in ("summary", "evidence", "suggestion")},
        "reply_to": reply_to,
    }


def _publish_claimed_advice(
    pipe: Pipeline,
    pool: SharedMessagePool,
    *,
    agent: str,
    claimed_path: Path,
    backend: str,
    binding: dict,
    turn_id: str,
    payload: dict | None = None,
) -> dict:
    """Idempotently commit and archive one already-claimed protocol file."""

    payload = payload if payload is not None else _read_agent_protocol(claimed_path)
    normalized = _normalized_advice_payload(
        payload, agent=agent, backend=backend, turn_id=turn_id
    )
    event = pool.publish(
        sender=agent,
        binding=binding,
        producer_turn_id=turn_id,
        **normalized,
    )
    try:
        pipe.log(
            "shared_advice_published",
            agent=agent,
            producer_turn_id=turn_id,
            message_id=event["id"],
            event_sha256=event["event_sha256"],
            recipients=event["recipients"],
            topic=event["topic"],
            binding=event["binding"],
        )
    except OperationCancelled:
        raise
    except Exception:
        # The shared event itself is the durable audit authority.
        pass
    try:
        claimed_path.rename(
            claimed_path.with_name(f"ADVISE.sent.{time.time_ns()}.json")
        )
    except OSError as error:
        # The producer turn id is the durable idempotency key.  A later bounded
        # reconciliation can safely retry without appending another event.
        try:
            pipe.log(
                "shared_advice_archive_error",
                agent=agent,
                producer_turn_id=turn_id,
                message_id=event["id"],
                error=str(error)[:300],
            )
        except OperationCancelled:
            raise
        except Exception:
            pass
    return event


def reconcile_claimed_advice(
    pipe: Pipeline,
    pool: SharedMessagePool | None,
    *,
    workspace: Path,
    agent: str,
    backend: str,
) -> None:
    """Recover at most the one claim named by the prior turn context.

    No directory scan is needed: a driver turn has one random id and one legal
    claim pathname.  If recovery cannot commit, keep both files intact and do
    not overwrite the context; the next process can retry without message loss.
    """

    if pool is None:
        return
    try:
        context = _validated_turn_context(
            workspace, agent=agent, backend=backend
        )
    except (OSError, ValueError) as error:
        pipe.log(
            "shared_advice_reconcile_context_error",
            agent=agent,
            error=str(error)[:300],
        )
        raise RuntimeError("cannot safely reconcile prior shared-advice turn") from error
    if context is None:
        return
    turn_id = context["turn_id"]
    claimed_path = workspace / f"ADVISE.claimed.{turn_id}.json"
    if not claimed_path.exists() and not claimed_path.is_symlink():
        return
    try:
        event = _publish_claimed_advice(
            pipe,
            pool,
            agent=agent,
            claimed_path=claimed_path,
            backend=backend,
            binding=context["binding"],
            turn_id=turn_id,
        )
    except (OSError, ValueError, SharedMessagePoolError) as error:
        pipe.log(
            "shared_advice_reconcile_pending",
            agent=agent,
            producer_turn_id=turn_id,
            error=str(error)[:300],
        )
        raise RuntimeError("shared-advice claim remains pending recovery") from error
    pipe.log(
        "shared_advice_reconciled",
        agent=agent,
        producer_turn_id=turn_id,
        message_id=event["id"],
    )


def source_review_policy(pipe: Pipeline) -> dict:
    """Describe evaluator configuration, never a review verdict or authority.

    Only the two shipped reviewer implementations have known semantics.
    Custom implementations/subclasses are unknown, even if they claim a mode.
    This projection never invokes a reviewer or changes promotion checks.
    """
    layers = ("L0", "L1", "L2")
    with getattr(pipe, "_lock", nullcontext()):
        missing = object()
        reviewer = getattr(pipe, "reviewer", missing)
        selected = getattr(pipe, "review_layers", frozenset(layers))
        selection_known = (type(selected) in {set, frozenset, tuple, list}
                           and all(type(item) is str and item in layers for item in selected))
        if reviewer is None or type(reviewer) is DisabledReviewer:
            kind = "none" if reviewer is None else "disabled"
            statuses = dict.fromkeys(layers, "disabled")
        else:
            kind = "codex" if type(reviewer) is CodexReviewer else "unknown"
            statuses = {
                layer: ("unknown" if not selection_known else
                        "disabled" if layer not in selected else
                        "configured" if kind == "codex" else "unknown")
                for layer in layers
            }
    return {
        "schema_version": "leka.source_review_policy.v1",
        "authority": "evaluator_configuration_only",
        "reviewer_kind": kind,
        "layer_status": statuses,
        "meaning": (
            "configured means source review is configured to be attempted at promotion; "
            "not evidence that it ran, completed or approved. disabled means no source "
            "review for that layer. unknown grants no assumption of review. "
            "Correctness and promotion gates still apply; this field grants no authority."
        ),
    }


def prepare_turn_context(
    pipe: Pipeline,
    pool: SharedMessagePool | None,
    *,
    workspace: Path,
    agent: str,
    backend: str,
    binding: dict,
    shared_message_ids: tuple[str, ...],
) -> str:
    """Retire orphan output and bind new agent output to one exact turn."""

    reconcile_claimed_advice(
        pipe, pool, workspace=workspace, agent=agent, backend=backend
    )
    advice_path = workspace / "ADVISE.json"
    if advice_path.exists() or advice_path.is_symlink():
        try:
            metadata = advice_path.lstat()
            if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
                raise ValueError("orphan ADVISE.json is not a regular file")
            orphan = workspace / f"ADVISE.orphan.{time.time_ns()}.json"
            advice_path.rename(orphan)
            pipe.log("shared_advice_orphaned", agent=agent, path=orphan.name)
        except (OSError, ValueError) as error:
            pipe.log(
                "shared_advice_orphan_error", agent=agent, error=str(error)[:300]
            )
    turn_id = uuid.uuid4().hex
    context = {
        "schema_version": _TURN_CONTEXT_SCHEMA,
        "turn_id": turn_id,
        "agent": agent,
        "backend": backend,
        "binding": binding,
        "source_review": source_review_policy(pipe),
        "shared_message_ids": list(shared_message_ids),
        "advice_authority": "untrusted_non_binding_advice",
        "dialogue": {
            "inbox": "PEER_INBOX.json", "max_messages_this_turn": MAX_DIALOGUE_SLOTS,
            "outbox_slots": [f"DIALOGUE.{turn_id}.{slot}.json"
                             for slot in range(1, MAX_DIALOGUE_SLOTS + 1)],
            "rule": "write each slot once; use the exact turn_id; never wait for a reply",
            "available_roles": {
                role: name for role, name in {
                    "L0": "relax_agent", "L1": "tir_agent",
                    "L2": BACKENDS[backend]["l2_agent"],
                    "HW": BACKENDS[backend]["hw_sense"],
                }.items() if pool is not None and name in pool.participants
            },
        },
    }
    _write_read_only(
        workspace / "TURN_CONTEXT.json",
        json.dumps(context, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
    )
    bridge = getattr(pipe, "_peer_dialogue", None)
    if bridge is not None:
        bridge.begin(agent, context)
    delivery = getattr(pipe, "_hardware_delivery", None)
    if delivery is not None:
        delivery.begin(agent, context)
    observations = getattr(pipe, "_hardware_observations", None)
    if observations is not None:
        observations.begin(agent, context)
    return turn_id


def forward_shared_advice(
    pipe: Pipeline,
    pool: SharedMessagePool | None,
    *,
    agent: str,
    workspace: Path,
    backend: str,
    binding: dict,
    turn_id: str,
) -> list[str]:
    """Validate and publish one optional ``ADVISE.json`` peer message.

    Logical targets keep profiles backend-independent.  The trusted driver maps
    them to the four concrete resident roles and supplies the frozen stack
    binding; agents cannot forge either the backend or epoch.
    """

    bridge = getattr(pipe, "_peer_dialogue", None)
    if bridge is not None:
        bridge.finish(agent)
    delivery = getattr(pipe, "_hardware_delivery", None)
    if delivery is not None:
        delivery.finish(agent)
    observations = getattr(pipe, "_hardware_observations", None)
    if observations is not None:
        observations.finish(agent)

    path = workspace / "ADVISE.json"
    if not path.exists() and not path.is_symlink():
        return []
    if pool is None:
        pipe.log("shared_advice_rejected", agent=agent, error="pool unavailable")
        try:
            _archive_agent_protocol(path, "rejected")
        except OSError as error:
            pipe.log(
                "shared_advice_archive_error", agent=agent, error=str(error)[:300]
            )
        return ["shared advice rejected: pool unavailable"]

    claimed_path: Path | None = None
    try:
        payload = _read_agent_protocol(path)
        # Reject malformed output before crossing the atomic claim boundary.
        _normalized_advice_payload(
            payload, agent=agent, backend=backend, turn_id=turn_id
        )
        claimed_path = workspace / f"ADVISE.claimed.{turn_id}.json"
        if claimed_path.exists() or claimed_path.is_symlink():
            raise ValueError("ADVISE turn claim already exists")
        path.rename(claimed_path)
        event = _publish_claimed_advice(
            pipe,
            pool,
            agent=agent,
            claimed_path=claimed_path,
            backend=backend,
            binding=binding,
            turn_id=turn_id,
            payload=payload,
        )
    except (OSError, ValueError, SharedMessagePoolError) as error:
        pipe.log("shared_advice_rejected", agent=agent, error=str(error)[:300])
        if isinstance(error, PeerTextLimitError):
            _report_peer_protocol_rejection(pipe, agent, "ADVISE.json", error)
        rejected_source = (
            claimed_path
            if claimed_path is not None
            and (claimed_path.exists() or claimed_path.is_symlink())
            else path
        )
        if rejected_source.exists() or rejected_source.is_symlink():
            try:
                rejected_source.rename(
                    workspace / f"ADVISE.rejected.{time.time_ns()}.json"
                )
            except OSError as archive_error:
                pipe.log(
                    "shared_advice_archive_error",
                    agent=agent,
                    error=str(archive_error)[:300],
                )
        return ["shared advice rejected"]

    return [f"shared advice published as {event['id']}"]


def forward_tree_branch_request(
    pipe: Pipeline,
    *,
    agent: str,
    workspace: Path,
    backend: str,
    turn_id: str,
    promotion_requested: bool,
    turn_authority: dict | None = None,
) -> list[str]:
    if getattr(pipe, "turn_authority_enabled", False):
        with pipe._lock:
            layer = pipe._turn_authority_roles.get(agent)
            valid, error = pipe.validate_turn_authority(
                agent, layer, turn_authority, turn_id=turn_id)
            if not valid:
                path = workspace / "BRANCH_REQUEST.json"
                if path.exists() or path.is_symlink():
                    _archive_agent_protocol(path, "rejected")
                    pipe.log("operator_tree_request_rejected", agent=agent, error=error,
                             reason="turn_authority")
                    return [f"tree checkout rejected: {error}"]
                return []
            return _forward_tree_branch_request(pipe, agent=agent, workspace=workspace,
                backend=backend, turn_id=turn_id, promotion_requested=promotion_requested)
    return _forward_tree_branch_request(pipe, agent=agent, workspace=workspace,
        backend=backend, turn_id=turn_id, promotion_requested=promotion_requested)


def _forward_tree_branch_request(
    pipe: Pipeline, *, agent: str, workspace: Path, backend: str,
    turn_id: str, promotion_requested: bool,
) -> list[str]:
    """Apply one agent-authored request to explore a stored compatible node."""

    path = workspace / "BRANCH_REQUEST.json"
    if not path.exists() and not path.is_symlink():
        return []
    role_layer = {
        "relax_agent": "L0",
        "tir_agent": "L1",
        BACKENDS[backend]["l2_agent"]: "L2",
    }
    requester_layer = role_layer.get(agent)
    try:
        if requester_layer is None:
            raise ValueError("only optimizer agents may request a tree checkout")
        if promotion_requested:
            raise ValueError(
                "PROMOTE and BRANCH_REQUEST are mutually exclusive in one turn"
            )
        payload = _read_agent_protocol(path)
        if payload.get("turn_id") not in {turn_id, "CURRENT"}:
            raise ValueError(
                "BRANCH_REQUEST turn_id differs from immutable TURN_CONTEXT.json"
            )
        node_id = payload.get("node_id")
        if not isinstance(node_id, str) or re.fullmatch(
            r"(?:root-[0-9a-f]{16}|n[0-9a-f]{8})", node_id
        ) is None:
            raise ValueError("BRANCH_REQUEST node_id is invalid")
        reason = diagnostics.sanitize_report_text(
            payload.get("reason"), max_chars=1200
        )
        out = pipe.activate_tree_node(node_id, requester_layer=requester_layer)
        pipe.log(
            "operator_tree_request_accepted",
            agent=agent,
            requester_layer=requester_layer,
            node=node_id,
            reason=reason,
        )
        _archive_agent_protocol(path, "accepted")
        return [f"tree checkout {node_id}: {out}"]
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        pipe.log(
            "operator_tree_request_rejected",
            agent=agent,
            requester_layer=requester_layer,
            error=str(error)[:300],
        )
        try:
            _archive_agent_protocol(path, "rejected")
        except OSError as archive_error:
            pipe.log(
                "operator_tree_request_archive_error",
                agent=agent,
                error=str(archive_error)[:300],
            )
        return [f"tree checkout rejected: {str(error)[:200]}"]


def apply_agent_reply(pipe: Pipeline, reply_text: str, *, agent: str | None = None,
                      turn_id: str = "", turn_authority: dict | None = None) -> list[str]:
    if getattr(pipe, "turn_authority_enabled", False):
        with pipe._lock:
            valid, error = pipe.validate_turn_authority(
                agent, pipe._turn_authority_roles.get(agent), turn_authority, turn_id=turn_id)
            if not valid:
                if re.search(r"PROMOTE\s+n[0-9a-f]{8}", reply_text or ""):
                    pipe.log("reply_action_rejected", agent=agent, turn_id=turn_id,
                             error=error, reason="turn_authority")
                    return [f"promote rejected: {error}"]
                return ["noop"] if re.search(r"^NOOP\b", reply_text or "", re.MULTILINE) else []
            return _apply_agent_reply(pipe, reply_text)
    return _apply_agent_reply(pipe, reply_text)


def _apply_agent_reply(pipe: Pipeline, reply_text: str) -> list[str]:
    actions = []
    for node_id in re.findall(r"PROMOTE\s+(n[0-9a-f]{8})", reply_text or ""):
        candidate = pipe.candidate_by_id(node_id)
        if candidate is None:
            actions.append(f"promote {node_id}: not found in ledger")
            continue
        out = pipe.promote(candidate)
        actions.append(f"promote {node_id}: {out}")
    if re.search(r"^NOOP\b", reply_text or "", re.MULTILINE):
        actions.append("noop")
    return actions


def require_baseline_latency(result) -> float:
    """Return a valid baseline latency or surface its sanitized diagnostic."""
    latency = result.public.get("latency_ms")
    if (
        result.ok
        and not isinstance(latency, bool)
        and isinstance(latency, (int, float))
        and latency > 0
    ):
        return float(latency)
    diagnostic = result.public.get("diagnostic") or {
        "owning_layer": "infra",
        "lines": ["default measurement returned no positive latency"],
    }
    raise RuntimeError(
        "default baseline measurement failed: "
        + json.dumps(diagnostic, ensure_ascii=False)
    )


def require_final_latency(result) -> float:
    """Fail the stage unless restored global-best is correct and measurable."""

    latency = result.public.get("latency_ms")
    if (
        result.ok is True
        and result.public.get("correct") is True
        and not isinstance(latency, bool)
        and isinstance(latency, (int, float))
        and 0 < latency <= sys.float_info.max
        and math.isfinite(latency)
    ):
        return float(latency)
    diagnostic = result.public.get("diagnostic") or {
        "owning_layer": "infra",
        "lines": ["restored global-best failed correctness or returned no latency"],
    }
    raise RuntimeError(
        "final global-best measurement failed: "
        + json.dumps(diagnostic, ensure_ascii=False)
    )


def finalize_and_report(pipe: Pipeline, *, backend: str, baseline_latency: float,
                        finalize_only: bool = False) -> float:
    """Emit a successful delivery only after final correctness and timing.

    Restoration may still preserve a useful checkpoint when the device is
    lost. That is a failed final delivery, and must propagate to the CLI exit
    status instead of turning a skipped measurement into a successful run.
    """
    try:
        final = pipe.finalize_global_best()
        unavailable, reason = device_unavailable(pipe)
        if unavailable:
            source = final.public.get("final_source", "restored_global_best")
            pipe.log("final_measurement_skipped", backend=backend, reason=reason,
                     source=source)
            raise RuntimeError(
                f"final device measurement unavailable for {backend}: {reason}"
            )
        final_latency = require_final_latency(final)
        checks = final.public.get("requalification_checks")
        if type(checks) is not int or checks != PROMOTION_AUDIT_SEED_COUNT:
            raise RuntimeError("final delivery lacks three completed requalification checks")
        final_source = final.public.get("final_source")
        allowed = {"restored_global_best"} if finalize_only else {
            "restored_global_best", "naive_imported_fallback"}
        if final_source not in allowed:
            raise RuntimeError("final result lacks a trusted delivery source")
        pipe.log("final_default", latency_ms=final_latency, source=final_source, ok=True,
                 snapshot_id=final.public.get("final_snapshot_id"),
                 requalification_checks=checks,
                 **({"provider_invoked": False} if finalize_only else {}))
        print(f"final {final_source} stack T1: {final_latency} ms "
              f"(baseline {baseline_latency} ms), epoch={pipe.stack.epoch}")
        return final_latency
    except Exception as err:
        pipe.log("final_default_error", error=str(err)[:400],
                 **({"provider_invoked": False} if finalize_only else {}))
        print(f"global-best restore/final measurement failed ({err}); "
              f"stack epoch={pipe.stack.epoch}, promoted artifacts and private "
              "checkpoints remain in the run dir")
        raise


def device_unavailable(pipe: Pipeline) -> tuple[bool, str]:
    """Return the SSH runner's terminal device-loss state, when supported."""

    runner = getattr(getattr(pipe, "service", None), "runner", None)
    unavailable = bool(getattr(runner, "device_unavailable", False))
    reason = str(getattr(runner, "device_unavailable_reason", "") or "")
    return unavailable, reason


def record_layer_turn(
    pipe: Pipeline,
    *,
    agent: str,
    workspace: Path,
    backend: str,
    result: dict | None,
    resumed_pending: bool = False,
    shared_pool: SharedMessagePool | None = None,
    turn_binding: dict | None = None,
    turn_id: str = "",
    turn_authority: dict | None = None,
) -> None:
    options = dict(agent=agent, workspace=workspace, backend=backend, result=result,
                   resumed_pending=resumed_pending, shared_pool=shared_pool,
                   turn_binding=turn_binding, turn_id=turn_id, turn_authority=turn_authority)
    if getattr(pipe, "turn_authority_enabled", False):
        with pipe._lock:
            valid, error = pipe.validate_turn_authority(
                agent, pipe._turn_authority_roles.get(agent), turn_authority,
                turn_id=turn_id, require_optimization=False)
            if not valid:
                pipe.log("turn_result_authority_rejected", agent=agent, turn_id=turn_id,
                         error=error, result_applied=False)
                return
            _record_layer_turn(pipe, **options)
    else:
        _record_layer_turn(pipe, **options)


def _record_layer_turn(
    pipe: Pipeline, *, agent: str, workspace: Path, backend: str, result: dict | None,
    resumed_pending: bool, shared_pool: SharedMessagePool | None,
    turn_binding: dict | None, turn_id: str, turn_authority: dict | None,
) -> None:
    """Apply and ledger exactly one provider result delivered to the driver."""

    reply = (result or {}).get("result", "")
    actions = apply_agent_reply(pipe, reply, agent=agent, turn_id=turn_id,
                                turn_authority=turn_authority)
    # A timed-out provider call is still running.  Do not consume any possibly
    # half-written protocol file until the corresponding result is delivered.
    if result is not None:
        actions += forward_escalations(
            pipe, agent, workspace, backend, binding=turn_binding, turn_id=turn_id)
        actions += forward_help_reports(pipe, agent, workspace, backend)
        actions += forward_tree_branch_request(
            pipe,
            agent=agent,
            workspace=workspace,
            backend=backend,
            turn_id=turn_id,
            promotion_requested=any(action.startswith("promote ") for action in actions),
            turn_authority=turn_authority,
        )
        actions += forward_shared_advice(
            pipe,
            shared_pool,
            agent=agent,
            workspace=workspace,
            backend=backend,
            binding=turn_binding or pipe.stack.binding().to_dict(),
            turn_id=turn_id,
        )
    pipe.log(
        "turn",
        agent=agent,
        actions=actions,
        cost_usd=(result or {}).get("total_cost_usd"),
        timed_out=result is None,
        resumed_pending=resumed_pending,
        reply_head=(reply or "")[:400],
    )
    discussion = ((turn_authority or {}).get("payload") or {}).get("purpose") == "discussion"
    if provider_turn_succeeded(result) and not discussion:
        acknowledge_hardware_messages(pipe, agent)


def run_concurrent_turns(
    calls: dict[str, Callable[[], dict | None]],
) -> dict[str, tuple[dict | None, Exception | None]]:
    """Run independent provider turns together; never let one hide another.

    Device access is still serialized by ``MeasureServer``/``Pipeline``.  This
    helper only overlaps model reasoning and workspace editing, which gives
    each layer a real share of the wall-clock budget.
    """

    if not calls:
        return {}
    outcomes: dict[str, tuple[dict | None, Exception | None]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as pool:
        future_to_agent = {pool.submit(call): agent for agent, call in calls.items()}
        for future in concurrent.futures.as_completed(future_to_agent):
            agent = future_to_agent[future]
            try:
                outcomes[agent] = (future.result(), None)
            except Exception as error:
                outcomes[agent] = (None, error)
    return outcomes


def prepare_finalization(
    *,
    server: MeasureServer,
    sessions: dict[str, AgentSession],
    pipe: Pipeline,
    final_operation_deadline: float,
    resident_tasks: ResidentTasks | None = None,
) -> None:
    """Drain exploration and arm a fresh, bounded finalization phase.

    This helper intentionally resets the shared cancellation token only after
    ``server.quiesce`` has joined.  Any quiesce failure therefore raises before
    finalization can restore or measure a stack.
    """

    if resident_tasks is not None:
        resident_tasks.close_admission()
    if getattr(pipe, "turn_authority_enabled", False):
        pipe.close_turn_admissions()
    quiesce_error: Exception | None = None
    try:
        server.quiesce(
            timeout_s=min(
                QUIESCE_JOIN_MAX_S,
                max(0.0, final_operation_deadline - time.monotonic()),
            )
        )
    except Exception as error:
        quiesce_error = error
        pipe.log("measure_server_quiesce_failed", error=str(error)[:400])
    # Resident CLIs are independent and each stop path is internally bounded.
    # Drain them together so four worst-case 30s shutdowns do not consume two
    # minutes of a seven-minute finalization reserve serially.
    if sessions:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(sessions)
        ) as pool:
            stopped = {
                pool.submit(session.stop): name
                for name, session in sessions.items()
            }
            for future in concurrent.futures.as_completed(stopped):
                try:
                    future.result()
                except Exception as error:
                    pipe.log(
                        "session_stop_error",
                        agent=stopped[future],
                        error=str(error)[:400],
                    )
    # Unlike batch waits, asynchronous hardware work can outlive both the
    # measure server and the resident CLI. Never reset cancellation or restore
    # global_best until every such worker has actually returned.
    if resident_tasks is not None:
        resident_tasks.join(timeout_s=min(ASYNC_WORKER_JOIN_MAX_S, max(0.0,
            final_operation_deadline - time.monotonic() - MIN_FINALIZATION_START_BUDGET_S)))
    if quiesce_error is not None or not server.is_quiesced:
        raise RuntimeError(
            "finalization refused because the measure server is not quiescent"
        ) from quiesce_error
    remaining_final_s = final_operation_deadline - time.monotonic()
    if remaining_final_s < MIN_FINALIZATION_START_BUDGET_S:
        raise RuntimeError(
            "finalization reserve was exhausted during quiesce/session shutdown"
        )
    pipe.begin_operation_phase(deadline_monotonic=final_operation_deadline)


def bind_correctness_policy(run_dir: Path, correctness_policy: dict) -> None:
    """Do not silently reinterpret an older run under a new input policy."""
    policy_path = run_dir / "correctness_policy.json"
    if policy_path.is_symlink():
        raise RuntimeError("correctness policy authority cannot be a symlink")
    if policy_path.exists():
        if json.loads(policy_path.read_text()) != correctness_policy:
            raise RuntimeError("run directory is already bound to a different correctness policy")
    elif (run_dir / "ledger.jsonl").exists():
        raise RuntimeError("legacy run has no bound correctness policy; preserve it and use a new run directory")
    else:
        policy_path.write_text(json.dumps(correctness_policy, indent=2, allow_nan=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Directory for run outputs")
    ap.add_argument("--backend", default="jetson_gpu", choices=sorted(BACKENDS),
                    help="Target device backend")
    ap.add_argument(
        "--task",
        type=Path,
        required=True,
        help="PyTorch task file with explicit input shapes and numerical tolerances",
    )
    ap.add_argument("--task-id", help="Task identifier (defaults to the task filename stem)")
    ap.add_argument("--parameter-seed", type=int, default=0,
                    help="Seed for fixed model parameters")
    ap.add_argument(
        "--correctness-profile", choices=(ORDINARY_CORRECTNESS_PROFILE,),
        default=ORDINARY_CORRECTNESS_PROFILE,
        help="Numerical validation profile",
    )
    ap.add_argument("--wall-minutes", type=float, default=30, help="Total run time budget in minutes")
    ap.add_argument("--turn-minutes", type=float, default=8,
                    help="Wait slice per agent turn in minutes; an unfinished turn remains pending")
    ap.add_argument(
        "--finalize-minutes",
        type=float,
        default=7,
        help="Reserve time for final output within the total run budget",
    )
    ap.add_argument("--model", required=True, help="Model identifier configured by the external coding client")
    ap.add_argument("--model-effort", choices=("low", "high"), default=None,
                    help="Explicit CLI effort for all optimizer and hardware sessions; omitted preserves CLI defaults")
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="Maximum new layer turns per optimizer; pending waits do not count (unset: wall budget only)")
    ap.add_argument("--scheduling", choices=("batch", "async"), default="batch",
                    help="Batch waits (default) or opt-in independent resident completion events")
    ap.add_argument("--early-second-hardware", action="store_true",
                    help="Opt-in earlier second HW opportunity for an active lower optimizer; requires async, hardware feedback and --max-rounds 2")
    ap.add_argument("--inline-hardware-feedback", action="store_true",
                    help="Opt-in bounded complete sealed HW packet in measure responses; facts precede unverified model text; requires hardware feedback")
    ap.add_argument("--protected-discussion-turns", action="store_true",
                    help="Opt-in L0 read/peer-only opportunity during protected downstream work; requires async and --max-rounds 1 or 2")
    ap.add_argument("--historical-discussion-replies", action="store_true",
                    help="Opt-in proven ancestor-root direct replies only; requires --protected-discussion-turns")
    ap.add_argument("--session-max-turns", type=int, default=None,
                    help="Maximum number of turns per agent session")
    ap.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Explicit output-token limit (1..28000); omitted uses the external client's default",
    )
    ap.add_argument(
        "--finalize-only",
        action="store_true",
        help="Recheck the saved result without starting optimization agents",
    )
    ap.add_argument(
        "--review",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable optional independent source review",
    )
    ap.add_argument("--review-model", help="Explicit model for optional independent source review")
    ap.add_argument(
        "--rk-collector",
        choices=("off", "devfreq", "hwcpipe"),
        default="off",
        help=(
            "RK3588 profiling mode: TVM only, frequency sampling, or HWCpipe"
        ),
    )
    ap.add_argument("--agents", default="relax,tir,l2",
                    help="Comma-separated optimization agents: relax,tir,l2")
    ap.add_argument("--hardware-feedback", action=argparse.BooleanOptionalAction,
                    default=True, help="Enable device profiling and optimization feedback")
    ap.add_argument("--jobs-root", type=Path, default=None,
                    help="Directory for device jobs; default is RUN_DIR/private/jobs")
    ap.add_argument("--early-hardware-observations", action="store_true",
                    help="Opt-in source-bound VM facts before hardware model analysis; requires inline hardware feedback")
    args = ap.parse_args()
    if args.review and not args.review_model:
        ap.error("--review requires --review-model")
    if args.early_hardware_observations and not (args.hardware_feedback and args.inline_hardware_feedback
                                               and args.protected_discussion_turns):
        ap.error("--early-hardware-observations requires hardware feedback, --inline-hardware-feedback and --protected-discussion-turns")
    if args.inline_hardware_feedback and not args.hardware_feedback:
        ap.error("--inline-hardware-feedback requires hardware feedback")
    if args.early_second_hardware and (
            args.scheduling != "async" or not args.hardware_feedback or args.max_rounds != 2):
        ap.error("--early-second-hardware requires --scheduling async, hardware feedback and explicit --max-rounds 2")
    if args.protected_discussion_turns and (args.scheduling != "async" or args.max_rounds not in {1, 2}):
        ap.error("--protected-discussion-turns requires --scheduling async and explicit --max-rounds 1 or 2")
    if args.historical_discussion_replies and not args.protected_discussion_turns:
        ap.error("--historical-discussion-replies requires --protected-discussion-turns")
    if args.max_output_tokens is not None and not (
        1 <= args.max_output_tokens <= 28000
    ):
        ap.error("--max-output-tokens must be between 1 and 28000")
    try:
        validate_turn_limits(max_rounds=args.max_rounds, session_max_turns=args.session_max_turns)
        validate_time_budgets(
            wall_minutes=args.wall_minutes,
            turn_minutes=args.turn_minutes,
            finalize_minutes=args.finalize_minutes,
            scheduling=args.scheduling,
        )
    except ValueError as error:
        ap.error(str(error))

    optional_client_options = []
    if args.model_effort is not None:
        optional_client_options.append("--effort")
    if args.session_max_turns is not None:
        optional_client_options.append("--max-turns")
    if optional_client_options and not args.finalize_only:
        from ..__main__ import client_capabilities
        capable, detail = client_capabilities(
            os.environ.get("LEKA_AGENT_BINARY", "claude"),
            extra_options=tuple(optional_client_options),
        )
        if not capable:
            ap.error("model client does not support the requested options: " + detail)

    backend = args.backend
    deadline_monotonic = time.monotonic() + args.wall_minutes * 60
    work_deadline = deadline_monotonic - args.finalize_minutes * 60
    final_operation_deadline = deadline_monotonic - FINALIZATION_EXIT_MARGIN_S
    startup_deadline = final_operation_deadline if args.finalize_only else work_deadline
    startup_control = OperationControl()
    startup_control.reset(deadline_monotonic=startup_deadline)
    # Task import/export runs in this process. Check its elapsed time before
    # admitting later work; these checkpoints cannot interrupt a native call.
    startup_control.checkpoint()
    run_dir = Path(args.run_dir)
    task = load_pytorch_task(
        args.task,
        parameter_seed=args.parameter_seed,
        task_id=args.task_id,
    )
    startup_control.checkpoint()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.task is not None:
        manifest_path = run_dir / "task_manifest.json"
        manifest = task.manifest()
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError("run directory is already bound to a different task")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2))
    service = MeasureService(task=task, backend=backend, tvm_python=TVM_PYTHON,
                             correctness_profile=args.correctness_profile,
                             jobs_root=(args.jobs_root if args.jobs_root is not None
                                        else run_dir.resolve() / "private" / "jobs"))
    correctness_policy = service.correctness_policy()
    bind_correctness_policy(run_dir, correctness_policy)
    reviewer = CodexReviewer(model=args.review_model) if args.review else DisabledReviewer()
    startup_control.checkpoint()
    pipe = Pipeline(run_dir, task, service, reviewer=reviewer)
    # Bind Git, lowering, device leases and baseline jobs before bootstrap.
    # Startup consumes the existing work budget, preserving finalization time.
    pipe.begin_operation_phase(deadline_monotonic=startup_deadline)
    pipe.operation_control.checkpoint()
    pipe.log("correctness_policy", **correctness_policy)
    pipe.promote_requires_gain = True
    pipe.bootstrap()
    pipe.operation_control.checkpoint()
    baseline = pipe.measure_default()
    pipe.operation_control.checkpoint()
    baseline_latency = baseline.public.get("latency_ms")
    pipe.log(
        "default_baseline",
        ok=baseline.ok,
        latency_ms=baseline_latency,
        diagnostic=baseline.public.get("diagnostic"),
    )
    baseline_latency = require_baseline_latency(baseline)
    best_state = pipe.initialize_global_best(baseline)
    pipe.operation_control.checkpoint()
    print(f"default stack T1: {baseline_latency:.4f} ms")
    if best_state.get("recovered"):
        print(
            "recovered sealed global best: "
            f"{best_state['selected_latency_ms']:.4f} ms "
            f"({best_state['snapshot_id']})"
        )

    if args.finalize_only:
        if not best_state.get("recovered"):
            error = "--finalize-only requires an existing sealed global-best checkpoint"
            pipe.log("final_default_error", error=error, provider_invoked=False)
            raise RuntimeError(error)
        pipe.log(
            "finalization_only_recovery",
            provider_invoked=False,
            snapshot_id=best_state["snapshot_id"],
        )
        pipe.begin_operation_phase(deadline_monotonic=final_operation_deadline)
        finalize_and_report(pipe, backend=backend, baseline_latency=baseline_latency,
                            finalize_only=True)
        return

    agents = layer_agents(backend, args.agents)          # name -> layer
    if args.protected_discussion_turns:
        pipe.enable_admission_authority(agents)
    hw_name = BACKENDS[backend]["hw_sense"]
    roles = resident_roles(agents, backend, hardware_feedback=args.hardware_feedback)
    pipe.log("hardware_feedback_policy", enabled=args.hardware_feedback,
             inline_feedback=args.inline_hardware_feedback,
             static_hardware_knowledge=True, ordinary_measurements=True,
             resident_roles=roles)
    shared_pool = SharedMessagePool(
        run_dir / "inbox" / "_shared_pool",
        backend=backend,
        participants=roles,
    )
    workspaces = {a: stage_workspace(run_dir, a, l, pipe, backend) for a, l in agents.items()}
    if args.hardware_feedback:
        workspaces[hw_name] = stage_workspace(run_dir, hw_name, None, pipe, backend)
    pipe._peer_dialogue = LivePeerDialogue(pipe, shared_pool, workspaces, backend)
    if args.hardware_feedback:
        pipe._hardware_delivery = HardwareEvidenceDelivery(
            pipe, workspaces, agents, backend=backend,
            inline_feedback=args.inline_hardware_feedback)
    if args.early_hardware_observations:
        pipe._hardware_observations = HardwareObservationDelivery(pipe, agents)
        pipe.log("early_hardware_observation_policy", enabled=True,
                 authority="deterministic_selected_facts_not_model_hypothesis",
                 additional_capture_or_model_requests=False)

    server = MeasureServer(pipe, agents, {a: workspaces[a] for a in agents})

    sessions = {}
    session_logs = run_dir / "private" / "session_logs"
    if session_logs.is_symlink() or (session_logs.exists() and not session_logs.is_dir()):
        raise RuntimeError("private session log root must be a real directory")
    session_logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    session_logs.chmod(0o700)
    first = {a: True for a in agents}
    by_layer = {l: a for a, l in agents.items()}
    order = [by_layer[l] for l in ("L0", "L1", "L2") if l in by_layer]
    priority: list[str] = []
    rebase = DescendantRebaseScheduler()
    round_budget = RoundBudget(agents, args.max_rounds)
    resident_tasks = None
    dead_sessions_logged: set[str] = set()

    def observe_rebase(before: dict, after: dict) -> None:
        changed_layer = rebase.observe_binding_change(before, after)
        if changed_layer is not None:
            pipe.log(
                "descendant_rebase_scheduled",
                changed_layer=changed_layer,
                protected_layers=list(rebase.layers),
                new_binding=after,
            )

    try:
        # All exploration-side transports/subprocesses inherit the same
        # monotonic work deadline.  A request admitted just before the cutoff
        # therefore cannot consume the reserve intended for finalization.
        pipe.begin_operation_phase(deadline_monotonic=work_deadline)
        server.start()
        # Session startup is inside the cleanup boundary.  If a later provider
        # fails to start, every earlier one is still stopped before propagating
        # the startup error.
        session_extra_env: dict[str, str] = {}
        if args.max_output_tokens is not None:
            session_extra_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(
                args.max_output_tokens
            )
        pipe.log("agent_client_policy", client="claude", model=args.model,
                 ir_and_evaluator_unchanged=True)
        for name in roles:
            s = AgentSession(
                name,
                workspaces[name],
                profile_text=profile_text(
                    name if name in agents else "hw_sense",
                    backend,
                    task_facts=task.public_facts(),
                    hardware_feedback=args.hardware_feedback,
                ),
                model=args.model,
                model_effort=args.model_effort,
                extra_env=session_extra_env,
                log_path=session_logs / f"{name}.jsonl",
                **session_turn_options(args.session_max_turns),
            )
            # Register before start(): Popen/thread startup can fail after the
            # session log was opened, and the finally block must still own and
            # close that partially initialized session.
            sessions[name] = s
            s.poll_callback = pipe._peer_dialogue.poll
            s.start()
            pipe.log("session_started", agent=name, agent_client="claude",
                     provider_policy=s.provider_policy)

        pipe.log("scheduling_policy", mode=args.scheduling,
                 maximum_new_turns_per_optimizer=args.max_rounds,
                 work_deadline_is_authoritative=True,
                 **({"early_second_hardware": True} if args.early_second_hardware else {}))
        if args.scheduling == "async":
            resident_tasks = ResidentTasks(roles)
            run_event_loop(tasks=resident_tasks, pipe=pipe, sessions=sessions, agents=agents,
                workspaces=workspaces, shared_pool=shared_pool, rebase=rebase,
                round_budget=round_budget, first=first, backend=backend, args=args,
                work_deadline=work_deadline,
                **({"claimed_request_count": lambda: server.claimed_request_count}
                   if args.early_second_hardware else {}))

        while args.scheduling == "batch" and time.monotonic() < work_deadline:
            if round_budget.should_stop(
                live_agents=[name for name in agents if sessions[name].alive()],
                pending_agents=[name for name, session in sessions.items()
                                if session.has_pending() and session.alive()],
            ):
                pipe.log("round_limit_reached", **round_budget.snapshot(),
                         action="all_admitted_turns_drained_stop_exploration_and_finalize")
                break
            # Hardware evidence may prioritize descendants, but L0 must retain
            # one protected presentation slot every batch: otherwise repeated
            # lower-layer promotions can consume the whole graph-search wall
            # budget before the graph owner refreshes and proposes a branch.
            protected = next_admissible_rebase(rebase, by_layer, sessions, round_budget, pipe.log)
            if protected is not None:
                protected_layer, protected_agent = protected
                schedule = [protected_agent]
                pipe.log(
                    "descendant_rebase_turn_scheduled",
                    layer=protected_layer,
                    agent=protected_agent,
                    remaining_layers=list(rebase.layers),
                )
            else:
                protected_layer = protected_agent = None
                l0_agent = by_layer.get("L0")
                schedule = [
                    *([l0_agent] if l0_agent is not None else []),
                    *[a for a in priority if a != l0_agent],
                    *[a for a in order if a != l0_agent and a not in priority],
                ]
                priority = []
            calls: dict[str, Callable[[], dict | None]] = {}
            resumed_pending: dict[str, bool] = {}
            turn_bindings: dict[str, dict] = {}
            turn_message_ids: dict[str, tuple[str, ...]] = {}
            turn_ids: dict[str, str] = {}
            batch_binding = pipe.stack.binding().to_dict()
            for agent in schedule:
                if time.monotonic() >= work_deadline:
                    break
                s = sessions[agent]
                if not s.alive():
                    if agent not in dead_sessions_logged:
                        pipe.log("session_died", agent=agent)
                        dead_sessions_logged.add(agent)
                    if agent == protected_agent and protected_layer is not None:
                        disposition = rebase.finish_turn(
                            protected_layer,
                            completed=True,
                            provider_succeeded=False,
                        )
                        pipe.log(
                            "descendant_rebase_turn_finished",
                            layer=protected_layer,
                            agent=agent,
                            disposition=disposition,
                            remaining_layers=list(rebase.layers),
                        )
                    continue
                # A timed-out CLI turn remains active.  Deliver its formal
                # result before draining inbox messages or constructing a new
                # prompt; otherwise the old result could be overwritten and
                # the newly drained messages could be silently discarded.
                if s.has_pending():
                    pending_binding = getattr(
                        s, "_leka_pending_layer_binding", None
                    )
                    if not isinstance(pending_binding, dict):
                        raise RuntimeError(
                            f"pending layer turn {agent} has no frozen binding"
                        )
                    turn_bindings[agent] = pending_binding
                    turn_message_ids[agent] = tuple(
                        getattr(s, "_leka_pending_shared_message_ids", ())
                    )
                    turn_ids[agent] = str(
                        getattr(s, "_leka_pending_layer_turn_id", "")
                    )
                    calls[agent] = (
                        lambda session=s:
                        invoke_layer_turn(session, prompt=None,
                            turn_minutes=args.turn_minutes, work_deadline=work_deadline)
                    )
                    resumed_pending[agent] = True
                    continue
                if not round_budget.can_start(agent):
                    continue
                prompt, turn_binding, shared_message_ids, turn_id = prepare_layer_turn(
                    pipe,
                    shared_pool,
                    agent=agent,
                    layer=agents[agent],
                    workspace=workspaces[agent],
                    backend=backend,
                    first=first[agent],
                )
                prompt += budget_prompt_suffix(round_budget, agent, work_deadline)
                round_budget.admit(agent, turn_id)
                pipe.log("layer_turn_admitted", agent=agent, turn_id=turn_id,
                         **round_budget.snapshot())
                s._leka_pending_layer_binding = turn_binding
                s._leka_pending_shared_message_ids = shared_message_ids
                s._leka_pending_layer_turn_id = turn_id
                turn_bindings[agent] = turn_binding
                turn_message_ids[agent] = shared_message_ids
                turn_ids[agent] = turn_id
                first[agent] = False
                calls[agent] = (
                    lambda session=s, text=prompt:
                    invoke_layer_turn(session, prompt=text,
                        turn_minutes=args.turn_minutes, work_deadline=work_deadline)
                )
                resumed_pending[agent] = False

            if not calls:
                hw_pending = (args.hardware_feedback and sessions[hw_name].has_pending()
                              and sessions[hw_name].alive())
                if not any(sessions[name].alive() for name in agents) and not hw_pending:
                    pipe.log(
                        "all_optimizer_sessions_terminal",
                        action="stop_exploration_and_finalize",
                    )
                    break
                # A protected rebase can momentarily point at a terminal
                # session.  Avoid a hot empty loop while the scheduler moves
                # to another live participant.
                if not hw_pending:
                    time.sleep(min(0.1, max(0.0, work_deadline - time.monotonic())))
                    continue
                # Optimizer quotas may be exhausted while one previously
                # admitted hardware reply still needs delivery. Fall through
                # with an empty layer batch and drain that exact hardware turn.

            outcomes = run_concurrent_turns(calls)
            # ``./measure promote`` can change the stack while model calls are
            # still running, before their result text is applied below.
            observe_rebase(batch_binding, pipe.stack.binding().to_dict())
            # Apply replies in the deterministic presentation order.  A
            # promotion may make a sibling's already-measured node stale; the
            # pipeline's binding/epoch gate then rejects it explicitly.
            for agent in schedule:
                if agent not in outcomes:
                    continue
                result, error = outcomes[agent]
                if error is not None:
                    pipe.log("turn_error", agent=agent, error=str(error)[:400])
                    if agent == protected_agent and protected_layer is not None:
                        disposition = rebase.finish_turn(
                            protected_layer,
                            completed=True,
                            provider_succeeded=False,
                        )
                        pipe.log(
                            "descendant_rebase_turn_finished",
                            layer=protected_layer,
                            agent=agent,
                            disposition=disposition,
                            remaining_layers=list(rebase.layers),
                        )
                    continue
                provider_status = terminal_provider_error(result)
                if provider_status is not None:
                    pipe.log(
                        "turn_terminal_provider",
                        agent=agent,
                        status=provider_status,
                        resumed_pending=resumed_pending[agent],
                        action="leave_peer_advice_unacknowledged",
                    )
                    if agent == protected_agent and protected_layer is not None:
                        disposition = rebase.finish_turn(
                            protected_layer,
                            completed=True,
                            provider_succeeded=False,
                        )
                        pipe.log(
                            "descendant_rebase_turn_finished",
                            layer=protected_layer,
                            agent=agent,
                            disposition=disposition,
                            remaining_layers=list(rebase.layers),
                        )
                    continue
                if result is not None and not provider_turn_succeeded(result):
                    pipe.log(
                        "turn_provider_error",
                        agent=agent,
                        resumed_pending=resumed_pending[agent],
                        action="leave_peer_advice_unacknowledged",
                        cost_usd=result.get("total_cost_usd"),
                    )
                    pipe.mailbox.send(
                        agent,
                        "provider_retry",
                        {
                            "note": (
                                "The prior provider stream ended without a successful "
                                "formal result. Re-read the current binding and inspect "
                                "your retained workspace edits before continuing; do not "
                                "assume any old candidate is still promotable."
                            ),
                            "prior_binding": turn_bindings[agent],
                        },
                    )
                    for attribute in (
                        "_leka_pending_layer_binding",
                        "_leka_pending_shared_message_ids",
                        "_leka_pending_layer_turn_id",
                    ):
                        if hasattr(sessions[agent], attribute):
                            delattr(sessions[agent], attribute)
                    if agent == protected_agent and protected_layer is not None:
                        disposition = rebase.finish_turn(
                            protected_layer,
                            completed=True,
                            provider_succeeded=False,
                        )
                        pipe.log(
                            "descendant_rebase_turn_finished",
                            layer=protected_layer,
                            agent=agent,
                            disposition=disposition,
                            remaining_layers=list(rebase.layers),
                        )
                    continue
                before_reply = pipe.stack.binding().to_dict()
                record_layer_turn(
                    pipe,
                    agent=agent,
                    workspace=workspaces[agent],
                    backend=backend,
                    result=result,
                    resumed_pending=resumed_pending[agent],
                    shared_pool=shared_pool,
                    turn_binding=turn_bindings[agent],
                    turn_id=turn_ids[agent],
                )
                observe_rebase(before_reply, pipe.stack.binding().to_dict())
                if result is not None:
                    acknowledge_shared_advice(
                        pipe,
                        shared_pool,
                        agent,
                        turn_message_ids[agent],
                        turn_id=turn_ids[agent],
                        binding=turn_bindings[agent],
                    )
                    for attribute in (
                        "_leka_pending_layer_binding",
                        "_leka_pending_shared_message_ids",
                        "_leka_pending_layer_turn_id",
                    ):
                        if hasattr(sessions[agent], attribute):
                            delattr(sessions[agent], attribute)
                if agent == protected_agent and protected_layer is not None:
                    # A successful provider reply to an old pending turn is
                    # not proof that this layer has rebased. Keep its protected
                    # slot until it gets a turn on the actual current inputs.
                    with pipe._lock:
                        projection_current, _ = workspace_projection_status(
                            pipe, workspaces[agent], protected_layer
                        )
                    disposition = rebase.finish_turn(
                        protected_layer,
                        completed=result is not None,
                        provider_succeeded=provider_turn_succeeded(result),
                        projection_current=projection_current,
                    )
                    pipe.log(
                        "descendant_rebase_turn_finished",
                        layer=protected_layer,
                        agent=agent,
                        disposition=disposition,
                        remaining_layers=list(rebase.layers),
                    )
            terminal_provider_agents = [
                (agent, terminal_provider_error(result))
                for agent, (result, error) in outcomes.items()
                if error is None and terminal_provider_error(result) is not None
            ]
            if terminal_provider_agents:
                statuses = sorted({status for _, status in terminal_provider_agents})
                affected = sorted(agent for agent, _ in terminal_provider_agents)
                pipe.log(
                    "provider_unavailable",
                    statuses=statuses,
                    agents=affected,
                    action="stop_before_hardware_profile",
                )
                print(
                    "model provider returned non-retryable HTTP "
                    f"{','.join(statuses)} for {','.join(affected)}; stopping "
                    "before another hardware profile"
                )
                break
            unavailable, unavailable_reason = device_unavailable(pipe)
            if unavailable:
                pipe.log(
                    "device_unavailable",
                    backend=backend,
                    reason=unavailable_reason,
                    action="stop_exploration",
                )
                print(
                    f"{backend} became unavailable ({unavailable_reason}); "
                    "stopping exploration without further device retries"
                )
                break
            if not args.hardware_feedback:
                continue
            if time.monotonic() >= work_deadline or not sessions[hw_name].alive():
                continue
            if rebase.active and not sessions[hw_name].has_pending():
                pipe.log(
                    "hardware_turn_deferred",
                    reason="protected_descendant_rebase",
                    remaining_layers=list(rebase.layers),
                )
                continue
            if not sessions[hw_name].has_pending() and not any(
                sessions[name].alive() and round_budget.can_start(name) for name in agents
            ):
                # No future optimizer turn can consume a new profile. Already
                # pending hardware turns are still drained above/below.
                continue
            try:
                delivered = hw_turn(
                    pipe,
                    sessions[hw_name],
                    workspaces[hw_name],
                    backend,
                    bounded_turn_wait_seconds(
                        turn_minutes=args.turn_minutes, work_deadline=work_deadline),
                    rk_collector=args.rk_collector,
                    shared_pool=shared_pool,
                )
            except TerminalProviderError as err:
                pipe.log(
                    "provider_unavailable",
                    agent=hw_name,
                    error=str(err)[:400],
                    action="stop_after_hardware_agent_failure",
                )
                print(f"{err}; stopping hardware turns")
                break
            except Exception as err:  # a failed collection must not kill the run
                pipe.log("hw_turn_error", error=str(err)[:400])
                delivered = []
            unavailable, unavailable_reason = device_unavailable(pipe)
            if unavailable:
                pipe.log(
                    "device_unavailable",
                    backend=backend,
                    reason=unavailable_reason,
                    action="stop_after_hardware_turn",
                )
                print(
                    f"{backend} became unavailable ({unavailable_reason}); "
                    "hardware collection will not be retried"
                )
                break
            priority = [{"L0": by_layer.get("L0"), "L1": by_layer.get("L1"),
                         "L2": by_layer.get("L2")}.get(p.get("target_layer"))
                        for p in delivered]
            priority = [a for a in dict.fromkeys(priority) if a]
    finally:
        pipe.log("layer_turn_budget_summary", **round_budget.snapshot())
        # Admission closes before cancellation.  Once quiesce returns, its
        # joined server thread is the proof that no request still owns the
        # Pipeline RLock; finalization must never race an exploratory write.
        # Cancellation is cleared only after the server join.  The restored
        # best's final T1 is a new single-threaded phase, bounded just inside
        # the run wall deadline so failures surface before hard SIGTERM.
        prepare_finalization(
            server=server,
            sessions=sessions,
            pipe=pipe,
            final_operation_deadline=final_operation_deadline,
            resident_tasks=resident_tasks,
        )
        # No worker or CLI can publish another protocol file after this point.
        # In particular, never force-poll while an async HW collector may own
        # the Pipeline lock. A failed quiesce skips this snapshot and fails
        # closed before restoration rather than risking a blocking summary.
        try:
            pipe._peer_dialogue.poll(force=True)
            pipe.log("peer_dialogue_summary", **shared_pool.dialogue_summary())
        except (OSError, ValueError, SharedMessagePoolError) as error:
            pipe.log("peer_dialogue_summary_error", error=str(error)[:300])
        finalize_and_report(pipe, backend=backend, baseline_latency=baseline_latency)



if __name__ == "__main__":
    main()
