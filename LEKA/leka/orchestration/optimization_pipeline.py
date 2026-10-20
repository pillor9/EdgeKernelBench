"""Validate, measure, and promote candidates for one task and backend.

The pipeline maintains representation dependencies and candidate ancestry.
It records transitions in ledger.jsonl and saves candidate artifacts under
candidates/<node_id>/.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..evaluation.evaluation_service import MeasureResult, MeasureService
from ..evaluation.timing_quality import assess_timing
from ..operation_control import OperationCancelled, OperationControl
from ..worktree.candidate_workspace import PipelineRepo
from .message_bus import Mailbox
from .candidate_tree import OperatorTree
from .program_state import ArtifactStack, StackBinding

AGENT_OF_LAYER = {"L0": "relax_agent", "L1": "tir_agent", "L2": "cuda_agent"}
LAYERS_BELOW = {"L0": ("L1", "L2"), "L1": ("L2",), "L2": ()}
GLOBAL_BEST_SCHEMA_VERSION = 1
# Stable paired measurements must also clear the minimum improvement rule.
PAIRED_GAIN_FLOOR = 0.005


@dataclass
class Candidate:
    node_id: str
    layer: str
    binding: dict
    dir: Path
    projected_inputs: dict | None = None
    branch_id: str | None = None

    @property
    def source(self) -> str:
        return (self.dir / "source.py").read_text()

    def result(self) -> dict | None:
        p = self.dir / "result.json"
        return json.loads(p.read_text()) if p.exists() else None


class Pipeline:
    def __init__(self, run_dir: Path, task, service: MeasureService, *, reviewer=None):
        self.run_dir = run_dir
        self.task = task
        self.service = service
        self.reviewer = reviewer
        l2_ext = getattr(service, "cfg", {}).get("l2_ext", "cu") if hasattr(service, "cfg") else "cu"
        self.stack = ArtifactStack.open(run_dir / "stack", l2_ext=l2_ext)
        self.operator_tree = OperatorTree(run_dir, self.stack)
        self.mailbox = Mailbox(run_dir)
        self.repo = PipelineRepo(run_dir)
        # the measure-server thread and the main loop both call into the
        # pipeline; stack mutations and measurements are serialized coarsely
        self._lock = threading.RLock()
        # Every long trusted subprocess/transport seam observes this phase
        # token.  Exploration binds it to work_deadline; shutdown cancels it;
        # only after the measure thread is joined is it reset for final T1.
        self.operation_control = OperationControl()
        self.repo.set_operation_control(self.operation_control)
        service_control = getattr(self.service, "set_operation_control", None)
        if callable(service_control):
            service_control(self.operation_control)
        reviewer_control = getattr(self.reviewer, "set_operation_control", None)
        if callable(reviewer_control):
            reviewer_control(self.operation_control)
        # Per-driver capability key used to bind each agent workspace to the
        # exact stack snapshot that was projected into it.  The key never
        # enters an agent workspace; agents only receive a signed envelope and
        # therefore cannot relabel an old-base request as current.
        self._workspace_binding_key = secrets.token_bytes(32)
        # Opt-in only. A discussion admission may read/advise, never mutate
        # the active program through a measurement or final-reply action.
        self.turn_authority_enabled = False
        self._turn_authority_roles = {}
        self._turn_authorities = {}
        self._turn_admissions_closed = threading.Event()
        (run_dir / "candidates").mkdir(parents=True, exist_ok=True)

    # -- graph-branch lineage ------------------------------------------------
    @property
    def _branches_root(self) -> Path:
        return self.run_dir / "private" / "branches"

    def _seal_branch_root(
        self,
        *,
        branch_id: str,
        parent_branch_id: str | None,
        promoted_node: str | None,
    ) -> dict:
        """Persist one immutable root checkpoint for an L0 graph branch."""

        if not isinstance(branch_id, str) or not branch_id.startswith("b-"):
            raise ValueError("invalid graph branch identity")
        private = self.run_dir / "private"
        root = self._branches_root
        branch = root / branch_id
        for directory in (private, root, branch):
            if directory.is_symlink():
                raise RuntimeError("graph branch state may not traverse a symlink")
            directory.mkdir(mode=0o700, exist_ok=True)
            if not directory.is_dir():
                raise RuntimeError("graph branch state must be a real directory")
            os.chmod(directory, 0o700)
        pointer = branch / "root.json"
        if pointer.exists():
            if pointer.is_symlink() or not pointer.is_file():
                raise RuntimeError("graph branch root pointer must be a regular file")
            state = json.loads(pointer.read_text(encoding="utf-8"))
            if (
                not isinstance(state, dict)
                or state.get("branch_id") != branch_id
                or state.get("parent_branch_id") != parent_branch_id
                or state.get("root_l0_sha") != self.stack.sha("L0")
            ):
                raise RuntimeError("existing graph branch root identity differs")
            return state

        checkpoint, manifest = self.stack.seal_snapshot(
            branch / "checkpoints",
            metadata={
                "selection_kind": "l0_branch_root",
                "branch_id": branch_id,
                "parent_branch_id": parent_branch_id,
                "promoted_node": promoted_node,
                "root_l0_sha": self.stack.sha("L0"),
            },
        )
        state = {
            "schema_version": 1,
            "branch_id": branch_id,
            "parent_branch_id": parent_branch_id,
            "promoted_node": promoted_node,
            "root_l0_sha": self.stack.sha("L0"),
            "checkpoint": f"checkpoints/{checkpoint.name}",
            "manifest_sha256": self._manifest_sha(checkpoint),
            "root_binding": self.stack.binding().to_dict(),
            "created_at": time.time(),
        }
        encoded = json.dumps(
            state, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        temporary = branch / f".root-{uuid.uuid4().hex}.tmp"
        ArtifactStack._write_private_text(temporary, encoded)
        try:
            os.link(temporary, pointer)
            temporary.unlink()
        finally:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
        self.log(
            "l0_branch_root",
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            promoted_node=promoted_node,
            root_l0_sha=state["root_l0_sha"],
            checkpoint=f"private/branches/{branch_id}/{state['checkpoint']}",
        )
        return state

    def prove_discussion_root_relation(self, *, original_binding: dict,
                                      reply_binding: dict,
                                      expected_current_binding: dict) -> dict:
        """Read-only proof Q <= R < current; never repairs or restores state.

        Peer bindings contain no branch id. A source hash must identify exactly
        one evaluator-recorded branch, then every ancestor edge is checked
        against its sealed root. Ambiguity and incomplete legacy state fail
        closed. This is provenance for discussion, never optimization authority.
        """
        with self._lock:
            remaining = 64 * 1024 * 1024

            def read(parts, maximum):
                nonlocal remaining
                # Walk from the trusted run directory using no-follow dirfds;
                # checking only the final path would miss a linked ancestor.
                flags = os.O_RDONLY | os.O_NOFOLLOW
                descriptor = os.open(self.run_dir, flags | os.O_DIRECTORY)
                try:
                    for part in parts[:-1]:
                        child = os.open(part, flags | os.O_DIRECTORY, dir_fd=descriptor)
                        os.close(descriptor)
                        descriptor = child
                    # A FIFO can block before fstat; no-follow alone is not
                    # enough to make reads bounded. Non-regular files fail below.
                    fd = os.open(parts[-1], flags | os.O_NONBLOCK, dir_fd=descriptor)
                    try:
                        info = os.fstat(fd)
                        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                                or info.st_size > min(maximum, remaining)):
                            raise ValueError("root proof file is not bounded regular data")
                        data = bytearray()
                        limit = min(maximum, remaining)
                        while len(data) <= limit:
                            chunk = os.read(fd, min(65536, limit + 1 - len(data)))
                            if not chunk:
                                break
                            data.extend(chunk)
                        if len(data) > limit:
                            raise ValueError("root proof read budget exceeded")
                        remaining -= len(data)
                        return bytes(data)
                    finally:
                        os.close(fd)
                finally:
                    os.close(descriptor)

            def sha(value):
                return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

            def branch(value):
                return isinstance(value, str) and re.fullmatch(r"b-[A-Za-z0-9_-]{1,96}", value) is not None

            def valid_binding(binding):
                return (isinstance(binding, dict)
                        and type(binding.get("epoch")) is int and binding["epoch"] >= 0
                        and sha(binding.get("l0_sha"))
                        and all(binding.get(k) == "absent" or sha(binding.get(k))
                                for k in ("l1_sha", "l2_sha")))

            try:
                for binding in (original_binding, reply_binding, expected_current_binding):
                    if not valid_binding(binding):
                        raise ValueError("source binding is malformed")
                roots = self.stack.meta.get("branch_root_l0")
                parents = self.stack.meta.get("branch_parent")
                if (not isinstance(roots, dict) or not isinstance(parents, dict)
                        or not 1 <= len(roots) <= 256 or set(roots) != set(parents)
                        or any(not branch(k) or not sha(v) for k, v in roots.items())
                        or any(v is not None and (not branch(v) or v not in roots)
                               for v in parents.values())):
                    raise ValueError("root index incomplete or outside proof bounds")
                current_id = self.stack.active_branch_id()
                if current_id not in roots:
                    raise ValueError("current branch is unknown")
                # The caller supplies a frozen binding under this same lock.
                # Verify current bytes with bounded reads, including lower slots.
                current = {"epoch": self.stack.epoch}
                for slot in ("L0", "L1", "L2"):
                    try:
                        data = read(("stack", self.stack.path(slot).name), 16 * 1024 * 1024)
                        current[slot.lower() + "_sha"] = hashlib.sha256(data).hexdigest()
                    except FileNotFoundError:
                        current[slot.lower() + "_sha"] = "absent"
                if current != expected_current_binding or current["l0_sha"] != roots[current_id]:
                    raise ValueError("current root changed or differs from its registered source")
                selected = []
                for binding in (original_binding, reply_binding):
                    if not isinstance(binding, dict) or not sha(binding.get("l0_sha")):
                        raise ValueError("source root hash is missing or invalid")
                    matches = [key for key, value in roots.items() if value == binding["l0_sha"]]
                    if len(matches) != 1:
                        raise ValueError("source root is unknown or ambiguous")
                    selected.append(matches[0])
                question_id, reply_id = selected
                chain, seen = [], set()
                cursor = current_id
                while cursor is not None:
                    if cursor in seen or len(chain) >= 64:
                        raise ValueError("root parent cycle or depth limit")
                    seen.add(cursor)
                    prefix = ("private", "branches", cursor)
                    pointer = json.loads(read((*prefix, "root.json"), 64 * 1024))
                    expected = {"schema_version": 1, "branch_id": cursor,
                                "parent_branch_id": parents[cursor], "root_l0_sha": roots[cursor]}
                    if (not isinstance(pointer, dict) or type(pointer.get("schema_version")) is not int
                            or any(pointer.get(k) != v for k, v in expected.items())):
                        raise ValueError("sealed root pointer disagrees with branch index")
                    checkpoint = pointer.get("checkpoint")
                    if not isinstance(checkpoint, str) or re.fullmatch(r"checkpoints/s[0-9]+-[0-9a-f]{10}", checkpoint) is None:
                        raise ValueError("unsafe root checkpoint identity")
                    checkpoint_parts = (*prefix, *checkpoint.split("/"))
                    raw = read((*checkpoint_parts, "manifest.json"), 128 * 1024)
                    if not sha(pointer.get("manifest_sha256")) or hashlib.sha256(raw).hexdigest() != pointer["manifest_sha256"]:
                        raise ValueError("root manifest hash differs")
                    manifest = json.loads(raw)
                    metadata = manifest.get("metadata", {})
                    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
                            or manifest.get("snapshot_id") != checkpoint.split("/")[1]
                            or manifest.get("l2_ext") != self.stack.l2_ext
                            or metadata.get("selection_kind") != "l0_branch_root"
                            or any(metadata.get(k) != expected[k] for k in ("branch_id", "parent_branch_id", "root_l0_sha"))
                            or manifest.get("source_binding") != pointer.get("root_binding")
                            or not valid_binding(pointer.get("root_binding"))
                            or type(manifest.get("source_epoch")) is not int
                            or manifest["source_epoch"] != pointer["root_binding"]["epoch"]
                            or pointer["root_binding"].get("l0_sha") != roots[cursor]):
                        raise ValueError("root checkpoint metadata differs")
                    record = manifest.get("slots", {}).get("L0")
                    if (not isinstance(record, dict) or record.get("present") is not True
                            or record != {"present": True, "path": "L0.txt", "sha256": roots[cursor]}):
                        raise ValueError("root L0 slot identity differs")
                    if hashlib.sha256(read((*checkpoint_parts, "L0.txt"), 16 * 1024 * 1024)).hexdigest() != roots[cursor]:
                        raise ValueError("root L0 source hash differs")
                    chain.append({"branch_id": cursor, "parent_branch_id": parents[cursor],
                                  "l0_sha": roots[cursor], "manifest_sha256": pointer["manifest_sha256"]})
                    cursor = parents[cursor]
                if (question_id not in seen or reply_id not in seen
                        or reply_id == current_id
                        or [x["branch_id"] for x in chain].index(question_id)
                           < [x["branch_id"] for x in chain].index(reply_id)):
                    raise ValueError("required original <= reply < current ancestry does not hold")
                return {"verified": True, "relation": "original<=reply<current",
                        "authority": "evaluator_root_provenance_only_not_claim_validation",
                        "original_root": {"branch_id": question_id, "l0_sha": roots[question_id]},
                        "reply_root": {"branch_id": reply_id, "l0_sha": roots[reply_id]},
                        "current_root": {"branch_id": current_id, "l0_sha": roots[current_id]},
                        "captured_current_binding": current, "verified_chain_current_to_root": chain}
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
                return {"verified": False, "reason": str(error)[:200]}

    def begin_operation_phase(self, *, deadline_monotonic: float | None) -> None:
        """Reset cancellation and bind subsequent work to a phase deadline."""

        self.operation_control.reset(deadline_monotonic=deadline_monotonic)

    def cancel_inflight(self, reason: str) -> None:
        """Cooperatively stop service/reviewer work currently holding the lock."""

        self.operation_control.cancel(reason)

    # -- driver-private global-best checkpoint ------------------------------
    @property
    def _global_best_root(self) -> Path:
        return self.run_dir / "private" / "global_best"

    def _prepare_global_best_root(self) -> Path:
        private = self.run_dir / "private"
        if private.is_symlink():
            raise RuntimeError("driver-private state root may not be a symlink")
        private.mkdir(mode=0o700, exist_ok=True)
        if not private.is_dir():
            raise RuntimeError("driver-private state root must be a directory")
        root = self._global_best_root
        if root.is_symlink():
            raise RuntimeError("global-best state root may not be a symlink")
        root.mkdir(mode=0o700, exist_ok=True)
        if not root.is_dir():
            raise RuntimeError("global-best state root must be a directory")
        os.chmod(private, 0o700)
        os.chmod(root, 0o700)
        return root

    @staticmethod
    def _positive_latency(value: object) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("global-best measurement has no positive finite latency")
        return float(value)

    @staticmethod
    def _manifest_sha(checkpoint: Path) -> str:
        manifest_path = checkpoint / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise RuntimeError("global-best checkpoint manifest is not a regular file")
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def _write_pointer_state(
        self, filename: str, state: dict, *, replace: bool
    ) -> None:
        root = self._prepare_global_best_root()
        target = root / filename
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise RuntimeError(f"global-best {filename} pointer must be a regular file")
        if target.exists() and not replace:
            raise RuntimeError(f"global-best {filename} pointer is immutable")
        encoded = json.dumps(
            state, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        temporary = root / f".{filename}-{uuid.uuid4().hex}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            if replace:
                os.replace(temporary, target)
            else:
                # Publish the fallback pointer without an overwrite race.  It
                # is immutable for the lifetime of the run; later best-pointer
                # updates must never retarget the naive/imported rollback.
                os.link(temporary, target)
                temporary.unlink()
        finally:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()

    def _write_best_state(self, state: dict) -> None:
        self._write_pointer_state("best.json", state, replace=True)

    def _read_pointer(
        self, filename: str, *, label: str
    ) -> tuple[dict, Path, dict, dict[str, str | None]] | None:
        root = self._global_best_root
        pointer = root / filename
        if not pointer.exists():
            return None
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError("global-best state root must be a real directory")
        if pointer.is_symlink() or not pointer.is_file():
            raise RuntimeError(f"{label} pointer must be a regular file")
        if pointer.stat().st_size > 1024 * 1024:
            raise RuntimeError(f"{label} pointer exceeds its size limit")
        state = json.loads(pointer.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("schema_version") != GLOBAL_BEST_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported {label} pointer schema")
        relative = state.get("checkpoint")
        if not isinstance(relative, str):
            raise RuntimeError(f"{label} checkpoint identity is missing")
        parts = Path(relative).parts
        if len(parts) != 2 or parts[0] != "checkpoints" or parts[1] in {"", ".", ".."}:
            raise RuntimeError(f"unsafe {label} checkpoint identity")
        checkpoint = root / parts[0] / parts[1]
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise RuntimeError(f"{label} checkpoint must be a real directory")
        if self._manifest_sha(checkpoint) != state.get("manifest_sha256"):
            raise RuntimeError(f"{label} checkpoint manifest hash mismatch")
        manifest, contents = self.stack.read_snapshot(checkpoint)
        if manifest["snapshot_id"] != state.get("snapshot_id"):
            raise RuntimeError(f"{label} pointer/checkpoint identity mismatch")
        selected_latency = self._positive_latency(state.get("selected_latency_ms"))
        metadata = manifest.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("selection_kind") != state.get("selection_kind")
            or metadata.get("selected_latency_ms") != selected_latency
            or manifest.get("source_binding") != state.get("selected_binding")
        ):
            raise RuntimeError(f"{label} pointer/checkpoint metadata mismatch")
        return state, checkpoint, manifest, contents

    def _read_best(self) -> tuple[dict, Path, dict, dict[str, str | None]] | None:
        return self._read_pointer("best.json", label="global-best")

    def _read_fallback(self) -> tuple[dict, Path, dict, dict[str, str | None]] | None:
        loaded = self._read_pointer(
            "fallback.json", label="naive/imported fallback"
        )
        if loaded is None:
            return None
        state, _checkpoint, _manifest, contents = loaded
        if (
            state.get("pointer_role") != "naive_imported_fallback"
            or state.get("selection_kind") != "baseline"
            or contents.get("L0") != self.task.naive_relax_source()
            or contents.get("T12") is not None
            or contents.get("L2") is not None
        ):
            raise RuntimeError("naive/imported fallback identity mismatch")
        return loaded

    def _initialize_fallback_pointer(self, state: dict) -> None:
        loaded = self._read_fallback()
        if loaded is not None:
            if loaded[0].get("snapshot_id") != state.get("snapshot_id"):
                raise RuntimeError("immutable fallback differs from baseline snapshot")
            return
        fallback = {**state, "pointer_role": "naive_imported_fallback"}
        self._write_pointer_state("fallback.json", fallback, replace=False)

    def _seal_current_stack(
        self,
        *,
        selected_latency_ms: float,
        selection_kind: str,
        metadata: dict,
    ) -> dict:
        root = self._prepare_global_best_root()
        checkpoint, manifest = self.stack.seal_snapshot(
            root / "checkpoints",
            metadata={
                "selection_kind": selection_kind,
                "selected_latency_ms": selected_latency_ms,
                **metadata,
                "branch_id": self.stack.active_branch_id(),
            },
        )
        state = {
            "schema_version": GLOBAL_BEST_SCHEMA_VERSION,
            "snapshot_id": manifest["snapshot_id"],
            "checkpoint": f"checkpoints/{checkpoint.name}",
            "manifest_sha256": self._manifest_sha(checkpoint),
            "selected_at": time.time(),
            "selection_kind": selection_kind,
            "selected_latency_ms": selected_latency_ms,
            "selected_binding": self.stack.binding().to_dict(),
            "branch_id": self.stack.active_branch_id(),
            "selection": metadata,
        }
        self._write_best_state(state)
        return state

    def initialize_global_best(self, baseline: MeasureResult) -> dict:
        """Seal the measured baseline once, or validate a recovered best.

        A resumed run must not relabel its possibly exploratory current stack
        as best merely because it is the first stack measured by this process.
        """

        with self._lock:
            recovered = self._read_best()
            if recovered is not None:
                state, _, _, _ = recovered
                fallback = self._read_fallback()
                if fallback is None:
                    if state.get("selection_kind") != "baseline":
                        raise RuntimeError(
                            "recovered promoted best lacks an immutable naive/imported fallback"
                        )
                    self._initialize_fallback_pointer(state)
                self.log(
                    "global_best_recovered",
                    snapshot_id=state["snapshot_id"],
                    selected_latency_ms=state["selected_latency_ms"],
                    selected_binding=state["selected_binding"],
                    current_binding=self.stack.binding().to_dict(),
                )
                selected_node = (state.get("selection") or {}).get("promoted_node")
                if isinstance(selected_node, str):
                    try:
                        self.operator_tree.mark_trunk(
                            selected_node,
                            latency_ms=state["selected_latency_ms"],
                        )
                    except KeyError:
                        # Runs created before the full-tree index retain their
                        # global-best authority; bootstrap migration starts a
                        # new tree at the current checkout without fabricating
                        # historical nodes.
                        pass
                return {"initialized": False, "recovered": True, **state}
            baseline = self._validate_timing_result(baseline, paired=False)
            if not baseline.ok:
                raise RuntimeError("cannot initialize global best from an unqualified baseline")
            latency = self._positive_latency(baseline.public.get("latency_ms"))
            state = self._seal_current_stack(
                selected_latency_ms=latency,
                selection_kind="baseline",
                metadata={"measurement_binding": baseline.public.get("binding")},
            )
            self._initialize_fallback_pointer(state)
            self.log(
                "global_best_initialized",
                snapshot_id=state["snapshot_id"],
                selected_latency_ms=latency,
                selected_binding=state["selected_binding"],
            )
            tree_state = self.operator_tree.initialize()
            self.operator_tree.mark_trunk(
                tree_state["root_node_id"], latency_ms=latency
            )
            return {"initialized": True, "recovered": False, **state}

    def global_best_state(self) -> dict | None:
        """Return verified global-best metadata without exposing stack source."""

        with self._lock:
            loaded = self._read_best()
            return None if loaded is None else dict(loaded[0])

    @staticmethod
    def _validate_timing_result(result: MeasureResult, *, paired: bool) -> MeasureResult:
        """Recompute admission from raw samples, including recovered results."""
        raw = result.trusted.get("raw")
        if not isinstance(raw, dict):
            raw = result.public
        quality = assess_timing(raw, paired=paired)
        qualified = (result.ok and raw.get("correct") is True
                     and (not paired or raw.get("reference_correct") is True)
                     and quality["valid"] and quality["stable"])
        return MeasureResult(
            ok=qualified, tier=result.tier, layer=result.layer,
            public={**result.public, "pass": qualified, "measurement_quality": quality,
                    **({} if qualified else {"diagnostic": {
                        "fault_stage": "timing_qualification",
                        "errors": quality["errors"],
                        "required_rcv": "<= 0.05",
                    }})},
            trusted=dict(result.trusted),
        )

    @staticmethod
    def _gain_threshold(vs_base: dict) -> tuple[float, float]:
        ratio = float(vs_base["median_ratio"])
        if ratio <= 0 or not math.isfinite(ratio):
            raise ValueError("paired global-best statistics are invalid")
        return ratio, PAIRED_GAIN_FLOOR

    def _snapshot_reference(self, contents: dict[str, str | None]) -> dict:
        module_source = contents.get("L1")
        if module_source is None:
            raise RuntimeError("global-best checkpoint has no executable L1 module")
        patch_source = contents.get("L2")
        t12 = contents.get("T12")
        if patch_source is not None and t12 is None:
            raise RuntimeError("global-best L2 patch has no bound T12 source")
        from .program_state import sha256_text
        return {
            "module_source": module_source,
            "patch_source": patch_source,
            "t12_sha": sha256_text(t12) if patch_source is not None else None,
        }

    def _consider_current_global_best(self, *, node: str, layer: str) -> dict:
        loaded = self._read_best()
        if loaded is None:
            self.log("global_best_not_evaluated", node=node, layer=layer,
                     reason="baseline_not_sealed")
            return {"evaluated": False, "updated": False,
                    "reason": "baseline_not_sealed"}
        previous, _, _, best_contents = loaded
        self.ensure_derived()
        patch = self.stack.read("L2")
        if patch is not None:
            self.ensure_t12()
        measurement = self.service.t1(
            self.final_module_source(),
            layer="stack_global_best",
            binding=self.stack.binding().to_dict(),
            patch_source=patch,
            t12_sha=self.stack.sha("T12") if patch is not None else None,
            reference=self._snapshot_reference(best_contents),
        )
        measurement = self._validate_timing_result(measurement, paired=True)
        vs_base = measurement.public.get("vs_base")
        if not measurement.ok or not isinstance(vs_base, dict):
            self.log(
                "global_best_evaluation",
                node=node,
                layer=layer,
                ok=measurement.ok,
                updated=False,
                reason="paired_measurement_failed",
                diagnostic=measurement.public.get("diagnostic"),
            )
            return {
                "evaluated": True,
                "updated": False,
                "reason": "paired_measurement_failed",
            }
        try:
            ratio, threshold = self._gain_threshold(vs_base)
            latency = self._positive_latency(measurement.public.get("latency_ms"))
            reference_latency = self._positive_latency(vs_base.get("base_latency_ms"))
        except (KeyError, TypeError, ValueError) as error:
            self.log(
                "global_best_evaluation",
                node=node,
                layer=layer,
                ok=False,
                updated=False,
                reason="invalid_paired_statistics",
            )
            return {"evaluated": True, "updated": False,
                    "reason": f"invalid_paired_statistics: {type(error).__name__}"}

        # Match the layer-local promotion gate: equality clears the bound;
        # only a ratio above it is rejected.
        improved = ratio <= 1.0 - threshold
        state = previous
        if improved:
            state = self._seal_current_stack(
                selected_latency_ms=latency,
                selection_kind="promotion",
                metadata={
                    "promoted_node": node,
                    "promoted_layer": layer,
                    "previous_snapshot_id": previous["snapshot_id"],
                    "median_ratio": ratio,
                    "ratio_cv": float(vs_base.get("ratio_cv") or 0.0),
                    "pairs": max(1, int(vs_base.get("pairs") or 1)),
                    "noise_threshold": threshold,
                    "reference_latency_ms": reference_latency,
                },
            )
        self.log(
            "global_best_evaluation",
            node=node,
            layer=layer,
            ok=True,
            updated=improved,
            snapshot_id=state["snapshot_id"],
            candidate_latency_ms=latency,
            reference_latency_ms=reference_latency,
            median_ratio=ratio,
            ratio_cv=vs_base.get("ratio_cv"),
            pairs=vs_base.get("pairs"),
            noise_threshold=threshold,
        )
        return {
            "evaluated": True,
            "updated": improved,
            "snapshot_id": state["snapshot_id"],
            "candidate_latency_ms": latency,
            "reference_latency_ms": reference_latency,
            "median_ratio": ratio,
            "noise_threshold": threshold,
        }

    # -- workspace/base capabilities -----------------------------------------
    @staticmethod
    def _canonical_json(value: dict) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def _workspace_binding_payload(self, layer: str) -> dict:
        if layer not in AGENT_OF_LAYER:
            raise ValueError(f"unknown workspace layer: {layer}")
        # T01 and T12 are derived slots not represented in StackBinding.  Sign
        # them as well so a lower-layer request is tied to the actual projected
        # input rather than only to its epoch.
        projected = {
            "L0": {"L0": self.stack.sha("L0")},
            "L1": {
                "T01": self.stack.sha("T01"),
                "L1": self.stack.sha("L1"),
            },
            "L2": {
                # Local L1 contraction can change the host call graph even
                # when device-source bytes happen to be identical (or absent).
                "L1": self.stack.sha("L1"),
                "T12": self.stack.sha("T12"),
                "L2": self.stack.sha("L2"),
            },
        }[layer]
        return {
            "schema_version": 2,
            "layer": layer,
            "branch_id": self.stack.active_branch_id(),
            "binding": self.stack.binding().to_dict(),
            "projected_inputs": projected,
        }

    def _workspace_projection_matches(
        self, layer: str, payload: object
    ) -> tuple[bool, str]:
        """Check only dependencies owned or consumed by ``layer``.

        The full binding revision is deliberately not compared here.  A lower
        layer promotion changes the observation epoch but does not invalidate
        an L0 graph projection (and an L2-only change does not invalidate L1).
        Branch identity plus exact projected hashes remain mandatory.
        """

        if not isinstance(payload, dict):
            return False, "workspace binding capability payload is malformed"
        current = self._workspace_binding_payload(layer)
        if payload.get("schema_version") != current["schema_version"]:
            return False, "workspace binding capability schema differs"
        if payload.get("layer") != layer:
            return False, "workspace binding capability targets another layer"
        if payload.get("branch_id") != current["branch_id"]:
            return False, "stale graph branch; refresh the workspace projection"
        if payload.get("projected_inputs") != current["projected_inputs"]:
            return False, "stale workspace inputs; refresh this layer projection"
        return True, ""

    def workspace_binding_envelope(self, layer: str) -> dict:
        """Issue a signed identity for one workspace projection."""
        with self._lock:
            payload = self._workspace_binding_payload(layer)
            signature = hmac.new(
                self._workspace_binding_key,
                self._canonical_json(payload),
                hashlib.sha256,
            ).hexdigest()
            return {"payload": payload, "signature": signature}

    def validate_workspace_binding(self, layer: str, envelope: object) -> tuple[bool, str]:
        """Fail closed unless a request targets the exact current projection."""
        if not isinstance(envelope, dict):
            return False, "workspace binding capability is missing"
        payload = envelope.get("payload")
        signature = envelope.get("signature")
        if not isinstance(payload, dict) or not isinstance(signature, str):
            return False, "workspace binding capability is malformed"
        try:
            expected_signature = hmac.new(
                self._workspace_binding_key,
                self._canonical_json(payload),
                hashlib.sha256,
            ).hexdigest()
        except (TypeError, ValueError):
            return False, "workspace binding capability is malformed"
        if not hmac.compare_digest(signature, expected_signature):
            return False, "workspace binding capability is invalid"
        with self._lock:
            return self._workspace_projection_matches(layer, payload)

    def enable_admission_authority(self, roles: dict[str, str]) -> None:
        """Enable once before sessions start; never re-enable after shutdown."""
        with self._lock:
            if self.turn_authority_enabled or self._turn_admissions_closed.is_set():
                raise RuntimeError("turn authority is already configured or closed")
            if not roles or any(layer not in AGENT_OF_LAYER for layer in roles.values()):
                raise ValueError("turn authority requires known optimizer layers")
            self._turn_authority_roles = dict(roles)
            self.turn_authority_enabled = True

    def issue_turn_authority(self, agent: str, layer: str, turn_id: str,
                             purpose: str = "optimization") -> dict:
        """Activate one fresh bearer capability under the admission lock."""
        with self._lock:
            if not self.turn_authority_enabled or self._turn_admissions_closed.is_set():
                raise RuntimeError("turn admissions are disabled or closed")
            if self._turn_authority_roles.get(agent) != layer:
                raise ValueError("turn authority role/layer mismatch")
            if purpose not in {"optimization", "discussion"} or not isinstance(turn_id, str) or not turn_id:
                raise ValueError("invalid turn authority purpose or identity")
            if purpose == "discussion" and layer != "L0":
                raise ValueError("discussion pilot is restricted to L0")
            if agent in self._turn_authorities:
                raise RuntimeError("prior turn authority is still active")
            payload = {"schema_version": "leka.turn_authority.v1", "agent": agent,
                       "layer": layer, "turn_id": turn_id, "purpose": purpose,
                       "nonce": secrets.token_hex(32),
                       "workspace_projection": self._workspace_binding_payload(layer)}
            envelope = {"payload": payload, "signature": hmac.new(
                self._workspace_binding_key, self._canonical_json(payload), hashlib.sha256).hexdigest()}
            self._turn_authorities[agent] = json.loads(self._canonical_json(envelope))
            return envelope

    def validate_turn_authority(self, agent: str, layer: str, envelope: object,
                                *, require_optimization: bool = True,
                                turn_id: str | None = None) -> tuple[bool, str]:
        """Caller keeps the Pipeline lock through any authorized side effect.

        Authority lifetime and workspace freshness are separate: an active
        turn can promote and thereby change its source projection, but its
        nonce must still be revocable without comparing the new stack.
        """
        if not self.turn_authority_enabled:
            return True, ""
        with self._lock:
            if self._turn_admissions_closed.is_set():
                return False, "turn admissions are closed"
            if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
                return False, "turn authority is missing or malformed"
            payload, signature = envelope["payload"], envelope.get("signature")
            if (not isinstance(signature, str) or len(signature) != 64
                    or any(char not in "0123456789abcdef" for char in signature)):
                return False, "turn authority signature is malformed"
            try:
                expected = hmac.new(self._workspace_binding_key, self._canonical_json(payload),
                                    hashlib.sha256).hexdigest()
            except (TypeError, ValueError):
                return False, "turn authority payload is malformed"
            if not hmac.compare_digest(signature, expected):
                return False, "turn authority signature is invalid"
            if (self._turn_authority_roles.get(agent) != layer or payload.get("agent") != agent
                    or payload.get("layer") != layer or (turn_id is not None and payload.get("turn_id") != turn_id)):
                return False, "turn authority belongs to another role or turn"
            if envelope != self._turn_authorities.get(agent):
                return False, "turn authority nonce is inactive or superseded"
            if require_optimization and payload.get("purpose") != "optimization":
                return False, "discussion turn cannot measure, submit, promote or checkout"
            return True, ""

    def end_turn_authority(self, agent: str, envelope: object) -> bool:
        """Compare-and-revoke, including after a successful same-turn promotion."""
        with self._lock:
            if envelope is not None and self._turn_authorities.get(agent) == envelope:
                del self._turn_authorities[agent]
                return True
            return False

    def close_turn_admissions(self) -> None:
        # Nonblocking so teardown can first cancel/join an in-flight device
        # operation holding the Pipeline lock. No later validation may pass.
        self._turn_admissions_closed.set()

    # -- ledger ---------------------------------------------------------------
    def log(self, kind: str, **payload) -> None:
        entry = {"t": time.time(), "kind": kind, **payload}
        with (self.run_dir / "ledger.jsonl").open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    # -- lifecycle ------------------------------------------------------------
    def bootstrap(self) -> None:
        """Fresh run: L0 default (naive relax) + derived slots; repo init."""
        self.repo.ensure()
        if self.stack.read("L0") is None:
            self.stack.write("L0", self.task.naive_relax_source())
            self.log("stack_default", slot="L0")
        self.ensure_derived()
        branch_id = self.stack.initialize_root_branch()
        self._seal_branch_root(
            branch_id=branch_id,
            parent_branch_id=self.stack.branch_parent(branch_id),
            promoted_node=None,
        )
        self.operator_tree.initialize()
        self.repo.commit("bootstrap: default stack")

    def ensure_derived(self) -> None:
        """Re-run T01 whenever derived slots are missing (fresh or invalidated)."""
        if self.stack.read("T01") is not None and self.stack.read("L1") is not None:
            return
        r = self.service.t0("L0", self.stack.read("L0"))
        if not r.ok:
            self.log("t01_failed", diagnostic=r.public.get("diagnostic"))
            raise RuntimeError(f"T01 failed on current L0: {r.trusted.get('fault_stage')}")
        self.stack.write("T01", r.trusted["canonical"])
        if self.stack.read("L1") is None:
            self.stack.write("L1", r.trusted["scheduled_default"])
            self.log(
                "stack_default",
                slot="L1",
                note=(
                    "visible conservative executable seed derived after T01; "
                    "L1 owns all replacement scheduling"
                ),
                scheduler=r.trusted.get("l1_seed_scheduler"),
            )

    def final_module_source(self) -> str:
        """Composition: the L1 slot IS the full module (relax main + scheduled TIR)."""
        return self.stack.read("L1")

    def ensure_t12(self) -> None:
        """Extract the generated device source for the current L1 base (L2's input)."""
        with self._lock:
            self.ensure_derived()
            if self.stack.read("T12") is not None:
                return
            r = self.service.extract(self.final_module_source(),
                                     self.stack.binding().to_dict())
            if not r.ok:
                self.log("t12_failed", diagnostic=r.public.get("diagnostic"))
                raise RuntimeError("T12 extraction failed on current L1 base")
            self.stack.write("T12", r.trusted["generated_source"])
            self.log("t12_extracted", sha=self.stack.sha("T12"))

    # -- candidate flow --------------------------------------------------------
    def submit_candidate(
        self,
        layer: str,
        source: str,
        *,
        note: str = "",
        expected_workspace_payload: dict | None = None,
    ) -> tuple[Candidate, MeasureResult]:
      with self._lock:
        if layer == "L2":
            # T12 is part of L2's signed dependency projection. Materialize it
            # before capturing the payload so the candidate does not stale
            # itself merely by asking for its generated source.
            self.ensure_t12()
        if expected_workspace_payload is not None:
            projection_ok, projection_error = self._workspace_projection_matches(
                layer, expected_workspace_payload
            )
            if not projection_ok:
                raise RuntimeError(projection_error)
        current_payload = self._workspace_binding_payload(layer)
        projected_inputs = dict(current_payload["projected_inputs"])
        branch_id = str(current_payload["branch_id"])
        binding = self.stack.binding().to_dict()
        node_id = f"n{uuid.uuid4().hex[:8]}"
        cdir = self.run_dir / "candidates" / node_id
        cdir.mkdir(parents=True)
        (cdir / "source.py").write_text(source)
        cand = Candidate(
            node_id,
            layer,
            binding,
            cdir,
            projected_inputs=projected_inputs,
            branch_id=branch_id,
        )
        self.log(
            "candidate",
            node=node_id,
            layer=layer,
            binding=binding,
            projected_inputs=projected_inputs,
            branch_id=branch_id,
            note=note,
        )

        t0 = self.service.t0(layer, source, canonical_text=self.stack.read("T01"),
                             t12_text=self.stack.read("T12"))
        (cdir / "t0.json").write_text(json.dumps(t0.public, indent=2))
        if layer == "L1" and t0.ok:
            # Bind local graph authority to the exact bytes that go to T1.
            # Candidate workspaces never own this driver-private record.
            contract = {
                "schema": "l1_local_joint_v1",
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "canonical_sha256": self.stack.sha("T01"),
                "local_rewrite": t0.trusted.get("local_rewrite"),
            }
            (cdir / "l1_contract.json").write_text(json.dumps(contract, indent=2))
            self.log("l1_contract", node=node_id, **contract)
        if not t0.ok:
            self.log("t0_fail", node=node_id, layer=layer)
            self.operator_tree.record_candidate(
                node_id=node_id,
                layer=layer,
                source=source,
                note=note,
                branch_id=branch_id,
                binding=binding,
                projected_inputs=projected_inputs,
                t0_public=t0.public,
                t0_trusted=t0.trusted,
                t1_public=None,
                t1_ok=False,
            )
            return cand, t0

        reference = self._incumbent_reference(layer)
        if layer == "L0":
            # measure the candidate with its own derived dlight default below
            t1 = self.service.t1(t0.trusted["scheduled_default"], layer=layer, binding=binding,
                                 reference=reference)
        elif layer == "L1":
            t1 = self.service.t1(source, layer=layer, binding=binding, reference=reference)
        else:  # L2: current L1 base + the candidate patch, bound to the T12 sha
            t1 = self.service.t1(self.final_module_source(), layer=layer, binding=binding,
                                 patch_source=source, t12_sha=self.stack.sha("T12"),
                                 reference=reference)
        t1 = self._validate_timing_result(t1, paired=reference is not None)
        (cdir / "result.json").write_text(json.dumps(t1.public, indent=2))
        self.log("t1", node=node_id, layer=layer, ok=t1.ok,
                 latency_ms=t1.public.get("latency_ms"))
        self.operator_tree.record_candidate(
            node_id=node_id,
            layer=layer,
            source=source,
            note=note,
            branch_id=branch_id,
            binding=binding,
            projected_inputs=projected_inputs,
            t0_public=t0.public,
            t0_trusted=t0.trusted,
            t1_public=t1.public,
            t1_ok=t1.ok,
        )
        return cand, t1

    def _incumbent_reference(self, candidate_layer: str) -> dict | None:
        """Return a layer-matched incumbent for interleaved paired timing.

        An upper-layer candidate is measured before lower-layer optimizations
        are re-applied.  Comparing it with the fully optimized current stack
        would systematically close L0/L1 re-entry.  Pair each candidate with
        the incumbent at the same lowering boundary instead:

        * L0: current L0 lowered through its own default scheduler;
        * L1: current L1 without an L2 source patch;
        * L2: current L1 plus the current L2 patch.
        """
        if candidate_layer == "L0":
            current_l0 = self.stack.read("L0")
            if current_l0 is None:
                return None
            lowered = self.service.t0("L0", current_l0)
            if not lowered.ok:
                raise RuntimeError("unable to derive the layer-matched L0 incumbent")
            return {
                "module_source": lowered.trusted["scheduled_default"],
                "patch_source": None,
                "t12_sha": None,
            }
        module_source = self.final_module_source()
        if module_source is None:
            return None
        if candidate_layer == "L1":
            return {"module_source": module_source, "patch_source": None,
                    "t12_sha": None}
        if candidate_layer != "L2":
            raise ValueError(f"unknown candidate layer: {candidate_layer}")
        patch = self.stack.read("L2")
        return {"module_source": module_source, "patch_source": patch,
                "t12_sha": self.stack.sha("T12") if patch is not None else None}

    def _audit_candidate(self, cand: Candidate) -> MeasureResult:
        """Rebuild the measured candidate and run hidden, untimed checks."""
        if cand.layer == "L0":
            lowered = self.service.t0("L0", cand.source)
            if not lowered.ok:
                return MeasureResult(
                    ok=False,
                    tier="audit",
                    layer=cand.layer,
                    public={
                        "pass": False,
                        "correct": False,
                        "checks_completed": 0,
                        "diagnostic": lowered.public.get("diagnostic"),
                    },
                )
            return self.service.audit(
                lowered.trusted["scheduled_default"],
                layer=cand.layer,
                binding=cand.binding,
            )
        if cand.layer == "L1":
            return self.service.audit(
                cand.source,
                layer=cand.layer,
                binding=cand.binding,
            )
        if cand.layer == "L2":
            self.ensure_t12()
            module_source = self.final_module_source()
            t12_sha = self.stack.sha("T12")
            audited = self.service.audit(
                module_source,
                layer=cand.layer,
                binding=cand.binding,
                patch_source=cand.source,
                t12_sha=t12_sha,
            )
            # The dependency binding intentionally describes the pre-change
            # stack. An absent old L2 does not mean the supplied patch was
            # absent from this audit. Bind the actual call inputs separately.
            audited.public = dict(audited.public)
            audited.public["candidate_execution"] = {
                "schema": "leka.l2_audit_execution.v1",
                "binding_role": "pre_change_dependency_stack_not_candidate_patch",
                "module_source_sha256": hashlib.sha256(module_source.encode("utf-8")).hexdigest(),
                "patch_source_sha256": hashlib.sha256(cand.source.encode("utf-8")).hexdigest(),
                "t12_sha256": t12_sha,
                "audit_passed": audited.ok,
                "explanation": "The audit receives the complete module plus the submitted L2 patch. binding.l2_sha identifies the prior stack; it is not the submitted patch hash.",
            }
            return audited
        raise ValueError(f"unknown candidate layer: {cand.layer}")

    def candidate_by_id(self, node_id: str) -> Candidate | None:
        """Recover one candidate and its dependency projection from the ledger."""

        cdir = self.run_dir / "candidates" / node_id
        if not cdir.exists():
            return None
        layer = None
        binding = None
        projected_inputs = None
        branch_id = None
        for line in (self.run_dir / "ledger.jsonl").read_text().splitlines():
            entry = json.loads(line)
            if entry.get("kind") == "candidate" and entry.get("node") == node_id:
                layer = entry["layer"]
                binding = entry["binding"]
                projected_inputs = entry.get("projected_inputs")
                branch_id = entry.get("branch_id")
        if layer is None:
            return None
        return Candidate(
            node_id,
            layer,
            binding,
            cdir,
            projected_inputs=(
                dict(projected_inputs) if isinstance(projected_inputs, dict) else None
            ),
            branch_id=branch_id if isinstance(branch_id, str) else None,
        )

    def promote_by_id(self, node_id: str) -> dict:
        """Recover a candidate from the ledger and promote it (shim/reply path)."""

        candidate = self.candidate_by_id(node_id)
        if candidate is None:
            return {"promoted": False, "reason": f"node {node_id} not found in ledger"}
        return self.promote(candidate)

    def _candidate_dependency_status(self, cand: Candidate) -> tuple[bool, str]:
        """Return whether the candidate's exact layer dependencies are current."""

        current = self.stack.binding().to_dict()
        if cand.projected_inputs is None or cand.branch_id is None:
            # Backwards-compatible fail-closed handling for pre-v2 ledgers.
            if cand.binding.get("epoch") != current["epoch"]:
                return False, "stale_epoch"
            return True, ""
        expected = self._workspace_binding_payload(cand.layer)
        if cand.branch_id != expected["branch_id"]:
            # An already measured L0 proposal is a prospective child of the
            # graph branch on which it was authored.  Promoting a sibling L0
            # branch must not destroy that independent proposal: it can later
            # become another child as long as its signed L0 base still equals
            # the immutable recorded root of its parent branch. Descendant
            # candidates remain branch-local and therefore fail closed here.
            if cand.layer == "L0":
                parent_root = self.stack.branch_root_l0_sha(cand.branch_id)
                candidate_base = cand.projected_inputs.get("L0")
                if parent_root is not None and candidate_base == parent_root:
                    return True, ""
            return False, "stale_branch"
        if cand.projected_inputs != expected["projected_inputs"]:
            return False, "stale_projection"
        return True, ""

    # -- promotion --------------------------------------------------------------
    def promote(self, cand: Candidate) -> dict:
      with self._lock:
        self.operation_control.checkpoint()
        current = self.stack.binding().to_dict()
        dependencies_current, stale_reason = self._candidate_dependency_status(cand)
        if not dependencies_current:
            self.log(
                "promote_rejected",
                node=cand.node_id,
                reason=stale_reason,
                candidate_epoch=cand.binding.get("epoch"),
                current_epoch=current["epoch"],
                candidate_branch_id=cand.branch_id,
                current_branch_id=self.stack.active_branch_id(),
                candidate_projected_inputs=cand.projected_inputs,
                current_projected_inputs=self._workspace_binding_payload(cand.layer)[
                    "projected_inputs"
                ],
            )
            return {
                "promoted": False,
                "reason": (
                    f"{stale_reason}: candidate dependencies changed after measurement; "
                    "the measured result is retained, but promotion requires a fresh branch projection"
                ),
            }
        result = cand.result()
        if not result or not result.get("pass"):
            self.log("promote_rejected", node=cand.node_id, reason="no_passing_t1")
            return {"promoted": False, "reason": "no passing T1 result for this candidate"}
        quality = assess_timing(result, paired=True)
        if not (quality["valid"] and quality["stable"]
                and result.get("correct") is True and result.get("reference_correct") is True):
            self.log("promote_rejected", node=cand.node_id, reason="timing_not_qualified")
            return {"promoted": False, "reason": "promotion requires valid paired samples with all rCV values <= 0.05"}
        from .program_state import sha256_text
        if sha256_text(cand.source) == self.stack.sha(cand.layer):
            self.log("promote_rejected", node=cand.node_id, reason="no_change")
            return {"promoted": False, "reason": "candidate is byte-identical to the "
                    "current slot content; promoting a no-change candidate only resets "
                    "the layers below it"}
        local_rewrite = None
        if cand.layer == "L1":
            proof_path = cand.dir / "l1_contract.json"
            try:
                proof = json.loads(proof_path.read_text())
                proof_matches = (
                    proof.get("schema") == "l1_local_joint_v1"
                    and proof.get("source_sha256") == sha256_text(cand.source)
                    and proof.get("canonical_sha256") == self.stack.sha("T01")
                )
            except (OSError, ValueError, AttributeError):
                proof_matches = False
            if not proof_matches:
                self.log("promote_rejected", node=cand.node_id,
                         reason="l1_contract_identity_changed")
                return {"promoted": False, "reason":
                        "L1 contract/source identity is missing or changed; re-submit and measure"}
            # Re-run the structural check on the exact measured L1 source.
            # A T1 pass can never grant broad Relax graph editing authority.
            contract_check = self.service.t0(
                "L1", cand.source, canonical_text=self.stack.read("T01")
            )
            if (not contract_check.ok or
                    contract_check.trusted.get("local_rewrite") != proof.get("local_rewrite")):
                self.log("promote_rejected", node=cand.node_id,
                         reason="l1_contract_revalidation_failed")
                return {"promoted": False, "reason": "L1 structural contract revalidation failed"}
            local_rewrite = contract_check.trusted.get("local_rewrite")
        # Joint L1 proposals always require paired evidence, including in a
        # driver configuration that allows unpaired scheduling.
        if getattr(self, "promote_requires_gain", False) or local_rewrite:
            vs = result.get("vs_base")
            if not vs:
                self.log("promote_rejected", node=cand.node_id, reason="no_pair_evidence")
                return {"promoted": False, "reason": "gain gate: no paired ratio evidence "
                        "on this candidate (re-measure against the current base)"}
            try:
                ratio, threshold = self._gain_threshold(vs)
            except (KeyError, TypeError, ValueError):
                self.log(
                    "promote_rejected",
                    node=cand.node_id,
                    reason="invalid_paired_statistics",
                    vs_base=vs,
                )
                return {
                    "promoted": False,
                    "reason": "gain gate: paired median ratio is invalid",
                }
            if ratio > 1.0 - threshold:
                self.log("promote_rejected", node=cand.node_id, reason="gain_below_noise",
                         vs_base=vs, threshold=threshold)
                return {"promoted": False, "reason":
                        f"gain gate: paired median ratio {ratio:.4f} does not clear "
                        f"the fixed 0.5% improvement floor ({1.0 - threshold:.4f})"}

        # The ordinary T1 uses one fixed hidden input set so repeated timing is
        # comparable.  It is not sufficient authority for promotion: a
        # candidate could accidentally (or deliberately) specialize to those
        # values. Re-execute the configured multi-seed correctness-only audit
        # immediately before reviewer approval and stack mutation. The ordinary
        # profile uses seeds 0/1/2 rather than newly sampled seed identities.
        try:
            audit = self._audit_candidate(cand)
        except OperationCancelled:
            raise
        except Exception:
            self.log("promotion_audit", node=cand.node_id, layer=cand.layer,
                     ok=False, checks_completed=0, reason="audit_infrastructure_error")
            self.log("promote_rejected", node=cand.node_id,
                     reason="correctness_audit_incomplete")
            return {"promoted": False,
                    "reason": "hidden correctness audit could not complete; promotion "
                              "fails closed"}
        self.log(
            "promotion_audit",
            node=cand.node_id,
            layer=cand.layer,
            ok=audit.ok,
            checks_completed=audit.public.get("checks_completed", 0),
            diagnostic=audit.public.get("diagnostic"),
        )
        if not audit.ok:
            self.log("promote_rejected", node=cand.node_id,
                     reason="correctness_audit_failed")
            return {"promoted": False,
                    "reason": "candidate failed hidden multi-seed correctness audit"}

        review_layers = getattr(self, "review_layers", {"L0", "L1", "L2"})
        if self.reviewer is not None and cand.layer in review_layers:
            # Give the reviewer the already-completed independent audit.  The
            # benchmark's tolerance checks, rather than bitwise expression
            # identity, are the numerical authority for every layer.
            review_result = dict(result)
            review_result["promotion_audit"] = audit.public
            if local_rewrite:
                review_result["l1_local_joint_contract"] = {
                    "schema": "l1_local_joint_v1",
                    "canonical_sha256": self.stack.sha("T01"),
                    "local_rewrite": local_rewrite,
                    "structural_gate_passed": True,
                }
            semantic_context = None
            if cand.layer == "L0" or local_rewrite:
                semantic_projection = getattr(self.task, "pytorch_semantics", None)
                if callable(semantic_projection):
                    semantic_context = semantic_projection()
            verdict = self.reviewer.review(
                layer=cand.layer, source=cand.source, result=review_result,
                task_facts=self.task.public_facts(),
                reference_source=self.stack.read("T12" if cand.layer == "L2" else cand.layer),
                semantic_context=semantic_context)
            self.log("review", node=cand.node_id, layer=cand.layer, verdict=verdict)
            if not verdict.get("approved"):
                self.log("promote_rejected", node=cand.node_id, reason="review_rejected",
                         review=verdict)
                return {"promoted": False, "reason": f"review rejected: {verdict.get('reasons')}"}

        parent_branch_id = (
            cand.branch_id
            if cand.layer == "L0" and cand.branch_id is not None
            else self.stack.active_branch_id()
        )
        self.stack.write(cand.layer, cand.source)
        cleared = self.stack.invalidate_below(cand.layer)
        if cand.layer == "L0":
            self.stack.activate_branch(
                f"b-{cand.node_id}", parent_branch_id=parent_branch_id
            )
        self.log("promoted", node=cand.node_id, layer=cand.layer, cleared=cleared,
                 new_epoch=self.stack.epoch,
                 branch_id=self.stack.active_branch_id(),
                 parent_branch_id=parent_branch_id,
                 local_rewrite=local_rewrite)
        self.ensure_derived()
        if cand.layer == "L0":
            self._seal_branch_root(
                branch_id=self.stack.active_branch_id(),
                parent_branch_id=parent_branch_id,
                promoted_node=cand.node_id,
            )
        self.log(
            "branch_leaf",
            branch_id=self.stack.active_branch_id(),
            promoted_node=cand.node_id,
            promoted_layer=cand.layer,
            binding=self.stack.binding().to_dict(),
        )
        self.operator_tree.mark_promoted(cand.node_id)
        new_binding = self.stack.binding().to_dict()
        # A layer-local promotion is an exploration step, not automatically a
        # new deliverable.  Re-measure the *complete* current stack against the
        # sealed global best on the same device/input stream.  Infrastructure
        # failure preserves both the successful exploratory promotion and the
        # prior best checkpoint; it never silently relabels current as best.
        try:
            global_best = self._consider_current_global_best(
                node=cand.node_id, layer=cand.layer
            )
        except OperationCancelled:
            raise
        except Exception as error:
            self.log(
                "global_best_evaluation",
                node=cand.node_id,
                layer=cand.layer,
                ok=False,
                updated=False,
                reason="evaluation_infrastructure_error",
                error_type=type(error).__name__,
            )
            global_best = {
                "evaluated": False,
                "updated": False,
                "reason": "evaluation_infrastructure_error",
            }
        if global_best.get("updated"):
            self.operator_tree.mark_trunk(
                cand.node_id,
                latency_ms=global_best["candidate_latency_ms"],
            )
        self.operation_control.checkpoint()
        agent_of_layer = dict(AGENT_OF_LAYER)
        if hasattr(self.service, "cfg"):
            agent_of_layer["L2"] = self.service.cfg.get("l2_agent", "cuda_agent")
        # Every optimizer receives a fresh capability revision after every
        # promotion.  Only the promoted layer and its descendants have changed
        # projected artifacts; upper layers merely refresh their signed
        # observation binding and remain eligible to continue exploring.
        invalidated_layers = (cand.layer, *LAYERS_BELOW[cand.layer])
        for layer_name in ("L0", "L1", "L2"):
            agent = agent_of_layer.get(layer_name)
            if agent:
                self.mailbox.send(agent, "base_change", {
                    "promoted_layer": cand.layer, "promoted_node": cand.node_id,
                    "new_binding": new_binding,
                    "branch_id": self.stack.active_branch_id(),
                    "projection_changed": layer_name in invalidated_layers,
                    "note": (
                        "stack revision changed; refresh your signed workspace projection. "
                        "Only candidates whose branch or layer inputs changed are stale; "
                        "unaffected upper-layer candidates may finish and promote."
                    ),
                })
        self.operation_control.checkpoint()
        self.repo.commit(f"promote {cand.layer} {cand.node_id} epoch={self.stack.epoch}")
        return {"promoted": True, "new_binding": new_binding,
                "global_best": global_best}

    def activate_tree_node(self, node_id: str, *, requester_layer: str) -> dict:
        """Checkout a measured tree node for bounded cross-branch exploration.

        Activation never changes the globally best trunk.  It only restores a
        hash-sealed T1-passing snapshot and refreshes every resident optimizer
        capability; ordinary promotion/audit/review remain mandatory before a
        descendant can become a deliverable.
        """

        with self._lock:
            self.operation_control.checkpoint()
            before = self.stack.binding().to_dict()
            activated = self.operator_tree.activate(
                node_id, requester_layer=requester_layer
            )
            node = activated["node"]
            if node["layer"] == "L0":
                child = node["child_branch_id"]
                parent = node["branch_id"]
                self.stack.activate_branch(child, parent_branch_id=parent)
                self._seal_branch_root(
                    branch_id=child,
                    parent_branch_id=parent,
                    promoted_node=node_id,
                )
            self.ensure_derived()
            new_binding = self.stack.binding().to_dict()
            agent_of_layer = dict(AGENT_OF_LAYER)
            if hasattr(self.service, "cfg"):
                agent_of_layer["L2"] = self.service.cfg.get(
                    "l2_agent", "cuda_agent"
                )
            for layer_name in ("L0", "L1", "L2"):
                agent = agent_of_layer.get(layer_name)
                if agent:
                    self.mailbox.send(
                        agent,
                        "tree_checkout",
                        {
                            "requested_by": requester_layer,
                            "node_id": node_id,
                            "node_layer": node["layer"],
                            "old_binding": before,
                            "new_binding": new_binding,
                            "branch_id": self.stack.active_branch_id(),
                            "projection_changed": True,
                            "note": (
                                "A hash-sealed T1-passing tree node is now the exploratory "
                                "checkout. Re-read signed inputs. The global-best trunk did "
                                "not change."
                            ),
                        },
                    )
            self.log(
                "operator_tree_checkout",
                node=node_id,
                node_layer=node["layer"],
                requester_layer=requester_layer,
                old_binding=before,
                new_binding=new_binding,
                branch_id=self.stack.active_branch_id(),
            )
            return {
                "activated": True,
                "node_id": node_id,
                "node_layer": node["layer"],
                "new_binding": new_binding,
                "global_best_unchanged": True,
            }

    # -- current-stack measurement -------------------------------------------
    def measure_default(self, *, profile: bool = False) -> MeasureResult:
      with self._lock:
        self.ensure_derived()
        patch = self.stack.read("L2")
        t12_sha = self.stack.sha("T12") if patch is not None else None
        return self.service.t1(self.final_module_source(), layer="stack_default",
                               binding=self.stack.binding().to_dict(),
                               patch_source=patch, t12_sha=t12_sha, profile=profile)

    def _audit_current_stack(self, *, source: str) -> MeasureResult:
        """Requalify the exact restored stack on fresh, timing-free inputs."""

        self.ensure_derived()
        patch = self.stack.read("L2")
        if patch is not None:
            self.ensure_t12()
        result = self.service.audit(
            self.final_module_source(),
            layer="stack_final_requalification",
            binding=self.stack.binding().to_dict(),
            patch_source=patch,
            t12_sha=self.stack.sha("T12") if patch is not None else None,
        )
        self.log(
            "final_requalification",
            source=source,
            ok=result.ok,
            checks_completed=result.public.get("checks_completed", 0),
            binding=self.stack.binding().to_dict(),
            diagnostic=result.public.get("diagnostic"),
        )
        return result

    @staticmethod
    def _is_correctness_failure(result: MeasureResult) -> bool:
        """Distinguish candidate mismatch from transport/runner unavailability."""

        if result.ok:
            return False
        raw_results: list[dict] = []
        trusted_raw = result.trusted.get("raw")
        if isinstance(trusted_raw, dict):
            raw_results.append(trusted_raw)
        trusted_runs = result.trusted.get("runs")
        if isinstance(trusted_runs, list):
            raw_results.extend(
                item["raw"]
                for item in trusted_runs
                if isinstance(item, dict) and isinstance(item.get("raw"), dict)
            )
        if raw_results:
            return any(
                raw.get("correct") is False
                or str(raw.get("fault_stage") or "")
                in {"correctness", "run_correctness", "reference_correctness"}
                for raw in raw_results
            )
        # Test doubles and legacy services may expose only the public result.
        return result.public.get("correct") is False

    @staticmethod
    def _tag_final_result(
        result: MeasureResult,
        *,
        source: str,
        snapshot_id: str,
        requalification: MeasureResult,
        selection: dict | None = None,
    ) -> MeasureResult:
        result = Pipeline._validate_timing_result(result, paired=True)
        qualified = (result.ok and requalification.ok
                     and requalification.public.get("checks_completed", 0) >= 3)
        return MeasureResult(
            ok=qualified,
            tier=result.tier,
            layer=result.layer,
            public={
                **result.public,
                "pass": qualified,
                "final_source": source,
                "final_snapshot_id": snapshot_id,
                "requalification_checks": requalification.public.get(
                    "checks_completed", 0
                ),
                **({} if selection is None else {"final_selection": selection}),
            },
            trusted=dict(result.trusted),
        )

    def _measure_current_against_snapshot(
        self,
        contents: dict[str, str | None],
        *,
        layer: str,
        profile: bool,
    ) -> MeasureResult:
        """Replay current and immutable reference together on one device session."""

        self.ensure_derived()
        patch = self.stack.read("L2")
        if patch is not None:
            self.ensure_t12()
        return self.service.t1(
            self.final_module_source(),
            layer=layer,
            binding=self.stack.binding().to_dict(),
            patch_source=patch,
            t12_sha=self.stack.sha("T12") if patch is not None else None,
            profile=profile,
            reference=self._snapshot_reference(contents),
        )

    def finalize_global_best(self, *, profile: bool = False) -> MeasureResult:
        """Freshly requalify the sealed best, with one immutable safe fallback.

        Before restoration, the exploratory current stack is itself sealed as
        a non-selected private checkpoint and its binding is written to the
        append-only ledger.  This keeps exact recovery/debug evidence without
        allowing the final deliverable to regress to the last promotion.  A
        candidate correctness failure restores and requalifies the original
        naive/imported stack; transport failures do not trigger extra device
        work, and neither pointer nor historical checkpoint is overwritten.
        """

        with self._lock:
            loaded = self._read_best()
            if loaded is None:
                self.log("global_best_finalize_skipped", reason="baseline_not_sealed")
                measured = self.measure_default(profile=profile)
                return MeasureResult(
                    ok=False, tier=measured.tier, layer=measured.layer,
                    public={**measured.public, "pass": False,
                            "diagnostic": {"fault_stage": "final_qualification",
                                           "reason": "baseline_not_sealed"}},
                    trusted=dict(measured.trusted),
                )
            state, checkpoint, _, _ = loaded
            fallback_loaded = self._read_fallback()
            if fallback_loaded is None:
                raise RuntimeError(
                    "global-best finalization lacks an immutable naive/imported fallback"
                )
            exploratory_binding = self.stack.binding().to_dict()
            exploratory_snapshot_id = None
            try:
                exploratory_checkpoint, exploratory_manifest = self.stack.seal_snapshot(
                    self._prepare_global_best_root() / "checkpoints",
                    metadata={
                        "selection_kind": "exploratory_final",
                        "selected": False,
                        "best_snapshot_id": state["snapshot_id"],
                    },
                )
                exploratory_snapshot_id = exploratory_manifest["snapshot_id"]
                self.log(
                    "exploratory_final_checkpoint",
                    snapshot_id=exploratory_snapshot_id,
                    checkpoint=f"checkpoints/{exploratory_checkpoint.name}",
                    binding=exploratory_binding,
                )
            except Exception as error:
                # Delivery of the already sealed best has priority.  The
                # binding/epoch still leaves an append-only trace if preserving
                # the extra exploratory source snapshot itself fails.
                self.log(
                    "exploratory_final_checkpoint_failed",
                    binding=exploratory_binding,
                    error_type=type(error).__name__,
                )

            restored_binding = self.stack.restore_snapshot(checkpoint).to_dict()
            self.log(
                "global_best_restored",
                snapshot_id=state["snapshot_id"],
                selected_latency_ms=state["selected_latency_ms"],
                exploratory_snapshot_id=exploratory_snapshot_id,
                exploratory_binding=exploratory_binding,
                restored_binding=restored_binding,
            )
            self.repo.commit(
                f"restore global best {state['snapshot_id']} epoch={self.stack.epoch}"
            )
            fallback_state, fallback_checkpoint, _, fallback_contents = fallback_loaded
            selected_is_fallback = (
                fallback_state["snapshot_id"] == state["snapshot_id"]
            )
            requalification = self._audit_current_stack(
                source="restored_global_best"
            )
            no_regression: dict = {
                "evaluated": False,
                "decision": "not_evaluated",
                "reference_snapshot_id": fallback_state["snapshot_id"],
            }
            regression = False
            if not requalification.ok:
                result = requalification
                no_regression["reason"] = "selected_requalification_failed"
            elif selected_is_fallback:
                result = self._measure_current_against_snapshot(
                    fallback_contents, layer="stack_final_fallback", profile=profile
                )
                result = self._validate_timing_result(result, paired=True)
                no_regression.update(
                    {
                        "decision": "selected_is_immutable_fallback",
                        "reason": "paired_fallback_qualification",
                    }
                )
            else:
                result = self._measure_current_against_snapshot(
                    fallback_contents,
                    layer="stack_final_no_regression",
                    profile=profile,
                )
                result = self._validate_timing_result(result, paired=True)
                vs_base = result.public.get("vs_base")
                if result.ok and isinstance(vs_base, dict):
                    try:
                        ratio, threshold = self._gain_threshold(vs_base)
                        reference_latency = self._positive_latency(
                            vs_base.get("base_latency_ms")
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        result = MeasureResult(
                            ok=False,
                            tier=result.tier,
                            layer=result.layer,
                            public={
                                **result.public,
                                "pass": False,
                                "diagnostic": {
                                    "fault_stage": "paired_no_regression_evidence",
                                    "error_type": type(error).__name__,
                                },
                            },
                            trusted=dict(result.trusted),
                        )
                        no_regression["reason"] = "invalid_paired_statistics"
                    else:
                        regression = ratio > 1.0 + threshold
                        no_regression.update(
                            {
                                "evaluated": True,
                                "decision": (
                                    "fallback_required"
                                    if regression
                                    else "selected_no_regression"
                                ),
                                "median_ratio": ratio,
                                "ratio_cv": float(vs_base.get("ratio_cv") or 0.0),
                                "pairs": max(1, int(vs_base.get("pairs") or 1)),
                                "noise_threshold": threshold,
                                "reference_latency_ms": reference_latency,
                            }
                        )
                else:
                    no_regression["reason"] = "paired_measurement_failed"
                    if result.ok:
                        result = MeasureResult(
                            ok=False,
                            tier=result.tier,
                            layer=result.layer,
                            public={
                                **result.public,
                                "pass": False,
                                "diagnostic": {
                                    "fault_stage": "paired_no_regression_evidence",
                                    "error_type": "MissingPairedStatistics",
                                },
                            },
                            trusted=dict(result.trusted),
                        )
            tagged = self._tag_final_result(
                result,
                source="restored_global_best",
                snapshot_id=str(state["snapshot_id"]),
                requalification=requalification,
                selection=no_regression,
            )
            correctness_failure = self._is_correctness_failure(tagged)
            reference_failure = tagged.public.get("reference_correct") is False
            self.log(
                "final_global_best",
                snapshot_id=state["snapshot_id"],
                ok=tagged.ok,
                correctness_failure=correctness_failure,
                requalification_ok=requalification.ok,
                requalification_checks=requalification.public.get(
                    "checks_completed", 0
                ),
                latency_ms=tagged.public.get("latency_ms"),
                selected_latency_ms=state["selected_latency_ms"],
                final_selection=no_regression,
                regression=regression,
                reference_failure=reference_failure,
                binding=self.stack.binding().to_dict(),
                diagnostic=tagged.public.get("diagnostic"),
            )
            if tagged.ok and not regression:
                return tagged
            if not regression and (not correctness_failure or reference_failure):
                # Missing paired evidence, transport faults, or a broken
                # reference fail closed; do not disguise them as a valid
                # fallback delivery.
                return tagged
            if selected_is_fallback:
                self.log(
                    "final_fallback_skipped",
                    reason="selected_best_is_naive_imported_fallback",
                    snapshot_id=state["snapshot_id"],
                )
                return tagged

            failed_binding = self.stack.binding().to_dict()
            fallback_binding = self.stack.restore_snapshot(
                fallback_checkpoint
            ).to_dict()
            self.log(
                "naive_fallback_restored",
                snapshot_id=fallback_state["snapshot_id"],
                failed_snapshot_id=state["snapshot_id"],
                failed_binding=failed_binding,
                restored_binding=fallback_binding,
            )
            self.repo.commit(
                "restore naive/imported fallback "
                f"{fallback_state['snapshot_id']} epoch={self.stack.epoch}"
            )
            fallback_requalification = self._audit_current_stack(
                source="naive_imported_fallback"
            )
            fallback_result = (
                self._measure_current_against_snapshot(
                    fallback_contents, layer="stack_final_fallback", profile=profile
                )
                if fallback_requalification.ok
                else fallback_requalification
            )
            fallback_tagged = self._tag_final_result(
                fallback_result,
                source="naive_imported_fallback",
                snapshot_id=str(fallback_state["snapshot_id"]),
                requalification=fallback_requalification,
                selection={
                    "decision": "immutable_fallback_delivered",
                    "reason": (
                        "paired_regression" if regression else "correctness_failure"
                    ),
                    "failed_snapshot_id": state["snapshot_id"],
                    "paired_evidence": no_regression,
                },
            )
            self.log(
                "final_naive_fallback",
                snapshot_id=fallback_state["snapshot_id"],
                failed_snapshot_id=state["snapshot_id"],
                ok=fallback_tagged.ok,
                requalification_ok=fallback_requalification.ok,
                requalification_checks=fallback_requalification.public.get(
                    "checks_completed", 0
                ),
                latency_ms=fallback_tagged.public.get("latency_ms"),
                fallback_reason=(
                    "paired_regression" if regression else "correctness_failure"
                ),
                paired_evidence=no_regression,
                binding=self.stack.binding().to_dict(),
                diagnostic=fallback_tagged.public.get("diagnostic"),
            )
            return fallback_tagged
