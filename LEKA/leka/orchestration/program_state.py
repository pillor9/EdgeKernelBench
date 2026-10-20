"""Artifact stack: one per pipeline (operator x backend).

Slots (CUDA filenames shown; OpenCL uses .cl and LLVM uses .ll):
  L0   relax_mod.py      producer: relax_agent   default: task naive relax
  T01  canonical_tir.py  producer: driver        (deterministic translation)
  L1   scheduled_tir.py  producer: tir_agent     default: dlight fallback
  T12  generated.cu      producer: driver        (extracted on device at T1)
  L2   patched.cu        producer: source agent  default: absent (== generated)

Invalidation: promoting layer k resets every slot below k on the active graph
branch and bumps the stack epoch.  Epoch is a monotonic observation revision,
not by itself a dependency identity: an upper-layer candidate remains usable
when only unrelated lower slots changed.  Promotion is checked against the
candidate's signed layer projection and graph-branch lineage.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

LAYERS = ("L0", "L1", "L2")
STACK_SLOTS = ("L0", "T01", "L1", "T12", "L2")
_DERIVED_BELOW = {"L0": ("T01", "L1", "T12", "L2"), "L1": ("T12", "L2"), "L2": ()}
SNAPSHOT_SCHEMA_VERSION = 1


def slot_files(l2_ext: str = "cu") -> dict[str, str]:
    return {
        "L0": "relax_mod.py",
        "T01": "canonical_tir.py",
        "L1": "scheduled_tir.py",
        "T12": f"generated.{l2_ext}",
        "L2": f"patched.{l2_ext}",
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class StackBinding:
    """Immutable identity of one full stack state; travels with every request."""

    epoch: int
    l0_sha: str
    l1_sha: str
    l2_sha: str

    def to_dict(self) -> dict:
        return {"epoch": self.epoch, "l0_sha": self.l0_sha, "l1_sha": self.l1_sha, "l2_sha": self.l2_sha}

    @classmethod
    def from_dict(cls, d: dict) -> "StackBinding":
        return cls(int(d["epoch"]), str(d["l0_sha"]), str(d["l1_sha"]), str(d["l2_sha"]))


@dataclass
class ArtifactStack:
    root: Path                      # runs/<pipeline_id>/stack/
    epoch: int = 0
    meta: dict = field(default_factory=dict)
    l2_ext: str = "cu"

    # -- persistence ---------------------------------------------------------
    @classmethod
    def open(cls, root: Path, *, l2_ext: str = "cu") -> "ArtifactStack":
        root.mkdir(parents=True, exist_ok=True)
        meta_path = root / "stack.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(root=root, epoch=int(meta.get("epoch", 0)), meta=meta, l2_ext=l2_ext)

    def _save(self) -> None:
        self.meta.update({"epoch": self.epoch, "updated_at": time.time()})
        (self.root / "stack.json").write_text(json.dumps(self.meta, indent=2))

    # -- slot access ---------------------------------------------------------
    def path(self, slot: str) -> Path:
        return self.root / slot_files(self.l2_ext)[slot]

    def read(self, slot: str) -> str | None:
        p = self.path(slot)
        return p.read_text() if p.exists() else None

    def write(self, slot: str, text: str) -> str:
        self.path(slot).write_text(text)
        sha = sha256_text(text)
        self.meta.setdefault("slot_sha", {})[slot] = sha
        self._save()
        return sha

    def sha(self, slot: str) -> str:
        text = self.read(slot)
        return sha256_text(text) if text is not None else "absent"

    def binding(self) -> StackBinding:
        return StackBinding(self.epoch, self.sha("L0"), self.sha("L1"), self.sha("L2"))

    def active_branch_id(self) -> str:
        """Return the active L0 lineage identity, creating the root lazily."""

        branch_id = self.meta.get("active_branch_id")
        if isinstance(branch_id, str) and branch_id:
            return branch_id
        # L0 may be absent only during first bootstrap.  The stable literal
        # keeps pre-bootstrap callers deterministic; bootstrap replaces it
        # with a content-addressed root as soon as L0 exists.
        return "b-uninitialized"

    def initialize_root_branch(self) -> str:
        """Bind a fresh/resumed stack to a stable content-addressed root."""

        current = self.meta.get("active_branch_id")
        current_l0 = self.sha("L0")
        roots = self.meta.setdefault("branch_root_l0", {})
        if isinstance(current, str) and current and current != "b-uninitialized":
            recorded = roots.get(current)
            if recorded is None:
                # One-time migration for stacks created before branch roots
                # were explicit.
                roots[current] = current_l0
                self._save()
                return current
            if recorded == current_l0:
                return current
            # A resumed process may observe an exploratory L0 write that
            # happened immediately before a crash. Preserve the old lineage
            # and give that materialized graph a new recovery branch.
            branch_id = f"b-resume-{current_l0[:16]}"
            self.meta.setdefault("branch_parent", {})[branch_id] = current
            roots[branch_id] = current_l0
            self.meta["active_branch_id"] = branch_id
            self._save()
            return branch_id
        branch_id = f"b-root-{current_l0[:16]}"
        self.meta["active_branch_id"] = branch_id
        self.meta.setdefault("branch_parent", {})[branch_id] = None
        roots[branch_id] = current_l0
        self._save()
        return branch_id

    def branch_parent(self, branch_id: str | None = None) -> str | None:
        selected = self.active_branch_id() if branch_id is None else branch_id
        parent = self.meta.get("branch_parent", {}).get(selected)
        return parent if isinstance(parent, str) else None

    def branch_root_l0_sha(self, branch_id: str) -> str | None:
        """Return the immutable L0 base hash recorded for one known branch."""

        value = self.meta.get("branch_root_l0", {}).get(branch_id)
        return value if isinstance(value, str) and value else None

    def activate_branch(self, branch_id: str, *, parent_branch_id: str) -> None:
        """Move the materialized stack onto a newly promoted L0 branch."""

        if not isinstance(branch_id, str) or not branch_id.startswith("b-"):
            raise ValueError("invalid branch identity")
        if not isinstance(parent_branch_id, str) or not parent_branch_id.startswith("b-"):
            raise ValueError("invalid parent branch identity")
        parents = self.meta.setdefault("branch_parent", {})
        existing = parents.get(branch_id, parent_branch_id)
        if existing != parent_branch_id:
            raise RuntimeError("branch identity is already bound to another parent")
        parents[branch_id] = parent_branch_id
        self.meta.setdefault("branch_root_l0", {})[branch_id] = self.sha("L0")
        self.meta["active_branch_id"] = branch_id
        self._save()

    # -- sealed full-stack snapshots ----------------------------------------
    @staticmethod
    def _require_real_directory(path: Path, *, label: str) -> None:
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"{label} must be a real directory")

    @staticmethod
    def _write_private_text(path: Path, text: str) -> None:
        """Create a driver-only regular file without following a symlink."""

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(text)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    def seal_contents_snapshot(
        self,
        checkpoints_root: Path,
        *,
        contents: dict[str, str | None],
        metadata: dict,
    ) -> tuple[Path, dict]:
        """Persist an immutable snapshot assembled from explicit slot texts.

        This is used by the operator-tree store to retain a measured branch
        without first making it the one materialized working stack.  The
        caller must describe every slot, including explicit ``None`` values
        for invalidated lower artifacts. ``checkpoints_root`` lives below
        ``run_dir/private``.  A fresh
        directory is atomically published only after every source and the
        manifest have been written.  Missing derived/lower slots are recorded
        explicitly, so restore reproduces the complete stack state rather
        than overlaying a checkpoint on the exploratory current stack.
        """

        if set(contents) != set(STACK_SLOTS):
            raise ValueError("snapshot contents must describe every stack slot")

        if checkpoints_root.is_symlink():
            raise RuntimeError("checkpoint root may not be a symlink")
        checkpoints_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._require_real_directory(checkpoints_root, label="checkpoint root")
        os.chmod(checkpoints_root, 0o700)

        snapshot_id = f"s{time.time_ns()}-{uuid.uuid4().hex[:10]}"
        temporary = checkpoints_root / f".{snapshot_id}.tmp"
        final = checkpoints_root / snapshot_id
        temporary.mkdir(mode=0o700)
        slots: dict[str, dict] = {}
        try:
            for slot in STACK_SLOTS:
                text = contents[slot]
                if text is not None and not isinstance(text, str):
                    raise ValueError(f"snapshot slot {slot} must be text or null")
                if text is None:
                    slots[slot] = {"present": False, "sha256": "absent"}
                    continue
                relative = f"{slot}.txt"
                self._write_private_text(temporary / relative, text)
                slots[slot] = {
                    "present": True,
                    "path": relative,
                    "sha256": sha256_text(text),
                }
            source_binding = StackBinding(
                self.epoch,
                sha256_text(contents["L0"]) if contents["L0"] is not None else "absent",
                sha256_text(contents["L1"]) if contents["L1"] is not None else "absent",
                sha256_text(contents["L2"]) if contents["L2"] is not None else "absent",
            )
            manifest = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "snapshot_id": snapshot_id,
                "created_at": time.time(),
                "l2_ext": self.l2_ext,
                "source_epoch": self.epoch,
                "source_binding": source_binding.to_dict(),
                "slots": slots,
                "metadata": metadata,
            }
            manifest_text = json.dumps(
                manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            self._write_private_text(temporary / "manifest.json", manifest_text)
            os.replace(temporary, final)
            return final, manifest
        except Exception:
            # The temporary directory is private and contains only files
            # created by this call.  Leave no half-published checkpoint.
            if temporary.exists() and not temporary.is_symlink():
                for child in temporary.iterdir():
                    if child.is_file() and not child.is_symlink():
                        child.unlink()
                temporary.rmdir()
            raise

    def seal_snapshot(self, checkpoints_root: Path, *, metadata: dict) -> tuple[Path, dict]:
        """Persist an immutable, hash-sealed copy of the materialized stack."""

        contents: dict[str, str | None] = {}
        for slot in STACK_SLOTS:
            source = self.path(slot)
            if source.is_symlink() or (source.exists() and not source.is_file()):
                raise RuntimeError(f"stack slot must be a regular file: {slot}")
            contents[slot] = source.read_text() if source.exists() else None
        return self.seal_contents_snapshot(
            checkpoints_root,
            contents=contents,
            metadata=metadata,
        )

    def read_snapshot(self, checkpoint: Path) -> tuple[dict, dict[str, str | None]]:
        """Verify a sealed snapshot and return its manifest and slot texts."""

        self._require_real_directory(checkpoint, label="stack checkpoint")
        manifest_path = checkpoint / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise RuntimeError("stack checkpoint manifest must be a regular file")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise RuntimeError("stack checkpoint manifest must be a JSON object")
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise RuntimeError("unsupported stack checkpoint schema")
        if manifest.get("snapshot_id") != checkpoint.name:
            raise RuntimeError("stack checkpoint identity mismatch")
        if manifest.get("l2_ext") != self.l2_ext:
            raise RuntimeError("stack checkpoint backend extension mismatch")
        slot_records = manifest.get("slots")
        if not isinstance(slot_records, dict) or set(slot_records) != set(STACK_SLOTS):
            raise RuntimeError("stack checkpoint does not describe every slot")

        contents: dict[str, str | None] = {}
        for slot in STACK_SLOTS:
            record = slot_records[slot]
            if not isinstance(record, dict) or not isinstance(record.get("present"), bool):
                raise RuntimeError(f"invalid checkpoint record for {slot}")
            if not record["present"]:
                if record.get("sha256") != "absent":
                    raise RuntimeError(f"invalid absent checkpoint record for {slot}")
                contents[slot] = None
                continue
            expected_name = f"{slot}.txt"
            if record.get("path") != expected_name:
                raise RuntimeError(f"unsafe checkpoint path for {slot}")
            source_path = checkpoint / expected_name
            if source_path.is_symlink() or not source_path.is_file():
                raise RuntimeError(f"checkpoint source for {slot} must be a regular file")
            text = source_path.read_text(encoding="utf-8")
            if not hmac.compare_digest(sha256_text(text), str(record.get("sha256"))):
                raise RuntimeError(f"checkpoint source hash mismatch for {slot}")
            contents[slot] = text
        return manifest, contents

    def restore_snapshot(self, checkpoint: Path) -> StackBinding:
        """Restore a verified full-stack checkpoint and invalidate old work.

        The saved epoch is provenance, not a capability to resurrect.  The
        restored stack therefore receives a strictly newer epoch than both
        the exploratory current state and the checkpoint's original epoch.
        """

        manifest, contents = self.read_snapshot(checkpoint)
        restore_id = uuid.uuid4().hex[:10]
        prepared: dict[str, Path] = {}
        try:
            # Validate every destination before mutating any slot, so an
            # unexpected directory/symlink cannot leave a half-restored stack.
            for slot in STACK_SLOTS:
                target = self.path(slot)
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    raise RuntimeError(f"stack slot is not a regular file: {slot}")
            for slot, text in contents.items():
                if text is None:
                    continue
                temporary = self.root / f".{self.path(slot).name}.restore-{restore_id}"
                self._write_private_text(temporary, text)
                prepared[slot] = temporary
            for slot in STACK_SLOTS:
                target = self.path(slot)
                if contents[slot] is None:
                    if target.exists():
                        target.unlink()
                else:
                    os.replace(prepared.pop(slot), target)
            self.epoch = max(self.epoch, int(manifest["source_epoch"])) + 1
            self.meta["slot_sha"] = {
                slot: sha256_text(text)
                for slot, text in contents.items()
                if text is not None
            }
            self.meta["restored_from"] = manifest["snapshot_id"]
            metadata = manifest.get("metadata")
            restored_branch = (
                metadata.get("branch_id") if isinstance(metadata, dict) else None
            )
            if isinstance(restored_branch, str) and restored_branch.startswith("b-"):
                self.meta["active_branch_id"] = restored_branch
            self._save()
            return self.binding()
        finally:
            for temporary in prepared.values():
                if temporary.exists() and not temporary.is_symlink():
                    temporary.unlink()

    # -- invalidation --------------------------------------------------------
    def invalidate_below(self, layer: str) -> list[str]:
        """Promoting `layer` resets derived slots; caller re-runs translations."""
        cleared = []
        for slot in _DERIVED_BELOW[layer]:
            p = self.path(slot)
            if p.exists():
                p.unlink()
                cleared.append(slot)
            self.meta.setdefault("slot_sha", {}).pop(slot, None)
        self.epoch += 1
        self._save()
        return cleared
