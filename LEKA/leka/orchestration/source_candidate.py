"""Seed an editable native candidate from the evaluator's frozen source snapshot.

This module only writes workspace projections. It has no pipeline, compiler,
measurement, stack mutation or promotion capability.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

ORIGIN_FILE = "L2_CANDIDATE_ORIGIN.json"


def _atomic(fd, name, text, mode):
    try:
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            raise RuntimeError("L2 workspace projection must be a file")
    except FileNotFoundError:
        pass
    temporary = ".l2-seed-" + uuid.uuid4().hex
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=fd)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(text.encode("utf-8"))
            stream.flush()
            os.fchmod(stream.fileno(), mode)
        # Replacing the directory entry detaches hostile prior links, without
        # opening or chmod-ing their targets. Temporary names are never inputs.
        os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
    finally:
        try: os.unlink(temporary, dir_fd=fd)
        except FileNotFoundError: pass


def project_native_candidate(workspace: Path, *, extension: str,
                             generated_source: str, active_override: str | None,
                             binding_payload: dict) -> dict:
    """Caller holds the Pipeline lock for source capture and all projections."""
    if extension not in {"cu", "cl", "ll"}:
        raise ValueError("unsupported native workspace extension")
    if (not isinstance(generated_source, str)
            or (active_override is not None and not isinstance(active_override, str))):
        raise ValueError("native seed requires captured source text")
    if ".." in Path(workspace).parts:
        raise ValueError("native workspace path must not traverse parents")
    captured = json.loads(json.dumps(binding_payload, allow_nan=False))
    projected = captured.get("projected_inputs") or {}
    generated_sha = hashlib.sha256(generated_source.encode("utf-8")).hexdigest()
    source_slot = "L2" if active_override is not None else "T12"
    source = active_override if active_override is not None else generated_source
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if projected.get("T12") != generated_sha or projected.get(source_slot) != source_sha:
        raise ValueError("native seed source differs from signed projected input")
    origin = {
        "schema_version": "leka.l2_candidate_seed.v1", "recipient_layer": "L2",
        "candidate_file": f"patched.{extension}",
        "generated_file": f"generated.{extension}", "source_slot": source_slot,
        "source_kind": "active_override" if active_override is not None else "generated_native",
        "source_sha256": source_sha, "initial_candidate_sha256": source_sha,
        "generated_source_sha256": generated_sha,
        "branch_id": captured.get("branch_id"),
        "workspace_capability_schema": captured.get("schema_version"),
        "binding": captured.get("binding"), "projected_inputs": projected,
        "authority": "evaluator_workspace_seed_not_measurement_or_promotion",
        "meaning": "Origin describes initially copied bytes only. The editable candidate may "
                   "diverge after edits. This copy does not change active L2 or authorize a "
                   "speedup; original signed binding, ABI, T0/T1 and promotion gates still apply.",
    }
    fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if os.fstat(fd).st_uid != os.getuid():
            raise ValueError("native workspace has another owner")
        _atomic(fd, f"generated.{extension}", generated_source, 0o444)
        _atomic(fd, f"patched.{extension}", source, 0o644)
        _atomic(fd, ORIGIN_FILE, json.dumps(origin, ensure_ascii=False, allow_nan=False,
                                          sort_keys=True, indent=2) + "\n", 0o444)
    finally:
        os.close(fd)
    return origin
