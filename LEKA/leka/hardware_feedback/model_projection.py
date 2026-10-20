"""Evaluator-private raw hardware digests and de-identified model projections.

This boundary applies to hardware report data, not operational launcher paths.
Metric values, units, source hashes/bindings and NCU grouping IDs stay intact.
Private paths/hosts/serials get stable aliases within an evaluator archive root;
the alias HMAC key and raw-to-public receipt never enter an agent workspace.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import uuid

from .collectors.profile_artifacts import _open_root
from ..evaluation.diagnostics import scrub_report_text

SCHEMA = "leka.hardware_model_projection.v1"
_IDENTITY_KEYS = frozenset({
    "host", "hostname", "host_name", "host_alias", "ssh_host", "ssh_alias",
    "rpc_host", "remote_host", "ip", "ip_address", "serial", "adb_serial",
    "vivo_serial", "device_serial", "username", "user_name", "remote_user",
})
_PROTECTED_KEYS = frozenset({
    "binding", "source_binding", "digest_binding", "source_digest_id",
    "group_id", "ncu_group_id", "digest_id",
})
_IPV4 = re.compile(r"(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])")
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,}[0-9A-Fa-f]{0,4}(?![\w:])")


def _key(value: str) -> str:
    return re.sub(r"[\s-]+", "_", value.lower())


def _protected(key: str) -> bool:
    return key in _PROTECTED_KEYS or key.endswith(("_sha256", "_sha", "_hash")) or key in {"sha256", "hash"}


def _absolute_path(value: str) -> bool:
    return value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value) is not None


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def project_digest(raw: dict, *, alias_key: bytes, private_values=()) -> dict:
    """Pure deep projection; preserve row fields so exact-row sealing still works.

    ``private_values`` may only come from evaluator configuration (e.g. ADB
    serial/SSH host alias), never environment enumeration or credential reads.
    All numbers, including process IDs and physical device ordinals, are kept.
    """
    if not isinstance(alias_key, bytes) or len(alias_key) < 16:
        raise ValueError("hardware aliases need a private key of at least 16 bytes")

    def alias(kind, value):
        digest = hmac.new(alias_key, (kind + "\0" + value).encode("utf-8"), hashlib.sha256).hexdigest()[:24]
        return "<" + kind + ":" + digest + ">"

    identities = {str(value) for value in private_values if isinstance(value, str) and value}

    def collect(value, field=""):
        if _protected(field):
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "post_snapshot_artifacts":
                    continue
                collect(child, _key(str(key)))
        elif isinstance(value, list):
            for child in value:
                collect(child, field)
        elif field in _IDENTITY_KEYS and isinstance(value, str) and value:
            identities.add(value)

    collect(raw)
    # A short alias must not rewrite substrings of kernel names or words.
    known = [(re.compile(r"(?<![\w])" + re.escape(value) + r"(?![\w])"), alias("identity", value))
             for value in sorted(identities, key=lambda x: (-len(x), x))]

    def ip_alias(match):
        try:
            ipaddress.ip_address(match[0])
        except ValueError:
            return match[0]
        return alias("identity", match[0])

    def visit(value, field=""):
        if _protected(field):
            return copy.deepcopy(value)
        if isinstance(value, dict):
            # Field names are part of the evidence schema; never rename them.
            # The optional post-snapshot archive is evaluator-only. Existing
            # context/scopes remain; neither raw text nor its receipt enters
            # the model digest or introduces a new model-facing post field.
            return {key: visit(child, _key(str(key))) for key, child in value.items()
                    if key != "post_snapshot_artifacts"}
        if isinstance(value, list):
            return [visit(child, field) for child in value]
        if not isinstance(value, str):
            return value
        if field in _IDENTITY_KEYS and value:
            return alias("identity", value)
        path_field = field in {"path", "cwd", "home", "job_dir", "artifact_dir"} or field.endswith(("_path", "_dir"))
        if _absolute_path(value) and (path_field or not any(c.isspace() for c in value)):
            return alias("artifact", value)
        text = scrub_report_text(value, path_replacer=lambda path: alias("artifact", path))
        for pattern, replacement in known:
            text = pattern.sub(lambda _: replacement, text)
        return _IPV6.sub(ip_alias, _IPV4.sub(ip_alias, text))

    return visit(raw)


def _write(fd: int, name: str, payload: bytes) -> None:
    handle = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=fd)
    with os.fdopen(handle, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _alias_key(fd: int) -> bytes:
    name = "alias.key"
    try:
        _write(fd, name, os.urandom(32))
    except FileExistsError:
        pass
    handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    try:
        info = os.fstat(handle)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise ValueError("hardware alias key must be evaluator-owned regular mode 0600")
        key = os.read(handle, 33)
        if len(key) != 32:
            raise ValueError("hardware alias key must have exactly 32 bytes")
        return key
    finally:
        os.close(handle)


def archive_and_project(raw: dict, *, archive_root: Path, workspace: Path,
                        private_values=()) -> dict:
    """Archive first, then expose a sealed projection; fail closed on I/O errors.

    Production passes run_dir/private/hardware_model_digests, under the session
    sandbox's denied run root, outside the workspace allowlist. Unix permissions
    alone do not isolate same-user model processes; the existing OS sandbox is
    required. Archive descendants are opened via dirfds with no symlink follow.
    """
    root = Path(archive_root).absolute()
    model_root = Path(workspace).resolve()
    if root.is_relative_to(model_root) or model_root.is_relative_to(root):
        raise ValueError("raw hardware archive must be outside the model workspace")
    # Validate strict JSON before any model-facing artifact is written.
    raw_bytes = canonical_bytes(raw)
    parent = _open_root(root)
    try:
        key = _alias_key(parent)
        name = "digest-" + uuid.uuid4().hex
        os.mkdir(name, mode=0o700, dir_fd=parent)
        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        _write(child, "raw_digest.json", raw_bytes + b"\n")
        # SHA describes the exact saved bytes (including its final newline).
        raw_sha = hashlib.sha256(raw_bytes + b"\n").hexdigest()
        public = project_digest(raw, alias_key=key, private_values=private_values)
        public.pop("digest_id", None)
        public["model_projection"] = {
            "schema_version": SCHEMA,
            "raw_digest_sha256": raw_sha,
            "alias_scope": "evaluator_run_private_archive",
        }
        public["digest_id"] = hashlib.sha256(canonical_bytes(public)).hexdigest()
        _write(child, "projection_receipt.json", canonical_bytes({
            "schema_version": SCHEMA,
            "raw_digest_file": "raw_digest.json",
            "raw_digest_sha256": raw_sha,
            "model_digest_id": public["digest_id"],
            "source_binding": raw.get("binding"),
        }) + b"\n")
        return public
    finally:
        os.close(child)
