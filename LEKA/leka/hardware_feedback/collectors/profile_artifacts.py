"""Bounded raw collector evidence written only beneath a caller-owned root.

No model fields, remote paths or process text are used as filesystem names.
Directory descriptors and O_NOFOLLOW prevent symlink redirection during writes.
The final digest gets a compact manifest receipt, never the raw streams.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

POST_SNAPSHOT_ARTIFACT_SCHEMA = "leka.vivo_post_snapshot_artifacts.v1"
POST_SNAPSHOT_PARSER_SCHEMA = "leka.vivo_post_runtime_snapshot.v1"


def artifact_dir_for_jobs_root(jobs_root) -> Path:
    """Canonicalize evaluator configuration before adding archive descendants.

    macOS tempfile roots commonly start with the OS alias ``/var``. Only the
    trusted, evaluator-selected jobs root is resolved here. Callers must never
    pass model/remote paths, nor resolve the appended ``hardware_artifacts``
    directory: archive descendants still undergo `_open_root`'s nofollow walk.
    Resolution pins the chosen destination before any archive write; a changed
    or newly introduced symlink in that canonical path is rejected by the walk.
    """
    root = Path(jobs_root)
    if ".." in root.parts:
        raise ValueError("trusted jobs root must not contain parent traversal")
    root = root.absolute().resolve()
    if root == Path("/"):
        raise ValueError("trusted jobs root must be a specific directory")
    return root / "hardware_artifacts"


def _open_root(root: Path) -> int:
    if not root.is_absolute() or ".." in root.parts or root == Path("/"):
        raise ValueError("artifact_dir must be a specific absolute trusted directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)
    try:
        for part in root.parts[1:]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("artifact_dir must be owned by this user with mode 0700")
        return fd
    except BaseException:
        os.close(fd)
        raise


class WindowArtifacts:
    """Keep a bounded prefix plus the selected window, even after overflow.

    The reserved selected slot is replaced whenever a strictly busier window
    arrives. This preserves the exact selected raw data without retaining an
    unbounded history or stopping sampling during a long compile phase.
    """

    def __init__(self, artifact_dir, *, collector: str, parser_schema: str,
                 max_windows: int = 64, max_stream_bytes: int = 65_536):
        if (type(max_windows) is not int or type(max_stream_bytes) is not int
                or not 2 <= max_windows <= 256 or not 256 <= max_stream_bytes <= 1_048_576):
            raise ValueError("artifact limits outside supported bounds")
        root = Path(artifact_dir)
        parent = _open_root(root)
        name = "capture-" + uuid.uuid4().hex
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
            self._fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        finally:
            os.close(parent)
        self.path = root / name
        self.collector = collector
        self.parser_schema = parser_schema
        self.max_windows = max_windows
        self.max_stream_bytes = max_stream_bytes
        self.count = 0
        self.truncated_windows = 0
        self.windows = []
        self.selected = None

    def _write(self, name: str, payload: bytes) -> dict:
        # Names are entirely evaluator generated; replacing a selected slot
        # replaces the directory entry and never follows a prior symlink.
        temporary = ".pending-" + uuid.uuid4().hex
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=self._fd)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
            os.replace(temporary, name, src_dir_fd=self._fd, dst_dir_fd=self._fd)
        except BaseException:
            try:
                os.unlink(temporary, dir_fd=self._fd)
            except FileNotFoundError:
                pass
            raise
        return {"path": str(self.path / name), "sha256": hashlib.sha256(payload).hexdigest(),
                "saved_bytes": len(payload)}

    def _bounded(self, text: str) -> tuple[bytes, dict]:
        raw = text.encode("utf-8", errors="replace")
        return raw[:self.max_stream_bytes], {
            "original_bytes": len(raw), "truncated": len(raw) > self.max_stream_bytes,
            "original_sha256": hashlib.sha256(raw).hexdigest()}

    def record(self, stdout: str, stderr: str, metadata: dict, *, selected: bool = False) -> int:
        index = self.count
        self.count += 1
        streams = {"stdout": self._bounded(stdout), "stderr": self._bounded(stderr)}
        truncated = any(info["truncated"] for _, info in streams.values())
        self.truncated_windows += int(truncated)
        details = {"schema_version": "leka.collector_window.v1", "collector": self.collector,
                   "parser_schema": self.parser_schema, "window_index": index, **metadata}
        encoded = json.dumps(details, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
        metadata_truncated = len(encoded) > self.max_stream_bytes
        if metadata_truncated:
            encoded = json.dumps({"schema_version": "leka.collector_window.v1",
                "window_index": index, "metadata_truncated": True,
                "original_sha256": hashlib.sha256(encoded).hexdigest(),
                "original_bytes": len(encoded)}).encode()

        def save(prefix):
            receipt = {"window_index": index, "raw_complete": not truncated,
                       "metadata_complete": not metadata_truncated}
            for label, (payload, info) in streams.items():
                receipt[label] = {**self._write(prefix + "." + label + ".log", payload), **info}
            receipt["metadata"] = self._write(prefix + ".json", encoded)
            return receipt

        if index < self.max_windows - 1:
            self.windows.append(save(f"window-{index:06d}"))
        if selected:
            self.selected = save("selected-window")
        return index

    def finish(self, *, process_stderr: str | None = None, **context) -> dict:
        manifest = {"schema_version": "leka.collector_artifacts.v1", "collector": self.collector,
            "parser_schema": self.parser_schema, "observed_windows": self.count,
            "prefix_windows_retained": len(self.windows),
            "prefix_windows_omitted": max(0, self.count - len(self.windows)),
            "selected_retained_separately": self.selected is not None,
            "truncated_stream_windows": self.truncated_windows,
            "max_window_slots": self.max_windows, "max_bytes_per_stream_or_metadata": self.max_stream_bytes,
            "stream_encoding": "utf8_from_subprocess_text",
            "windows": self.windows, "selected_window": self.selected, "context": context}
        try:
            if process_stderr is not None:
                payload, info = self._bounded(process_stderr)
                capture = context.get("process_stderr_capture") or {}
                manifest["process_stderr"] = {**self._write("process.stderr.log", payload), **info,
                    "capture_truncated": bool(capture.get("truncated")),
                    "source_bytes": capture.get("original_bytes", info["original_bytes"]),
                    "raw_complete": not (info["truncated"] or capture.get("truncated"))}
            receipt = self._write("manifest.json", json.dumps(manifest, ensure_ascii=False,
                sort_keys=True, allow_nan=False, indent=2).encode())
        finally:
            os.close(self._fd)
        return {"schema_version": manifest["schema_version"], "manifest": receipt,
                "observed_windows": self.count, "retained_prefix_windows": len(self.windows),
                "prefix_windows_omitted": manifest["prefix_windows_omitted"],
                "truncated_stream_windows": self.truncated_windows,
                "process_stderr_complete": (manifest["process_stderr"]["raw_complete"]
                    if "process_stderr" in manifest else None),
                "selected_window_index": self.selected["window_index"] if self.selected else None,
                "selected_raw_complete": self.selected["raw_complete"] if self.selected else False,
                "selected_metadata_complete": self.selected["metadata_complete"] if self.selected else False}


def archive_runtime_snapshot(artifact_dir, *, stdout: str, stderr: str,
                             metadata: dict, max_stream_bytes: int = 65_536) -> dict:
    """One post-measure capture, separate from every PMU window/selection.

    Reuse the private directory, atomic writer and byte bounds without calling
    record()/finish(): a snapshot is not a PMU window. Fixed names never come
    from the remote output. All three captured texts are prefixes if oversized;
    no truncated identity or metadata is considered independently replayable.
    """
    details = json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)
    sink = WindowArtifacts(artifact_dir, collector="vivo_post_runtime_snapshot",
                           parser_schema=POST_SNAPSHOT_PARSER_SCHEMA,
                           max_windows=2, max_stream_bytes=max_stream_bytes)
    try:
        files = {}
        for label, text, suffix in (("stdout", stdout, ".stdout.log"),
                                    ("stderr", stderr, ".stderr.log"),
                                    ("metadata", details, ".json")):
            payload, bounds = sink._bounded(text)
            files[label] = {**sink._write("post-runtime-snapshot" + suffix, payload), **bounds}
        raw_complete = not any(files[label]["truncated"] for label in ("stdout", "stderr"))
        metadata_complete = not files["metadata"]["truncated"]
        manifest = {
            "schema_version": POST_SNAPSHOT_ARTIFACT_SCHEMA,
            "collector": "vivo_post_runtime_snapshot", "parser_schema": POST_SNAPSHOT_PARSER_SCHEMA,
            "scope": "post_measure_owned_rpc_root_tasks_and_separate_global_sysfs",
            "exact_bundle": False, "timed_phase_overlap": "unproven",
            "stream_encoding": "utf8_from_subprocess_text", "file_count": 3,
            "max_bytes_per_stream_or_metadata": max_stream_bytes,
            "raw_complete": raw_complete, "metadata_complete": metadata_complete, "files": files,
        }
        receipt = sink._write("manifest.json", json.dumps(manifest, ensure_ascii=False,
            sort_keys=True, allow_nan=False, indent=2).encode())
    finally:
        os.close(sink._fd)
    return {"schema_version": POST_SNAPSHOT_ARTIFACT_SCHEMA, "status": "archived",
            "manifest": receipt, "raw_complete": raw_complete, "metadata_complete": metadata_complete}
