"""File-based measure bridge between agent workspaces and the driver.

The agent's per-workspace trusted measure launcher writes requests/<id>.json
into the workspace and polls for responses/<id>.json; the driver loop services
requests through the Pipeline.  No sockets or shared trusted state enter the
agent's writable sandbox.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
import os
import re
import stat
import threading
import time
import uuid
from pathlib import Path

from .hardware_delivery import compact_response_bytes


# Relax/TIR/device-source candidates are text.  Explicit L0 decomposition can
# legitimately expand a full scheduled module beyond 8 MiB before L1 removes
# redundant buffers/loops; keep the read bounded while allowing that handoff.
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = 64 * 1024
MAX_PENDING_REQUESTS = 64
MAX_OVERFLOW_REJECTIONS_PER_POLL = 64
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_ALLOWED_TIERS = frozenset({"t0", "t1", "promote"})


def _workspace_refresh_required(error: str) -> dict:
    return {
        "ok": False, "error": error,
        "error_code": "workspace_refresh_required",
        "measurement_executed": False,
        "correctness": "not_evaluated",
        "recovery": (
            "Preserve your idea in NOTES.md and end this turn now. "
            "The driver will refresh the signed projection at the next "
            "turn in the same session. Rebase and measure again then; "
            "do not retry this stale request or edit BASE_BINDING.json."
        ),
    }

# The ordinary launcher timeout is a liveness guard for requests that should
# normally complete within one agent turn.  Promotion is deliberately longer:
# its synchronous trusted path can run two fresh-seed audits, two independent
# reviewer attempts, L0 lowering before and after mutation, and a bounded Git
# checkpoint.  For the current production configuration, the slowest SSH audit
# also includes all transport retry deadlines:
#
#   setup 3*60+6, upload 3*300+6, worker 600,
#   pull 3*60+6, cleanup 3*30+6 = 1,974 seconds / audit.
#
# Two audits + two 600-second reviews + two 120-second lowerings + four
# 60-second Git operations + 15 minutes of scheduling/filesystem margin is
# 6,528 seconds. Round up to two hours. The run wall deadline remains
# authoritative and kills the whole stage process group if it expires first.
STANDARD_RESPONSE_TIMEOUT_S = 900
_PROMOTION_REVIEWED_WORST_CASE_S = (
    2 * 1974
    + 2 * 600
    + 2 * 120
    + 4 * 60
    + 15 * 60
)
PROMOTE_RESPONSE_TIMEOUT_S = max(2 * 60 * 60, _PROMOTION_REVIEWED_WORST_CASE_S)

SHIM = '''#!/usr/bin/env python3
"""measure <t0|t1> <artifact> [--note TEXT] | measure promote <node_id>"""
import json, sys, time, uuid
from pathlib import Path

WORKSPACE = Path(__LEKA_WORKSPACE__)
STANDARD_RESPONSE_TIMEOUT_S = __LEKA_STANDARD_RESPONSE_TIMEOUT_S__
PROMOTE_RESPONSE_TIMEOUT_S = __LEKA_PROMOTE_RESPONSE_TIMEOUT_S__

def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("t0", "t1", "promote"):
        print(__doc__); sys.exit(2)
    tier = sys.argv[1]
    note = ""
    if "--note" in sys.argv:
        note = sys.argv[sys.argv.index("--note") + 1]
    binding_path = WORKSPACE / "BASE_BINDING.json"
    try:
        workspace_binding = json.loads(binding_path.read_text())
    except Exception as error:
        print(json.dumps({"ok": False, "error":
            f"workspace base capability unavailable: {error}"})); sys.exit(1)
    if tier == "promote":
        payload = {"tier": "promote", "node_id": sys.argv[2],
                   "workspace_binding": workspace_binding}
    else:
        artifact = Path(sys.argv[2])
        if not artifact.exists():
            print(json.dumps({"ok": False, "error": f"no such file: {artifact}"})); sys.exit(1)
        payload = {"tier": tier, "artifact": str(artifact.resolve()), "note": note,
                   "workspace_binding": workspace_binding}
    authority_path = WORKSPACE / "TURN_AUTHORITY.json"
    if authority_path.exists() or authority_path.is_symlink():
        try:
            if authority_path.is_symlink() or authority_path.stat().st_size > 8192:
                raise ValueError("invalid turn authority file")
            payload["turn_authority"] = json.loads(authority_path.read_text())
        except Exception as error:
            print(json.dumps({"ok": False, "error": f"turn authority unavailable: {error}"})); sys.exit(1)
    req_id = uuid.uuid4().hex[:12]
    ws = WORKSPACE
    (ws / "requests").mkdir(exist_ok=True)
    (ws / "responses").mkdir(exist_ok=True)
    (ws / "requests" / f"{req_id}.json").write_text(json.dumps({
        "id": req_id, "submitted_at": time.time(), **payload}))
    response_timeout_s = (PROMOTE_RESPONSE_TIMEOUT_S
                          if tier == "promote" else STANDARD_RESPONSE_TIMEOUT_S)
    deadline = time.monotonic() + response_timeout_s
    resp = ws / "responses" / f"{req_id}.json"
    while time.monotonic() < deadline:
        if resp.exists():
            print(resp.read_text()); sys.exit(0)
        time.sleep(0.5)
    print(json.dumps({"ok": False, "error":
        f"measure timed out after {response_timeout_s}s (driver busy or down)"}))
    sys.exit(1)

main()
'''


def validate_workspace_path(workspace: Path) -> None:
    """Reject workspace identities that can redirect the driver trust root.

    Agent workspaces are ordinary directories owned by an untrusted model.  A
    resumed workspace must not be a symlink and must not contain its own Git
    boundary: either would let workspace-controlled state influence where
    Claude or the driver looks for trusted configuration.
    """

    if workspace.is_symlink():
        raise RuntimeError("agent workspace may not be a symlink")
    if not workspace.exists():
        raise RuntimeError("agent workspace does not exist")
    if not workspace.is_dir():
        raise RuntimeError("agent workspace must be a directory")
    nested_git = workspace / ".git"
    if nested_git.exists() or nested_git.is_symlink():
        raise RuntimeError("agent workspace may not contain a nested .git boundary")


def trusted_launcher_path(workspace: Path) -> Path:
    """Return the per-workspace launcher path outside the untrusted workspace."""

    validate_workspace_path(workspace)
    identity = hashlib.sha256(
        str(workspace.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    # Never inspect ``workspace/.git``.  That directory is model-writable and
    # therefore cannot establish the location of a trusted launcher.
    boundary = workspace.parent.resolve()
    while boundary != boundary.parent:
        git_dir = boundary / ".git"
        if git_dir.is_dir() and not git_dir.is_symlink():
            # PipelineRepo ignores private/, so trusted launchers never enter
            # candidate commits or experiment result scans.
            return boundary / "private" / "measure-launchers" / f"measure-{identity}"
        boundary = boundary.parent
    return workspace.parent.resolve() / ".leka-measure-launchers" / f"measure-{identity}"


def install_shim(workspace: Path) -> Path:
    """Install an immutable-to-the-agent launcher outside its writable cwd.

    The Claude process may edit any normal file in ``workspace``.  Therefore a
    workspace-local executable must never be the sole allowed Bash command: it
    could be overwritten and then used as a confused deputy.  Restricted mode
    confines file tools to ``workspace`` while the settings allow only this
    exact absolute launcher path.
    """

    launcher = trusted_launcher_path(workspace)
    for trusted_dir in (launcher.parent.parent, launcher.parent):
        if trusted_dir.is_symlink():
            raise RuntimeError(
                "trusted measure launcher directory must be a real directory"
            )
        if trusted_dir.exists():
            if not trusted_dir.is_dir():
                raise RuntimeError(
                    "trusted measure launcher directory must be a real directory"
                )
        else:
            # Create one component at a time, only after its real parent was
            # checked above, so a pre-existing symlink is never traversed.
            trusted_dir.mkdir()
    if launcher.exists() or launcher.is_symlink():
        if launcher.is_symlink() or not launcher.is_file():
            raise RuntimeError("trusted measure launcher path is not a regular file")
        launcher.chmod(0o700)
    source = (
        SHIM.replace("__LEKA_WORKSPACE__", json.dumps(str(workspace.resolve())))
        .replace(
            "__LEKA_STANDARD_RESPONSE_TIMEOUT_S__",
            str(STANDARD_RESPONSE_TIMEOUT_S),
        )
        .replace(
            "__LEKA_PROMOTE_RESPONSE_TIMEOUT_S__",
            str(PROMOTE_RESPONSE_TIMEOUT_S),
        )
    )
    launcher.write_text(source)
    launcher.chmod(0o500)

    # Remove the former writable executable.  Its command spelling is no
    # longer permission-allowed; leaving it in place would invite accidental
    # use by an agent following an older prompt.
    legacy = workspace / "measure"
    if legacy.exists() or legacy.is_symlink():
        legacy.unlink()
    for name in ("requests", "responses"):
        bridge = workspace / name
        if bridge.is_symlink() or (bridge.exists() and not bridge.is_dir()):
            raise RuntimeError(f"measure bridge {name} must be a real directory")
        bridge.mkdir(exist_ok=True)
    return launcher.resolve()


class MeasureServer:
    """Polls agent workspaces; services one request at a time (device is serial)."""

    def __init__(self, pipeline, layer_of_agent: dict[str, str], workspaces: dict[str, Path]):
        self.pipeline = pipeline
        self.layer_of_agent = layer_of_agent
        self.workspaces = workspaces
        self._accepting = threading.Event()
        self._accepting.set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._quiesced = False
        self._claimed_lock = threading.Lock()
        self._claimed_requests = 0

    @property
    def claimed_request_count(self) -> int:
        """Conservative evaluator observation, not signature validation/authority.

        The counter includes valid-shape requests already removed from the
        queue, including ones waiting for Pipeline ownership or failing later.
        This short lock never spans a Pipeline call or response projection.
        """
        with self._claimed_lock:
            return self._claimed_requests

    def _change_claimed_count(self, delta: int) -> None:
        with self._claimed_lock:
            next_count = self._claimed_requests + delta
            if next_count < 0:
                raise RuntimeError("unbalanced measure request claim")
            self._claimed_requests = next_count

    def start(self, *, interval_s: float = 0.5) -> threading.Thread:
        """Start and retain the one-shot request-serving thread."""

        with self._lifecycle_lock:
            if self._thread is not None:
                raise RuntimeError("measure server has already been started")
            self._thread = threading.Thread(
                target=self.serve_forever,
                args=(self._stop_event, interval_s),
                name="leka-measure-server",
                daemon=True,
            )
            self._thread.start()
            return self._thread

    @property
    def is_quiesced(self) -> bool:
        with self._lifecycle_lock:
            return self._quiesced

    def quiesce(self, *, timeout_s: float) -> None:
        """Stop admission, cancel in-flight work, and join the server.

        Returning is the synchronization proof required before finalization:
        no request can still hold the Pipeline RLock or mutate the stack.  A
        join timeout fails closed; callers must not finalize in that case.
        """

        timeout = max(0.0, float(timeout_s))
        self._accepting.clear()
        self._stop_event.set()
        self.pipeline.cancel_inflight(
            "exploration closed; cancelling in-flight measure request"
        )
        with self._lifecycle_lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        if thread is not None and thread.is_alive():
            raise RuntimeError(
                "measure server did not quiesce before the finalization reserve"
            )
        with self._lifecycle_lock:
            self._quiesced = True

    @staticmethod
    def _open_directory(path: Path) -> int:
        """Open a real directory without following its final path component."""

        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"request bridge path is not a real directory: {path}")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode):
            os.close(fd)
            raise RuntimeError(f"request bridge path is not a directory: {path}")
        return fd

    @staticmethod
    def _request_stem(filename: str) -> str | None:
        """Return the safe response id encoded by a direct ``*.json`` entry."""

        if not filename.endswith(".json"):
            return None
        stem = filename[:-5]
        if not _REQUEST_ID_RE.fullmatch(stem):
            return None
        return stem

    def _safe_log(self, event: str, *, agent: str, error: str) -> None:
        """Diagnostics must not be able to terminate the polling thread."""

        try:
            self.pipeline.log(event, agent=agent, error=error[:400])
        except Exception:
            pass

    @staticmethod
    def _unlink_request(request_dir_fd: int, filename: str) -> None:
        try:
            os.unlink(filename, dir_fd=request_dir_fd)
        except FileNotFoundError:
            pass

    @staticmethod
    def _write_response(response_dir_fd: int, request_id: str, response: dict, *, compact=False) -> int:
        """Atomically write below an already-opened response directory.

        Both filenames are derived from a strictly validated request id.  The
        temporary file is opened exclusively and the final rename replaces a
        hostile pre-existing symlink rather than following it.
        """

        if not _REQUEST_ID_RE.fullmatch(request_id):
            raise ValueError("unsafe response request id")
        final_name = f"{request_id}.json"
        temp_name = f".{request_id}.{uuid.uuid4().hex}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(
            os, "O_NOFOLLOW", 0
        )
        fd = os.open(temp_name, flags, 0o600, dir_fd=response_dir_fd)
        try:
            payload = (compact_response_bytes(response) if compact
                       else json.dumps(response, indent=2).encode("utf-8"))
            with os.fdopen(fd, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
            os.replace(
                temp_name,
                final_name,
                src_dir_fd=response_dir_fd,
                dst_dir_fd=response_dir_fd,
            )
        finally:
            os.close(fd)
            try:
                os.unlink(temp_name, dir_fd=response_dir_fd)
            except FileNotFoundError:
                pass
        return len(payload)

    @staticmethod
    def _read_request(request_dir_fd: int, filename: str) -> dict:
        """Read and validate one bounded, direct, regular request file."""

        metadata = os.stat(filename, dir_fd=request_dir_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("request must be a regular file (symlinks are rejected)")
        if metadata.st_size > MAX_REQUEST_BYTES:
            raise ValueError(f"request exceeds {MAX_REQUEST_BYTES} byte limit")

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(filename, flags, dir_fd=request_dir_fd)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("request must be a regular file")
            if opened.st_size > MAX_REQUEST_BYTES:
                raise ValueError(f"request exceeds {MAX_REQUEST_BYTES} byte limit")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                raw = handle.read(MAX_REQUEST_BYTES + 1)
        finally:
            os.close(fd)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError(f"request exceeds {MAX_REQUEST_BYTES} byte limit")
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise ValueError(f"request must be valid UTF-8 JSON: {err}") from err
        if not isinstance(request, dict):
            raise ValueError("request JSON must be an object")
        return request

    @staticmethod
    def _validate_request(request: dict, filename_stem: str) -> None:
        request_id = request.get("id")
        if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
            raise ValueError("invalid request id")
        if request_id != filename_stem:
            raise ValueError("request id must match request filename")
        tier = request.get("tier")
        if not isinstance(tier, str) or tier not in _ALLOWED_TIERS:
            raise ValueError("unsupported measure tier")

    def poll_once(self) -> int:
        if not self._accepting.is_set():
            return 0
        served = 0
        # L0 graph exploration owns the highest-level search space.  Always
        # admit its pending request before descendants in each poll so a flood
        # of L1/L2 work cannot starve a newly opened graph branch.  The device
        # remains serialized and every other workspace still receives one
        # bounded request opportunity per poll.
        ordered_workspaces = sorted(
            self.workspaces.items(),
            key=lambda item: (
                {"L0": 0, "L1": 1, "L2": 2}.get(
                    self.layer_of_agent.get(item[0]), 3
                ),
                item[0],
            ),
        )
        for agent, ws in ordered_workspaces:
            request_dir_fd: int | None = None
            response_dir_fd: int | None = None
            try:
                request_dir_fd = self._open_directory(ws / "requests")
                response_dir_fd = self._open_directory(ws / "responses")
                # Bound both memory use and cleanup work.  Entries beyond this
                # window remain queued for later polls; at most one retained
                # request reaches the device/pipeline in this poll.
                window = MAX_PENDING_REQUESTS + MAX_OVERFLOW_REJECTIONS_PER_POLL
                filenames: list[str] = []
                scanned = 0
                with os.scandir(request_dir_fd) as entries:
                    for entry in entries:
                        scanned += 1
                        if scanned > window:
                            break
                        if entry.name.endswith(".json"):
                            filenames.append(entry.name)
                filenames.sort()

                retained = filenames[:MAX_PENDING_REQUESTS]
                overflow = filenames[MAX_PENDING_REQUESTS:]
                for filename in overflow:
                    stem = self._request_stem(filename)
                    self._unlink_request(request_dir_fd, filename)
                    if stem is None:
                        continue
                    try:
                        self._write_response(
                            response_dir_fd,
                            stem,
                            {"ok": False, "error": "measure request queue is full"},
                        )
                        served += 1
                    except Exception as err:
                        self._safe_log(
                            "measure_server_error",
                            agent=agent,
                            error=f"unable to reject queued request: {err}",
                        )

                executed = False
                for filename in retained:
                    stem = self._request_stem(filename)
                    if stem is None:
                        # An unsafe filename cannot be reflected into a response
                        # pathname.  Remove it without parsing.
                        self._unlink_request(request_dir_fd, filename)
                        continue
                    claimed = False
                    try:
                        response: dict
                        request: dict | None = None
                        request_valid = False
                        hardware_delivery = getattr(self.pipeline, "_hardware_delivery", None)
                        inline_turn = None
                        observation_delivery = getattr(self.pipeline, "_hardware_observations", None)
                        observation_turn = None
                        try:
                            request = self._read_request(request_dir_fd, filename)
                            self._validate_request(request, stem)
                            request_valid = True
                        except Exception as err:
                            response = {"ok": False, "error": str(err)[:300]}
                            self._unlink_request(request_dir_fd, filename)
                        else:
                            if executed:
                                # Leave valid later requests in place for a future
                                # poll, preserving one pipeline execution per agent.
                                continue
                            self._change_claimed_count(1)
                            claimed = True
                            self._unlink_request(request_dir_fd, filename)
                            if (hardware_delivery is not None
                                    and getattr(hardware_delivery, "inline_feedback", False)
                                    and self._accepting.is_set() and not self._stop_event.is_set()):
                                try:
                                    inline_turn = hardware_delivery.feedback_turn(agent)
                                except Exception as err:
                                    self._safe_log("hardware_evidence_inline_feedback_error", agent=agent, error=str(err))
                            if (observation_delivery is not None
                                    and self._accepting.is_set() and not self._stop_event.is_set()):
                                try:
                                    observation_turn = observation_delivery.feedback_turn(agent)
                                except Exception as err:
                                    self._safe_log("hardware_observation_response_error", agent=agent, error=str(err)[:300])
                            if not self._accepting.is_set():
                                response = {
                                    "ok": False,
                                    "error": "measure service is quiescing for finalization",
                                }
                            else:
                                try:
                                    response = self._serve(agent, request)
                                except Exception as err:
                                    self._safe_log(
                                        "measure_server_error", agent=agent, error=str(err)
                                    )
                                    response = {
                                        "ok": False,
                                        "error": "driver-side failure while serving this "
                                        f"request: {str(err)[:300]}",
                                    }
                            executed = True
                        advisory = None
                        dialogue = getattr(self.pipeline, "_peer_dialogue", None)
                        if request_valid and dialogue is not None:
                            try:
                                advisory = dialogue.measure_response_advisory(
                                    agent, request_id=stem, tier=request["tier"],
                                    result_binding=response.get("binding"))
                            except Exception as err:
                                # Optional peer transport must never suppress a
                                # correctness/timing/error response from the evaluator.
                                self._safe_log("peer_tool_advisory_error", agent=agent, error=str(err))
                        if advisory is not None:
                            response = {**response, "peer_advisories": advisory}
                        hardware_notification = None
                        if request_valid and hardware_delivery is not None:
                            try:
                                hardware_notification = hardware_delivery.notification(
                                    agent, request_id=stem, tier=request["tier"],
                                    result_binding=response.get("binding"))
                            except Exception as err:
                                self._safe_log("hardware_evidence_notification_error",
                                               agent=agent, error=str(err)[:300])
                        if hardware_notification is not None:
                            response = {**response, "hardware_evidence_notifications": hardware_notification}
                        feedback = deferred = None
                        compact = False
                        if (request_valid and hardware_delivery is not None
                                and getattr(hardware_delivery, "inline_feedback", False)
                                and self._accepting.is_set() and not self._stop_event.is_set()):
                            try:
                                if response.get("error_code") == "turn_authority_rejected":
                                    # A queued old request must not consume new
                                    # turn feedback even if admitted to the file
                                    # bridge after the new turn began.
                                    if inline_turn is not None:
                                        deferred = {"turn_id": inline_turn, "request_id": stem,
                                            "measure_tier": request["tier"], "reason": "request_turn_authority_rejected",
                                            "fallback_in_response": False}
                                else:
                                    feedback, deferred = hardware_delivery.prepare_feedback(
                                        agent, request_id=stem, tier=request["tier"],
                                        result_binding=response.get("binding"), turn_id=inline_turn,
                                        base_response=response)
                                candidate_response = ({**response, "hardware_evidence_feedback": feedback}
                                                      if feedback is not None else response)
                                compact_response_bytes(candidate_response)
                            except Exception as err:
                                # Keep the original measurement and peer/path
                                # notifications even if optional display fails.
                                feedback = deferred = None
                                self._safe_log("hardware_evidence_inline_feedback_error", agent=agent, error=str(err))
                            else:
                                response, compact = candidate_response, True
                        observation_feedback = observation_deferred = None
                        if (request_valid and observation_delivery is not None
                                and self._accepting.is_set() and not self._stop_event.is_set()):
                            try:
                                if response.get("error_code") == "turn_authority_rejected":
                                    if observation_turn is not None:
                                        observation_deferred = {"turn_id": observation_turn, "request_id": stem,
                                            "measure_tier": request["tier"], "reason": "request_turn_authority_rejected"}
                                else:
                                    observation_feedback, observation_deferred = observation_delivery.prepare(
                                        agent, request_id=stem, tier=request["tier"],
                                        result_binding=response.get("binding"), turn_id=observation_turn,
                                        base_response=response)
                                observed_response = ({**response, "hardware_observations": observation_feedback}
                                                     if observation_feedback is not None else response)
                                compact_response_bytes(observed_response)
                            except Exception as err:
                                observation_feedback = observation_deferred = None
                                self._safe_log("hardware_observation_response_error", agent=agent, error=str(err)[:300])
                            else:
                                response, compact = observed_response, True
                        try:
                            if compact:
                                written_bytes = self._write_response(response_dir_fd, stem, response, compact=True)
                            else:
                                written_bytes = self._write_response(response_dir_fd, stem, response)
                            served += 1
                        except Exception as err:
                            self._safe_log(
                                "measure_server_error",
                                agent=agent,
                                error=f"unable to write response: {err}",
                            )
                        else:
                            if observation_feedback is not None:
                                try:
                                    observation_delivery.record_delivery(agent, observation_feedback, response_bytes=written_bytes)
                                except Exception as err:
                                    self._safe_log("hardware_observation_response_error", agent=agent, error=str(err)[:300])
                            if observation_deferred is not None:
                                try:
                                    observation_delivery.record_deferral(agent, observation_deferred, response_bytes=written_bytes)
                                except Exception as err:
                                    self._safe_log("hardware_observation_response_error", agent=agent, error=str(err)[:300])
                            if feedback is not None and feedback.get("status") == "complete_packet":
                                try:
                                    hardware_delivery.record_feedback_delivery(agent, feedback, response_bytes=written_bytes)
                                except Exception as err:
                                    self._safe_log("hardware_evidence_inline_feedback_error", agent=agent, error=str(err))
                            if deferred is not None:
                                try:
                                    hardware_delivery.record_feedback_deferral(agent, deferred, response_bytes=written_bytes)
                                except Exception as err:
                                    self._safe_log("hardware_evidence_inline_feedback_error", agent=agent, error=str(err))
                            if hardware_notification is not None:
                                try:
                                    hardware_delivery.record_notification_delivery(agent, hardware_notification)
                                except Exception as err:
                                    self._safe_log("hardware_evidence_notification_error",
                                                   agent=agent, error=str(err)[:300])
                            if advisory is not None:
                                try:
                                    dialogue.record_tool_response_delivery(agent, advisory)
                                except Exception as err:
                                    self._safe_log("peer_tool_advisory_error", agent=agent, error=str(err))
                    finally:
                        if claimed:
                            self._change_claimed_count(-1)
            except Exception as err:
                self._safe_log("measure_server_error", agent=agent, error=str(err))
            finally:
                if request_dir_fd is not None:
                    os.close(request_dir_fd)
                if response_dir_fd is not None:
                    os.close(response_dir_fd)
        return served

    def _read_artifact(self, agent: str, value: object) -> tuple[str | None, str | None]:
        """Read one bounded regular file strictly below ``agent``'s workspace.

        Agent requests are untrusted even though the normal trusted launcher
        submits a resolved absolute path.  Reject lexical traversal, paths
        outside the workspace, and every symlink below the trusted workspace
        root before opening the file read-only.
        """

        if not isinstance(value, str) or not value:
            return None, "artifact must be a non-empty path string"
        try:
            raw = Path(value)
        except (TypeError, ValueError) as err:
            return None, f"invalid artifact path: {err}"
        if ".." in raw.parts:
            return None, "artifact path traversal is not allowed"

        try:
            workspace_path = self.workspaces[agent]
            root = workspace_path.resolve(strict=True)
            lexical_root = workspace_path.absolute()

            if raw.is_absolute():
                # Accept either the spelling supplied to the driver or its
                # canonical spelling (notably /tmp versus /private/tmp on
                # macOS), but nothing else.
                relative = None
                for base in (lexical_root, root):
                    try:
                        relative = raw.relative_to(base)
                        break
                    except ValueError:
                        pass
                if relative is None:
                    return None, "artifact is outside the agent workspace"
            else:
                relative = raw

            candidate = root
            for component in relative.parts:
                if component in ("", "."):
                    continue
                candidate = candidate / component
                if candidate.is_symlink():
                    return None, "artifact path may not contain symlinks"

            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError:
                return None, "artifact is outside the agent workspace"

            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(resolved, flags)
            try:
                metadata = os.fstat(fd)
                if not stat.S_ISREG(metadata.st_mode):
                    return None, "artifact must be a regular file"
                if metadata.st_size > MAX_ARTIFACT_BYTES:
                    return None, (
                        f"artifact exceeds {MAX_ARTIFACT_BYTES} byte limit"
                    )
                with os.fdopen(fd, "rb", closefd=False) as handle:
                    data = handle.read(MAX_ARTIFACT_BYTES + 1)
            finally:
                os.close(fd)
        except (OSError, RuntimeError, ValueError) as err:
            return None, f"unable to read artifact: {err}"

        if len(data) > MAX_ARTIFACT_BYTES:
            return None, f"artifact exceeds {MAX_ARTIFACT_BYTES} byte limit"
        try:
            return data.decode("utf-8"), None
        except UnicodeDecodeError:
            return None, "artifact must be UTF-8 text"

    def _serve(self, agent: str, request: dict) -> dict:
        if getattr(self.pipeline, "turn_authority_enabled", False):
            # The token is captured by the launcher, never substituted from
            # the current workspace at dequeue time. Keep this lock through
            # action execution so a queued old request cannot cross a new
            # admission between authorization and use.
            with self.pipeline._lock:
                valid, error = self.pipeline.validate_turn_authority(
                    agent, self.layer_of_agent[agent], request.get("turn_authority"))
                if not valid:
                    return {"ok": False, "error": error, "error_code": "turn_authority_rejected",
                            "measurement_executed": False, "correctness": "not_evaluated"}
                return self._serve_authorized(agent, request)
        return self._serve_authorized(agent, request)

    def _serve_authorized(self, agent: str, request: dict) -> dict:
        layer = self.layer_of_agent[agent]
        tier = request.get("tier")
        # Defence in depth for direct callers: an unknown tier must never fall
        # through to the expensive T1 candidate path.
        if tier not in _ALLOWED_TIERS:
            return {"ok": False, "error": "unsupported measure tier"}
        valid_binding, binding_error = self.pipeline.validate_workspace_binding(
            layer, request.get("workspace_binding")
        )
        if not valid_binding:
            return _workspace_refresh_required(binding_error)
        if tier == "promote":
            out = self.pipeline.promote_by_id(str(request.get("node_id", "")))
            return {"ok": bool(out.get("promoted")), "tier": "promote", **out}
        if tier == "t0":
            # Async result application can promote another role after the
            # initial capability check. Revalidate and use all T0 inputs in
            # one critical section, like submit_candidate does for T1.
            # Production Pipeline always has this RLock; nullcontext supports
            # existing single-threaded protocol/security test doubles only.
            with getattr(self.pipeline, "_lock", None) or nullcontext():
                if not self._accepting.is_set():
                    return {"ok": False, "error": "measure service is quiescing for finalization",
                            "measurement_executed": False}
                valid_binding, binding_error = self.pipeline.validate_workspace_binding(
                    layer, request.get("workspace_binding"))
                if not valid_binding:
                    return _workspace_refresh_required(binding_error)
                artifact, error = self._read_artifact(agent, request.get("artifact"))
                if error is not None:
                    return {"ok": False, "error": error}
                assert artifact is not None
                if layer == "L2":
                    self.pipeline.ensure_t12()
                    valid_binding, binding_error = self.pipeline.validate_workspace_binding(
                        layer, request.get("workspace_binding"))
                    if not valid_binding:
                        return _workspace_refresh_required(binding_error)
                r = self.pipeline.service.t0(layer, artifact,
                                             canonical_text=self.pipeline.stack.read("T01"),
                                             t12_text=self.pipeline.stack.read("T12"))
                return {"ok": r.ok, "tier": "t0", **r.public}
        artifact, error = self._read_artifact(agent, request.get("artifact"))
        if error is not None:
            return {"ok": False, "error": error}
        assert artifact is not None
        assert tier == "t1"
        try:
            cand, r = self.pipeline.submit_candidate(
                layer,
                artifact,
                note=request.get("note", ""),
                expected_workspace_payload=request["workspace_binding"]["payload"],
            )
        except RuntimeError as error:
            if str(error).startswith("stale "):
                return {"ok": False, "error": str(error)}
            raise
        payload = {"ok": r.ok, "tier": r.tier, "node_id": cand.node_id, **r.public}
        if r.tier == "t1" and r.ok:
            launcher = trusted_launcher_path(self.workspaces[agent]).resolve()
            payload["promote_hint"] = (
                f"this node can be promoted; run `{launcher} promote {cand.node_id}` "
                "now (preferred — works mid-turn), or reply PROMOTE "
                f"{cand.node_id} at the end of your turn")
        return payload

    def serve_forever(self, stop_event, interval_s: float = 0.5) -> None:
        while self._accepting.is_set() and not stop_event.is_set():
            if self.poll_once() == 0:
                stop_event.wait(interval_s)
