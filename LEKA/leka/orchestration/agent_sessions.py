"""Run one persistent Claude Code stream-json session per agent.

Retryable provider failures retain the same session.

Each agent gets an isolated workspace with a private CLAUDE_CONFIG_DIR and a
scoped settings file (edit/write inside the workspace, plus one trusted measure
launcher outside it).  Messages go in as stream-json user turns; events stream
back into an evaluator-private log (workspace/session.log only for legacy
callers); the final "result" event of each turn is returned.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

from .evaluation_server import install_shim, validate_workspace_path
from ..provider_policy import (
    claude_provider_environment, is_claude_synthetic_api_error, provider_policy,
    response_model_evidence, validate_claude_provider_argv, require_model,
    secret_environment_names,
)


_FILE_TOOLS = ["Read", "Edit", "Write", "Glob", "Grep"]

# ``--restricted`` removes command-running tools unless they are explicitly
# named with ``--tools``.  Bash is made available here solely so the scoped
# permission above can admit ``./measure``; it does not grant arbitrary Bash.
_RESTRICTED_TOOLS = "Read,Edit,Write,Glob,Grep,Bash"

_IMMUTABLE_WORKSPACE_FILES = (
    "PYTORCH_SEMANTICS.json",
    "PYTORCH_TASK.py",
    "IMPORTED_RELAX.py",
    "PARAMETER_PLAN_SCHEMA.json",
    "canonical_tir.py",
    "generated.cu",
    "generated.cl",
    "generated.ll",
    "TASK_FACTS.json",
    "BASE_BINDING.json",
    "INTERFACE_FACTS.json",
    "PROFILE_DIGEST.json",
    "QUANTITY_CONTRACT.json",
    "HYPOTHESES.json",
    "EVIDENCE_SCHEMA.json",
    "EVIDENCE_FEEDBACK.json",
    "HELP_RESPONSE.json",
    "TURN_CONTEXT.json",
    "TURN_AUTHORITY.json",
    "PEER_INBOX.json",
    "HW_EVIDENCE_INDEX.json",
    "L2_CANDIDATE_ORIGIN.json",
)

_IMMUTABLE_WORKSPACE_DIRECTORIES = (
    ".claude", ".git", "wiki", "stack_view", "operator_tree", "HW_EVIDENCE",
)

_PROCESS_EXIT_DRAIN_GRACE_S = 1.0
_NONRETRYABLE_API_ERROR = re.compile(r"^\s*API Error:\s*(401|402|403)\b", re.I)


class AgentProcessExited(RuntimeError):
    """The resident Claude transport ended before a requested turn completed."""

    def __init__(self, name: str, *, reason: str, returncode: int | None):
        status = (
            f"return code {returncode}"
            if returncode is not None
            else "return code unavailable"
        )
        super().__init__(f"Claude CLI session {name!r} exited ({status}): {reason}")
        self.name = name
        self.reason = reason
        self.returncode = returncode


def _git_boundary(workspace: Path) -> Path | None:
    # The workspace is model-writable and may never nominate itself as a trust
    # root.  In particular, ignore an embedded ``workspace/.git`` entirely.
    boundary = workspace.parent.resolve()
    while boundary != boundary.parent:
        git_dir = boundary / ".git"
        if git_dir.is_dir() and not git_dir.is_symlink():
            return boundary
        boundary = boundary.parent
    return None


class AgentSession:
    def __init__(self, name: str, workspace: Path, *, profile_text: str,
                 model: str | None = None, extra_env: dict | None = None,
                 max_turns: int | None = None, log_path: Path | None = None,
                 model_effort: str | None = None):
        if max_turns is not None and (isinstance(max_turns, bool)
                                     or not isinstance(max_turns, int) or max_turns < 1):
            raise ValueError("max_turns must be an integer >= 1 or None")
        self.name = name
        self.workspace = workspace
        self.profile_text = profile_text
        self.model = model
        self.model_effort = model_effort
        self.extra_env = extra_env or {}
        self.max_turns = max_turns
        self.log_path = Path(log_path) if log_path is not None else workspace / "session.log"
        self.provider_policy = provider_policy(self.model, model_effort=self.model_effort)
        self._response_model_events: list[dict] = []
        self.proc: subprocess.Popen | None = None
        self._log = None
        self._completed_results: deque[dict] = deque()
        self._unowned_result_count = 0
        self._turn_done = threading.Event()
        self._state_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._pending = False
        # EOF is a terminal transport event even in the narrow race where the
        # child has closed stdout but waitpid has not exposed its return code
        # yet.  Without retaining that state, a pending turn sleeps until its
        # full budget and every later send writes into a dead session.
        self._terminal: tuple[str, int | None] | None = None
        self._reader: threading.Thread | None = None
        self._watcher: threading.Thread | None = None
        # Cooperative workspace delivery, never an extra user message on the
        # resident transport. The driver callback has no measurement authority.
        self.poll_callback: Callable[[], None] | None = None

    # -- workspace -------------------------------------------------------------
    def prepare(self) -> None:
        ws = self.workspace
        validate_workspace_path(ws)
        self._measure_path = install_shim(ws)
        git_boundary = _git_boundary(ws)
        security_boundary = git_boundary or ws.resolve().parent
        settings = {
            "permissions": {
                "allow": [
                    *_FILE_TOOLS,
                    f"Bash({self._measure_path}:*)",
                ],
                "deny": [
                    "WebFetch", "WebSearch", "Task",
                    # Claude's file-edit permission category covers both Edit
                    # and Write, including resolved symlink targets. Built-in
                    # tools do not run inside the Bash filesystem sandbox;
                    # atomic rename can replace a chmod-444 file in a writable
                    # directory, so enforce these as tool deny rules as well.
                    *[f"Edit({name})" for name in _IMMUTABLE_WORKSPACE_FILES],
                    *[f"Edit({name}/**)" for name in _IMMUTABLE_WORKSPACE_DIRECTORIES],
                    # Do not deny Edit on requests/responses: Claude merges
                    # Edit deny paths into the Bash sandbox's denyWrite list.
                    # The trusted launcher must create its request there. The
                    # queue is untrusted input; MeasureServer validates paths,
                    # capabilities and tiers before any authoritative action.
                    # Claude otherwise auto-approves several read-only shell
                    # commands.  The OS sandbox below is authoritative; these
                    # rules make the intended tool surface explicit too.
                    "Bash(cat:*)", "Bash(ls:*)", "Bash(head:*)",
                    "Bash(tail:*)", "Bash(grep:*)", "Bash(find:*)",
                    "Bash(wc:*)", "Bash(diff:*)", "Bash(stat:*)",
                    "Bash(du:*)", "Bash(cd:*)", "Bash(git:*)",
                ],
                "defaultMode": "dontAsk",
            },
            # Bash and its children are OS-confined.  The run Git root holds
            # the trusted manifest/stack and sibling workspaces; home holds the
            # benchmark source and credentials.  Re-open only this workspace
            # plus the exact immutable-to-the-agent measure launcher.
            "sandbox": {
                "enabled": True,
                "failIfUnavailable": True,
                "allowUnsandboxedCommands": False,
                "filesystem": {
                    "denyRead": [
                        str(Path.home().resolve()),
                        str(security_boundary.resolve()),
                    ],
                    "allowRead": [
                        str(ws.resolve()),
                        str(self._measure_path),
                    ],
                    "denyWrite": [
                        str(self._measure_path),
                        *[str((ws / name).resolve())
                          for name in _IMMUTABLE_WORKSPACE_DIRECTORIES],
                        *[
                            str((ws / name).resolve())
                            for name in _IMMUTABLE_WORKSPACE_FILES
                        ],
                    ],
                },
                "network": {"allowedDomains": [], "strictAllowlist": True},
                "credentials": {
                    "envVars": [
                        {"name": name, "mode": "deny"}
                        for name in secret_environment_names({**os.environ, **self.extra_env})
                    ]
                },
            },
        }
        claude_dir = ws / ".claude"
        if claude_dir.is_symlink() or (claude_dir.exists() and not claude_dir.is_dir()):
            raise RuntimeError("workspace .claude must be a real directory")
        claude_dir.mkdir(parents=True, exist_ok=True)
        settings_path = ws / ".claude" / "settings.json"
        # A previous prepare may have left the file read-only.  Recreate it
        # deterministically, then make accidental mutation outside Claude's
        # restricted-mode policy harder as well.
        if settings_path.is_symlink() or (
            settings_path.exists() and not settings_path.is_file()
        ):
            raise RuntimeError("workspace settings path must be a regular file")
        if settings_path.exists():
            settings_path.chmod(0o600)
        settings_path.write_text(json.dumps(settings, indent=2))
        settings_path.chmod(0o400)
        self._settings_path = settings_path.resolve()
        notes_path = ws / "NOTES.md"
        if notes_path.is_symlink() or (notes_path.exists() and not notes_path.is_file()):
            raise RuntimeError("workspace notes path must be a regular file")
        notes_path.touch()
        config_dir = (
            git_boundary / "private" / "claude-configs" / self.name
            if git_boundary is not None
            else ws.parent / f".leka-claude-{self.name}"
        )
        if config_dir.is_symlink() or (config_dir.exists() and not config_dir.is_dir()):
            raise RuntimeError("Claude config path must be a real directory")
        config_dir.mkdir(parents=True, exist_ok=True)
        self._config_dir = config_dir
        # Pre-trust the workspace, else the settings.json allow list is ignored
        # and Bash(./measure) is denied.
        # macOS canonicalizes /tmp to /private/tmp, so register both spellings.
        trust_paths = {str(ws.resolve()), str(ws)}
        # Claude Code scopes project trust to the nearest Git boundary rather
        # than to the process cwd.  LEKA workspaces are nested under the run's
        # private Git repository, so trust both identities explicitly.
        if git_boundary is not None:
            trust_paths.add(str(git_boundary))
        config_file = config_dir / ".claude.json"
        if config_file.is_symlink() or (
            config_file.exists() and not config_file.is_file()
        ):
            raise RuntimeError("Claude config file must be a regular file")
        config = json.loads(config_file.read_text()) if config_file.exists() else {}
        projects = config.setdefault("projects", {})
        for p in trust_paths:
            projects.setdefault(p, {})["hasTrustDialogAccepted"] = True
        config_file.write_text(json.dumps(config, indent=2))

    def start(self) -> None:
        require_model(self.model)
        self.prepare()
        with self._state_lock:
            if self._pending or self._completed_results:
                raise RuntimeError("cannot start a session with undelivered turn state")
            self._terminal = None
            self._response_model_events.clear()
            self._turn_done.clear()
        env = dict(os.environ)
        env["CLAUDE_CONFIG_DIR"] = str(self._config_dir)
        env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
        env.update(self.extra_env)
        env = claude_provider_environment(self.model, env)
        # Provider credentials remain available to the parent Claude process
        # for model requests but are stripped from Bash/hooks/MCP children.
        # On Linux this also hides the parent environment through /proc.  Set
        # these after extra_env so caller overrides cannot weaken them.
        env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        measure_notice = (
            "\n\nSECURITY-SCOPED MEASUREMENT COMMAND: replace every older "
            f"`./measure` example with `{self._measure_path}`. This exact trusted "
            "absolute launcher is the only Bash command available; do not create or "
            "run a workspace-local measure executable."
        )
        scoped_profile = self.profile_text.replace("./measure", str(self._measure_path))
        argv = [os.environ.get("LEKA_AGENT_BINARY", "claude"), "--input-format", "stream-json", "--output-format", "stream-json",
                "--verbose", "-p", "--restricted", "--strict-mcp-config",
                "--tools", _RESTRICTED_TOOLS,
                "--settings", str(self._settings_path),
                "--name", f"leka-{self.name}",
                "--append-system-prompt", scoped_profile + measure_notice]
        if self.model:
            argv += ["--model", self.model]
        if self.model_effort is not None:
            argv += ["--effort", self.model_effort]
        if self.max_turns is not None:
            argv += ["--max-turns", str(self.max_turns)]
        argv += ["--setting-sources", ""]
        validate_claude_provider_argv(self.model, argv, model_effort=self.model_effort)
        # Recompute only from the validated launch options. Neither an env
        # override nor model-authored result metadata can claim another effort.
        self.provider_policy = provider_policy(self.model, model_effort=self.model_effort)
        log_path = self.log_path
        if log_path.is_symlink() or (log_path.exists() and not log_path.is_file()):
            raise RuntimeError("session log path must be a regular file")
        self._log = log_path.open("a")
        log_path.chmod(0o600)
        self.proc = subprocess.Popen(
            argv, cwd=str(self.workspace), env=env, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            # The run supervisor owns one process group per stage. Claude
            # must inherit that group so a stage timeout/termination reaches the
            # provider process (and its descendants) too.
            start_new_session=False,
        )
        self._reader = threading.Thread(target=self._read_events, daemon=True)
        self._reader.start()
        # A crashed CLI can leave stdout inherited by one of its descendants,
        # in which case EOF alone is not a reliable exit notification.  Poll
        # the resident parent independently and wake waiters after a short
        # drain grace for any final buffered result event.
        self._watcher = threading.Thread(target=self._watch_process, daemon=True)
        self._watcher.start()

    def _watch_process(self) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            while True:
                returncode = proc.poll()
                if returncode is not None:
                    break
                time.sleep(0.05)
        except Exception as error:
            self._mark_terminal(
                f"process monitor failed ({type(error).__name__})", None
            )
            return
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=_PROCESS_EXIT_DRAIN_GRACE_S)
        self._mark_terminal("process exited", returncode)

    def _read_events(self) -> None:
        assert self.proc and self.proc.stdout
        failure: str | None = None
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                self._log.write(line + "\n")
                self._log.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._record_event(event)
        except Exception as error:
            # Do not echo stream contents or provider diagnostics through this
            # control-plane exception; the private session log remains the
            # detailed diagnostic authority.
            failure = f"stdout reader failed ({type(error).__name__})"
        finally:
            proc = self.proc
            returncode = proc.poll() if proc is not None else None
            self._mark_terminal(failure or "stdout reached EOF", returncode)

    def _mark_terminal(self, reason: str, returncode: int | None) -> None:
        """Publish terminal transport state and wake every pending waiter."""

        with self._state_lock:
            if self._terminal is None:
                self._terminal = (reason, returncode)
            elif self._terminal[1] is None and returncode is not None:
                # EOF can beat waitpid by a few milliseconds.  Preserve the
                # first (most specific) reason while enriching later reports
                # with the eventual process status observed by the watcher.
                self._terminal = (self._terminal[0], returncode)
            self._turn_done.set()

    def _detect_process_exit_locked(self) -> None:
        """Close the poll-before-reader race while holding ``_state_lock``."""

        if self._terminal is not None or self.proc is None:
            return
        returncode = self.proc.poll()
        if returncode is not None:
            self._terminal = ("process exited", returncode)
            self._turn_done.set()

    def _terminal_exception_locked(self) -> AgentProcessExited | None:
        self._detect_process_exit_locked()
        if self._terminal is None:
            return None
        reason, returncode = self._terminal
        return AgentProcessExited(
            self.name, reason=reason, returncode=returncode
        )

    def _record_event(self, event: dict) -> None:
        """Record one parsed CLI event, preserving every completed turn FIFO."""

        with self._state_lock:
            if self.model:
                # Retain identity-only observations, not assistant contents.
                message = event.get("message")
                identity = {"type": event.get("type")}
                if isinstance(message, dict) and isinstance(message.get("model"), str):
                    identity["message"] = {"model": message["model"]}
                for marker in ("isApiErrorMessage", "is_api_error_message", "apiError", "api_error"):
                    if marker in event:
                        identity[marker] = event[marker]
                if isinstance(event.get("modelUsage"), dict):
                    identity["modelUsage"] = {name: {} for name in event["modelUsage"]}
                if len(identity) > 1 and identity not in self._response_model_events:
                    self._response_model_events.append(identity)
                try:
                    evidence = response_model_evidence(self.model, self._response_model_events)
                except ValueError:
                    self._terminal = ("response model identity mismatch", None)
                    self._turn_done.set()
                    return
                if is_claude_synthetic_api_error(event):
                    # Claude Code emits a synthetic assistant envelope followed
                    # by an ``is_error`` result for a failed API turn.  The
                    # resident CLI process remains usable (and retains the
                    # conversation), so do not turn a retryable *turn* failure
                    # into a terminal *session* failure.  Wait for the trailing
                    # result, deliver it to the scheduler, and let the next
                    # round retry through this same stdin/session/workspace.
                    #
                    # The identity evidence above retains the precise error
                    # class for the failed turn.  Real EOF/process exit and
                    # response-model mismatch remain terminal.
                    return
                if event.get("type") == "result":
                    reply = event.get("result", "")
                    nonretryable = (
                        _NONRETRYABLE_API_ERROR.match(reply)
                        if isinstance(reply, str)
                        else None
                    )
                    if event.get("is_error") is True and nonretryable is not None:
                        self._terminal = (
                            f"non-retryable provider API error {nonretryable.group(1)}",
                            None,
                        )
                        self._turn_done.set()
                        return
                    event = {**event, "provider_policy": self.provider_policy,
                             "provider_model_identity": evidence}
            if event.get("type") == "result":
                self._completed_results.append(event)
                if not self._pending:
                    self._archive_unowned_results_locked()
                    return
                self._turn_done.set()

    def _archive_unowned_results_locked(self) -> None:
        """Keep extra CLI results in the raw log, not in a future turn's queue.

        Only one user request can be pending. Extra completion envelopes cannot
        be assigned a new workspace binding or replayed as another PROMOTE.
        _read_events already persisted their full original bytes to session.log.
        """
        count = len(self._completed_results)
        if not count:
            return
        self._completed_results.clear()
        self._unowned_result_count += count
        if self._log is not None:
            self._log.write(json.dumps({
                "type": "leka_session_diagnostic",
                "event": "unowned_result_archived",
                "count": count,
                "action": "retain_raw_log_without_replaying_as_new_turn",
            }) + "\n")
            self._log.flush()
        if not self._pending and self._terminal is None:
            self._turn_done.clear()

    def has_pending(self) -> bool:
        """Whether a previously submitted turn still awaits driver delivery."""

        with self._state_lock:
            return self._pending

    def _finish_pending_locked(self, timeout_s: float) -> dict | None:
        with self._state_lock:
            if self._completed_results:
                if not self._pending:
                    self._archive_unowned_results_locked()
                    return None
                event = self._completed_results.popleft()
                self._pending = False
                self._archive_unowned_results_locked()
                if not self._completed_results and self._terminal is None:
                    self._turn_done.clear()
                return event
            terminal = self._terminal_exception_locked()
            if terminal is not None:
                self._pending = False
                raise terminal
            if not self._pending:
                return None
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            if self.poll_callback is not None:
                self.poll_callback()
            remaining = max(0.0, deadline - time.monotonic())
            interval = min(0.25, remaining) if self.poll_callback else remaining
            if self._turn_done.wait(timeout=interval):
                completed = True
                break
            if time.monotonic() >= deadline:
                completed = False
                break
        if not completed:
            # EOF and timeout can race at the wait boundary.  Re-check under
            # the state lock so an already-observed exit is never reported as
            # an ordinary still-running turn.
            with self._state_lock:
                terminal = self._terminal_exception_locked()
                if terminal is not None:
                    self._pending = False
                    raise terminal
                return None
        with self._state_lock:
            if self._completed_results:
                event = self._completed_results.popleft()
                self._pending = False
                self._archive_unowned_results_locked()
                if not self._completed_results and self._terminal is None:
                    self._turn_done.clear()
                return event
            terminal = self._terminal_exception_locked()
            if terminal is not None:
                self._pending = False
                raise terminal
            raise RuntimeError("turn completion signalled without a result event")

    def finish_pending(self, *, timeout_s: float = 1800) -> dict | None:
        """Wait for and deliver the outstanding turn without submitting another."""

        with self._send_lock:
            return self._finish_pending_locked(timeout_s)

    # -- messaging ---------------------------------------------------------------
    def send(self, text: str, *, timeout_s: float = 1800) -> dict | None:
        """One turn: deliver text, wait for the turn's result event (or timeout).

        A timed-out turn keeps running inside the CLI.  If callers invoke send()
        again before consuming it, this method returns that previous turn's
        result and deliberately does *not* inject ``text``.  This guarantees the
        driver observes each formal reply before a later turn can replace it.
        """
        text = text.replace("./measure", str(self._measure_path))
        with self._send_lock:
            with self._state_lock:
                terminal = self._terminal_exception_locked()
                if terminal is not None:
                    self._pending = False
                    raise terminal
                pending = self._pending
            if pending:
                return self._finish_pending_locked(timeout_s)

            proc = self.proc
            if proc is None or proc.stdin is None:
                raise RuntimeError("Claude CLI session has not been started")

            with self._state_lock:
                if self._completed_results:
                    self._archive_unowned_results_locked()
                # Identity and synthetic-error evidence is per turn.  Clearing
                # it here prevents one recovered transport failure from
                # labelling every later successful reply in the same session.
                self._response_model_events.clear()
                self._turn_done.clear()
                self._pending = True
            message = {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                },
            }
            try:
                proc.stdin.write(json.dumps(message) + "\n")
                proc.stdin.flush()
            except Exception as error:
                self._mark_terminal(
                    f"stdin write failed ({type(error).__name__})", proc.poll()
                )
                with self._state_lock:
                    self._pending = False
                    terminal = self._terminal_exception_locked()
                assert terminal is not None
                raise terminal from error
            return self._finish_pending_locked(timeout_s)

    def alive(self) -> bool:
        with self._state_lock:
            if self._terminal_exception_locked() is not None:
                return False
            return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        proc = self.proc
        if proc and proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Do not let one broken CLI prevent cleanup of the
                        # remaining resident sessions.  The enclosing stage
                        # process group is the final run-level backstop.
                        pass
        if proc and proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass
        if self._reader and self._reader is not threading.current_thread():
            self._reader.join(timeout=5)
        if self._watcher and self._watcher is not threading.current_thread():
            self._watcher.join(timeout=5)
        if self._log:
            self._log.close()
