"""Cooperative cancellation and deadlines for trusted long-running operations.

Subprocess calls poll ``OperationControl`` through ``run_cancellable``.
Multi-step operations check cancellation between steps. Reset the control only
after the exploration server has stopped, before finalizing the selected stack.
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from collections.abc import Sequence
from typing import Any


class OperationCancelled(RuntimeError):
    """A trusted operation was cancelled or exceeded its phase deadline."""


class OperationControl:
    """Thread-safe cancellation token with an optional monotonic deadline."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._deadline_monotonic: float | None = None
        self._reason = "operation cancelled"

    def reset(self, *, deadline_monotonic: float | None = None) -> None:
        """Begin a new phase.

        Resetting a token while an old operation is still running would allow
        that operation to escape cancellation.  The live driver therefore
        calls this only after ``MeasureServer.quiesce`` has joined its thread.
        """

        if deadline_monotonic is not None:
            deadline_monotonic = float(deadline_monotonic)
            if not math.isfinite(deadline_monotonic):
                raise ValueError("operation deadline must be finite")
        with self._lock:
            self._deadline_monotonic = deadline_monotonic
            self._reason = "operation cancelled"
            self._cancelled.clear()

    def cancel(self, reason: str = "operation cancelled") -> None:
        with self._lock:
            self._reason = str(reason)[:300] or "operation cancelled"
            self._cancelled.set()

    def is_cancelled(self) -> bool:
        if self._cancelled.is_set():
            return True
        with self._lock:
            deadline = self._deadline_monotonic
        return deadline is not None and time.monotonic() >= deadline

    def checkpoint(self) -> None:
        with self._lock:
            deadline = self._deadline_monotonic
            reason = self._reason
        if self._cancelled.is_set():
            raise OperationCancelled(reason)
        if deadline is not None and time.monotonic() >= deadline:
            # Latch deadline expiry so every sibling operation observes the
            # same terminal phase state until the driver explicitly resets it.
            self.cancel("operation phase deadline reached")
            raise OperationCancelled("operation phase deadline reached")

    def clamp_timeout(self, timeout_s: float) -> float:
        """Return a positive timeout bounded by the current phase deadline."""

        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("subprocess timeout must be positive and finite")
        self.checkpoint()
        with self._lock:
            deadline = self._deadline_monotonic
        if deadline is None:
            return timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self.checkpoint()
        return min(timeout, remaining)

    def wait(self, delay_s: float) -> None:
        """Interruptible retry backoff."""

        end = time.monotonic() + max(0.0, float(delay_s))
        while True:
            self.checkpoint()
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            if self._cancelled.wait(min(0.2, remaining)):
                self.checkpoint()


def _stop_process(process: subprocess.Popen[Any], grace_s: float = 2.0) -> None:
    """Bounded best-effort termination of one directly spawned process."""

    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        process.communicate(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        process.communicate(timeout=grace_s)
    except subprocess.TimeoutExpired:
        # The supervisor can still terminate the stage process group.
        pass


def run_cancellable(
    argv: Sequence[str],
    *,
    control: OperationControl | None,
    timeout: float,
    stop_event: threading.Event | None = None,
    input: str | bytes | None = None,
    capture_output: bool = False,
    text: bool = False,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[Any]:
    """A cancellation-aware subset of :func:`subprocess.run`.

    One communicator thread drains stdout/stderr while the caller polls for
    completion, cancellation, and timeout. The child remains in the stage
    process group, preserving the supervisor's hard-timeout backstop.
    """

    if stop_event is not None and stop_event.is_set():
        raise OperationCancelled("local operation stop requested")
    if control is not None:
        control.checkpoint()
        effective_timeout = control.clamp_timeout(timeout)
    else:
        effective_timeout = float(timeout)
    if not math.isfinite(effective_timeout) or effective_timeout <= 0:
        raise ValueError("subprocess timeout must be positive and finite")

    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    process = subprocess.Popen(
        tuple(argv),
        stdin=subprocess.PIPE if input is not None else None,
        stdout=stdout,
        stderr=stderr,
        text=text,
        cwd=cwd,
        env=env,
    )
    started = time.monotonic()
    communication: dict[str, Any] = {}
    communication_finished = threading.Event()

    def communicate_once() -> None:
        try:
            communication["result"] = process.communicate(input=input)
        except BaseException as error:  # propagated on the owning thread below
            communication["error"] = error
        finally:
            communication_finished.set()

    # ``Popen.communicate(input=..., timeout=...)`` cannot be safely restarted
    # with ``input=None`` on every Python/platform combination: stdin may stay
    # open after the first timeout and a child blocked in ``sys.stdin.read()``
    # will then wait forever.  One dedicated communicator owns all three pipes;
    # this thread only polls its completion so cancellation remains responsive.
    communicator = threading.Thread(
        target=communicate_once,
        name="leka-subprocess-communicator",
        daemon=True,
    )
    communicator.start()
    try:
        while True:
            if control is not None:
                control.checkpoint()
            if stop_event is not None and stop_event.is_set():
                raise OperationCancelled("local operation stop requested")
            remaining = effective_timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(tuple(argv), effective_timeout)
            if communication_finished.wait(min(0.2, remaining)):
                error = communication.get("error")
                if error is not None:
                    raise error
                out, err = communication["result"]
                return subprocess.CompletedProcess(
                    tuple(argv), process.returncode, out, err
                )
    except (OperationCancelled, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        if not communication_finished.wait(2.0) and process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            communication_finished.wait(2.0)
        raise
