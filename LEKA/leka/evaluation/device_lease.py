"""Host-wide, reentrant ownership of a physical device's measurement window.

Different devices remain independent. Profilers and ordinary measurements on
the same device share this lease, including across separate LEKA processes.
This is host-local coordination; unrelated programs and other hosts must still
be kept off the measurement device by the experiment operator.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import time


_registry_lock = threading.Lock()
_registry = {}


class DeviceLease:
    def __init__(self, identity: str, root: Path | None = None):
        root = root or Path(tempfile.gettempdir()) / f"leka-device-leases-{os.getuid()}"
        self.path = root / (hashlib.sha256(identity.encode()).hexdigest() + ".lock")
        self._lock = threading.RLock()
        self._depth = 0
        self._fd = None

    @contextmanager
    def hold(self, control=None, timeout_s=600):
        deadline = time.monotonic() + timeout_s

        def wait():
            if control is not None:
                control.checkpoint()
            if time.monotonic() >= deadline:
                raise TimeoutError("device is busy with another LEKA measurement")
            if control is not None:
                control.wait(0.05)
            else:
                time.sleep(0.05)

        while not self._lock.acquire(blocking=False):
            wait()
        try:
            if control is not None:
                control.checkpoint()
            if self._depth == 0:
                self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(self.path, flags, 0o600)
                try:
                    while True:
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            wait()
                    self._fd = fd
                except BaseException:
                    os.close(fd)
                    raise
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
                    os.close(self._fd)
                    self._fd = None
        finally:
            self._lock.release()


def device_lease(identity: str) -> DeviceLease:
    # A fork must not reuse a parent's thread lock or descriptor ownership.
    key = (os.getpid(), identity)
    with _registry_lock:
        return _registry.setdefault(key, DeviceLease(identity))
