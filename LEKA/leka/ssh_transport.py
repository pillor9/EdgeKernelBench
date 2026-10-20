"""Standard-library-only classification of SSH connection failures.

Used by both measurement transport and environment diagnostics, including
installations where numerical/compiler dependencies are unavailable.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

SSH_DEVICE_UNAVAILABLE_MARKERS = (
    "host is down",
    "no route to host",
    "network is unreachable",
    "connection timed out",
    "operation timed out",
    "connection refused",
)


def network_unavailable_marker(proc) -> str | None:
    if isinstance(proc, subprocess.TimeoutExpired):
        return None
    # A compiler/runtime error can quote network-related text.  Only an
    # unsuccessful SSH transport with a client-shaped connection diagnostic
    # may stop the whole backend; remote stdout is never transport evidence.
    argv = getattr(proc, "args", ())
    command = Path(argv[0]).name if isinstance(argv, (tuple, list)) and argv else ""
    codes = {255} if command in {"ssh", "scp"} else {10, 12, 255} if command == "rsync" else set()
    if getattr(proc, "returncode", None) not in codes:
        return None
    stderr = getattr(proc, "stderr", None)
    if not isinstance(stderr, str):
        return None
    for line in stderr.splitlines():
        match = re.fullmatch(
            r"ssh: connect to host [^\r\n]+ port [0-9]+: (.+)",
            line.strip(), flags=re.IGNORECASE,
        )
        if match is not None:
            reason = match.group(1).strip().lower()
            if reason in SSH_DEVICE_UNAVAILABLE_MARKERS:
                return reason
    return None
