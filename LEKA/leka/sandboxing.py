"""Process isolation helpers for parsing agent-authored TVMScript.

Static source validation is the primary language-level boundary.  These
helpers add a second process boundary: provider credentials are absent, Python
user-site loading is disabled, and macOS Seatbelt denies network access,
Keychain execution, writes under the user's home, and reads of common
credential stores.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence


_BASIC_ENVIRONMENT = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "VIVO_SERIAL",
    "VIVO_RPC_PORT",
    "VIVO_NUM_THREADS",
    "TVM_NDK_CC",
    "LEKA_VIVO_STATE_DIR",
    "LEKA_VIVO_DEVICE_LIB_DIR",
    "LEKA_JETSON_SSH_ALIAS",
    "LEKA_RK3588_SSH_ALIAS",
)


def candidate_environment(
    *,
    path: str,
    pythonpath: str,
    home: str | None = None,
    source: Mapping[str, str] | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a minimal candidate-worker environment with no provider secret."""

    inherited = os.environ if source is None else source
    environment = {
        name: inherited[name]
        for name in _BASIC_ENVIRONMENT
        if inherited.get(name)
    }
    environment.update(
        {
            "PATH": path,
            "PYTHONPATH": pythonpath,
            "PYTHONNOUSERSITE": "1",
            "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1",
        }
    )
    if home is not None:
        environment["HOME"] = home
    if extra:
        environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def _sbpl_string(path: Path | str) -> str:
    return json.dumps(str(Path(path).expanduser().resolve(strict=False)))


def macos_candidate_profile() -> str:
    """Return a defense-in-depth Seatbelt policy for local candidate workers.

    TVM and the Python runtime span system, Homebrew, uv, and repository paths,
    so the policy retains normal reads/executes.  The AST contract prevents
    arbitrary Python operations; Seatbelt independently blocks the useful
    exfiltration primitives and high-value credential locations.
    """

    home = Path.home()
    sensitive_subpaths = (
        home / ".ssh",
        home / ".aws",
        home / ".config" / "leka",
        home / ".claude",
        home / "Library" / "Keychains",
    )
    sensitive_files = (
        home / ".zshrc",
        home / ".zprofile",
        home / ".bashrc",
        home / ".bash_profile",
        home / ".netrc",
    )
    rules = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        f"(deny process-exec (literal {_sbpl_string('/usr/bin/security')}))",
        f"(deny file-write* (subpath {_sbpl_string(home)}))",
    ]
    rules.extend(
        f"(deny file-read* (subpath {_sbpl_string(path)}))"
        for path in sensitive_subpaths
    )
    rules.extend(
        f"(deny file-read* (literal {_sbpl_string(path)}))"
        for path in sensitive_files
    )
    return "\n".join(rules)


def sandboxed_candidate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Wrap a local candidate parser in Seatbelt when available on macOS."""

    command = tuple(str(part) for part in argv)
    sandbox_exec = Path("/usr/bin/sandbox-exec")
    require = os.environ.get("LEKA_REQUIRE_CANDIDATE_OS_SANDBOX") == "1"
    already_confined = os.environ.get("CODEX_SANDBOX") == "seatbelt"
    if require and not already_confined:
        if sys.platform != "darwin":
            raise RuntimeError("required candidate OS sandbox is only configured on macOS")
        if not sandbox_exec.is_file():
            raise RuntimeError("required macOS sandbox-exec is unavailable")
    if (
        sys.platform == "darwin"
        and sandbox_exec.is_file()
        and (require or not already_confined)
    ):
        return (str(sandbox_exec), "-p", macos_candidate_profile(), *command)
    return command


__all__ = [
    "candidate_environment",
    "macos_candidate_profile",
    "sandboxed_candidate_argv",
]
