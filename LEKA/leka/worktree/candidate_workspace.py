"""Per-pipeline git repository: content versioning, crash recovery, diff engine."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..operation_control import OperationControl, run_cancellable


_GITIGNORE = """\
agents/
candidates/
inbox/
jobs/
private/
requests/
responses/
session.log
*.log
*.pid
*.npz
"""

# Only driver-owned state needed to identify and review a promoted stack is
# versioned.  Candidate workspaces and measurement traffic deliberately never
# enter the per-run Git history.
_TRUSTED_PATHS = (
    ".gitignore",
    "task_manifest.json",
    "ledger.jsonl",
    "stack/stack.json",
    "stack/relax_mod.py",
    "stack/canonical_tir.py",
    "stack/scheduled_tir.py",
    "stack/generated.cu",
    "stack/patched.cu",
    "stack/generated.cl",
    "stack/patched.cl",
    "stack/generated.ll",
    "stack/patched.ll",
)


class PipelineRepo:
    def __init__(self, root: Path):
        self.root = root
        self.operation_control: OperationControl | None = None

    def set_operation_control(self, control: OperationControl) -> None:
        self.operation_control = control

    def _run_git(self, *argv: str) -> subprocess.CompletedProcess:
        # Run history must not inherit the host's identity, hooks, or repository.
        environment = {name: value for name, value in os.environ.items()
                       if not name.startswith("GIT_")}
        environment.update({
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "LEKA",
            "GIT_AUTHOR_EMAIL": "anonymous@example.invalid",
            "GIT_COMMITTER_NAME": "LEKA",
            "GIT_COMMITTER_EMAIL": "anonymous@example.invalid",
        })
        proc = run_cancellable(
            ("git", "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={os.devnull}",
             "-C", str(self.root), *argv),
            control=self.operation_control,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
        return proc

    def _git(self, *argv: str) -> subprocess.CompletedProcess:
        proc = self._run_git(*argv)
        proc.check_returncode()
        return proc

    def _stage_trusted(self) -> None:
        tracked = {
            path
            for path in self._git("ls-files", "-z").stdout.split("\0")
            if path
        }
        untrusted = sorted(tracked.difference(_TRUSTED_PATHS))
        # A repository created by an older LEKA version may already track
        # session traffic.  Stop carrying it forward without deleting any
        # working-tree file; old history is deliberately left untouched.
        for offset in range(0, len(untrusted), 128):
            self._git(
                "rm", "--cached", "--force", "--ignore-unmatch", "--",
                *untrusted[offset:offset + 128],
            )
        selected: list[str] = []
        for relative in _TRUSTED_PATHS:
            path = self.root / relative
            if path.is_symlink():
                raise RuntimeError(f"trusted pipeline state may not be a symlink: {relative}")
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(
                        f"trusted pipeline state must be a regular file: {relative}"
                    )
                selected.append(relative)
            elif relative in tracked:
                # ``git add <exact-path>`` also records deletion of an already
                # tracked file, which is needed when an upper-layer promotion
                # invalidates generated/patched lower slots.
                selected.append(relative)
        if selected:
            self._git("add", "--", *selected)

    def ensure(self) -> None:
        if self.root.is_symlink():
            raise RuntimeError("pipeline repository root may not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        git_dir = self.root / ".git"
        if git_dir.is_symlink() or (git_dir.exists() and not git_dir.is_dir()):
            raise RuntimeError("pipeline .git must be a real directory")
        new_repository = not git_dir.is_dir()
        if new_repository:
            self._git("init", "--quiet")
        ignore = self.root / ".gitignore"
        if ignore.is_symlink():
            raise RuntimeError("pipeline .gitignore may not be a symlink")
        ignore.write_text(_GITIGNORE)
        self._stage_trusted()
        if new_repository:
            self._git("commit", "--quiet", "-m", "init", "--allow-empty")

    def commit(self, message: str) -> None:
        self._stage_trusted()
        staged = self._run_git("diff", "--cached", "--quiet")
        if staged.returncode == 0:
            return
        if staged.returncode != 1:
            raise RuntimeError(staged.stderr.strip() or "unable to inspect staged state")
        self._git("commit", "--quiet", "-m", message)

    def diff_summary(self, max_lines: int = 40) -> str:
        out = self._git("diff", "HEAD~1", "--stat").stdout
        return "\n".join(out.splitlines()[:max_lines])
