"""Local run-history tests. No model or device access is required."""

import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from leka.worktree.candidate_workspace import PipelineRepo


@unittest.skipUnless(shutil.which("git"), "git is required")
class PipelineRepoTests(unittest.TestCase):
    def test_history_uses_neutral_identity_without_global_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            inherited = {
                "GIT_CONFIG_GLOBAL": str(root / "missing-config"),
                "GIT_AUTHOR_NAME": "external-author",
                "GIT_AUTHOR_EMAIL": "author@private.invalid",
                "GIT_COMMITTER_NAME": "external-committer",
                "GIT_COMMITTER_EMAIL": "committer@private.invalid",
                "GIT_DIR": str(root / "foreign-repository"),
                "GIT_INDEX_FILE": str(root / "foreign-index"),
            }
            with patch.dict(os.environ, inherited):
                repository = PipelineRepo(run)
                repository.ensure()
                (run / "task_manifest.json").write_text('{}\n', encoding="utf-8")
                repository.commit("record task")
                identities = repository._git(
                    "log", "--format=%an|%ae|%cn|%ce"
                ).stdout.splitlines()
            self.assertEqual(identities, [
                "LEKA|anonymous@example.invalid|LEKA|anonymous@example.invalid"
            ] * 2)
            self.assertFalse((root / "foreign-repository").exists())
            self.assertFalse((root / "foreign-index").exists())

    def test_run_history_does_not_execute_hooks_or_require_signing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = PipelineRepo(root / "run")
            repository.ensure()
            repository._git("config", "commit.gpgsign", "true")
            hook = repository.root / ".git" / "hooks" / "pre-commit"
            hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            hook.chmod(0o700)
            (repository.root / "task_manifest.json").write_text('{}\n', encoding="utf-8")
            repository.commit("record task")
            self.assertEqual(repository._git("log", "-1", "--format=%s").stdout.strip(),
                             "record task")


if __name__ == "__main__":
    unittest.main()
