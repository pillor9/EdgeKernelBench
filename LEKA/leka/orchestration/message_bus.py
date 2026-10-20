"""File-based per-agent inboxes.  One JSON file per message, FIFO by name."""

from __future__ import annotations

import json
import time
import threading
import uuid
from pathlib import Path


class Mailbox:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.RLock()

    def _inbox(self, agent: str) -> Path:
        p = self.root / "inbox" / agent
        p.mkdir(parents=True, exist_ok=True)
        return p

    def send(self, agent: str, kind: str, payload: dict) -> Path:
        inbox = self._inbox(agent)
        name = f"{time.time_ns()}_{uuid.uuid4().hex}_{kind}.json"
        path = inbox / name
        temporary = inbox / f".{uuid.uuid4().hex}.tmp"
        # A measurement callback can send while another layer drains its
        # mailbox. Only completed JSON becomes visible to consumers.
        with self._lock:
            try:
                with temporary.open("x", encoding="utf-8") as stream:
                    json.dump({**payload, "kind": kind, "sent_at": time.time()},
                              stream, indent=2, allow_nan=False)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path

    def pending(self, agent: str) -> list[Path]:
        return sorted(self._inbox(agent).glob("*.json"))

    def pop(self, agent: str) -> dict | None:
        with self._lock:
            items = self.pending(agent)
            if not items:
                return None
            payload = json.loads(items[0].read_text())
            items[0].unlink()
            return payload
