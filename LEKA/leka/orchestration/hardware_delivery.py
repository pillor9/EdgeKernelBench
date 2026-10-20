"""Read-only live projections of evaluator-sealed HW packets, separate from peers.

Only the trusted evidence-sealing callsite publishes here. No model file, request
or tool can import a packet into this store. IDs are integrity checks, not MACs;
authority comes from that callsite and the original frozen digest validation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import uuid

from ..hardware_feedback.evidence import MAX_PACKET_BYTES
from .shared_message_pool import binding_status

DIRECTORY = "HW_EVIDENCE"
INDEX = "HW_EVIDENCE_INDEX.json"
SCHEMA = "leka.live_hardware_evidence.v1"
FEEDBACK_SCHEMA = "leka.live_hardware_feedback.v1"
MAX_INLINE_RESPONSE_BYTES = 24 * 1024
MAX_NOTIFICATION_BYTES = 4096
MAX_NOTIFICATION_ITEMS = 3
MAX_INDEX_ITEMS = 12
MAX_STORED_PACKETS = 128
MAX_SAVED_PACKET_BYTES = 64 * 1024
_ID = re.compile(r"[0-9a-f]{64}\Z")


def _json(value, *, pretty=False):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       indent=2 if pretty else None) + "\n").encode("utf-8")


def compact_response_bytes(value):
    """The exact opt-in response wire encoding, including multibyte UTF-8."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def display_packet(packet):
    """Order a complete sealed copy; never derive facts or change any value/ID."""
    priority = ("evidence_id", "source_digest_id", "source_binding", "profile_data", "claim_assessment")
    model = ("target_layer", "verdict", "confidence", "rationale", "suggested_direction")
    order = [key for key in priority if key in packet]
    order += [key for key in packet if key not in priority and key not in model]
    order += [key for key in model if key in packet]
    return {key: copy.deepcopy(packet[key]) for key in order}


def _check_regular(fd, name):
    try:
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.getuid()):
        raise ValueError("hardware projection target must be an owned regular file without links")
    return info


def _atomic(fd, name, payload):
    _check_regular(fd, name)
    temporary = ".hw-pending-" + uuid.uuid4().hex
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=fd)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
        # Replacement never follows a swapped final symlink or hardlink.
        os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=fd)
        except FileNotFoundError:
            pass


def _open_workspace(path):
    # Parent is evaluator-configured; the workspace itself and every projection
    # descendant are opened without following links, including after startup.
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


class HardwareEvidenceDelivery:
    def __init__(self, pipe, workspaces, layers, *, backend, inline_feedback=False):
        self.pipe, self.backend = pipe, backend
        if type(inline_feedback) is not bool:
            raise ValueError("inline_feedback must be a boolean")
        self.inline_feedback = inline_feedback
        self.layers = dict(layers)
        self.workspaces = {}
        for agent in self.layers:
            workspace = Path(workspaces[agent]).absolute()
            if ".." in workspace.parts or workspace.is_symlink():
                raise ValueError("invalid hardware projection workspace")
            self.workspaces[agent] = workspace.parent.resolve() / workspace.name
        self._lock = threading.RLock()
        self._packets = {agent: {} for agent in self.layers}
        self._contexts = {}
        self._notified = {}
        self._sealed = {agent: {} for agent in self.layers}
        self._inline_delivered = {}
        for agent in self.layers:
            self._write_index(agent)

    def _directory(self, workspace_fd):
        try:
            os.mkdir(DIRECTORY, 0o555, dir_fd=workspace_fd)
        except FileExistsError:
            pass
        fd = os.open(DIRECTORY, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                     dir_fd=workspace_fd)
        if os.fstat(fd).st_uid != os.getuid():
            os.close(fd)
            raise ValueError("hardware projection directory has another owner")
        os.fchmod(fd, 0o555)
        return fd

    def _write_index(self, agent):
        entries = list(self._packets[agent].values())
        index = {
            "schema_version": SCHEMA, "agent": agent, "backend": self.backend,
            "authority": "evaluator_verified_source_with_untrusted_hypothesis",
            "delivery_semantics": "file_projection_only_not_read_or_adoption",
            "status_rule": "Source bindings are immutable; compare against your current turn. "
                           "This index does not assert current-stack applicability.",
            "entries": entries[-MAX_INDEX_ITEMS:], "total_packet_count": len(entries),
            "omitted_older_count": max(0, len(entries) - MAX_INDEX_ITEMS),
        }
        fd = _open_workspace(self.workspaces[agent])
        try:
            child = self._directory(fd)
            os.close(child)
            _atomic(fd, INDEX, _json(index, pretty=True))
        finally:
            os.close(fd)

    def begin(self, agent, context):
        if agent not in self.layers:
            return
        with self._lock:
            self._contexts[agent] = copy.deepcopy(context)
            self._notified[agent] = set()
            self._inline_delivered[agent] = set()

    def finish(self, agent):
        with self._lock:
            self._contexts.pop(agent, None)
        # Packets/index remain available; finishing a model turn never means
        # an observation was read or an evidence claim became true.

    def publish_sealed(self, agent, packet):
        if agent not in self.layers:
            return False
        if not isinstance(packet, dict):
            raise ValueError("sealed hardware packet must be an object")
        packet = copy.deepcopy(packet)
        identifier = packet.get("evidence_id")
        source_id = packet.get("source_digest_id")
        if (not isinstance(identifier, str) or not _ID.fullmatch(identifier)
                or not isinstance(source_id, str) or not _ID.fullmatch(source_id)):
            raise ValueError("invalid sealed hardware identity")
        expected = hashlib.sha256(json.dumps({k: v for k, v in packet.items() if k != "evidence_id"},
                                              sort_keys=True, allow_nan=False).encode()).hexdigest()
        if identifier != expected:
            raise ValueError("sealed hardware evidence checksum mismatch")
        if packet.get("target_layer") != self.layers[agent]:
            raise ValueError("sealed hardware target does not own this workspace")
        if not isinstance(packet.get("source_binding"), dict) or not packet["source_binding"]:
            raise ValueError("sealed hardware source binding is unavailable")
        if not isinstance(packet.get("claim_assessment"), dict):
            raise ValueError("sealed hardware claim assessment is unavailable")
        if len(json.dumps(packet, ensure_ascii=False, allow_nan=False).encode()) > MAX_PACKET_BYTES:
            raise ValueError("sealed hardware packet exceeds source contract")
        payload = _json(packet, pretty=True)
        if len(payload) > MAX_SAVED_PACKET_BYTES:
            raise ValueError("sealed hardware packet projection is too large; no packet was truncated")
        entry = {
            "evidence_id": identifier, "packet_file": f"{DIRECTORY}/{identifier}.json",
            "packet_sha256": hashlib.sha256(payload).hexdigest(), "packet_bytes": len(payload),
            "source_digest_id": source_id, "source_binding": packet["source_binding"],
            "target_layer": packet["target_layer"], "verdict": packet["verdict"],
        }
        with self._lock:
            known = self._packets[agent]
            if identifier not in known and len(known) >= MAX_STORED_PACKETS:
                raise ValueError("live hardware projection packet limit reached; formal mailbox retained")
            fd = _open_workspace(self.workspaces[agent])
            try:
                child = self._directory(fd)
                try:
                    name = identifier + ".json"
                    info = _check_regular(child, name)
                    if info is not None:
                        if info.st_size != len(payload):
                            raise ValueError("existing hardware packet content differs")
                        handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=child)
                        with os.fdopen(handle, "rb") as stream:
                            actual = os.fstat(stream.fileno())
                            if (not stat.S_ISREG(actual.st_mode) or actual.st_nlink != 1
                                    or actual.st_uid != os.getuid()
                                    or (actual.st_dev, actual.st_ino) != (info.st_dev, info.st_ino)):
                                raise ValueError("hardware packet changed during read")
                            if stream.read(MAX_SAVED_PACKET_BYTES + 1) != payload:
                                raise ValueError("existing hardware packet content differs")
                    else:
                        os.fchmod(child, 0o755)
                        _atomic(child, name, payload)
                finally:
                    os.fchmod(child, 0o555)
                    os.close(child)
            finally:
                os.close(fd)
            first = identifier not in known
            known[identifier] = entry
            try:
                self._write_index(agent)
            except Exception:
                if first:
                    known.pop(identifier, None)
                raise
            # The evaluator's successful publication is the only import path.
            # Later tool responses never reopen model-accessible projections.
            if self.inline_feedback:
                self._sealed[agent][identifier] = packet
            if first:
                self.pipe.log("hardware_evidence_projected", agent=agent,
                    evidence_id=identifier, packet_sha256=entry["packet_sha256"],
                    packet_bytes=len(payload), source_digest_id=source_id,
                    source_binding=packet["source_binding"],
                    meaning="complete_readonly_packet_written_not_model_read_or_adoption")
            return first

    def feedback_turn(self, agent):
        """Capture before _serve; a late response must retain its old admission."""
        if not self.inline_feedback:
            return None
        with self._lock:
            context = self._contexts.get(agent)
            return context.get("turn_id") if context else None

    def prepare_feedback(self, agent, *, request_id, tier, result_binding,
                         turn_id, base_response):
        """Return (optional envelope, optional deferral), without consuming it.

        Prefer current-binding packets, then newest publication within each
        status class, without interpreting verdicts or speed. Path notifications
        and complete-packet delivery have independent per-turn state. Current
        binding is sampled without waiting for collection or taking the
        Pipeline lock under this store's lock.
        """
        if not self.inline_feedback or turn_id is None:
            return None, None
        deferred = {"turn_id": turn_id, "request_id": request_id, "measure_tier": tier}
        current = None
        if self.pipe._lock.acquire(blocking=False):
            try:
                current = self.pipe.stack.binding().to_dict()
            finally:
                self.pipe._lock.release()
        with self._lock:
            context = self._contexts.get(agent)
            if context is None or context.get("turn_id") != turn_id:
                return None, {**deferred, "reason": "recipient_turn_changed", "fallback_in_response": False}
            context = copy.deepcopy(context)
            pending = [(copy.deepcopy(entry), self._sealed[agent][identifier])
                       for identifier, entry in self._packets[agent].items()
                       if identifier in self._sealed[agent]
                       and identifier not in self._inline_delivered.get(agent, set())]
        if not pending:
            return None, None
        pending.reverse()
        pending.sort(key=lambda item: binding_status(item[0]["source_binding"], current) != "current")
        if "hardware_evidence_feedback" in base_response:
            raise ValueError("reserved inline feedback field already present")
        source = pending[0][0]
        deferred.update(evidence_id=source["evidence_id"], source_digest_id=source["source_digest_id"],
                        packet_file=source["packet_file"], packet_sha256=source["packet_sha256"])
        if len(compact_response_bytes(base_response)) > MAX_INLINE_RESPONSE_BYTES:
            return None, {**deferred, "reason": "base_response_exceeds_inline_budget", "fallback_in_response": False}
        first = None
        for entry, packet in pending:
            feedback = {
                "schema_version": FEEDBACK_SCHEMA, "status": "complete_packet",
                "agent": agent, "turn_id": turn_id, "request_id": request_id, "measure_tier": tier,
                "authority": "evaluator_sealed_source_values_and_checks_not_model_semantic_validation",
                "completeness": "one_complete_selected_packet_not_full_profile_or_hardware_state",
                "source_file": {key: entry[key] for key in ("packet_file", "packet_sha256", "packet_bytes")},
                "recipient_turn_binding": context["binding"],
                "measurement_result_binding": copy.deepcopy(result_binding) if isinstance(result_binding, dict) else None,
                "current_stack_binding_at_feedback": current,
                "binding_status": binding_status(entry["source_binding"], current),
                "recipient_turn_binding_status": binding_status(entry["source_binding"], context["binding"]),
                "measurement_result_binding_status": binding_status(entry["source_binding"], result_binding if isinstance(result_binding, dict) else None),
                "unverified_model_fields": ["target_layer", "verdict", "confidence", "rationale", "suggested_direction"],
                "interpretation_rule": "Read profile_data and claim_assessment first. Model fields are unverified hypotheses; target_layer is experiment ownership, not exclusion of other layers. Historical/unknown sources are not current facts.",
                "packet": display_packet(packet),
                "delivery_semantics": "response_file_written_not_model_visibility_or_adoption",
                "remaining_pending_count": len(pending) - 1,
            }
            if first is None:
                first = feedback
            if len(compact_response_bytes({**base_response, "hardware_evidence_feedback": feedback})) <= MAX_INLINE_RESPONSE_BYTES:
                return feedback, None
        # Do not truncate any part of a sealed packet or displace measurement,
        # peer advice, or the existing notification to make it fit.
        fallback = {key: first[key] for key in (
            "schema_version", "agent", "turn_id", "request_id", "measure_tier", "source_file")}
        fallback.update(status="path_fallback", reason="response_budget_exceeded", index_file=INDEX,
                        instruction="The complete sealed packet remains at source_file.packet_file; no packet was truncated or marked fully delivered.")
        fits = len(compact_response_bytes({**base_response, "hardware_evidence_feedback": fallback})) <= MAX_INLINE_RESPONSE_BYTES
        return (fallback if fits else None), {**deferred, "reason": "response_budget_exceeded", "fallback_in_response": fits}

    def record_feedback_delivery(self, agent, feedback, *, response_bytes):
        """Call only after the complete response's atomic write succeeds."""
        if feedback.get("status") != "complete_packet" or feedback.get("agent") != agent:
            raise ValueError("not a complete inline hardware feedback envelope")
        with self._lock:
            packet = feedback["packet"]
            identifier = packet["evidence_id"]
            stored = self._sealed.get(agent, {}).get(identifier)
            entry = self._packets.get(agent, {}).get(identifier)
            if (stored is None or entry is None or _json(stored) != _json(packet)
                    or feedback["source_file"] != {key: entry[key] for key in ("packet_file", "packet_sha256", "packet_bytes")}):
                raise ValueError("inline packet differs from evaluator publication")
            self.pipe.log("hardware_evidence_inline_feedback", agent=agent,
                turn_id=feedback["turn_id"], request_id=feedback["request_id"], measure_tier=feedback["measure_tier"],
                evidence_id=identifier, source_digest_id=stored["source_digest_id"], packet_sha256=entry["packet_sha256"],
                feedback_sha256=hashlib.sha256(_json(feedback)).hexdigest(), response_bytes=response_bytes,
                meaning="response_file_written_not_model_visibility_or_adoption")
            context = self._contexts.get(agent)
            if context and context.get("turn_id") == feedback["turn_id"]:
                self._inline_delivered.setdefault(agent, set()).add(identifier)

    def record_feedback_deferral(self, agent, deferred, *, response_bytes):
        self.pipe.log("hardware_evidence_inline_feedback_deferred", agent=agent,
                      **deferred, response_bytes=response_bytes,
                      meaning="original_measurement_preserved_no_complete_inline_delivery")

    def notification(self, agent, *, request_id, tier, result_binding):
        # Never wait behind collection or invert Pipeline/store lock order.
        current = None
        if self.pipe._lock.acquire(blocking=False):
            try:
                current = self.pipe.stack.binding().to_dict()
            finally:
                self.pipe._lock.release()
        with self._lock:
            context = self._contexts.get(agent)
            if context is None:
                return None
            pending = [entry for key, entry in self._packets[agent].items()
                       if key not in self._notified.get(agent, set())]
            if not pending:
                return None
            envelope = {
                "schema_version": SCHEMA, "agent": agent, "turn_id": context["turn_id"],
                "request_id": request_id, "measure_tier": tier,
                "authority": "evaluator_verified_source_with_untrusted_hypothesis",
                "delivery_semantics": "notification_written_not_packet_read_or_adoption",
                "index_file": INDEX, "recipient_turn_binding": context["binding"],
                "measurement_result_binding": copy.deepcopy(result_binding) if isinstance(result_binding, dict) else None,
                "current_stack_binding_at_notification": current,
                "instruction": "New sealed HW hypotheses are available. Read a relevant packet_file "
                               "in this workspace for the COMPLETE packet; packet_sha256 covers saved bytes. "
                               "Recheck source_binding and claim_assessment. Historical/unknown evidence "
                               "is not current evidence. A seal verifies provenance, not causal truth.",
                "packets": [], "omitted_pending_count": len(pending),
            }
            for entry in pending[:MAX_NOTIFICATION_ITEMS]:
                item = copy.deepcopy(entry)
                item["binding_status"] = binding_status(item["source_binding"], current)
                item["recipient_turn_binding_status"] = binding_status(item["source_binding"], context["binding"])
                envelope["packets"].append(item)
                envelope["omitted_pending_count"] = len(pending) - len(envelope["packets"])
                if len(json.dumps({"hardware_evidence_notifications": envelope}, allow_nan=False, indent=2).encode()) > MAX_NOTIFICATION_BYTES:
                    envelope["packets"].pop()
                    envelope["omitted_pending_count"] = len(pending) - len(envelope["packets"])
                    break
            return envelope if envelope["packets"] else None

    def record_notification_delivery(self, agent, notification):
        with self._lock:
            identifiers = [item["evidence_id"] for item in notification["packets"]]
            self.pipe.log("hardware_evidence_tool_notification", agent=agent,
                turn_id=notification["turn_id"], request_id=notification["request_id"],
                evidence_ids=identifiers, notification_sha256=hashlib.sha256(_json(notification)).hexdigest(),
                meaning="response_file_written_not_packet_read_or_adoption")
            # A response from an old turn must not consume a newer turn's notice.
            context = self._contexts.get(agent)
            if context and context["turn_id"] == notification["turn_id"]:
                self._notified.setdefault(agent, set()).update(identifiers)
