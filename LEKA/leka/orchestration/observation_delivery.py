"""Opt-in early measured facts, independent of hardware model hypotheses.

Only the trusted archive-and-project callsite publishes. No model protocol can
import cards. Publication/response writes do not establish model visibility.
"""
from __future__ import annotations

import copy
import hashlib
import threading
import time

from ..hardware_feedback.observations import build_observation_card
from .hardware_delivery import MAX_INLINE_RESPONSE_BYTES, compact_response_bytes
from .shared_message_pool import binding_status

SCHEMA = "leka.hardware_observation_feedback.v1"
FIELD = "hardware_observations"
MAX_STORED_CARDS = 32


class HardwareObservationDelivery:
    def __init__(self, pipe, agents):
        self.pipe = pipe
        self.agents = frozenset(agents)
        self._lock = threading.RLock()
        self._cards = {}
        self._contexts = {}
        self._delivered = {}

    def begin(self, agent, context):
        if agent in self.agents:
            with self._lock:
                self._contexts[agent] = copy.deepcopy(context)
                self._delivered[agent] = set()

    def finish(self, agent):
        with self._lock:
            self._contexts.pop(agent, None)

    def feedback_turn(self, agent):
        with self._lock:
            return self._contexts.get(agent, {}).get("turn_id")

    def publish_projected_digest(self, digest):
        # The builder validates source identity and whitelists factual fields;
        # neither diagnostic model prose nor full private archives enter here.
        card = build_observation_card(digest)
        identifier = card["observation_id"]
        with self._lock:
            if identifier in self._cards:
                return False
            if len(self._cards) >= MAX_STORED_CARDS:
                raise ValueError("hardware observation store limit reached")
            self.pipe.log("hardware_observation_published", card=card, monotonic_s=time.monotonic(),
                          meaning="trusted_factual_subset_available_not_model_visibility_or_diagnosis")
            self._cards[identifier] = copy.deepcopy(card)
        return True

    def prepare(self, agent, *, request_id, tier, result_binding, turn_id, base_response):
        if turn_id is None or agent not in self.agents:
            return None, None
        current = None
        if self.pipe._lock.acquire(blocking=False):
            try:
                current = self.pipe.stack.binding().to_dict()
            finally:
                self.pipe._lock.release()
        deferred = {"turn_id": turn_id, "request_id": request_id, "measure_tier": tier}
        with self._lock:
            context = self._contexts.get(agent)
            if not context or context.get("turn_id") != turn_id:
                return None, {**deferred, "reason": "recipient_turn_changed"}
            context = copy.deepcopy(context)
            pending = [copy.deepcopy(card) for identifier, card in self._cards.items()
                       if identifier not in self._delivered.get(agent, set())]
        if not pending:
            return None, None
        if FIELD in base_response:
            raise ValueError("reserved observation field already present")
        pending.reverse()
        pending.sort(key=lambda card: binding_status(card["source_binding"], current) != "current")
        if len(compact_response_bytes(base_response)) > MAX_INLINE_RESPONSE_BYTES:
            return None, {**deferred, "reason": "base_response_exceeds_budget"}
        for card in pending:
            envelope = {
                "schema_version": SCHEMA, "status": "complete_observation_card",
                "agent": agent, "turn_id": turn_id, "request_id": request_id, "measure_tier": tier,
                "authority": "trusted_selected_measurement_facts_not_model_diagnosis",
                "recipient_turn_binding": context["binding"],
                "measurement_result_binding": copy.deepcopy(result_binding) if isinstance(result_binding, dict) else None,
                "current_stack_binding_at_feedback": current,
                "binding_status": binding_status(card["source_binding"], current),
                "recipient_turn_binding_status": binding_status(card["source_binding"], context["binding"]),
                "measurement_result_binding_status": binding_status(card["source_binding"], result_binding if isinstance(result_binding, dict) else None),
                "card": card, "remaining_pending_count": len(pending)-1,
                "delivery_semantics": "response_file_written_not_model_visibility_or_adoption",
                "instruction": "Use only within the card's source binding and scopes. Historical/unknown facts are not current facts. This selected VM view does not attribute a bottleneck to an IR layer; later hardware-agent hypotheses remain separate. Do not repeat measurement only to fetch observations.",
            }
            if len(compact_response_bytes({**base_response, FIELD: envelope})) <= MAX_INLINE_RESPONSE_BYTES:
                return envelope, None
        return None, {**deferred, "reason": "response_budget_exceeded", "pending_count": len(pending)}

    def record_delivery(self, agent, envelope, *, response_bytes):
        # Called only after successful atomic response write. An old response
        # may be logged but must never consume another turn's pending cards.
        with self._lock:
            card = envelope.get("card", {})
            stored = self._cards.get(card.get("observation_id"))
            if (envelope.get("schema_version") != SCHEMA
                    or envelope.get("status") != "complete_observation_card"
                    or envelope.get("agent") != agent or stored is None
                    or compact_response_bytes(card) != compact_response_bytes(stored)):
                raise ValueError("observation feedback differs from trusted publication")
            self.pipe.log("hardware_observation_response_written", agent=agent, monotonic_s=time.monotonic(),
                turn_id=envelope["turn_id"], request_id=envelope["request_id"],
                measure_tier=envelope["measure_tier"], observation_id=card["observation_id"],
                source_digest_id=card["source_digest_id"],
                envelope_sha256=hashlib.sha256(compact_response_bytes(envelope)).hexdigest(),
                response_bytes=response_bytes,
                meaning="response_file_written_not_model_visibility_or_adoption")
            current = self._contexts.get(agent)
            if current and current.get("turn_id") == envelope["turn_id"]:
                self._delivered.setdefault(agent, set()).add(card["observation_id"])

    def record_deferral(self, agent, deferred, *, response_bytes):
        self.pipe.log("hardware_observation_response_deferred", agent=agent,
                      **deferred, response_bytes=response_bytes,
                      meaning="original_response_preserved_no_complete_observation_delivery")
