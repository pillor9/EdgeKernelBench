"""Schedule resident agent work from completion events and pending feedback."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import time
import threading
import uuid

from .shared_message_pool import MAX_DIALOGUE_MESSAGES


# Admission floor only: this is not an estimate of collection/model duration.
# The existing operation deadline still bounds device waits and collection.
MIN_FRESH_HARDWARE_START_BUDGET_S = 1.0


class HardwareCaptureState:
    """At most two evaluator-owned attempts, with independent bookkeeping.

    Actual capture and stack snapshots still require the Pipeline lock. This
    store protects only capture identities, statuses and timestamps; seal or
    digest publication never needs to wait for an unrelated device operation.
    Its log callback must not acquire the Pipeline lock. No workspace/ledger
    data is imported. Returned snapshots are detached from mutable state.
    """
    def __init__(self, log):
        self._lock = threading.RLock()
        self._attempts = {}
        self._sequence = 0
        self._log = log

    def admit(self, attempt_id, fresh_round):
        with self._lock:
            if fresh_round not in (1, 2) or attempt_id in self._attempts or len(self._attempts) >= 2:
                raise ValueError("invalid hardware capture attempt")
            self._attempts[attempt_id] = {"fresh_attempt_id": attempt_id, "fresh_round": fresh_round,
                                          "measurement_status": "not_started"}

    def start(self, attempt_id):
        with self._lock:
            item = self._attempts[attempt_id]
            if item["measurement_status"] != "not_started":
                raise ValueError("hardware attempt cannot collect twice")
            self._sequence += 1
            item.update(capture_seq=self._sequence, capture_started_monotonic=time.monotonic(),
                        measurement_status="started")
            self._log("hardware_capture_started", **item)
            return self._sequence

    def finish(self, attempt_id, sequence, *, binding, slot_sha, tree_node_id, profile_status):
        with self._lock:
            item = self._attempts[attempt_id]
            if item.get("capture_seq") != sequence or item["measurement_status"] != "started":
                raise ValueError("hardware capture identity mismatch")
            item.update(actual_binding=dict(binding), slot_sha=dict(slot_sha), tree_node_id=tree_node_id,
                        measurement_status="verified" if profile_status.get("measurement_ok") is True else "unverified",
                        capture_finished_monotonic=time.monotonic())
            if item["measurement_status"] == "verified":
                item["capture_verified_monotonic"] = item["capture_finished_monotonic"]
            self._log("hardware_capture_finished", **item)

    def close_capture(self, attempt_id, sequence):
        with self._lock:
            item = self._attempts[attempt_id]
            if item.get("capture_seq") == sequence and item["measurement_status"] == "started":
                item["measurement_status"] = "failed"
                self._log("hardware_capture_failed", **item)

    def bind_digest(self, attempt_id, sequence, digest_id):
        with self._lock:
            item = self._attempts[attempt_id]
            if item.get("capture_seq") != sequence or item["measurement_status"] not in {"verified", "unverified"}:
                raise ValueError("digest lacks its evaluator capture")
            if not isinstance(digest_id, str) or len(digest_id) != 64:
                raise ValueError("capture digest identity is malformed")
            if item.get("digest_id") not in (None, digest_id):
                raise ValueError("capture digest identity cannot change")
            item["digest_id"] = digest_id
            self._log("hardware_capture_digest_bound", **item)

    def record_seal(self, digest_id, evidence_id, occurred_monotonic):
        with self._lock:
            for item in self._attempts.values():
                if item.get("digest_id") != digest_id or item["measurement_status"] != "verified":
                    continue
                if "first_seal_monotonic" not in item:
                    if (not math.isfinite(occurred_monotonic)
                            or occurred_monotonic < item["capture_verified_monotonic"]):
                        raise ValueError("hardware seal precedes verified capture")
                    item.update(first_seal_monotonic=occurred_monotonic, first_evidence_id=evidence_id)
                    self._log("hardware_capture_first_seal", **item,
                        capture_to_first_seal_seconds=occurred_monotonic-item["capture_started_monotonic"])
                return

    def first_ready(self):
        with self._lock:
            item = next((x for x in self._attempts.values() if x["fresh_round"] == 1), None)
            if item is None or item["measurement_status"] != "verified" or "first_seal_monotonic" not in item:
                return None
            return json.loads(json.dumps(item))

    def latest_verified(self):
        with self._lock:
            completed = [item for item in self._attempts.values()
                         if item["measurement_status"] == "verified"]
            return json.loads(json.dumps(max(completed, key=lambda item:item["capture_seq"]))) if completed else None


@dataclass(frozen=True)
class HardwareReceiver:
    ticket: object
    future: object
    session: object
    workspace: object
    layer: str


@dataclass(frozen=True)
class EarlyHardwareOpportunity:
    receivers: tuple[HardwareReceiver, ...]
    admission_binding_json: str
    first_capture_seq: int
    required_budget_s: float
    work_deadline: float
    claimed_count: object


def _receiver_is_current(api, pipe, receiver):
    """No ResidentTasks access; only a frozen ticket/Future and trusted state."""
    ticket, session = receiver.ticket, receiver.session
    if ticket.hardware or ticket.purpose != "optimization" or receiver.layer not in {"L1", "L2"}:
        return False
    if getattr(session, "_leka_pending_layer_turn_id", None) != ticket.turn_id:
        return False
    if getattr(session, "_leka_pending_turn_purpose", "optimization") != "optimization":
        return False
    state_lock = getattr(session, "_state_lock", None)
    if state_lock is None:
        return False
    # has_pending() means delivery is pending, which can include a completed
    # model result. Never count those queued completions as an active receiver.
    with state_lock:
        if not session._pending or session._terminal is not None or session._completed_results:
            return False
    if receiver.future.done():
        # Completion may have raced the first session snapshot. A timeout
        # Future can leave the same provider request pending; a consumed final
        # result cannot. Re-read after observing Future completion.
        with state_lock:
            if (not session._pending or session._completed_results
                    or session._terminal is not None or receiver.future.cancelled()):
                return False
    try:
        current, _ = api.workspace_projection_status(pipe, receiver.workspace, receiver.layer)
    except (OSError, ValueError, TypeError, RuntimeError):
        # This opportunity is not a new authority. A malformed/unavailable
        # projection never relaxes the ordinary measurement validation path.
        return False
    if not current:
        return False
    if getattr(pipe, "turn_authority_enabled", False):
        valid, _ = pipe.validate_turn_authority(ticket.role, receiver.layer, ticket.authority,
                                              require_optimization=True, turn_id=ticket.turn_id)
        if not valid:
            return False
    # Projection validation may itself take time. A wrapper Future still
    # running after send() consumed its final result is not a live provider.
    with state_lock:
        return bool(session._pending and not session._completed_results and session._terminal is None)


def _content_identity(binding):
    return tuple(binding.get(key) for key in ("l0_sha", "l1_sha", "l2_sha"))


def early_hardware_opportunity(*, api, pipe, tasks, sessions, agents, workspaces,
                               protected, captures, claimed_count, work_deadline):
    """Owner-thread eligibility; no quota spent until this returns a snapshot."""
    first = captures.first_ready()
    if first is None:
        return None, "first_capture_or_seal_unavailable"
    binding = pipe.stack.binding().to_dict()
    latest = captures.latest_verified()
    if latest is None or _content_identity(binding) == _content_identity(latest["actual_binding"]):
        return None, "unchanged_committed_content"
    duration = first["first_seal_monotonic"] - first["capture_started_monotonic"]
    required = max(60.0, 1.25 * duration) + 30.0
    if work_deadline - time.monotonic() < required:
        return None, "insufficient_feedback_budget"
    if claimed_count() != 0:
        return None, "claimed_measure_request"
    receivers = []
    for role, layer in agents.items():
        if layer not in {"L1", "L2"} or (protected is not None and role != protected[1]):
            continue
        # Only the owner touches the resident slot table. Workers receive this
        # detached identity, not a callback that consults mutable _slots.
        slot = tasks._slots.get(role)
        if slot is None or slot.state != "running":
            continue
        receiver = HardwareReceiver(slot.ticket, slot.future, sessions[role], workspaces[role], layer)
        if _receiver_is_current(api, pipe, receiver):
            receivers.append(receiver)
    if not receivers:
        return None, "no_current_active_lower_receiver"
    return EarlyHardwareOpportunity(tuple(receivers), json.dumps(binding, sort_keys=True),
        first["capture_seq"], required, work_deadline, claimed_count), None


@contextmanager
def early_hardware_capture_gate(pipe, opportunity):
    """A short Pipeline wait followed by a zero-wait device lease, never reversed."""
    from . import optimization_loop as api
    acquired = False
    reason = None
    started = time.monotonic()
    remaining = opportunity.work_deadline - started
    max_wait = min(0.5, max(0.0, remaining-opportunity.required_budget_s))
    try:
        while True:
            pipe.operation_control.checkpoint()
            if opportunity.work_deadline-time.monotonic() < opportunity.required_budget_s:
                reason = "insufficient_feedback_budget"
                break
            if opportunity.claimed_count() != 0:
                reason = "claimed_measure_request"
                break
            if pipe._lock.acquire(blocking=False):
                acquired = True
                break
            if time.monotonic()-started >= max_wait:
                reason = "pipeline_busy"
                break
            pipe.operation_control.wait(min(0.025, max_wait-(time.monotonic()-started)))
        if acquired:
            if pipe.stack.binding().to_dict() != json.loads(opportunity.admission_binding_json):
                reason = "binding_changed_after_admission"
            elif not any(_receiver_is_current(api, pipe, r) for r in opportunity.receivers):
                reason = "receiver_finished_or_invalidated"
            elif opportunity.claimed_count() != 0:
                reason = "claimed_measure_request"
        if reason is not None:
            pipe.log("early_hardware_capture_skipped", reason=reason,
                     pipeline_wait_seconds=time.monotonic()-started, attempt_refunded=False)
            yield False
            return
        # Do not catch TimeoutError raised *inside* the capture body as a
        # lease failure. Enter and exit the lease explicitly around the yield.
        lease = pipe.service.device_lease(timeout_s=0)
        try:
            lease.__enter__()
        except TimeoutError:
            pipe.log("early_hardware_capture_skipped", reason="device_lease_busy",
                     pipeline_wait_seconds=time.monotonic()-started, attempt_refunded=False)
            yield False
            return
        try:
            pipe.operation_control.checkpoint()
            if (opportunity.work_deadline-time.monotonic() < opportunity.required_budget_s
                    or opportunity.claimed_count() != 0):
                pipe.log("early_hardware_capture_skipped", reason="deadline_or_request_after_lease",
                         attempt_refunded=False)
                yield False
                return
            pipe.log("early_hardware_capture_gate_passed",
                     pipeline_wait_seconds=time.monotonic()-started,
                     receiver_turn_ids=[r.ticket.turn_id for r in opportunity.receivers],
                     meaning="eligible_feedback_opportunity_not_delivery_or_read")
            yield True
        finally:
            lease.__exit__(None, None, None)
    finally:
        if acquired:
            pipe._lock.release()


@dataclass(frozen=True)
class TurnTicket:
    role: str
    turn_id: str
    binding_json: str = "{}"
    message_ids: tuple[str, ...] = ()
    resumed: bool = False
    hardware: bool = False
    purpose: str = "optimization"
    authority_json: str | None = None

    @property
    def binding(self):
        return json.loads(self.binding_json)

    @property
    def authority(self):
        return json.loads(self.authority_json) if self.authority_json is not None else None


@dataclass
class _Slot:
    ticket: TurnTicket
    future: object
    state: str = "running"


class ResidentTasks:
    """One slot per role until its result is *committed*, not merely returned.

    All methods run on the driver's owner thread. Worker functions never alter
    the task table. Ready results keep their slot until explicit commit, closing
    the session.has_pending=False / unapplied-result admission race.
    """
    def __init__(self, roles):
        self.roles = tuple(roles)
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError("resident roles must be nonempty and unique")
        self._pool = ThreadPoolExecutor(max_workers=len(self.roles), thread_name_prefix="leka-resident")
        self._slots = {}
        self._closed = False

    def busy(self, role):
        return role in self._slots

    @property
    def has_work(self):
        return bool(self._slots)

    def submit(self, ticket, call):
        if self._closed:
            raise RuntimeError("resident admission closed")
        if ticket.role not in self.roles or self.busy(ticket.role):
            raise RuntimeError("unknown or occupied resident role")
        self._slots[ticket.role] = _Slot(ticket, self._pool.submit(call))

    def poll(self, timeout_s=0.25):
        if not self._slots:
            time.sleep(max(0.0, min(timeout_s, 0.25)))
            return []
        if not any(slot.future.done() for slot in self._slots.values()):
            wait([slot.future for slot in self._slots.values()],
                 timeout=max(0.0, min(timeout_s, 0.25)), return_when=FIRST_COMPLETED)
        ready = []
        for role in self.roles:
            slot = self._slots.get(role)
            if slot is not None and slot.future.done() and slot.state != "committing":
                slot.state = "ready"
                ready.append(slot.ticket)
        return ready

    def begin_commit(self, ticket):
        slot = self._slots[ticket.role]
        if slot.ticket != ticket or slot.state != "ready":
            raise RuntimeError("stale or unready resident ticket")
        slot.state = "committing"
        try:
            return slot.future.result(), None
        except Exception as error:
            return None, error

    def finish_commit(self, ticket):
        slot = self._slots[ticket.role]
        if slot.ticket != ticket or slot.state != "committing":
            raise RuntimeError("resident result was not committed exactly once")
        del self._slots[ticket.role]

    def close_admission(self):
        self._closed = True

    def join(self, *, timeout_s):
        """Prove every worker exited before cancellation may be reset.

        Cancelling queued futures never proves running workers exited. On a
        timeout the caller must refuse finalization; admission stays closed.
        """
        self.close_admission()
        futures = [slot.future for slot in self._slots.values()]
        for future in futures:
            future.cancel()
        pending = wait(futures, timeout=max(0.0, timeout_s))[1] if futures else set()
        self._pool.shutdown(wait=False, cancel_futures=True)
        if pending:
            raise RuntimeError("resident workers did not quiesce before finalization")
        self._slots.clear()


def budget_prompt_suffix(budget, agent, work_deadline, *, purpose="optimization"):
    # Same observational wording in batch and async; this never adds time.
    snapshot = budget.snapshot()
    remaining_s = max(0, int(work_deadline - time.monotonic()))
    next_turn = snapshot["new_layer_turns"][agent] + 1
    if purpose == "discussion":
        return (f"\n\n[existing exploration budget] Approximately {remaining_s} seconds remain; "
                f"this is ordinary admission {next_turn} of {snapshot['maximum_new_turns_per_optimizer']}. "
                "This discussion uses that quota, adds no time, and authorizes no measurement. "
                "Resolve the relevant dependency or finish NOOP promptly; do not wait for replies.")
    return (
        f"\n\n[exploration budget observed at prompt preparation] "
        f"Approximately {remaining_s} seconds remain for exploration across "
        f"all roles. This is your new layer turn {next_turn}; the configured "
        f"maximum is {snapshot['maximum_new_turns_per_optimizer']} "
        "(null means no turn-count limit). This is a snapshot, not a fresh "
        "time allowance; API latency and device waits consume it. The driver "
        "enforces the hard deadline and reserves final verification time. "
        "Choose one small falsifiable change, inspect its T0/T1 feedback, "
        "and finish promptly. Read PEER_INBOX.json at the start and after "
        "measurement; answer relevant requests with source-bound observations "
        "or an explicitly unmeasured hypothesis. Do not wait or generate "
        "messages solely to fill a conversation quota."
    )


def _clear_layer_pending(session):
    for name in ("_leka_pending_layer_binding", "_leka_pending_shared_message_ids", "_leka_pending_layer_turn_id",
                 "_leka_pending_turn_authority", "_leka_pending_turn_purpose"):
        if hasattr(session, name):
            delattr(session, name)


def next_event_rebase(rebase, by_layer, sessions, budget, tasks, log):
    """Ready/uncommitted futures retain rebase debt even after pending clears."""
    protected = rebase.next(by_layer)
    while protected is not None:
        layer, role = protected
        if tasks.busy(role) or sessions[role].has_pending() or budget.can_start(role):
            return protected
        disposition = rebase.skip_exhausted(layer)
        log("descendant_rebase_turn_skipped", layer=layer, agent=role,
            disposition=disposition, remaining_layers=list(rebase.layers))
        protected = rebase.next(by_layer)
    return None


def _consume_layer(api, pipe, pool, workspaces, sessions, backend, ticket, result, error):
    try:
        if getattr(pipe, "turn_authority_enabled", False):
            valid, reason = pipe.validate_turn_authority(ticket.role,
                pipe._turn_authority_roles.get(ticket.role), ticket.authority,
                require_optimization=False, turn_id=ticket.turn_id)
            if not valid:
                pipe.log("turn_result_authority_rejected", agent=ticket.role,
                         turn_id=ticket.turn_id, error=reason, result_applied=False)
                return None  # Rejected result is neither completion nor a HW trigger.
        return _consume_layer_result(api, pipe, pool, workspaces, sessions, backend, ticket, result, error)
    finally:
        if getattr(pipe, "turn_authority_enabled", False) and (result is not None or error is not None):
            ended = pipe.end_turn_authority(ticket.role, ticket.authority)
            if ended and getattr(sessions[ticket.role], "_leka_pending_layer_turn_id", None) == ticket.turn_id:
                _clear_layer_pending(sessions[ticket.role])


def _consume_layer_result(api, pipe, pool, workspaces, sessions, backend, ticket, result, error):
    """Existing trusted reply handlers; invoked only by main under Pipeline lock."""
    agent = ticket.role
    if error is not None:
        pipe.log("turn_error", agent=agent, error=str(error)[:400])
        return False
    status = api.terminal_provider_error(result)
    if status is not None:
        pipe.log("turn_terminal_provider", agent=agent, status=status,
                 resumed_pending=ticket.resumed, action="leave_peer_advice_unacknowledged")
        return True
    if result is not None and not api.provider_turn_succeeded(result):
        pipe.log("turn_provider_error", agent=agent, resumed_pending=ticket.resumed,
                 action="leave_peer_advice_unacknowledged", cost_usd=result.get("total_cost_usd"))
        pipe.mailbox.send(agent, "provider_retry", {
            "note": "Prior stream failed. Re-read the current binding and retained edits before continuing.",
            "prior_binding": ticket.binding,
        })
        _clear_layer_pending(sessions[agent])
        return False
    api.record_layer_turn(pipe, agent=agent, workspace=workspaces[agent], backend=backend,
        result=result, resumed_pending=ticket.resumed, shared_pool=pool,
        turn_binding=ticket.binding, turn_id=ticket.turn_id,
        **({"turn_authority": ticket.authority} if getattr(pipe, "turn_authority_enabled", False) else {}))
    if result is not None:
        api.acknowledge_shared_advice(pipe, pool, agent, ticket.message_ids,
                                    turn_id=ticket.turn_id, binding=ticket.binding)
        _clear_layer_pending(sessions[agent])
    return False


def _invoke_hardware_turn(api, *, pipe, session, workspace, backend, args,
                          work_deadline, shared_pool, capture_state=None,
                          fresh_attempt_id=None, early_opportunity=None):
    """Recheck admission time at worker execution without renewing its budget."""
    remaining = work_deadline - time.monotonic()
    pending = session.has_pending()
    if remaining <= 0 or (not pending and remaining < MIN_FRESH_HARDWARE_START_BUDGET_S):
        pipe.log("hardware_round_start_skipped", reason="insufficient_exploration_budget",
                 resumed_pending=pending, remaining_seconds=max(0.0, remaining),
                 minimum_fresh_start_seconds=MIN_FRESH_HARDWARE_START_BUDGET_S)
        return []
    extra = {}
    if capture_state is not None:
        extra = {"capture_state": capture_state, "fresh_attempt_id": fresh_attempt_id,
                 "early_opportunity": early_opportunity}
        if early_opportunity is not None:
            session._leka_early_capture_deadline = work_deadline
    return api.hw_turn(pipe, session, workspace, backend,
        api.bounded_turn_wait_seconds(turn_minutes=args.turn_minutes, work_deadline=work_deadline),
        rk_collector=args.rk_collector, shared_pool=shared_pool,
        **extra)


def run_event_loop(*, tasks, pipe, sessions, agents, workspaces, shared_pool,
                   rebase, round_budget, first, backend, args, work_deadline,
                   claimed_request_count=None):
    enabled = bool(getattr(args, "protected_discussion_turns", False))
    if getattr(args, "historical_discussion_replies", False) and not enabled:
        raise ValueError("historical replies require the protected discussion pilot")
    if enabled and (not getattr(pipe, "turn_authority_enabled", False)
                    or round_budget.maximum_new_turns not in {1, 2}):
        raise ValueError("discussion pilot requires configured authority and one or two admissions")
    if getattr(args, "early_second_hardware", False) and (
            getattr(args, "scheduling", None) != "async" or not args.hardware_feedback
            or round_budget.maximum_new_turns != 2 or not callable(claimed_request_count)):
        raise ValueError("early second hardware requires async, hardware feedback, exactly two rounds and a trusted request counter")
    try:
        return _run_event_loop(tasks=tasks, pipe=pipe, sessions=sessions, agents=agents,
            workspaces=workspaces, shared_pool=shared_pool, rebase=rebase,
            round_budget=round_budget, first=first, backend=backend, args=args,
            work_deadline=work_deadline, claimed_request_count=claimed_request_count)
    finally:
        if enabled:
            pipe.close_turn_admissions()


def measurement_discussion_reply_context(*, history, event, role, agents, current):
    """A measured answer to L0's own directed question can merit a response.

    The common L0 source is provenance only. Lower slots may have changed;
    peer prose and claimed measurements remain untrusted. This does not admit
    unsolicited observations, old graph sources, or replies to observations.
    """
    by_id = {item["id"]: item for item in history}
    parent = by_id.get(event.get("reply_to"))
    reply = event.get("dialogue") or {}
    original = (parent or {}).get("dialogue") or {}
    root_sha = current.get("l0_sha")
    if (not isinstance(root_sha, str) or len(root_sha) != 64
            or event.get("topic") != "measurement_observation"
            or reply.get("intent") != "reply" or reply.get("depth") != 1
            or event.get("recipients") != [role]
            or agents.get(event.get("sender")) not in {"L1", "L2"}
            or parent is None or parent.get("sender") != role
            or parent.get("recipients") != [event.get("sender")]
            or parent.get("reply_to") is not None
            or original.get("intent") not in {"request", "handoff"}
            or original.get("depth") != 0
            or reply.get("root_id") != parent["id"]
            or any((item.get("binding") or {}).get("l0_sha") != root_sha
                   for item in (parent, event))
            or any(not (item.get("content") or {}).get(field, "").strip()
                   for item in (parent, event) for field in ("evidence", "suggestion"))
            or reply.get("depth", 99) >= reply.get("max_depth", 0)
            or any(item.get("sender") == role and item.get("reply_to") == event["id"]
                   for item in history)
            or sum(item["id"] == parent["id"] or
                   (item.get("dialogue") or {}).get("root_id") == parent["id"]
                   for item in history) >= MAX_DIALOGUE_MESSAGES):
        return None
    return {"original_message_id": parent["id"], "reply_message_id": event["id"],
            "shared_l0_sha": root_sha,
            "scope": "same_graph_source_only; lower_bindings_may_be_historical",
            "measurement_claim_verified": False}


def historical_discussion_reply_proof(*, pipe, history, event, role, agents, current):
    """Narrow original directed L0 question -> direct lower reply exception."""
    by_id = {item["id"]: item for item in history}
    parent = by_id.get(event.get("reply_to"))
    dialogue = event.get("dialogue") or {}
    original = (parent or {}).get("dialogue") or {}
    if (dialogue.get("intent") != "reply" or dialogue.get("depth") != 1
            or event.get("recipients") != [role]
            or agents.get(event.get("sender")) not in {"L1", "L2"}
            or parent is None or parent.get("sender") != role
            or parent.get("recipients") != [event.get("sender")]
            or parent.get("reply_to") is not None
            or original.get("intent") not in {"request", "handoff"}
            or original.get("depth") != 0
            or parent.get("topic") not in {"interface_constraint", "question", "counterexample"}
            or dialogue.get("root_id") != parent["id"]
            or event.get("topic") not in {"interface_constraint", "question", "counterexample"}
            or any(not (item.get("content") or {}).get(field, "").strip()
                   for item in (parent, event) for field in ("evidence", "suggestion"))
            or dialogue.get("depth", 99) >= dialogue.get("max_depth", 0)
            or any(item.get("sender") == role and item.get("reply_to") == event["id"]
                   for item in history)
            or sum(item["id"] == parent["id"] or
                   (item.get("dialogue") or {}).get("root_id") == parent["id"]
                   for item in history) >= MAX_DIALOGUE_MESSAGES):
        return None
    proof = pipe.prove_discussion_root_relation(original_binding=parent["binding"],
        reply_binding=event["binding"], expected_current_binding=current)
    return {**proof, "original_message_id": parent["id"], "reply_message_id": event["id"]} if proof.get("verified") else None


def discussion_opportunity(*, pipe, pool, agents, protected, tasks, sessions, budget,
                            successful_optimizers, allow_historical_replies=False):
    """Conservative routing gate, not proof that a peer claim is meaningful."""
    if pool is None or protected is None or protected[0] not in {"L1", "L2"}:
        return None
    role = next((a for a, layer in agents.items() if layer == "L0"), None)
    if (role is None or role not in successful_optimizers or not budget.can_start(role)
            or tasks.busy(role) or sessions[role].has_pending() or not sessions[role].alive()):
        return None
    protected_role = protected[1]
    if not (tasks.busy(protected_role) or sessions[protected_role].has_pending()):
        return None
    current = pipe.stack.binding().to_dict()
    root_sha = current.get("l0_sha")
    if not isinstance(root_sha, str) or len(root_sha) != 64:
        return None
    try:
        pending = {m["message_id"] for m in pool.peek(role, limit=32, binding=current)}
        if not pending:
            return None
        history = pool.history()
        by_id = {m["id"]: m for m in history}
        answered = {m.get("reply_to") for m in history if m["sender"] == role}
        eligible = []
        for event in history:
            dialogue = event.get("dialogue") or {}
            content = event.get("content") or {}
            if (event["id"] not in pending or event["id"] in answered
                    or event["recipients"] != [role]
                    or agents.get(event["sender"]) not in {"L1", "L2"}
                    or dialogue.get("intent") not in {"request", "reply", "handoff"}
                    or dialogue.get("depth", 99) >= dialogue.get("max_depth", 0)
                    or not content.get("evidence", "").strip()
                    or not content.get("suggestion", "").strip()):
                continue
            if event["topic"] not in {"interface_constraint", "question", "counterexample"}:
                if measurement_discussion_reply_context(history=history, event=event,
                        role=role, agents=agents, current=current) is None:
                    continue
            if dialogue["intent"] == "reply" and by_id.get(event.get("reply_to"), {}).get("sender") != role:
                continue
            if event["binding"].get("l0_sha") != root_sha:
                if not allow_historical_replies or historical_discussion_reply_proof(
                        pipe=pipe, history=history, event=event, role=role,
                        agents=agents, current=current) is None:
                    continue
            eligible.append(event["id"])
        return (role, tuple(eligible[:3])) if eligible else None
    except (OSError, ValueError) as error:
        pipe.log("discussion_opportunity_unavailable", error=type(error).__name__)
        return None


def _run_event_loop(*, tasks, pipe, sessions, agents, workspaces, shared_pool,
                    rebase, round_budget, first, backend, args, work_deadline,
                    claimed_request_count=None):
    """Keep model waits concurrent and commit each completed role independently.

    Hardware uses one entire existing hw_turn future. It remains the sole
    writer of its digest/correction state; finalization must join this table.
    The event loop never blocks on the Pipeline lock merely to inspect state.
    """
    from . import optimization_loop as api

    by_layer = {layer: agent for agent, layer in agents.items()}
    order = [by_layer[layer] for layer in ("L0", "L1", "L2") if layer in by_layer]
    hw = api.BACKENDS[backend]["hw_sense"]
    priority = []
    dead_logged = set()
    last_binding = None
    # Baseline and workspaces exist before entry. Offer one initial snapshot
    # after admitting the optimizers, instead of waiting for their first wait
    # timeout. The worker still captures the authoritative current binding.
    hardware_generation = 1 if args.hardware_feedback else 0
    profiled_generation = 0
    fresh_hardware_rounds = 0
    successful_optimizers = set()
    discussion_enabled = bool(getattr(args, "protected_discussion_turns", False))
    early_enabled = bool(getattr(args, "early_second_hardware", False))
    captures = HardwareCaptureState(pipe.log) if early_enabled else None
    if captures is not None:
        with pipe._lock:
            pipe._hardware_capture_state = captures
    last_early_defer = None

    def observe():
        nonlocal last_binding, hardware_generation
        current = pipe.stack.binding().to_dict()
        if last_binding is not None:
            changed = rebase.observe_binding_change(last_binding, current)
            if changed is not None:
                hardware_generation += 1
                pipe.log("descendant_rebase_scheduled", changed_layer=changed,
                         protected_layers=list(rebase.layers), new_binding=current)
            elif early_enabled and _content_identity(last_binding) != _content_identity(current):
                # L2 commits do not create upper/lower rebase debt, but are a
                # real content change eligible for the same second attempt.
                hardware_generation += 1
        last_binding = current

    while time.monotonic() < work_deadline:
        ready = tasks.poll(timeout_s=(min(0.25, max(0.0, work_deadline - time.monotonic()))
                                     if tasks.has_work else 0.0))
        if time.monotonic() >= work_deadline:
            break
        # A collector/device operation can hold this lock. Preserve ready
        # tickets and process liveness/deadline on the next bounded tick.
        if not pipe._lock.acquire(blocking=False):
            time.sleep(min(0.025, max(0.0, work_deadline - time.monotonic())))
            continue
        stop = False
        try:
            observe()
            for ticket in ready:
                result, error = tasks.begin_commit(ticket)
                try:
                    if ticket.hardware:
                        if isinstance(error, api.TerminalProviderError):
                            pipe.log("provider_unavailable", agent=hw, error=str(error)[:400])
                            stop = True
                        elif error is not None:
                            pipe.log("hw_turn_error", error=str(error)[:400])
                        else:
                            priority = list(dict.fromkeys(by_layer.get(p.get("target_layer"))
                                                          for p in (result or [])))
                            priority = [role for role in priority if role is not None]
                    else:
                        consumed = _consume_layer(api, pipe, shared_pool, workspaces, sessions,
                                                  backend, ticket, result, error)
                        if consumed is None:
                            continue
                        stop |= consumed
                        observe()  # Include a promotion caused by the final reply.
                        if result is not None and ticket.purpose == "optimization":
                            hardware_generation += 1
                            if api.provider_turn_succeeded(result) and error is None:
                                successful_optimizers.add(ticket.role)
                        protected = rebase.next(by_layer)
                        if ticket.purpose == "optimization" and protected is not None and protected[1] == ticket.role:
                            layer = protected[0]
                            current, _ = api.workspace_projection_status(pipe, workspaces[ticket.role], layer)
                            disposition = rebase.finish_turn(layer,
                                completed=result is not None or error is not None,
                                provider_succeeded=api.provider_turn_succeeded(result),
                                projection_current=current)
                            pipe.log("descendant_rebase_turn_finished", layer=layer, agent=ticket.role,
                                     disposition=disposition, remaining_layers=list(rebase.layers))
                finally:
                    tasks.finish_commit(ticket)
            if stop:
                break
            unavailable, reason = api.device_unavailable(pipe)
            if unavailable:
                pipe.log("device_unavailable", backend=backend, reason=reason, action="stop_exploration")
                break
            pending = [role for role, session in sessions.items() if session.has_pending() and session.alive()]
            live = [role for role in agents if sessions[role].alive()]
            if not tasks.has_work and round_budget.should_stop(live_agents=live, pending_agents=pending):
                pipe.log("round_limit_reached", **round_budget.snapshot(),
                         action="all_admitted_turns_drained_stop_exploration_and_finalize")
                break
            if not live and not tasks.has_work and not pending:
                pipe.log("all_optimizer_sessions_terminal", action="stop_exploration_and_finalize")
                break

            # Existing pending work is always drained, including siblings of a
            # newly protected rebase. Protection restricts only new admissions.
            for role in order:
                session = sessions[role]
                if tasks.busy(role) or not session.has_pending():
                    continue
                binding = getattr(session, "_leka_pending_layer_binding", None)
                if not isinstance(binding, dict):
                    raise RuntimeError(f"pending layer turn {role} lacks its frozen binding")
                ticket = TurnTicket(role, getattr(session, "_leka_pending_layer_turn_id", ""),
                    json.dumps(binding, sort_keys=True), tuple(getattr(session, "_leka_pending_shared_message_ids", ())),
                    resumed=True, purpose=getattr(session, "_leka_pending_turn_purpose", "optimization"),
                    authority_json=getattr(session, "_leka_pending_turn_authority", None))
                tasks.submit(ticket, lambda s=session: api.invoke_layer_turn(s, prompt=None,
                    turn_minutes=args.turn_minutes, work_deadline=work_deadline))

            protected = next_event_rebase(rebase, by_layer, sessions, round_budget, tasks, pipe.log)
            l0 = by_layer.get("L0")
            schedule = [protected[1]] if protected else list(dict.fromkeys(
                ([l0] if l0 else []) + priority + order))
            # The normal protected optimizer is admitted first. A separate
            # optional entry can then consume L0's remaining existing quota.
            for role in schedule:
                if time.monotonic() >= work_deadline:
                    break
                session = sessions[role]
                if tasks.busy(role) or session.has_pending():
                    continue
                if not session.alive():
                    if role not in dead_logged:
                        pipe.log("session_died", agent=role)
                        dead_logged.add(role)
                    if protected and role == protected[1]:
                        rebase.finish_turn(protected[0], completed=True, provider_succeeded=False)
                    continue
                if not round_budget.can_start(role):
                    continue
                prompt, binding, ids, turn_id = api.prepare_layer_turn(pipe, shared_pool,
                    agent=role, layer=agents[role], workspace=workspaces[role], backend=backend,
                    first=first[role])
                prompt += budget_prompt_suffix(round_budget, role, work_deadline)
                if time.monotonic() >= work_deadline:
                    break
                round_budget.admit(role, turn_id)
                authority = None
                if discussion_enabled:
                    authority = api.activate_turn_authority(pipe, agent=role, layer=agents[role],
                        workspace=workspaces[role], turn_id=turn_id, purpose="optimization")
                pipe.log("layer_turn_admitted", agent=role, turn_id=turn_id, **round_budget.snapshot(),
                         **({"purpose": "optimization"} if discussion_enabled else {}))
                session._leka_pending_layer_binding = dict(binding)
                session._leka_pending_shared_message_ids = tuple(ids)
                session._leka_pending_layer_turn_id = turn_id
                if authority is not None:
                    session._leka_pending_turn_purpose = "optimization"
                    session._leka_pending_turn_authority = json.dumps(authority, sort_keys=True)
                first[role] = False
                ticket = TurnTicket(role, turn_id, json.dumps(binding, sort_keys=True), tuple(ids),
                                    authority_json=json.dumps(authority, sort_keys=True) if authority is not None else None)
                tasks.submit(ticket, lambda s=session, text=prompt: api.invoke_layer_turn(s,
                    prompt=text, turn_minutes=args.turn_minutes, work_deadline=work_deadline))

            opportunity = (discussion_opportunity(pipe=pipe, pool=shared_pool, agents=agents,
                protected=protected, tasks=tasks, sessions=sessions, budget=round_budget,
                successful_optimizers=successful_optimizers,
                allow_historical_replies=bool(getattr(args, "historical_discussion_replies", False))) if discussion_enabled else None)
            if opportunity is not None and time.monotonic() < work_deadline:
                role, message_ids = opportunity
                prompt, binding, ids, turn_id = api.prepare_discussion_turn(pipe, shared_pool,
                    agent=role, layer=agents[role], workspace=workspaces[role], backend=backend,
                    message_ids=message_ids,
                    **({"allow_historical_replies": True} if getattr(args, "historical_discussion_replies", False) else {}))
                prompt += budget_prompt_suffix(round_budget, role, work_deadline, purpose="discussion")
                if time.monotonic() < work_deadline:
                    round_budget.admit(role, turn_id)
                    authority = api.activate_turn_authority(pipe, agent=role, layer=agents[role],
                        workspace=workspaces[role], turn_id=turn_id, purpose="discussion")
                    session = sessions[role]
                    session._leka_pending_layer_binding = dict(binding)
                    session._leka_pending_shared_message_ids = tuple(ids)
                    session._leka_pending_layer_turn_id = turn_id
                    session._leka_pending_turn_purpose = "discussion"
                    session._leka_pending_turn_authority = json.dumps(authority, sort_keys=True)
                    pipe.log("layer_turn_admitted", agent=role, turn_id=turn_id, purpose="discussion",
                             protected_layer=protected[0], trigger_message_ids=list(ids),
                             opportunity_meaning="eligible_reply_opportunity_not_read_or_semantic_success",
                             **round_budget.snapshot())
                    ticket = TurnTicket(role, turn_id, json.dumps(binding, sort_keys=True), tuple(ids),
                                        purpose="discussion", authority_json=json.dumps(authority, sort_keys=True))
                    tasks.submit(ticket, lambda s=session, text=prompt: api.invoke_layer_turn(s,
                        prompt=text, turn_minutes=args.turn_minutes, work_deadline=work_deadline))

            if args.hardware_feedback and not tasks.busy(hw) and sessions[hw].alive():
                hw_pending = sessions[hw].has_pending()
                opportunity = None
                fresh_due = (not rebase.active and hardware_generation > profiled_generation
                             and (round_budget.maximum_new_turns is None
                                  or fresh_hardware_rounds < round_budget.maximum_new_turns)
                             and any(sessions[role].alive() and
                                     (round_budget.can_start(role) or tasks.busy(role)
                                      or sessions[role].has_pending()) for role in agents)
                             and work_deadline - time.monotonic() >= MIN_FRESH_HARDWARE_START_BUDGET_S)
                if early_enabled and fresh_hardware_rounds == 1 and not hw_pending:
                    fresh_due = False
                    if hardware_generation > profiled_generation:
                        opportunity, reason = early_hardware_opportunity(api=api, pipe=pipe,
                            tasks=tasks, sessions=sessions, agents=agents, workspaces=workspaces,
                            protected=protected, captures=captures, claimed_count=claimed_request_count,
                            work_deadline=work_deadline)
                        fresh_due = opportunity is not None
                        marker = (reason, json.dumps(last_binding, sort_keys=True))
                        if not fresh_due and marker != last_early_defer:
                            pipe.log("early_hardware_opportunity_deferred", reason=reason,
                                     attempt_spent=False, observed_binding=dict(last_binding))
                        last_early_defer = marker
                if time.monotonic() < work_deadline and (hw_pending or fresh_due):
                    capture_attempt_id = None
                    if not hw_pending:
                        profiled_generation = hardware_generation
                        fresh_hardware_rounds += 1
                        if captures is not None:
                            capture_attempt_id = uuid.uuid4().hex
                            captures.admit(capture_attempt_id, fresh_hardware_rounds)
                        pipe.log("hardware_round_admitted", scheduling="async",
                            fresh_round=fresh_hardware_rounds,
                            maximum_fresh_rounds=round_budget.maximum_new_turns,
                            trigger_generation=hardware_generation,
                            initial_opportunity=(fresh_hardware_rounds == 1),
                            admission_observed_binding=dict(last_binding),
                            snapshot_binding_authority="hw_turn_capture_under_pipeline_lock",
                            meaning="fresh_collect_reason_verify_attempt; pending_and_corrections_share_this_round",
                            **({"fresh_attempt_id": capture_attempt_id,
                                "early_second_hardware": opportunity is not None,
                                "eligible_receiver_turn_ids": [r.ticket.turn_id for r in opportunity.receivers] if opportunity else [],
                                "required_feedback_budget_seconds": opportunity.required_budget_s if opportunity else None}
                               if early_enabled else {}))
                    else:
                        pipe.log("hardware_round_resumed", scheduling="async",
                            fresh_round=fresh_hardware_rounds,
                            digest_id=getattr(sessions[hw], "_leka_pending_digest", {}).get("digest_id"),
                            new_model_prompt=False)
                    ticket = TurnTicket(hw, uuid.uuid4().hex, hardware=True, resumed=hw_pending)
                    extra = ({"capture_state": captures, "fresh_attempt_id": capture_attempt_id,
                              "early_opportunity": opportunity} if early_enabled else {})
                    tasks.submit(ticket, lambda extra=extra: _invoke_hardware_turn(api, pipe=pipe,
                        session=sessions[hw], workspace=workspaces[hw], backend=backend,
                        args=args, work_deadline=work_deadline, shared_pool=shared_pool, **extra))
        finally:
            pipe._lock.release()
    tasks.close_admission()
    pipe.log("hardware_round_budget_summary", scheduling="async",
        fresh_rounds=fresh_hardware_rounds, maximum_fresh_rounds=round_budget.maximum_new_turns,
        observed_trigger_generations=hardware_generation,
        unlimited_policy="initial opportunity then one fresh attempt per eligible completion/binding generation; never per wait timeout or polling tick")
