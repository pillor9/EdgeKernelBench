"""Bound new optimizer turns without counting waits as optimization rounds.

This is driver-thread admission accounting, not a clock, a provider retry loop,
or a claim that an admitted turn completed. AgentSession remains the authority
for pending work; the caller must keep its existing hard monotonic deadline.
"""

from __future__ import annotations

from collections.abc import Iterable


class RoundBudget:
    """A per-optimizer limit on new layer turns, including failed admissions.

    Check an existing session's pending state *before* ``can_start``. Resuming
    that same turn never calls ``admit``. Admit immediately before scheduling
    ``AgentSession.send``; a provider/transport failure still consumes that
    admission, so retries cannot silently exceed the requested model budget.

    ``should_stop`` also accepts pending hardware roles. This lets a completed
    optimizer quota drain already-started feedback, without authorizing a new
    hardware turn or extending the run deadline. Hardware admission/cadence is
    deliberately separate from this optimizer-only counter.
    """

    def __init__(self, agents: Iterable[str], maximum_new_turns: int | None) -> None:
        names = tuple(agents)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("at least one named optimizer is required")
        if len(set(names)) != len(names):
            raise ValueError("optimizer names must be unique")
        if maximum_new_turns is not None and (
            type(maximum_new_turns) is not int or not 1 <= maximum_new_turns <= 128
        ):
            raise ValueError("maximum_new_turns must be None or an integer from 1 to 128")
        self.maximum_new_turns = maximum_new_turns
        self._admitted: dict[str, list[str]] = {name: [] for name in names}

    def can_start(self, agent: str) -> bool:
        admitted = self._admitted[agent]
        return self.maximum_new_turns is None or len(admitted) < self.maximum_new_turns

    def admit(self, agent: str, turn_id: str) -> bool:
        """Record one new turn; exact same-agent receipt replay is idempotent.

        Returns True for a newly charged admission, False for an existing one.
        A role cannot reuse another role's identity or exceed its own quota.
        """
        admitted = self._admitted[agent]
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise ValueError("turn_id must be nonempty")
        if turn_id in admitted:
            return False
        if any(turn_id in values for name, values in self._admitted.items() if name != agent):
            raise ValueError("turn_id is already owned by another optimizer")
        if not self.can_start(agent):
            raise RuntimeError(f"new layer-turn quota exhausted for {agent}")
        admitted.append(turn_id)
        return True

    def should_stop(self, *, live_agents: Iterable[str], pending_agents: Iterable[str]) -> bool:
        """Stop only after quota exhaustion and delivery of all in-flight work.

        The caller handles dead-session diagnostics and always checks the wall
        deadline first. A dead optimizer cannot keep a finite run open forever.
        For unlimited runs this method never requests early termination.
        """
        live = tuple(live_agents)
        for agent in live:
            if agent not in self._admitted:
                raise KeyError(agent)
        if self.maximum_new_turns is None or tuple(pending_agents):
            return False
        return all(not self.can_start(agent) for agent in live)

    def snapshot(self) -> dict:
        """Explicit admission statistics; these are not completion statistics."""
        return {
            "maximum_new_turns_per_optimizer": self.maximum_new_turns,
            "new_layer_turns": {name: len(turns) for name, turns in self._admitted.items()},
            "admitted_turn_ids": {name: list(turns) for name, turns in self._admitted.items()},
            "meaning": "new_turn_admissions; waits_and_resumes_not_counted; completion_not_implied",
        }
