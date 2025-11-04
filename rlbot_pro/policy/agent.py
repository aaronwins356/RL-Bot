"""High-level policy translating game state into controls."""

from __future__ import annotations

from dataclasses import dataclass

from ..control import Controls
from ..planning.options import evaluate, execute
from ..state import GameState


@dataclass
class ProStyleAgent:
    """Deterministic policy using handcrafted heuristics."""

    name: str = "ProStyle"

    def step(self: ProStyleAgent, state: GameState) -> Controls:
        """Return controls for a single simulated frame."""
        option_type = evaluate(state)
        option = execute(state, option_type)
        return option.controls


__all__ = ["ProStyleAgent"]
