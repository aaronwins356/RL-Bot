from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto


class OptionKind(Enum):
    AERIAL = auto()
    AIR_DRIBBLE = auto()
    CEILING = auto()
    FLIP_RESET = auto()
    DOUBLE_TAP = auto()
    RECOVERY = auto()
    CHALLENGE = auto()
    CLEAR = auto()


@dataclass(frozen=True, slots=True)
class Option:
    kind: OptionKind
    esv: float
    description: str
    metadata: Mapping[str, float] | None = None

    def better_than(self, other: Option) -> bool:
        return self.esv > other.esv


def mechanic_enabled(config: Mapping[str, object], kind: OptionKind) -> bool:
    mechanics = config.get("mechanics", {})
    if not isinstance(mechanics, Mapping):
        return True
    mapping = {
        OptionKind.AERIAL: "aerial",
        OptionKind.AIR_DRIBBLE: "air_dribble",
        OptionKind.CEILING: "ceiling",
        OptionKind.FLIP_RESET: "flip_reset",
        OptionKind.DOUBLE_TAP: "double_tap",
        OptionKind.RECOVERY: "recoveries",
    }
    key = mapping.get(kind)
    if key is None:
        return True
    value = mechanics.get(key, True)
    return bool(value)


__all__ = ["OptionKind", "Option", "mechanic_enabled"]
