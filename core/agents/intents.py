"""High-level action intents for the bot.

This module defines discrete intents that represent strategic decisions
the bot can make during gameplay.
"""
from enum import Enum, auto


class Intent(Enum):
    """High-level action intents for strategic decision-making."""
    
    # Basic movement
    DRIVE_TO_BALL = auto()
    DRIVE_TO_POSITION = auto()
    
    # Offensive
    CHALLENGE = auto()
    SHOOT = auto()
    DRIBBLE = auto()
    FLICK = auto()
    AIR_DRIBBLE = auto()
    PASS = auto()
    
    # Defensive
    ROTATE_BACK = auto()
    SHADOW_DEFENSE = auto()
    SAVE = auto()
    CLEAR = auto()
    
    # Positioning
    POSITION_OFFENSE = auto()
    POSITION_DEFENSE = auto()
    POSITION_MIDFIELD = auto()
    
    # Special
    KICKOFF = auto()
    DEMO = auto()
    BOOST_PICKUP = auto()
    RECOVERY = auto()
    WAIT = auto()
    
    def __str__(self) -> str:
        """String representation of the intent."""
        return self.name.replace('_', ' ').title()
