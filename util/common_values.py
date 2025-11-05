"""Common constants used throughout the bot project."""

# Goal locations (in Unreal Units)
ORANGE_GOAL_CENTER = [0, 5120, 642.775 / 2]
BLUE_GOAL_CENTER = [0, -5120, 642.775 / 2]

# Team identifiers
BLUE_TEAM = 0
ORANGE_TEAM = 1

# Action space size for discrete controller inputs
NUM_ACTIONS = 8

# Rocket League vehicle and boost constants (uu = unreal units)
CAR_MAX_SPEED = 2300
MAX_CAR_VELOCITY = CAR_MAX_SPEED
BOOST_USAGE_PER_SECOND = 33.3
MAX_BOOST = 100
