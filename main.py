from __future__ import annotations

from rlbot_pro.adapters.rlbot_adapter import RLBot_AVAILABLE
from rlbot_pro.policy.agent import ProStyleAgent


def main() -> None:
    if not RLBot_AVAILABLE:
        print("RLBot is not installed. Install the 'rlbot' package to enable in-game control.")
        return
    ProStyleAgent.from_settings()
    print("RLBot detected. Connect this agent using rlbot_pro.adapters.rlbot_adapter.tick_loop().")


if __name__ == "__main__":
    main()
