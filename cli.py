from __future__ import annotations

from rlbot_pro.adapters.sim_adapter import SimWorld
from rlbot_pro.policy.agent import ProStyleAgent


def main() -> None:
    world = SimWorld()
    agent = ProStyleAgent.from_settings()
    state = world.tick()
    controls = agent.step(state)
    print(controls)


if __name__ == "__main__":
    main()
