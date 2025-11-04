from typing import Any, Dict, List, Union
import numpy as np
import rlgym
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.state_setters import StateSetter

class RLBotEnv(rlgym.RLGym):
    def __init__(self,
                 scenario: str,
                 reward_fn: RewardFunction,
                 obs_builder: ObsBuilder,
                 state_setter: StateSetter,
                 terminal_condition: TerminalCondition,
                 team_size: int = 1):
        super().__init__(
            reward_fn=reward_fn,
            obs_builder=obs_builder,
            state_setter=state_setter,
            terminal_condition=terminal_condition,
            team_size=team_size
        )
        self.scenario = scenario

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs = super().reset(seed=seed)
        self.state_setter.reset(self.state)
        return obs

    def step(self, actions: Union[np.ndarray, List[np.ndarray]]):
        obs, reward, done, info = super().step(actions)
        info['scenario'] = self.scenario
        return obs, reward, done, info