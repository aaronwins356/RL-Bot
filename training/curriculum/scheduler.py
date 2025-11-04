from typing import List, Dict
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm


class CurriculumScheduler:
    def __init__(self, config: Dict):
        self.scenarios = config['scenarios']
        self.thresholds = config['thresholds']
        self.current_stage = 0
        self.eval_history: List[float] = []
        
    def current_scenario(self) -> str:
        return self.scenarios[self.current_stage]
        
    def should_progress(self, model: BaseAlgorithm) -> bool:
        """Determine if we should move to the next curriculum stage"""
        # Get recent evaluation performance
        recent_evals = model.eval_env.get_episode_rewards()[-10:]
        
        if not recent_evals:
            return False
            
        # Calculate success metrics
        avg_reward = np.mean(recent_evals)
        self.eval_history.append(avg_reward)
        
        # Check if we've hit the threshold for this stage
        if avg_reward >= self.thresholds[self.current_stage]:
            return True
            
        return False
        
    def next_scenario(self) -> str:
        """Progress to next curriculum stage"""
        if self.current_stage < len(self.scenarios) - 1:
            self.current_stage += 1
        return self.current_scenario()