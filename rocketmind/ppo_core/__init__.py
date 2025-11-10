"""
PPO Core - Advanced PPO implementation for RocketMind.
"""

from .network import ActorCritic, create_actor_critic
from .memory import RolloutBuffer, EpisodeBuffer, PrioritizedReplayBuffer
from .losses import ppo_loss, value_loss, total_ppo_loss, compute_gae
from .trainer import PPOTrainer
from .agent import PPOAgent, MultiAgentWrapper
from .hyperopt import (
    AdaptiveHyperparameters,
    CurriculumManager,
    AdaptiveRolloutLength,
    RewardMixer
)
from .utils import (
    get_device,
    set_seed,
    get_schedule,
    save_checkpoint,
    load_checkpoint,
    load_config,
    count_parameters,
    RunningMeanStd,
    RewardNormalizer,
    AdaptiveEntropyCoef
)

__all__ = [
    'ActorCritic',
    'create_actor_critic',
    'RolloutBuffer',
    'EpisodeBuffer',
    'PrioritizedReplayBuffer',
    'ppo_loss',
    'value_loss',
    'total_ppo_loss',
    'compute_gae',
    'PPOTrainer',
    'PPOAgent',
    'MultiAgentWrapper',
    'AdaptiveHyperparameters',
    'CurriculumManager',
    'AdaptiveRolloutLength',
    'RewardMixer',
    'get_device',
    'set_seed',
    'get_schedule',
    'save_checkpoint',
    'load_checkpoint',
    'load_config',
    'count_parameters',
    'RunningMeanStd',
    'RewardNormalizer',
    'AdaptiveEntropyCoef'
]
