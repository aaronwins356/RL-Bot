from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import yaml
import numpy as np

from rlbot_pro.state import GameState
from rlbot_pro.mechanics import aerial, flip_reset, ceiling, double_tap


class NodeStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class Node(ABC):
    def __init__(self):
        self.status = NodeStatus.FAILURE
        
    @abstractmethod
    def tick(self, state: GameState) -> NodeStatus:
        pass
    
    def reset(self):
        self.status = NodeStatus.FAILURE


@dataclass
class ShotOpportunity:
    probability: float
    angle_to_goal: float
    shot_speed: float
    boost_required: float
    pressure_level: float
    

class BehaviorTree:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.root = self._build_tree()
        
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)['planner']
            
    def _build_tree(self) -> Node:
        """Construct the behavior tree hierarchy"""
        return Selector([
            DefensiveRotation(self.config),
            Sequence([
                Approach(),
                Selector([
                    FirstTouch(),
                    CarryControl(),
                    FreestyleSelector(self.config),
                ])
            ]),
            FallbackClear()
        ])
    
    def tick(self, state: GameState) -> NodeStatus:
        return self.root.tick(state)


class Selector(Node):
    """Runs child nodes in order until one succeeds"""
    def __init__(self, children: List[Node]):
        super().__init__()
        self.children = children
        
    def tick(self, state: GameState) -> NodeStatus:
        for child in self.children:
            status = child.tick(state)
            if status != NodeStatus.FAILURE:
                return status
        return NodeStatus.FAILURE


class Sequence(Node):
    """Runs child nodes in order until all succeed"""
    def __init__(self, children: List[Node]):
        super().__init__()
        self.children = children
        self.current_child = 0
        
    def tick(self, state: GameState) -> NodeStatus:
        while self.current_child < len(self.children):
            status = self.children[self.current_child].tick(state)
            if status == NodeStatus.FAILURE:
                self.reset()
                return NodeStatus.FAILURE
            elif status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            self.current_child += 1
        
        self.reset()
        return NodeStatus.SUCCESS
        
    def reset(self):
        super().reset()
        self.current_child = 0
        for child in self.children:
            child.reset()