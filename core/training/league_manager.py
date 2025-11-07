"""League-based self-play manager for RL-Bot.

This module implements a league system with:
- Population of 8-12 agents (main + past + exploiters)
- Elo-based matchmaking with diversity sampling
- Exponential decay for older checkpoints
- Role-based agent management (main, past, exploiters)
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class LeagueAgent:
    """Represents an agent in the league."""
    
    agent_id: str
    role: str  # "main", "past", "exploiter"
    checkpoint_path: Optional[Path] = None
    elo: float = 1500.0
    timestep: int = 0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    creation_time: datetime = field(default_factory=datetime.now)
    last_played: datetime = field(default_factory=datetime.now)
    frozen: bool = False  # If True, agent doesn't train
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.5
        return self.wins / self.games_played
    
    @property
    def age_hours(self) -> float:
        """Calculate age in hours."""
        return (datetime.now() - self.creation_time).total_seconds() / 3600.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "elo": self.elo,
            "timestep": self.timestep,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "frozen": self.frozen,
            "age_hours": self.age_hours,
        }


class LeagueManager:
    """Manages a population of agents for league-based self-play."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        log_dir: Optional[Path] = None,
    ):
        """Initialize league manager.
        
        Args:
            config: Configuration dictionary
            log_dir: Directory for saving league state
        """
        self.config = config or {}
        self.log_dir = Path(log_dir) if log_dir else Path("logs/league")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # League configuration
        self.min_population = self.config.get("min_population", 8)
        self.max_population = self.config.get("max_population", 12)
        self.max_past_agents = self.config.get("max_past_agents", 6)
        self.max_exploiters = self.config.get("max_exploiters", 3)
        
        # Elo configuration
        self.k_factor = self.config.get("elo_k_factor", 32)
        self.initial_elo = self.config.get("initial_elo", 1500.0)
        
        # Sampling configuration
        self.diversity_temperature = self.config.get("diversity_temperature", 1.0)
        self.age_decay_rate = self.config.get("age_decay_rate", 0.1)  # exponential decay
        self.min_sample_prob = self.config.get("min_sample_prob", 0.01)
        
        # Agent population
        self.agents: Dict[str, LeagueAgent] = {}
        self.main_agent_id: Optional[str] = None
        
        # Statistics
        self.total_games = 0
        self.elo_history: List[Dict[str, Any]] = []
        
        # Initialize main agent
        self._initialize_main_agent()
        
        logger.info(f"LeagueManager initialized (pop: {self.min_population}-{self.max_population})")
        logger.info(f"  - Max past agents: {self.max_past_agents}")
        logger.info(f"  - Max exploiters: {self.max_exploiters}")
    
    def _initialize_main_agent(self):
        """Initialize the main training agent."""
        main_id = "main_agent"
        self.main_agent_id = main_id
        self.agents[main_id] = LeagueAgent(
            agent_id=main_id,
            role="main",
            elo=self.initial_elo,
            frozen=False,
        )
        logger.info(f"Main agent initialized: {main_id}")
    
    def add_past_checkpoint(
        self,
        checkpoint_path: Path,
        timestep: int,
        elo: Optional[float] = None,
    ) -> str:
        """Add a past checkpoint to the league.
        
        Args:
            checkpoint_path: Path to checkpoint file
            timestep: Training timestep when checkpoint was created
            elo: Elo rating (uses main agent's elo if None)
            
        Returns:
            Agent ID of added checkpoint
        """
        # Use main agent's elo if not provided
        if elo is None and self.main_agent_id:
            elo = self.agents[self.main_agent_id].elo
        else:
            elo = elo or self.initial_elo
        
        # Create agent ID
        agent_id = f"past_{timestep}"
        
        # Check if we need to prune old agents
        past_agents = [a for a in self.agents.values() if a.role == "past"]
        if len(past_agents) >= self.max_past_agents:
            # Remove oldest agent with lowest Elo
            to_remove = min(past_agents, key=lambda a: (a.elo, -a.age_hours))
            logger.info(f"Pruning past agent: {to_remove.agent_id} (Elo: {to_remove.elo:.0f})")
            del self.agents[to_remove.agent_id]
        
        # Add new past agent
        self.agents[agent_id] = LeagueAgent(
            agent_id=agent_id,
            role="past",
            checkpoint_path=checkpoint_path,
            elo=elo,
            timestep=timestep,
            frozen=True,  # Past agents don't train
        )
        
        logger.info(f"Added past checkpoint: {agent_id} (Elo: {elo:.0f}, timestep: {timestep})")
        return agent_id
    
    def add_exploiter(
        self,
        checkpoint_path: Path,
        target_agent_id: Optional[str] = None,
    ) -> str:
        """Add an exploiter agent trained against specific target.
        
        Args:
            checkpoint_path: Path to exploiter checkpoint
            target_agent_id: ID of target agent to exploit (defaults to main)
            
        Returns:
            Agent ID of exploiter
        """
        target_id = target_agent_id or self.main_agent_id
        
        # Check population limit
        exploiters = [a for a in self.agents.values() if a.role == "exploiter"]
        if len(exploiters) >= self.max_exploiters:
            # Remove oldest exploiter
            to_remove = min(exploiters, key=lambda a: a.timestep)
            logger.info(f"Pruning exploiter: {to_remove.agent_id}")
            del self.agents[to_remove.agent_id]
        
        # Create agent ID
        agent_id = f"exploiter_{len(exploiters)}"
        
        # Exploiter starts at similar elo as target
        target_elo = self.agents.get(target_id, self.agents[self.main_agent_id]).elo
        
        self.agents[agent_id] = LeagueAgent(
            agent_id=agent_id,
            role="exploiter",
            checkpoint_path=checkpoint_path,
            elo=target_elo,
            frozen=False,  # Exploiters can train
        )
        
        logger.info(f"Added exploiter: {agent_id} targeting {target_id}")
        return agent_id
    
    def select_opponent(self, current_agent_id: Optional[str] = None) -> Optional[LeagueAgent]:
        """Select opponent using Elo-based diversity sampling.
        
        Uses softmax over rating differences with exponential age decay.
        
        Args:
            current_agent_id: ID of agent selecting opponent (defaults to main)
            
        Returns:
            Selected opponent agent or None if no valid opponents
        """
        current_id = current_agent_id or self.main_agent_id
        if not current_id or current_id not in self.agents:
            logger.warning(f"Invalid current agent: {current_id}")
            return None
        
        current_agent = self.agents[current_id]
        
        # Get all valid opponents (exclude self)
        opponents = [a for a in self.agents.values() if a.agent_id != current_id]
        
        if not opponents:
            logger.warning("No opponents available in league")
            return None
        
        # Calculate sampling probabilities
        probabilities = []
        for opp in opponents:
            # Elo difference (higher difference = more interesting match)
            elo_diff = abs(current_agent.elo - opp.elo)
            diversity_score = np.exp(-elo_diff / (400.0 * self.diversity_temperature))
            
            # Age decay (older agents sampled less frequently)
            age_factor = np.exp(-self.age_decay_rate * opp.age_hours)
            
            # Combined probability
            prob = diversity_score * age_factor
            probabilities.append(max(prob, self.min_sample_prob))
        
        # Normalize probabilities
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        
        # Sample opponent
        selected = np.random.choice(opponents, p=probabilities)
        selected.last_played = datetime.now()
        
        logger.debug(f"Selected opponent: {selected.agent_id} (Elo: {selected.elo:.0f}, p={probabilities[opponents.index(selected)]:.3f})")
        return selected
    
    def update_elo(
        self,
        agent1_id: str,
        agent2_id: str,
        agent1_score: float,
    ):
        """Update Elo ratings after a match.
        
        Args:
            agent1_id: ID of first agent
            agent2_id: ID of second agent
            agent1_score: Score for agent1 (1.0 = win, 0.5 = draw, 0.0 = loss)
        """
        if agent1_id not in self.agents or agent2_id not in self.agents:
            logger.warning(f"Invalid agent IDs: {agent1_id}, {agent2_id}")
            return
        
        agent1 = self.agents[agent1_id]
        agent2 = self.agents[agent2_id]
        
        # Calculate expected scores
        expected1 = 1.0 / (1.0 + 10 ** ((agent2.elo - agent1.elo) / 400.0))
        expected2 = 1.0 - expected1
        
        # Update elo ratings
        agent1.elo += self.k_factor * (agent1_score - expected1)
        agent2.elo += self.k_factor * ((1.0 - agent1_score) - expected2)
        
        # Update game statistics
        agent1.games_played += 1
        agent2.games_played += 1
        
        if agent1_score == 1.0:
            agent1.wins += 1
            agent2.losses += 1
        elif agent1_score == 0.0:
            agent1.losses += 1
            agent2.wins += 1
        # Draws don't count as wins/losses
        
        self.total_games += 1
        
        # Record Elo history for main agent
        if agent1_id == self.main_agent_id:
            self.elo_history.append({
                "timestep": agent1.timestep,
                "elo": agent1.elo,
                "opponent": agent2_id,
                "opponent_elo": agent2.elo,
                "score": agent1_score,
                "games_played": self.total_games,
            })
        
        logger.debug(f"Elo update: {agent1_id} {agent1.elo:.0f} vs {agent2_id} {agent2.elo:.0f}")
    
    def get_main_agent_elo(self) -> float:
        """Get current Elo of main agent."""
        if self.main_agent_id and self.main_agent_id in self.agents:
            return self.agents[self.main_agent_id].elo
        return self.initial_elo
    
    def get_league_stats(self) -> Dict[str, Any]:
        """Get league statistics.
        
        Returns:
            Dictionary with league stats
        """
        stats = {
            "total_agents": len(self.agents),
            "total_games": self.total_games,
            "main_agent_elo": self.get_main_agent_elo(),
        }
        
        # Count by role
        for role in ["main", "past", "exploiter"]:
            role_agents = [a for a in self.agents.values() if a.role == role]
            stats[f"num_{role}"] = len(role_agents)
            if role_agents:
                stats[f"avg_elo_{role}"] = np.mean([a.elo for a in role_agents])
        
        return stats
    
    def get_agents_by_role(self, role: str) -> List[LeagueAgent]:
        """Get all agents with specific role.
        
        Args:
            role: Agent role ("main", "past", "exploiter")
            
        Returns:
            List of agents with that role
        """
        return [a for a in self.agents.values() if a.role == role]
    
    def save_state(self, filepath: Optional[Path] = None):
        """Save league state to file.
        
        Args:
            filepath: Path to save file (defaults to log_dir/league_state.json)
        """
        filepath = filepath or self.log_dir / "league_state.json"
        
        state = {
            "config": self.config,
            "total_games": self.total_games,
            "main_agent_id": self.main_agent_id,
            "agents": {aid: agent.to_dict() for aid, agent in self.agents.items()},
            "elo_history": self.elo_history,
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"League state saved to {filepath}")
    
    def save_elo_history(self, filepath: Optional[Path] = None):
        """Save Elo history to file.
        
        Args:
            filepath: Path to save file (defaults to log_dir/elo_history.json)
        """
        filepath = filepath or self.log_dir / "elo_history.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.elo_history, f, indent=2, default=str)
        
        logger.info(f"Elo history saved to {filepath}")
