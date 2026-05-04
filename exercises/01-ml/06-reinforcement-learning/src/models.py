"""Model training with experiment framework for AgentAI

This module provides:
- Abstract RLAgent interface for plug-and-play RL algorithms
- Concrete implementations: Q-Learning, DQN (with TODOs)
- ExperimentRunner for comparing multiple agents
- Immediate feedback with rich console output

Learning objectives:
1. Implement Q-Learning and DQN with TODOs
2. Compare agents using plug-and-play registry pattern
3. See episode rewards immediately after each episode
4. Experiment with hyperparameters and observe learning curves
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.table import Table
from tensorflow import keras

from src.utils import timer, validate_positive

logger = logging.getLogger("agentai")
console = Console()


@dataclass
class AgentConfig:
    """Configuration for agent training."""
    episodes: int = 500
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 32
    buffer_size: int = 10000
    hidden_units: List[int] = None
    random_state: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [128, 64]


class RLAgent(ABC):
    """Abstract base class for all RL agents.
    
    Provides common interface for plug-and-play experimentation.
    Subclasses implement train_episode() and select_action() methods.
    """
    
    def __init__(self, name: str, state_dim: int, action_dim: int):
        """Initialize RL agent with name for display.
        
        Args:
            name: Display name for agent
            state_dim: Dimension of state space
            action_dim: Number of actions
        """
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = None
        self.metrics = {}
        self.training_history = []
    
    @abstractmethod
    def train_episode(
        self,
        env_wrapper,
        epsilon: float,
        config: AgentConfig
    ) -> Dict[str, float]:
        """Train for one episode and return metrics with immediate console feedback.
        
        Args:
            env_wrapper: Environment wrapper instance
            epsilon: Current exploration rate
            config: Training configuration
        
        Returns:
            Dictionary with episode metrics: {"episode_reward": float, "steps": int, ...}
        """
        pass
    
    @abstractmethod
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using current policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate for epsilon-greedy
        
        Returns:
            Selected action
        """
        pass
    
    def save(self, path: str) -> None:
        """Save trained agent to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained agent")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and history
        save_dict = {
            "model": self.model,
            "metrics": self.metrics,
            "training_history": self.training_history,
            "name": self.name
        }
        joblib.dump(save_dict, path)
        logger.info(f"Saved {self.name} to {path}")
    
    @classmethod
    def load(cls, path: str) -> "RLAgent":
        """Load trained agent from disk."""
        save_dict = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = save_dict["model"]
        instance.metrics = save_dict["metrics"]
        instance.training_history = save_dict.get("training_history", [])
        instance.name = save_dict["name"]
        return instance


class QLearningAgent(RLAgent):
    """Q-Learning agent with tabular Q-table.
    
    Q-Learning learns Q-values: Q(s, a) = expected return from taking action a in state s.
    Uses Bellman equation: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
    
    Key concepts:
    - Q-table: Dictionary mapping (state, action) → Q-value
    - Epsilon-greedy: Balance exploration (random) vs exploitation (best Q)
    - Temporal difference: Learn from one-step lookahead
    """
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.1):
        """Initialize Q-Learning agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            learning_rate: Learning rate (α) for Q-updates
        """
        super().__init__(f"Q-Learning (α={learning_rate})", state_dim, action_dim)
        self.learning_rate = learning_rate
        self.model = {}  # Q-table: (state_key, action) → Q-value
    
    def _discretize_state(self, state: np.ndarray, bins: int = 10) -> Tuple[int, ...]:
        """Discretize continuous state for Q-table lookup.
        
        Args:
            state: Continuous state vector
            bins: Number of bins per dimension
        
        Returns:
            Tuple of discrete bin indices
        """
        # Simple discretization: divide each dimension into bins
        discretized = tuple(
            np.clip(int(s * bins), 0, bins - 1) for s in state
        )
        return discretized
    
    def train_episode(
        self,
        env_wrapper,
        epsilon: float,
        config: AgentConfig
    ) -> Dict[str, float]:
        """
        TODO: Implement Q-Learning episode training
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement Q-Learning episode training")
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        TODO: Implement epsilon-greedy action selection
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement action selection")


class DQNAgent(RLAgent):
    """Deep Q-Network (DQN) agent with experience replay.
    
    DQN uses neural network to approximate Q-function: Q(s, a; θ) ≈ Q*(s, a)
    Key innovations:
    - Experience replay: Store and sample transitions to break correlation
    - Neural network: Handle continuous state spaces without discretization
    
    Architecture:
    - Input: State vector (e.g., [x, ẋ, θ, θ̇] for CartPole)
    - Hidden: 2-3 fully connected layers with ReLU
    - Output: Q-values for each action [Q(s,a₀), Q(s,a₁), ...]
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: List[int] = None
    ):
        """Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_units: Neural network architecture
        """
        super().__init__(f"DQN", state_dim, action_dim)
        self.hidden_units = hidden_units or [128, 64]
        self.replay_buffer = deque(maxlen=10000)
        self.model = None  # Will be built in first training call
        self.optimizer = None
    
    def _build_network(self) -> keras.Model:
        """
        TODO: Build Q-network architecture
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement Q-network")
    
    def _store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        TODO: Store transition in replay buffer
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement replay buffer storage")
    
    def _sample_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """
        TODO: Sample random batch from replay buffer
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement batch sampling")
    
    def train_episode(
        self,
        env_wrapper,
        epsilon: float,
        config: AgentConfig
    ) -> Dict[str, float]:
        """
        TODO: Implement DQN episode training with experience replay
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement DQN episode training")
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        TODO: Implement epsilon-greedy action selection using Q-network
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement DQN action selection")


class ExperimentRunner:
    """Run RL experiments with multiple agents and compare results.
    
    Provides plug-and-play framework for trying different agents:
    1. Register agents to try
    2. Train all agents with episode-by-episode feedback
    3. Print learning curves and final leaderboard
    
    Example:
        >>> runner = ExperimentRunner(env_wrapper)
        >>> runner.register("Q-Learning", QLearningAgent(4, 2))
        >>> runner.register("DQN", DQNAgent(4, 2))
        >>> runner.run_experiment(AgentConfig(episodes=500))
        >>> runner.print_leaderboard()
    """
    
    def __init__(self, env_wrapper):
        """Initialize experiment runner.
        
        Args:
            env_wrapper: Environment wrapper instance
        """
        self.env_wrapper = env_wrapper
        self.agents: Dict[str, RLAgent] = {}
        self.results: List[Dict[str, Any]] = []
    
    def register(self, name: str, agent: RLAgent):
        """Register an agent to try in experiments.
        
        Args:
            name: Display name for results
            agent: Agent instance to train
        """
        self.agents[name] = agent
        console.print(f"Registered: {name}", style="dim")
    
    def run_experiment(self, config: AgentConfig):
        """
        TODO: Train all registered agents and track progress
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement experiment runner")
    
    def print_leaderboard(self):
        """
        TODO: Print sorted leaderboard of all agents
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement leaderboard")
    
    def plot_learning_curves(self):
        """
        TODO (Optional): Plot learning curves for all agents
        
        Use matplotlib to plot episode rewards over time.
        Shows how quickly each agent learns.
        
        Steps:
        1. Import matplotlib: import matplotlib.pyplot as plt
        2. Create figure: plt.figure(figsize=(12, 6))
        3. For each result, plot:
           plt.plot(result["episode_rewards"], label=result["agent"], alpha=0.7)
        4. Add labels and legend: plt.xlabel("Episode"), plt.ylabel("Reward"), plt.legend()
        5. Save figure: plt.savefig("models/learning_curves.png")
        
        Time estimate: 20 minutes
        """
        pass
    
    def get_best_agent(self) -> RLAgent:
        """Return agent with highest average reward."""
        if not self.results:
            raise ValueError("No experiments run yet")
        best_result = max(self.results, key=lambda x: x["avg_reward_100"])
        return self.agents[best_result["agent"]]
