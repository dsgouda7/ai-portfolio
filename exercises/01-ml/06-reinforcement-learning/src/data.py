"""Environment wrapper and episode collection for AgentAI

Provides: Environment loading, episode rollout, replay buffer
"""

import logging
from collections import deque
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import gymnasium as gym

from src.utils import set_seed, validate_positive


logger = logging.getLogger("agentai")


class EnvironmentWrapper:
    """Wrapper for Gymnasium environments with validation.
    
    Provides:
    - Environment initialization with error handling
    - State/action space validation
    - Episode reset and step execution
    - Rendering support
    
    Attributes:
        env: Gymnasium environment instance
        env_name: Name of environment
        state_dim: Dimension of state space
        action_dim: Number of actions
        max_steps: Maximum steps per episode
    
    Example:
        >>> wrapper = EnvironmentWrapper("CartPole-v1")
        >>> state, info = wrapper.reset()
        >>> action = wrapper.sample_action()
        >>> next_state, reward, done, truncated, info = wrapper.step(action)
    """
    
    def __init__(
        self,
        env_name: str = "CartPole-v1",
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        random_state: int = 42
    ):
        """Initialize environment wrapper.
        
        Args:
            env_name: Name of Gymnasium environment
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            random_state: Random seed for reproducibility
        
        Raises:
            ValueError: If environment doesn't exist or max_steps is invalid
            RuntimeError: If environment initialization fails
        """
        validate_positive(max_steps, "max_steps")
        set_seed(random_state)
        
        logger.info(f"Initializing environment: {env_name}")
        
        try:
            # Create environment
            self.env = gym.make(env_name, render_mode=render_mode)
            self.env_name = env_name
            self.max_steps = max_steps
            
            # Extract space dimensions
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.n
            
            logger.info(
                f"Environment loaded - State dim: {self.state_dim}, "
                f"Action dim: {self.action_dim}, Max steps: {max_steps}"
            )
            
        except Exception as e:
            logger.error(f"Failed to create environment {env_name}: {e}")
            raise RuntimeError(f"Environment creation failed: {e}") from e
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.
        
        Args:
            seed: Optional seed for episode reset
        
        Returns:
            Tuple of (initial_state, info_dict)
        
        Example:
            >>> state, info = wrapper.reset(seed=42)
        """
        try:
            state, info = self.env.reset(seed=seed)
            return np.array(state, dtype=np.float32), info
        except Exception as e:
            logger.error(f"Environment reset failed: {e}")
            raise RuntimeError(f"Reset failed: {e}") from e
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action in environment.
        
        Args:
            action: Action to execute
        
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        
        Example:
            >>> next_state, reward, done, truncated, info = wrapper.step(1)
        """
        try:
            state, reward, terminated, truncated, info = self.env.step(action)
            return (
                np.array(state, dtype=np.float32),
                float(reward),
                bool(terminated),
                bool(truncated),
                info
            )
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            raise RuntimeError(f"Step failed: {e}") from e
    
    def sample_action(self) -> int:
        """Sample random action from action space.
        
        Returns:
            Random action
        
        Example:
            >>> action = wrapper.sample_action()
        """
        return self.env.action_space.sample()
    
    def close(self) -> None:
        """Close environment and cleanup resources."""
        try:
            self.env.close()
            logger.info("Environment closed")
        except Exception as e:
            logger.warning(f"Error closing environment: {e}")


class EpisodeCollector:
    """Collect episodes for training.
    
    Attributes:
        env_wrapper: Environment wrapper
        episode_count: Total episodes collected
        total_steps: Total steps across all episodes
    
    Example:
        >>> collector = EpisodeCollector(env_wrapper)
        >>> episode = collector.collect_episode(policy_fn)
        >>> print(f"Episode reward: {episode['total_reward']}")
    """
    
    def __init__(self, env_wrapper: EnvironmentWrapper):
        """Initialize episode collector.
        
        Args:
            env_wrapper: Initialized environment wrapper
        """
        self.env_wrapper = env_wrapper
        self.episode_count = 0
        self.total_steps = 0
        
        logger.info("Initialized EpisodeCollector")
    
    def collect_episode(
        self,
        policy_fn: callable,
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """Collect a single episode.
        
        Args:
            policy_fn: Function that takes state and returns action
            max_steps: Maximum steps (uses env default if None)
        
        Returns:
            Dictionary with episode data:
                - states: List of states
                - actions: List of actions
                - rewards: List of rewards
                - total_reward: Sum of rewards
                - steps: Number of steps
        
        Example:
            >>> episode = collector.collect_episode(lambda s: np.argmax(policy.predict(s)))
        """
        max_steps = max_steps or self.env_wrapper.max_steps
        
        states, actions, rewards = [], [], []
        state, _ = self.env_wrapper.reset()
        
        for step in range(max_steps):
            # Select action
            action = policy_fn(state)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = self.env_wrapper.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            
            if terminated or truncated:
                break
        
        self.episode_count += 1
        self.total_steps += len(states)
        
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "total_reward": float(np.sum(rewards)),
            "steps": len(states),
        }


class ReplayBuffer:
    """Experience replay buffer for DQN.
    
    Attributes:
        buffer: Deque of transitions
        max_size: Maximum buffer size
        size: Current number of transitions
    
    Example:
        >>> buffer = ReplayBuffer(max_size=10000)
        >>> buffer.add(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=32)
    """
    
    def __init__(self, max_size: int = 10000):
        """Initialize replay buffer.
        
        Args:
            max_size: Maximum number of transitions to store
        """
        validate_positive(max_size, "max_size")
        
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        
        logger.info(f"Initialized ReplayBuffer with max_size={max_size}")
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add transition to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Dictionary with batched transitions
        
        Raises:
            ValueError: If batch_size > buffer size
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Batch size {batch_size} exceeds buffer size {len(self.buffer)}"
            )
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "next_states": np.array(next_states),
            "dones": np.array(dones),
        }
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
