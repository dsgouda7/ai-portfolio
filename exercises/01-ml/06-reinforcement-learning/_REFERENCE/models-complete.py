"""Model training and registry for AgentAI

Provides: ModelRegistry for training REINFORCE and DQN policies
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.utils import timer, validate_positive


logger = logging.getLogger("agentai")


class ModelRegistry:
    """Registry for training and managing RL policies.
    
    Supported algorithms:
    - REINFORCE (policy gradient)
    - DQN (Deep Q-Network with experience replay)
    
    Attributes:
        policies: Dictionary of trained policies
        best_policy_name: Name of best performing policy
        training_history: Training metrics history
    
    Example:
        >>> registry = ModelRegistry(state_dim=4, action_dim=2)
        >>> registry.train_reinforce(episodes, learning_rate=0.001)
        >>> action = registry.select_action(state, "reinforce")
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        """Initialize model registry.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
        """
        validate_positive(state_dim, "state_dim")
        validate_positive(action_dim, "action_dim")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policies = {}
        self.best_policy_name = None
        self.training_history = {}
        
        logger.info(
            f"Initialized ModelRegistry (state_dim={state_dim}, action_dim={action_dim})"
        )
    
    def _build_policy_network(
        self,
        hidden_units: List[int] = [128, 64],
        activation: str = "relu"
    ) -> keras.Model:
        """Build neural network for policy.
        
        Args:
            hidden_units: List of hidden layer sizes
            activation: Activation function
        
        Returns:
            Keras model
        """
        model = keras.Sequential([
            keras.layers.Dense(
                hidden_units[0],
                activation=activation,
                input_shape=(self.state_dim,)
            )
        ])
        
        for units in hidden_units[1:]:
            model.add(keras.layers.Dense(units, activation=activation))
        
        # Output layer (action probabilities for REINFORCE, Q-values for DQN)
        model.add(keras.layers.Dense(self.action_dim, activation="softmax"))
        
        return model
    
    @timer
    def train_reinforce(
        self,
        episodes_data: List[Dict[str, np.ndarray]],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        hidden_units: List[int] = [128, 64]
    ) -> Dict[str, float]:
        """Train REINFORCE policy gradient algorithm.
        
        Args:
            episodes_data: List of episode dictionaries with states, actions, rewards
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            hidden_units: Policy network architecture
        
        Returns:
            Dictionary with training metrics
        
        Raises:
            ValueError: If learning_rate or gamma are invalid
            RuntimeError: If training fails
        
        Example:
            >>> metrics = registry.train_reinforce(
            ...     episodes,
            ...     learning_rate=0.001,
            ...     gamma=0.99
            ... )
        """
        validate_positive(learning_rate, "learning_rate")
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        
        logger.info(
            f"Training REINFORCE (lr={learning_rate}, gamma={gamma}, "
            f"episodes={len(episodes_data)})"
        )
        
        try:
            # Build or get policy network
            if "reinforce" not in self.policies:
                policy = self._build_policy_network(hidden_units)
                optimizer = keras.optimizers.Adam(learning_rate)
                self.policies["reinforce"] = {"model": policy, "optimizer": optimizer}
            else:
                policy = self.policies["reinforce"]["model"]
                optimizer = self.policies["reinforce"]["optimizer"]
            
            total_loss = 0.0
            episode_count = len(episodes_data)
            
            # Train on episodes
            for episode in episodes_data:
                states = episode["states"]
                actions = episode["actions"]
                rewards = episode["rewards"]
                
                # Compute discounted returns
                returns = self._compute_returns(rewards, gamma)
                
                # Compute policy gradient loss
                with tf.GradientTape() as tape:
                    # Forward pass
                    action_probs = policy(states, training=True)
                    
                    # Log probabilities of taken actions
                    action_indices = tf.stack(
                        [tf.range(len(actions)), actions], axis=1
                    )
                    log_probs = tf.math.log(
                        tf.gather_nd(action_probs, action_indices) + 1e-10
                    )
                    
                    # REINFORCE loss: -log(π) * G
                    loss = -tf.reduce_mean(log_probs * returns)
                
                # Backpropagation
                gradients = tape.gradient(loss, policy.trainable_variables)
                optimizer.apply_gradients(zip(gradients, policy.trainable_variables))
                
                total_loss += float(loss)
            
            avg_loss = total_loss / episode_count
            avg_reward = np.mean([ep["total_reward"] for ep in episodes_data])
            
            metrics = {
                "avg_loss": avg_loss,
                "avg_reward": avg_reward,
                "episodes": episode_count,
            }
            
            # Store history
            if "reinforce" not in self.training_history:
                self.training_history["reinforce"] = []
            self.training_history["reinforce"].append(metrics)
            
            logger.info(
                f"REINFORCE trained - Avg loss: {avg_loss:.4f}, "
                f"Avg reward: {avg_reward:.2f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"REINFORCE training failed: {e}")
            raise RuntimeError(f"REINFORCE training failed: {e}") from e
    
    @timer
    def train_dqn(
        self,
        batch: Dict[str, np.ndarray],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        hidden_units: List[int] = [128, 64]
    ) -> Dict[str, float]:
        """Train DQN (Deep Q-Network) on a batch of experiences.
        
        Args:
            batch: Dictionary with states, actions, rewards, next_states, dones
            learning_rate: Learning rate
            gamma: Discount factor
            hidden_units: Q-network architecture
        
        Returns:
            Dictionary with training metrics
        
        Example:
            >>> batch = replay_buffer.sample(batch_size=32)
            >>> metrics = registry.train_dqn(batch, learning_rate=0.001)
        """
        validate_positive(learning_rate, "learning_rate")
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        
        try:
            # Build or get Q-network
            if "dqn" not in self.policies:
                q_network = self._build_q_network(hidden_units)
                optimizer = keras.optimizers.Adam(learning_rate)
                self.policies["dqn"] = {"model": q_network, "optimizer": optimizer}
            else:
                q_network = self.policies["dqn"]["model"]
                optimizer = self.policies["dqn"]["optimizer"]
            
            states = batch["states"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_states = batch["next_states"]
            dones = batch["dones"]
            
            # Compute Q-learning targets
            with tf.GradientTape() as tape:
                # Current Q-values
                q_values = q_network(states, training=True)
                action_indices = tf.stack([tf.range(len(actions)), actions], axis=1)
                q_values_for_actions = tf.gather_nd(q_values, action_indices)
                
                # Target Q-values
                next_q_values = q_network(next_states, training=False)
                max_next_q = tf.reduce_max(next_q_values, axis=1)
                targets = rewards + gamma * max_next_q * (1 - dones)
                
                # MSE loss
                loss = tf.reduce_mean(tf.square(targets - q_values_for_actions))
            
            # Backpropagation
            gradients = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
            
            metrics = {
                "loss": float(loss),
                "avg_q_value": float(tf.reduce_mean(q_values)),
                "batch_size": len(states),
            }
            
            # Store history
            if "dqn" not in self.training_history:
                self.training_history["dqn"] = []
            self.training_history["dqn"].append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"DQN training failed: {e}")
            raise RuntimeError(f"DQN training failed: {e}") from e
    
    def _build_q_network(
        self,
        hidden_units: List[int] = [128, 64]
    ) -> keras.Model:
        """Build Q-network for DQN.
        
        Args:
            hidden_units: List of hidden layer sizes
        
        Returns:
            Keras model
        """
        model = keras.Sequential([
            keras.layers.Dense(
                hidden_units[0],
                activation="relu",
                input_shape=(self.state_dim,)
            )
        ])
        
        for units in hidden_units[1:]:
            model.add(keras.layers.Dense(units, activation="relu"))
        
        # Output layer (Q-values for each action)
        model.add(keras.layers.Dense(self.action_dim, activation="linear"))
        
        return model
    
    def select_action(
        self,
        state: np.ndarray,
        policy_name: str = "reinforce",
        epsilon: float = 0.0
    ) -> int:
        """Select action using trained policy.
        
        Args:
            state: Current state
            policy_name: Name of policy to use
            epsilon: Exploration rate (for epsilon-greedy)
        
        Returns:
            Selected action
        
        Raises:
            RuntimeError: If policy not trained
        
        Example:
            >>> action = registry.select_action(state, "reinforce", epsilon=0.1)
        """
        if policy_name not in self.policies:
            raise RuntimeError(f"Policy '{policy_name}' not trained")
        
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        # Exploit learned policy
        policy = self.policies[policy_name]["model"]
        state_batch = state.reshape(1, -1)
        
        if policy_name == "reinforce":
            # Sample from policy distribution
            action_probs = policy(state_batch, training=False).numpy()[0]
            return np.random.choice(self.action_dim, p=action_probs)
        else:  # DQN
            # Select action with highest Q-value
            q_values = policy(state_batch, training=False).numpy()[0]
            return int(np.argmax(q_values))
    
    def _compute_returns(self, rewards: np.ndarray, gamma: float) -> np.ndarray:
        """Compute discounted returns for an episode.
        
        Args:
            rewards: Array of rewards
            gamma: Discount factor
        
        Returns:
            Array of discounted returns
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        # Normalize returns
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
        
        return returns
    
    def save_policy(self, policy_name: str, filepath: str) -> None:
        """Save trained policy to file.
        
        Args:
            policy_name: Name of policy to save
            filepath: Path to save file
        
        Raises:
            RuntimeError: If policy not found
        """
        if policy_name not in self.policies:
            raise RuntimeError(f"Policy '{policy_name}' not found")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        policy = self.policies[policy_name]["model"]
        policy.save(filepath)
        
        logger.info(f"Policy '{policy_name}' saved to {filepath}")
    
    def load_policy(self, policy_name: str, filepath: str) -> None:
        """Load policy from file.
        
        Args:
            policy_name: Name to assign to loaded policy
            filepath: Path to saved policy
        
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Policy file not found: {filepath}")
        
        policy = keras.models.load_model(filepath)
        optimizer = keras.optimizers.Adam()
        
        self.policies[policy_name] = {"model": policy, "optimizer": optimizer}
        
        logger.info(f"Policy '{policy_name}' loaded from {filepath}")
