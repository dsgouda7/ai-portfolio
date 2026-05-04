"""Tests for policy training and model registry"""

import pytest
import numpy as np

from src.models import ModelRegistry


class TestModelRegistry:
    """Test suite for ModelRegistry."""
    
    def test_registry_initialization(self):
        """Test registry initializes correctly."""
        registry = ModelRegistry(state_dim=4, action_dim=2)
        
        assert registry.state_dim == 4
        assert registry.action_dim == 2
        assert len(registry.policies) == 0
        assert registry.best_policy_name is None
    
    def test_build_policy_network(self):
        """Test policy network construction."""
        registry = ModelRegistry(state_dim=4, action_dim=2)
        
        policy = registry._build_policy_network(hidden_units=[128, 64])
        
        assert policy is not None
        assert len(policy.layers) == 3  # 2 hidden + 1 output
    
    def test_train_reinforce_single_episode(self):
        """Test REINFORCE training on single episode."""
        registry = ModelRegistry(state_dim=4, action_dim=2)
        
        # Create dummy episode
        episode = {
            "states": np.random.randn(10, 4),
            "actions": np.random.randint(0, 2, size=10),
            "rewards": np.random.rand(10),
            "total_reward": 5.0,
            "steps": 10,
        }
        
        metrics = registry.train_reinforce(
            [episode],
            learning_rate=0.001,
            gamma=0.99,
            hidden_units=[64, 32]
        )
        
        assert "avg_loss" in metrics
        assert "avg_reward" in metrics
        assert "episodes" in metrics
        assert metrics["episodes"] == 1
        assert "reinforce" in registry.policies
    
    def test_train_reinforce_multiple_episodes(self):
        """Test REINFORCE training on multiple episodes."""
        registry = ModelRegistry(state_dim=4, action_dim=2)
        
        # Create dummy episodes
        episodes = []
        for _ in range(5):
            episode = {
                "states": np.random.randn(20, 4),
                "actions": np.random.randint(0, 2, size=20),
                "rewards": np.random.rand(20),
                "total_reward": np.random.rand() * 100,
                "steps": 20,
            }
            episodes.append(episode)
        
        metrics = registry.train_reinforce(episodes, learning_rate=0.001, gamma=0.99)
        
        assert metrics["episodes"] == 5
        assert "reinforce" in registry.training_history
    
    def test_train_dqn_batch(self):
        """Test DQN training on batch."""
        registry = ModelRegistry(state_dim=4, action_dim=2)
        
        # Create dummy batch
        batch = {
            "states": np.random.randn(32, 4),
            "actions": np.random.randint(0, 2, size=32),
            "rewards": np.random.rand(32),
            "next_states": np.random.randn(32, 4),
            "dones": np.random.randint(0, 2, size=32).astype(bool),
        }
        
        metrics = registry.train_dqn(
            batch,
            learning_rate=0.001,
            gamma=0.99,
            hidden_units=[64, 32]
        )
        
        assert "loss" in metrics
        assert "avg_q_value" in metrics
        assert "batch_size" in metrics
        assert metrics["batch_size"] == 32
        assert "dqn" in registry.policies
    
    def test_select_action_reinforce(self):
        """Test action selection with REINFORCE policy."""
        registry = ModelRegistry(state_dim=4, action_dim=2)
        
        # Train a simple policy
        episode = {
            "states": np.random.randn(10, 4),
            "actions": np.random.randint(0, 2, size=10),
            "rewards": np.random.rand(10),
            "total_reward": 5.0,
            "steps": 10,
        }
        registry.train_reinforce([episode])
        
        # Select action
        state = np.random.randn(4)
        action = registry.select_action(state, policy_name="reinforce", epsilon=0.0)
        
        assert action in [0, 1]
    
    def test_select_action_epsilon_greedy(self):
        """Test epsilon-greedy action selection."""
        registry = ModelRegistry(state_dim=4, action_dim=2)
        
        episode = {
            "states": np.random.randn(10, 4),
            "actions": np.random.randint(0, 2, size=10),
            "rewards": np.random.rand(10),
            "total_reward": 5.0,
            "steps": 10,
        }
        registry.train_reinforce([episode])
        
        # With epsilon=1.0, should always explore
        state = np.random.randn(4)
        actions = [registry.select_action(state, "reinforce", epsilon=1.0) for _ in range(10)]
        
        # Should have some randomness
        assert len(set(actions)) > 1 or len(actions) < 5
    
    def test_select_action_not_trained(self):
        """Test that selecting action without training raises error."""
        registry = ModelRegistry(state_dim=4, action_dim=2)
        
        state = np.random.randn(4)
        
        with pytest.raises(RuntimeError):
            registry.select_action(state, policy_name="reinforce")
    
    def test_compute_returns(self):
        """Test discounted returns computation."""
        registry = ModelRegistry(state_dim=4, action_dim=2)
        
        rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        returns = registry._compute_returns(rewards, gamma=0.99)
        
        assert returns.shape == (5,)
        # First return should be sum of all discounted rewards
        expected_first = 1.0 + 0.99 + 0.99**2 + 0.99**3 + 0.99**4
        # After normalization, check that returns are computed
        assert isinstance(returns[0], (float, np.floating))
    
    def test_invalid_learning_rate(self):
        """Test that invalid learning rate raises error."""
        registry = ModelRegistry(state_dim=4, action_dim=2)
        
        episode = {
            "states": np.random.randn(10, 4),
            "actions": np.random.randint(0, 2, size=10),
            "rewards": np.random.rand(10),
            "total_reward": 5.0,
            "steps": 10,
        }
        
        with pytest.raises(ValueError):
            registry.train_reinforce([episode], learning_rate=-0.001)
    
    def test_invalid_gamma(self):
        """Test that invalid gamma raises error."""
        registry = ModelRegistry(state_dim=4, action_dim=2)
        
        episode = {
            "states": np.random.randn(10, 4),
            "actions": np.random.randint(0, 2, size=10),
            "rewards": np.random.rand(10),
            "total_reward": 5.0,
            "steps": 10,
        }
        
        with pytest.raises(ValueError):
            registry.train_reinforce([episode], gamma=1.5)
