"""Tests for environment wrapper and episode collection"""

import pytest
import numpy as np

from src.data import EnvironmentWrapper, EpisodeCollector, ReplayBuffer


class TestEnvironmentWrapper:
    """Test suite for EnvironmentWrapper."""
    
    def test_environment_initialization(self):
        """Test environment loads successfully."""
        wrapper = EnvironmentWrapper("CartPole-v1", max_steps=500)
        
        assert wrapper.env_name == "CartPole-v1"
        assert wrapper.state_dim == 4  # CartPole has 4D state
        assert wrapper.action_dim == 2  # CartPole has 2 actions
        assert wrapper.max_steps == 500
    
    def test_environment_reset(self):
        """Test environment reset returns valid state."""
        wrapper = EnvironmentWrapper("CartPole-v1")
        state, info = wrapper.reset()
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (4,)  # CartPole state dimension
        assert isinstance(info, dict)
    
    def test_environment_step(self):
        """Test environment step execution."""
        wrapper = EnvironmentWrapper("CartPole-v1")
        state, _ = wrapper.reset()
        
        action = wrapper.sample_action()
        next_state, reward, terminated, truncated, info = wrapper.step(action)
        
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_environment_sample_action(self):
        """Test action sampling."""
        wrapper = EnvironmentWrapper("CartPole-v1")
        
        action = wrapper.sample_action()
        assert action in [0, 1]  # CartPole has 2 actions
    
    def test_environment_close(self):
        """Test environment cleanup."""
        wrapper = EnvironmentWrapper("CartPole-v1")
        wrapper.close()  # Should not raise
    
    def test_invalid_max_steps(self):
        """Test that invalid max_steps raises ValueError."""
        with pytest.raises(ValueError):
            EnvironmentWrapper("CartPole-v1", max_steps=-1)


class TestEpisodeCollector:
    """Test suite for EpisodeCollector."""
    
    def test_collector_initialization(self):
        """Test collector initializes correctly."""
        wrapper = EnvironmentWrapper("CartPole-v1")
        collector = EpisodeCollector(wrapper)
        
        assert collector.episode_count == 0
        assert collector.total_steps == 0
    
    def test_collect_episode_random_policy(self):
        """Test episode collection with random policy."""
        wrapper = EnvironmentWrapper("CartPole-v1")
        collector = EpisodeCollector(wrapper)
        
        # Random policy
        def random_policy(state):
            return np.random.randint(0, 2)
        
        episode = collector.collect_episode(random_policy, max_steps=100)
        
        assert "states" in episode
        assert "actions" in episode
        assert "rewards" in episode
        assert "total_reward" in episode
        assert "steps" in episode
        
        assert episode["steps"] <= 100
        assert len(episode["states"]) == episode["steps"]
        assert len(episode["actions"]) == episode["steps"]
        assert len(episode["rewards"]) == episode["steps"]
    
    def test_collector_increments_counts(self):
        """Test that collector tracks episode counts."""
        wrapper = EnvironmentWrapper("CartPole-v1")
        collector = EpisodeCollector(wrapper)
        
        def random_policy(state):
            return np.random.randint(0, 2)
        
        collector.collect_episode(random_policy, max_steps=50)
        collector.collect_episode(random_policy, max_steps=50)
        
        assert collector.episode_count == 2
        assert collector.total_steps > 0


class TestReplayBuffer:
    """Test suite for ReplayBuffer."""
    
    def test_buffer_initialization(self):
        """Test buffer initializes with correct size."""
        buffer = ReplayBuffer(max_size=1000)
        
        assert len(buffer) == 0
        assert buffer.max_size == 1000
    
    def test_buffer_add_transition(self):
        """Test adding transition to buffer."""
        buffer = ReplayBuffer(max_size=100)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        next_state = np.array([0.5, 0.6, 0.7, 0.8])
        
        buffer.add(state, action=1, reward=1.0, next_state=next_state, done=False)
        
        assert len(buffer) == 1
    
    def test_buffer_sample_batch(self):
        """Test sampling batch from buffer."""
        buffer = ReplayBuffer(max_size=100)
        
        # Add 50 transitions
        for i in range(50):
            state = np.random.rand(4)
            next_state = np.random.rand(4)
            buffer.add(state, action=i % 2, reward=1.0, next_state=next_state, done=False)
        
        batch = buffer.sample(batch_size=32)
        
        assert batch["states"].shape == (32, 4)
        assert batch["actions"].shape == (32,)
        assert batch["rewards"].shape == (32,)
        assert batch["next_states"].shape == (32, 4)
        assert batch["dones"].shape == (32,)
    
    def test_buffer_sample_too_large(self):
        """Test that sampling larger than buffer size raises error."""
        buffer = ReplayBuffer(max_size=100)
        
        # Add only 10 transitions
        for i in range(10):
            state = np.random.rand(4)
            next_state = np.random.rand(4)
            buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)
        
        with pytest.raises(ValueError):
            buffer.sample(batch_size=20)
    
    def test_buffer_max_size_limit(self):
        """Test buffer respects max_size limit."""
        buffer = ReplayBuffer(max_size=10)
        
        # Add 20 transitions (should keep only last 10)
        for i in range(20):
            state = np.random.rand(4)
            next_state = np.random.rand(4)
            buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)
        
        assert len(buffer) == 10
