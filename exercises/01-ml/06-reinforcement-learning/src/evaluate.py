"""Model evaluation and diagnostics for AgentAI

Provides: RLEvaluator for comprehensive RL agent assessment
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


logger = logging.getLogger("agentai")


class RLEvaluator:
    """Automated RL evaluation with visualizations and diagnostics.
    
    Provides:
    - Episode reward tracking
    - Convergence analysis
    - Learning curves
    - Win rate calculation
    - Success threshold checking
    
    Attributes:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        success_threshold: Threshold for success (e.g., 195 for CartPole-v1)
    
    Example:
        >>> evaluator = RLEvaluator(success_threshold=195.0)
        >>> evaluator.add_episode(reward=200.0, length=200)
        >>> metrics = evaluator.compute_metrics()
        >>> evaluator.plot_learning_curve()
    """
    
    def __init__(self, success_threshold: float = 195.0):
        """Initialize RL evaluator.
        
        Args:
            success_threshold: Reward threshold to consider agent successful
        """
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_threshold = success_threshold
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        logger.info(f"Initialized RLEvaluator (threshold={success_threshold})")
    
    def add_episode(self, reward: float, length: int) -> None:
        """Add episode results.
        
        Args:
            reward: Total episode reward
            length: Number of steps in episode
        
        Example:
            >>> evaluator.add_episode(reward=195.0, length=195)
        """
        self.episode_rewards.append(float(reward))
        self.episode_lengths.append(int(length))
    
    def compute_metrics(
        self,
        window_size: int = 100
    ) -> Dict[str, float]:
        """Compute RL evaluation metrics.
        
        Args:
            window_size: Window for moving average
        
        Returns:
            Dictionary with metrics:
                - avg_reward: Mean reward across all episodes
                - std_reward: Standard deviation of rewards
                - avg_length: Mean episode length
                - win_rate: % episodes above threshold
                - moving_avg: Recent moving average
                - convergence: Whether agent converged
        
        Example:
            >>> metrics = evaluator.compute_metrics(window_size=100)
            >>> print(f"Win rate: {metrics['win_rate']:.1f}%")
        """
        if len(self.episode_rewards) == 0:
            logger.warning("No episodes recorded")
            return {}
        
        rewards_array = np.array(self.episode_rewards)
        lengths_array = np.array(self.episode_lengths)
        
        # Basic metrics
        avg_reward = float(np.mean(rewards_array))
        std_reward = float(np.std(rewards_array))
        avg_length = float(np.mean(lengths_array))
        
        # Win rate (episodes above threshold)
        wins = np.sum(rewards_array >= self.success_threshold)
        win_rate = 100.0 * wins / len(rewards_array)
        
        # Moving average (last N episodes)
        if len(rewards_array) >= window_size:
            recent_rewards = rewards_array[-window_size:]
            moving_avg = float(np.mean(recent_rewards))
            convergence = moving_avg >= self.success_threshold
        else:
            moving_avg = avg_reward
            convergence = False
        
        metrics = {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "avg_length": avg_length,
            "win_rate": win_rate,
            "moving_avg": moving_avg,
            "convergence": convergence,
            "total_episodes": len(self.episode_rewards),
        }
        
        logger.info(
            f"Metrics - Avg reward: {avg_reward:.2f}, "
            f"Win rate: {win_rate:.1f}%, "
            f"Convergence: {convergence}"
        )
        
        return metrics
    
    def plot_learning_curve(
        self,
        window_size: int = 100,
        save_path: Optional[str] = None
    ) -> None:
        """Plot episode rewards over time with moving average.
        
        Args:
            window_size: Window for moving average
            save_path: Optional path to save figure
        
        Example:
            >>> evaluator.plot_learning_curve(window_size=100)
        """
        if len(self.episode_rewards) == 0:
            logger.warning("No episodes to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        rewards = np.array(self.episode_rewards)
        
        # Plot raw rewards
        ax.plot(episodes, rewards, alpha=0.3, label="Episode Reward", color="blue")
        
        # Plot moving average
        if len(rewards) >= window_size:
            moving_avg = np.convolve(
                rewards,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            ax.plot(
                episodes[window_size-1:],
                moving_avg,
                label=f"Moving Avg ({window_size})",
                color="red",
                linewidth=2
            )
        
        # Success threshold line
        ax.axhline(
            y=self.success_threshold,
            color="green",
            linestyle="--",
            label=f"Success Threshold ({self.success_threshold})"
        )
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Learning Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_episode_lengths(
        self,
        save_path: Optional[str] = None
    ) -> None:
        """Plot episode lengths over time.
        
        Args:
            save_path: Optional path to save figure
        
        Example:
            >>> evaluator.plot_episode_lengths()
        """
        if len(self.episode_lengths) == 0:
            logger.warning("No episodes to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = np.arange(1, len(self.episode_lengths) + 1)
        lengths = np.array(self.episode_lengths)
        
        ax.plot(episodes, lengths, alpha=0.6, color="purple")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Length (Steps)")
        ax.set_title("Episode Length Over Time")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Episode lengths plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_reward_distribution(
        self,
        save_path: Optional[str] = None
    ) -> None:
        """Plot histogram of episode rewards.
        
        Args:
            save_path: Optional path to save figure
        
        Example:
            >>> evaluator.plot_reward_distribution()
        """
        if len(self.episode_rewards) == 0:
            logger.warning("No episodes to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rewards = np.array(self.episode_rewards)
        
        ax.hist(rewards, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        ax.axvline(
            x=self.success_threshold,
            color="red",
            linestyle="--",
            label=f"Threshold ({self.success_threshold})"
        )
        ax.axvline(
            x=np.mean(rewards),
            color="green",
            linestyle="--",
            label=f"Mean ({np.mean(rewards):.1f})"
        )
        
        ax.set_xlabel("Episode Reward")
        ax.set_ylabel("Frequency")
        ax.set_title("Reward Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reward distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def check_convergence(
        self,
        window_size: int = 100,
        required_episodes: int = 100
    ) -> bool:
        """Check if agent has converged to success threshold.
        
        Args:
            window_size: Window for moving average
            required_episodes: Minimum episodes to consider convergence
        
        Returns:
            True if converged, False otherwise
        
        Example:
            >>> if evaluator.check_convergence(window_size=100):
            ...     print("Agent converged!")
        """
        if len(self.episode_rewards) < required_episodes:
            return False
        
        recent_rewards = np.array(self.episode_rewards[-window_size:])
        moving_avg = np.mean(recent_rewards)
        
        converged = moving_avg >= self.success_threshold
        
        if converged:
            logger.info(
                f"Convergence detected! Moving avg: {moving_avg:.2f} "
                f">= {self.success_threshold}"
            )
        
        return converged
    
    def get_summary(self) -> str:
        """Get text summary of evaluation results.
        
        Returns:
            Formatted summary string
        
        Example:
            >>> print(evaluator.get_summary())
        """
        if len(self.episode_rewards) == 0:
            return "No episodes recorded"
        
        metrics = self.compute_metrics()
        
        summary = f"""
RL Evaluation Summary
{'='*50}
Total Episodes: {metrics['total_episodes']}
Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}
Average Length: {metrics['avg_length']:.1f} steps
Win Rate: {metrics['win_rate']:.1f}%
Moving Average (last 100): {metrics['moving_avg']:.2f}
Convergence: {'✓ Yes' if metrics['convergence'] else '✗ No'}
Success Threshold: {self.success_threshold}
{'='*50}
"""
        return summary
