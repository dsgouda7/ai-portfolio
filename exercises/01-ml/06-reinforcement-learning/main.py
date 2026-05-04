"""AgentAI - Interactive Reinforcement Learning Training

This script demonstrates:
1. Q-Learning with tabular Q-table
2. Deep Q-Network (DQN) with experience replay
3. Episode-by-episode learning curve
4. Plug-and-play agent comparison

Usage:
    python main.py

Expected runtime: 10-15 minutes (500 episodes per agent)
Expected output: Console shows episode rewards, learning curves, and final leaderboard
"""

from rich.console import Console
from rich.panel import Panel

from src.data import EnvironmentWrapper
from src.features import StatePreprocessor
from src.models import (
    QLearningAgent,
    DQNAgent,
    ExperimentRunner,
    AgentConfig,
)

console = Console()


def main():
    """Run complete RL pipeline with interactive feedback."""
    
    console.print(Panel.fit(
        "[bold cyan]AgentAI[/bold cyan]\n"
        "Interactive Reinforcement Learning Training",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 1: Initialize Environment
    # ============================================
    console.print("\n[bold cyan]🎮 ENVIRONMENT SETUP[/bold cyan]")
    
    env_wrapper = EnvironmentWrapper(
        env_name="CartPole-v1",
        max_steps=500,
        random_state=42
    )
    
    console.print(f"  ✓ Environment: CartPole-v1")
    console.print(f"  ✓ State dimension: {env_wrapper.state_dim}")
    console.print(f"  ✓ Action dimension: {env_wrapper.action_dim}")
    console.print(f"  ✓ Max steps per episode: 500")
    console.print("\n  [dim]Goal: Balance pole on cart by moving left/right[/dim]")
    console.print("  [dim]Solved: Average reward >195 over 100 consecutive episodes[/dim]")
    
    # ============================================
    # STEP 2: Configure Training
    # ============================================
    console.print("\n[bold cyan]⚙️  TRAINING CONFIGURATION[/bold cyan]")
    
    config = AgentConfig(
        episodes=500,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        buffer_size=10000,
        hidden_units=[128, 64],
        random_state=42
    )
    
    console.print(f"  ✓ Episodes: {config.episodes}")
    console.print(f"  ✓ Learning rate: {config.learning_rate}")
    console.print(f"  ✓ Discount factor (γ): {config.gamma}")
    console.print(f"  ✓ Epsilon decay: {config.epsilon_start} → {config.epsilon_end}")
    console.print(f"  ✓ DQN batch size: {config.batch_size}")
    console.print(f"  ✓ Replay buffer size: {config.buffer_size}")
    
    # ============================================
    # STEP 3: Initialize Agents (Plug-and-Play)
    # ============================================
    console.print("\n[bold cyan]🤖 AGENT INITIALIZATION[/bold cyan]")
    
    runner = ExperimentRunner(env_wrapper)
    
    # TODO: Register Q-Learning agent
    runner.register(
        "Q-Learning (α=0.1)",
        QLearningAgent(
            state_dim=env_wrapper.state_dim,
            action_dim=env_wrapper.action_dim,
            learning_rate=0.1
        )
    )
    
    # TODO: Register DQN agent
    runner.register(
        "DQN (128-64)",
        DQNAgent(
            state_dim=env_wrapper.state_dim,
            action_dim=env_wrapper.action_dim,
            hidden_units=[128, 64]
        )
    )
    
    # Optional: Try different hyperparameters
    # runner.register("DQN (256-128)", DQNAgent(..., hidden_units=[256, 128]))
    # runner.register("Q-Learning (α=0.05)", QLearningAgent(..., learning_rate=0.05))
    
    # ============================================
    # STEP 4: Train All Agents (See Progress Live!)
    # ============================================
    console.print("\n[bold yellow]⏱️  Training will take ~10-15 minutes...[/bold yellow]")
    console.print("  [dim]You'll see episode rewards immediately as agents learn![/dim]\n")
    
    # Run experiment (trains all registered agents)
    runner.run_experiment(config)
    
    # ============================================
    # STEP 5: Compare Results
    # ============================================
    runner.print_leaderboard()
    
    # Optional: Plot learning curves
    # runner.plot_learning_curves()
    
    # ============================================
    # STEP 6: Save Best Agent
    # ============================================
    console.print("\n[bold cyan]💾 SAVING BEST AGENT[/bold cyan]")
    
    try:
        best_agent = runner.get_best_agent()
        best_agent.save("models/best_agent.pkl")
        console.print(f"  ✓ Saved best agent: {best_agent.name}", style="green")
    except Exception as e:
        console.print(f"  ✗ Failed to save agent: {e}", style="red")
    
    # ============================================
    # STEP 7: Test Best Agent (Optional)
    # ============================================
    console.print("\n[bold cyan]🎯 TESTING BEST AGENT[/bold cyan]")
    console.print("  [dim]Run test episodes with epsilon=0 (no exploration)[/dim]")
    
    test_episodes = 10
    test_rewards = []
    
    for episode in range(test_episodes):
        state, info = env_wrapper.reset(seed=42 + episode)
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            # Select action with no exploration (epsilon=0)
            action = best_agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, info = env_wrapper.step(action)
            
            episode_reward += reward
            state = next_state
            steps += 1
            done = terminated or truncated
        
        test_rewards.append(episode_reward)
        console.print(f"  Test episode {episode+1}: Reward={episode_reward:.1f}", style="cyan")
    
    avg_test_reward = sum(test_rewards) / len(test_rewards)
    console.print(f"\n  Average test reward: {avg_test_reward:.1f}", style="bold green")
    
    if avg_test_reward > 195:
        console.print("  🎉 [bold green]Environment SOLVED![/bold green] (>195 avg reward)")
    else:
        console.print(f"  ⚠️  [yellow]Not solved yet. Need {195 - avg_test_reward:.1f} more reward.[/yellow]")
    
    # ============================================
    # COMPLETE!
    # ============================================
    console.print("\n" + "="*60)
    console.print("[bold green]✓ TRAINING COMPLETE![/bold green]")
    console.print("="*60)
    
    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    console.print("  1. Check models/best_agent.pkl (saved model)")
    console.print("  2. Try different hyperparameters (learning rate, epsilon decay)")
    console.print("  3. Visualize learning curves: runner.plot_learning_curves()")
    console.print("  4. Deploy API: make docker-up (see README.md)")
    console.print("  5. Run tests: pytest tests/")
    
    console.print("\n[dim]💡 Tip: Increase episodes to 1000+ for better convergence[/dim]")
    
    # Cleanup
    env_wrapper.close()


if __name__ == "__main__":
    main()
