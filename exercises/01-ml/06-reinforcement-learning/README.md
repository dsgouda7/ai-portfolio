# Exercise 06: AgentAI — Interactive Reinforcement Learning System

> **Learning Goal:** Implement Q-Learning and Deep Q-Network (DQN) with plug-and-play experimentation and episode-by-episode feedback  
> **Prerequisites:** Completed [notes/01-ml/06-reinforcement-learning/](../../../notes/01-ml/06-reinforcement-learning/)  
> **Time Estimate:** 6-8 hours (coding) + 1 hour (deployment, optional)  
> **Difficulty:** ⭐⭐⭐ Advanced

---

## 🎯 **What You'll Implement**

Starting from function stubs and inline TODOs, you'll build a complete RL system with:

### **Core Implementation (6-8 hours)**

| File | What You Implement | TODOs | Time |
|------|-------------------|-------|------|
| `src/models.py` | Q-Learning with Bellman updates | 2 methods | 1h |
| `src/models.py` | DQN with experience replay | 4 methods | 2.5h |
| `src/models.py` | ExperimentRunner with leaderboard | 2 methods | 35min |
| `src/features.py` | State normalization with StandardScaler | 4 methods | 1h |
| `main.py` | Episode-by-episode training loop | Test evaluation | 30min |

**Interactive Experience:**
- ✅ See episode rewards immediately after each episode
- ✅ Watch epsilon decay in real-time (exploration → exploitation)
- ✅ Track Q-table size growth (Q-Learning)
- ✅ Monitor replay buffer filling (DQN)
- ✅ Leaderboard shows best agent automatically
- ✅ Rich console output with colors and progress bars

**Total:** 6-8 hours of focused coding (RL is more complex than supervised learning!)

---

### **What's Already Done (Utilities)**

These files are complete and reusable:
- ✅ `src/data.py` — Environment wrapper and episode collection
- ✅ `src/evaluate.py` — RL metrics computation
- ✅ `src/utils.py` — Logging and validation
- ✅ `src/monitoring.py` — Prometheus metrics
- ✅ `src/api.py` — Flask REST API (pre-built for deployment)

**Philosophy:** Focus on RL algorithms, not boilerplate.

---

> **📌 Infrastructure Note:** This exercise focuses purely on RL implementation. Docker, deployment, and monitoring infrastructure have been removed. Use `setup.ps1` / `setup.sh` for local environment setup.

---

## 🧠 **What Makes RL Different?**

Unlike supervised learning (01-regression/), RL has unique challenges:

| Supervised Learning | Reinforcement Learning |
|---------------------|------------------------|
| Fixed dataset | Agent collects data through exploration |
| Labels provided (y) | Agent receives delayed rewards |
| Train once on full dataset | Train incrementally after each episode |
| Cross-validation | Episode-based evaluation |
| MSE/R² metrics | Average episode reward |

**Key RL Concepts:**
1. **Exploration vs Exploitation:** Balance trying new actions (explore) vs using best known action (exploit)
2. **Credit Assignment:** Which actions led to reward? (Bellman equation solves this)
3. **Temporal Difference Learning:** Learn from one-step lookahead, not full episode
4. **Experience Replay:** Store transitions, sample randomly to break correlation

---

## 🚀 **Quick Start**

### **1. Setup Environment**

**PowerShell (Windows):**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

**Bash (Linux/Mac/WSL):**
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### **2. Run Interactive Training**

```bash
python main.py
```

**Expected output:**
```
🎮 ENVIRONMENT SETUP
  ✓ Environment: CartPole-v1
  ✓ State dimension: 4
  ✓ Action dimension: 2
  Goal: Balance pole on cart by moving left/right
  Solved: Average reward >195 over 100 consecutive episodes

🤖 REINFORCEMENT LEARNING TRAINING

→ Training Q-Learning (α=0.1)...
  Reward=15.0 | Steps=15 | ε=1.000 | Q-size=47
  Reward=23.0 | Steps=23 | ε=0.995 | Q-size=132
  ...
    Episode 50/500: Avg Reward (last 50) = 45.3
    Episode 100/500: Avg Reward (last 50) = 89.7
    Episode 150/500: Avg Reward (last 50) = 156.2
    Episode 200/500: Avg Reward (last 50) = 195.8  ← SOLVED!

→ Training DQN (128-64)...
  Reward=18.0 | Steps=18 | Loss=0.0234 | ε=1.000 | Buffer=18
  Reward=32.0 | Steps=32 | Loss=0.0187 | ε=0.995 | Buffer=50
  ...
    Episode 50/500: Avg Reward (last 50) = 52.1
    Episode 100/500: Avg Reward (last 50) = 112.4
    Episode 150/500: Avg Reward (last 50) = 183.9
    Episode 200/500: Avg Reward (last 50) = 210.3  ← SOLVED!

📊 LEADERBOARD (CartPole-v1)
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Agent             ┃ Avg Reward (Last 100) ┃ Success (>195) ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ DQN (128-64)      │              215.3    │       ✓       │
│ Q-Learning (α=0.1)│              198.7    │       ✓       │
└───────────────────┴───────────────────────┴───────────────┘

🏆 Best agent: DQN (128-64) | Avg Reward: 215.3
```

---

## 📚 **TODO Guide**

### **Q-Learning TODOs (1.5 hours)**

#### **1. `QLearningAgent.train_episode()` (40-60 min)**
Implement Q-Learning with Bellman equation:
```python
# Bellman equation for Q-Learning:
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                    ↑   ↑       ↑
                  reward discount  best next Q
```

**Key steps:**
1. Discretize continuous state (CartPole has continuous positions/velocities)
2. Epsilon-greedy action selection
3. Execute action, observe reward
4. Update Q-value using Bellman equation
5. Print episode reward immediately

**Hints:**
- Q-table is a dictionary: `{(state_key, action): Q-value}`
- Initialize unseen Q-values to 0
- High epsilon = explore, low epsilon = exploit
- Q-table grows as agent explores new states

#### **2. `QLearningAgent.select_action()` (10 min)**
Epsilon-greedy policy:
```python
if random() < epsilon:
    return random_action  # Explore
else:
    return argmax_a Q(s, a)  # Exploit
```

---

### **DQN TODOs (2.5 hours)**

#### **3. `DQNAgent._build_network()` (10 min)**
Build neural network Q-function approximator:
```
Input (state) → Dense(128, ReLU) → Dense(64, ReLU) → Output (Q-values)
   [x, ẋ, θ, θ̇]                                        [Q(s,left), Q(s,right)]
```

**Why neural network?**
- Q-table doesn't scale to continuous/high-dimensional states
- NN can generalize: Similar states → similar Q-values

#### **4. `DQNAgent._store_transition()` (5 min)**
Store experience in replay buffer:
```python
transition = (state, action, reward, next_state, done)
replay_buffer.append(transition)
```

**Why replay buffer?**
- Break temporal correlation between consecutive transitions
- Reuse past experiences for more sample-efficient learning

#### **5. `DQNAgent._sample_batch()` (15 min)**
Sample random minibatch from buffer:
```python
indices = random.choice(len(buffer), batch_size)
batch = [buffer[i] for i in indices]
return {"states": ..., "actions": ..., "rewards": ...}
```

#### **6. `DQNAgent.train_episode()` (60-90 min) — MOST COMPLEX**
Full DQN training loop with experience replay:

```python
# Pseudo-code:
for each step in episode:
    action = epsilon_greedy(state)
    next_state, reward = env.step(action)
    store_transition(state, action, reward, next_state, done)
    
    if buffer has enough samples:
        batch = sample_batch()
        
        # Compute Bellman target
        target = reward + gamma * max_a' Q(next_state, a')
        
        # Compute loss
        loss = MSE(Q(state, action), target)
        
        # Backprop
        update_network(loss)
```

**Key differences from Q-Learning:**
- No discretization (NN handles continuous states)
- Batch learning (not single transition)
- Experience replay (random sampling)
- Loss function (MSE between predicted and target Q)

---

### **ExperimentRunner TODOs (35 min)**

#### **7. `ExperimentRunner.run_experiment()` (20 min)**
Train all registered agents:
```python
for agent in agents:
    epsilon = epsilon_start
    for episode in range(num_episodes):
        metrics = agent.train_episode(env, epsilon, config)
        epsilon *= epsilon_decay  # Gradually reduce exploration
        
        if episode % 50 == 0:
            print_progress()  # Show learning curve
```

#### **8. `ExperimentRunner.print_leaderboard()` (15 min)**
Sort and display results:
```python
sorted_agents = sorted(results, key=lambda x: x["avg_reward_100"], reverse=True)
table.add_row(agent_name, avg_reward, "✓" if solved else "✗")
```

---

### **StatePreprocessor TODOs (1 hour)**

#### **9-12. State Normalization (15 min each)**
- `fit()`: Compute mean/std from training states
- `transform()`: Normalize states to zero mean, unit variance
- `fit_transform()`: Convenience method (fit + transform)
- `inverse_transform()`: Convert back to original scale

**Why normalize?**
- Neural networks train faster with normalized inputs
- Prevents large state values from dominating gradients
- CartPole states have different scales: position [-4.8, 4.8], velocity [-∞, ∞]

---

## 🔬 **Success Criteria**

Your exercise is complete when:
- [ ] All tests pass: `pytest tests/`
- [ ] Q-Learning achieves >195 average reward over 100 episodes
- [ ] DQN achieves >195 average reward over 100 episodes
- [ ] Episode rewards printed in real-time during training
- [ ] Leaderboard shows both agents sorted by performance
- [ ] Epsilon decays from 1.0 to 0.01 over training
- [ ] Code passes linting: `black . && flake8 src/`

**Stretch goals:**
- [ ] Plot learning curves: `runner.plot_learning_curves()`
- [ ] Try different hyperparameters (learning rate, network size)
- [ ] Implement target network for DQN (stabilizes learning)
- [ ] Add prioritized experience replay (weight important transitions)

---

## 🧪 **Testing**

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py::test_q_learning_agent

# Run with coverage
pytest --cov=src tests/
```

**Key tests:**
- `test_q_learning_bellman_update`: Verifies Q-value update is correct
- `test_dqn_network_architecture`: Checks Q-network shape
- `test_dqn_experience_replay`: Validates replay buffer sampling
- `test_epsilon_greedy`: Ensures exploration/exploitation balance

---

## 📖 **RL Concepts Explained**

### **Bellman Equation**
The heart of Q-Learning:
```
Q(s,a) = r + γ·max Q(s',a')
         ↑   ↑       ↑
      reward discount best future Q
```

- **Current Q:** Expected return from action `a` in state `s`
- **Target Q:** Immediate reward + discounted best future Q
- **Update rule:** Move current Q toward target Q

### **Exploration vs Exploitation**
- **Exploration:** Try random actions to discover new strategies
- **Exploitation:** Use best known action to maximize reward
- **Epsilon-greedy:** With probability ε, explore; otherwise exploit
- **Decay schedule:** Start with high ε (explore), end with low ε (exploit)

### **Discount Factor (γ)**
- **γ = 0:** Only care about immediate reward (myopic)
- **γ = 1:** Care equally about all future rewards
- **γ = 0.99:** Typical value — balance immediate and future rewards

### **Experience Replay**
Problem: Consecutive transitions are highly correlated  
Solution: Store transitions in buffer, sample randomly for training

**Benefits:**
- Breaks correlation between training samples
- Reuses past experiences (sample-efficient)
- Stabilizes learning (diverse minibatches)

---

## 🎨 **Visualization (Optional)**

### **Plot Learning Curves**
```python
runner.plot_learning_curves()
```

Shows:
- Episode reward over time for each agent
- Smoothed average reward (50-episode window)
- Success threshold (195 for CartPole-v1)

### **Watch Agent Play**
```python
env = gym.make("CartPole-v1", render_mode="human")
for episode in range(10):
    state, _ = env.reset()
    while True:
        action = agent.select_action(state, epsilon=0)
        state, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            break
```

---

## 🚀 **Optional: Production Deployment (1 hour)**

After implementing core features, deploy via Docker:
```bash
# Build container (uses shared infrastructure)
make docker-build

# Start API + Prometheus + Grafana
make docker-up

# Test API
curl -X POST http://localhost:5001/act \
  -H "Content-Type: application/json" \
  -d '{"state": [0.1, 0.2, -0.3, 0.4], "epsilon": 0.05}'
```

**Infrastructure:** All Docker/Prometheus configs live in `../../_infrastructure/`.  
No need to modify — just use it!

---

## 🔍 **Common Pitfalls**

### **Q-Learning doesn't converge**
- **Problem:** Q-table never stabilizes, reward stays low
- **Fix:** Reduce learning rate (α), increase episodes, tune discretization bins

### **DQN explodes (loss → NaN)**
- **Problem:** Gradient explosion, Q-values become infinite
- **Fix:** Lower learning rate, add gradient clipping, normalize states

### **Epsilon decay too fast**
- **Problem:** Agent stops exploring before finding good policy
- **Fix:** Slower decay (epsilon_decay=0.995 instead of 0.99), higher epsilon_end

### **Replay buffer not filling**
- **Problem:** Training starts before buffer has enough samples
- **Fix:** Check `if len(buffer) < batch_size: skip training`

---

## 📚 **Resources**

- [Sutton & Barto: Reinforcement Learning Book](http://incompleteideas.net/book/the-book.html)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [DQN Paper (Mnih et al., 2015)](https://arxiv.org/abs/1312.5602)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

## 🎯 **Next Steps**

After completing this exercise:
1. Try other Gymnasium environments (LunarLander, MountainCar)
2. Implement advanced RL algorithms (A3C, PPO, SAC)
3. Apply to real-world problems (robotics, game playing)
4. Explore multi-agent RL (competitive/cooperative agents)

---

**Happy Learning! 🤖**
