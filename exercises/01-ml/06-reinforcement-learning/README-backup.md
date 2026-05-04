# Exercise 06: AgentAI — Production Reinforcement Learning System

> **Grand Challenge:** Build a production-grade RL agent API that achieves >195 average reward on CartPole-v1 over 100 episodes while meeting 5 production constraints.

**Scaffolding Level:** 🟢 Heavy (learn the workflow)

---

## Objective

Implement a complete RL pipeline with production patterns:
- >195 average reward on CartPole-v1 (100 consecutive episodes)
- <100ms inference latency for action selection (p99)
- Policy persistence and versioning
- Error handling and input validation
- Configuration-driven training
- Episode-based training with replay buffer (for DQN)

---

## What You'll Learn

- Environment interaction (Gymnasium/Gym API)
- Policy gradient methods (REINFORCE)
- Value-based methods (DQN with experience replay)
- Episode collection and reward tracking
- State preprocessing and normalization
- Policy network architecture (TensorFlow/Keras)
- REST API for agent deployment
- Unit testing for RL systems

---

## Setup

**Unix/macOS/WSL:**
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

---

## Project Structure

```
06-reinforcement-learning/
├── requirements.txt          # Dependencies
├── setup.sh / setup.ps1      # Environment setup
├── config.yaml               # Hyperparameters
├── Makefile                  # Common commands
├── README.md                 # This file
├── src/
│   ├── __init__.py           # Package initialization
│   ├── utils.py              # Logging, seeding, timing
│   ├── data.py               # Environment wrapper, episode collection
│   ├── features.py           # State preprocessing, normalization
│   ├── models.py             # ModelRegistry: REINFORCE, DQN
│   ├── evaluate.py           # RL metrics: reward, convergence
│   ├── monitoring.py         # Prometheus metrics
│   └── api.py                # Flask API: /act, /train
├── tests/
│   ├── conftest.py           # Pytest configuration
│   ├── test_data.py          # Environment tests
│   ├── test_features.py      # State preprocessing tests
│   ├── test_models.py        # Policy training tests
│   └── test_api.py           # API endpoint tests
├── models/                   # Saved policies
├── data/                     # Episode logs
└── logs/                     # Application logs
```

---

## Success Criteria

Your exercise is complete when:
- [ ] All tests pass: `pytest tests/`
- [ ] Average reward >195 over 100 consecutive episodes on CartPole-v1
- [ ] API `/act` endpoint returns actions in <100ms
- [ ] Code passes linting: `black . && flake8 src/`
- [ ] Learning curve shows convergence
- [ ] Policy network successfully saved/loaded

---

## Key Differences from Supervised Learning

**Environment Interaction:**
- No fixed dataset — agent collects data through exploration
- State → Action → Reward feedback loop
- Episode-based training (multiple steps per episode)

**Objective:**
- Maximize cumulative reward (not minimize loss on labels)
- Balance exploration (try new actions) vs exploitation (use known good actions)

**Evaluation:**
- Success measured by average episode reward
- Convergence tracked over training episodes
- Win rate (e.g., CartPole: episode length = 500)

---

## API Endpoints

### `POST /act`
Select action given current state.

**Request:**
```json
{
  "state": [0.1, 0.2, -0.3, 0.4]
}
```

**Response:**
```json
{
  "action": 1,
  "policy": "REINFORCE",
  "epsilon": 0.05
}
```

### `POST /train`
Run training episodes.

**Request:**
```json
{
  "episodes": 100,
  "save_policy": true
}
```

**Response:**
```json
{
  "episodes_completed": 100,
  "avg_reward": 198.5,
  "policy_saved": true
}
```

### `GET /health`
Health check.

**Response:**
```json
{
  "status": "healthy",
  "policy_loaded": true,
  "environment": "CartPole-v1"
}
```

---

## Resources

**Concept Review:**
- [notes/01-ml/06-reinforcement-learning/](../../notes/01-ml/06-reinforcement-learning/) — Complete track (if available)
- Gymnasium docs: https://gymnasium.farama.org/

**Implementation Guides:**
- See `src/` for scaffolded code and hints
- Refer to Track 01 (Regression) for production patterns

---

## Training Commands

```bash
# Train policy (REINFORCE)
make train

# Run test suite
make test

# Start API server
make serve

# View MLflow dashboard
make mlflow-ui

# Docker deployment
make docker-build
make docker-run
```

---

## Expected Performance

**CartPole-v1 Benchmarks:**
- Random policy: ~22 average reward
- REINFORCE (well-tuned): >195 after 500-1000 episodes
- DQN: >195 after 300-500 episodes
- Convergence time: 5-10 minutes on CPU

---

## Common Pitfalls

1. **No state normalization**: CartPole states have different scales
2. **Exploration too low**: Agent gets stuck in local optima
3. **Learning rate too high**: Policy oscillates, doesn't converge
4. **Batch size too small (DQN)**: High variance in Q-value estimates
5. **No early stopping**: Wasted computation after convergence

---

## Next Steps

After completing this exercise:
- Experiment with Lunar Lander (harder environment)
- Implement Actor-Critic (A2C/A3C)
- Add multi-agent coordination
- Deploy to cloud with autoscaling

---

**Good luck! 🎮🤖**

