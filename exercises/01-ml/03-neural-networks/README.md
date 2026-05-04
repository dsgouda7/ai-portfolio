# Exercise 03: UnifiedAI — Neural Network Classification System

> **Learning Goal:** Implement MLP and CNN architectures with backpropagation, live training feedback, and architectural comparison  
> **Prerequisites:** Completed [notes/01-ml/03-neural-networks/](../../../notes/01-ml/03-neural-networks/) and exercises/01-ml/01-regression/  
> **Time Estimate:** 6-8 hours (coding) + 1-2 hours (experiments)  
> **Difficulty:** ⭐⭐⭐ Advanced

> **📝 Note:** This exercise focuses on core ML implementation. Infrastructure files (Docker, Makefile, prometheus.yml) have been removed to keep the exercise focused on neural network concepts. For production deployment patterns, see `exercises/06-ai_infrastructure/`.

---

## 🎯 What You'll Implement

**Core Implementation (6-8 hours):**

| File | What You Implement | TODOs | Time |
|------|-------------------|-------|------|
| `src/models.py` | MLP architecture (Dense + Dropout layers) | 1 build + 1 train | 45-60min |
| `src/models.py` | CNN architecture (Conv1D + MaxPool) | 1 build + 1 train | 45-60min |
| `src/models.py` | ExperimentRunner with leaderboard | 2 methods | 30min |
| `main.py` | Test evaluation + visualization | 2 sections | 45min |

**Interactive Experience:**
- ✅ See loss/accuracy for every epoch as training happens
- ✅ Compare 6 different architectures automatically
- ✅ Leaderboard shows best model with overfitting detection
- ✅ Rich console output with tables

**Total:** 6-8 hours + 1-2 hours experimentation

---

## 🧠 Neural Network Concepts You'll Master

### 1. Architecture Design

**Multi-Layer Perceptron (MLP):**
- Input → Dense(128) → ReLU → Dropout → Dense(64) → ReLU → Dropout → Output (Softmax)

**Convolutional Neural Network (CNN):**
- Input → Conv1D → MaxPool → Dropout → Conv1D → MaxPool → GlobalAvgPool → Dense → Output

### 2. Training Process

**Forward Pass:** Input flows through layers, each computing output = activation(weights * input + bias)

**Backpropagation:** Compute gradients using chain rule, starting from output layer

**Gradient Descent:** Update weights: W_new = W_old - learning_rate * gradient

### 3. Key Hyperparameters

| Parameter | Typical Values | Effect |
|-----------|----------------|--------|
| `learning_rate` | 0.0001 - 0.01 | Too high: divergence; Too low: slow |
| `batch_size` | 16, 32, 64, 128 | Larger: smoother gradients |
| `dropout` | 0.1 - 0.5 | Higher: more regularization |
| `architecture` | [64,32], [128,64,32] | Deeper: more capacity, more overfitting risk |

---

## 🚀 Quick Start

### 1. Setup Environment

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

### 2. Run Interactive Training

```bash
python main.py
```

**Expected output:**
```
🧠 NEURAL NETWORK TRAINING
→ Training Shallow MLP (64, 32)...
    Epoch  1: loss=0.8234, acc=0.7120, val_loss=0.6543, val_acc=0.7890
    Epoch  2: loss=0.5123, acc=0.8234, val_loss=0.4987, val_acc=0.8456
    ...
  ✓ Shallow MLP: Val Acc = 89.67% | Epochs: 23 | Time: 12.3s

🏆 LEADERBOARD
Best model: Deep CNN (64, 32, 16) | Val Acc: 91.23%
```

---

## 📚 Implementation Guide

### Phase 1: MLP Architecture (45-60 min)

**File:** `src/models.py` → `MLPClassifier.build_model()`

**What you'll implement:**
1. Create Sequential model
2. Add input layer
3. Loop through architecture to add Dense + Dropout layers
4. Add output layer (softmax)
5. Compile with Adam optimizer

**Key concepts:**
- **Dense layer:** Fully connected
- **ReLU:** f(x) = max(0, x)
- **Dropout:** Randomly zero out neurons (regularization)
- **Softmax:** Convert logits to probabilities

### Phase 2: MLP Training (30-45 min)

**File:** `src/models.py` → `MLPClassifier.train()`

**What you'll implement:**
1. Build model
2. Create custom Callback for live epoch feedback
3. Setup early stopping
4. Train with model.fit()
5. Extract and print metrics

**Concepts:**
- **Batch:** Subset of data (e.g., 32 samples)
- **Epoch:** One complete pass through dataset
- **Early stopping:** Stop if val loss stops improving

### Phase 3: CNN Architecture (45-60 min)

**File:** `src/models.py` → `CNNClassifier.build_model()`

**What you'll implement:**
1. Create Sequential model
2. Add input layer (2D: features × 1 channel)
3. Loop to add Conv1D + MaxPooling + Dropout
4. Add GlobalAveragePooling
5. Add Dense + output layer

**CNN concepts:**
- **Conv1D:** Learn local patterns
- **MaxPooling:** Downsample
- **GlobalAvgPool:** Reduce to 1D

### Phase 4: CNN Training (20-30 min)

**Key difference:** Must reshape input to (samples, features, 1)

### Phase 5: Experiment Framework (30 min)

**File:** `src/models.py` → `ExperimentRunner`

Implement:
1. `run_experiment()`: Train all registered networks
2. `print_leaderboard()`: Sort by val accuracy

---

## 🔬 Experimentation Guide

### Experiment 1: Architecture Depth
```python
runner.register("Shallow", MLPClassifier([64]))
runner.register("Deep", MLPClassifier([256, 128, 64, 32]))
```

**Question:** Does deeper always mean better?

### Experiment 2: Dropout Rates
```python
for dropout in [0.0, 0.1, 0.3, 0.5]:
    runner.register(f"dropout={dropout}", MLPClassifier([128, 64], dropout=dropout))
```

**Question:** What's the optimal dropout rate?

### Experiment 3: Learning Rates
```python
for lr in [0.0001, 0.001, 0.01]:
    runner.register(f"lr={lr}", MLPClassifier([128, 64], learning_rate=lr))
```

**Question:** What happens with too high or too low learning rates?

---

## 📊 Understanding Training Curves

### Healthy Training
- Loss decreases, accuracy increases
- Train and val metrics track each other

### Overfitting
- Train loss decreases, val loss increases
- Large gap between train/val accuracy

**Solutions:** Increase dropout, reduce capacity, early stopping

### Underfitting
- Both train and val metrics poor
- Plateaus early

**Solutions:** Increase capacity, train longer, decrease dropout

---

## 🐛 Common Issues

### Issue 1: Shape mismatch in CNN
**Solution:** Reshape input to (samples, features, 1)

### Issue 2: Loss is NaN
**Solutions:**
- Reduce learning rate
- Check data normalization

### Issue 3: Overfitting
**Solutions:**
- Increase dropout
- Reduce model capacity
- Early stopping (already implemented)

---

## ✅ Success Criteria

- [ ] All TODOs implemented
- [ ] `python main.py` runs without errors
- [ ] Live epoch feedback displays
- [ ] Leaderboard shows 6 models sorted by Val Acc
- [ ] Best model achieves Val Acc > 85%
- [ ] Test evaluation prints final metrics

---

## 🎓 What You've Learned

### Neural Network Fundamentals
✅ Forward pass, backpropagation, gradient descent  
✅ Activation functions: ReLU, softmax  

### Architecture Design
✅ MLP: Fully connected networks  
✅ CNN: Convolutional networks  
✅ Layer types: Dense, Conv1D, MaxPooling, Dropout  
✅ Regularization techniques  

### Training Dynamics
✅ Batch vs epoch concepts  
✅ Early stopping  
✅ Overfitting vs underfitting detection  

### Hyperparameter Tuning
✅ Learning rate, dropout rate, batch size effects  
✅ Architecture depth/width trade-offs  

---

## 📚 Resources

- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [notes/01-ml/03-neural-networks/](../../../notes/01-ml/03-neural-networks/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n Neural Networks Notes](http://cs231n.github.io/neural-networks-1/)

---

**Estimated completion time:** 6-8 hours + 1-2 hours experimentation

Good luck! 🚀

