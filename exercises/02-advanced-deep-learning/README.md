# Exercise 02: Advanced Deep Learning — Transfer Learning & Model Comparison

> **Learning Goal:** Implement ResNet, Vision Transformer, and EfficientNet with transfer learning, mixed precision training, and advanced data augmentation  
> **Prerequisites:** Completed [notes/02-advanced_deep_learning/](../../notes/02-advanced_deep_learning/)  
> **Time Estimate:** 6-8 hours (coding) + 2-3 hours (experimentation)  
> **Difficulty:** ⭐⭐⭐ Advanced

> **📦 Infrastructure Note:** This exercise uses simplified infrastructure. Docker and containerization features have been removed to focus on core deep learning concepts. For production deployment patterns, see [exercises/06-ai_infrastructure/](../06-ai_infrastructure/).

---

## 🎯 **What You'll Implement**

Starting from function stubs and inline TODOs, you'll build an advanced deep learning system with:

### **Core Implementation (6-8 hours)**

| File | What You Implement | TODOs | Time |
|------|-------------------|-------|------|
| `src/models.py` | ResNet with residual connections & transfer learning | 2 methods | 90-135min |
| `src/models.py` | Vision Transformer (ViT) with self-attention | 2 methods | 90-135min |
| `src/models.py` | EfficientNet with compound scaling | 2 methods | 50-60min |
| `src/models.py` | ExperimentRunner with leaderboard | 2 methods | 30-40min |
| `src/features.py` | Data augmentation (rotation, flip, color jitter) | 3 classes | 90-110min |
| `src/features.py` | Mixup/CutMix augmentation | 1 method | 45-60min |
| `main.py` | Test evaluation + model saving | 2 sections | 30min |

**Interactive Experience:**
- ✅ See training progress per epoch (loss, accuracy)
- ✅ Compare ResNet vs Transformer vs EfficientNet
- ✅ Leaderboard shows best model automatically
- ✅ Experiment with frozen/unfrozen layers
- ✅ Rich console output with training curves

**Total:** 6-8 hours of focused coding

---

### **Advanced Concepts Covered**

#### **1. Residual Networks (ResNet)**
```python
# Residual block: F(x) + x
output = conv_block(x) + x  # Skip connection
```
- **Problem solved:** Vanishing gradients in deep networks
- **How it works:** Identity shortcuts allow gradients to flow backward
- **Result:** Can train 50-200+ layer networks

#### **2. Vision Transformers (ViT)**
```python
# Self-attention: Learn relationships between all patches
attention = softmax(Q @ K.T / sqrt(d)) @ V
```
- **Problem solved:** CNNs have limited receptive field
- **How it works:** Attention across all image patches (global view)
- **Result:** Better for large-scale pretraining (14M+ images)

#### **3. Transfer Learning**
```python
# Freeze early layers (generic features)
for param in model.layer1.parameters():
    param.requires_grad = False  # Don't update

# Fine-tune last layers (task-specific)
model.fc = nn.Linear(2048, num_classes)  # New random weights
```
- **When to use:** Limited data (<100k images)
- **Why it works:** Early layers learn edges/textures (universal)
- **Result:** Faster training, less overfitting

#### **4. Mixed Precision Training**
```python
with autocast():  # FP16 computation
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()  # Scaled gradients
```
- **Benefit:** 2-3x faster training, 50% less memory
- **Trade-off:** Slight numerical precision loss (negligible)

#### **5. Data Augmentation**
```python
transforms = [
    RandomCrop(224),      # Spatial variation
    RandomHorizontalFlip(), # Left-right symmetry
    ColorJitter(0.4),     # Lighting robustness
    RandomErasing(0.25)   # Occlusion robustness
]
```
- **Why critical:** Deep models need 50k+ images to avoid overfitting
- **Effect:** Augmentation = 5-10x more "virtual" training data

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
📊 LOADING DATA
  ✓ Train: 45,000 samples
  ✓ Val:   5,000 samples
  ✓ Test:  10,000 samples

🔧 DATA AUGMENTATION
  • Random crop + padding
  • Horizontal flip (p=0.5)
  • Color jitter (strength=0.4)
  • Random erasing (p=0.25)

🤖 MODEL TRAINING
Comparing 4 architectures...

Training ResNet-50 (freeze=3)...
  Epoch 1/5: Train Loss=1.8234 | Val Loss=1.4521 | Val Acc=48.23%
  Epoch 2/5: Train Loss=1.3891 | Val Loss=1.2156 | Val Acc=56.78%
  ...
  ✓ ResNet-50: Best Val Acc = 72.45% (epoch 4) | Time: 123.4s

Training ViT-B/16 (freeze=True)...
  ...

═══════════════════════════════════════════════════════════
  MODEL LEADERBOARD
═══════════════════════════════════════════════════════════
Rank  Model                    Val Acc   Val Loss  Time (s)
#1    ResNet-50 (freeze=0)     74.12%    0.7234    245.6
#2    ResNet-50 (freeze=3)     72.45%    0.8156    123.4
#3    EfficientNet-B0          71.89%    0.8421    98.7
#4    ViT-B/16 (freeze=True)   68.34%    0.9876    187.3
```

---

## 📚 **Implementation Guide**

### **Phase 1: Models (src/models.py)**

#### **TODO 1: ResNet with Transfer Learning** (90-135 min)
```python
def build_model(self) -> nn.Module:
    # 1. Load pretrained ResNet-50
    model = models.resnet50(pretrained=True)
    
    # 2. Freeze early layers
    if self.freeze_layers >= 1:
        for param in model.layer1.parameters():
            param.requires_grad = False
    # ... freeze layer2, layer3, layer4 as needed
    
    # 3. Replace final FC layer
    model.fc = nn.Linear(2048, self.num_classes)
    
    return model
```

**Key concepts:**
- **Residual block:** `F(x) + x` (identity shortcut)
- **Freezing:** `param.requires_grad = False` prevents updates
- **Fine-tuning:** Only train last layers (faster, less overfitting)

#### **TODO 2: ResNet Training with Mixed Precision** (60-90 min)
```python
def train(self, train_loader, val_loader, config):
    scaler = GradScaler()  # For FP16
    
    for epoch in range(config.epochs):
        for images, labels in train_loader:
            with autocast():  # FP16 computation
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
```

**Key concepts:**
- **autocast():** Automatic FP16/FP32 casting
- **GradScaler:** Scales gradients to prevent underflow
- **Speed:** 2-3x faster than FP32

#### **TODO 3: Vision Transformer** (90-135 min)
```python
def build_model(self) -> nn.Module:
    # 1. Load pretrained ViT-B/16
    model = models.vit_b_16(pretrained=True)
    
    # 2. Freeze encoder (12 transformer layers)
    if self.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    # 3. Replace classification head
    model.heads = nn.Linear(768, self.num_classes)
    
    return model
```

**Key concepts:**
- **Patch embedding:** Image → 196 patches (14×14 grid)
- **Self-attention:** Each patch attends to all others
- **[CLS] token:** Special token for classification

#### **TODO 4: EfficientNet** (50-60 min)
```python
def build_model(self) -> nn.Module:
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(1280, self.num_classes)
    return model
```

**Key concepts:**
- **Compound scaling:** Balance depth, width, resolution
- **MBConv:** Mobile inverted bottleneck (efficient)
- **SE blocks:** Channel attention

#### **TODO 5: Experiment Runner** (30-40 min)
```python
def run_experiment(self, train_loader, val_loader, config):
    for name, model in self.models.items():
        metrics = model.train(train_loader, val_loader, config)
        self.results.append({"name": name, **metrics})
    
    # Sort by validation accuracy
    self.results.sort(key=lambda x: x["val_acc"], reverse=True)
```

---

### **Phase 2: Data Augmentation (src/features.py)**

#### **TODO 6: Training Augmentation** (30-45 min)
```python
def get_train_transform(self):
    return T.Compose([
        T.RandomCrop(224, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(0.4, 0.4, 0.4, 0.04),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25)
    ])
```

**Why each augmentation:**
- **RandomCrop:** Spatial translation invariance
- **HorizontalFlip:** Left-right symmetry
- **Rotation:** Orientation invariance
- **ColorJitter:** Lighting robustness
- **RandomErasing:** Occlusion robustness

#### **TODO 7: Mixup/CutMix** (45-60 min)
```python
def __call__(self, batch):
    images, labels = batch
    lambda_param = np.random.beta(self.alpha, self.alpha)
    
    # Mixup: blend images and labels
    indices = torch.randperm(len(images))
    mixed_images = lambda_param * images + (1 - lambda_param) * images[indices]
    mixed_labels = lambda_param * labels + (1 - lambda_param) * labels[indices]
    
    return mixed_images, mixed_labels
```

**Why Mixup works:**
- Smooths decision boundaries
- Improves calibration (confidence matches accuracy)
- +2-3% accuracy boost

#### **TODO 8: Dataset Builder** (30-45 min)
```python
def build_datasets(self):
    train_transform = self.augmenter.get_train_transform()
    val_transform = self.augmenter.get_val_transform()
    
    train_dataset = CIFAR10(root=self.data_dir, train=True, 
                             download=True, transform=train_transform)
    test_dataset = CIFAR10(root=self.data_dir, train=False,
                            download=True, transform=val_transform)
    
    # Split train → train/val
    train_dataset, val_dataset = random_split(train_dataset, [45000, 5000])
    
    return train_dataset, val_dataset, test_dataset
```

---

### **Phase 3: Main Script (main.py)**

#### **TODO 9: Test Evaluation** (15-20 min)
```python
test_correct = 0
test_total = 0

best_model.model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = best_model.predict(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100.0 * test_correct / test_total
console.print(f"Test Accuracy: {test_acc:.2f}%", style="green")
```

#### **TODO 10: Model Saving** (10-15 min)
```python
best_model.save("models/best_model.pth")
console.print("✓ Model saved to models/best_model.pth", style="green")
```

---

## 🧪 **Experimentation Guide**

### **Experiment 1: Transfer Learning — How Much to Freeze?**
```python
# Try different freeze levels
runner.register("ResNet (freeze=0)", ResNetModel(freeze_layers=0))  # Train all
runner.register("ResNet (freeze=3)", ResNetModel(freeze_layers=3))  # Freeze early
runner.register("ResNet (freeze=4)", ResNetModel(freeze_layers=4))  # Freeze most
```

**Expected results:**
- `freeze=0`: Best accuracy, slowest (5-10% better)
- `freeze=3`: Good accuracy, 2x faster
- `freeze=4`: Lower accuracy, 3x faster

### **Experiment 2: ResNet vs Transformer**
```python
runner.register("ResNet-50", ResNetModel(num_classes=10))
runner.register("ViT-B/16", TransformerModel(num_classes=10))
```

**Expected results on CIFAR-10:**
- ResNet: 70-75% accuracy (stronger inductive bias)
- ViT: 65-70% accuracy (needs more data, but better at scale)

### **Experiment 3: Data Augmentation Ablation**
```python
# Baseline: No augmentation
augmenter = ImageAugmenter(
    horizontal_flip_prob=0.0,
    rotation_degrees=0,
    color_jitter_strength=0.0,
    random_erasing_prob=0.0
)

# Full augmentation
augmenter = ImageAugmenter(
    horizontal_flip_prob=0.5,
    rotation_degrees=15,
    color_jitter_strength=0.4,
    random_erasing_prob=0.25
)
```

**Expected improvement:** +5-10% accuracy with full augmentation

---

## 📊 **Expected Performance**

### **CIFAR-10 Benchmarks** (5 epochs, batch_size=32)

| Model | Val Acc | Train Time | Parameters |
|-------|---------|------------|------------|
| ResNet-50 (freeze=0) | 74-76% | ~4 min | 23.5M |
| ResNet-50 (freeze=3) | 71-73% | ~2 min | 2.1M trainable |
| EfficientNet-B0 | 71-73% | ~1.5 min | 4.0M |
| ViT-B/16 (freeze=True) | 68-70% | ~3 min | 0.8M trainable |

**Hardware:** NVIDIA GTX 1080 or equivalent  
**Note:** Increase to 20-50 epochs for production accuracy (85-90%)

---

## 🎓 **Key Takeaways**

### **1. When to Use Transfer Learning**
✅ **Use when:**
- Small dataset (<100k images)
- Similar domain to pretraining (ImageNet → natural images)
- Limited compute budget

❌ **Avoid when:**
- Large dataset (>1M images) — train from scratch
- Very different domain (natural → medical/satellite)

### **2. ResNet vs Transformer**
**ResNet:**
- ✅ Better inductive bias (convolution = translation invariance)
- ✅ Works well on small datasets (10k-100k)
- ✅ Faster inference

**Vision Transformer:**
- ✅ Better at scale (millions of images)
- ✅ Global receptive field (long-range dependencies)
- ❌ Needs more data (100k+ images)

### **3. Mixed Precision Training**
**Benefits:**
- 2-3x faster training
- 50% less GPU memory
- Enables larger batch sizes

**Trade-off:**
- Negligible accuracy loss (<0.1%)
- Requires GradScaler for stability

### **4. Data Augmentation**
**Critical for:**
- Small datasets (augmentation = 5-10x more data)
- Preventing overfitting
- Improving robustness

**Don't overdo it:**
- Too strong → blurry/unrealistic images
- Semantic preservation (don't flip digits, faces)

---

## 🐛 **Troubleshooting**

### **CUDA Out of Memory**
```python
# Reduce batch size
config.batch_size = 16  # Or 8

# Enable gradient accumulation (simulates larger batch)
config.gradient_accumulation_steps = 4  # Effective batch = 16×4 = 64
```

### **Model Overfitting (Train Acc > Val Acc by >10%)**
- Increase data augmentation strength
- Add weight decay: `config.weight_decay = 0.01`
- Freeze more layers: `freeze_layers=4`
- Add dropout (modify model)

### **Model Underfitting (Both Train/Val Acc Low)**
- Unfreeze more layers: `freeze_layers=0`
- Increase epochs: `config.epochs = 50`
- Increase learning rate: `config.learning_rate = 1e-3`
- Check data augmentation (not too strong)

### **Transformer Worse than ResNet**
- **Expected on small datasets** (CIFAR-10 = 50k images)
- Freeze encoder: `freeze_encoder=True`
- Increase data augmentation
- Try larger dataset (ImageNet, CIFAR-100)

---

## 📖 **Further Reading**

### **Papers**
- **ResNet:** [Deep Residual Learning (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- **Vision Transformer:** [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- **EfficientNet:** [Rethinking Model Scaling (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
- **Mixup:** [Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412)
- **CutMix:** [Regularization Strategy (Yun et al., 2019)](https://arxiv.org/abs/1905.04899)

### **Related Chapters**
- [notes/02-advanced_deep_learning/01-Introduction.md](../../notes/02-advanced_deep_learning/01-Introduction.md)
- [notes/02-advanced_deep_learning/02-ResNet.md](../../notes/02-advanced_deep_learning/02-ResNet.md)
- [notes/02-advanced_deep_learning/04-Vision-Transformers.md](../../notes/02-advanced_deep_learning/04-Vision-Transformers.md)

---

## ✅ **Success Criteria**

- [ ] ResNet build_model() implemented with transfer learning
- [ ] ResNet train() implemented with mixed precision
- [ ] Vision Transformer build_model() implemented
- [ ] Vision Transformer train() implemented with warm-up
- [ ] EfficientNet build_model() implemented
- [ ] Training augmentation pipeline implemented
- [ ] Validation augmentation pipeline implemented
- [ ] Mixup/CutMix augmentation implemented (optional but recommended)
- [ ] Dataset builder implemented
- [ ] DataLoader creation implemented
- [ ] Experiment runner implemented
- [ ] Leaderboard printing implemented
- [ ] Test evaluation implemented
- [ ] Model saving implemented

**Grand Challenge:** Achieve **75%+ validation accuracy** on CIFAR-10 with ResNet-50 in **20 epochs** using transfer learning and data augmentation.

---

## 🎯 **Next Steps**

1. **Optimize hyperparameters:** Grid search over learning rates, weight decay
2. **Two-stage training:** Freeze layers → train 10 epochs → unfreeze → train 20 more
3. **Advanced augmentation:** Implement CutMix, AutoAugment, RandAugment
4. **Model ensembling:** Average predictions from ResNet + ViT + EfficientNet
5. **Knowledge distillation:** Train small model using ResNet-50 as teacher

---

**Good luck! Remember:** Transfer learning is the secret to training deep models with limited data. Start with frozen layers, experiment, and iterate! 🚀
