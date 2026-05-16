# Ch.1 — Autoencoders for Compression and Reconstruction

> **The story.** In **2006**, Geoffrey Hinton and Ruslan Salakhutdinov published *"Reducing the Dimensionality of Data with Neural Networks"* in *Science* — demonstrating that deep neural networks could learn efficient, nonlinear compressions of high-dimensional data. Their autoencoder architecture was deceptively simple: an encoder network that compressed input to a low-dimensional "bottleneck," followed by a decoder that reconstructed the original input. The bottleneck forced the network to learn only the most essential features — throw away noise, keep structure. Training was unsupervised (no labels needed) via reconstruction loss: $\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$. By 2008, autoencoders were being used for dimensionality reduction (better than PCA for nonlinear manifolds), denoising (train to reconstruct clean images from corrupted inputs), and anomaly detection (high reconstruction error = anomaly, covered in [05-anomaly-detection/ch03_autoencoders](../../05-anomaly-detection/ch03_autoencoders/README.md)). Autoencoders became the foundation for variational autoencoders (VAEs, 2013) and ultimately latent diffusion models (Stable Diffusion, 2022) — all rely on learned compression via encoder-decoder pairs.
>
> **Where you are in the curriculum.** This is the first chapter in the Generative Models track. You've seen supervised learning (regression, classification) and unsupervised learning (clustering, PCA). Autoencoders bridge these: they're trained **without labels** (unsupervised) but learn a **task** (reconstruction). The bottleneck dimension is the key hyperparameter — too narrow and reconstruction fails, too wide and the network learns an identity mapping (no compression). By the end, you'll understand why autoencoders can't generate NEW samples (only reconstruct existing ones) and why they produce blurry outputs (MSE loss). This sets the stage for Ch.2 VAEs (probabilistic generation) and Ch.3 GANs (photorealistic quality).
>
> **Notation in this chapter.** $\mathbf{x} \in \mathbb{R}^d$ — input image (MNIST: $d=784$); $\mathbf{z} = f_{\text{enc}}(\mathbf{x}) \in \mathbb{R}^k$ — latent code, $k \ll d$; $\hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{z}) \in \mathbb{R}^d$ — reconstructed image; $\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{d}\|\mathbf{x} - \hat{\mathbf{x}}\|^2$ — reconstruction loss (MSE); $k$ — bottleneck dimension (compression ratio = $d/k$).

---

## 0 · The Challenge — Where We Are

> **The mission**: Build **SynthGen Studio** — a synthetic data generation system satisfying 5 constraints:
> 1. **QUALITY**: >90% classifier fooling rate (synthetic samples indistinguishable from real)
> 2. **DIVERSITY**: Cover all 10 digit classes; avoid mode collapse
> 3. **CONTROLLABILITY**: Generate specific digit on demand
> 4. **EFFICIENCY**: <200ms per 64-sample batch
> 5. **LATENT INTERPRETABILITY**: Smooth interpolation; meaningful latent arithmetic

**What we know so far (starting point):**
- MNIST: 60,000 training images, 28×28 pixels = 784 dimensions
- High-dimensional data often lies on a **low-dimensional manifold** (PCA showed this in [07-unsupervised-learning/ch02_dimensionality_reduction](../../07-unsupervised-learning/ch02_dimensionality_reduction/README.md))
- Question: Can a neural network learn a nonlinear compression better than PCA?

**What's blocking us:**
We don't yet have a way to:
1. **Compress** high-dimensional images to a meaningful low-dimensional representation
2. **Reconstruct** the original image from that representation
3. **Generate** new samples (this chapter won't solve this — Ch.2 VAEs will)

**What this chapter unlocks:**
**Autoencoders** — Encoder-decoder architecture trained to minimize reconstruction error:
- Encoder $f_{\text{enc}}$ maps 784D → 32D (learns compression)
- Decoder $f_{\text{dec}}$ maps 32D → 784D (learns reconstruction)
- Bottleneck forces network to learn essential features (digit identity, not noise)

By the end:
- Reconstructions capture digit identity but are **blurry** (MSE loss penalizes sharp edges)
- Latent space organizes by digit class (clusters of "3"s, "8"s)
- **Cannot generate new digits** — only reconstruct existing ones
- Fooling rate: ~60% (reconstructions distinguishable from originals)

---

## Animation

![Autoencoder compression and reconstruction](img/ch01-autoencoder-reconstruction.gif)

*Visual takeaway: Encoder compresses 784D MNIST digits → 32D latent space. Decoder reconstructs them. Latent space organizes by digit class. Reconstructions are blurry but recognizable.*

---

## 1 · Core Idea

An autoencoder learns to compress input data through a narrow bottleneck and then reconstruct it. Think of it as **lossy compression** (like JPEG): the bottleneck forces the network to throw away noise and keep only essential structure.

**Key components**:
1. **Encoder**: $\mathbf{z} = f_{\text{enc}}(\mathbf{x})$ — maps input to low-dimensional latent code
2. **Decoder**: $\hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{z})$ — reconstructs input from latent code
3. **Reconstruction loss**: $\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$ — penalizes differences between input and reconstruction

Training: Minimize reconstruction error via gradient descent. The bottleneck dimension $k$ controls compression ratio — small $k$ forces aggressive compression.

**What autoencoders learn**:
- Encoder learns which features matter (digit shape, stroke thickness)
- Decoder learns how to reconstruct from those features
- Latent space organizes by semantic similarity (similar digits → nearby latent codes)

**What autoencoders DON'T learn**:
- How to generate NEW samples (can only reconstruct existing ones)
- Sharp reconstructions (MSE loss favors blur)

---

## 2 · Running Example — CompressMNIST

**SynthGen Studio brief**: Before generating synthetic data, we need to understand data compression. Can we compress 28×28 MNIST digits (784 dimensions) to 32 dimensions without losing digit identity?

**Baseline (no compression)**:
- Store each digit: 784 floats × 4 bytes = 3.1 KB
- 60,000 training images = 188 MB

**Autoencoder compression (784→32→784)**:
- Compress to 32 floats × 4 bytes = 128 bytes latent code
- Compression ratio: 3.1 KB / 128 B = **24.5×**
- Quality: Reconstruction MSE = 0.02 (on normalized [0,1] scale)

**Reconstruction quality**:
```python
# Original digit "3"
x_original = mnist_test[0] # Shape: [784]

# Compress and reconstruct
z = encoder(x_original) # Shape: [32]
x_recon = decoder(z) # Shape: [784]

# MSE: 0.02
# Visual quality: Blurry but recognizable
```

**Latent space visualization**:
- Project 32D latent codes to 2D with t-SNE
- Result: Digit classes form clusters — "3"s cluster together, "8"s cluster together
- Similar digits (3 and 8) are closer than dissimilar digits (1 and 0)

**Why this matters**: Compression proves the bottleneck captures semantic information. But can we **sample** from this space to generate new digits? Not yet — the latent space is deterministic, not probabilistic. Ch.2 VAEs fix this.

---

## 3 · The Math

### 3.1 · Architecture

**Encoder**:
$$\mathbf{z} = f_{\text{enc}}(\mathbf{x}; \theta_{\text{enc}}) = \sigma_2(W_2 \cdot \sigma_1(W_1 \cdot \mathbf{x} + b_1) + b_2)$$

For MNIST:
- Input: $\mathbf{x} \in \mathbb{R}^{784}$
- Hidden layer: $W_1 \in \mathbb{R}^{400 \times 784}$, $\sigma_1 = \text{ReLU}$
- Latent layer: $W_2 \in \mathbb{R}^{32 \times 400}$, $\sigma_2 = \text{ReLU}$ (or linear)
- Output: $\mathbf{z} \in \mathbb{R}^{32}$

**Decoder**:
$$\hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{z}; \theta_{\text{dec}}) = \sigma_4(W_4 \cdot \sigma_3(W_3 \cdot \mathbf{z} + b_3) + b_4)$$

- Input: $\mathbf{z} \in \mathbb{R}^{32}$
- Hidden layer: $W_3 \in \mathbb{R}^{400 \times 32}$, $\sigma_3 = \text{ReLU}$
- Output layer: $W_4 \in \mathbb{R}^{784 \times 400}$, $\sigma_4 = \text{Sigmoid}$ (maps to [0,1])
- Output: $\hat{\mathbf{x}} \in \mathbb{R}^{784}$

### 3.2 · Loss Function

Mean Squared Error (MSE) between input and reconstruction:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{d} \sum_{i=1}^d (x_i - \hat{x}_i)^2$$

For MNIST ($d=784$):

$$\mathcal{L} = \frac{1}{784} \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

**Alternative losses**:
- **Binary Cross-Entropy** (for images in [0,1]): $\mathcal{L}_{\text{BCE}} = -\frac{1}{d}\sum_{i=1}^d [x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)]$
- **Mean Absolute Error**: $\mathcal{L}_{\text{MAE}} = \frac{1}{d}\sum_{i=1}^d |x_i - \hat{x}_i|$

BCE often produces sharper reconstructions than MSE for binary/grayscale images.

### 3.3 · Training Objective

Minimize reconstruction error over training set:

$$\theta^* = \arg\min_{\theta_{\text{enc}}, \theta_{\text{dec}}} \sum_{i=1}^N \mathcal{L}(\mathbf{x}^{(i)}, f_{\text{dec}}(f_{\text{enc}}(\mathbf{x}^{(i)})))$$

Gradients computed via backpropagation:
1. Forward pass: $\mathbf{x} \to \mathbf{z} \to \hat{\mathbf{x}}$
2. Compute loss: $\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}})$
3. Backward pass: Gradients flow through decoder → latent → encoder
4. Update weights: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$

---

## 4 · How It Works — Step by Step

### Step 1: Encoder Compresses Input

```python
# MNIST digit (28×28 = 784D)
x = torch.tensor([0.0, 0.1, ..., 0.9]) # Shape: [784]

# Encoder forward pass
h1 = torch.relu(encoder.fc1(x)) # [784] → [400]
z = torch.relu(encoder.fc2(h1)) # [400] → [32]

# Latent code: 32 dimensions
print(f"Latent code: {z[:5]}") # [0.23, 0.81, 0.03, 0.67, 0.12]
```

**What encoder learns**:
- Dim 0-5: Digit identity (which class?)
- Dim 6-15: Stroke thickness, slant
- Dim 16-31: Fine details, noise

### Step 2: Decoder Reconstructs from Latent

```python
# Decoder forward pass
h2 = torch.relu(decoder.fc1(z)) # [32] → [400]
x_recon = torch.sigmoid(decoder.fc2(h2)) # [400] → [784]

# Reconstructed digit
print(f"Reconstruction MSE: {torch.mean((x - x_recon)**2):.4f}") # 0.0187
```

**What decoder learns**:
- Map latent dimension 0 (digit class) → overall shape
- Map latent dimensions 6-15 → stroke details
- Ignore dimensions 16-31 (noise) → smooth reconstruction

### Step 3: Compute Reconstruction Loss

```python
# MSE loss
loss = nn.functional.mse_loss(x_recon, x)

# Backpropagate
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Gradient flow**:
```
x → encoder → z → decoder → x_recon → L_MSE
                ↑                      ↓
                └──────── ∇L ──────────┘
```

### Step 4: Visualize Latent Space

```python
# Encode all test samples
z_all = []
labels_all = []
for x, y in test_loader:
    z = encoder(x)
    z_all.append(z.detach())
    labels_all.append(y)

z_all = torch.cat(z_all) # Shape: [10000, 32]
labels_all = torch.cat(labels_all) # Shape: [10000]

# Reduce 32D → 2D with t-SNE
from sklearn.manifold import TSNE
z_2d = TSNE(n_components=2).fit_transform(z_all.numpy())

# Plot colored by digit class
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_all, cmap='tab10', s=1)
plt.colorbar(label='Digit class')
plt.title('Autoencoder latent space (t-SNE projection)')
```

**Result**: Clear clusters by digit class. Similar digits (3 and 5, or 4 and 9) overlap slightly. Dissimilar digits (1 and 0) are far apart.

---

## 5 · Key Diagrams

### 5.1 · Autoencoder Architecture

```
INPUT LAYER          ENCODER          BOTTLENECK        DECODER           OUTPUT LAYER
  (784D)             (400D)             (32D)            (400D)              (784D)

    x₁  ──┐                              z₁  ──┐                              x̂₁
    x₂  ──┤                              z₂  ──┤                              x̂₂
    x₃  ──┤─→ Linear → ReLU ──→ Linear → z₃  ──┤─→ Linear → ReLU ──→ Linear → x̂₃
    ...   │   (784→400)          (400→32) ...  │   (32→400)          (400→784) ...
   x₇₈₄ ──┘                             z₃₂ ──┘                             x̂₇₈₄
                                                                               ↓
                                                                            Sigmoid
                                                                            [0, 1]
                                    ↑                                          ↓
                               COMPRESSION                            RECONSTRUCTION
                              (24.5× smaller)                         (MSE loss ≈ 0.02)
```

### 5.2 · Latent Space (2D t-SNE Projection)

```
          Latent Space (z ∈ R³², projected to R²)

    ┌──────────────────────────────────────────┐
    │          ●●●                              │  ● = Digit class
    │        ●●3●3●●                            │
    │       ●●3●3●3●●          ○○○             │
    │        ●●3●3●●          ○○8○8○○           │
    │          ●●●            ○8○○8○8○          │
    │                          ○○8○8○○           │
    │   ●●●                     ○○○             │
    │  ●1●1●      △△△△                         │  Classes cluster
    │ ●●1●1●●    △△5△5△△                       │  by semantic
    │  ●1●1●    △△5△5△5△                       │  similarity
    │   ●●●      △△5△5△△                       │
    │             △△△△                          │
    │                      ×××××                │
    │   ■■■              ×××0×0×××              │
    │  ■4■4■            ××0×0×0×0××             │
    │ ■■4■4■■            ×××0×0×××              │
    │  ■4■4■              ×××××                 │
    │   ■■■                                     │
    └──────────────────────────────────────────┘

    Digit 3 and 8 closer than 1 and 0 (visual similarity)
```

### 5.3 · Compression-Reconstruction Trade-off

```
Bottleneck Size vs Reconstruction Quality

MSE
0.10│
    │  ●                       Too narrow: can't capture digit identity
0.08│    ●
    │      ●
0.06│        ●
    │          ●               Sweet spot: k=32 (compression 24.5×)
0.04│            ●●●
    │                ●●●
0.02│                    ●●●●●●●
    │_________________________●●●●●●●●●___
0.00│                                Too wide: learns identity
    └────────────────────────────────────────> Bottleneck size k
    2    4    8   16   32   64  128  256  512  784
```

---

## 6 · The Hyperparameter Dial — Bottleneck Dimension $k$

**The critical choice**: How narrow should the bottleneck be?

**Typical values**:
- $k = 2$ — Ultra-narrow (visualization only, terrible reconstruction)
- $k = 32$ — MNIST sweet spot (24.5× compression, MSE ≈ 0.02)
- $k = 128$ — CelebA faces (less compression, better quality)
- $k = 784$ — No compression (network learns identity, useless)

**Rule of thumb**: $k$ should be large enough to capture data variability, small enough to force meaningful compression. For MNIST (10 classes), $k \geq 10$ is minimum (need at least one dimension per class). $k=32$ gives headroom for intra-class variation (stroke thickness, slant).

**Underfitting vs overfitting**:
- $k$ too small → Reconstruction blurry, high MSE → **underfitting**
- $k$ too large → Network learns near-identity mapping → **overfitting** (no compression)

**Test**: Plot reconstruction MSE vs $k$. Find the "elbow" — smallest $k$ where MSE stops improving significantly.

---

## 7 · What Can Go Wrong

1. **Identity mapping (no compression)** — Bottleneck too wide ($k \approx d$), network learns $\hat{\mathbf{x}} \approx \mathbf{x}$ without compressing.
   - **Symptom**: MSE → 0, but latent codes don't cluster by digit class
   - **Fix**: Reduce $k$ (force compression), add sparsity penalty to latent code

2. **Blurry reconstructions** — MSE loss penalizes sharp edges (pixel-wise squared error).
   - **Symptom**: Reconstructions look "fuzzy," edges smeared
   - **Fix**: Use BCE loss instead of MSE, or switch to perceptual loss (Ch.3 GANs)

3. **Can't generate new samples** — Sampling random $\mathbf{z}$ produces garbage.
   - **Cause**: Latent space is deterministic, not probabilistic. Only codes produced by encoder are valid.
   - **Fix**: Use VAE (Ch.2) — probabilistic latent space allows sampling from prior

4. **Training doesn't converge** — Loss oscillates or plateaus early.
   - **Fix**: Reduce learning rate, check data normalization (MNIST should be in [0,1] or [-1,1])

5. **Latent space not interpretable** — Dimensions don't correspond to meaningful features.
   - **Fix**: Use β-VAE (Ch.2) for disentangled representations, or train with supervision (e.g., auxiliary classifier on latent code)

---

## 8 · Progress Check — What We Can Solve Now

**Unlocked capabilities:**
- **Compression**: 784D → 32D with 24.5× compression ratio, MSE ≈ 0.02
- **Reconstruction**: Can reconstruct training samples with recognizable digit identity
- **Latent space**: Organizes by semantic similarity (similar digits cluster together)

**Constraint progress**:
- **#1 QUALITY**: ~60% fooling rate (reconstructions distinguishable from originals due to blur)
- **#5 LATENT INTERPRETABILITY**: ✓ Latent space meaningful (clusters by digit class)

**Still can't solve:**
- **#1 QUALITY**: <90% fooling rate — MSE loss produces blur
- **#2 DIVERSITY**: Can't generate new digits — only reconstruct existing ones
- **#3 CONTROLLABILITY**: No generation capability
- **#4 EFFICIENCY**: Not relevant yet (no generation)

**Real-world status**: We can compress and reconstruct MNIST digits, proving a 32D latent space captures semantic information. But we can't **generate** new synthetic samples for data augmentation (SynthGen Studio requirement). Autoencoders are a **representation learning tool**, not a generative model.

**Next up:** Ch.2 VAEs make the bottleneck **probabilistic** — each image maps to a distribution $q(z|x)$, not a single point. This unlocks sampling from the prior $p(z) = \mathcal{N}(0, I)$ to generate NEW digits. Fooling rate improves to ~75%.

---

## 9 · Bridge to the Next Chapter

This chapter established the encoder-decoder framework and showed that a 32D bottleneck can capture digit identity (compression works). But the latent space is **deterministic** — only codes produced by the encoder are valid. Sample a random $\mathbf{z}$ and decode → garbage. **Ch.2 (VAEs)** makes the bottleneck **probabilistic**: the encoder outputs a distribution $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$, and the ELBO loss regularizes this distribution to match a unit Gaussian prior $p(z) = \mathcal{N}(0, I)$. After training, you can sample $z \sim \mathcal{N}(0, I)$, decode, and get a brand new digit. This unlocks generation capability (Constraint #2) and improves quality to ~75% fooling rate (still blurry, but better).

---

## Cross-Chapter Connections

### Autoencoders Reappear In:

1. **[05-anomaly-detection/ch03_autoencoders](../../05-anomaly-detection/ch03_autoencoders/README.md)** — Train autoencoder on normal credit card transactions. High reconstruction error → anomaly (fraud detection).

2. **[07-unsupervised-learning/ch02_dimensionality_reduction](../../07-unsupervised-learning/ch02_dimensionality_reduction/README.md)** — Compare autoencoder compression vs PCA/t-SNE. Autoencoders learn nonlinear manifolds (PCA is linear).

3. **Denoising autoencoders** — Train to reconstruct clean images from corrupted inputs. Used in image restoration, data cleaning.

4. **Sparse autoencoders** — Add $L_1$ penalty to latent code: $\mathcal{L} = \text{MSE} + \lambda \|\mathbf{z}\|_1$. Forces sparse representations (most latent dimensions → 0).

5. **Convolutional autoencoders** — Replace linear layers with convolutions. Better for high-resolution images (CelebA, ImageNet).

### Why This Matters for the Curriculum:

- **Compression unlocks downstream tasks** — Anomaly detection, clustering, visualization all benefit from learned latent representations
- **Autoencoders are the foundation for VAEs** — Understanding deterministic bottlenecks makes probabilistic bottlenecks (Ch.2) intuitive
- **MSE blur problem motivates GANs** — Pixel-wise losses fail perceptually → adversarial training (Ch.3)
