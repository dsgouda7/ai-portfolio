# Ch.3 — Generative Adversarial Networks (GANs)

> **The story.** In **June 2014**, Ian Goodfellow (then a PhD student at the University of Montreal, advised by Yoshua Bengio) sketched the GAN algorithm on a napkin at a bar — the idea came to him during a debate with friends about generative models. Within hours, he went home and coded the first implementation. Published at NIPS 2014, *"Generative Adversarial Nets"* introduced a radically different approach: instead of maximizing likelihood or minimizing reconstruction error, **train two networks in opposition** — a generator $G$ that creates fake samples, and a discriminator $D$ that learns to distinguish real from fake. The discriminator's job is to catch forgeries; the generator's job is to fool the discriminator. As training progresses, the generator gets better at forgery, the discriminator gets better at detection, until they reach a Nash equilibrium where generated samples are **indistinguishable from real data**. Yann LeCun called GANs "the most interesting idea in machine learning in the last 10 years." By 2016, GANs were generating photorealistic faces (ProgressiveGAN, StyleGAN), translating photos to paintings (CycleGAN), and super-resolving images (SRGAN). They dominated generative modeling until diffusion models emerged in 2020-2021, but GANs remain the gold standard for fast, high-fidelity synthesis where training data is abundant.
>
> **Where you are in the curriculum.** Ch.1 Autoencoders learned deterministic compression (60% fooling rate, blurry reconstructions). Ch.2 VAEs added probabilistic latent spaces for generation (75% fooling rate, still blurry due to MSE loss). Both approaches optimize pixel-wise reconstruction error — which penalizes sharp edges. This chapter **abandons reconstruction loss entirely**. The generator learns from a discriminator's binary signal: "real or fake?" This adversarial training forces the generator to capture perceptual realism (sharp edges, textures, fine details) that MSE-based losses miss. By the end, you'll have a GAN that generates MNIST digits with >90% fooling rate and scales to CelebA faces (64×64 RGB) with photorealistic quality.
>
> **Notation in this chapter.** $\mathbf{z} \in \mathbb{R}^k$ — latent noise vector, $\mathbf{z} \sim p_z(\mathbf{z}) = \mathcal{N}(0, \mathbf{I})$; $G_\theta : \mathbb{R}^k \to \mathbb{R}^d$ — generator network (forger); $D_\phi : \mathbb{R}^d \to [0,1]$ — discriminator network (detective), outputs probability "image is real"; $\mathbf{x} \sim p_{\text{data}}$ — real training sample; $\tilde{\mathbf{x}} = G(\mathbf{z})$ — generated (fake) sample; $V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$ — GAN objective (value function); $\mathcal{L}_D = -\log D(\mathbf{x}) - \log(1 - D(G(\mathbf{z})))$ — discriminator loss (maximize $V$); $\mathcal{L}_G = -\log D(G(\mathbf{z}))$ — generator loss (non-saturating variant, minimize $V$); BCE — Binary Cross-Entropy.

---

## 0 · The Challenge — Where We Are

> **The mission**: Build **SynthGen Studio** — a synthetic data generation system satisfying 5 constraints:
> 1. **QUALITY**: >90% classifier fooling rate (synthetic samples indistinguishable from real)
> 2. **DIVERSITY**: Cover all 10 digit classes; avoid mode collapse
> 3. **CONTROLLABILITY**: Generate specific digit on demand
> 4. **EFFICIENCY**: <200ms per 64-sample batch
> 5. **LATENT INTERPRETABILITY**: Smooth interpolation; meaningful latent arithmetic

**What we know so far (after Ch.1-2):**
- **Ch.1 Autoencoders**: Deterministic compression, MSE reconstruction → 60% fooling rate (blurry)
- **Ch.2 VAEs**: Probabilistic latent space, ELBO loss → 75% fooling rate (less blurry, can generate)
- Both use **pixel-wise reconstruction losses** (MSE) → penalize sharp edges, favor blur

**What's blocking us:**
The VAE's ELBO includes reconstruction term $\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]$, which for Gaussian likelihood becomes **MSE**: $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$. MSE computes per-pixel squared error:
- Sharp edge (black pixel next to white pixel): large gradient → high loss
- Blurred edge (black → gray → white): smaller gradient → lower loss

**The model learns to blur to minimize MSE**. But perceptually, sharp edges look real; blurred edges look synthetic. MSE doesn't measure perceptual quality — it measures pixel-wise distance.

**What this chapter unlocks:**
**Generative Adversarial Networks (GANs)** — Replace reconstruction loss with **adversarial training**:
1. **Discriminator $D$**: Binary classifier trained on real vs fake images
2. **Generator $G$**: Learns to fool $D$ by generating images $D$ classifies as real
3. **Training**: Alternating optimization — update $D$ to better detect fakes, update $G$ to better fool $D$

The discriminator becomes a **learned perceptual loss** — it tells the generator "this looks fake because the edges are blurry" or "the texture is wrong." The generator adapts. No pixel-wise MSE. Only the binary question: "Does this fool the expert?"

Result: **>90% fooling rate** on MNIST, photorealistic 64×64 CelebA faces.

---

## Animation

![GAN training dynamics — generator vs discriminator](img/ch03-gan-training.gif)

*Visual takeaway: Generator starts with noise → gibberish images. Discriminator easily classifies them as fake. Over training, generator learns to create sharper, more realistic digits that fool the discriminator. At equilibrium, discriminator accuracy → 50% (can't distinguish real from fake).*

---

## 1 · Core Idea — The Master Forger and the Art Detective

**Imagine a master art forger** trying to create counterfeit paintings, and a team of expert art detectives (discriminators) trained to spot fakes. This is GAN training:

### The Forger (Generator)

The forger starts with zero art knowledge. Their first attempts are laughably bad — random brush strokes, nonsensical compositions. But every time a painting is submitted for authentication:

- **If detected as fake**: The detective explains why — "The brush strokes are too uniform," "The aging patterns are wrong," "The color palette doesn't match the period."
- **If accepted as real**: The forger knows they succeeded and reinforces that technique.

Over time, the forger incorporates every lesson:
1. **Iteration 1**: Random noise → detective says "fake" with 100% confidence
2. **Iteration 100**: Vague shapes emerge → detective says "fake" with 80% confidence
3. **Iteration 1000**: Recognizable digits → detective says "fake" with 60% confidence
4. **Iteration 5000**: Photorealistic output → detective says "real" 50% of the time (guessing!)

The forger **never sees real paintings directly during training** — only the detective's binary verdict: "real or fake?" Yet by the end, their forgeries are indistinguishable from masterpieces.

### The Detective (Discriminator)

The detective's job is equally challenging. Early on, fake paintings are obvious — trivial to classify. But as the forger improves:

- **Iteration 1**: Easy classification (100% accuracy) → detective learns nothing useful
- **Iteration 1000**: Difficult classification (~70% accuracy) → detective must learn subtle features (edge sharpness, texture consistency)
- **Iteration 5000**: Near-impossible classification (~50% accuracy) → equilibrium

The detective **gets better at spotting fakes precisely because the forger gets better at creating them**. This adversarial pressure forces both networks to improve.

### The Equilibrium (Nash Equilibrium)

When the generator creates samples that perfectly match the real data distribution:
- $D(\mathbf{x}_{\text{real}}) = 0.5$ — "Could be real, could be fake, I genuinely can't tell."
- $D(G(\mathbf{z})) = 0.5$ — Same uncertainty on generated samples

At this point, **the forgeries are indistinguishable from originals**. The detective can't improve without more data. The forger can't improve without better feedback. Training converges.

**This is the GAN magic**: Two networks in opposition, pushing each other toward perfection. No reconstruction loss. No MSE. Just the evolutionary pressure of adversarial training.

---

## 2 · Running Example — ForgeMNIST GAN

**SynthGen Studio brief**: Generate 10,000 synthetic MNIST digits for data augmentation. Medical Director's requirement: "Radiologists must not be able to distinguish synthetic from real X-rays by visual inspection."

**Why VAEs fail the visual inspection test:**
```python
# Ch.2 VAE output
z = torch.randn(1, 32)
x_vae = vae_decoder(z) # Generated digit

# Radiologist verdict: "Blurry edges, lacks fine detail. Clearly synthetic."
# Fooling rate: 75% (classifier is fooled, but humans aren't)
```

**GAN solution:**
```python
# Ch.3 GAN output
z = torch.randn(1, 100) # Sample noise
x_gan = generator(z) # Generated digit

# Discriminator trained to distinguish real vs fake
# Generator trained to maximize D(G(z)) ("fool the discriminator")

# Radiologist verdict: "Sharp edges, realistic stroke variation. Could be real."
# Fooling rate: >90%
```

**The forger analogy in action:**
1. **Training begins**: Generator outputs random noise → discriminator correctly classifies as fake (99% confidence)
2. **Early training**: Generator learns digit shapes → discriminator detects imperfections (80% confidence)
3. **Mid training**: Generator adds texture detail → discriminator accuracy drops to 60%
4. **Convergence**: Generator produces photorealistic digits → discriminator guesses (50% accuracy) → **Nash equilibrium**

---

## 3 · The Math — Min-Max Game and Training Dynamics

### 3.1 · The GAN Objective (Value Function)

Goodfellow's original GAN formulation is a **two-player min-max game**:

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$$

Decompose this:
- **First term** $\mathbb{E}_{\mathbf{x}}[\log D(\mathbf{x})]$ — Discriminator reward for correctly classifying real samples as real ($D(\mathbf{x}) \to 1$)
- **Second term** $\mathbb{E}_{\mathbf{z}}[\log(1 - D(G(\mathbf{z})))]$ — Discriminator reward for correctly classifying fake samples as fake ($D(G(\mathbf{z})) \to 0$)

**Discriminator's goal**: **Maximize $V$** (maximize both terms)
- Real samples: $\log D(\mathbf{x}) \to \log(1) = 0$ (high confidence "real")
- Fake samples: $\log(1 - D(G(\mathbf{z}))) \to \log(1) = 0$ (high confidence "fake")

**Generator's goal**: **Minimize $V$** (minimize second term, can't affect first)
- Make $D(G(\mathbf{z})) \to 1$ (fool discriminator into thinking fake is real)
- This minimizes $\log(1 - D(G(\mathbf{z}))) \to \log(0) = -\infty$

**The adversarial game**: $D$ tries to maximize $V$, $G$ tries to minimize it. Training alternates between updating $D$ and $G$.

### 3.2 · Training Procedure (Alternating Optimization)

```python
# Hyperparameters
k = 5 # Train discriminator k times per generator update

for epoch in range(num_epochs):
    for real_batch in dataloader:
        # -------------------
        # Train Discriminator
        # -------------------
        for _ in range(k):
            # Sample real data
            x_real = real_batch

            # Sample fake data
            z = torch.randn(batch_size, latent_dim)
            x_fake = generator(z).detach() # Detach to not update G

            # Discriminator loss (Binary Cross-Entropy)
            d_loss_real = bce_loss(discriminator(x_real), torch.ones(batch_size, 1))
            d_loss_fake = bce_loss(discriminator(x_fake), torch.zeros(batch_size, 1))
            d_loss = d_loss_real + d_loss_fake

            # Update discriminator
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # -------------------
        # Train Generator
        # -------------------
        z = torch.randn(batch_size, latent_dim)
        x_fake = generator(z)

        # Generator loss (fool discriminator)
        g_loss = bce_loss(discriminator(x_fake), torch.ones(batch_size, 1))

        # Update generator
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
```

**Why $k > 1$?** (Train discriminator multiple times per generator update)
- Early training: Generator is terrible → discriminator saturates at 100% accuracy → gradients vanish
- Solution: Keep discriminator "ahead" of generator by training it more frequently
- Typical: $k=5$ for MNIST, $k=1$ for CelebA (adjust based on convergence)

### 3.3 · The Non-Saturating Generator Loss

The original generator loss $\mathcal{L}_G = \log(1 - D(G(\mathbf{z})))$ suffers from **vanishing gradients** early in training:
- When generator is bad: $D(G(\mathbf{z})) \approx 0$ → $\log(1 - 0) \approx 0$ → flat gradient
- No gradient signal → generator doesn't learn

**Goodfellow's fix** (non-saturating loss): Flip the objective:

$$\mathcal{L}_G = -\log D(G(\mathbf{z}))$$

**Why this works:**
- When generator is bad: $D(G(\mathbf{z})) \approx 0$ → $-\log(0) \to +\infty$ → **large gradient**
- Strong gradient signal → generator learns quickly even when outputs are terrible

This is the loss used in practice (including code above via `bce_loss(D(G(z)), torch.ones())` which computes $-\log D(G(z))$).

### 3.4 · Theoretical Guarantee (Goodfellow, 2014)

If discriminator is optimal (given generator $G$):

$$D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$$

Where $p_g$ is the distribution induced by generator. Plugging this into $V(D^*, G)$:

$$V(D^*, G) = -2\log 2 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)$$

Where $\text{JSD}$ is the Jensen-Shannon divergence. The generator's optimal strategy is to **minimize JSD** between real and generated distributions. When $p_g = p_{\text{data}}$:
- $\text{JSD} = 0$ → $D^*(\mathbf{x}) = 0.5$ everywhere → generator perfectly matches real data

**The forger's endgame**: When the generated distribution exactly matches the real distribution, even the perfect detective can't tell them apart (50% accuracy = random guessing).

---

## 4 · How It Works — Step by Step

### Step 1: Generator Architecture (DCGAN)

**Deep Convolutional GAN** (Radford et al., 2015) stabilized GAN training with architectural guidelines:

```
Input: z ∈ R^100 (latent noise)

Generator G:
    Linear(100 → 7×7×256) → Reshape(7, 7, 256)
    ↓
    ConvTranspose2d(256 → 128, kernel=4, stride=2, pad=1) → BatchNorm → ReLU
    [Output: 14×14×128]
    ↓
    ConvTranspose2d(128 → 64, kernel=4, stride=2, pad=1) → BatchNorm → ReLU
    [Output: 28×28×64]
    ↓
    Conv2d(64 → 1, kernel=3, stride=1, pad=1) → Tanh
    [Output: 28×28×1 (MNIST digit)]
```

**Key design choices:**
- **Transposed convolutions** (fractional-strided convolutions) for upsampling
- **BatchNorm** after every layer (except output) — stabilizes training
- **ReLU** in generator, **LeakyReLU** in discriminator
- **Tanh** output activation → images in [-1, 1]

### Step 2: Discriminator Architecture

```
Input: x ∈ R^{28×28×1} (real or fake MNIST digit)

Discriminator D:
    Conv2d(1 → 64, kernel=4, stride=2, pad=1) → LeakyReLU(0.2)
    [Output: 14×14×64]
    ↓
    Conv2d(64 → 128, kernel=4, stride=2, pad=1) → BatchNorm → LeakyReLU(0.2)
    [Output: 7×7×128]
    ↓
    Flatten → Linear(7×7×128 → 1) → Sigmoid
    [Output: scalar in (0, 1) = P(image is real)]
```

**Key design choices:**
- **Strided convolutions** (no pooling) — let network learn downsampling
- **LeakyReLU** (slope 0.2 for negative values) — prevents dying ReLUs
- **No BatchNorm in first layer** — allows discriminator to learn from raw pixels
- **Sigmoid output** → binary classification probability

### Step 3: Training Loop (Epoch-Level View)

```
Epoch 0:
    Generator: z ~ N(0,I) → random noise pixels
    Discriminator: 100% accuracy (trivial to spot fakes)
    D_loss: ~0.01 (confident), G_loss: ~5.0 (failing badly)

Epoch 10:
    Generator: Vague digit shapes emerge
    Discriminator: 90% accuracy
    D_loss: ~0.15, G_loss: ~2.3

Epoch 50:
    Generator: Clear digits with some artifacts
    Discriminator: 70% accuracy
    D_loss: ~0.45, G_loss: ~1.1

Epoch 200:
    Generator: Sharp, realistic digits
    Discriminator: 55% accuracy (near guessing)
    D_loss: ~0.68, G_loss: ~0.70

    → EQUILIBRIUM REACHED
```

### Step 4: Generation (After Training)

```python
# Generate 64 new MNIST digits
with torch.no_grad():
    z = torch.randn(64, 100) # Sample latent noise
    fake_images = generator(z) # Shape: [64, 1, 28, 28]

# Discriminator evaluation
d_output = discriminator(fake_images).mean()
print(f"Discriminator confidence (fake is real): {d_output:.2%}")
# Expected: ~50% (equilibrium)
```

**The forger's mastery**: Sample random noise → decode to photorealistic digit. No reconstruction. No MSE. Just learned mapping from noise to data distribution.

### Step 5: Interpolation in Latent Space

```python
z1 = torch.randn(1, 100) # Latent code for random digit A
z2 = torch.randn(1, 100) # Latent code for random digit B

for alpha in torch.linspace(0, 1, 10):
    z_interp = alpha * z1 + (1 - alpha) * z2
    x_interp = generator(z_interp)
    # Smooth morph from digit A to digit B
```

**Unlike VAE**: GANs don't have an encoder. Latent space is **implicit** — you can't encode a real image to $z$. But interpolation still works because the generator learns a smooth mapping.

---

## 5 · Key Diagrams

### 5.1 · GAN Training Dynamics

```
                    ADVERSARIAL TRAINING LOOP

    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   Real Data (x_real)              Noise (z ~ N(0,I))    │
    │        │                                 │               │
    │        │                                 ↓               │
    │        │                          ┌─────────────┐        │
    │        │                          │  GENERATOR  │        │
    │        │                          │  (Forger)   │        │
    │        │                          └──────┬──────┘        │
    │        │                                 │               │
    │        ↓                                 ↓               │
    │   ┌────────────────────────────────────────┐            │
    │   │        DISCRIMINATOR (Detective)       │            │
    │   │  Input: x (real or fake)               │            │
    │   │  Output: D(x) ∈ [0,1]                  │            │
    │   │  ("Probability image is real")         │            │
    │   └────────────┬───────────────────────────┘            │
    │                │                                         │
    │                ↓                                         │
    │   ┌────────────────────────────────────────┐            │
    │   │  Binary Classification Loss            │            │
    │   │  D_loss = -log D(x_real)               │            │
    │   │           -log(1 - D(G(z)))            │            │
    │   │  G_loss = -log D(G(z))                 │            │
    │   └────────────┬───────────────────────────┘            │
    │                │                                         │
    │                ↓                                         │
    │   Update D (improve fake detection)                     │
    │   Update G (improve forgery quality)                    │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

    After N iterations → D(x_real) ≈ D(G(z)) ≈ 0.5
    (Discriminator can't distinguish real from fake → Nash equilibrium)
```

### 5.2 · Loss Curves Over Training

```
    D_loss, G_loss

    5.0 │ G_loss
        │  *
    4.0 │   *
        │    *
    3.0 │     *___
        │         *___
    2.0 │             *___
        │                 *___
    1.0 │ D_loss               *___________
        │  ___---~~~~                      ~~~~~~~~~~
    0.0 │~~
        └────────────────────────────────────────────> Epoch
        0        50       100      150      200

    Early: G_loss high (generator failing), D_loss low (easy to classify)
    Late: G_loss ≈ D_loss ≈ 0.7 (equilibrium, both networks uncertain)
```

### 5.3 · Mode Collapse Visualization

```
    HEALTHY GAN (all modes covered)          MODE COLLAPSE

    Real Data Distribution                Real Data Distribution
    ┌────────────────────┐                ┌────────────────────┐
    │ ●1  ●2  ●3  ●4  ●5 │                │ ●1  ●2  ●3  ●4  ●5 │
    │                    │                │                    │
    │ ●6  ●7  ●8  ●9  ●0 │                │ ●6  ●7  ●8  ●9  ●0 │
    └────────────────────┘                └────────────────────┘

    Generated Distribution                Generated Distribution
    ┌────────────────────┐                ┌────────────────────┐
    │ ○1  ○2  ○3  ○4  ○5 │                │                    │
    │                    │                │         ○3  ○3  ○3 │
    │ ○6  ○7  ○8  ○9  ○0 │                │         ○3  ○3  ○3 │
    └────────────────────┘                └────────────────────┘
    All digit classes                     Only generates "3"s
    generated uniformly                   (found one "good" mode)
```

**The forger's failure mode**: Generator discovers that digit "3" reliably fools discriminator → produces only "3"s → ignores other digits. Discriminator adapts, generator collapses to a different mode. Cycle repeats.

---

## 6 · The Hyperparameter Dial — Stabilizing GAN Training

### 6.1 · Discriminator Update Frequency ($k$)

**The critical balance**: Train discriminator $k$ times per generator update.

**Typical values**:
- $k = 1$ — Standard (Goodfellow, 2014). Works if both networks are balanced.
- $k = 5$ — Conservative (DCGAN). Keeps discriminator "ahead" → prevents generator collapse.
- $k \to \infty$ — Train discriminator to convergence before updating generator (Unrolled GAN).

**Rule of thumb**: If generator loss plateaus early (discriminator saturates at 100% accuracy), increase $k$. If discriminator loss explodes (generator fools it too easily), decrease $k$.

**The forger-detective dance**: If the detective gets lazy (undertrained), the forger stops improving (no feedback). If the detective is too vigilant (overtrained), the forger gives up (task seems impossible).

### 6.2 · Learning Rates

**Typical values**:
- Discriminator: $\text{lr}_D = 0.0002$
- Generator: $\text{lr}_G = 0.0002$ (same as discriminator)

**Adam optimizer**: $\beta_1 = 0.5$, $\beta_2 = 0.999$ (DCGAN recommendation — $\beta_1 = 0.5$ reduces momentum oscillations)

**Asymmetric learning rates** (advanced):
- $\text{lr}_D = 0.0004$, $\text{lr}_G = 0.0001$ — Slow down generator to prevent collapse
- Used in StyleGAN, progressive GAN

### 6.3 · Architecture Choices (DCGAN Guidelines)

1. **Replace pooling with strided convolutions** — Let network learn spatial downsampling
2. **Use BatchNorm in both G and D** (except G output, D input) — Stabilizes training
3. **Remove fully-connected hidden layers** — All-convolutional architecture
4. **Use ReLU in G (except output), LeakyReLU in D** — Prevents dying neurons
5. **Use Tanh for G output, Sigmoid for D output** — Matches data range and classification task

### 6.4 · Preventing Mode Collapse

**Mode collapse**: Generator finds one "good" sample that fools discriminator, produces only that sample forever.

**Solutions**:
1. **Minibatch discrimination** (Salimans et al., 2016) — Discriminator sees batch statistics, penalizes lack of diversity
2. **Unrolled GAN** (Metz et al., 2016) — Train discriminator for $k$ steps, backprop through unrolled optimization to generator
3. **Wasserstein GAN (WGAN)** (Arjovsky et al., 2017) — Replace JS divergence with Wasserstein distance → smoother gradients, no mode collapse
4. **Spectral Normalization** (Miyato et al., 2018) — Constrain discriminator Lipschitz constant → stable training

---

## 7 · What Can Go Wrong

1. **Mode collapse** — Generator produces only one or few digit classes, ignores others.
   - **Symptom**: All generated samples look identical or cover <5 digit classes
   - **Fix**: Increase $k$ (train discriminator more), use minibatch discrimination, switch to WGAN

2. **Discriminator wins too easily** (generator collapse) — $D$ saturates at 100% accuracy, generator gradients vanish.
   - **Symptom**: $D_{\text{loss}} \to 0$, $G_{\text{loss}} \to$ constant (flat), generated images don't improve
   - **Fix**: Reduce $k$ (train discriminator less frequently), lower $\text{lr}_D$, add noise to discriminator inputs

3. **Generator wins too easily** (discriminator collapse) — $D$ outputs 50% on everything, provides no feedback.
   - **Symptom**: $D_{\text{loss}} \to \log(2) \approx 0.69$, $D(\mathbf{x}_{\text{real}}) \approx D(G(\mathbf{z})) \approx 0.5$
   - **Fix**: Increase $k$, raise $\text{lr}_D$, check if discriminator architecture is too shallow

4. **Training instability / oscillation** — Losses oscillate wildly, never converge.
   - **Symptom**: $D_{\text{loss}}$ and $G_{\text{loss}}$ swing between 0 and 5+, no equilibrium
   - **Fix**: Reduce both learning rates by 10×, use BatchNorm, clip discriminator weights (WGAN), use spectral normalization

5. **Checkerboard artifacts** — Generated images have grid-like patterns.
   - **Cause**: Transposed convolution kernel size not divisible by stride
   - **Fix**: Use kernel size 4, stride 2 (or kernel 6, stride 3) — divisibility prevents overlap artifacts

6. **Forgetting to normalize inputs** — MNIST must be in [-1, 1] if using Tanh output.
   - **Fix**: `x_real = (x_real - 0.5) / 0.5` (map [0,1] → [-1,1])

---

## 8 · Progress Check — What We Can Solve Now

**Unlocked capabilities:**
- **Photorealistic generation**: >90% fooling rate on MNIST (classifier + human visual inspection)
- **Sharp, detailed outputs**: No MSE blur — adversarial loss captures perceptual quality
- **Fast inference**: Generate 64 samples in <200ms (Constraint #4 ✓)
- **Scalability**: Same architecture works for CelebA 64×64 RGB faces (shown in notebook)

**Constraint progress**:
- **#1 QUALITY**: ✓ >90% fooling rate (target achieved!)
- **#2 DIVERSITY**: ✓ Covers all 10 digit classes (mode collapse prevented with minibatch discrimination)
- **#3 CONTROLLABILITY**: Partial — conditional GAN (cGAN) extends this (not covered in detail but straightforward)
- **#4 EFFICIENCY**: ✓ <200ms per 64-sample batch
- **#5 LATENT INTERPRETABILITY**: Partial — no encoder (can't map real image → $z$), but interpolation works

**Still can't solve:**
- **Encode real images to latent space** — GANs have no encoder (generator-only). VAEs have this.
- **Training stability without careful tuning** — GANs require hyperparameter sensitivity (learning rates, $k$, architecture)

**Real-world status**: We can generate synthetic MNIST digits that pass radiologist visual inspection (>90% fooling rate). Medical imaging use case **UNBLOCKED** — can augment training sets with synthetic X-rays that are HIPAA-compliant (no patient data) and perceptually indistinguishable from real scans. For CelebA faces, DCGAN generates 64×64 photorealistic portraits (notebook demonstrates this).

**Next steps:** For production deployment, consider **StyleGAN** (higher resolution, style control), **Progressive GAN** (incremental resolution growth), or **Conditional GAN** (controllable generation — "generate smiling face"). For medical imaging specifically, **CycleGAN** enables domain transfer (MRI → CT) without paired training data.

---

## 9 · Bridge to Advanced Generative Models

This chapter established **adversarial training** as an alternative to reconstruction-based losses — the forger-detective game produces photorealistic quality by directly optimizing perceptual similarity. But GANs have known issues: **mode collapse, training instability, no encoder**. Modern generative modeling has largely moved to **diffusion models** (covered in [04-multimodal-ai/ch04_diffusion_models](../../../04-multimodal-ai/ch04_diffusion_models/diffusion-models.md)):

- **Diffusion** replaces adversarial training with iterative denoising → stable training, no mode collapse
- **Latent diffusion** (Stable Diffusion) combines VAE encoder-decoder with diffusion U-Net → best of both worlds
- **Guidance** (classifier-free guidance) enables controllable generation without conditional training

Yet **GANs remain relevant** for:
- **Fast inference** (single forward pass vs 50 diffusion steps)
- **Style transfer** (CycleGAN, Pix2Pix)
- **Super-resolution** (SRGAN, ESRGAN)
- **Domain adaptation** (sim-to-real for robotics)

The forger-detective framework introduced here reappears in contrastive learning ([02-advanced-deep-learning/ch07_contrastive_learning](../../../02-advanced-deep-learning/ch07_contrastive_learning/README.md)) and adversarial robustness research.

---

## Cross-Chapter Connections

### GANs Reappear In:

1. **[04-multimodal-ai/ch04_diffusion_models](../../../04-multimodal-ai/ch04_diffusion_models/diffusion-models.md)** — Explains why diffusion models replaced GANs for high-resolution image synthesis (mode coverage, training stability)

2. **Adversarial training concepts** — Used in adversarial robustness (train classifier on adversarial examples to improve robustness)

3. **CycleGAN / Pix2Pix** — Domain translation (horse ↔ zebra, sketch ↔ photo) without paired training data. Used in sim-to-real robotics.

4. **StyleGAN** — State-of-the-art face generation (1024×1024). Introduces style-based generator architecture, used in deepfakes, character generation.

5. **Super-resolution (SRGAN)** — Upscale low-res images to high-res using GAN discriminator for perceptual loss.

### Why This Matters for Production:

**Medical imaging use case** (our running example):
- Train GAN on 800 real patient X-rays
- Generate 9,200 synthetic X-rays (mode collapse prevented with minibatch discrimination)
- Augmented dataset: 10,000 images (800 real + 9,200 synthetic)
- Train diagnostic CNN on augmented dataset → **92% AUC** (vs 85% on real-only)
- **Compliance**: Synthetic images contain zero patient information → HIPAA-safe

**E-commerce use case**:
- Train conditional GAN on product photos with different backgrounds
- Generate product images on white background → **consistent catalog appearance**
- **10× faster** than manual photo editing

**The forger's legacy**: GANs proved that adversarial training works. Even though diffusion models dominate high-resolution synthesis today, the adversarial game concept persists in contrastive learning, robustness research, and fast inference applications.
