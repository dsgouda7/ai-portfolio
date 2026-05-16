# Ch.2 — Variational Autoencoders (VAEs)

> **The story.** In **2013**, Diederik Kingma (University of Amsterdam) and Max Welling (Google DeepMind) published *"Auto-Encoding Variational Bayes"* at ICLR 2014 — a paper that unified deep learning and probabilistic inference by replacing the deterministic autoencoder bottleneck with a **learned probability distribution**. Their key insight: instead of encoding an image to a single point in latent space, encode it to a Gaussian distribution $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))$. This seemingly small change unlocked generation — sample $z \sim \mathcal{N}(0, I)$ from the prior, decode, and get a brand new image. The ELBO (Evidence Lower BOund) loss trains the encoder-decoder pair to maximize reconstruction quality while keeping the latent distribution close to a unit Gaussian (via KL divergence regularization). VAEs became the first practical deep generative model that could both compress existing data AND generate novel samples, paving the way for modern latent diffusion models (Stable Diffusion diffuses in VAE latent space). Within two years, VAEs were generating faces, interpolating between images, and discovering interpretable latent dimensions (smile direction, age axis) without supervision.
>
> **Where you are in the curriculum.** Ch.1 Autoencoders learned deterministic compression — 784D MNIST digits → 32D latent → reconstruction. But the bottleneck was a single point per image; you couldn't sample NEW digits, only reconstruct existing ones. This chapter makes the bottleneck **probabilistic**: each image maps to a distribution, not a point. The decoder becomes a generative model: sample from the prior, decode, generate. By the end, you'll have a VAE that generates novel handwritten digits and smoothly interpolates between them.
>
> **Notation in this chapter.** $\mathbf{x} \in \mathbb{R}^d$ — input image (MNIST: $d=784$); $\mathbf{z} \in \mathbb{R}^k$ — latent variable ($k=32$); $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x) \mathbf{I})$ — approximate posterior (encoder output); $p_\theta(x|z)$ — likelihood (decoder output); $p(z) = \mathcal{N}(0, \mathbf{I})$ — prior distribution; $\text{ELBO} = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$ — Evidence Lower BOund (loss function); $\mu_\phi(x), \log \sigma^2_\phi(x)$ — encoder outputs (mean and log-variance); $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ — noise for reparameterization trick; $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$ — reparameterized sample.

---

## 0 · The Challenge — Where We Are

> **The mission**: Build **SynthGen Studio** — a synthetic data generation system satisfying 5 constraints:
> 1. **QUALITY**: >90% classifier fooling rate (synthetic samples indistinguishable from real)
> 2. **DIVERSITY**: Cover all 10 digit classes; avoid mode collapse
> 3. **CONTROLLABILITY**: Generate specific digit on demand
> 4. **EFFICIENCY**: <200ms per 64-sample batch
> 5. **LATENT INTERPRETABILITY**: Smooth interpolation; meaningful latent arithmetic

**What we know so far (after Ch.1 Autoencoders):**
- Autoencoders compress 784D MNIST → 32D latent → 784D reconstruction
- MSE reconstruction loss: $\mathcal{L} = \frac{1}{d}\|\mathbf{x} - \hat{\mathbf{x}}\|^2$
- Latent space captures digit identity (cluster "3"s together, "8"s together)
- **But we can't generate NEW digits** — only reconstruct existing ones
- Fooling rate: ~60% (reconstructions blurry, distinguishable from real MNIST)

**What's blocking us:**
The autoencoder bottleneck is **deterministic** — each image maps to exactly one latent code $\mathbf{z} = f_{\text{enc}}(\mathbf{x})$. There's no "prior distribution" we can sample from. If we randomly sample a point in latent space and decode it, we get garbage — the decoder was never trained on random latent codes, only on codes produced by the encoder on real images.

**What this chapter unlocks:**
**Variational Autoencoders (VAEs)** — Replace the deterministic bottleneck with a learned probability distribution. The encoder outputs $\mu_\phi(\mathbf{x})$ and $\sigma^2_\phi(\mathbf{x})$ (mean and variance), defining a Gaussian $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$. The ELBO loss regularizes this distribution to match a unit Gaussian prior $p(z) = \mathcal{N}(0, I)$. After training:
1. **Sample $z \sim \mathcal{N}(0, I)$** — draw from the prior
2. **Decode $\hat{\mathbf{x}} = f_{\text{dec}}(z)$** — generate a NEW digit
3. **Interpolation**: linearly interpolate latent codes $z_1 \to z_2$ → smooth morphing between digits

Fooling rate improves to ~75% (still blurry due to MSE reconstruction term, but better than deterministic autoencoder).

---

## Animation

![VAE generation and interpolation](img/ch02-vae-generation.gif)

*Visual takeaway: VAE learns a smooth latent space. Sample from prior → generate new digits. Interpolate between two latent codes → watch one digit morph into another.*

---

## 1 · Core Idea — The Master Sculptor's Intuitive Map

**Think of a master sculptor training interns** in their studio. The master doesn't memorize every exact measurement of every model they've ever seen — that would be impossible and wasteful. Instead, over decades of experience, they build an **intuitive map** of human anatomy:

- **Proportions**: Shoulder-to-hip ratio typically 1.6:1 for adults; head is ~1/7.5 of body height
- **Ranges**: Arm span roughly equals height ± 5%; leg length is 45-50% of total height
- **Variability**: Some models are taller, some broader, some more muscular — but all lie within **anatomical plausibility bounds**

When the master hands these compressed anatomical principles to an intern (the decoder), the intern can:
1. **Reconstruct** the original model's pose by applying the stored proportions
2. **Generate** a brand new, anatomically-plausible human figure by sampling from the learned distribution of proportions

This is exactly what a VAE does:
- **Encoder (master sculptor)**: Maps each training image to a distribution over latent variables — "this digit is probably a '3', with slight variations in slant and curvature"
- **Latent space (intuitive anatomical map)**: A probability distribution $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$ that captures "what makes a valid digit" — the **plausibility bounds**
- **Decoder (intern)**: Takes a latent sample and reconstructs or generates a digit that respects those learned constraints

The ELBO loss trains both:
1. **Reconstruction accuracy** (can the intern recreate the original model?)
2. **Plausibility regularization** (does the intern's output stay within anatomical bounds?)

---

## 2 · Running Example — DigitForge VAE

**SynthGen Studio brief**: Generate 5,000 synthetic MNIST digits to augment a small training set (500 real digits). The Medical Director needs diverse samples — not 5,000 copies of the same "3".

**Why the autoencoder from Ch.1 fails:**
```python
# Ch.1 Autoencoder attempt
z = encoder(real_digit) # Deterministic latent code
x_recon = decoder(z) # Reconstructs the input digit

# Try to generate NEW digit:
z_random = torch.randn(32) # Sample random latent code
x_gen = decoder(z_random) # Output: GARBAGE (random noise)
```

The decoder was never trained on random latent codes — only on codes produced by the encoder. Random sampling gives nonsense.

**VAE solution:**
```python
# Ch.2 VAE
mu, log_var = encoder(real_digit) # Outputs mean and log-variance
z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu) # Sample from N(mu, var)
x_recon = decoder(z) # Reconstruction

# ELBO loss forces latent distribution to match N(0, I)
# After training, we CAN sample from N(0, I):
z_random = torch.randn(32) # Sample from prior
x_gen = decoder(z_random) # Output: Valid NEW digit!
```

**The sculptor analogy in action:**
- **Training**: Encoder learns that "3"s cluster around $\mu \approx [0.8, -0.3, \ldots]$ with $\sigma^2 \approx [0.1, 0.2, \ldots]$
- **Generation**: Sample $z \sim \mathcal{N}(0, I)$ → decoder interprets this as "generate a plausible digit based on the learned anatomical map"
- **Interpolation**: $z_1$ (a "3") $\to$ $z_2$ (an "8") → decoder smoothly morphs one digit into another

---

## 3 · The Math — ELBO and the Reparameterization Trick

### 3.1 · The Probabilistic Formulation

In a standard autoencoder, the encoder is deterministic: $\mathbf{z} = f_{\text{enc}}(\mathbf{x})$. In a VAE, the encoder is **probabilistic**:

$$q_\phi(z|x) = \mathcal{N}(z \mid \mu_\phi(x), \sigma^2_\phi(x) \mathbf{I})$$

The encoder outputs $\mu_\phi(x) \in \mathbb{R}^k$ and $\log \sigma^2_\phi(x) \in \mathbb{R}^k$ (we output log-variance for numerical stability). To sample a latent code:

$$z \sim q_\phi(z|x) \implies z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

This is the **reparameterization trick** (Kingma & Welling, 2013) — it makes the sampling operation differentiable by moving the randomness to $\epsilon$.

The decoder defines a likelihood: $p_\theta(x|z)$. For MNIST (continuous pixels in [0,1]), we use:

$$p_\theta(x|z) = \mathcal{N}(x \mid f_{\text{dec}}(z), \sigma_I^2 \mathbf{I})$$

In practice, $\sigma_I^2$ is fixed (often to 1), so the reconstruction loss becomes MSE.

### 3.2 · The ELBO Loss

The VAE objective is to maximize the log-likelihood $\log p_\theta(x)$ over training data. Since this is intractable (requires marginalizing over all possible $z$), we instead maximize the **Evidence Lower BOund (ELBO)**:

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$$

This decomposes into two terms:

1. **Reconstruction term** $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ — how well does the decoder reconstruct the input after sampling from the encoder's distribution?
   - For Gaussian $p_\theta(x|z)$, this becomes MSE: $-\frac{1}{2}\|\mathbf{x} - f_{\text{dec}}(z)\|^2$ (up to constants)

2. **KL regularization term** $\text{KL}(q_\phi(z|x) \| p(z))$ — how far is the encoder's distribution from the prior $p(z) = \mathcal{N}(0, \mathbf{I})$?
   - For Gaussian $q_\phi$ and Gaussian prior, this has a closed form:
   $$\text{KL}(q_\phi \| p) = \frac{1}{2} \sum_{j=1}^k \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)$$

**The sculptor analogy for ELBO:**
- **Reconstruction term**: "Can the intern (decoder) recreate the original model's pose from the compressed instructions?"
- **KL term**: "Are the compressed instructions (latent distribution) expressed in the standard anatomical vocabulary (unit Gaussian)?" — This ensures that **any** sample from $\mathcal{N}(0, I)$ can be decoded into a plausible digit.

### 3.3 · Training Loop (PyTorch)

```python
def vae_loss(x, mu, log_var, x_recon):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')

    # KL divergence (closed form for Gaussian)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_div

# Training step
optimizer.zero_grad()
mu, log_var = encoder(x) # Outputs: [batch, latent_dim]
z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu) # Reparameterization trick
x_recon = decoder(z)
loss = vae_loss(x, mu, log_var, x_recon)
loss.backward()
optimizer.step()
```

---

## 4 · How It Works — Step by Step

### Step 1: Encoder Maps Image to Latent Distribution

**Input**: MNIST digit $\mathbf{x} \in \mathbb{R}^{784}$ (28×28 flattened, normalized to [0,1])

**Encoder architecture**:
```
Linear(784 → 400) → ReLU → Linear(400 → 32) [μ branch]
                          └→ Linear(400 → 32) [log σ² branch]
```

**Output**: $\mu_\phi(\mathbf{x}) \in \mathbb{R}^{32}$, $\log \sigma^2_\phi(\mathbf{x}) \in \mathbb{R}^{32}$

**The sculptor learns**: "This digit '3' has mean proportions $\mu = [0.8, -0.3, \ldots]$ with uncertainty $\sigma^2 = [0.1, 0.2, \ldots]$"

### Step 2: Sample Latent Code via Reparameterization Trick

Instead of directly sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ (non-differentiable), we use:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

This moves the randomness to $\epsilon$, allowing gradients to flow through $\mu$ and $\sigma$.

```python
z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
```

**The sculptor's intern**: "I'll take the master's proportions $\mu$ and add natural variation $\sigma \odot \epsilon$ to make this pose slightly unique."

### Step 3: Decoder Reconstructs Image from Latent Sample

**Decoder architecture**:
```
Linear(32 → 400) → ReLU → Linear(400 → 784) → Sigmoid
```

**Output**: $\hat{\mathbf{x}} \in \mathbb{R}^{784}$, reshaped to 28×28

**The sculptor's output**: "Here's the reconstructed digit based on those proportions."

### Step 4: Compute ELBO Loss

$$\mathcal{L} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|^2}_{\text{reconstruction}} + \underbrace{\frac{1}{2} \sum_{j=1}^{32} \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)}_{\text{KL regularization}}$$

**Backpropagate through**:
1. Decoder (reconstruction quality)
2. Reparameterization (latent sampling)
3. Encoder (latent distribution parameters)

### Step 5: Generation (After Training)

**To generate a NEW digit**:
```python
z = torch.randn(1, 32) # Sample from N(0, I)
with torch.no_grad():
    x_gen = decoder(z)
```

**The sculptor's generative power**: "Give me a random set of anatomical proportions from the learned distribution → I'll create a brand new, plausible human figure."

**To interpolate between two digits**:
```python
z1 = mu1 # Latent code for digit "3"
z2 = mu2 # Latent code for digit "8"
for alpha in np.linspace(0, 1, 10):
    z_interp = alpha * z1 + (1 - alpha) * z2
    x_interp = decoder(z_interp) # Smooth morph from "3" to "8"
```

---

## 5 · Key Diagrams

### 5.1 · VAE Architecture

```
                    ┌─────────────────────────────────┐
                    │   ENCODER (Master Sculptor)     │
                    │                                  │
     x (784D)       │   Linear(784→400) → ReLU         │
     ──────────────>│      ┌→ Linear(400→32) = μ      │
     MNIST digit    │      └→ Linear(400→32) = log σ² │
                    └──────────┬───────────────────────┘
                               │
                               v
                    ┌─────────────────────────────────┐
                    │  REPARAMETERIZATION TRICK       │
                    │  z = μ + σ ⊙ ε, ε ~ N(0,I)      │
                    └──────────┬───────────────────────┘
                               │
                               v (z: 32D latent sample)
                    ┌─────────────────────────────────┐
                    │   DECODER (Sculptor's Intern)   │
                    │                                  │
                    │   Linear(32→400) → ReLU          │
                    │   Linear(400→784) → Sigmoid      │
                    └──────────┬───────────────────────┘
                               │
                               v
                           x_recon (784D)
                           Reconstructed digit
```

### 5.2 · Latent Space Visualization (2D Projection)

```
          Latent Space z ∈ R²

    ┌────────────────────────────┐
    │     ●●● ○○○                │  ● = μ (mean)
    │    ●3●3●●●○8○8○             │  ○ = samples from N(μ,σ²)
    │   ●●3●3●    ○8○8○           │
    │                             │
    │  ●●1 1●      ○○7 7○         │  Clusters: encoder maps
    │ ●●1●1●      ○7○7○○          │  similar digits to nearby μ
    │                             │
    │    ●●5●5●  ○○0○0○            │  Overlap: σ² creates spread
    │     ●5●5●○0○○0○              │  → smooth transitions
    └────────────────────────────┘

    Sample z ~ N(0,I) anywhere → decoder generates plausible digit
```

**Why this works**: KL regularization forces all $q_\phi(z|x)$ to be close to $\mathcal{N}(0, I)$. The latent space becomes **smooth** — nearby points decode to similar digits.

### 5.3 · Interpolation Between Digits

```
z₁ (digit "3")  α=0.0  α=0.25  α=0.5  α=0.75  α=1.0  z₂ (digit "8")
    ●───────────────────────────────────────────────────●
    │                                                   │
    └─────> z_interp = α·z₁ + (1-α)·z₂ ──────> decoder

Generated images along interpolation:
┌────┬────┬────┬────┬────┐
│ 3  │ 3~ │ ~8 │ ~8 │ 8  │  Smooth morphing
└────┴────┴────┴────┴────┘
```

**The sculptor's mastery**: The latent space respects anatomical plausibility. Interpolating between two poses yields intermediate poses that are themselves valid.

---

## 6 · The Hyperparameter Dial — Balancing Reconstruction and Regularization

### 6.1 · The β-VAE Trade-off

The ELBO has two competing terms:
- **Reconstruction**: Minimize $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$ → prioritize accurate reconstructions
- **KL regularization**: Minimize $\text{KL}(q_\phi \| p)$ → force latent to match prior $\mathcal{N}(0, I)$

**β-VAE** (Higgins et al., 2017) introduces a weight:

$$\mathcal{L}_{\beta\text{-VAE}} = \text{Recon} + \beta \cdot \text{KL}$$

**Typical values**:
- $\beta = 1.0$ — Standard VAE (ELBO)
- $\beta < 1.0$ (e.g., 0.5) — Prioritize reconstruction → sharper images, less smooth latent space
- $\beta > 1.0$ (e.g., 4.0) — Prioritize disentanglement → more interpretable latent dimensions (e.g., dim 5 = digit slant, dim 12 = stroke thickness)

**The sculptor's choice**:
- Low $\beta$: "Prioritize exact pose reconstruction, even if it means the anatomical map is less general."
- High $\beta$: "Ensure every latent dimension has a clear anatomical meaning (shoulder width, leg length), even if reconstructions are less precise."

### 6.2 · Latent Dimensionality $k$

**Typical values**:
- $k = 2$ — Visualization (plot entire latent space in 2D)
- $k = 32$ — MNIST (captures digit identity + style variations)
- $k = 128$ — CelebA faces (needs more dimensions for facial attributes)

**Rule of thumb**: $k$ should be large enough to capture dataset variability, small enough to enforce meaningful compression.

---

## 7 · What Can Go Wrong

1. **Posterior collapse** — Decoder ignores latent code $z$, generates same image regardless of input. KL term → 0 because encoder outputs $\mu \approx 0, \sigma \approx 1$ (prior) for all $\mathbf{x}$. Decoder learns to generate average digit without using $z$.
   - **Fix**: Increase $\beta$ (strengthen KL penalty), use warm-up schedule ($\beta$ starts low, increases over epochs), anneal learning rate

2. **Blurry reconstructions** — MSE loss penalizes sharp edges (pixel-wise squared error prefers blurring). VAE outputs are characteristically "fuzzy."
   - **Fix**: Replace MSE with perceptual loss (GAN discriminator, covered in Ch.3), or use VQ-VAE (vector-quantized latent space)

3. **Mode collapse in generation** — Decoder learns to map many $z$ values to the same output digit (e.g., all $z$ near origin → "1").
   - **Fix**: Increase latent dimensionality $k$, reduce $\beta$ (weaken KL pressure), check dataset balance (if training set has 90% "1"s, decoder will favor "1"s)

4. **Non-smooth latent space** — Interpolation between $z_1$ and $z_2$ produces garbage frames.
   - **Fix**: Increase $\beta$ (stronger regularization toward unit Gaussian), train longer (latent space smoothness emerges over epochs)

5. **Forgetting to clamp decoder output** — If decoder doesn't use Sigmoid activation, outputs can be outside [0,1], breaking image rendering.
   - **Fix**: `x_recon = torch.sigmoid(decoder(z))` or use `nn.Sigmoid()` as final layer

---

## 8 · Progress Check — What We Can Solve Now

**Unlocked capabilities:**
- **Generate NEW digits**: Sample $z \sim \mathcal{N}(0, I)$ → decode → get novel MNIST digit
- **Smooth interpolation**: Morph between any two digits by interpolating latent codes
- **Constraint progress**:
  - **#1 QUALITY**: ~75% fooling rate (up from 60% with autoencoders) — still blurry but better
  - **#2 DIVERSITY**: ✓ Covers all 10 digit classes (KL regularization prevents mode collapse)
  - **#5 LATENT INTERPRETABILITY**: ✓ Smooth latent space, meaningful arithmetic (e.g., $z_{\text{"3"}} + (z_{\text{"8"}} - z_{\text{"3"}}) \approx z_{\text{"8"}}$)

**Still can't solve:**
- **Sharp, photorealistic generation** (fooling rate <90%) — MSE reconstruction loss produces blur
- **Controllability** (Constraint #3) — Can't easily say "generate digit 7" without conditional architecture
- **Training instability** — VAE training is stable (no adversarial game), but outputs are never crisp

**Real-world status**: We can generate diverse, novel digits and explore latent space smoothly. But synthetic samples are distinguishable from real MNIST by visual inspection (blur artifacts). Medical imaging use case still blocked — radiologists reject blurry X-rays.

**Next up:** Ch.3 GANs give us **adversarial training** — generator learns to fool a discriminator, unlocking photorealistic quality (>90% fooling rate) without MSE blur.

---

## 9 · Bridge to the Next Chapter

This chapter established **probabilistic latent spaces** that enable generation and smooth interpolation — the sculptor's intern can create new anatomically-plausible figures. But the ELBO's MSE reconstruction term produces blurry outputs. **Ch.3 (GANs)** introduces **adversarial training**: a discriminator (art detective) that learns to distinguish real from generated images, forcing the generator (forger) to produce sharp, photorealistic samples. No MSE — only the binary question: "Can you fool the detector?" This adversarial game unlocks the final 15% in fooling rate (75% → >90%) and solves the medical imaging use case.

---

## Cross-Chapter Connections

### VAEs Reappear In:

1. **[04-multimodal-ai/ch06_latent_diffusion](../../../04-multimodal-ai/ch06_latent_diffusion/latent-diffusion.md)** — Stable Diffusion uses a VAE encoder to compress 512×512 images → 64×64 latent, diffuses there (16× cheaper), then decodes back to pixels. The VAE is pretrained on millions of images.

2. **Conditional VAEs (CVAE)** — Extend encoder input: $q_\phi(z|x, c)$ where $c$ is a class label. Enables "generate digit 7 specifically" by conditioning decoder: $p_\theta(x|z, c)$.

3. **Disentangled representations** — β-VAE with $\beta > 1$ learns latent dimensions that capture independent factors of variation (e.g., digit slant, stroke thickness). Used in representation learning research.

4. **VAE-GAN hybrids** — Combine VAE encoder-decoder with GAN discriminator. Replace MSE reconstruction loss with adversarial loss → sharp reconstructions + smooth latent space.

### Why This Matters for the Curriculum:

- **Latent diffusion is the architecture behind Stable Diffusion, DALL-E 2, Midjourney** — all rely on VAE compression
- **Conditional generation** unlocks controllable synthesis (Constraint #3 in Ch.3)
- **Probabilistic thinking** prepares you for Bayesian deep learning, variational inference, normalizing flows
