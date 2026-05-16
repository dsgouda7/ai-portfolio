# Multimodal AI — Interview Primer

← Back to learning chapter: [Multimodal AI](../04-multimodal-ai/README.md)

> **Purpose**: Get a senior AI/ML engineer from zero to confident on multimodal AI interview questions in 2–4 hours. Covers vision transformers, contrastive learning, diffusion models, and production text-to-image/video systems. Every answer is grounded in **VisualForge Studio**, a production text-to-image pipeline serving 10k generations/day.

---

> **How to use the junior/senior answer comparisons** — Each question below includes a junior-level answer and a senior-level answer. Junior answers are technically correct but surface-level. Senior answers demonstrate production experience, failure awareness, and trade-off reasoning. Hiring managers at FAANG and growth-stage AI companies distinguish these instantly. Study the DIFFERENCE between the two, not just the senior answer.

## 1 · Concept Map — The 10 Questions That Matter

| # | Cluster | What the interviewer is testing |
|---|---------|----------------------------------|
| 1 | **Multimodal Foundations & Patch Tokenization** | Can you explain how images become token sequences for a transformer? |
| 2 | **Vision Transformers (ViT)** | Do you know CLS token pooling, patch size tradeoffs, and why ViT beats CNNs at scale? |
| 3 | **CLIP & InfoNCE Loss** | Can you derive the contrastive objective? Know batch size's role in negative mining? |
| 4 | **Forward/Reverse Diffusion** | Can you describe the noise schedule and the denoising score matching objective? |
| 5 | **Schedulers (DDPM/DDIM/DPM-Solver)** | Do you know the speed/quality tradeoff? Can you explain DDIM's deterministic sampling? |
| 6 | **Latent Diffusion & VAE** | Do you know why Stable Diffusion runs in latent space, not pixel space? |
| 7 | **Guidance & CFG** | Can you derive classifier-free guidance and explain what happens above CFG≈7? |
| 8 | **U-Net Architecture** | Do you know skip connections, cross-attention injection, and their roles in conditioning? |
| 9 | **Evaluation Metrics (FID/CLIP Score)** | Can you explain both metrics, their correlation, and what each misses? |
| 10 | **Production Latency & LoRA Fine-Tuning** | Do you know batch size, step count, and precision tradeoffs for real-time generation? |

---

## 2 · Section-by-Section Deep Dives

### Ch.1 — Multimodal Foundations

#### What They're Testing
Do you understand the fundamental representation gap between modalities and why naive concatenation fails?

#### Must Know
- What shape is a colour image tensor in PyTorch? `(N, C, H, W)` — batch, channels, height, width
- What is the modality gap and why does it require a solution beyond simple concatenation?
- What normalisation does ImageNet-pretrained models expect, and why does it matter?

### Likely Asked
- "How would you convert a 5-second 16 kHz audio clip to a tensor ready for a ViT-based model?"
 → resample → STFT → mel filterbank → log scale → treat as 2-D image
- "Why does the same pixel value mean something different in a diffusion model vs a classification model?"
 → different normalisation conventions (`[-1,1]` vs `[0,1]` vs ImageNet stats)

#### The Junior Answer vs Senior Answer

**Q: "How do you handle different input modalities in a unified model?"**
**Junior**: "You normalize them and concatenate."
*Why this signals junior:* Ignores the modality gap — different modalities have different statistical properties and semantic densities.
**Senior**: "Each modality needs its own encoder (ViT for images, BERT for text) that projects to a shared embedding space. The key is contrastive alignment — CLIP uses InfoNCE loss to make matching (text, image) pairs close in cosine similarity while pushing unrelated pairs apart. For VisualForge Studio, we use a frozen CLIP text encoder to condition image generation."
*Why this signals senior:* Names the architecture pattern (separate encoders + shared space), cites a specific loss function, grounds in a production system.

#### The Key Tradeoffs

| Approach | When It Wins | When It Loses | Decision Criterion |
|----------|--------------|---------------|-------------------|
| ImageNet normalization | Standard RGB images, pretrained CNNs/ViTs | Medical imaging, satellite, infrared | If domain matches ImageNet distribution, use it; otherwise compute dataset-specific stats |
| Float16 training | 2× memory savings, 1.5–2× speed | Gradient underflow in attention, loss instability | Use mixed precision (fp16 forward, fp32 accumulation) for attention layers |
| HWC vs CHW layout | NumPy/PIL interoperability | PyTorch/GPU efficiency | Convert to CHW before feeding to model; use einops for clarity |

#### Failure Mode Gotchas

**Trap:** Forgetting to convert channel order (HWC → CHW) when going from PIL/NumPy to PyTorch causes silent shape mismatches.
**How to detect:** Model forward pass fails with "expected 3 channels, got 224" or similar dimension error.
**Fix:** Always `torchvision.transforms.ToTensor()` which handles HWC→CHW + uint8→float32 + [0,255]→[0,1].

**Warning — Trap:** Using ImageNet normalization for medical images or satellite imagery — domain mismatch will hurt transfer learning.
**How to detect:** Validation accuracy plateau well below expected.
**Fix:** Compute dataset-specific mean/std or use domain-pretrained models (e.g., MedCLIP for medical).

#### The Production Angle

**In VisualForge Studio (10k generations/day):**
- All inputs are normalized to CLIP's expected format (`mean=[0.48145466, 0.4578275, 0.40821073]`, `std=[0.26862954, 0.26130258, 0.27577711]`) before encoding
- Float16 inference throughout except for VAE decoder final layer (prevents color banding artifacts)
- Batch size = 4 for optimal A100 utilization (higher batch doesn't improve throughput due to memory bandwidth limits)

---

### Ch.2 — Vision Transformers

#### What They're Testing
Do you understand why ViT needs massive data to beat CNNs, and can you explain the patch tokenization + positional embedding mechanics?

#### Must Know
- How does ViT convert an image to a sequence? (split into $P \times P$ patches, flatten, linear project)
- Why does ViT struggle at low data regimes compared to CNN?
- What does the CLS token output represent?

### Likely Asked
- "A ViT-B/16 receives a 224×224 image. How many tokens does the transformer encoder process?"
 → $196 \text{ patches} + 1 \text{ CLS} = 197$
- "How would you adapt ViT to process a 512×512 image without retraining positional embeddings?"
 → Bicubic interpolation of the learned position embeddings from 14×14 to 32×32 grid
- "CLIP uses ViT-L/14 as its image encoder. Why L/14 and not B/16?"
 → Smaller patch size (14 vs 16) → more patches → finer detail → better zero-shot performance

#### The Junior Answer vs Senior Answer

**Q: "Why does ViT struggle on small datasets compared to CNNs?"**
**Junior**: "It doesn't have convolutional layers."
*Why this signals junior:* Surface-level observation without understanding the root cause.
**Senior**: "ViT has minimal spatial inductive bias — no local connectivity, no translation equivariance beyond tied patch embeddings. CNNs bake in 2D locality through convolution kernels. On ImageNet-1k, ViT-B underperforms ResNet-50 when both are trained from scratch. ViT needs either massive scale (JFT-300M) or compensating techniques (DeiT's distillation, MAE's masked pretraining) to compete in data-constrained regimes. In VisualForge, we use pretrained ViT-L/14 from CLIP — 400M image-text pairs provide the necessary scale."
*Why this signals senior:* Names the specific bias (locality), quantifies the data requirement, cites compensating techniques, grounds in production context.

#### The Key Tradeoffs

| Approach | When It Wins | When It Loses | Decision Criterion |
|----------|--------------|---------------|-------------------|
| ViT | Large-scale pretraining (>100M samples), global reasoning tasks | Small datasets (<1M samples), dense prediction without pretraining | If you have pretrained ViT weights, use it; otherwise use CNN or hybrid |
| Swin Transformer | Dense prediction (detection, segmentation), efficiency-critical | Global semantic tasks (CLIP, classification), simplicity | If output needs spatial resolution, use Swin; for embeddings/classification, plain ViT |
| Patch size 14 vs 16 | Fine detail critical (CLIP, high-res), 1.3× more tokens | Speed-critical, memory-constrained | Smaller patch = better accuracy but quadratic cost; 16 is the standard, 14 for high-end |

#### Failure Mode Gotchas

**Warning — Trap:** Forgetting that attention in ViT is over patches, not pixels — patch count $N = (H/P)^2$, not $H \times W$.
**Impact:** Memory scales as $O(N^2)$, not $O(HW)$; doubling resolution = 4× memory.
**Fix:** For high-res images, use hierarchical architectures (Swin) or local attention windows.

**Gotcha:** "ViT always beats ResNet" — only with sufficient scale and data; DeiT and MAE address the constrained-data regime.
**Interview distinction:** A senior candidate knows ViT-B on ImageNet-1k alone underperforms ResNet-50; the win comes from pretraining scale.

#### The Production Angle

**In VisualForge Studio:**
- CLIP ViT-L/14 as frozen text-to-embedding encoder (400M samples pretrained)
- 224×224 input → 256 patch tokens (16×16 grid) + 1 CLS → 257 total
- Positional embeddings fixed (no interpolation) — all inputs resized to 224×224
- Attention is full $O(N^2)$ but batched efficiently on A100 (faster than hierarchical for batch=4)

---

### Ch.3 — CLIP

#### What They're Testing
Can you derive the InfoNCE loss, explain why batch size matters, and describe CLIP's role in conditional generation?

#### Must Know
- What are the two encoders in CLIP and what do they output?
- What is the InfoNCE loss — what are the positives and negatives?
- How does zero-shot classification work with CLIP?

### Likely Asked
- "Why does CLIP use large batch sizes?"
 → More negatives per sample → harder negatives → sharper representations
- "How is CLIP's text encoder used in Stable Diffusion?"
 → Frozen CLIP text encoder converts prompt to 77 × 768 token embeddings → fed as cross-attention keys/values inside the U-Net denoiser at every layer
- "What does CLIP's embedding space geometry look like?"
 → All embeddings are on the unit hypersphere (`L2 norm = 1`); matching pairs are close (high cosine sim); unrelated pairs are near-orthogonal

#### The Junior Answer vs Senior Answer

**Q: "Why does CLIP use such large batch sizes during training?"**
**Junior**: "To train faster with more parallelism."
*Why this signals junior:* Confuses batch size with GPU utilization — misses the contrastive learning mechanics.
**Senior**: "Large batch = more negatives per sample. InfoNCE loss treats the N-1 other images in a batch of N as hard negatives for each text. Batch size 32,768 means 32,767 negatives per positive pair — this forces the model to learn fine-grained distinctions. Small batch = easy negatives = poor representations. CLIP's batch size is a hyperparameter for negative mining difficulty, not just hardware efficiency."
*Why this signals senior:* Explains the contrastive loss mechanics, quantifies the effect, distinguishes batch size roles.

**Q: "How is CLIP used in Stable Diffusion?"**
**Junior**: "It encodes the text prompt."
*Why this signals junior:* Technically correct but incomplete — doesn't explain the conditioning mechanism.
**Senior**: "CLIP's frozen text encoder converts the prompt to 77 token embeddings (each 768-dim for SD 1.5). These embeddings become keys and values in cross-attention layers throughout the U-Net denoiser — at every resolution stage. Query comes from the noisy latent features. This is how text steers the denoising trajectory. In VisualForge Studio, we also use CLIP for prompt embeddings and for computing CLIP Score during evaluation."
*Why this signals senior:* Describes the full data flow (text → embeddings → cross-attention K/V), names the architecture component, grounds in production.

#### The Key Tradeoffs

| Approach | When It Wins | When It Loses | Decision Criterion |
|----------|--------------|---------------|-------------------|
| CLIP | Zero-shot classification, text-image retrieval, coarse semantic alignment | Fine-grained visual understanding, spatial reasoning | If task is "does this image match this text?", use CLIP; for "where is the object?", fine-tune or use detection model |
| SigLIP | Smaller batch training (<1k), simpler loss, better at smaller scales | Marginal quality at very large scale | If batch size <2k, consider SigLIP; at 32k+, CLIP's softmax normalization wins |
| CLIP embeddings for retrieval | Cross-modal search (text→image, image→text), zero-shot | Exact duplicate detection, pixel-level similarity | Use CLIP for semantic retrieval; use perceptual hashes for exact duplicates |

#### Failure Mode Gotchas

**Warning — Trap:** Confusing CLIP (contrastive training, no generation) with DALL-E (generative, uses CLIP as a component).
**Interview impact:** Shows you don't understand the architecture landscape.
**Clarification:** CLIP = embedding model; DALL-E = generative model conditioned on CLIP embeddings.

**Gotcha:** "CLIP embeddings can be compared with raw dot product" — CLIP embeddings are L2-normalized; use cosine similarity.
**Why it matters:** Unnormalized dot product gives wrong rankings if vectors aren't on the unit sphere.
**Fix:** `cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2)` or `dot_product(normalize(emb1), normalize(emb2))`.

#### The Production Angle

**In VisualForge Studio (10k generations/day):**
- CLIP ViT-L/14 text encoder frozen, loaded once at server startup (1.2 GB VRAM)
- Prompt → 77 CLIP tokens (max length), padded with EOS tokens if shorter
- Cross-attention K/V cached per prompt (reused across 50 denoising steps)
- CLIP Score computed post-generation for quality monitoring: `cosine_sim(CLIP_image(output), CLIP_text(prompt))`
- Threshold: CLIP Score < 0.25 triggers alert (likely prompt-image mismatch or generation failure)

---

### Ch.4 — Diffusion Models

#### What They're Testing
Can you derive the forward process, explain the U-Net training objective, and distinguish training from inference?

#### Must Know
- What does the U-Net predict in DDPM — the image or the noise?
- Write the closed-form forward process equation $q(x_t | x_0)$
- Why is the DDPM loss just MSE on noise prediction?

### Likely Asked
- "Why does DDPM need $T = 1000$ steps? Why not just use $T = 10$?"
 → Fewer steps → each $\beta_t$ must be larger → the Gaussian approximation of the reverse step breaks down → poor generation quality. Fast samplers (DDIM) solve inference speed without retraining.
- "What is the signal-to-noise ratio at step $t$, and what does $\bar{\alpha}_t$ represent?"
 → $\text{SNR}(t) = \bar{\alpha}_t / (1 - \bar{\alpha}_t)$; $\bar{\alpha}_t$ = fraction of original signal remaining
- "Why are diffusion models more stable than GANs?"
 → No adversarial game; the loss is a simple MSE; no generator/discriminator equilibrium required

#### The Junior Answer vs Senior Answer

**Q: "Walk me through the diffusion training objective."**
**Junior**: "The model learns to denoise images."
*Why this signals junior:* Vague — doesn't specify what the model predicts or the loss function.
**Senior**: "Training: Sample a timestep $t \sim \text{Uniform}(1, T)$, add Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ to the clean image $x_0$ according to $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$. The U-Net $\epsilon_\theta(x_t, t)$ predicts the noise $\epsilon$. Loss is simple MSE: $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$. This is denoising score matching. Inference reverses this: start from pure noise, iteratively subtract predicted noise over T steps to recover $x_0$."
*Why this signals senior:* Writes the forward process equation, names the loss, distinguishes training (single step) from inference (T steps).

#### The Key Tradeoffs

| Approach | When It Wins | When It Loses | Decision Criterion |
|----------|--------------|---------------|-------------------|
| Predict noise $\epsilon$ | Standard DDPM, numerically stable | - | Default choice; v-prediction is emerging alternative |
| Predict $x_0$ directly | - | Less stable gradients, worse sample quality | Avoid unless using v-parameterization |
| DDPM (T=1000 steps) | Highest quality, full stochastic sampling | Slow inference (50s on GPU) | Use for offline generation or quality-critical tasks |
| DDIM (T=50 steps) | 20× faster, deterministic, same quality | Slightly less diversity | Production default for T2I |

#### Failure Mode Gotchas

**Warning — Trap:** Confusing $\beta_t$ (noise variance) with $\alpha_t = 1 - \beta_t$ (signal retention fraction).
**Impact:** Wrong schedule = wrong signal-to-noise ratio = poor generation.
**Mnemonic:** $\alpha_t$ = fraction of original signal remaining; $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ = cumulative signal.

**Gotcha:** "Diffusion generates in a single forward pass" — NO. Inference requires T repeated U-Net calls (typically 20–50).
**Why it matters:** Latency = T × forward_time; this is why fast schedulers (DPM-Solver, LCM) are critical for production.

#### The Production Angle

**In VisualForge Studio:**
- Training: DDPM objective with T=1000, cosine noise schedule, batch=256 on 8× A100s
- Inference: DDIM with T=25 steps (default), DPM-Solver++ for T=15 (premium tier)
- Latency: 25 steps × 40ms/step = 1.0s end-to-end (512×512, A100, batch=1)
- Quality monitoring: FID computed daily on 5k samples vs. LAION-Aesthetics-6.5+ subset

---

### Ch.5 — Schedulers

#### What They're Testing
Do you understand why DDPM needs 1000 steps, how DDIM achieves deterministic sampling, and the speed/quality tradeoff?

#### Must Know
- Why DDPM needs 1 000 steps at inference: each step is a Gaussian approximation that only holds for small β — larger strides violate the Markov assumption
- DDIM key insight: rewrite as ODE, enabling deterministic sub-sequence sampling
- The trade-off axes: steps ↓ speed ↑ quality ↓ diversity (generally true)

### Likely Asked
- *"What scheduler does Stable Diffusion use by default?"* — PNDM (pseudo-numerical) or DDIM; SDXL defaults to EulerDiscreteScheduler
- *"How would you halve inference time without quality loss?"* — Switch from 50-step DDIM to 15-step DPM-Solver++
- *"What is the relationship between DDIM and DDPM?"* — DDIM is a non-Markovian generalization that reduces to DDPM when σ=original noise level; at σ=0 it becomes fully deterministic

#### The Junior Answer vs Senior Answer

**Q: "How does DDIM achieve faster sampling than DDPM?"**
**Junior**: "It uses fewer steps."
*Why this signals junior:* Describes the outcome, not the mechanism.
**Senior**: "DDIM reinterprets the reverse process as an ODE instead of an SDE — it picks a non-Markovian trajectory that allows skipping timesteps without quality loss. Where DDPM samples every step from $t=1000$ to $t=0$, DDIM can sample a subsequence like [1000, 950, 900, ..., 50, 0] (50 steps) by solving the ODE deterministically. At $\sigma=0$ it's fully deterministic; at $\sigma=\text{DDPM}$ it reduces to stochastic DDPM. In VisualForge, we default to 25-step DDIM for 1.0s latency."
*Why this signals senior:* Explains the ODE vs SDE distinction, describes the subsequence sampling, quantifies production latency.

#### The Key Tradeoffs

| Scheduler | Steps | Latency (512×512, A100) | Quality vs DDPM | Use Case |
|-----------|-------|------------------------|----------------|----------|
| DDPM | 1000 | ~50s | Baseline (highest) | Offline, quality-critical |
| DDIM | 50 | ~2s | ~95% | Standard production |
| DDIM | 25 | ~1s | ~90% | Default VisualForge |
| DPM-Solver++ | 15 | ~0.6s | ~92% | Premium tier, low-latency |
| LCM (1-shot) | 1–4 | ~0.15s | ~70% | Real-time preview |
| SDXL-Turbo | 1–4 | ~0.2s | ~75% | Real-time, sacrifices diversity |

**Decision criterion:** If latency budget >2s, use 50-step DDIM. If 1–2s, use 25-step. If <1s, use DPM-Solver++ or LCM. If real-time (<200ms), use distilled 1-shot models.

#### Failure Mode Gotchas

**Warning — Trap:** Confusing the **training** noise schedule (always 1000 steps) with the **inference** step count (scheduler-specific).
**Clarification:** Training defines $q(x_t|x_0)$ over 1000 steps. Inference schedulers can use any subsequence — 50, 25, or even 1 step with distillation.
**Why it matters:** Candidates who say "diffusion always needs 1000 steps" fail this question.

**Gotcha:** "LCM 1-step images match 50-step DDIM quality" — NO.
**Reality:** 1–4 step models sacrifice fine detail and diversity; they're for real-time preview, not final output.
**In VisualForge:** LCM is for interactive preview; final generation uses 25-step DDIM.

#### The Production Angle

**In VisualForge Studio:**
- Default: 25-step DDIM (1.0s, batch=1)
- Premium tier: 15-step DPM-Solver++ (0.6s)
- Preview mode: 4-step LCM-LoRA (0.15s)
- Scheduler switching: determined by user tier and latency SLA
- Quality gate: FID monitored per scheduler; alert if FID degrades >5 points vs baseline

---

### Ch.6 — Latent Diffusion

#### What They're Testing
Why does Stable Diffusion run in latent space, not pixel space? Can you explain the VAE compression ratio and cross-attention conditioning?

#### Must Know
- The three components of Stable Diffusion: **VAE** (compress/decompress), **U-Net** (diffuse in latent), **CLIP** (condition on text)
- Why latent space: ~48× cheaper diffusion without meaningful quality loss
- Cross-attention mechanism for text conditioning: Q from image features, K/V from text tokens

### Likely Asked
- *"What is the latent scaling factor and why is it needed?"* — 0.18215; rescales VAE output to unit variance to match DDPM's N(0,I) prior
- *"How does SDXL improve on SD 1.5?"* — 2× larger latent spatial resolution (128×128), 3× more U-Net params, two CLIP encoders concatenated, trained on aspect-ratio bucketing
- *"What is a Diffusion Transformer (DiT)?"* — Replace U-Net with a pure Transformer; patches of the latent are tokens; SD 3.5 and Flux use this architecture

#### The Junior Answer vs Senior Answer

**Q: "Why does Stable Diffusion run diffusion in latent space instead of pixel space?"**
**Junior**: "It's faster."
*Why this signals junior:* Correct but unquantified — doesn't explain the compression ratio or computational savings.
**Senior**: "A 512×512 RGB image is 786k pixels. The VAE encoder compresses it to 64×64×4 latent (16k values) — 8× spatial downsampling per dimension, 49× fewer values. Diffusion cost scales as $O(H^2 W^2)$ for attention, so latent diffusion is ~64× cheaper per step than pixel-space DDPM. Quality is preserved because the VAE is trained to reconstruct faithfully. In VisualForge, this makes 25-step generation feasible at 1.0s; pixel-space would be ~60s."
*Why this signals senior:* Quantifies compression (8× spatial, 64× compute), explains the quadratic attention cost, grounds in production.

#### The Key Tradeoffs

| Approach | When It Wins | When It Loses | Decision Criterion |
|----------|--------------|---------------|-------------------|
| Latent diffusion (SD) | Standard resolution (512×512), speed-critical | Very high-res (2048×2048+), pixel-perfect tasks | Use latent diffusion for <1024×1024; for higher res, consider cascaded diffusion or DiT |
| Pixel-space diffusion | Ultra-high-res, medical imaging (no compression loss) | Speed, memory | If quality loss from VAE compression is unacceptable, use pixel-space; otherwise latent wins |
| VAE scaling factor 0.18215 | Required for unit variance matching | - | Always apply: `latent = (vae_encode(img) * 0.18215)` |

#### Failure Mode Gotchas

**Warning — Trap:** "The VAE is trained as part of SD" — NO.
**Reality:** VAE is pretrained separately and frozen during SD training. Only the U-Net denoiser is trained.
**Why it matters:** Shows you understand the training pipeline separation.

**Gotcha:** Forgetting the VAE scaling factor (0.18215).
**Impact:** Without scaling, latent variance doesn't match the $\mathcal{N}(0, I)$ prior DDPM expects → poor generation.
**In VisualForge:** Scaling is baked into the model config; forgetting it during custom fine-tuning is a common bug.

#### The Production Angle

**In VisualForge Studio:**
- VAE: KL-f8 (8× spatial downsampling, 4-channel latent)
- Compression: 512×512×3 RGB → 64×64×4 latent = 49× fewer values
- VAE encode: 20ms (amortized across batch)
- VAE decode: 50ms (decoder is larger than encoder)
- Total latency breakdown: encode 20ms + 25 steps × 40ms/step + decode 50ms = 1.07s
- Memory: VAE decoder kept in VRAM (700 MB), encoder loaded on-demand for img2img

---

### Ch.7 — Guidance & Conditioning

#### What They're Testing
Can you derive the CFG equation, explain the two model calls, and describe what happens when guidance scale exceeds 12?

#### Must Know
- Write the CFG equation. What are the two model calls?
- What does guidance scale $w = 1$ vs $w = 7.5$ vs $w = 15$ produce?
- How is condition dropout used during CFG training?

### Likely Asked
- "How does a negative prompt work mechanically?"
 → Replace the unconditioned embedding with the negative prompt embedding; the CFG equation then steers the image away from the negative and toward the positive
- "Why is CFG inference twice as slow as unconditioned inference?"
 → Two separate U-Net forward passes per denoising step — one conditioned, one not
- "What is attention control / prompt-to-prompt?"
 → Manipulate the cross-attention maps directly (instead of changing the embedding) to achieve localised edits: change "a photo of a cat" to "a photo of a dog" while keeping the composition

#### The Junior Answer vs Senior Answer

**Q: "What is classifier-free guidance and why do we need it?"**
**Junior**: "It makes the model follow the prompt better."
*Why this signals junior:* Outcome without mechanism — doesn't explain the two predictions or the blending equation.
**Senior**: "CFG trains the U-Net both conditionally (with prompt) and unconditionally (prompt dropped). At inference, we make two predictions per step: $\epsilon_\text{cond}$ with the prompt, $\epsilon_\text{uncond}$ without. The guided prediction is $\epsilon_\text{guided} = \epsilon_\text{uncond} + w (\epsilon_\text{cond} - \epsilon_\text{uncond})$. When $w > 1$, we amplify the direction toward the conditional score. This increases prompt adherence at the cost of diversity. In VisualForge, we default to $w=7.5$ for quality; $w=12$ for strict prompt adherence."
*Why this signals senior:* Writes the equation, explains the training procedure (condition dropout), names the tradeoff, grounds in production.

#### The Key Tradeoffs

| Guidance scale $w$ | Effect | Quality | Diversity | Use Case |
|--------------------|--------|---------|-----------|----------|
| 1.0 | Unconditional (prompt ignored) | Low | Highest | Style exploration |
| 3.0–5.0 | Weak guidance | Medium | High | Artistic, abstract |
| 7.0–8.0 | Standard | High | Medium | Production default |
| 10.0–12.0 | Strong guidance | Very high | Low | Strict prompt adherence |
| 15.0+ | Oversaturated | Artifacts appear | Very low | Avoid — neon colors, oversharpening |

**Decision criterion:** If user specifies "exactly as described", use $w=10$. If "creative interpretation", use $w=5$. Default: $w=7.5$.

#### Failure Mode Gotchas

**Warning — Trap:** Confusing guidance scale with classifier temperature — they are different mechanisms.
**Clarification:** Classifier guidance uses a pretrained classifier to steer (requires external model); CFG uses the same U-Net with/without conditioning (no extra model).
**Interview distinction:** Classifier guidance is mostly deprecated; CFG is the standard.

**Gotcha:** "Negative prompts block content" — NO, they steer.
**Mechanism:** Negative prompt replaces $\epsilon_\text{uncond}$ with $\epsilon_\text{negative}$ in the CFG equation, steering *away* from the negative concept.
**In VisualForge:** Common negatives: "blurry, low quality, watermark, text" — steers toward high-quality, text-free outputs.

#### The Production Angle

**In VisualForge Studio:**
- CFG scale: user-selectable 1.0–15.0, default 7.5
- Two U-Net forward passes per step: 1 conditional + 1 unconditional (or negative)
- Latency penalty: 2× forward time per step (50ms → 100ms at batch=1)
- Optimization: batch conditional and unconditional together (effective batch=2) to amortize memory loads
- Quality monitoring: CFG scale >12 triggers automatic saturation check (color histogram analysis)

---

### Ch.8 — Text-to-Image

#### What They're Testing
Do you understand img2img, ControlNet architecture, and LoRA fine-tuning mechanics?

#### Must Know
- img2img = add partial noise then denoise; strength parameter controls how much of the original is preserved
- Negative prompts extend CFG: replace unconditional baseline with negatively-conditioned baseline
- ControlNet architecture: frozen original encoder + trainable cloned encoder whose residuals are injected as skip connection additions

### Likely Asked
- *"How would you make SD generate images in a specific person's style?"* — LoRA fine-tune on ~15 images (~15 min on a consumer GPU)
- *"What is the difference between ControlNet and img2img?"* — ControlNet injects a spatial map (edges, depth) as structural guidance at every attention layer; img2img starts from a partially-noised version of an image
- *"Why do people use negative prompts like 'lowres, bad anatomy'?"* — These are common LAION dataset artifacts; subtracting their embeddings pushes the output away from low-quality image cluster in latent space

#### The Junior Answer vs Senior Answer

**Q: "How would you fine-tune Stable Diffusion on a specific art style with only 15 images?"**
**Junior**: "Fine-tune the whole model on those images."
*Why this signals junior:* Ignores overfitting risk and computational cost — full fine-tuning on 15 samples will overfit catastrophically.
**Senior**: "Use LoRA — low-rank adaptation of the U-Net's attention weight matrices. Freeze the base model, train only low-rank residuals (rank=8 typical) on the 15 images for ~1000 steps. This adds only ~3 MB of parameters vs 2 GB for the full model. LoRA prevents overfitting through the rank bottleneck while capturing style effectively. In VisualForge, we fine-tune custom LoRAs per client in ~15 minutes on a single A100."
*Why this signals senior:* Names the technique, explains the rank bottleneck mechanism, quantifies parameters and training time, grounds in production.

#### The Key Tradeoffs

| Approach | When It Wins | When It Loses | Decision Criterion |
|----------|--------------|---------------|-------------------|
| LoRA fine-tuning | Few samples (10–100), style/composition adaptation | Needs to change base model capabilities | If ≤100 samples, use LoRA; if >10k samples and need new concepts, consider full fine-tuning |
| Textual inversion | Single concept (e.g., product), <10 images | Composition/style changes | If object appearance only, use textual inversion; for style, use LoRA |
| DreamBooth | Specific subject identity (person, pet, product) | Generic style | If "this exact person", use DreamBooth; for artistic style, use LoRA |
| ControlNet | Spatial conditioning (pose, depth, edges) | No spatial reference available | If you have a reference structure (sketch, pose), use ControlNet; for text-only, use base model |

#### Failure Mode Gotchas

**Warning — Trap:** Confusing **classifier guidance** (Ch.5, needs pretrained classifier) with **classifier-free guidance** (Ch.5) and with the **negative prompt extension** of CFG.
**Clarification:** All use the word "guidance" but are different mechanisms. Classifier guidance is deprecated; CFG is standard; negative prompts extend CFG.

**Gotcha:** "ControlNet is just img2img with extra conditioning."
**Difference:** img2img starts from a partially noised image (strength parameter); ControlNet starts from pure noise but injects spatial guidance at every U-Net layer via cross-attention and skip connections. ControlNet gives structural control without pixel-level initialization.

#### The Production Angle

**In VisualForge Studio (10k generations/day):**
- Base model: Stable Diffusion 1.5 (2 GB)
- Client-specific LoRAs: ~50 active, 3 MB each, loaded on-demand (150 MB total)
- LoRA switching: <50ms overhead (swap attention weight matrices)
- ControlNet: Canny edge model loaded for "sketch-to-image" requests (additional 1.5 GB VRAM)
- Fine-tuning pipeline: 15 images → LoRA rank=8 → 1000 steps → 15 min on A100 → automatic quality check (FID on held-out set)

---

### Ch.9 — Text-to-Video

#### What They're Testing
Do you understand temporal consistency challenges, temporal attention mechanics, and AnimateDiff's freeze-spatial-train-temporal strategy?

#### Must Know
- Why video adds temporal consistency as a fundamental new challenge (image-by-image generates flicker)
- Temporal attention: attend across frames at same spatial position
- AnimateDiff design: freeze spatial SD layers, train only temporal modules on video data

### Likely Asked
- *"How would you generate a 30-fps video cheaply?"* — Generate 8-frame keyframes with a T2V model, interpolate with a frame interpolation model (RIFE, EMA-VFI)
- *"What is a spacetime patch in Sora/DiT?"* — A 3D patch $(t_p, h_p, w_p)$ treated as a single transformer token; enables arbitrary-length, arbitrary-resolution video generation
- *"What's the difference between animating a static image vs. text-to-video?"* — Animating (img2video, Stable Video Diffusion) starts from a real image latent; T2V starts from pure noise; both use temporal attention

#### The Junior Answer vs Senior Answer

**Q: "Why can't you just generate video frame-by-frame with a T2I model?"**
**Junior**: "It would be slow."
*Why this signals junior:* Misses the main problem — temporal consistency and flicker.
**Senior**: "The core issue is temporal inconsistency. Each frame is generated independently — the DDPM noise process is i.i.d. per frame. This causes flicker, object drift, and style shifts across frames. The solution is temporal attention: each pixel/patch attends not just spatially within its frame but also to the same spatial position in neighboring frames. AnimateDiff adds temporal attention modules to a frozen spatial SD U-Net, training only the temporal layers on video data. This enforces frame-to-frame coherence while preserving spatial generation quality."
*Why this signals senior:* Identifies the root cause (i.i.d. noise), names the solution (temporal attention), describes a specific architecture (AnimateDiff), explains the freeze-spatial-train-temporal strategy.

#### The Key Tradeoffs

| Approach | When It Wins | When It Loses | Decision Criterion |
|----------|--------------|---------------|-------------------|
| Frame-by-frame T2I | Very high per-frame quality | Flicker, no motion coherence | Avoid for video; use only for storyboards |
| Temporal attention (AnimateDiff) | Smooth motion, coherent style | Slightly lower per-frame detail than T2I | Default for T2V generation |
| Frame interpolation (RIFE) | Cheap upsampling (8 fps → 24 fps) | Doesn't fix motion errors, only smooths | Use after T2V to increase frame rate cheaply |
| DiT spacetime patches (Sora) | Arbitrary length/resolution, best quality | 10× compute cost vs AnimateDiff | Use for flagship demos, not production at scale |

#### Failure Mode Gotchas

**Warning — Trap:** "Temporal consistency is free from DDPM" — NO.
**Reality:** The Markovian noise process is independent per frame; correlation must be learned via temporal attention.
**Interview distinction:** A senior candidate knows that naive frame-by-frame generation produces flicker.

**Gotcha:** Confusing AnimateDiff (spatial frozen, temporal trained) with full T2V training (both trained).
**Why AnimateDiff wins:** Leverages pretrained SD spatial quality, only learns temporal coherence — much cheaper and faster to train.

#### The Production Angle

**In VisualForge Studio (video generation module):**
- Base: AnimateDiff on SD 1.5, temporal attention at 3 U-Net layers
- Generation: 16 frames at 8 fps (2-second clips), 512×512 resolution
- Latency: 16 frames × 25 DDIM steps × 50ms/step = 20s per clip (A100, batch=1)
- Post-processing: RIFE interpolation to 24 fps (16 → 48 frames, adds 2s)
- Quality monitoring: Per-frame FID + temporal coherence metric (LPIPS distance between adjacent frames, target <0.15)

---

### Ch.10 — Multimodal LLMs

#### What They're Testing
Do you understand the vision-encoder → alignment-layer → LLM architecture, and can you compare LLaVA vs BLIP-2 token compression strategies?

#### Must Know
- General MLLM recipe: vision encoder → alignment layer → LLM
- Difference between LLaVA (linear projection, 576 tokens) and BLIP-2 (Q-Former, 32 tokens)
- Visual instruction tuning: freeze ViT, train projection + LLM on (image, instruction, answer) triples

### Likely Asked
- *"How would you add vision to LLaMA-3?"* — Attach a CLIP or SigLIP ViT, project visual tokens to LLaMA's embed dimension with an MLP, fine-tune on instruction-following visual QA data
- *"What is the Q-Former and when would you use it?"* — A cross-attention transformer that compresses many visual tokens into few learnable query outputs; use when the LLM has short context limits or when visual compression is needed
- *"Why freeze the ViT during initial training?"* — Prevents catastrophic interference; the ViT's features are already strong; frozen ViT lets you focus the compute budget on learning the alignment

#### The Junior Answer vs Senior Answer

**Q: "How would you add vision capabilities to an existing LLM like LLaMA?"**
**Junior**: "Attach a vision encoder and fine-tune."
*Why this signals junior:* Vague — doesn't specify the alignment mechanism or training strategy.
**Senior**: "Three components: (1) Vision encoder (e.g., CLIP ViT-L/14) to convert 224×224 image → 256 patch tokens. (2) Alignment layer (e.g., linear projection or Q-Former) to map ViT's 768-dim embeddings to LLaMA's 4096-dim space. (3) Freeze ViT, train projection + LLaMA on instruction-following visual QA data (e.g., LLaVA-150k). Freezing prevents catastrophic interference. For token compression, LLaVA uses 576 tokens (24×24 patches), BLIP-2 uses a Q-Former to compress to 32 learnable queries — choose based on context window budget."
*Why this signals senior:* Names all three components, specifies dimensions, describes freeze strategy, compares token compression approaches, cites datasets.

#### The Key Tradeoffs

| Approach | When It Wins | When It Loses | Decision Criterion |
|----------|--------------|---------------|-------------------|
| LLaVA (linear projection, 576 tokens) | Simple, no extra training for alignment, full spatial detail | Large context usage (576 tokens per image) | If LLM has 32k+ context, use LLaVA; for shorter context, use Q-Former |
| BLIP-2 (Q-Former, 32 tokens) | Context-efficient, compresses heavily | Loses fine-grained spatial detail | If context budget <4k, use Q-Former; if detail critical, use full tokens |
| Freeze ViT | Prevents catastrophic forgetting, cheaper training | Can't adapt to new visual domains | Default for initial training; unfreeze ViT only if domain shift is large |

#### Failure Mode Gotchas

**Warning — Trap:** "MLLMs 'see' images like humans" — NO.
**Reality:** They process a sequence of numerical patch embeddings; spatial understanding is learned, not built-in.
**Interview impact:** Shows you understand the representation gap.

**Gotcha:** "More visual tokens = better performance."
**Reality:** 576 tokens (LLaVA) vs 32 tokens (BLIP-2 Q-Former) — performance difference is task-dependent. Fine-grained spatial tasks (OCR, counting) favor more tokens; semantic QA works fine with 32.
**In VisualForge:** Use Q-Former for caption generation (semantic), full tokens for layout analysis (spatial).

#### The Production Angle

**In VisualForge Studio (planned MLLM module for prompt enhancement):**
- Vision encoder: CLIP ViT-L/14 (frozen, shared with T2I pipeline)
- LLM: LLaMA-3-8B fine-tuned on visual instruction data
- Alignment: Linear projection (768 → 4096 dim) + 2-layer MLP
- Token budget: 256 image tokens (compressed from 576 via learned pooling)
- Use case: User uploads reference image → MLLM generates enhanced prompt → T2I pipeline generates similar image
- Latency: ViT encode 50ms + LLM inference 200ms = 250ms overhead per request

---

### Ch.11 — Generative Evaluation

#### What They're Testing
Can you explain FID mechanics, distinguish FID from CLIP Score, and describe what each metric misses?

#### Must Know
- FID formula: Fréchet distance between Gaussians fitted to Inception features
- Why FID needs large N (bias, variance)
- CLIP Score: cosine similarity between CLIP text and image embeddings, scaled by 2.5
- Trade-off: no single metric captures fidelity *and* diversity *and* text alignment
- LPIPS vs. SSIM — learned vs. hand-crafted perceptual similarity

### Likely Asked
- "What's the difference between FID and IS?" (FID uses real images; IS does not)
- "How would you evaluate text-to-image generation?" (FID + CLIP Score + human eval)
- "Why does FID increase when you use fewer samples?" (Gaussian fit becomes noisier)
- "Name a metric for evaluating compositional text prompts" (GenEval, T2I-CompBench)
- "What is Precision/Recall in the context of generative models?"

#### The Junior Answer vs Senior Answer

**Q: "How do you evaluate a text-to-image model's quality?"**
**Junior**: "Compute FID score."
*Why this signals junior:* Single metric, no understanding of what FID misses (prompt alignment).
**Senior**: "You need complementary metrics: (1) FID for distributional realism — fit Gaussians to Inception features of real vs generated images, compute Fréchet distance. Need 5k+ samples to reduce bias. (2) CLIP Score for text-image alignment — cosine similarity between CLIP embeddings of prompt and generated image. (3) Human eval on 100–200 samples for compositional accuracy (attribute binding, spatial relations) — FID and CLIP Score both miss fine-grained composition failures. In VisualForge, we track FID daily (5k samples) and CLIP Score per generation, with monthly human eval sweeps."
*Why this signals senior:* Names multiple metrics, explains what each measures, notes sample size requirements, identifies blind spots, grounds in production monitoring.

#### The Key Tradeoffs

| Metric | What It Measures | What It Misses | Sample Size | Use Case |
|--------|------------------|----------------|-------------|----------|
| FID | Distributional realism (mode coverage, diversity) | Prompt alignment, composition | 5k–10k | Primary quality metric |
| CLIP Score | Text-image semantic alignment | Fine-grained composition, quality | 1 per image | Per-generation monitoring |
| Inception Score (IS) | Class diversity (ignores real data) | Realism, prompt alignment | 5k+ | Mostly deprecated |
| LPIPS | Perceptual similarity (needs reference) | Semantic alignment (reference-free) | Pairwise | A/B testing, img2img |
| Human eval | Ground truth for composition, aesthetics | Expensive, slow, subjective | 100–200 | Monthly sweeps |

**Decision criterion:** Use FID for model selection, CLIP Score for per-generation filtering, human eval for compositional failure detection.

#### Failure Mode Gotchas

**Warning — Trap:** Reporting FID on <5k samples without flagging the bias.
**Impact:** Small-sample FID is noisy — Gaussian fit variance dominates.
**Fix:** Always report sample size with FID; use bootstrapping to estimate confidence intervals.

**Gotcha:** "CLIP Score captures compositional accuracy" — NO.
**Reality:** CLIP Score is global semantic similarity; it can't verify attribute binding. "a red cube and blue sphere" with swapped colors scores nearly identically to the correct image.
**Solution:** Use GenEval or T2I-CompBench for compositional eval.

**Warning — Video trap:** "High per-frame FID means good video" — NO.
**Missing:** Temporal coherence. A strobing video can have excellent per-frame FID but unwatchable flicker.
**Use instead:** FVD (Fréchet Video Distance) with I3D features, or VBench for comprehensive temporal eval.

#### The Production Angle

**In VisualForge Studio (10k generations/day):**
- **FID:** Computed daily on 5k random samples vs LAION-Aesthetics-6.5+ subset. Target: <25 (lower is better). Alert if >30.
- **CLIP Score:** Computed per generation, cached. Target: >0.28 (median). Generations <0.22 flagged for review.
- **Human eval:** Monthly 200-sample sweep, Likert scale (1–5) on prompt adherence, composition, quality. Target: mean >4.0.
- **A/B testing:** LPIPS for comparing img2img output vs reference, threshold <0.25 for "similar enough."
- **Compositional failure detection:** Weekly GenEval run on 100 prompts testing attribute binding, spatial relations, counting.

---

### Ch.12 — Local Diffusion Lab

#### What They're Testing
Can you walk through the full T2I pipeline from prompt to pixel, and explain key speed-up levers?

#### Must Know
- The full T2I pipeline: CLIP encode → latent DDIM → VAE decode.
- Key speed-up levers: DDIM (fewer steps), latent space (smaller feature maps), LCM/Turbo distillation.
- CFG: two forward passes per step, guidance scale w, why w>1 improves quality but hurts diversity.
- FID as the standard quality metric; its N-sample bias.

### Likely Asked
- "Walk me through generating an image from a text prompt with Stable Diffusion."
- "What is the role of the VAE in latent diffusion?"
- "How does ControlNet inject spatial conditioning?"
- "How would you systematically evaluate a new generative model?"
- "What's the trade-off between guidance scale and diversity?"

#### The Junior Answer vs Senior Answer

**Q: "Walk me through generating an image from a text prompt with Stable Diffusion."**
**Junior**: "The model takes the prompt and generates pixels."
*Why this signals junior:* Hand-waves the entire pipeline — no mention of CLIP, latent space, or denoising.
**Senior**: "Five stages: (1) CLIP text encoder converts prompt to 77 token embeddings (768-dim each for SD 1.5). (2) Start from Gaussian noise in 64×64×4 latent space. (3) DDIM denoiser runs 25 steps — at each step, the U-Net predicts noise $\epsilon_\theta(z_t, t, c)$ conditioned on text via cross-attention, subtract noise to get $z_{t-1}$. (4) CFG blends conditional and unconditional predictions with scale $w=7.5$. (5) VAE decoder upsamples 64×64×4 latent to 512×512×3 RGB image. Latency: 1.0s on A100. In VisualForge, we cache CLIP embeddings per prompt and batch CFG predictions for efficiency."
*Why this signals senior:* Names all components, specifies dimensions and step count, explains CFG, quantifies latency, mentions production optimizations.

#### The Key Tradeoffs

| Speed-up lever | Latency gain | Quality impact | When to use |
|----------------|--------------|----------------|-------------|
| Fewer DDIM steps (50→25) | 2× faster | ~5–10% FID increase | Standard production |
| DPM-Solver++ (15 steps) | 3× faster | ~2% FID increase | Premium tier |
| LCM distillation (1–4 steps) | 10× faster | ~20–30% FID increase | Real-time preview |
| Latent vs pixel space | 64× faster | Minimal (VAE compression) | Always use latent |
| Float16 vs Float32 | 1.5–2× faster | Negligible if mixed precision | Always use fp16 forward, fp32 accumulation |
| Batch size 1→4 | 1.5× higher throughput | None (latency per request increases) | High-throughput offline |

**Decision criterion:** If user-facing interactive, use 25-step DDIM. If batch processing, use 15-step DPM-Solver++ at batch=8.

#### Failure Mode Gotchas

**Warning — Trap:** "Stable Diffusion runs the denoiser in pixel space" — NO, it runs in **latent** space (64×64×4).
**Why it matters:** Shows you understand the core SD innovation.

**Gotcha:** Conflating LoRA (parameter-efficient fine-tuning, updates U-Net attention weights) with textual inversion (token-based, updates only embedding).
**When to use which:** LoRA for style/composition changes, textual inversion for single-concept appearance.

#### The Production Angle

**VisualForge Studio — Full T2I Pipeline (10k generations/day):**
- **Hardware:** 2× A100 80GB, load-balanced (5k req/day each)
- **Model:** SD 1.5 base (2 GB) + 50 client LoRAs (3 MB each, hot-swapped)
- **Prompt processing:** CLIP ViT-L/14 text encoder (20ms) → 77×768 embeddings → cached per unique prompt
- **Denoising:** 25 DDIM steps, CFG scale 7.5 (2× U-Net calls per step), batch=1 for low latency
- **Per-step time:** 40ms (U-Net forward on A100 at 512×512 latent)
- **VAE decode:** 50ms (64×64×4 → 512×512×3)
- **Total latency:** 20ms (encode) + 25 × 80ms (denoise with CFG) + 50ms (decode) = 2.07s
- **Optimization:** CFG predictions batched (effective batch=2), reduces per-step time 40ms→80ms (not 2×40ms)
- **Quality gates:** CLIP Score <0.22 → retry with higher CFG; FID tracked daily on 5k samples

---

<details>
<summary> 5-Minute Crammer — last-resort prep</summary>

## 5 · The 5-Minute Concept Cram

> **Use this section for last-minute prep on concepts you're shaky on.** Each entry: ultra-dense 5-minute explanation to get you through basic interview questions without embarrassment.

### 1. Vision Transformers (ViT) — Core Mechanics

**What it is:** Treat images as sequences of patches. Split image into 16×16 patches, flatten each, linearly project to embeddings, add positional encodings, feed to standard transformer.

**Key insight:** No 2D convolution — uses global self-attention over patches. This removes spatial inductive bias (good at scale, bad for small data).

**Interview formula:** For 224×224 image with 16×16 patches: $(224/16)^2 = 196$ patches + 1 CLS token = 197 tokens total.

**When to use:** When you have pretrained weights (CLIP ViT-L/14) or massive dataset (>100M samples). For small data (<1M), use CNN or DeiT.

**Common trap:** "ViT beats CNNs always" — NO. Only with sufficient scale or compensating techniques (DeiT, MAE).

---

### 2. CLIP — Contrastive Learning Essentials

**What it is:** Train image encoder (ViT) and text encoder (Transformer) to align via contrastive loss. Batch of N (image, text) pairs → maximize cosine similarity for matching pairs, minimize for non-matching.

**Loss:** InfoNCE — softmax over N² possible pairings per batch. Large batch = more hard negatives = better representations.

**Why batch size matters:** Batch 32k = 32,767 negatives per positive. Small batch = easy negatives = poor separation.

**How it's used in SD:** CLIP text encoder (frozen) converts prompt → 77×768 embeddings → fed as cross-attention K/V in U-Net.

**Interview answer:** "CLIP creates a shared embedding space where semantically related text and images are close. It's the conditioning mechanism for SD and the backbone for zero-shot image classification."

---

### 3. Diffusion Models — Training vs Inference

**Training (one step):** Sample timestep $t$, add noise to image: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$. U-Net predicts $\epsilon$. Loss: MSE on noise.

**Inference (T steps):** Start from pure noise $x_T \sim \mathcal{N}(0,I)$. For $t = T, T-1, ..., 1$: predict noise $\epsilon_\theta(x_t, t)$, subtract it, get $x_{t-1}$. After T steps → clean image.

**Why T=1000 for DDPM:** Each step is a small Gaussian perturbation — larger steps break the approximation.

**Why DDIM is faster:** Reinterprets as ODE, skips timesteps (e.g., use every 20th step) without quality loss.

**Interview answer:** "Diffusion training is cheap — single-step MSE on noise prediction. Inference is expensive — 50+ U-Net forward passes. DDIM solves this with non-Markovian sampling."

---

### 4. Latent Diffusion (Stable Diffusion) — Why Latent Space?

**Core idea:** Run diffusion in compressed latent space, not pixel space.

**Components:** (1) VAE encoder: 512×512×3 → 64×64×4 latent. (2) U-Net denoiser: diffuse in latent. (3) VAE decoder: 64×64×4 → 512×512×3 image.

**Speedup:** 8× spatial downsampling per dimension = $(8^2)^2 = 64×$ fewer operations for attention (which is $O(n^2)$).

**VAE scaling:** Latent is scaled by 0.18215 to match unit variance for DDPM's $\mathcal{N}(0,I)$ prior.

**Interview answer:** "SD runs diffusion in latent space for 64× compute savings. The VAE is pretrained and frozen; only the U-Net denoiser is trained. Quality is preserved because the VAE reconstructs faithfully."

---

### 5. Classifier-Free Guidance (CFG) — Mechanism

**Training:** Randomly drop text conditioning 10–20% of the time → model learns both conditional and unconditional generation.

**Inference:** Make two predictions per step: $\epsilon_\text{cond}$ (with prompt), $\epsilon_\text{uncond}$ (without). Blend: $\epsilon_\text{guided} = \epsilon_\text{uncond} + w (\epsilon_\text{cond} - \epsilon_\text{uncond})$.

**Guidance scale $w$:** 1.0 = unconditional (prompt ignored). 7.5 = standard. 12+ = oversaturated (neon colors, artifacts).

**Why it works:** Amplifies the direction toward conditional score, increasing prompt adherence at cost of diversity.

**Interview answer:** "CFG trains one model for both conditional and unconditional generation. At inference, blending the two predictions with scale $w > 1$ increases prompt adherence. Higher $w$ = stronger alignment, lower diversity."

---

### 6. Evaluation Metrics — FID vs CLIP Score

**FID (Fréchet Inception Distance):**
- **What:** Distance between real and generated image distributions in Inception feature space.
- **Lower is better.** Captures realism and diversity.
- **Blind spot:** Ignores prompt alignment.
- **Sample size:** Need 5k+ for stable estimate.

**CLIP Score:**
- **What:** Cosine similarity between CLIP embeddings of prompt and generated image.
- **Higher is better.** Captures text-image alignment.
- **Blind spot:** Doesn't measure quality or fine-grained composition.

**When to use both:** FID for model selection (is the distribution realistic?), CLIP Score for per-generation filtering (does this image match the prompt?).

**Interview answer:** "FID measures distributional realism but misses prompt alignment. CLIP Score measures semantic alignment but misses quality. You need both plus human eval for compositional accuracy."

---

> 📖 **Optional:** For deeper dives on any concept, see the full deep-dive sections above or the parent [Multimodal AI track](../multimodal_ai).

</details>

---

## Related Topics

> ➡ **Forward pointers:** Where to go next after mastering this guide.

- [Agentic AI Interview Guide](agentic-ai.md) — CoT, ReAct, RAG, embeddings fundamentals
- [Multi-Agent AI Interview Guide](multi-agent-ai.md) — agent protocols, MCP, A2A, event-driven systems
- [AI / LLM Fundamentals](../ai/llm_fundamentals) — transformer architecture, attention mechanism
- [AI / Fine-tuning](../ai/fine_tuning) — LoRA, full fine-tuning, adapter methods

> **Prerequisite concepts:** If you're shaky on these, review them first:
- Linear algebra: matrix multiplication, dot products, L2 normalization (CLIP embeddings)
- Probability: Gaussian distributions, KL divergence (VAE loss)
- Transformers: self-attention, positional encodings, multi-head attention (ViT, U-Net)

---

## 3 · The Rapid-Fire Round

> 20 Q&A pairs. Each answer: ≤ 3 sentences.

**1. How does a ViT turn an image into tokens?**
Divide the image into fixed-size patches (e.g. 16×16 pixels). Flatten each patch, project it to a d-dimensional embedding, and prepend a learnable CLS token. Add 2D positional embeddings and feed the sequence to a standard transformer.

**2. What does the CLS token do in ViT?**
It is a learnable token prepended to the patch sequence. After all transformer layers, its output embedding is used as the global image representation for classification. It aggregates information from all patches via attention.

**3. What is the InfoNCE loss in CLIP?**
A contrastive objective that treats each (image, text) pair in a batch as a positive and all other combinations as negatives. It maximizes cosine similarity for matching pairs and minimizes it for non-matching. Large batch size is critical because more negatives = harder negatives = better representations.

**4. Can you use CLIP embeddings for retrieval directly?**
Yes — CLIP produces a shared embedding space where L2 distance or cosine similarity measures image-text semantic alignment. Normalize embeddings first; then use FAISS or HNSW for approximate nearest neighbor search.

**5. What happens during the forward diffusion process?**
Gaussian noise is progressively added to the image over T timesteps according to a variance schedule. After T steps the image becomes pure Gaussian noise. This process is fixed and has no learnable parameters.

**6. What does the U-Net learn in a diffusion model?**
To predict the noise ε that was added at each timestep t, given the noisy image $x_t$ and the timestep $t$. The training objective is to minimize $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$. During sampling, the predicted noise is subtracted to recover the clean image.

**7. DDPM vs. DDIM — key difference?**
DDPM is stochastic (adds Gaussian noise at each reverse step). DDIM is deterministic — it skips timesteps by choosing a non-Markovian reverse process. DDIM can generate in 20–50 steps vs DDPM's 1000, at comparable quality.

**8. What is classifier-free guidance?**
Train the same U-Net with and without the text conditioning (randomly drop the text during training). At inference, blend the conditional and unconditional predictions: $\epsilon_\text{guided} = \epsilon_\text{uncond} + w(\epsilon_\text{cond} - \epsilon_\text{uncond})$. Higher $w$ increases adherence to the prompt but reduces diversity.

**9. What does the VAE do in Stable Diffusion?**
Compresses a 512×512 pixel image into a 64×64×4 latent representation. The diffusion process runs in this latent space, making it ~64× cheaper computationally. The VAE decoder reconstructs the pixel image from the denoised latent.

**10. Why does Stable Diffusion run in latent space and not pixel space?**
The latent space is 8× smaller in each spatial dimension. Running 1000 denoising steps on 64×64 tensors vs 512×512 pixels reduces compute by ~64×. Quality is preserved because the VAE is trained to reconstruct faithfully.

**11. What is a cross-attention layer and what does it add to diffusion?**
Cross-attention allows the U-Net to attend to text token embeddings at every layer. Query comes from the image features; key and value come from the text embeddings. This is what makes diffusion models steerable by language.

**12. What is ControlNet?**
A trainable copy of the U-Net encoder that accepts additional spatial conditioning (edges, depth, pose). It injects conditioning via zero-convolutions (initialized to zero so they don't disturb the base model). Enables precise spatial control beyond what text prompts provide.

**13. What is FID and what does it measure?**
Fréchet Inception Distance measures the distance between the distributions of real and generated images in Inception feature space. Lower is better. Sensitive to both quality and diversity; doesn't capture individual image quality.

**14. What does CLIP Score measure?**
The cosine similarity between a generated image's CLIP embedding and the prompt's CLIP embedding. Measures prompt alignment, not image quality. High CLIP Score can coexist with artifacts; low FID can have poor prompt adherence.

**15. What is textual inversion?**
A fine-tuning method that learns a new token embedding $S^*$ representing a concept (e.g. a specific product or style). Only the embedding is trained; the model weights are frozen. Much cheaper than full fine-tuning but less expressive.

**16. LoRA vs. textual inversion for product fine-tuning?**
LoRA trains low-rank weight updates in the U-Net's attention layers — captures style and composition changes, not just appearance. Textual inversion only captures embedding-level appearance. Use LoRA when the output structure needs to change, textual inversion for simpler appearance adaptation.

**17. CFG scale: what happens above 12?**
The guidance becomes oversaturated — colors become neon, textures oversharp, and artifacts appear. The model is pulled too strongly toward the conditional score. Typical production range: 6–12 for image generation, 3–6 for video.

**18. What is DPM-Solver++?**
A high-order ODE solver for diffusion reverse process. Achieves near-DDPM quality in 10–20 steps by using high-order Taylor approximations. Now the default scheduler in many production pipelines.

**19. Batch size tradeoff for real-time diffusion?**
Batch=1 per request minimizes latency but underutilizes the GPU. Batch=4–8 amortizes memory loads but increases TTFT for any single request. For interactive generation, use batch=1 with SDXL-Turbo (1–4 step).

**20. When would you choose SDXL over SD 1.5?**
When image quality and prompt adherence matter more than speed. SDXL requires ~6 GB more VRAM for base+refiner, runs 2–3× slower, but produces significantly better text rendering, composition, and photorealism. Use SD 1.5 for high-throughput or resource-constrained deployments.

---

## 4 · Signal Words That Distinguish Answers

### Senior Signals — Say This

**Architecture & Components:**
- "Vision encoder → alignment layer → LLM" (MLLM architecture pattern)
- "Cross-attention injection at every U-Net layer" (how text controls image generation)
- "Q-Former compresses 256 ViT tokens to 32 learnable queries" (BLIP-2 efficiency)
- "VAE encoder/decoder with 8× spatial downsampling" (latent diffusion mechanics)
- "Zero-convolution initialization prevents base model disruption" (ControlNet training trick)

**Training & Losses:**
- "InfoNCE loss over a batch of 32k pairs" (CLIP training, shows you understand negative mining)
- "Denoising score matching with MSE on predicted noise" (diffusion training objective)
- "Condition dropout for classifier-free guidance" (CFG training procedure)
- "LoRA rank=8 prevents overfitting through bottleneck" (parameter-efficient fine-tuning)

**Mechanics & Tradeoffs:**
- "Guidance scale saturates above 12 — neon colors and artifacts" (shows you know the failure mode)
- "DDIM skips timesteps via non-Markovian ODE" (not just "fewer steps")
- "Latent space is 64× cheaper for $O(n^2)$ attention" (quantified computational savings)
- "Temporal attention across frames at same spatial position" (video coherence mechanism)

**Evaluation & Production:**
- "FID measures distribution distance; CLIP Score measures prompt alignment" (separate metrics for separate concerns)
- "Need 5k+ samples for stable FID — Gaussian fit variance" (shows you understand metric bias)
- "In VisualForge, we default to 25-step DDIM for 1.0s latency" (grounds in production SLAs)
- "I'd instrument this with daily FID tracking and per-generation CLIP Score thresholds" (monitoring mindset)

### Junior Signals — Don't Say This

**Vague Mechanism:**
- "It generates from noise randomly" → (DDIM is deterministic; even DDPM is structured)
- "The model learns patterns" → (what patterns? What loss? What architecture?)
- "It depends on the use case" → (depends on WHAT specifically?)

**Wrong Mental Model:**
- "Stable Diffusion is pixel-based" → (NO — it runs in latent space)
- "CLIP just finds similar images" → (it's a zero-shot cross-modal reasoning model)
- "ViT has no inductive biases" → (it has tied patch weights = translation equivariance)
- "MLLMs see images like humans" → (they process numerical patch embeddings)

**Incomplete Thinking:**
- "Just increase steps for quality" → (quality plateaus around 50 steps for DDIM)
- "Fine-tuning means retraining the whole model" → (LoRA/textual inversion are parameter-efficient)
- "Negative prompts block content" → (they steer via CFG equation, not filter)
- "More visual tokens = better" → (depends on task: semantic QA works fine with 32 tokens)

**Missing Production Awareness:**
- "I'd look at the logs" → (what metrics? What thresholds? What's the alert trigger?)
- "I'd test it" → (how? What eval dataset? What's passing criteria?)
- "It might be slow" → (quantify: 1s? 10s? At what batch size? What's the bottleneck?)

