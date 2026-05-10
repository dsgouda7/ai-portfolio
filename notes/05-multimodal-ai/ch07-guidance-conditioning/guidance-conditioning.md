# Guidance & Conditioning — Making Diffusion Models Follow Instructions

> **The story.** A bare diffusion model produces *plausible* images — but you can't tell it what to draw. **Classifier guidance** (**Dhariwal & Nichol**, OpenAI, **May 2021**) was the first answer: train a separate classifier on noisy images and use its gradients to nudge sampling toward a chosen class. The breakthrough was **Classifier-Free Guidance (CFG)** by **Jonathan Ho and Tim Salimans** at Google in **July 2022** — train a single conditional model that occasionally ignores its condition, then at inference linearly extrapolate between conditional and unconditional predictions. Cleaner, faster, and the foundation of every modern text-to-image model. **ControlNet** (Zhang & Agrawala, Stanford, **February 2023**) added structural conditioning — edges, depth, poses, scribbles — by training a parallel "control branch" that copies and fine-tunes the U-Net's encoder. **IP-Adapter** (Tencent, 2023) added image-prompt conditioning. The diffusion-control toolbox you ship with in 2026 — negative prompts, CFG scale, ControlNet, IP-Adapter — is built on these three papers.
>
> **Where you are in the curriculum.** [LatentDiffusion](../ch06_latent_diffusion) wired CLIP into the U-Net so the model could *see* a text prompt. This chapter explains how it actually *follows* it: classifier-free guidance, what the guidance scale parameter does geometrically, why a CFG of 7.5 produces sharper but sometimes oversaturated images, how negative prompts work mechanically, and how ControlNet imposes structural constraints on top.

![Guidance conditioning flow animation](img/guidance-conditioning-flow.gif)

*Flow: conditional and unconditional branches are blended by CFG before each denoising update, increasing prompt adherence.*

---

## 0 · The VisualForge Studio Challenge

**Mission**: You're the Lead ML Engineer at VisualForge Studio. Your team needs <5% unusable generations to avoid wasting iteration time on regenerations.

**Current blocker at Chapter 7**: Your Stable Diffusion pipeline (Ch.6) generates from text prompts but outputs are **unpredictable**. Client brief: "Mango leather crossbody bag, center frame, white background" — but you're getting cluttered backgrounds, wrong composition, weird angles. **~25% of outputs are unusable** = your design team spends hours regenerating the same brief.

**What this chapter unlocks**: **Classifier-Free Guidance (CFG)** — you get a guidance scale parameter that controls prompt adherence. Scale 1.0 = creative but ignores your prompt. Scale 7.5 = balanced (production default). Scale 12.0 = strict prompt following. Add **negative prompts** to subtract unwanted concepts ("blurry, cluttered, text, shadow"). Together: you drop unusable rate to <15%.

---

### The 6 Constraints — Snapshot After Chapter 7

| Constraint | Target | Status | Evidence |
|------------|--------|--------|----------|
| #1 Quality | ≥4.0/5.0 | **~3.8/5.0** | Better prompt adherence improves client ratings |
| #2 Speed | <30 seconds | **20s** | Unchanged from Ch.6 |
| #3 Cost | <$5k hardware | **$2.5k laptop** | Unchanged from Ch.6 |
| #4 Control | <5% unusable | **<15% unusable** | CFG scale 12.0 + negative prompts improve success rate |
| #5 Throughput | 100+ images/day | **~50 images/day** | Lower unusable rate = less regeneration time |
| #6 Versatility | 3 modalities | **Text→Image enabled** | Still only text→image, no video/understanding |

---

### What's Still Blocking Us After This Chapter?

**Composition control**: CFG improves prompt adherence but can't guarantee **exact composition**. Your brief says "product at 45-degree angle" — still fails 60% of the time. Text alone isn't precise enough. You need structural control, not just semantic guidance.

**Next unlock (Ch.8)**: **ControlNet** — you condition on edge maps, depth maps, pose skeletons. Your designer sketches the layout in 30 seconds → ControlNet enforces that structure → 95% first-try success rate.

---

## 1 · Core Idea

An unconditional diffusion model generates plausible images but has no mechanism for you to direct the output. **Guidance** is the technique that adds this direction. The key insight is: during denoising, instead of always moving toward "any plausible image", move toward "a plausible image **that satisfies this condition**". The condition can be a class label, a text embedding, or any other signal.

**Classifier-free guidance (CFG)** is the dominant method. Train the same U-Net for both conditional and unconditional generation (randomly dropping the condition during training). At inference, run the model twice: once with the condition and once without. Then amplify the difference: lean further in the direction of the conditioned prediction than the unconditioned prediction. The guidance scale $w$ controls how much you amplify.

---

## 2 · Running Example — PixelSmith v3.5

The VisualForge creative team has a new problem: unconditioned generation produces varied results — sometimes the bag faces the wrong direction, sometimes the background is off-white. **Classifier-Free Guidance (CFG)** locks the model onto the brief.

```
VisualForge brief: "spring-collection hero shot"
Prompt: "Mango leather crossbody bag, center frame, white background, studio lighting"
Negative: "cluttered background, shadow, people, logo, blur"

CFG scale sweep on this brief:
 scale=1.0 → unconditioned (barely follows the prompt — varied, creative but unfocused)
 scale=7.5 → production default (follows brief closely, high quality)
 scale=12.0 → over-conditioned (artifacts, oversaturated colors)
```

> 📖 **Educational proxy:** The math is illustrated with digit-class labels (0–9) because it shows the conditional direction clearly with only 10 classes. The production mechanism is identical — replace "digit 3" with "white background product shot."

```
Educational math proxy:
 Digit class label 3 → guides denoising toward "threes"
 Class label dropped 10% during training → model learns both conditional and unconditional
 CFG: ε̃ = ε_uncond + γ(ε_cond - ε_uncond)
 γ=7.5 in production (same γ for digits or product shots)
```

---

## 3 · The Math

### 3.1 Conditioning via Cross-Attention (text)

In text-conditioned diffusion (Stable Diffusion), the U-Net receives a sequence of text token embeddings $\mathbf{c} \in \mathbb{R}^{77 \times d}$ from the frozen CLIP text encoder. Each ResBlock in the U-Net includes a **cross-attention** layer:

$$\text{CrossAttn}(Q, K, V) = \text{softmax} \left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where:
- $Q = W_Q \cdot \mathbf{z}$ — queries from the spatial feature map
- $K = W_K \cdot \mathbf{c}$ — keys from the text embedding
- $V = W_V \cdot \mathbf{c}$ — values from the text embedding

Each spatial location in the feature map attends to all 77 text tokens and learns which words to pay attention to for that image region.

### 3.2 Classifier Guidance (original)

The original guidance method (Dhariwal & Nichol 2021) needs a separately trained classifier $p_\phi(y | x_t)$:

$$\nabla_{x_t} \log p_\theta(x_t | y) = \nabla_{x_t} \log p_\theta(x_t) + w \cdot \nabla_{x_t} \log p_\phi(y | x_t)$$

The denoising score is shifted by the gradient of the classifier. Guidance scale $w$ amplifies this shift. **Problem:** requires training a separate noisy-image classifier at every noise level — expensive and impractical for open-domain text.

### 3.3 Classifier-Free Guidance (CFG)

Ho & Salimans (2021) eliminate the separate classifier. Train a single model $\boldsymbol{\epsilon}_\theta(x_t, t, \mathbf{c})$ with condition dropout:

$$\mathbf{c} = \emptyset \text{ (null)} \quad \text{with probability } p_{uncond} \approx 0.1$$

At inference, run the model twice:

$$\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(x_t, t, \emptyset) + w \cdot \left[ \boldsymbol{\epsilon}_\theta(x_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(x_t, t, \emptyset) \right]$$

**Decomposing:**

| Term | Meaning |
|------|---------|
| $\boldsymbol{\epsilon}_\theta(x_t, t, \emptyset)$ | Unconditioned noise prediction (what any image looks like) |
| $\boldsymbol{\epsilon}_\theta(x_t, t, \mathbf{c})$ | Conditioned noise prediction (what a cat image looks like) |
| difference | The direction in noise space that moves toward the condition |
| $w \cdot \text{difference}$ | Amplify that direction by guidance scale $w$ |

**When $w = 1$:** standard conditional sampling (no extra amplification).
**When $w = 7.5$:** the model overshoots toward the condition — images are crisper and more on-topic but may become oversaturated or lose texture diversity.
**When $w = 0$:** unconditioned generation. Your prompt is ignored.

### 3.4 Negative Prompts

A negative prompt $\mathbf{c}_{neg}$ extends CFG:

$$\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(x_t, t, \mathbf{c}_{neg}) + w \cdot \left[ \boldsymbol{\epsilon}_\theta(x_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(x_t, t, \mathbf{c}_{neg}) \right]$$

The model moves away from $\mathbf{c}_{neg}$ and toward $\mathbf{c}$. Common negative prompts include "blurry, low quality, watermark" — the model actively steers away from those image features.

---

## 4 · Visual Intuition

### How CFG Works — Step by Step

**Class-conditional training:**
1. Assign each training image a class label (e.g., digit 0–9)
2. Embed the class label as a vector $\mathbf{c}$ (learnable lookup table)
3. Add $\mathbf{c}$ to the timestep embedding and inject into every ResBlock
4. With probability $p = 0.1$, replace $\mathbf{c}$ with a null embedding $\emptyset$
5. Train with the same MSE noise prediction loss

**Class-conditional inference:**
1. Start from $x_T \sim \mathcal{N}(0, \mathbf{I})$
2. At each step $t$:
 - Compute $\boldsymbol{\epsilon}_{cond} = \boldsymbol{\epsilon}_\theta(x_t, t, \mathbf{c})$ (one forward pass)
 - Compute $\boldsymbol{\epsilon}_{uncond} = \boldsymbol{\epsilon}_\theta(x_t, t, \emptyset)$ (second forward pass)
 - CFG: $\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_{uncond} + w \cdot (\boldsymbol{\epsilon}_{cond} - \boldsymbol{\epsilon}_{uncond})$
 - Sample $x_{t-1}$ using $\hat{\boldsymbol{\epsilon}}$

**Note:** CFG doubles the compute cost at inference (two U-Net calls per step). Distilled models like LCM/Turbo can do CFG in a single pass.

---

### CFG Geometry in Noise Space

```
Unconditioned score direction: "move toward any plausible image"
 ε_uncond ────────────▶

Conditioned score direction: "move toward an image of a cat"
 ε_cond ────────────────────▶

CFG direction (w=7.5): "move MUCH more strongly toward cat"
 ε_cfg ────────────────────────────────────────────▶

 = ε_uncond + 7.5 × (ε_cond - ε_uncond)

Effect: at the cost of diversity, the image will be much more obviously a cat.
```

### Cross-Attention in the U-Net

```
Feature map (spatial) Text tokens
(B, d, H, W) (B, 77, d_text)
 │ │
 ▼ Linear W_Q ▼ Linear W_K, W_V
 Q (B, H*W, d_k) K (B, 77, d_k)
 │ V (B, 77, d_v)
 └─────────────────────────┐
 ▼
 softmax(Q·Kᵀ / √d_k) · V
 │
 ▼
 (B, H*W, d_v)

Each image region (row in Q) learns which text tokens to attend to.
"fluffy" → attends to fur-like spatial regions.
"blue sky" → attends to the top of the image.
```

---

## 5 · Production Example — VisualForge in Action

**Brief type: Brand-Constrained Product-on-White with Negative Prompt Guardrails**

VisualForge brand guidelines prohibit: visible logos on competitive products, cluttered backgrounds, model faces, and excessive shadows. CFG + negative prompts enforce these guardrails automatically.

```python
# Production: CFG scale sweep for VisualForge spring-collection brief
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained(
 "stabilityai/stable-diffusion-2-1",
 scheduler=DPMSolverMultistepScheduler.from_pretrained(
 "stabilityai/stable-diffusion-2-1", subfolder="scheduler"),
 torch_dtype=torch.float16
).to("cuda")

brief_prompt = "Mango leather crossbody bag, center frame, white background, studio lighting, sharp product photography"
brand_negative = "people, faces, logo, text, watermark, shadow, cluttered background, blur, grain, competitor brand"

# CFG scale sweep — pick scale that hits brief compliance without artifacts
for scale in [1.0, 7.5, 12.0]:
 img = pipe(
 brief_prompt,
 negative_prompt=brand_negative,
 guidance_scale=scale,
 num_inference_steps=20,
 generator=torch.manual_seed(42),
 ).images[0]
 img.save(f"vf_cfg_scale_{scale}.png")
```

**CFG scale effect on VisualForge product brief:**

| CFG scale | Brief compliance | Background | Artifact risk | Verdict |
|-----------|-----------------|------------|----------------|---------|
| 1.0 | Low (ignores prompt) | Variable | None | Not usable |
| 5.0 | Medium | Usually white | Low | Experimental |
| 7.5 | High | White | Low | **Production default** |
| 10.0 | Very high | White | Medium (color saturation) | Sometimes useful |
| 12.0 | Rigid | White | High (burnt highlights) | Avoid |

> VisualForge uses `guidance_scale=7.5` as the default. For unusual brief types (e.g., "abstract brand pattern"), experiment up to 9.0. Above 10.0, artifacts appear in >30% of outputs on product shots.

---

## 6 · Common Failure Modes

**Oversaturation from high guidance scales**
Guidance scale >12 produces vivid but unnatural colors. The model overfits to the text embedding, losing natural image statistics.
**Fix:** Use guidance_scale=7.5 as default; test up to 10.0 for specific briefs.

**Negative prompts failing to remove concepts**
Negative prompts steer semantically but cannot target spatial regions. "no cat" reduces cat-like features overall but doesn't guarantee absence.
**Fix:** Use ControlNet (Ch.8) for structural control; negative prompts are for semantic steering only.

**CFG ignoring complex prompts**
Long, multi-clause prompts (>77 tokens) get truncated. CFG only sees first 77 tokens.
**Fix:** Simplify prompts; use ControlNet for structural requirements; consider SDXL (longer context).

**Artifacts at extreme scales**
Guidance scale >15 causes burnt highlights, posterization, unnatural edges.
**Fix:** Stay in 5-12 range; check outputs at multiple scales during experimentation.

**Doubled inference cost**
CFG requires two U-Net forward passes per denoising step. 50 steps × 2 passes = 100 U-Net calls.
**Fix:** Use distilled models (LCM, SDXL-Turbo) for single-pass CFG; reduce steps with better schedulers (Ch.5).

**Null embedding confusion**
Some implementations use empty string `""`, others use learnable null token, others use zero vector. Behavior differs.
**Fix:** Check model docs; Stable Diffusion uses empty string encoding as null.

---

## 7 · When to Use This vs Alternatives

### CFG vs Classifier Guidance

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| **Classifier-Free Guidance** | No separate classifier; single model; works for open-domain text | 2× inference cost | **Default for text→image** (SD, DALL-E, Imagen) |
| **Classifier Guidance** | Historically first method; can use pre-trained classifiers | Requires noisy-image classifier at every noise level; impractical for text | Obsolete for production |

### CFG Guidance Scale Selection

| Guidance Scale | Effect | Use Case |
|----------------|--------|----------|
| **1.0** | Unconditioned (ignores prompt) | Creative exploration, dataset generation |
| **5.0-7.0** | Balanced creativity + adherence | Artistic, painterly, abstract briefs |
| **7.5** | **VisualForge default** | Professional product shots, hero images |
| **8.0-10.0** | High adherence | Strict brand guidelines, precise composition |
| **12.0+** | Oversaturated, artifacts | Avoid in production |

### Text Conditioning vs Structural Conditioning

| Method | Control Type | Precision | When to Use |
|--------|--------------|-----------|-------------|
| **CFG (this chapter)** | Semantic (text) | Medium | Prompt describes desired output |
| **ControlNet (Ch.8)** | Structural (edges, depth, pose) | High | Exact composition required |
| **IP-Adapter** | Image prompt (style, content) | High | Style transfer, reference images |
| **LoRA fine-tuning** | Domain-specific concepts | Very high | Custom products, brand-specific styles |

**VisualForge decision tree:**
- Brief specifies **semantic** requirements ("modern office", "spring colors") → **CFG**
- Brief specifies **spatial** requirements ("product at 45° angle", "person in this pose") → **ControlNet**
- Brief references **existing visual style** ("match our brand guide") → **LoRA + IP-Adapter**

### Scaling Parameters

| Parameter | Effect | Typical values |
|-----------|--------|---------------|
| Guidance scale $w$ | Higher = more faithful to prompt, less diversity | 5–15 (SD default: 7.5) |
| $p_{uncond}$ (dropout rate) | Higher = stronger unconditional capability | 0.05–0.2 |
| Text encoder frozen? | Fine-tuning CLIP with diffusion can hurt CLIP's zero-shot | Always frozen in SD |
| Cross-attention resolution | More attention layers → better text following | SD adds at 8×8, 16×16, 32×32, 64×64 |

**Perp-Neg (2023):** An extension of CFG that uses multiple negative prompts and handles the perpendicular direction in embedding space — prevents unintended semantic bleed between prompt elements.

---

## 8 · Connection to Prior Chapters

**From Ch.3 CLIP** — CLIP provided the text encoder that embeds prompts into the 77×768 vector space. CFG amplifies the direction defined by that embedding. Without CLIP's aligned text-image space, CFG would have no meaningful "conditioned direction" to amplify.

**From Ch.4 Diffusion Models** — DDPM established the denoising trajectory. CFG doesn't change the diffusion schedule or noise schedule — it only modifies the predicted noise $\boldsymbol{\epsilon}_t$ at each step by extrapolating between conditional and unconditional predictions.

**From Ch.5 Schedulers** — Schedulers (DDIM, DPM++) reduce the number of steps. CFG doubles the cost per step (two U-Net calls), so reducing steps from 1000 → 50 makes CFG practical. Without DDIM, CFG would require 2000 U-Net forward passes (1000 steps × 2 calls).

**From Ch.6 Latent Diffusion** — Latent diffusion moved from pixel space to latent space (8× compression). CFG operates in latent space — the two U-Net calls process 64×64 latents, not 512×512 pixels. This makes CFG fast enough for interactive use.

**Forward to Ch.8 Text-to-Image (ControlNet)** — CFG controls **semantic** adherence but not **structural** layout. ControlNet (next chapter) adds structural conditioning (edges, depth, pose) on top of CFG's semantic conditioning. VisualForge uses both: CFG for "modern office" + ControlNet for "product at 45° angle."

**Forward to Ch.11 Evaluation** — CFG guidance scale affects measurable quality metrics: higher scales improve CLIP score (text-image alignment) but can decrease FID (image realism). Evaluating CFG sweep requires automated metrics (Ch.11).

---

## 9 · Interview Checklist

### Must Know
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

### Trap to Avoid
- Confusing guidance scale with classifier temperature — they are different mechanisms
- Saying negative prompts "block" content — they steer, not filter
- Forgetting that $w=0$ ignores the prompt entirely (unconditioned), not $w=1$

---

## 10 · Further Reading

**Core papers:**
- **Classifier-Free Guidance (Ho & Salimans, 2021)** — [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — The CFG paper that replaced classifier guidance
- **Diffusion Models Beat GANs (Dhariwal & Nichol, 2021)** — [Paper](https://arxiv.org/abs/2105.05233) — Introduced classifier guidance (now obsolete but foundational)
- **High-Resolution Image Synthesis (Rombach et al., 2022)** — [Latent Diffusion paper](https://arxiv.org/abs/2112.10752) — How Stable Diffusion implements CFG in latent space

**Extensions and improvements:**
- **ControlNet (Zhang et al., 2023)** — [Paper](https://arxiv.org/abs/2302.05543) | [Code](https://github.com/lllyasviel/ControlNet) — Structural conditioning on top of CFG
- **IP-Adapter (Ye et al., 2023)** — [Paper](https://arxiv.org/abs/2308.06721) — Image-prompt conditioning
- **Perp-Neg (2023)** — [Repo](https://github.com/Perp-Neg/Perp-Neg) — Multiple negative prompts without semantic bleed

**Production tools:**
- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) — Interactive CFG scale and negative prompt experimentation
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — Node-based workflow for CFG + ControlNet pipelines
- [Diffusers library](https://github.com/huggingface/diffusers) — `guidance_scale` parameter in all pipelines

**Tutorials:**
- [Hugging Face Diffusers docs — CFG](https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation#classifier-free-guidance)
- [Fast.ai Stable Diffusion Deep Dive](https://www.fast.ai/posts/2022-11-21-stable-diffusion.html) — Jeremy Howard's walkthrough of CFG mechanism

---

## 11 · Notebook

**Interactive Jupyter notebook:**
📓 [guidance-conditioning.ipynb](guidance-conditioning.ipynb)

**What you'll build:**
1. **CFG scale sweep** — Generate same prompt at guidance scales 1.0, 5.0, 7.5, 10.0, 12.0 and compare outputs
2. **Negative prompt impact** — Compare generation with/without negative prompts
3. **Cross-attention visualization** — Extract and visualize attention maps for specific text tokens
4. **VisualForge brief test** — Run production brief ("product on white background") with optimal CFG settings

**Runtime:** ~15 minutes on Colab T4 GPU (free tier)
**Hardware:** Works on CPU (slow) or any GPU with 4GB+ VRAM
**Model:** Uses Stable Diffusion 2.1 (~5GB download)

**Learning goals:**
- Observe CFG geometry: how guidance scale amplifies the conditional direction
- Measure impact of negative prompts on output quality
- Verify that CFG doubles inference time (two U-Net calls per step)
- Find optimal guidance scale for VisualForge product brief type

---

## 11.5 · Progress Check — What Have We Unlocked?

### Before This Chapter
- **Constraint #1 (Quality)**: ~3.5/5.0, text prompts work but your outputs are unpredictable
- **Constraint #4 (Control)**: ~25% unusable (wrong composition, cluttered backgrounds)
- **VisualForge Status**: Your team spends hours regenerating to get usable outputs

### After This Chapter
- **Constraint #1 (Quality)**: **3.8/5.0** → CFG improves your prompt adherence
- **Constraint #4 (Control)**: **<15% unusable** → CFG scale 12.0 + negative prompts improve your success rate
- **Constraint #5 (Throughput)**: **~50 images/day** → Lower unusable rate = less wasted regeneration time
- **VisualForge Status**: Your prompt "modern office with natural light" + negative "cluttered, dark" → 85% success rate

---

### Key Wins

1. **CFG scale control**: You can tune guidance (7.5 = balanced, 12.0 = strict) for different brief types
2. **Negative prompts**: You subtract unwanted concepts ("blurry, low quality, watermark, cluttered")
3. **Cross-attention mechanism**: You understand how text tokens reach U-Net spatial layers (each pixel attends to 77 text tokens)

---

### What's Still Blocking Production?

**Composition/pose control**: CFG improves adherence but can't guarantee **exact layout**. Your brief says "product at 45-degree angle" — still fails 60% of the time. Text alone isn't precise enough for spatial constraints. You need **structural conditioning**.

**Next unlock (Ch.8)**: **Text-to-Image (ControlNet)** — you'll condition on edge maps, depth maps, pose skeletons. Your designer sketches a rough layout → ControlNet enforces that structure → 95% first-try success rate.

---

### VisualForge Status — Full Constraint View

| Constraint | Ch.4 | Ch.5 | Ch.6 | **Ch.7 (this)** | Target |
|------------|------|------|------|-----------------|--------|
| #1 Quality | 3.0/5.0 | 3.2/5.0 | 3.5/5.0 | ** 3.8/5.0** | ≥4.0/5.0 |
| #2 Speed | 5min | 30-60s | 20s | ** 20s** | <30s |
| #3 Cost | | | $2.5k | ** $2.5k** | <$5k |
| #4 Control | ~40% unusable | ~40% | ~25% | ** <15%** | <5% |
| #5 Throughput | ~10/day | ~15/day | ~40/day | ** ~50/day** | 100+/day |
| #6 Versatility | Can generate | | Text→Image | ** Text→Image** | 3 modalities |

**Progress this chapter:** CFG improved prompt adherence (#1 Quality ↑), reduced unusable rate (#4 Control ↑), increased throughput (#5 ↑ from less rework).

**Next chapter unlock:** ControlNet will push #4 Control from <15% → <5% unusable (target hit) by adding structural conditioning on top of CFG's semantic conditioning.

---

## 12 · Bridge to Chapter 8

**What you unlocked:** You now have tunable prompt adherence with CFG. Guidance scale 7.5 → 85% usable outputs for your VisualForge product briefs. Negative prompts subtract unwanted concepts ("cluttered, blurry"). **Your Constraint #4 (Control)** improved from 25% unusable → 15% unusable.

**What's still blocking you:** CFG controls **semantic** adherence but not **structural** layout. Your client brief says "product at 45-degree angle" — still fails 60% of the time. Text alone isn't precise enough for spatial constraints. "45-degree angle" means different things to different people, but a sketch is unambiguous.

**What you need next:** **ControlNet** — you'll condition on edge maps, depth maps, pose skeletons. Your designer sketches a rough layout in 30 seconds → ControlNet enforces that structure → 95% first-try success rate. CFG controls "what" (semantic), ControlNet controls "where" (spatial).

**The unlock:** You'll combine CFG + ControlNet → full control over generation. Text describes the concept ("modern office"), ControlNet enforces composition (edge map from designer sketch), negative prompt removes unwanted elements ("cluttered, dark"). Result: **<5% unusable rate** ( Constraint #4 target hit).

→ **Next chapter:** [Text-to-Image (ControlNet)](../ch08_text_to_image/README.md)

## Illustrations

![Guidance and conditioning - CFG, scale sweep, cross-attention, ControlNet family](img/guidance-conditioning.png)
