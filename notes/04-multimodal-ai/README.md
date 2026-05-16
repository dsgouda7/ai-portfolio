# Multimodal AI — How to Read This Collection

→ Interview prep: [Interview Guide](../interview-guides/multimodal-ai.md)

> This document is your **entry point and reading map**. It explains the conceptual arc across all notes, defines the running example that threads through every chapter, shows how each document connects to the others, and prescribes reading paths based on your goal.

---

## The Central Story in One Paragraph

Modern generative AI systems — the ones that turn a text prompt into a photorealistic image, generate a video from a sentence, or answer questions about a photograph — are all built on one foundational idea: **any signal (pixels, audio frames, video clips, text tokens) can be projected into the same high-dimensional embedding space, and a transformer (or diffusion model) can learn the joint distribution over those signals**. To build that intuition from the ground up, you need four layers of knowledge: (0) how raw signals become tensors and then tokens (Foundations + Vision Transformers), (1) how meaning is aligned across modalities without labels (CLIP and contrastive learning), (2) how models generate entirely new signals they were never shown — not just predict them (Diffusion Models and Latent Diffusion), and (3) how those generation capabilities are extended to video, conditioned on structure, and baked into language models (Text-to-Video + Multimodal LLMs). The documents in this collection cover each layer in depth, and they deliberately cross-reference the AI track (embeddings, transformers, fine-tuning) because the layers are not independent.

---

## The Grand Challenge: VisualForge Studio

> **The Mission**: Build **VisualForge Studio** — a local AI creative pipeline that replaces $600k/year freelancer costs while maintaining professional-grade quality on $5k hardware with zero cloud fees.

This is not a toy image generator. Every chapter threads through a single production challenge: you're the Lead ML Engineer at **Aperture Creative**, a boutique marketing agency that just lost 60% of its budget. The Creative Director needs to maintain 2,000 assets/year output (product shots, social ads, explainer videos, voiceovers) with a skeleton crew and no Midjourney/Runway subscriptions. You must prove that local AI can deliver stock-photo-grade quality at 10× throughput while running on hardware you can expense on a startup credit card.

---

### The 6 Core Constraints

| # | Constraint | Target | Why It Matters | Measurement Protocol |
|---|------------|--------|----------------|---------------------|
| **#1** | **QUALITY** | ≥4.0/5.0 HPSv2 score | Professional stock photo grade — clients reject anything below Unsplash/Pexels quality. Industry standard: 4.0-4.3 for paid stock libraries | Run HPSv2 aesthetic predictor on 100-image sample. Blind review by 3 designers (accept/reject). Field test: client approval rate >90% |
| **#2** | **SPEED** | <30 seconds per 512×512 image | Real-time client iteration during calls. Midjourney: ~15s. Waiting 2+ minutes kills momentum in creative reviews | Median generation time across 50 prompts on RTX 3060. Include model load time (cold start) |
| **#3** | **COST** | <$5,000 hardware, $0/month cloud | Bootstrapped startup — no budget for A100 rentals or API subscriptions. Must pay back investment in <3 months | One-time: RTX 3060 + RAM upgrade. Recurring: $0 (local inference only). ROI: $600k savings ÷ $5k = 2.5 months |
| **#4** | **CONTROL** | <5% unusable generations | Wasted iterations burn time and frustrate designers. CFG + ControlNet must reliably hit compositional requirements | Track "regeneration needed" rate over 200 diverse prompts (portraits, products, landscapes). "Unusable" = fails composition, anatomy, or brand guidelines |
| **#5** | **THROUGHPUT** | 100+ assets/day sustained | Agency baseline: 10 assets/day (2 designers × 5 assets each). Need 10× capacity to replace 3 departed freelancers | 8-hour workday simulation: automated batch pipeline generating hero images, variations, social crops. Measure total accepted assets |
| **#6** | **VERSATILITY** | 4 modalities working together | Real campaigns need text→image (hero shots), image→video (product demos), image→caption (alt text), text→speech (voiceovers) | Build end-to-end campaign: text prompt → hero image → 3s product video → image caption → voiceover. All steps <5 min total |

---

### Progressive Capability Unlock

| Ch | Title | What Unlocks | Constraints | Status |
|----|-------|--------------|-------------|--------|
| **1** | [Multimodal Foundations](ch01_multimodal_foundations/multimodal-foundations.md) | Understand tensors, modality gap, why raw pixels don't work | Foundation knowledge | Concepts only |
| **2** | [Vision Transformers](ch02_vision_transformers/vision-transformers.md) | Patch embeddings, ViT architecture, how images become tokens | Image representation | No generation yet |
| **3** | [CLIP](ch03_clip/clip.md) | **Text-image alignment, semantic search, zero-shot classification** | Foundation for #6 | Retrieval only |
| **4** | [Diffusion Models](ch04_diffusion_models/diffusion-models.md) | **DDPM generation, forward/reverse process, denoising** | **#1 Partial (quality), #4 Partial (control)** | MNIST: 28×28, slow |
| **5** | [Schedulers](ch05_schedulers/schedulers.md) | **DDIM, DPM-Solver — 4-step generation instead of 1,000** | **#2 <30s target viable!** | 10× faster sampling |
| **6** | [Latent Diffusion](ch06_latent_diffusion/latent-diffusion.md) | **VAE compression, Stable Diffusion architecture, 512×512 generation** | **#1 4.2 HPSv2! #3 $5k HW!** | SD runs locally! |
| **7** | [Guidance & Conditioning](ch07_guidance_conditioning/guidance-conditioning.md) | **CFG, negative prompts, semantic steering** | **#4 <3% unusable!** | Composition control |
| **8** | [Text-to-Image](ch08_text_to_image/text-to-image.md) | **ControlNet, img2img, inpainting, prompt engineering** | **#4 refined, #5 Partial** | Production pipeline |
| **9** | [Text-to-Video](ch09_text_to_video/text-to-video.md) | **Temporal consistency, AnimateDiff, video diffusion** | **#6 Partial (modality 2/4)** | 3s clips locally |
| **10** | [Multimodal LLMs](ch10_multimodal_llms/multimodal-llms.md) | **LLaVA, image captioning, VQA, projection layer** | **#6 Partial (modality 3/4)** | Alt text generation |
| **11** | [Audio Generation](ch11_audio_generation/README.md) | **MMS TTS, text-to-speech, waveform synthesis** | **#6 All 4 modalities!** | Voiceover complete |
| **12** | [Generative Evaluation](ch12_generative_evaluation/generative-evaluation.md) | **HPSv2, FID, CLIP score, human preference alignment** | **#1 measurement ** | Quality validation |
| **13** | [Local Diffusion Lab](ch13_local_diffusion_lab/local-diffusion-lab.md) | **End-to-end capstone: DDPM from scratch + local SD pipeline** | **#5 120 assets/day!** | **ALL CONSTRAINTS MET!** |

---

### Narrative Arc: From Theory to Production Pipeline

#### Act 1: Foundation (Ch.1-3)
**Understand the representation problem**

- **Ch.1**: Why raw pixels fail → Modality gap, tensor representation
- **Ch.2**: Vision Transformers → Patch embeddings, attention over images
- **Ch.3**: CLIP → Text-image alignment, semantic space

**Status**: No generation capability yet. Only retrieval and classification.

---

#### Act 2: Generation Breakthrough (Ch.4-6)
**Build the core text-to-image pipeline**

- **Ch.4**: Diffusion Models → **#1 Partial, #4 Partial** (DDPM works but slow, 28×28 only)
- **Ch.5**: Schedulers → **#2 ** 4-step DDIM makes <30s viable!
- **Ch.6**: Latent Diffusion → **#1 #3 ** HPSv2 4.2, SD runs on $5k hardware!

**Status**: Quality + Speed + Cost achieved! 512×512 images in 25s.

---

#### Act 3: Control & Composition (Ch.7-8)
**Eliminate unusable generations, build production workflow**

- **Ch.7**: Guidance & Conditioning → **#4 ** CFG + negative prompts = <3% unusable!
- **Ch.8**: Text-to-Image → ControlNet, img2img, inpainting, prompt engineering

**Status**: Core image pipeline production-ready!

---

#### Act 4: Multimodal Expansion (Ch.9-11)
**Add video, captioning, audio for full campaign creation**

- **Ch.9**: Text-to-Video → AnimateDiff, temporal consistency (modality 2/4)
- **Ch.10**: Multimodal LLMs → LLaVA captioning (modality 3/4)
- **Ch.11**: Audio Generation → **#6 ** MMS TTS voiceovers (modality 4/4)!

**Status**: All modalities working! Throughput optimization remains.

---

#### Act 5: Validation & Scale (Ch.12-13)
**Measure quality rigorously, prove throughput at scale**

- **Ch.12**: Generative Evaluation → HPSv2, FID, CLIP score, blind review
- **Ch.13**: Local Diffusion Lab → **#5 ** Batch pipeline achieves 120 assets/day!

**Status**: **ALL 6 CONSTRAINTS SATISFIED — PRODUCTION READY!**

---

### The Business Context: Aperture Creative Agency

**Current baseline (before budget cuts):**
- 5 in-house designers + 3 freelancers (avg. $75k/yr each)
- Freelancer annual cost: 3 × $200k = **$600,000/year**
- Output: 2,000 assets/year (product shots, social ads, explainer videos, voiceovers)
- Quality: Stock photo grade (Unsplash/Pexels equivalent)
- Turnaround: 2-3 days per asset batch

**After 60% budget cut:**
- Lost 3 freelancers = lost 750 assets/year capacity
- 2 remaining designers = 1,250 assets/year (shortfall of 750)
- Client commitments unchanged = existential crisis

**Creative Director's challenge:** "We can't afford Midjourney Enterprise ($60k/yr) or Runway ($12k/yr) or ComfyUI cloud rentals. Either you build me a local AI pipeline that matches freelancer quality on hardware I can expense in one purchase order — or I'm outsourcing to agencies in lower-cost markets and we're done."

**Your progression:**

```
After Ch.1-3: "Okay, I understand the theory. But can we actually generate images?"
After Ch.4: "DDPM works... but 28×28 is unusable. And 1,000 steps takes 4 minutes."
After Ch.5: "4-step DDIM is 10× faster! But MNIST isn't stock photos."
After Ch.6: "Wait... Stable Diffusion runs locally? In 25 seconds? And it looks professional?!"
After Ch.7: "CFG tuning dropped unusable gens from 22% to 3%. This is getting real."
After Ch.8: "ControlNet means I can give designers compositional control. They're believers now."
After Ch.9-11: "We have video, captions, and voiceovers. Full campaign creation in one pipeline."
After Ch.12: "HPSv2 4.2 = matches paid stock libraries. Blind test: clients can't tell the difference."
After Ch.13: "Batch pipeline churns out 120 assets/day. Payback in 2.5 months. This saved the agency."
```

**Financial impact:**
- Hardware investment: **$5,000** (RTX 3060 12GB, RAM upgrade, storage)
- Annual cloud fees: **$0** (fully local inference)
- Freelancer savings: **$600,000/year**
- **Payback period: 2.5 months**
- Year 1 ROI: **11,900%**

---

### What You'll Build

By the end of this track, you'll have:

1. **Vision understanding** of patch embeddings, ViT architecture, modality gap (Foundation)
2. **CLIP alignment** enabling semantic search and text-image retrieval (Constraint #6 foundation)
3. **Diffusion mastery** from DDPM forward/reverse process to latent diffusion (Constraints #1, #2, #3)
4. **Control systems** via CFG, negative prompts, ControlNet for composition (Constraint #4)
5. **Multi-modal pipeline** integrating text→image, image→video, image→caption, text→speech (Constraint #6)
6. **Evaluation rigor** using HPSv2, FID, CLIP score, human preference (Constraint #1 validation)
7. **Production deployment** with batch processing, quality gates, throughput optimization (Constraint #5)
8. **Deep understanding** of when to use pixel-space vs latent-space diffusion, DDPM vs DDIM vs DPM-Solver, and how to ship generative AI on constrained hardware

---

### Measurement Protocols (How Each Constraint Is Validated)

#### Constraint #1: Quality (≥4.0/5.0 HPSv2)
**Automated**: Run [HPSv2 aesthetic predictor](https://github.com/tgxs002/HPSv2) (human preference model trained on 430k comparisons) on 100-image sample. Must average ≥4.0/5.0.

**Human validation**: Blind review by 3 professional designers. Each rates 50 images (accept/reject for client use). Acceptance rate must exceed 90%.

**Field test**: Deploy to real client projects for 2 weeks. Track client approval rate (first draft accepted without revision). Target: >90%.

#### Constraint #2: Speed (<30s per 512×512 image)
**Benchmark**: Generate 50 diverse prompts (portraits, products, landscapes, abstract) on target hardware (RTX 3060 12GB). Measure:
- Median generation time (must be <30s)
- p95 generation time (must be <45s)
- Cold start overhead (model loading, first inference)

**Real-world test**: Designer iteration session. Measure time from "I want to change X" to new image on screen. Target: <40s including prompt modification.

#### Constraint #3: Cost (<$5k hardware, $0/month cloud)
**Hardware audit**:
- GPU: RTX 3060 12GB (~$350-400)
- RAM: 32GB DDR4 (~$80)
- Storage: 2TB NVMe (~$120)
- Workstation upgrade budget: ~$4,500 remaining

**Recurring costs**:
- Cloud API fees: $0 (local inference only)
- Electricity: ~$15/month (negligible vs $600k freelancer savings)

**ROI calculation**: $600k annual savings ÷ $5k investment = **2.5-month payback period**.

#### Constraint #4: Control (<5% unusable)
**Test set**: 200 diverse prompts across:
- Portraits (lighting, pose, expression)
- Products (angle, background, shadows)
- Landscapes (composition, time of day)
- Abstract (color palette, mood)

**Unusable criteria**: Generation fails if it violates:
- Composition guidelines (rule of thirds, focal point)
- Anatomy errors (extra fingers, distorted faces)
- Brand guidelines (wrong colors, mood, style)

**Measurement**: Designer review. Tag each generation as "accepted", "needs minor edit", or "unusable". <5% must be "unusable" (i.e., regeneration required).

#### Constraint #5: Throughput (100+ assets/day)
**8-hour workday simulation**:
1. Automated batch pipeline generates 150 hero images (50 prompts × 3 variations each)
2. Designer reviews and accepts best candidates (target: 2 min per prompt batch)
3. Generate social media crops (3 aspect ratios per accepted image)
4. Track total accepted assets ready for client delivery

**Success**: ≥100 client-ready assets in 8 hours (designer + AI collaboration). Breakdown:
- 40 hero images (50 prompts → 40 accepted after review)
- 120 social crops (40 images × 3 aspect ratios)
- **Total: 160 assets/day** (exceeds 100 target)

#### Constraint #6: Versatility (4 modalities)
**End-to-end campaign test**: Build complete product launch in <5 minutes total:
1. **Text → Image**: "A sleek smartwatch on a minimalist desk, golden hour lighting" → 512×512 hero image (25s)
2. **Image → Video**: Animate hero image with subtle parallax (3s clip, 60s generation)
3. **Image → Caption**: LLaVA generates alt text: "Modern smartwatch with black band displayed on white desk surface, bathed in warm natural light from the right" (5s)
4. **Text → Speech**: MMS TTS voiceover: "Introducing the Apex Watch — where design meets precision" (10s)

**Total pipeline time**: <2 minutes. **Success**: All 4 modalities produce professional-grade output without manual intervention.

---

## The Running Example — PixelSmith

Every note in this track is anchored to a single growing system: **PixelSmith**, the local AI-powered creative studio you are building from scratch to power **VisualForge Studio**.

Think of PixelSmith as the technical implementation behind VisualForge Studio's business mission. While VisualForge Studio is the production pipeline serving Aperture Creative's clients, PixelSmith is the educational artifact you build chapter-by-chapter to understand every component: from raw pixel tensors through patch embeddings, CLIP alignment, diffusion denoising, latent-space compression, and finally multimodal integration.

```
PixelSmith v1 (after Foundations):
 Input: raw image file → Output: pixel tensor, patch embeddings
 VisualForge capability: None yet (understanding representation)

PixelSmith v2 (after CLIP):
 Input: text query → Output: ranked images by semantic similarity
 VisualForge capability: Asset library semantic search

PixelSmith v3 (after Diffusion Models):
 Input: noise → Output: generated image (DDPM from scratch)
 VisualForge capability: Generation proof-of-concept (MNIST only)

PixelSmith v4 (after Latent Diffusion):
 Input: text prompt → Output: generated image (Stable Diffusion, locally)
 VisualForge capability: CORE PIPELINE — professional 512×512 images in 25s

PixelSmith v5 (after ControlNet / img2img):
 Input: text prompt + sketch → Output: controlled generated image
 VisualForge capability: Compositional control for designer workflows

PixelSmith v6 (after Multimodal LLMs + Audio):
 Input: photograph + question → Output: natural language answer
 Input: text script → Output: voiceover waveform
 VisualForge capability: Image captioning + TTS for complete campaigns
```

The key constraint: **PixelSmith must run on a stock developer laptop** — no A100, no cloud GPU budget. This forces every chapter to confront the same question production engineers face: *how do you get serious generative AI to run where you actually are?*

By Ch.13, PixelSmith v6 becomes the foundation for VisualForge Studio's production pipeline: batch processing, quality gates, throughput optimization, and multi-modal asset generation — all validated against the 6 constraints that determine if Aperture Creative survives.

---

## How We Got Here — A Short History of Multimodal & Generative AI

Image generation went from "blurry 32×32 digits" to "photoreal 4K video" in about a decade. **The detailed timeline now lives in each chapter's own prelude** — every chapter opens with a *"The story"* blockquote that names the papers, dates, and dramatic tensions behind that breakthrough.

**The through-line in one paragraph.** Each chapter corresponds to a specific obstacle that was removed. CNNs dominated vision until [ViT](ch02_vision_transformers) (Dosovitskiy et al., Oct 2020) showed patches + attention scaled better. Text and images lived in separate spaces until [CLIP](ch03_clip) (Radford et al., Jan 2021) aligned them with 400M web pairs. GANs (Goodfellow, 2014) were unstable until [DDPM](ch04_diffusion_models) (Ho et al., Jun 2020) replaced them with stable denoising. Pixel-space diffusion was too expensive until [LatentDiffusion / Stable Diffusion](ch06_latent_diffusion) (Rombach et al., Aug 2022) moved to VAE latents and went open-source. Sampling was slow until [DDIM](ch05_schedulers) (Song et al., Oct 2020) and DPM-Solver (Lu et al., 2022). Models were uncontrollable until [classifier-free guidance](ch07_guidance_conditioning) (Ho & Salimans, Jul 2022) and [ControlNet](ch08_text_to_image) (Zhang, Feb 2023). Images weren't enough → [MLLMs](ch10_multimodal_llms) (BLIP-2 Jan 2023, GPT-4V Sep 2023) and [video](ch09_text_to_video) (AnimateDiff 2023, Sora Feb 2024). Cloud GPUs were too expensive → [LocalDiffusionLab](ch13_local_diffusion_lab) via quantisation + LCM distillation. As quality saturated, [evaluation](ch12_generative_evaluation) shifted from FID (Heusel 2017) to human preference (HPSv2, 2023).

---

## Popular and Powerful Models by Modality (Apr 21, 2026 snapshot)

This is a practical field snapshot (not a leaderboard). It highlights model families most commonly cited for quality and adoption.

| Modality | Popular and powerful model families |
|----------|-------------------------------------|
| Text-to-Image | Midjourney latest, FLUX.1 family (BFL), SDXL/SD3.x ecosystem, Imagen 3 |
| Image Editing / Control | ControlNet-style pipelines, FLUX control/edit variants, SD inpainting stacks |
| Text-to-Video | Sora, Veo 2, Runway latest Gen family, Kling latest, open stacks such as HunyuanVideo and CogVideoX |
| Vision-Language (MLLM) | GPT-4o family, Gemini 2.x family, Claude 3.7+ vision, Qwen2.5-VL, Pixtral, Llama vision variants |
| Speech-to-Text (ASR) | Whisper large-v3 family, distil-whisper variants, Canary family |
| Text-to-Speech (TTS) | ElevenLabs latest voices, gpt-4o-mini-tts, Coqui XTTS v2, MMS-TTS, Kokoro-82M |
| Music / SFX Generation | Suno latest, Udio latest, Stable Audio family, MusicGen family |
| Cross-modal Embeddings / Retrieval | CLIP descendants, SigLIP family, EVA-CLIP style vision-text encoders |

### Local CPU quick-win picks (recommended)

If your goal is "few lines of code + visible output on stock hardware":

1. Text-to-speech: `facebook/mms-tts-eng` or `Kokoro-82M`
2. Speech-to-text: `distil-whisper` or smaller `Whisper` checkpoints
3. Image retrieval demo: CLIP/SigLIP embeddings on a small local image set

For this reason, the new [AudioGeneration](ch11_audio_generation/README.md) chapter uses MMS TTS as the first demonstration.

---

## GPU-Only Supplement Notebooks

Each MultimodalAI chapter now includes a dedicated GPU-only notebook named `notebook_supplement.ipynb_solution.ipynb` (reference) or `notebook_supplement.ipynb_exercise.ipynb` (practice).

- Purpose: keep optional GPU-first experiments separate from the main chapter walkthrough.
- Scope: one `notebook_supplement.ipynb_solution.ipynb` (reference) or `notebook_supplement.ipynb_exercise.ipynb` (practice) per chapter under `notes/MultimodalAI/<Chapter>/`.
- Safety guard: every `notebook_supplement.ipynb_solution.ipynb` (reference) or `notebook_supplement.ipynb_exercise.ipynb` (practice) checks for GPU availability in the first code cell and exits early with a clear message if no compatible GPU is detected.

### Requirements

- Hardware: NVIDIA GPU with drivers installed.
- Runtime: Python kernel with either:
 - `torch` with CUDA support, or
 - `nvidia-smi` available on `PATH` for fallback detection.

If neither check confirms a GPU, the notebook stops immediately by design.

---

## The Conceptual Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MULTIMODAL AI SYSTEM │
│ │
│ ┌────────────────────────────────────────────────────────────────────────┐ │
│ │ GENERATION LAYER │ │
│ │ │ │
│ │ Diffusion Models · Latent Diffusion · Schedulers │ │
│ │ Text-to-Image · Text-to-Video · ControlNet │ │
│ │ [DiffusionModels.md] [LatentDiffusion.md] [TextToImage.md] │ │
│ └─────────────────────────────┬──────────────────────────────────────────┘ │
│ │ │
│ ┌──────────────────────┴───────────────────┐ │
│ │ │ │
│ ┌───────▼───────────────────┐ ┌────────────────▼───────────────────┐ │
│ │ ALIGNMENT LAYER │ │ UNDERSTANDING LAYER │ │
│ │ │ │ │ │
│ │ How language and vision │ │ How models answer questions │ │
│ │ share the same space │ │ about images and video │ │
│ │ │ │ │ │
│ │ [CLIP.md] │ │ [MultimodalLLMs.md] │ │
│ │ [GuidanceConditioning.md] │ │ [TextToVideo.md] │ │
│ └───────────────────────────┘ └────────────────────────────────────┘ │
│ │
│ ┌────────────────────────────────────────────────────────────────────────┐ │
│ │ REPRESENTATION LAYER │ │
│ │ │ │
│ │ How raw signals become tokens the model can process │ │
│ │ [MultimodalFoundations.md] [VisionTransformers.md] │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Document Map

### Foundation Notes (read before the core notes)

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [MultimodalFoundations.md](ch01_multimodal_foundations/multimodal-foundations.md) | What multimodal AI is; how images, audio, and video become tensors; the representation problem; the modality gap | How does a photograph become a matrix? Why can't we just feed pixels to a transformer? What is a modality gap? |
| [VisionTransformers.md](ch02_vision_transformers/vision-transformers.md) | ViT: splitting images into patches, positional encoding, self-attention over patches; how ViT differs from CNNs and why it won | What is a patch embedding? How does ViT handle position? Why did attention beat convolution at scale? |

### Alignment Notes

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [CLIP.md](ch03_clip/clip.md) | Contrastive Language-Image Pretraining; dual-encoder architecture; InfoNCE loss; zero-shot classification without any labelled data | How does a model learn that a photo of a cat matches the text "a cat"? What is contrastive loss? What is zero-shot transfer? |
| [GuidanceConditioning.md](ch07_guidance_conditioning/guidance-conditioning.md) | Classifier guidance, classifier-free guidance (CFG), text conditioning via cross-attention; what the guidance scale actually does; negative prompts | Why does guidance scale 7.5 produce sharper images than 1.0? What does a negative prompt actually do mechanically? |

### Generation Core Notes

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [DiffusionModels.md](ch04_diffusion_models/diffusion-models.md) | The math of DDPM: the forward noising process, the reverse denoising process, score matching, noise schedules; why diffusion beat GANs | What is the forward process? What does the U-Net actually predict — the image or the noise? Why is diffusion more stable than GAN training? |
| [Schedulers.md](ch05_schedulers/schedulers.md) | DDPM vs DDIM vs DPM-Solver; how to generate in 4 steps instead of 1,000; deterministic sampling; the speed/quality trade-off | Why does DDIM need fewer steps? What changes when you switch from DDPM to DPM-Solver? What is a sampler doing geometrically? |
| [LatentDiffusion.md](ch06_latent_diffusion/latent-diffusion.md) | Why pixel-space diffusion is too slow; VAEs as a compression layer; the Stable Diffusion architecture (VAE + U-Net + CLIP text encoder); latent space geometry | What is a VAE? Why run diffusion in latent space instead of pixel space? How does text reach the U-Net in Stable Diffusion? |

### Application Notes

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [TextToImage.md](ch08_text_to_image/text-to-image.md) | End-to-end text-to-image pipeline; prompt engineering for images; img2img; inpainting; ControlNet for structural conditioning | How does prompt weight syntax work? What is ControlNet's conditioning signal? How does inpainting avoid repainting the whole image? |
| [TextToVideo.md](ch09_text_to_video/text-to-video.md) | Extending diffusion to the temporal dimension; the consistency problem; overview of video generation (Sora, CogVideo, AnimateDiff) | What makes video harder than images? How does Sora model spacetime? What is AnimateDiff doing differently from full video models? |
| [MultimodalLLMs.md](ch10_multimodal_llms/multimodal-llms.md) | Connecting vision encoders to LLM decoders; LLaVA, BLIP-2, GPT-4V, Gemini; visual instruction tuning; the projection layer | How does GPT-4V "see"? What is a Q-Former? How do you fine-tune an LLM to accept image tokens? |
| [AudioGeneration](ch11_audio_generation/README.md) | CPU-first text-to-speech quick win using a compact pretrained model and a minimal notebook flow | How can you generate speech locally without a GPU? What is the shortest path from text prompt to playable waveform? |

### Evaluation Note

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [GenerativeEvaluation.md](ch12_generative_evaluation/generative-evaluation.md) | How do you measure the quality of a generated image or video? FID, IS, CLIP score, LPIPS, human preference models; the alignment problem in evaluation | What does FID actually measure? Why is CLIP score better for text-image alignment than FID? Why is human evaluation still the gold standard? |

### Capstone

| File | Purpose |
|------|---------|
| [LocalDiffusionLab.md](ch13_local_diffusion_lab/local-diffusion-lab.md) | Hands-on capstone: train a DDPM from scratch on MNIST (runs in ~5 minutes on CPU), visualise every step of the diffusion process, then run Stable Diffusion locally using `diffusers` + a turbo/LCM checkpoint. Step-by-step output viewable in the notebook. |

---

## Reading Paths

### Path A — "I just want to understand how Stable Diffusion works"

```
MultimodalFoundations → VisionTransformers → CLIP → DiffusionModels → LatentDiffusion
```

### Path B — "I want to run image generation locally from scratch"

```
MultimodalFoundations → DiffusionModels → Schedulers → LatentDiffusion → LocalDiffusionLab
```

### Path C — "I'm already familiar with diffusion, I want to extend to video and multimodal LLMs"

```
CLIP → GuidanceConditioning → TextToImage → TextToVideo → MultimodalLLMs
```

### Path D — "I need to evaluate model outputs for a project"

```
GenerativeEvaluation (read alone — it is largely self-contained)
```

### Path E — "I want a fast CPU-only multimodal demo"

```
AudioGeneration (standalone quick win)
```

### Full Sequential Path (recommended)

```
MultimodalFoundations
 └─▶ VisionTransformers
 └─▶ CLIP
 └─▶ DiffusionModels
 └─▶ GuidanceConditioning
 └─▶ Schedulers
 └─▶ LatentDiffusion
 └─▶ TextToImage
 └─▶ TextToVideo
 └─▶ MultimodalLLMs
 └─▶ GenerativeEvaluation
 └─▶ LocalDiffusionLab (capstone)
```

### By VisualForge Constraint

**Need to hit quality targets?**
→ [Ch.4 Diffusion Models](ch04_diffusion_models/diffusion-models.md), [Ch.6 Latent Diffusion](ch06_latent_diffusion/latent-diffusion.md), [Ch.12 Generative Evaluation](ch12_generative_evaluation/generative-evaluation.md) **(Constraint #1)**

**Need faster generation?**
→ [Ch.5 Schedulers](ch05_schedulers/schedulers.md), [Ch.6 Latent Diffusion](ch06_latent_diffusion/latent-diffusion.md) **(Constraint #2)**

**Budget-constrained hardware?**
→ [Ch.6 Latent Diffusion](ch06_latent_diffusion/latent-diffusion.md), [Ch.13 Local Diffusion Lab](ch13_local_diffusion_lab/local-diffusion-lab.md) **(Constraint #3)**

**Need compositional control?**
→ [Ch.7 Guidance & Conditioning](ch07_guidance_conditioning/guidance-conditioning.md), [Ch.8 Text-to-Image](ch08_text_to_image/text-to-image.md) **(Constraint #4)**

**Need batch production throughput?**
→ [Ch.8 Text-to-Image](ch08_text_to_image/text-to-image.md), [Ch.13 Local Diffusion Lab](ch13_local_diffusion_lab/local-diffusion-lab.md) **(Constraint #5)**

**Need multi-modal capabilities?**
→ [Ch.3 CLIP](ch03_clip/clip.md), [Ch.9 Text-to-Video](ch09_text_to_video/text-to-video.md), [Ch.10 Multimodal LLMs](ch10_multimodal_llms/multimodal-llms.md), [Ch.11 Audio Generation](ch11_audio_generation/README.md) **(Constraint #6)**

---

## How This Track Connects to the AI Track and ML Track

| Concept from this track | Prerequisite from other tracks |
|------------------------|-------------------------------|
| Patch embeddings in ViT | Transformer architecture → [ML Ch.18 — Transformers](../ml/03_neural_networks/ch10_transformers/README.md) |
| InfoNCE contrastive loss | Embedding training objectives → [RAGAndEmbeddings.md](.03-ai/ch04_rag_and_embeddings/rag-and-embeddings.md) |
| CLIP text encoder inside Stable Diffusion | Tokenisation + transformer encoder → [LLMFundamentals.md](.03-ai/ch01_llm_fundamentals/llm-fundamentals.md) |
| CFG conditioning via cross-attention | Attention mechanics → [ML Ch.18 — Transformers](../ml/03_neural_networks/ch10_transformers/README.md) |
| VAE (encoder-decoder architecture) | Neural network layers + backprop → [ML Ch.4](../ml/03_neural_networks/ch02_neural_networks/README.md) + [ML Ch.5](../ml/03_neural_networks/ch03_backprop_optimisers/README.md) |
| Fine-tuning LLaVA on visual instructions | Fine-tuning concepts → [FineTuning.md](.03-ai/ch10_fine_tuning/fine-tuning.md) |
| FID as a distribution-level metric | Evaluation concepts → [EvaluatingAISystems.md](.03-ai/ch08_evaluating_ai_systems/evaluating-ai-systems.md) |

---

## The Build Plan

> This section tracks the chapter-by-chapter build of the Multimodal AI notes library. Each chapter lives under `notes/MultimodalAI/` in its own folder, containing a `.md` note and a Jupyter notebook. The running example (PixelSmith) progresses through each chapter.

Animation rollout tracker:

- [ANIMATION_PLAN.md](animation-plan.md) - data-flow animation coverage and chapter closeout status

### Chapter Structure

Every note follows this template (same order as the ML and AI tracks):

```
# [Topic] — [Subtitle]

> Blockquote: what you'll understand after reading this

## 0 · Core Idea ← 2–4 sentences, plain English
## 1 · Running Example ← how PixelSmith uses this concept
## 2 · The Math ← key equations, every symbol annotated
## 3 · How It Works — Step by Step ← numbered steps or flow diagram
## 4 · The Key Diagrams ← Mermaid / ASCII diagrams
## 5 · What Changes at Scale ← how this works in production systems
## 6 · Common Misconceptions ← what people get wrong
## 7 · Interview Checklist ← Must Know / Likely Asked / Trap to Avoid
## 8 · What's Next ← forward pointer to the next note
```

### Chapter Status

| # | Chapter | Folder | Status |
|---|---------|--------|--------|
| 1 | Multimodal Foundations | `ch01_multimodal_foundations/` | Complete |
| 2 | Vision Transformers | `ch02_vision_transformers/` | Complete |
| 3 | CLIP | `ch03_clip/` | Complete |
| 4 | Diffusion Models | `ch04_diffusion_models/` | Complete |
| 5 | Guidance & Conditioning | `ch07_guidance_conditioning/` | Complete |
| 6 | Schedulers | `ch05_schedulers/` | Complete |
| 7 | Latent Diffusion | `ch06_latent_diffusion/` | Complete |
| 8 | Text-to-Image | `ch08_text_to_image/` | Complete |
| 9 | Text-to-Video | `ch09_text_to_video/` | Complete |
| 10 | Multimodal LLMs | `ch10_multimodal_llms/` | Complete |
| 11 | Generative Evaluation | `ch12_generative_evaluation/` | Complete |
| 12 | Local Diffusion Lab (capstone) | `ch13_local_diffusion_lab/` | Complete |
| 13 | Audio Generation (CPU quick win) | `ch11_audio_generation/` | Complete |

---

## Hardware Expectations

> No chapter in this track requires a GPU. Every notebook runs on a stock developer laptop.

| Chapter | What runs locally | Typical time on CPU |
|---------|------------------|---------------------|
| 1–3 (Foundations, ViT, CLIP) | Numpy / PyTorch tensor operations | < 30 seconds |
| 4–6 (Diffusion, Guidance, Schedulers) | DDPM training on MNIST | < 5 minutes |
| 7–8 (Latent Diffusion, Text-to-Image) | SDXL-Turbo or LCM via `diffusers` | 30–90 seconds / image |
| 9 (Text-to-Video) | Theory + inspection of pretrained models | No local generation |
| 10 (Multimodal LLMs) | LLaVA-1.5-7B via `ollama` or `llama.cpp` | 10–60 seconds / response |
| 11 (Evaluation) | FID / CLIP score on pre-generated samples | < 2 minutes |
| 12 (Local Diffusion Lab) | Full pipeline: scratch DDPM + local SD | < 10 minutes total |
| 13 (Audio Generation) | MMS TTS inference via `transformers` on CPU | 5–20 seconds / sample |

---

## The PixelSmith Running Example — Master Design

### What PixelSmith Is

PixelSmith is the **educational artifact** — a minimal, from-scratch recreation of the core inference pipeline you'll use to build **VisualForge Studio** (the production challenge). Think of PixelSmith as your learning sandbox and VisualForge as the business application. As you work through the notes, PixelSmith grows from simple tensor operations to a complete multi-modal generation pipeline. At the end of the track, the notebook in `LocalDiffusionLab/` ties every component together into a runnable end-to-end system that demonstrates all 6 VisualForge constraints in action.

### Why This Running Example Works

| Property | Why it matters | VisualForge connection |
|----------|---------------|------------------------|
| Fully local | You never need an API key or a cloud account — every chapter runs offline | **Constraint #3**: Zero cloud costs, $5k hardware budget |
| Progressive | Each chapter adds exactly one new concept to the same growing system | Maps directly to 13-chapter capability unlock table |
| Grounded | Every abstraction (patch embeddings, latent space, noise schedule) is demonstrated with real tensor operations you can inspect | Understand *why* it works, not just *how* to use the API |
| Honest about constraints | A stock machine cannot run Stable Diffusion XL in full float32, so the notes explain *why*, not just *what* | **Constraint #2**: Speed optimization isn't optional — it's survival |

### What PixelSmith Is Not

PixelSmith is **not** a production image editor, a fine-tuning service, or a LoRA training pipeline. It is a teaching artefact — the simplest system that lets you verify you understand each concept with code you wrote or can read line-by-line.

**VisualForge Studio** is where you take that understanding and apply it to production constraints: batch processing, quality gates, client iteration workflows, and the business metrics that determine if the agency survives the budget cuts.

---

## Prerequisite Check

Before starting Chapter 1, you should be comfortable with:

| Prerequisite | Where to build it if needed |
|-------------|----------------------------|
| What a transformer is and how attention works | [ML Ch.17 — From Sequences to Attention](../ml/03_neural_networks/ch09_sequences_to_attention/README.md) then [ML Ch.18 — Transformers](../ml/03_neural_networks/ch10_transformers/README.md) |
| What embeddings are and why cosine similarity matters | [RAGAndEmbeddings.md](.03-ai/ch04_rag_and_embeddings/rag-and-embeddings.md) |
| Basic PyTorch tensor operations (`torch.Tensor`, `.view()`, matrix multiply) | [ML Ch.4 — Neural Networks](../ml/03_neural_networks/ch02_neural_networks/README.md) |
| What a convolutional layer does (for comparison with ViT) | [ML Ch.7 — CNNs](../ml/03_neural_networks/ch05_cnns/README.md) |

You do **not** need: prior experience with image generation, a GPU, or familiarity with `diffusers` / `huggingface_hub` before you start.
