# Text-to-Video — Adding the Temporal Dimension

> **The story.** Early text-to-video looked like wobbly GIFs — **Make-A-Video** (Meta, September 2022) and **Imagen Video** (Google, October 2022) were impressive *demonstrations* but produced 1–4 second clips with visible flicker. The first community-usable model was **AnimateDiff** (Guo et al., July **2023**), which inserted plug-in motion modules into existing Stable Diffusion checkpoints — video for free, on a consumer GPU. **Stable Video Diffusion** (Stability AI, November 2023) shipped the first open foundational video model. The watershed was **OpenAI Sora** (**February 2024**) — a transformer-based diffusion model trained on "spacetime patches" that produced 60-second high-fidelity clips and reset everyone's expectations of what was possible. **Runway Gen-3** (June 2024), **Luma Dream Machine** (June 2024), **Kling** (Kuaishou, June 2024), and **Pika 1.5** followed in months. The 2026 landscape is a market of foundation video models with steadily rising clip lengths and physics-awareness. Every time VisualForge generates a 15-second social media ad showing a product rotating 360°, this machinery runs 16–24 times to maintain frame-to-frame coherence.
>
> **Where you are in the curriculum.** You're the Lead ML Engineer at VisualForge Studio. Ch.8 Text-to-Image delivered static images at 18s per generation with 3% unusable rate. Now clients want **15-second video ads** — product demos, 360° rotations, zoom effects. Video is a sequence of frames: $(T, C, H, W)$. The three new challenges are temporal consistency (no flicker/teleportation), exploded compute cost (16 frames = 16× the work), and motion-quality evaluation. This chapter shows how temporal attention layers enforce frame-to-frame coherence and how AnimateDiff extends Stable Diffusion to the temporal dimension without retraining from scratch.

![Text-to-video flow animation](img/text-to-video-flow.gif)

*Flow: text conditioning and temporal modules co-drive latent frame denoising before decoding to a coherent short clip.*

---

## 0 · The VisualForge Studio Challenge

**Mission**: VisualForge needs text→image + image→video + image understanding (Constraint #6: 3 modalities).

**Current blocker at Chapter 9**: Static images = solved (Ch.8). But clients now want **15-second social media video ads** (product rotating 360°, zoom effects). Video generation = unsolved.

**What this chapter unlocks**: **Text-to-Video** — extend latent diffusion to temporal dimension. AnimateDiff adds temporal attention layers (attend across frames at same spatial position). Generate 16-frame 512×512 clips (~1 second at 15fps). Sora overview (spacetime patches).

---

### The 6 Constraints — Snapshot After Chapter 9

| Constraint | Target | Status | Evidence |
|------------|--------|--------|----------|
| #1 Quality | ≥4.0/5.0 | ⚡ **~3.8/5.0** | Video quality matches images, temporal consistency good |
| #2 Speed | <30 seconds | ✅ **~18s per image** | Video slower (10 clips/day) but acceptable |
| #3 Cost | <$5k hardware | ✅ **$2.5k laptop** | AnimateDiff runs on same hardware |
| #4 Control | <5% unusable | ✅ **~3% unusable** | ControlNet + temporal attention = consistent |
| #5 Throughput | 100+ images/day | ⚡ **~85 images/day** | 10 video clips/day added to workflow |
| #6 Versatility | 3 modalities | ⚡ **Text→Image + Video enabled** | Still need image understanding (auto-QA) |

---

### What's Still Blocking Us After This Chapter?

**QA bottleneck**: Every generated image/video requires **human QA** to verify it matches the brief. 100% manual review = bottleneck. Need automated verification.

**Next unlock (Ch.10)**: **Multimodal LLMs (VLMs)** — LLaVA can caption images, answer "Is the product centered?" Automate QA → remove bottleneck → unlock full 120+ images/day throughput.

---

## 1 · Core Idea

You're the Lead ML Engineer at VisualForge Studio. Static images (Ch.8) are production-ready, but clients now want **15-second video ads**. Video is a sequence of frames: $(T, C, H, W)$. Three new challenges:

### Challenge #1: Temporal Consistency

**The failure**: Generate 16 frames independently with Stable Diffusion (same prompt, different seeds) → massive **flicker, teleportation, morphing**. The dress changes color between frames, café furniture moves, the woman's face morphs.

**Why it fails**: Each frame is generated from independent noise. No cross-frame communication → no temporal consistency.

**The fix**: **Temporal attention layers** attend across frames at same spatial position $(h, w)$ → enforce that pixel colors, object positions, and lighting stay consistent frame-to-frame.

### Challenge #2: Computational Cost

**The problem**: Generating 16 frames at 512×512 is **16× the cost** of one image. At 18s per static image (Ch.8), naive 16-frame video = **4.8 minutes**. Unacceptable for client iteration.

**The optimization**: AnimateDiff generates all 16 frames **simultaneously** via 3D denoising, not sequentially. Shared spatial features across frames → **45 seconds total** (not 4.8 minutes).

### Challenge #3: Motion Modeling

**The need**: Natural motion trajectories (smooth camera dolly, realistic human movement, physics-aware object interactions).

**The solution**: Motion modules trained on video datasets (WebVid-10M) learn motion priors separately from spatial priors → plug into frozen SD checkpoints.

---

**Architecture strategy**: Extend latent diffusion by adding a **temporal axis**. Two approaches:

1. **Temporal attention layers** (AnimateDiff): Insert between spatial attention blocks → attend across frames at same $(h, w)$ position
2. **3D convolution inflation** (Video LDM): Inflate 2D Conv $(k \times k)$ → 3D Conv $(1 \times k \times k)$, fine-tune temporal kernel

AnimateDiff wins for VisualForge: freeze spatial SD layers (preserve photorealism), train only temporal modules (faster, cheaper).

## 2 · Running Example — VisualForge Product Demo Campaign

**Client brief**: "Generate a 15-second video ad for our spring collection — woman in floral dress spinning slowly in Parisian café, golden hour lighting, camera dolly zoom. Need 5 variations by tomorrow morning."

**Challenge**: Static image generation (Ch.8) delivers **one frame**. Generating 16 frames independently produces **flicker and teleportation** — the dress changes color, the café furniture moves, the woman's face morphs between frames. Unacceptable for client delivery.

**Solution preview**: AnimateDiff adds **temporal attention layers** between spatial attention blocks. At pixel position $(h, w)$, the model attends across all $T$ frames — enforcing that the dress, café, and lighting stay consistent frame-to-frame.

No local video generation (too expensive for CPU). Instead, we:
- Inspect a real AnimateDiff model's architecture via `diffusers` metadata
- Build a **temporal attention** module from scratch and show how it connects frames
- Visualise what "temporal consistency" looks like via frame-coherence plots on synthetic sequences
- Demonstrate the **failure case first**: frame-by-frame SD generation produces incoherent flicker
- Show the **fix**: temporal attention eliminates flicker

## 3 · The Math

### Video as 3D Tensor

An image batch: $(B, C, H, W)$

A video batch: $(B, T, C, H, W)$, where $T$ = number of frames (typically 8–24)

Alternatively flattened as $(B \cdot T, C, H, W)$ to reuse existing 2D spatial operations.

### Temporal Attention

The core idea in AnimateDiff (Guo et al. 2023): after each spatial self-attention layer, add a **temporal attention** layer that attends across frames at the *same spatial position*:

For spatial position $(h, w)$, stack the $T$ feature vectors:

$$\mathbf{F}_{hw} \in \mathbb{R}^{T \times d}$$

Run self-attention over the $T$-length sequence:

$$\text{TempAttn}(\mathbf{F}_{hw}) = \text{softmax} \left(\frac{\mathbf{F}_{hw}\mathbf{W}_Q (\mathbf{F}_{hw}\mathbf{W}_K)^\top}{\sqrt{d}}\right)\mathbf{F}_{hw}\mathbf{W}_V$$

This lets pixel $(h, w)$ at frame $t$ attend to the same pixel location at all other frames — enforcing coherence.

### 3D Convolution Inflation

An alternative approach (Video LDM, Chen et al. 2023): inflate the 2D spatial convolutions to 3D:

$$\text{Conv2D}(k \times k) \to \text{Conv3D}(1 \times k \times k)$$

then initialise the temporal kernel weight to be an identity (all frames equal), then fine-tune. This is a common trick to adapt pretrained image models to video with minimal new parameters.

### DDPM on Videos

The noise process extends naturally:

$$q(x_t^{1:T} | x_0^{1:T}) = \prod_{frame=1}^{T} \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0^{frame}, (1-\bar{\alpha}_t)\mathbf{I})$$

Noise is added independently per frame, but the denoiser must learn to share information across frames (via temporal attention).

### Optical Flow Guidance

Some models add an explicit **optical flow consistency loss** during fine-tuning:

$$\mathcal{L}_{\text{flow}} = \sum_{t=1}^{T-1} \|\hat{x}_0^{t+1} - \text{warp}(\hat{x}_0^t, \mathbf{f}_t)\|^2$$

where $\mathbf{f}_t$ is the estimated flow from frame $t$ to $t+1$.

---

## 4 · Visual Intuition — How Temporal Video Models Work

### AnimateDiff Architecture (Motion Module)

AnimateDiff trains only the **temporal attention layers** on a video dataset, with the spatial SD U-Net frozen:

1. Load a pretrained SD 1.5 (spatial layers frozen)
2. Insert temporal attention modules between existing spatial blocks
3. Train temporal modules on WebVid-10M (short web videos + captions)
4. At inference: use SD spatial layers (can swap in any LoRA/style checkpoint) + AnimateDiff temporal motion module

**Result**: style from the image checkpoint, motion from the motion module.

### Sora (OpenAI, 2024)

Sora uses a **Diffusion Transformer (DiT)** architecture extended to video:
- Input: video latent (VAE-encoded), patchified into 3D patches (spatial + temporal)
- Transformer processes spacetime patches with full 3D attention
- Scale: trained on variable-length, variable-resolution videos up to 1 minute

**Key innovation**: **spacetime patch** = a small cuboid of (frames, height, width), treated as a single token. The model can handle arbitrary resolutions and durations.

### CogVideo / CogVideoX

Tsinghua's open model. Uses a 3D U-Net with temporal attention. CogVideoX (2024) uses a larger DiT backbone and achieves state-of-the-art on T2V benchmarks. Fully open weights.

### Inference Memory Reduction

Generating 16 frames at 512×512 per step costs 16× an image. Optimisations:
- **Frame sliding window**: generate 8 overlapping frames at a time, stitch
- **Lower spatial resolution + temporal SR**: generate at 256×256, upscale to 512 with a video SR model
- **Temporal distillation**: LCM-style 4-step video generation

---

### Temporal Attention Visualization

```
How temporal attention connects frames:

 Frame 1: [f1_patch_1, f1_patch_2, ..., f1_patch_N] ← spatial attention (within frame)
 Frame 2: [f2_patch_1, f2_patch_2, ..., f2_patch_N]
 ...
 Frame T: [fT_patch_1, fT_patch_2, ..., fT_patch_N]

 Temporal attention at patch position i:
 [f1_patch_i, f2_patch_i, ..., fT_patch_i] ← attend across T frames
 ↕ (each frame attends to all others at this position)

 Result: patch_i in frame 3 knows what patch_i in frames 1,2,4..T looks like.
 → objects stay in place; motion is smooth.


SD → AnimateDiff extension:

 [Spatial ResBlock] → [Spatial Self-Attn] → [Temporal Attn ← NEW] → [Cross-Attn (text)]
 [Spatial ResBlock] → [Spatial Self-Attn] → [Temporal Attn ← NEW] → [Cross-Attn (text)]
 ...

 Spatial layers: frozen (from SD 1.5)
 Temporal layers: trained on video dataset
 Text conditioning: unchanged (same CLIP encoder)
```

---

**Model comparison — What changes at scale:**

| Model | Frames | Resolution | Architecture | Params | Open? |
|-------|--------|-----------|-------------|--------|-------|
| AnimateDiff | 16 | 512×512 | SD 1.5 + temporal attn | ~1.5B | Yes |
| ModelScope | 16 | 256×256 | 3D U-Net | ~1.7B | Yes |
| CogVideoX | 49 | 720p | 3D DiT | 5B | Yes |
| Sora | ~600 | up to 1080p | 3D DiT | ~20B? | No |
| Kling | 300 | 1080p | DiT | ~20B? | No |
| Wan2.1 | 81 | 720p | 3D DiT | 14B | Yes |

**Key insight**: DiT (Diffusion Transformer) has replaced U-Net as the dominant architecture above 5B params. AnimateDiff (U-Net based) is the sweet spot for VisualForge: open weights, 16 frames, fits on RTX 4090.

---

## 5 · Production Example — VisualForge in Action

**Scenario**: Client needs 5 variations of 15-second product demo (spring dress, 360° spin, café setting, golden hour).

**VisualForge pipeline**:

```python
# VisualForge: Text-to-Video generation with AnimateDiff
from diffusers import AnimateDiffPipeline, MotionAdapter, DPMSolverMultistepScheduler
import torch

# Load motion adapter (temporal attention modules)
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5")

# Load SD 1.5 base + LoRA for VisualForge style
pipe = AnimateDiffPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    motion_adapter=adapter,
    torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_slicing()  # Reduce VRAM usage

# VisualForge client brief — Product Demo campaign type
brief = """Woman in floral spring dress spinning slowly in Parisian café,
golden hour lighting, natural window light, professional photography,
cinematic camera movement, 4k quality"""

negative_prompt = "blurry, low quality, distorted, morphing, flicker, inconsistent lighting"

# Generate 16-frame clip (~1 second at 15fps)
video_frames = pipe(
    prompt=brief,
    negative_prompt=negative_prompt,
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cuda").manual_seed(42)
).frames[0]

# Export to MP4 for client review
import imageio
imageio.mimsave("visualforge_spring_dress_demo.mp4", video_frames, fps=15)
```

**Metrics**:
- **Generation time**: ~45 seconds for 16-frame clip (RTX 4090)
- **Temporal consistency**: No visible flicker or morphing between frames
- **Prompt adherence**: Dress color, café setting, lighting remain consistent
- **Client approval rate**: 4 of 5 variations approved on first iteration (80% success)

**Constraint impact**:
- **#5 Throughput**: ⚡ **~85 images/day + 10 video clips/day** — Video capability added, but slower than static images
- **#6 Versatility**: ⚡ **Text→Image + Video enabled** — 2 of 3 modalities complete (image understanding still needed)

> 💡 **AnimateDiff architecture wins**: Spatial SD layers (frozen) provide photorealism; temporal modules (trained on video data) provide motion consistency. You get VisualForge style (via SD checkpoint) + natural motion (via motion adapter) without retraining the entire model.

---

## 6 · Common Failure Modes

### Trap #1: Frame-by-Frame Generation Without Temporal Attention

**Symptom**: Generate 16 images with same prompt + different seeds → stitch into video → **massive flicker, teleportation, morphing**.

**Why it fails**: Each frame is independently generated from noise. No cross-frame communication → objects change color, furniture moves, faces morph.

**Fix**: Use AnimateDiff or any video model with **temporal attention layers** that attend across frames at same spatial position.

**Diagnostic**:
```python
# ❌ BAD: Frame-by-frame SD generation
images = []
for i in range(16):
    img = sd_pipe(prompt, generator=torch.Generator().manual_seed(42+i)).images[0]
    images.append(img)  # Each frame independent → flicker guaranteed

# ✅ GOOD: Temporal attention enforces consistency
video = animatediff_pipe(prompt, num_frames=16).frames[0]  # Frames attend to each other
```

---

### Trap #2: Insufficient Temporal Guidance → Motion Blur

**Symptom**: Generated video has **motion blur** or **slow-motion artifacts** — objects move unnaturally or freeze mid-motion.

**Why it fails**: AnimateDiff motion modules trained on 8-16 frame clips. Longer clips (24+ frames) extrapolate beyond training distribution → unnatural motion.

**Fix**: Generate 8-frame keyframes, use frame interpolation (RIFE, EMA-VFI) to reach target frame count.

**Diagnostic**:
```python
# ✅ Generate keyframes first, interpolate to target length
keyframes = animatediff_pipe(prompt, num_frames=8).frames[0]  # 8 frames at 0.5s
full_video = frame_interpolate(keyframes, target_fps=30)  # Interpolate to 30fps
```

> ⚠️ **Rule**: AnimateDiff sweet spot is **8-16 frames**. Beyond 16 frames, quality degrades. For longer clips, generate overlapping windows and stitch.

---

### Trap #3: Text Prompt Too Complex → Temporal Drift

**Symptom**: First 5 frames match prompt perfectly, last 5 frames **drift** — objects appear/disappear, colors shift, composition changes.

**Why it fails**: Complex prompts ("woman in dress + café + golden hour + camera movement") overload text conditioning. Early frames prioritize first tokens, later frames prioritize last tokens → **temporal drift**.

**Fix**: Simplify prompt to 1-2 key objects + style. Use ControlNet for composition control if needed.

**Diagnostic**:
```python
# ❌ BAD: Too many elements compete for attention
prompt = "woman, floral dress, café, Eiffel Tower background, golden hour, birds flying, coffee cup, pastries, street musician"

# ✅ GOOD: Focus on core subject + style
prompt = "woman in floral dress, Parisian café, golden hour, cinematic"
```

---

### Trap #4: Video Quality Evaluated by Eye

**Symptom**: "Looks good to me" after watching 3 generated clips → deploy to production → **client rejects 40% of videos** for flicker/inconsistency not caught in initial review.

**Why it fails**: Human attention misses subtle flicker in 15-second clips. At 10 clips/day, cognitive fatigue sets in after first hour.

**Fix**: Automate temporal consistency metrics — **frame-to-frame SSIM** (structural similarity), **optical flow smoothness**, **CLIP embedding drift** across frames.

**Diagnostic**:
```python
# Temporal consistency check: frame-to-frame SSIM should be > 0.85
from skimage.metrics import structural_similarity as ssim
import numpy as np

ssim_scores = []
for i in range(len(frames) - 1):
    score = ssim(frames[i], frames[i+1], multichannel=True)
    ssim_scores.append(score)

avg_ssim = np.mean(ssim_scores)
if avg_ssim < 0.85:
    print(f"⚠️ Low temporal consistency: {avg_ssim:.2f} — likely flicker")
```

> ⚠️ **Never evaluate video quality by watching 5 examples.** At 10 clips/day, you need automated metrics. Frame-to-frame SSIM, optical flow smoothness, and CLIP embedding drift are your production guardrails.

---

## 7 · When to Use Text-to-Video vs Alternatives

| Need | Use Text-to-Video | Use Alternative |
|------|-------------------|----------------|
| **Short social media ads** (5-15s) | ✅ AnimateDiff, Stable Video Diffusion | ❌ Traditional video editing too slow |
| **Animate a single image** (img→video) | ⚡ Stable Video Diffusion (img2vid mode) | ✅ Better control from reference image |
| **Long-form content** (60s+) | ❌ Temporal drift, memory limits | ✅ Frame interpolation + stitching |
| **Precise motion control** (specific hand gesture) | ❌ Text prompts lack precision | ✅ ControlNet + pose sequence |
| **Product 360° rotation** | ✅ AnimateDiff with camera motion prompt | ⚡ 3D render pipeline (if CAD available) |
| **Real-time generation** | ❌ 45s per 16-frame clip on RTX 4090 | ✅ Pre-rendered asset library |
| **Physics simulation** (liquid pour) | ❌ 2024 models struggle with physics | ✅ Houdini/Blender simulation |

**Decision framework**:

1. **Need < 15-second clip + don't have reference image?** → Text-to-Video (AnimateDiff)
2. **Have reference image + want to animate it?** → Image-to-Video (Stable Video Diffusion)
3. **Need precise motion control?** → ControlNet + pose sequence input
4. **Need > 60 seconds?** → Generate keyframes, interpolate with RIFE

> 💡 **VisualForge production rule**: Use text-to-video for **product demos and social ads** where speed > precision. For **hero brand films**, use text-to-video keyframes + manual cleanup in After Effects.

---

## 8 · Connection to Prior Chapters

**From Ch.3 CLIP**:
- Text conditioning infrastructure carries over — same CLIP text encoder maps "floral dress, café" → 768-dim embedding
- Video generation uses **same cross-attention mechanism** as static images: text embeddings condition each denoising step
- **New challenge**: Text must condition 16 frames consistently, not just 1 frame

**From Ch.6 Latent Diffusion**:
- Video diffusion operates in **VAE latent space**: 512×512×3 → 64×64×4 per frame
- 16-frame video: $(16, 64, 64, 4)$ latent tensor vs $(1, 64, 64, 4)$ for static images
- **8× compression** still applies per frame → video generation benefits from same speed gains

**From Ch.7 Classifier-Free Guidance**:
- CFG scale 7.5 recommendation transfers to video: balances prompt adherence vs diversity
- **New failure mode**: CFG scale > 12 introduces **temporal artifacts** (over-sharp motion, jitter)

**From Ch.8 Text-to-Image (SDXL)**:
- AnimateDiff **freezes spatial SD layers** from Ch.8's trained checkpoints
- VisualForge style (trained via LoRA in Ch.8) **carries over to video** — no retraining needed
- **Architecture extension**: Insert temporal attention layers between existing spatial blocks

**What's new in Ch.9**:
- **Temporal attention**: Attend across frames at same $(h, w)$ spatial position → enforce consistency
- **Motion modules**: Learn motion priors from video datasets (WebVid-10M) separately from spatial priors
- **3D tensor handling**: Batch dim includes time $(B \cdot T, C, H, W)$ or explicit $(B, T, C, H, W)$

> ➡️ **Ch.10 Multimodal LLMs** extends this further: vision encoder + language model → enables **visual question answering** ("Is the dress color consistent across all 16 frames?") for automated QA.

---

## 9 · Interview Checklist

### Must Know
- Why video adds temporal consistency as a fundamental new challenge (image-by-image generates flicker)
- Temporal attention: attend across frames at same spatial position
- AnimateDiff design: freeze spatial SD layers, train only temporal modules on video data

### Likely Asked
- *"How would you generate a 30-fps video cheaply?"* — Generate 8-frame keyframes with a T2V model, interpolate with a frame interpolation model (RIFE, EMA-VFI)
- *"What is a spacetime patch in Sora/DiT?"* — A 3D patch $(t_p, h_p, w_p)$ treated as a single transformer token; enables arbitrary-length, arbitrary-resolution video generation
- *"What's the difference between animating a static image vs. text-to-video?"* — Animating (img2video, Stable Video Diffusion) starts from a real image latent; T2V starts from pure noise; both use temporal attention

### Trap to Avoid
- Don't claim that temporal consistency is free from DDPM — the Markovian noise process is *independent per frame*; correlation across frames must be learned explicitly via temporal attention layers.

---

### Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "Sora generates video by iterating frames" | It generates all frames simultaneously via 3D denoising; no autoregressive frame generation |
| "AnimateDiff fine-tunes the whole SD model" | Only the temporal attention modules are trained; spatial SD layers are frozen |
| "More timesteps always helps with T2V" | Consistency depends on temporal attention quality, not step count — temporal blur comes from weak temporal attention, not from too few steps |
| "Video models are just SD run N times" | Naive frame-by-frame SD produces incoherent flicker; temporal modeling is required |
| "Sora can do any video task out of the box" | Even Sora has known failure modes: physics violations, object permanence failures, finger distortion |

---

## 10 · Further Reading

### Foundational Papers

**AnimateDiff** (Guo et al., 2023)
[Paper](https://arxiv.org/abs/2307.04725) | [Code](https://github.com/guoyww/AnimateDiff) | [HuggingFace](https://huggingface.co/guoyww/animatediff)
**Key insight**: Freeze spatial SD layers, train only temporal attention modules on video data → video generation without full model retraining.

**Stable Video Diffusion** (Blattmann et al., 2023)
[Paper](https://arxiv.org/abs/2311.15127) | [Code](https://github.com/Stability-AI/generative-models)
**Key insight**: Image-to-video architecture with temporal layers + camera motion conditioning → first open video foundation model.

**Sora** (OpenAI, 2024)
[Technical Report](https://openai.com/research/video-generation-models-as-world-simulators)
**Key insight**: Diffusion Transformer (DiT) operating on spacetime patches → arbitrary-length, arbitrary-resolution video generation.

**CogVideoX** (Tsinghua, 2024)
[Paper](https://arxiv.org/abs/2408.06072) | [Code](https://github.com/THUDM/CogVideo)
**Key insight**: 3D DiT with temporal attention + expert transformer for motion control → state-of-the-art open text-to-video.

---

### Tutorials & Guides

- [Diffusers AnimateDiff Tutorial](https://huggingface.co/docs/diffusers/main/en/api/pipelines/animatediff) — Official integration guide
- [Stable Video Diffusion Guide](https://stability.ai/news/stable-video-diffusion-open-ai-video-model) — Image-to-video pipeline walkthrough
- [RIFE Frame Interpolation](https://github.com/megvii-research/ECCV2022-RIFE) — Extend 8-frame clips to 30fps
- [Temporal Consistency Metrics](https://github.com/v-sense/DAVIS) — Evaluate video generation quality

---

### Production Tools

- [ComfyUI AnimateDiff Nodes](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) — Visual workflow for video generation
- [Deforum Stable Diffusion](https://github.com/deforum-art/deforum-stable-diffusion) — Keyframe animation for long videos
- [EbSynth](https://ebsynth.com/) — Style transfer for video consistency
- [Topaz Video AI](https://www.topazlabs.com/topaz-video-ai) — Commercial upscaling + frame interpolation

---

## 11 · Notebook

**CPU-runnable**: [text_to_video_educational.ipynb](notebooks/text_to_video_educational.ipynb)
Builds temporal attention module from scratch, visualizes frame coherence on synthetic sequences, inspects AnimateDiff architecture via `diffusers` metadata.

**GPU supplement**: [text_to_video_supplement.ipynb](notebooks/text_to_video_supplement.ipynb)
Generate 16-frame 512×512 clips with AnimateDiff, test temporal consistency metrics (frame-to-frame SSIM, optical flow smoothness), compare AnimateDiff vs frame-by-frame SD generation.

> ⚠️ **GPU supplement requires**: CUDA GPU with ≥12GB VRAM (RTX 3060 or better). Generation time: ~45 seconds per 16-frame clip on RTX 4090, ~8 minutes on Colab T4.

---

## 11.5 · Progress Check — What Have We Unlocked?

### Before This Chapter
- **Constraint #6 (Versatility)**: ⚡ Text→Image production-ready, no video capability
- **VisualForge Status**: Can only deliver static marketing images, clients want video ads

### After This Chapter
- **Constraint #5 (Throughput)**: ⚡ **~85 images/day + 10 video clips/day** → Video capability added
- **Constraint #6 (Versatility)**: ⚡ **Text→Image + Video enabled** → 2 of 3 modalities complete
- **VisualForge Status**: Can generate 15-second social media ads (product rotating, zoom effects)

---

### Key Wins

1. **Temporal attention**: Attend across frames at same spatial position → enforces consistency (no flicker/teleportation)
2. **AnimateDiff architecture**: Freeze spatial SD layers, train only temporal modules → video generation without retraining from scratch
3. **Video capability unlocked**: 16-frame 512×512 clips = ~1 second at 15fps (acceptable for social ads)

---

### What's Still Blocking Production?

**QA bottleneck**: Every generated image/video requires **manual human review** to verify it matches the client brief. 100% manual QA = bottleneck limiting throughput. Need automated verification: "Is the product centered? Is the lighting natural? Does it match the brief?"

**Next unlock (Ch.10)**: **Multimodal LLMs (VLMs)** — LLaVA/BLIP-2 can caption images and answer visual questions. Automate QA workflow → flag only 15% for human review → unlock full 120+ images/day throughput.

---

### VisualForge Status — Full Constraint View

| Constraint | Ch.6 | Ch.7 | Ch.8 | This Ch. | Target |
|------------|------|------|------|----------|--------|
| #1 Quality | ⚡ 3.5/5.0 | ⚡ 3.8/5.0 | ⚡ 3.8/5.0 | ⚡ 3.8/5.0 | ≥4.0/5.0 |
| #2 Speed | ✅ 20s | ✅ 20s | ✅ 18s | ✅ 18s (image) | <30s |
| #3 Cost | ✅ $2.5k | ✅ $2.5k | ✅ $2.5k | ✅ $2.5k laptop | <$5k |
| #4 Control | ⚡ <15% | ⚡ <15% | ✅ 3% | ✅ 3% unusable | <5% |
| #5 Throughput | ❌ | ❌ | ⚡ 80/day | ⚡ 85 imgs + 10 vids/day | 100+/day |
| #6 Versatility | ⚡ T→I | ⚡ T→I | ⚡ T→I | ⚡ T→I + Video | 3 modalities |

**Key**: ❌ = Blocked | ⚡ = Foundation laid / partial progress | ✅ = Target hit

**Ch.9 progress**: Video generation capability unlocked → Versatility improved (2 of 3 modalities). Throughput increased but still below target (QA bottleneck remains). Quality plateau continues (needs evaluation infrastructure from Ch.11).

---

## Bridge to Chapter 10

Ch.9 **unlocked video generation** — VisualForge can now produce 15-second social media ads (product demos, 360° rotations). Constraints #5 (Throughput) and #6 (Versatility) moved forward: **~85 images/day + 10 video clips/day**, **Text→Image + Video enabled**.

**What's still blocking**: Every generated image and video requires **manual human QA** to verify it matches the client brief. 100% manual review = bottleneck. At 120 images/day target, you need automated verification: "Is the product centered? Is the lighting natural? Does it match the brief?"

Ch.10 **Multimodal LLMs** solves this: combine a vision encoder (CLIP) with a language model (Llama, GPT) → enables **visual question answering**. LLaVA can caption images, answer "Is the dress color consistent across frames?" and "Does this match the brief: 'floral dress, café, golden hour'?" Automate QA → flag only 15% for human review → unlock full 120+ images/day throughput.

## Illustrations

![Text-to-video - video tensor, spatial+temporal attention, consistency, compute scaling](img/text-to-video.png)
