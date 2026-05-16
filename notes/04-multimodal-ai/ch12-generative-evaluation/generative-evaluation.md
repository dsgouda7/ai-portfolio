# Generative Evaluation — Measuring What You Made

> **Track:** Multimodal AI
> **Prerequisites:** [DiffusionModels.md](../ch04_diffusion_models/diffusion-models.md), [GuidanceConditioning.md](../ch07_guidance_conditioning/guidance-conditioning.md), [CLIP.md](../ch03_clip/clip.md)

> **The story.** Evaluating generative images is a 9-year-old subfield. **Inception Score (IS)** (Salimans et al., OpenAI, **2016**) was the first widely-used automatic metric — high IS meant images were both confident and diverse under an Inception classifier — but it was famously gameable. **FID** — *Fréchet Inception Distance* (**Heusel et al.**, **NIPS 2017**) — replaced IS by comparing the distribution of generated and real Inception features under a Gaussian assumption. FID became the field's default metric for half a decade despite its quirks (sample-size sensitivity, blindness to text alignment). **CLIPScore** (Hessel et al., 2021) added text-image alignment by reusing CLIP. **Human Preference Score (HPS)** (Wu et al., 2023) and **PickScore** (Kirstain et al., NeurIPS 2023) trained reward models on millions of human preference pairs from ChatGPT-style A/B comparisons — the first metrics that actually correlated well with what humans like. The 2026 evaluation stack pairs all of these with a multimodal LLM judge.
>
> **Where you are in the curriculum.** You can generate images. The honest question is: *are they any good?* This chapter gives the toolkit — FID, IS, CLIPScore, HPS, human preference — and explains why each one can mislead you and which combination to ship in production.

![Generative evaluation flow animation](img/generative-evaluation-flow.gif)

*Flow: generated and real sets feed metric pipelines, then aggregate into one quality signal for go/no-go decisions.*

---

## 0 · The VisualForge Studio Challenge

**Mission**: VisualForge needs ≥4.0/5.0 professional quality to match freelancer baseline (4.2/5.0).

**Current blocker at Chapter 11**: Client surveys report **~3.9/5.0 quality**, but manual surveys are slow (1 week turnaround) and expensive ($500/survey). Need objective, automated metrics to:
- Track quality improvements over time
- Validate A/B tests (e.g., guidance scale 7.5 vs 12.0)
- Prove to clients that AI quality matches freelancers

**What this chapter unlocks**: **Automated evaluation metrics** — FID (distribution similarity), CLIP Score (text-image alignment), HPSv2 (predicts human ratings). Run on 500-image test set in 10 minutes. Discover: HPSv2 = **4.1/5.0** (exceeds 4.0 target!). Client surveys were during ramp-up; current quality higher.

---

### The 6 Constraints — Snapshot After Chapter 11

| Constraint | Target | Status | Evidence |
|------------|--------|--------|----------|
| #1 Quality | ≥4.0/5.0 | **4.1/5.0** | HPSv2 score on 500-image test set (exceeds target!) |
| #2 Speed | <30 seconds | **~18s** | Unchanged from Ch.10 |
| #3 Cost | <$5k hardware | **$2.5k laptop** | Unchanged from Ch.10 |
| #4 Control | <5% unusable | **~3% unusable** | Unchanged from Ch.10 |
| #5 Throughput | 100+ images/day | **~120 images/day** | Unchanged from Ch.10 |
| #6 Versatility | 3 modalities | **All 3 enabled** | Unchanged from Ch.10 |

---

### What's Still Blocking Us After This Chapter?

**Optimization**: System works (all 6 constraints met!) but not optimized. Takes ~18 seconds per image (target <30s). Can we go faster? Hardware not fully tuned (FP16 vs INT8, batch processing, etc.).

**Next unlock (Ch.12)**: **Local Diffusion Lab (Production Optimization)** — SDXL-Turbo (4 steps, 8 seconds), quantization, production deployment patterns. Final assembly.

---

## 1 · Core Idea

**You're the Lead ML Engineer at VisualForge Studio.** You've just generated 100 spring-collection hero images. Before sending them to the creative director, you need to answer: *are they good enough to replace $600k/year of freelancer work?*

Generative evaluation is the science of measuring the **quality, fidelity, diversity, and alignment** of your generated images — without requiring a human judge for every sample.

Three orthogonal axes you must measure:

| Axis | Your question | Representative metric |
|------|----------|-----------------------|
| **Fidelity** | Do your generated images look real? | FID ↓ |
| **Diversity** | Does your model cover the full distribution? | FID ↓, Precision/Recall |
| **Alignment** | Does the output match your text prompt? | CLIP Score ↑ |

**Critical insight:** No single metric captures all three. You need at least two to make a defensible decision.

---

## 1.5 · The Practitioner Workflow — Your 4-Phase Quality Gate

> **Warning — Two ways to read this chapter:**
> - **Theory-first (recommended for learning):** Read §0→§5 sequentially to understand the concepts, then use this workflow as your reference
> - **Workflow-first (practitioners with existing knowledge):** Use this diagram as a jump-to guide when working with real data
>
> **Note:** Section numbers don't follow phase order because the chapter teaches concepts pedagogically (theory before application). The workflow below shows how to APPLY those concepts.

**What you'll build by the end:** A multi-metric quality scorecard combining FID (distribution similarity), CLIP Score (text-image alignment), and HPSv2 (human preference prediction) — the three-metric gate from §5 that answers "are my generated images ready to ship?"

```
Phase 1: IDENTIFY Phase 2: SELECT Phase 3: COMPUTE Phase 4: INTERPRET
────────────────────────────────────────────────────────────────────────────────────────────────────
Define evaluation goals: Choose metric suite: Run evaluation pipeline: Analyze results:

• Fidelity priority? • FID for distribution • Generate N≥5k images • Compare to thresholds
• Alignment priority? • CLIP for text match • torch-fidelity FID • Cross-check metrics
• Human preference? • HPSv2 for aesthetics • CLIPScore from HF • Identify failure modes
• Sample size N • Precision/Recall if • batch_size=50 • Make go/no-go call
 needed • Same resolution!

→ DECISION: → DECISION: → DECISION: → DECISION:
 What to measure? Which metrics? Ready to run? Ship or iterate?
 • Text-to-image task: • Core: FID + CLIP • N<5k? FID unstable • FID<50 + CLIP>0.25:
 FID + CLIP + HPSv2 • Optional: HPSv2 • Different resolutions? likely ready
 • Img2img task: • If diversity matters: Resize consistently • HPSv2<3.8: aesthetic
 LPIPS + CLIP add Precision/Recall • Reference set clean? issues
 • Video: FVD + CLIPSIM • Low CLIP + high FID:
 model ignores prompts
```

> **How to use this workflow:** Complete Phase 1→2 to define your evaluation strategy, then run Phase 3 once you have generated images. Phase 4 is where you make the production decision — refer back to §6 Failure Modes if metrics contradict each other.

---

### What to Measure

**Goal:** Define evaluation priorities based on your use case and business constraints.

**Input required:**
- **Use case type**: text-to-image generation (e.g., VisualForge campaigns), image-to-image refinement (e.g., inpainting, super-resolution), or video generation
- **Primary stakeholder concern**: visual realism (clients can't tell it's AI), prompt adherence (outputs match briefs), or aesthetic quality (professional-grade "looks good")
- **Sample budget**: how many images can you generate for evaluation? (Minimum N=500 for spot checks, N≥5,000 for stable FID estimates)
- **Reference corpus**: do you have approved real images for comparison? (Required for FID; not needed for CLIP Score)

**Decision tree for metric selection:**

```
START: What's your use case?
│
├─ Text-to-image generation (e.g., VisualForge product shots)
│ ├─ Concern: "Do images look real?"
│ │ └─ Measure: FID vs reference corpus (§3.1)
│ │ Threshold: <50 for production quality
│ │
│ ├─ Concern: "Do outputs match text prompts?"
│ │ └─ Measure: CLIP Score (§3.3)
│ │ Threshold: >0.25 for prompt adherence
│ │
│ └─ Concern: "Will clients like these?"
│ └─ Measure: HPSv2 (§3.3, §5)
│ Threshold: >4.0/5.0 for professional quality
│
├─ Image-to-image refinement (e.g., inpainting, style transfer)
│ ├─ Concern: "How close to the reference image?"
│ │ └─ Measure: LPIPS (§3.4)
│ │ Threshold: <0.15 for perceptually similar
│ │
│ └─ Concern: "Does edited region match prompt?"
│ └─ Measure: CLIP Score on edited region only
│
└─ Video generation
 ├─ Concern: "Per-frame realism"
 │ └─ Measure: FVD (§9 Interview Checklist)
 │
 └─ Concern: "Temporal consistency"
 └─ Measure: VBench suite (§10 Further Reading)
```

**Code snippet — Metric selection helper:**

```python
# Define evaluation strategy based on use case
def plan_evaluation(use_case: str,
 primary_concern: str,
 sample_count: int,
 has_reference_corpus: bool) -> dict:
 """
 Returns recommended metric suite and minimum sample requirements.

 Args:
 use_case: 'text-to-image', 'img2img', or 'video'
 primary_concern: 'realism', 'alignment', 'aesthetics', or 'perceptual_similarity'
 sample_count: number of generated images available for evaluation
 has_reference_corpus: True if you have real images for comparison

 Returns:
 dict with 'metrics' (list), 'min_samples' (int), 'warnings' (list)
 """
 plan = {'metrics': [], 'min_samples': 500, 'warnings': []}

 if use_case == 'text-to-image':
 if primary_concern == 'realism':
 if has_reference_corpus:
 plan['metrics'].append('FID')
 plan['min_samples'] = 5000 # FID needs large N
 if sample_count < 5000:
 plan['warnings'].append(
 f"FID unstable with N={sample_count} (±10 FID variance). "
 "Increase to N≥5k or use only as relative comparison."
 )
 else:
 plan['warnings'].append("FID requires reference corpus. Use CLIP + HPSv2 instead.")

 if primary_concern in ['alignment', 'realism']:
 plan['metrics'].append('CLIP Score')

 if primary_concern == 'aesthetics':
 plan['metrics'].append('HPSv2')

 # Default: use all three for comprehensive evaluation
 if not plan['metrics']:
 plan['metrics'] = ['FID', 'CLIP Score', 'HPSv2']
 plan['warnings'].append("No primary concern specified — using full suite.")

 elif use_case == 'img2img':
 plan['metrics'] = ['LPIPS', 'CLIP Score']
 plan['min_samples'] = 100 # LPIPS works per-image

 elif use_case == 'video':
 plan['metrics'] = ['FVD', 'CLIPSIM', 'VBench']
 plan['min_samples'] = 1000 # FVD needs large N like FID

 return plan

# Example usage for VisualForge
vf_plan = plan_evaluation(
 use_case='text-to-image',
 primary_concern='realism',
 sample_count=500,
 has_reference_corpus=True
)
print(f"Recommended metrics: {vf_plan['metrics']}")
print(f"Minimum samples: {vf_plan['min_samples']}")
for warning in vf_plan['warnings']:
 print(f" {warning}")
# Output:
# Recommended metrics: ['FID', 'CLIP Score']
# Minimum samples: 5000
# FID unstable with N=500 (±10 FID variance). Increase to N≥5k or use only as relative comparison.
```

> 🏭 **Industry standard:** OpenAI's DALL-E 3 evaluation uses FID (N=30k) + CLIP Score + human preference ratings (n=1000 human comparisons). Midjourney V6 uses HPSv2 + Aesthetic Predictor + manual QA on 500-image batches. Stability AI (SDXL) reports FID (N=50k) + CLIP Score + FID-CLIP harmonic mean.

> **Metric selection verdict:** FID + CLIP Score + HPSv2 covers VisualForge quality, alignment, and preference goals — FID is unstable at N=500; use for relative comparisons only until N ≥ 5,000.
> ➡ Stable metric implementations are selected next to ensure reproducible cross-experiment comparisons.

---

### Choose Metric Implementations

**Goal:** Select stable, validated implementations for your chosen metrics from Phase 1.

**Critical: Not all implementations are equivalent.** FID varies by ±5 points depending on which Inception model, feature layer, and image preprocessing you use. Always use the same implementation for all experiments you want to compare.

**Recommended libraries:**

| Metric | Library | Installation | Notes |
|--------|---------|--------------|-------|
| **FID** | `torch-fidelity` or `clean-fid` | `pip install torch-fidelity` | Use `torch-fidelity` (most stable). `clean-fid` fixes some Inception preprocessing bugs but gives different absolute scores. Never mix implementations. |
| **CLIP Score** | `torchmetrics` or `HuggingFace transformers` | `pip install torchmetrics` | Use `openai/clip-vit-base-patch32` for reproducibility. ViT-L/14 gives higher scores but is slower. |
| **HPSv2** | Official repo | `git clone https://github.com/tgxs002/HPSv2` | Requires `transformers>=4.25.1`. Checkpoint: `hpsv2_src/HPS_v2_compressed.pt` |
| **LPIPS** | `lpips` package | `pip install lpips` | Use `alex` backbone (default) unless you need VGG. |
| **Precision/Recall** | `prdc` | `pip install prdc` | k=5 neighbors default; increase to k=10 for smoother manifolds. |

**Code snippet — Environment setup and implementation validation:**

```python
# Install and validate metric implementations
import subprocess
import sys

def setup_metrics_env(metrics: list):
 """Install required packages for chosen metrics."""
 package_map = {
 'FID': ['torch-fidelity'],
 'CLIP Score': ['torchmetrics', 'transformers'],
 'HPSv2': ['transformers'], # Manual clone needed
 'LPIPS': ['lpips'],
 'Precision/Recall': ['prdc']
 }

 packages_needed = set()
 for metric in metrics:
 packages_needed.update(package_map.get(metric, []))

 for pkg in packages_needed:
 subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

 print(f" Installed packages: {', '.join(packages_needed)}")

 # Validate installations
 try:
 if 'FID' in metrics:
 from torch_fidelity import calculate_metrics
 print(" torch-fidelity ready")

 if 'CLIP Score' in metrics:
 from transformers import CLIPModel, CLIPProcessor
 model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
 print(" CLIP model loaded (openai/clip-vit-base-patch32)")

 if 'LPIPS' in metrics:
 import lpips
 loss_fn = lpips.LPIPS(net='alex')
 print(" LPIPS (AlexNet backbone) ready")

 if 'Precision/Recall' in metrics:
 from prdc import compute_prdc
 print(" PRDC (Precision/Recall) ready")

 except ImportError as e:
 print(f" Import failed: {e}")
 raise

# Example: VisualForge setup
setup_metrics_env(['FID', 'CLIP Score', 'HPSv2'])
```

> 🏭 **Industry standard:** Weights & Biases (W&B) provides unified metric tracking with `wandb.log({'fid': fid_score, 'clip_score': clip_mean})`. Google's Imagen paper uses `clean-fid` with Inception-v3 pool3 features at 299×299 resolution. Stable Diffusion XL evaluation uses `torch-fidelity` with 50k samples.

> **Implementation verdict:** torch-fidelity FID, CLIP ViT-B/32, and HPSv2 checkpoint validated — all cross-checks pass, consistent implementations locked for all experiments.
> ➡ Evaluation pipeline is run on the full VisualForge spring-collection batch in the next section.

---

### Run Evaluation Pipeline

**Goal:** Generate evaluation dataset, compute metrics, and log results for reproducibility.

**Critical steps:**
1. **Generate sufficient samples** — N≥5,000 for FID; N≥100 for CLIP/HPSv2 spot-checks
2. **Match preprocessing** — reference and generated images must use identical resize, normalization, and format
3. **Batch processing** — process in batches of 50–100 to avoid OOM on GPU
4. **Save raw scores** — log per-image CLIP scores and aggregated FID for debugging

**Code snippet — Production evaluation pipeline (VisualForge):**

```python
# Full evaluation pipeline for VisualForge campaign batch
import torch
from torch_fidelity import calculate_metrics
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import glob
from pathlib import Path
import json
from tqdm import tqdm

class GenerativeEvaluator:
 """
 Production-grade evaluation pipeline for text-to-image generation.
 Computes FID, CLIP Score, and optionally HPSv2.
 """
 def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
 self.device = device
 self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
 self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

 def compute_fid(self,
 real_dir: Path,
 generated_dir: Path,
 batch_size: int = 50) -> float:
 """
 Compute FID between real and generated image directories.

 Args:
 real_dir: directory containing reference images
 generated_dir: directory containing generated images
 batch_size: batch size for feature extraction

 Returns:
 FID score (lower is better; <50 = production quality)
 """
 metrics = calculate_metrics(
 input1=str(generated_dir),
 input2=str(real_dir),
 cuda=self.device == 'cuda',
 isc=False, # disable Inception Score (we don't need it)
 fid=True,
 kid=False,
 verbose=False,
 batch_size=batch_size
 )
 return metrics['frechet_inception_distance']

 def compute_clip_scores(self,
 image_paths: list,
 prompts: list,
 batch_size: int = 32) -> dict:
 """
 Compute CLIP Score for each (image, prompt) pair.

 Args:
 image_paths: list of paths to generated images
 prompts: list of text prompts (must match length of image_paths)
 batch_size: batch size for CLIP encoding

 Returns:
 dict with 'scores' (per-image), 'mean', 'std', 'min', 'max'
 """
 assert len(image_paths) == len(prompts), "Must have one prompt per image"

 scores = []
 for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP Score"):
 batch_imgs = [Image.open(p).convert('RGB') for p in image_paths[i:i+batch_size]]
 batch_prompts = prompts[i:i+batch_size]

 inputs = self.clip_processor(
 text=batch_prompts,
 images=batch_imgs,
 return_tensors="pt",
 padding=True
 ).to(self.device)

 with torch.no_grad():
 outputs = self.clip_model(**inputs)
 # CLIP Score = 2.5 * max(0, cosine_similarity)
 # logits_per_image is already scaled cosine similarity
 batch_scores = (outputs.logits_per_image.diagonal() / 100).cpu().numpy()

 scores.extend(batch_scores)

 return {
 'scores': scores,
 'mean': sum(scores) / len(scores),
 'std': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
 'min': min(scores),
 'max': max(scores)
 }

 def evaluate_campaign(self,
 real_dir: Path,
 generated_dir: Path,
 prompts_file: Path) -> dict:
 """
 Full evaluation for a VisualForge campaign batch.

 Args:
 real_dir: reference corpus (approved past campaigns)
 generated_dir: generated images for this campaign
 prompts_file: JSON file mapping filename -> prompt

 Returns:
 dict with all metrics and pass/fail thresholds
 """
 # Load prompts
 with open(prompts_file) as f:
 prompt_map = json.load(f)

 gen_paths = sorted(glob.glob(str(generated_dir / "*.png")))
 prompts = [prompt_map[Path(p).name] for p in gen_paths]

 print(f"Evaluating {len(gen_paths)} generated images...")

 # Compute FID (requires large N for stability)
 print("Computing FID...")
 fid_score = self.compute_fid(real_dir, generated_dir)

 # Compute CLIP Score (works with any N)
 print("Computing CLIP Scores...")
 clip_results = self.compute_clip_scores(gen_paths, prompts)

 # Assemble results
 results = {
 'fid': {
 'score': fid_score,
 'threshold': 50.0,
 'pass': fid_score < 50.0,
 'interpretation': 'Distribution similarity to reference corpus'
 },
 'clip_score': {
 'mean': clip_results['mean'],
 'std': clip_results['std'],
 'threshold': 0.25,
 'pass': clip_results['mean'] > 0.25,
 'interpretation': 'Text-image alignment (prompt adherence)'
 },
 'overall_pass': fid_score < 50.0 and clip_results['mean'] > 0.25,
 'sample_count': len(gen_paths),
 'warnings': []
 }

 # Add warnings for common issues
 if len(gen_paths) < 5000:
 results['warnings'].append(
 f"FID computed on N={len(gen_paths)} samples (unstable, ±10 variance). "
 "Use for relative comparison only or increase to N≥5k."
 )

 if clip_results['std'] > 0.15:
 results['warnings'].append(
 f"High CLIP Score variance (σ={clip_results['std']:.3f}) — "
 "some images may not match prompts. Inspect low-scoring outliers."
 )

 return results

# Example usage: VisualForge spring collection evaluation
evaluator = GenerativeEvaluator(device='cuda')
results = evaluator.evaluate_campaign(
 real_dir=Path("data/vf_reference_corpus"),
 generated_dir=Path("output/spring_collection_v1"),
 prompts_file=Path("output/spring_collection_v1/prompts.json")
)

print(json.dumps(results, indent=2))
# Output:
# {
# "fid": {
# "score": 42.3,
# "threshold": 50.0,
# "pass": true,
# "interpretation": "Distribution similarity to reference corpus"
# },
# "clip_score": {
# "mean": 0.31,
# "std": 0.08,
# "threshold": 0.25,
# "pass": true,
# "interpretation": "Text-image alignment (prompt adherence)"
# },
# "overall_pass": true,
# "sample_count": 500,
# "warnings": [
# "FID computed on N=500 samples (unstable, ±10 variance). Use for relative comparison only or increase to N≥5k."
# ]
# }
```

> 🏭 **Industry standard:** Stability AI's SDXL evaluation pipeline uses `torch-fidelity` with 50k samples, CLIP ViT-L/14 (not ViT-B/32), and Aesthetic Predictor v2.1. Images resized to 512×512 for FID (original training resolution). All metrics logged to Weights & Biases with per-checkpoint comparisons.

> **Compute verdict:** FID=42.3, CLIP=0.31 (σ=0.08) on 500 images — preprocessing consistent, prompt coverage complete; FID sample-size warning noted for relative comparison use only.
> ➡ Metric scores are interpreted against quality thresholds in the go/no-go section.

---

### Make Go/No-Go Decision

**Goal:** Interpret metric scores against thresholds, identify failure modes, and make the production decision.

**Threshold guidelines (text-to-image generation):**

| Metric | Target | Good | Acceptable | Poor | Interpretation |
|--------|--------|------|------------|------|----------------|
| **FID** | <30 | <50 | 50–100 | >100 | <50 = indistinguishable from real images in distribution; >100 = obvious artifacts |
| **CLIP Score** | >0.30 | >0.25 | 0.20–0.25 | <0.20 | >0.25 = prompt adherence; <0.20 = model ignores prompts |
| **HPSv2** | >4.2 | >4.0 | 3.8–4.0 | <3.8 | >4.0 = professional quality; <3.8 = aesthetic issues (composition, lighting) |

**Decision matrix (VisualForge quality gate):**

```
 CLIP Score
 Low (<0.25) High (≥0.25)
 ┌─────────────────┬─────────────────┐
 FID │ FAIL │ PARTIAL │
 High │ Model broken │ Realistic but │
 (≥50) │ (ignore prompts │ prompt-agnostic │
 │ + unrealistic) │ (tune CFG ↑) │
 ├─────────────────┼─────────────────┤
 FID │ PARTIAL │ PASS │
 Low │ Adheres to │ Realistic + │
 (<50) │ prompts but │ prompt-aligned │
 │ lacks realism │ → ready to ship │
 │ (tune steps ↑) │ │
 └─────────────────┴─────────────────┘
```

**Code snippet — Interpret results and generate report:**

```python
# Interpret evaluation results and make go/no-go decision
def interpret_results(results: dict, use_case: str = 'visualforge') -> dict:
 """
 Interpret evaluation metrics and generate actionable recommendations.

 Args:
 results: output from GenerativeEvaluator.evaluate_campaign()
 use_case: 'visualforge' (strict thresholds) or 'internal' (relaxed)

 Returns:
 dict with 'decision', 'blockers', 'warnings', 'recommendations'
 """
 thresholds = {
 'visualforge': {'fid': 50.0, 'clip': 0.25, 'hps': 4.0},
 'internal': {'fid': 100.0, 'clip': 0.20, 'hps': 3.5}
 }[use_case]

 fid = results['fid']['score']
 clip = results['clip_score']['mean']

 interpretation = {
 'decision': 'UNDECIDED',
 'blockers': [],
 'warnings': results.get('warnings', []),
 'recommendations': []
 }

 # Decision logic based on 2×2 matrix
 fid_pass = fid < thresholds['fid']
 clip_pass = clip > thresholds['clip']

 if fid_pass and clip_pass:
 interpretation['decision'] = 'PASS — Ready to ship'
 interpretation['recommendations'].append(
 f" FID={fid:.1f} < {thresholds['fid']} and CLIP={clip:.3f} > {thresholds['clip']} "
 "→ Images are realistic and prompt-aligned."
 )

 elif not fid_pass and not clip_pass:
 interpretation['decision'] = 'FAIL — Model fundamentally broken'
 interpretation['blockers'].append(
 f" FID={fid:.1f} ≥ {thresholds['fid']} (unrealistic) AND "
 f"CLIP={clip:.3f} ≤ {thresholds['clip']} (ignores prompts)"
 )
 interpretation['recommendations'].append(
 "Check model checkpoint, verify conditioning is enabled, inspect sample outputs for collapse."
 )

 elif fid_pass and not clip_pass:
 interpretation['decision'] = 'PARTIAL PASS — Realistic but prompt-agnostic'
 interpretation['warnings'].append(
 f" FID={fid:.1f} < {thresholds['fid']} (realistic) BUT "
 f"CLIP={clip:.3f} ≤ {thresholds['clip']} (weak prompt adherence)"
 )
 interpretation['recommendations'].append(
 "Increase CFG guidance scale (try 7.5→12.0) to strengthen prompt conditioning. "
 "Verify CLIP embeddings are passed to UNet correctly."
 )

 elif not fid_pass and clip_pass:
 interpretation['decision'] = 'PARTIAL PASS — Prompt-aligned but lacks realism'
 interpretation['warnings'].append(
 f" FID={fid:.1f} ≥ {thresholds['fid']} (lacks realism) BUT "
 f"CLIP={clip:.3f} > {thresholds['clip']} (prompt-aligned)"
 )
 interpretation['recommendations'].append(
 "Increase inference steps (25→50) or switch to better scheduler (Euler→DPM++) to improve quality. "
 "Check for training artifacts or dataset quality issues."
 )

 # Add HPSv2 interpretation if available
 if 'hps' in results:
 hps = results['hps']['score']
 if hps < thresholds['hps']:
 interpretation['warnings'].append(
 f" HPSv2={hps:.2f} < {thresholds['hps']} — "
 "Technically correct but aesthetically unpleasing. "
 "Check composition, lighting, color grading."
 )

 return interpretation

# Example: VisualForge spring collection decision
decision = interpret_results(results, use_case='visualforge')
print(f"\n{'='*60}")
print(f"DECISION: {decision['decision']}")
print(f"{'='*60}")
for blocker in decision['blockers']:
 print(blocker)
for warning in decision['warnings']:
 print(warning)
for rec in decision['recommendations']:
 print(f"→ {rec}")
```

> 🏭 **Industry standard:** Midjourney V6 uses HPSv2>4.2 + manual QA pass rate >90% as the release gate. OpenAI DALL-E 3 uses FID<20 (against LAION-5B subset) + CLIP Score>0.32 + human preference win rate >60% vs. DALL-E 2. Stable Diffusion XL uses FID<25 + Aesthetic Score>6.0/10 as the checkpoint selection criterion.

> **FID verdict:** FID 42.3 → CLIP 0.31 → HPSv2 4.1 — VisualForge spring collection passes all thresholds; ship to creative director for spot-check (expect ≥80% approval).
> ➡ CLIP similarity and HPSv2 confirm prompt alignment and aesthetic quality alongside distribution realism.

---

## 2 · Running Example

**VisualForge campaign evaluation suite.** You've just generated a batch of 100 spring-collection product images. Before you send them to the creative director, you need objective proof they're ready:

- Do your generated product shots look like real studio photographs? → **FID** against your reference product corpus (500 approved images from past campaigns)
- Are all VisualForge campaign types represented (product-on-white, lifestyle, brand-pattern)? → **class recall per brief type**
- Does "Mango leather crossbody bag, white background" produce a bag on white, not a lifestyle shot? → **CLIP Score** (text-image alignment)

**Why this matters:** At 120 images/day throughput, you can't manually review every output. Automated metrics give you a 10-minute quality gate instead of 2 hours of manual inspection.

> 📖 **Educational proxy:** FID math is illustrated using MNIST digit generation (reference = real digits, generated = DDPM output) because it's compact and verifiable. The VisualForge production evaluation (§5) applies the same metrics to campaign image batches.

---

## 3 · The Math — What Each Metric Measures

**Why evaluation math matters for VisualForge:** You need proof that your AI outputs match freelancer quality before replacing $600k/year of human work. "Looks good" isn't evidence. These metrics give you objective numbers you can defend to the CEO.

### 3.1 Fréchet Inception Distance (FID)

Extract features $\mu_r, \Sigma_r$ from **real** images and $\mu_g, \Sigma_g$ from **generated** images using a pre-trained feature extractor (canonically Inception-v3 pool3 layer):

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr} \left(\Sigma_r + \Sigma_g - 2 \left(\Sigma_r \Sigma_g\right)^{1/2}\right)
$$

- **Lower = better** (FID < 50 = production quality for VisualForge campaigns).
- Measures distance between the *distributions*, not individual images.
- **Biased at small N** — needs ≥ 5,000 samples for stable estimates (often 50k).

> 🏭 **Industry standard — torch-fidelity:** PyTorch Lightning's `torch-fidelity` library is the production standard for FID computation. Install with `pip install torch-fidelity`. Key advantages: (1) GPU acceleration with batched feature extraction, (2) matches original TensorFlow FID implementation within ±0.5 FID, (3) supports multiple backends (Inception-v3, SwAV). Usage: `calculate_metrics(input1='generated/', input2='real/', cuda=True, fid=True)`. Stability AI uses `torch-fidelity` with 50k samples for all SDXL checkpoints.

**How you compute it:**
1. **Generate** $N$ images from your model ($N \geq 5000$, ideally 50k).
2. **Extract features**: pass each real and generated image through Inception-v3 up to the `mixed_7c` pooling layer → 2048-dim vector.
3. **Fit Gaussians**: compute $(\mu_r, \Sigma_r)$ on real features, $(\mu_g, \Sigma_g)$ on generated features.
4. **Compute matrix square root**: $(\Sigma_r \Sigma_g)^{1/2}$ via eigendecomposition.
5. **Plug into formula above** — result is FID.

### 3.2 Inception Score (IS)

Uses marginal $p(y)$ and conditional $p(y \mid x)$ from the Inception classifier:

$$
\text{IS} = \exp \left(\mathbb{E}_x\bigl[D_\text{KL}(p(y|x)\|p(y))\bigr]\right)
$$

- **Higher = better** (sharp images → high $p(y|x)$; diverse images → high entropy $p(y)$).
- **Does not compare to real images** — a model memorising training data can achieve high IS.
- Rarely used alone after FID became standard.

### 3.3 CLIP Score

Given generated image $x$ and its text prompt $t$:

$$
\text{CLIP Score} = w \cdot \max(0, \cos(\text{CLIP}_I(x), \text{CLIP}_T(t)))
$$

where $w = 2.5$ is a scaling constant (originates from CLIPScore paper, Hessel et al. 2021).

- **Higher = better** (>0.25 = prompt-aligned for VisualForge briefs).
- Reference-free: no real image needed.
- The CLIP embedding space is **shared** across images and text, so cosine similarity measures semantic alignment.

> 🏭 **Industry standard — CLIP model selection:** OpenAI's `clip-vit-base-patch32` (ViT-B/32) is the reproducibility standard — fastest, widely benchmarked. `clip-vit-large-patch14` (ViT-L/14) gives 5–10% higher scores but 3× slower. **Do not mix models across experiments** — ViT-L/14 score of 0.28 ≠ ViT-B/32 score of 0.28. Midjourney uses ViT-L/14 for internal evaluation; Stable Diffusion papers report ViT-B/32 for reproducibility. Load via HuggingFace: `CLIPModel.from_pretrained('openai/clip-vit-base-patch32')`.

**How you compute it:**
1. Encode the prompt with `CLIPTextEncoder` → $\mathbf{t} \in \mathbb{R}^{512}$.
2. Encode the generated image with `CLIPImageEncoder` → $\mathbf{v} \in \mathbb{R}^{512}$.
3. Normalise both to unit length.
4. Score = $2.5 \cdot \max(0, \mathbf{t} \cdot \mathbf{v})$.

### 3.4 LPIPS (Learned Perceptual Image Patch Similarity)

Compare a generated image $\hat{x}$ to a reference $x$:

$$
\text{LPIPS}(\hat{x}, x) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\hat{y}^l_{hw} - y^l_{hw}) \|^2_2
$$

- $y^l$: VGG/AlexNet/SqueezeNet feature map at layer $l$, channel-normalised.
- $w_l$: learned channel weights.
- **Lower = more perceptually similar** to reference.
- Used for img2img tasks (e.g., inpainting quality).

### 3.5 Precision & Recall for Generative Models

Kynkäänniemi et al. 2019 formulation using $k$-NN manifold estimation:

- **Precision**: fraction of generated samples inside the real manifold (fidelity)
- **Recall**: fraction of real samples covered by the generated manifold (diversity)

$$
\text{Precision} = \frac{1}{|X_g|}\sum_i \mathbf{1}[x_{g,i} \in \text{manifold}(X_r)]
$$
**Connection to Constraint #1 (Quality):** HPSv2 score of 4.1/5.0 = exceeds target → freelancer replacement validated.

---

## 4 · Visual Intuition — Choosing the Right Metric

```
 GENERATIVE EVALUATION LANDSCAPE
 ─────────────────────────────────

 Reference-free Reference-based
 (no real images needed) (compares to real distribution)

 ┌─────────────────────┐ ┌──────────────────────────────┐
 │ CLIP Score │ │ FID (distribution match) │
 │ text ↔ image align │ │ IS (fidelity + diversity) │
 │ HPSv2, ImageReward │ │ Precision / Recall │
 │ (human preference) │ │ LPIPS (pixel-level, per img) │
 └─────────────────────┘ └──────────────────────────────┘

 Sample-level Distribution-level
 (per image score) (needs thousands of images)

 ┌─────────────────────┐ ┌──────────────────────────────┐
 │ LPIPS │ │ FID, IS, Precision/Recall │
 │ SSIM, PSNR │ │ (stable only with N≥5k) │
 │ CLIP Score │ └──────────────────────────────┘
 └─────────────────────┘
```

```
FID BIAS VS SAMPLE COUNT
─────────────────────────
FID
 ↑
300│ × N=100 ← unstable, don't trust this
200│ × N=500
100│ × N=1k
 50│ × N=5k ← starts to stabilise
 20│ × N=50k ← production-grade estimate
 └───────────────────────────→ N (log scale)

True FID attained only at large N; small N inflates FID.
```

**Why this matters:** You need to generate ≥5,000 test images to get a reliable FID score. Running FID on 100 samples will give you wildly inconsistent results (±50 FID variance). VisualForge evaluates on 500-image batches = minimum viable N for campaign-level decisions.

> 🏭 **Industry standard — Sample size requirements:** Google's Imagen uses N=30k (COCO-2014 validation set) for FID reporting. OpenAI DALL-E 3 uses N=30k. Stable Diffusion XL uses N=50k (COCO-2014 train subset). Academic papers often report N=5k (minimum for ±5 FID variance) with a disclaimer. **For A/B testing** (Model A vs Model B), N=1k is acceptable — the absolute FID is biased but the *difference* between models is stable. **For production go/no-go decisions**, use N≥10k to defend the threshold to stakeholders.

---

## 5 · Production Example — Running the Evaluation Pipeline

**Automated quality gate for spring-collection batch (100 product images)**

```python
# Production: FID + CLIP Score evaluation for VisualForge campaign batch
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import glob

# --- FID: compare generated batch to reference product corpus ---
fid = FrechetInceptionDistance(feature=2048, normalize=True)

# Reference: 500 approved product images from previous campaigns
ref_images = [Image.open(f).resize((299, 299)) for f in glob.glob("vf_reference/*.png")[:500]]
gen_images = [Image.open(f).resize((299, 299)) for f in glob.glob("vf_generated/*.png")]

def to_tensor_batch(images: list, batch_size=50):
 import torchvision.transforms.functional as TF
 return torch.stack([TF.to_tensor(img) for img in images])

fid.update(to_tensor_batch(ref_images), real=True)
fid.update(to_tensor_batch(gen_images), real=False)
fid_score = fid.compute().item()
print(f"FID: {fid_score:.1f} (target <50 for campaign quality)")

# --- CLIP Score: brief compliance check ---
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

brief_prompt = "product on white background, studio lighting, e-commerce photography"
clip_scores = []
for img_path in glob.glob("vf_generated/*.png")[:20]: # spot-check 20
 img = Image.open(img_path)
 inputs = processor(text=[brief_prompt], images=[img], return_tensors="pt", padding=True)
 with torch.no_grad():
 outputs = model(**inputs)
 score = outputs.logits_per_image.item() / 100 # normalised to 0–1
 clip_scores.append(score)
print(f"Mean CLIP Score: {sum(clip_scores)/len(clip_scores):.3f} (target >0.25)")
```

**VisualForge evaluation scorecard (spring-collection batch):**

| Metric | Target | Result | Interpretation |
|--------|--------|--------|----------------|
| FID vs. reference corpus | <50 | 42.3 | Generated images statistically similar to approved products |
| CLIP Score (brief match) | >0.25 | 0.31 | Images correctly depict brief prompt |
| Class recall (brief types) | ≥0.9 per type | 0.93 product-on-white, 0.87 lifestyle | All campaign types represented |
| Manual QA pass rate | >80% | 84% | Creative director sign-off rate |

> **Gate decision**: FID 42.3 < 50 threshold and CLIP 0.31 > 0.25 threshold — batch approved for creative review. This automated gate saves ~2 hours of manual review per 100-image batch.

> 🏭 **Industry standard — Multi-metric dashboards:** Weights & Biases (W&B) is the production standard for metric tracking. Log metrics with `wandb.log({'fid': fid_score, 'clip_score': clip_mean, 'hpsv2': hps_score})` and visualize trends across checkpoints. Stability AI's internal dashboard tracks FID + CLIP + Aesthetic Score + HPSv2 for every SDXL training run (logged every 5k steps). OpenAI logs to internal tools but published DALL-E 3 metrics follow the same multi-metric pattern. **Key insight:** Single-metric optimization ("minimize FID only") leads to mode collapse — always track ≥2 orthogonal metrics (fidelity + alignment or fidelity + aesthetics).

---

## 6 · Common Failure Modes — When Metrics Mislead

**Failure-first pattern:** Evaluation metrics can mislead you — here's how they break and what to watch for.

### Failure Mode 1: FID Overfitting

| What goes wrong | Reality |
|---------------|---------||
| "Lower FID is always better" | FID measures *match* to the training distribution; a model overfitting real images can get near-zero FID but zero diversity. You want FID <50, not FID=0. |

**Debug:** Compare FID with Precision/Recall. High precision + low recall = overfitting (matches training set but lacks diversity).

### Failure Mode 2: CLIP Score Misalignment

| What goes wrong | Reality |
|---------------|---------||
| "CLIP Score measures photorealism" | CLIP Score measures text-image alignment, not visual quality. A blurry image with correct colours/objects can score high. |

**Debug:** Use CLIP Score + HPSv2 together. CLIP validates prompt adherence; HPSv2 validates human preference for quality.

### Failure Mode 3: Small Sample Bias

| What goes wrong | Reality |
|---------------|---------||
| "FID on 1,000 samples is reliable" | FID has O(1/√N) variance; ±10 FID spread is common at N=1k. Production decisions need N≥5k for stable estimates. |

**Debug:** Run FID on 3 independent 1k-sample batches. If spread >±5 FID, increase N.

### Failure Mode 4: Metric Confusion

| What goes wrong | Reality |
|---------------|---------||
| "IS is equivalent to FID" | IS doesn't compare to real images at all — it only uses the generator's class distribution. A model memorizing training data can achieve high IS with zero generalization. |
| "LPIPS = SSIM" | LPIPS uses deep network features (learned); SSIM is a hand-crafted pixel similarity. LPIPS correlates better with human perception. |
| "CLIP embeddings are perceptually uniform" | CLIP can match text to semantically wrong images if colours/textures align spuriously (e.g., "red apple" → red car if both have red dominant hue). |

---

## 7 · When to Use This vs Alternatives

**Decision framework for VisualForge evaluation:**

| Your question | Use this | Not this | Why |
|--------------|----------|----------|-----|
| Are generated images photorealistic? | FID vs real corpus | IS | FID compares to real images; IS doesn't |
| Does output match text prompt? | CLIP Score | FID | CLIP measures text-image alignment; FID measures distribution similarity |
| Will clients like this? | HPSv2 / ImageReward | FID, CLIP Score | HPSv2 trained on human preferences; automated metrics miss subjective quality |
| How many samples do I need? | N=5k for FID | N=1k "good enough" | FID variance is O(1/√N); 1k samples = ±10 FID noise |
| Spot-check 20 images | CLIP Score | FID | FID needs thousands of samples; CLIP works per-image |
| Compare two models | FID + HPSv2 + CLIP | Any single metric | No metric captures fidelity + diversity + alignment alone |

**VisualForge production stack:**
- **Batch quality gate (100 images)**: FID <50, CLIP >0.25 → auto-approve
- **Model A/B test (5k images)**: FID + HPSv2 → choose winning model
- **Per-brief validation (20 images)**: CLIP Score + manual QA spot-check

---

## 8 · Connection to Prior Chapters

**This chapter closes the loop on Quality (Constraint #1)** by proving what earlier chapters built:

| Chapter | What it enabled | How evaluation validates it |
|---------|----------------|-----------------------------|
| Ch.3 CLIP | Text-image embedding space | CLIP Score uses Ch.3 embeddings to measure prompt adherence → validates that conditioning works |
| Ch.4 Diffusion | Generative capability | FID compares generated distribution to real → validates that denoising produces realistic outputs |
| Ch.5 Schedulers | Speed optimization (1000 → 50 steps) | HPSv2 score stays 4.1/5.0 with 50 steps → validates that fewer steps don't degrade quality |
| Ch.6 Latent Diffusion | Latent space compression | FID on latent-diffusion outputs <50 → validates 8× compression doesn't lose fidelity |
| Ch.7 Guidance | CFG scale tuning | CLIP Score 0.31 at scale 7.5 → validates that guidance improves alignment |
| Ch.10 Multimodal LLM | Image understanding for QA | Automated QA pass rate 84% → validates VLM can filter unusable outputs |

**Key unlock:** Before this chapter, you had to trust that your generations "looked good." Now you have objective proof: HPSv2=4.1/5.0 exceeds freelancer baseline (4.0 target). VisualForge can replace $600k/year of human work with confidence.
**Forward pointer:** Next chapter (Ch.12) optimizes speed (18s → 8s with SDXL-Turbo) while maintaining this 4.1/5.0 quality — evaluation metrics prove the optimization doesn't degrade outputs.

---

## 9 · Interview Checklist

### Must Know
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

### Traps to Avoid
- Confusing CLIP Score (semantic alignment) with FID (distributional realism).
- Reporting FID on < 5k samples without flagging the bias.
- Conflating LPIPS (reference-based perceptual similarity) with CLIP Score (reference-free text alignment).
- Forgetting that FID is scale-sensitive: spatial resolution must match between real and generated sets.
- **Video generation metrics:** FVD (Fréchet Video Distance) extends FID to video using an I3D 3D-CNN feature extractor; captures temporal coherence, not just per-frame quality. CLIPSIM averages CLIP Score across frames — measures text alignment but ignores temporal consistency. VBench is the current standardised suite (16 dimensions including subject consistency and motion smoothness). Trap: "high per-frame FID means good video" — per-frame FID ignores temporal coherence entirely; a strobing video can score well per-frame
- **Compositional text-to-image evaluation:** standard FID/CLIP Score miss attribute binding failures ("a red cube and blue sphere" where colours are swapped). GenEval and T2I-CompBench specifically test spatial relations, attribute-object binding, and counting. Trap: "CLIP Score captures compositional accuracy" — CLIP Score is a global semantic similarity; it cannot verify fine-grained binding and scores an image with swapped attributes almost identically to the correct one

---

## 10 · Further Reading

### Foundational Papers

- **FID** — Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (NIPS 2017) | [arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500) — The paper that made FID the standard metric; Fréchet distance formulation and Inception-v3 feature extraction.
- **Inception Score** — Salimans et al. "Improved Techniques for Training GANs" (2016) | [arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498) — First widely-used automatic metric; explains KL divergence formulation.
- **CLIP Score** — Hessel et al. "CLIPScore: A Reference-free Evaluation Metric for Image Captioning" (EMNLP 2021) | [arxiv.org/abs/2104.08718](https://arxiv.org/abs/2104.08718) — Adapts CLIP to image generation evaluation; explains 2.5× scaling constant.
- **HPSv2** — Wu et al. "Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis" (2023) | [arxiv.org/abs/2306.09341](https://arxiv.org/abs/2306.09341) — Human preference model trained on pairwise comparisons; correlates better with subjective quality than FID/CLIP.

### Compositional & Advanced Metrics

- **GenEval** — Ghosh et al. "GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment" (NeurIPS 2023) | [arxiv.org/abs/2310.11513](https://arxiv.org/abs/2310.11513) — Tests attribute binding, spatial relations, counting.
- **Precision & Recall** — Kynkäänniemi et al. "Improved Precision and Recall Metric for Assessing Generative Models" (NeurIPS 2019) | [arxiv.org/abs/1904.06991](https://arxiv.org/abs/1904.06991) — Separates fidelity (precision) from diversity (recall).
- **LPIPS** — Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (CVPR 2018) | [arxiv.org/abs/1801.03924](https://arxiv.org/abs/1801.03924) — Learned perceptual similarity metric.

### Video & Multimodal

- **FVD** — Unterthiner et al. "Towards Accurate Generative Models of Video: A New Metric & Challenges" (2018) | [arxiv.org/abs/1812.01717](https://arxiv.org/abs/1812.01717) — Extends FID to video using I3D features.
- **VBench** — Huang et al. "VBench: Comprehensive Benchmark Suite for Video Generative Models" (CVPR 2024) | [arxiv.org/abs/2311.17982](https://arxiv.org/abs/2311.17982) — 16-dimensional video evaluation suite.

### Implementations

- **`torchmetrics.image.fid`** — PyTorch implementation of FID | [lightning.ai/docs/torchmetrics](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html)
- **`clean-fid`** — PyTorch FID with improved feature extraction | [github.com/GaParmar/clean-fid](https://github.com/GaParmar/clean-fid)
- **`hpsv2`** — Official HPSv2 implementation | [github.com/tgxs002/HPSv2](https://github.com/tgxs002/HPSv2)

---

## 11 · Notebook

→ [`notebook.ipynb_solution.ipynb` (reference) or `notebook.ipynb_exercise.ipynb` (practice) (solution)](notebook.ipynb_solution.ipynb) | [`notebook.ipynb_solution.ipynb` (reference) or `notebook.ipynb_exercise.ipynb` (practice) (exercise)](notebook.ipynb_exercise.ipynb) — Compute FID, CLIP Score, and HPSv2 on a VisualForge campaign batch. Runs on laptop CPU (no GPU required for evaluation metrics).

> **Time estimate:** 10-15 minutes for 500-image batch on laptop CPU.

---

## 11.5 · Progress Check — What Have We Unlocked?

### Before This Chapter
- **Constraint #1 (Quality)**: ~3.9/5.0 via slow/expensive client surveys
- **VisualForge Status**: Cannot track quality improvements objectively

### After This Chapter
- **Constraint #1 (Quality)**: **4.1/5.0** → HPSv2 score on 500-image test set, exceeds 4.0 target!
- **VisualForge Status**: Automated metrics prove quality exceeds freelancer baseline (4.2/5.0 during ramp-up, now 4.1/5.0)

---

### Key Wins

1. **Objective measurement**: HPSv2 runs on 500 images in 10 minutes (vs 1-week manual survey)
2. **Quality validated**: 4.1/5.0 = exceeds 4.0 target (client surveys were during ramp-up, quality improved)
3. **A/B testing enabled**: Can now test guidance scales (7.5 vs 12.0), schedulers (DDIM vs DPM-Solver) objectively

---

### What's Still Blocking Production?

**Nothing critical** — all 6 constraints met! But **optimization opportunity**: System takes ~18s per image (target <30s = comfortable margin). Can we go faster? SDXL-Turbo promises 4-step sampling = 8 seconds. Hardware not fully optimized (FP16 vs INT8, etc.).

**Next unlock (Ch.12)**: **Local Diffusion Lab (Production Optimization)** — SDXL-Turbo deployment, quantization, production patterns. Final assembly of 12-chapter pipeline.

---

### VisualForge Status — Full Constraint View

| Constraint | Ch.1 | Ch.3 | Ch.6 | Ch.8 | Ch.10 | Ch.11 (This) | Target |
|------------|------|------|------|------|-------|--------------|--------|
| Quality | | | 3.5 | 3.8 | 3.9 | **4.1/5.0** | 4.0/5.0 |
| Speed | | | 20s | 18s | 18s | 18s | <30s |
| Cost | | | $2.5k | $2.5k | $2.5k | $2.5k | <$5k |
| Control | | | <15% | 3% | 3% | 3% | <5% |
| Throughput | | | | 80/day | 120/day | 120/day | 100+/day |
| Versatility | | | T2I | T2I+Video | All 3 | All 3 | 3 modes |

**Legend:** = Blocked | = Foundation laid | = Target hit

---

## Bridge to Chapter 12

**What's still blocking us?** Nothing critical — all 6 constraints are met! But there's an **optimization opportunity**: your system generates images in ~18 seconds (comfortably under the <30s target), but that leaves overhead for future feature creep. Can you go faster while maintaining 4.1/5.0 quality?

**The bottleneck:** SDXL uses 50 denoising steps. SDXL-Turbo promises 4-step sampling = 8 seconds. Hardware isn't fully optimized (running FP16; could use INT8 quantization). Deployment patterns are ad-hoc (manual Python scripts; should be Docker + ComfyUI workflows).

**Next unlock (Ch.12):** **Local Diffusion Lab (Production Optimization)** — Deploy SDXL-Turbo (4 steps → 8s), quantize to INT8, package as Docker container, set up ComfyUI for client-friendly UI. Final assembly of the 12-chapter VisualForge pipeline → production-ready system.

→ [LocalDiffusionLab.md](../ch13_local_diffusion_lab/local-diffusion-lab.md)

---

## Illustrations

![Generative evaluation - FID, CLIPScore, metric coverage, human eval pipeline](img/generative-evaluation.png)
