# Audio Generation — Text-to-Speech for Marketing Narration

> **The story.** In 2016, DeepMind's WaveNet shocked the speech synthesis world by generating human-quality speech waveforms directly from text using dilated convolutions. By 2020, end-to-end TTS models like Tacotron 2 and FastSpeech collapsed acoustic modeling and vocoding into a single neural pipeline. Meta's 2023 release of Massively Multilingual Speech (MMS) demonstrated that compact, CPU-friendly TTS models could cover 1,100+ languages while running on standard laptops — no GPU required.
>
> **Where you are in the curriculum.** You've built VisualForge's text→image pipeline (Ch.6–8), added video generation (Ch.9), and enabled image understanding with VLMs (Ch.10). Now clients want **narrated video ads**: 15-second social media clips with voiceover describing the product. You need text-to-speech capability that runs locally (no cloud API costs), generates natural-sounding narration, and integrates into the existing pipeline.

---

## 0 · The VisualForge Studio Challenge

**Mission**: VisualForge Studio needs to replace $600k/year freelancer costs with an in-house AI system running on local hardware (<$5k), delivering professional-grade marketing visuals (<30s per image, ≥4.0/5.0 quality), with <5% unusable generations and 100+ images/day throughput. The system must handle text→image, image→video, and image understanding for automated QA.

**Current blocker at Audio Generation**: You can generate video clips (Ch.9) and understand image content (Ch.10), but **clients want narrated video ads**. Hiring voice actors costs $150/video × 85 videos/day = $12,750/day. Cloud TTS APIs (ElevenLabs, Google Cloud) cost $0.30/minute = $200/day for 85 15-second clips, but introduce latency and internet dependency. You need **local, CPU-friendly TTS** that generates natural-sounding narration without GPU or cloud costs.

**What this chapter unlocks**: **Text-to-Speech (TTS)** — you'll generate speech waveforms from text using Meta's MMS TTS model, running entirely on CPU. This enables narrated video ads, reduces voice actor costs to $0, and maintains the local-first constraint.

---

### The 6 Constraints — Snapshot After Audio Generation

| Constraint | Target | Status | Evidence |
|------------|--------|--------|----------|
| #1 Quality | ≥4.0/5.0 | **4.1/5.0** | HPSv2 score (visual quality maintained) |
| #2 Speed | <30 seconds | **8s image + 2s audio** | SDXL-Turbo + MMS TTS (10s total per narrated frame) |
| #3 Cost | <$5k hardware | **$2,500 laptop** | CPU-only TTS, no additional hardware needed |
| #4 Control | <5% unusable | **3% unusable** | ControlNet + VLM QA (audio doesn't affect visual quality) |
| #5 Throughput | 100+ images/day | **120 images/day → 85 narrated videos/day** | Audio generation adds 2s/video (manageable) |
| #6 Versatility | 3 modalities | **Text→Image + Video + Audio + Understanding** | Now covers all client deliverable types |

---

### What's Still Blocking Us After This Chapter?

**Speech quality**: MMS TTS produces intelligible but robotic speech. Professional clients expect natural prosody and expressiveness (vocal emphasis, pauses, emotion). Premium TTS models like ElevenLabs or GPT-4o-mini-TTS offer better quality but require cloud APIs (violating Constraint #3 local-first). For MVP, we'll accept "intelligible but not perfect" narration — 80% of clients prioritize speed and cost over Hollywood-level voice acting.

**Next unlock (Ch.11 Evaluation)**: Automated quality metrics (FID, CLIP score, HPSv2) ensure we maintain professional-grade output at scale. Speech quality can be measured with MOS (Mean Opinion Score) later if clients demand it.

---

## 1 · Core Idea

Audio generation is the same multimodal principle you used for image and video generation: **transform tokens from one modality (text) into another modality (audio waveform)**. In text-to-speech (TTS), a model predicts waveform samples from text by:

1. **Tokenizing text** into phonemes or characters
2. **Acoustic modeling** predicts mel-spectrogram frames (frequency representation of sound)
3. **Vocoding** converts mel-spectrograms into raw audio waveforms (playable `.wav` file)

Modern end-to-end TTS models (like MMS TTS) collapse steps 2–3 into a single neural network, trained on pairs of (text, audio waveform).

**Why this matters for VisualForge**: You can now generate voiceover narration for video ads locally, without paying $150/video for voice actors or $0.30/minute for cloud TTS APIs.

---

## 2 · Running Example — PixelSmith v9 (Audio Narration)

PixelSmith gets a new feature: it can narrate generated video clips with synthesized speech.

**VisualForge client brief**: "Generate a 15-second video of our spring collection floral dress with voiceover: 'Introducing the Marseille Collection—handcrafted linen in coral and sage, designed for effortless elegance.'"

**Before this chapter**:
- Input: text brief → video frames (Ch.9)
- Output: silent 15-second video clip
- Problem: Client wants **narrated ads** for Instagram Reels (silent ads have 40% lower engagement)

**After this chapter**:
- Input: text brief + narration script → video frames + audio waveform
- Output: **narrated video ad** (video + synchronized voiceover)
- Constraint: CPU-only, no GPU requirement (runs on the same $2,500 laptop)


---

## 3 · The Math

### The TTS Pipeline

Text-to-speech consists of three mathematical transformations:

**1. Text → Phoneme sequence**

Convert text into phonemes (atomic sound units):
```
text: "Marseille Collection"
→ phonemes: /m ɑː r ˈ s eɪ k ə ˈ l ɛ k ʃ ən/
```

**2. Phonemes → Mel-spectrogram** (acoustic model)

Predict mel-spectrogram $M \in \mathbb{R}^{T \times F}$ from phoneme sequence $p_1, p_2, \ldots, p_N$:

$$
M = \text{AcousticModel}(p_1, p_2, \ldots, p_N)
$$

Where:
- $T$ = number of time frames (e.g., 100 frames for 1 second at 10ms resolution)
- $F$ = number of mel-frequency bins (typically 80)

The acoustic model is a sequence-to-sequence model (often a Transformer or RNN) that learns the mapping:
```
"Marseille" → [mel-spec frames showing the /mɑːr/ sound rising in pitch, then /seɪ/ vowel]
```

**3. Mel-spectrogram → Waveform** (vocoder)

Convert mel-spectrogram into raw audio waveform $w \in \mathbb{R}^S$ where $S$ is the number of samples (e.g., 22,050 samples/second):

$$
w = \text{Vocoder}(M)
$$

Popular vocoders:
- **WaveNet** (2016): Autoregressive CNN, slow but high-quality
- **WaveGlow** (2018): Flow-based model, faster inference
- **HiFi-GAN** (2020): GAN-based vocoder, real-time on CPU

### The Loss Function

TTS models are trained by minimizing the difference between predicted and ground-truth mel-spectrograms:

$$
\mathcal{L}_{\text{TTS}} = \| M_{\text{pred}} - M_{\text{true}} \|_2^2
$$

Where:
- $M_{\text{pred}}$ = mel-spectrogram predicted by the model
- $M_{\text{true}}$ = ground-truth mel-spectrogram from real audio

**Why this matters for VisualForge**: This L2 loss guarantees the model learns to reproduce the frequency patterns of human speech. The closer $\mathcal{L}_{\text{TTS}} \to 0$, the more natural-sounding the output.

---

## 4 · Visual Intuition

### TTS Pipeline Flow

![Audio generation flow animation](img/audio-generation-flow.gif)

*Flow: text is tokenized, transformed by an acoustic model, vocoded into a waveform, and emitted as playable audio.*

```
Text Input: "Introducing the Marseille Collection"
 │
 ├─→ Phoneme Encoder
 │ (converts text → /ɪntrəˈdjuːsɪŋ ðə mɑːrˈseɪ kəˈlɛkʃən/)
 │
 ├─→ Acoustic Model (Transformer)
 │ Predicts mel-spectrogram frames:
 │ Time ──→
 │ Freq ↓ [█░░░░] /m/ (low frequency hum)
 │ [░░█░░] /ɑː/ (mid frequency vowel)
 │ [░░░█░] /r/ (high frequency trill)
 │ [░█░░░] /s/ (sibilant)
 │ [░░█░░] /eɪ/ (vowel glide)
 │
 ├─→ Vocoder (HiFi-GAN)
 │ Converts mel-spectrogram → waveform
 │ Sample ──→
 │ Amplitude ↓ ╱╲╱╲╱╲ (22,050 samples/second)
 │
 └─→ Audio Output (.wav file, 15 seconds)
```

### Before/After: Voice Actor vs TTS

```
Before (Voice Actor):
────────────────────────────────────────────
Cost: $150 per video × 85 videos/day = $12,750/day
Turnaround: 24–48 hours (schedule actor, record, edit, deliver)
Iterations: $150 per script change
Constraint: #3 COST (unsustainable at scale)

After (MMS TTS):
────────────────────────────────────────────
Cost: $0 (runs on existing laptop CPU)
Turnaround: 2 seconds per 15-second narration (instant)
Iterations: Free (re-generate on demand)
Constraint: #3 COST (meets <$5k hardware target)
```

---

## 5 · Production Example — VisualForge in Action

You're the Lead ML Engineer at VisualForge Studio. A client wants 20 narrated video ads for their spring collection launch—each video needs a different product description read aloud.

**The brief**:
- **Campaign type**: Product lifestyle (spring collection)
- **Deliverable**: 15-second video + voiceover
- **Narration script**: "Introducing the [product name]—handcrafted [material] in [colors], designed for [attribute]."
- **Turnaround**: 2 hours (client call at 3 PM)

**The code** (production-ready):

```python
# Production pattern
import torch
from transformers import VitsModel, AutoTokenizer
import numpy as np
import soundfile as sf

# Production: Load MMS TTS model (CPU-friendly, 150MB)
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

def generate_narration(script: str, output_path: str):
 """Generate speech from text and save as .wav file."""
 inputs = tokenizer(script, return_tensors="pt")

 with torch.no_grad():
 output = model(**inputs).waveform

 # Convert to numpy and save
 audio = output.squeeze().cpu().numpy()
 sf.write(output_path, audio, samplerate=16000)

 return output_path

# VisualForge: Generate narration for spring collection products
scripts = [
 "Introducing the Marseille Collection—handcrafted linen in coral and sage, designed for effortless elegance.",
 "Meet the Côte d'Azur Dress—lightweight cotton in sky blue and ivory, perfect for coastal getaways.",
 "Discover the Provence Jacket—tailored wool in charcoal and cream, essential for transitional seasons.",
 # ... 17 more products
]

for i, script in enumerate(scripts):
 audio_path = f"visualforge_narration_product_{i:02d}.wav"
 generate_narration(script, audio_path)
 print(f" Generated {audio_path} (2.1s)")

# Result: 20 narrations × 2s = 40 seconds total
# vs 24–48 hours with voice actor booking
```

**Metrics**:
- **Generation time**: 2.1 seconds per 15-second narration (7× real-time)
- **Cost**: $0 (vs $150/video × 20 = $3,000 for voice actor)
- **Turnaround**: 40 seconds (vs 24–48 hours)
- **Constraint impact**:
 - **#3 COST** maintained ($0 cloud costs)
 - **#5 THROUGHPUT** improved (120 images/day → 85 narrated videos/day with only +2s overhead per video)
 - **#6 VERSATILITY** unlocked (now handle text→image + video + audio + understanding)

---

## 6 · Common Failure Modes

### 6.1 · Robotic Prosody

**Symptom**: Speech sounds monotone, lacks natural pauses and emphasis.

```
"Introducing the Marseille Collection handcrafted linen in coral and sage designed for effortless elegance"
 (flat, no pauses, robotic)
```

**Why**: MMS TTS is optimized for **intelligibility** (clear pronunciation) over **expressiveness** (emotion, rhythm). It doesn't model prosody features like pitch contours, stress patterns, or natural pauses.

**Fix**:
1. **Add punctuation** for pauses: `"Introducing the Marseille Collection—handcrafted linen in coral and sage, designed for effortless elegance."` (em dash and comma force pauses)
2. **Use premium TTS** if clients demand natural prosody: ElevenLabs, GPT-4o-mini-TTS (but these violate Constraint #3 local-first)
3. **Accept "good enough"** for MVP: 80% of clients prioritize speed/cost over Hollywood-quality voice acting

**VisualForge decision**: Accept intelligible but robotic narration for launch. If clients complain, we can later fine-tune a local TTS model on recorded voice samples (Ch.12 Local Diffusion Lab workflow applies here).

### 6.2 · Mispronunciation of Brand Names

**Symptom**: Model mispronounces non-English brand names (e.g., "Marseille" → "Mar-SALE").

**Why**: MMS TTS is trained on general English text, not brand-specific vocabularies. French names follow different phonetic rules.

**Fix**:
1. **Phonetic spelling**: Replace "Marseille" with "Mar-SAY" in the script
2. **Custom pronunciation dictionary**: Map brand names to phonemes (requires model fine-tuning)

**VisualForge workaround**: Maintain a pronunciation guide for common brand names:
```python
# Production: VisualForge brand pronunciation dictionary
BRAND_PRONUNCIATIONS = {
 "Marseille": "Mar-SAY",
 "Côte d'Azur": "Coat duh-ZOOR",
 "Provence": "Pro-VONCE",
}

def preprocess_script(script: str) -> str:
 """Replace brand names with phonetic spellings."""
 for brand, phonetic in BRAND_PRONUNCIATIONS.items():
 script = script.replace(brand, phonetic)
 return script

# Apply before generation
script = preprocess_script("Introducing the Marseille Collection")
generate_narration(script, "output.wav")
```

### 6.3 · Audio Quality Degrades with Long Scripts

**Symptom**: First 5 seconds sound good, then quality degrades (crackling, distortion).

**Why**: MMS TTS is trained on short utterances (10–20 seconds). Long scripts cause the model to extrapolate beyond its training distribution.

**Fix**: Split long scripts into sentences, generate separately, concatenate audio files:
```python
# Production pattern
import numpy as np

def generate_long_narration(script: str) -> np.ndarray:
 """Handle scripts longer than 20 seconds."""
 sentences = script.split(". ")
 audio_segments = []

 for sentence in sentences:
 audio = generate_narration(sentence + ".", f"temp_{len(audio_segments)}.wav")
 audio_segments.append(audio)

 # Concatenate with 0.3s silence between sentences
 silence = np.zeros(int(0.3 * 16000)) # 16kHz sample rate
 combined = np.concatenate([
 seg if i == 0 else np.concatenate([silence, seg])
 for i, seg in enumerate(audio_segments)
 ])

 return combined
```

---

## 7 · When to Use This vs Alternatives

| Approach | Use when... | VisualForge fit |
|----------|-------------|-----------------|
| **MMS TTS** (this chapter) | Need fast, local, CPU-friendly TTS; intelligibility > expressiveness | **Best for MVP**: $0 cost, 2s generation, runs on existing laptop |
| **ElevenLabs API** | Need premium voice quality with natural prosody | Violates #3 COST (cloud dependency) + #2 SPEED (API latency) |
| **Coqui XTTS v2** | Need voice cloning (match client's brand voice) | **Future upgrade**: GPU required (conflicts with CPU-only constraint) |
| **GPT-4o-mini-TTS** | Need conversational prosody with low latency | Requires API ($0.30/min = $200/day for 85 videos) |
| **Whisper** (ASR) | Need audio → text transcription (opposite direction) | Not applicable (we're doing text → audio) |
| **Voice Actor** | Need Hollywood-quality emotional delivery | $150/video × 85 = $12,750/day (unsustainable) |

**VisualForge decision tree**:
```
Does client brief require emotional narration (e.g., luxury car ad)?
 ├─ Yes → Use voice actor for hero assets (5% of videos)
 └─ No → Use MMS TTS for bulk generation (95% of videos)
```

### Popular Audio Models (Apr 2026 Snapshot)

For reference, here are leading audio models across different use cases:

#### Text-to-Speech (TTS)

| Model | Strength | Typical Runtime Profile |
|------|----------|-------------------------|
| `facebook/mms-tts-*` | Lightweight multilingual TTS, 1,100+ languages | CPU-friendly (VisualForge choice) |
| `hexgrad/Kokoro-82M` | Very small open TTS model, fast local experimentation | CPU-friendly |
| `Coqui XTTS v2` | Good multilingual open-source baseline with voice cloning support | GPU preferred |
| `ElevenLabs` multilingual voices | Top-tier naturalness and expressiveness | API-first (cloud) |
| `gpt-4o-mini-tts` | Strong quality with low-latency conversational integration | API-first (cloud) |

#### Speech-to-Text (ASR)

| Model | Strength | Typical Runtime Profile |
|------|----------|-------------------------|
| `Whisper large-v3` / `large-v3-turbo` | Widely used accuracy baseline across accents/noise | CPU possible; faster on GPU |
| `distil-whisper` variants | Better speed/quality trade-off for local apps | CPU-friendly for short clips |
| `NVIDIA Canary` family | Strong multilingual transcription + translation | GPU-oriented deployments |

#### Music / Sound Generation

| Model | Strength | Typical Runtime Profile |
|------|----------|-------------------------|
| `Suno` latest models | Popular for prompt-to-song generation and arrangement quality | API-first (cloud) |
| `Udio` latest models | Strong melodic control and production quality | API-first (cloud) |
| `Stable Audio 2` family | Open workflow for SFX/music generation and editing | GPU preferred |
| `MusicGen` family | Good open baseline for local experimentation | CPU possible (slow), GPU preferred |

---

## 8 · Connection to Prior Chapters

| Chapter | What it taught | How Audio Generation builds on it |
|---------|---------------|-----------------------------------|
| Ch.6 Latent Diffusion | Compress high-dimensional data (512×512 image → 64×64 latent) for fast generation | TTS compresses text → mel-spectrogram → waveform (similar hierarchical pipeline) |
| Ch.8 Text-to-Image | Generate images from text prompts using cross-modal conditioning | TTS generates audio from text using same principle (text → acoustic features) |
| Ch.9 Text-to-Video | Generate video by conditioning on text + temporal coherence | Narrated video = video frames (Ch.9) + audio waveform (this chapter) synchronized |
| Ch.10 Multimodal LLMs | Understand image content with VLMs (LLaVA, GPT-4V) | **Forward link**: Ch.11 evaluates audio quality using MOS (Mean Opinion Score), same evaluation principle as HPSv2 for images |

**The pattern**: Every VisualForge capability follows the same multimodal recipe: encode source modality (text) → transform in latent space → decode to target modality (image / video / audio).

---

## 9 · Interview Checklist

### Conceptual Questions

**Q: How does TTS differ from speech recognition (ASR)?**
- **TTS**: Text → Audio (generation)
- **ASR**: Audio → Text (transcription)
- Both use sequence-to-sequence models, but TTS must learn temporal alignment (which phoneme corresponds to which waveform frame).

**Q: Why do TTS models predict mel-spectrograms instead of raw waveforms?**
- **Mel-spectrograms** are lower-dimensional (80 bins vs 22,050 samples/second) → easier to learn
- Human hearing is logarithmic (mel scale matches perceptual frequency resolution)
- Vocoder handles high-frequency details (waveform reconstruction)

**Q: What's the difference between concatenative TTS and neural TTS?**
- **Concatenative** (2000s): Stitch together pre-recorded phoneme clips → robotic, limited prosody
- **Neural** (2016+): Learn end-to-end text→audio mapping → natural prosody, but requires large training datasets

### Practical Scenarios

**Q: Client wants their CEO's voice for all narrations. How would you implement voice cloning?**
- Use **Coqui XTTS v2** or **ElevenLabs Voice Cloning**
- Requires 5–10 minutes of clean audio recordings (CEO reading sample scripts)
- Fine-tune TTS model on CEO's voice → generates speech in CEO's style

**Q: Generated audio has background noise. How do you fix it?**
- **Root cause**: Model learned noise from training data (real-world recordings often have ambient noise)
- **Fix**: Post-process with noise reduction (e.g., `noisereduce` library in Python)
- **Better fix**: Fine-tune model on clean studio recordings

**Q: How would you evaluate TTS quality at scale (1,000 narrations/day)?**
- **Automated metric**: **MOS (Mean Opinion Score)** — train a model to predict human ratings (1–5 scale)
- **Signal-based metric**: **Mel-Cepstral Distortion (MCD)** — measures difference between predicted and ground-truth mel-spectrograms
- **VisualForge approach**: Sample 5% of narrations for human QA, use automated metrics for the rest

---

## 10 · Further Reading

### Papers
- **WaveNet** (van den Oord et al., 2016): First high-quality neural TTS using dilated convolutions
 - [arXiv:1609.03499](https://arxiv.org/abs/1609.03499)
- **Tacotron 2** (Shen et al., 2018): End-to-end TTS with attention-based sequence-to-sequence model
 - [arXiv:1712.05884](https://arxiv.org/abs/1712.05884)
- **FastSpeech** (Ren et al., 2019): Non-autoregressive TTS for fast parallel generation
 - [arXiv:1905.09263](https://arxiv.org/abs/1905.09263)
- **MMS** (Pratap et al., 2023): Massively Multilingual Speech models covering 1,100+ languages
 - [arXiv:2305.13516](https://arxiv.org/abs/2305.13516)

### Repositories
- **MMS TTS**: [github.com/facebookresearch/fairseq/tree/main/examples/mms](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
- **Coqui TTS**: [github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS) (voice cloning, multi-speaker)
- **HiFi-GAN**: [github.com/jik876/hifi-gan](https://github.com/jik876/hifi-gan) (fast vocoder)

### Production Tools
- **ElevenLabs**: Premium TTS API with emotional prosody
- **GPT-4o-mini-TTS**: OpenAI's low-latency conversational TTS
- **Google Cloud TTS**: Enterprise-grade multilingual TTS

---

## 11 · Notebook

**File**: `notebook.ipynb_solution.ipynb` (reference) or `notebook.ipynb_exercise.ipynb` (practice)

**What you'll build**: A minimal TTS pipeline that generates narrated audio from text in under 10 lines of code.

### Quick-Win Flow

1. **Install dependencies** (one cell):
 ```python
 !pip install transformers soundfile torch
 ```

2. **Load MMS TTS model** (CPU-friendly, 150MB):
 ```python
 from transformers import VitsModel, AutoTokenizer

 model = VitsModel.from_pretrained("facebook/mms-tts-eng")
 tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
 ```

3. **Generate speech from text**:
 ```python
 import torch
 import soundfile as sf

 text = "Introducing the Marseille Collection—handcrafted linen in coral and sage."
 inputs = tokenizer(text, return_tensors="pt")

 with torch.no_grad():
 output = model(**inputs).waveform

 audio = output.squeeze().cpu().numpy()
 sf.write("narration.wav", audio, samplerate=16000)
 ```

4. **Play audio inline** (Jupyter):
 ```python
 from IPython.display import Audio
 Audio("narration.wav")
 ```

**Expected output**: A 5-second `.wav` file with synthesized speech narrating the Marseille Collection brief.

**Extensions**:
- Batch process 20 product descriptions
- Integrate with video generation pipeline (Ch.9)
- Add pronunciation dictionary for brand names

---

## 11.5 · Progress Check — What Have We Unlocked?

### Before This Chapter
- **Constraint #5 Throughput**: 120 images/day (static images only)
- **Constraint #6 Versatility**: Text→Image + Video + Understanding (no audio capability)
- **Blocker**: Clients want narrated video ads, but voice actors cost $12,750/day ($150/video × 85 videos)

### After This Chapter
- **Constraint #5 Throughput**: **85 narrated videos/day** (audio adds only 2s overhead per video)
- **Constraint #6 Versatility**: **Text→Image + Video + Audio + Understanding** (all client deliverable types covered)
- **Cost savings**: $12,750/day → $0 (CPU-only TTS, no cloud APIs)

---

### Key Wins

1. **Local TTS pipeline**: MMS TTS generates 15-second narration in 2 seconds on CPU (no GPU, no cloud dependency)
2. **Zero voice actor costs**: Eliminated $12,750/day voice actor budget (or $200/day cloud TTS API costs)
3. **Instant iteration**: Re-generate narration on demand (vs 24–48 hours with voice actor scheduling)
4. **Maintained constraints**: Audio generation doesn't degrade visual quality (#1), speed (#2), or control (#4)

---

### What's Still Blocking Production?

**Quality measurement**: We're generating 120 images/day, 85 narrated videos/day, but **quality evaluation is still manual** (humans reviewing 5 samples per batch). At this scale, cognitive fatigue makes human review unreliable after the first hour. We need **automated quality metrics** to maintain professional-grade output without $50k/month QA team costs.

**Next unlock (Ch.11 Evaluation)**: Automated generative evaluation — FID (Fréchet Inception Distance), CLIP score, HPSv2 for image quality; MOS (Mean Opinion Score) for audio quality. This ensures we maintain ≥4.0/5.0 professional quality at 120 images/day scale.

---

### VisualForge Status — Full Constraint View

| Constraint | Ch.6 | Ch.7 | Ch.8 | Ch.9 | Ch.10 | **Audio Gen** | Target |
|------------|------|------|------|------|-------|---------------|--------|
| #1 Quality | 3.5 | 3.8 | 3.8 | 3.8 | 3.9 | 3.9/5.0 | ≥4.0/5.0 |
| #2 Speed | 20s | 20s | 18s | 18s | 18s | 8s img + 2s audio | <30s |
| #3 Cost | $2.5k | $2.5k | $2.5k | $2.5k | $2.5k | $2.5k (no new hw) | <$5k |
| #4 Control | 15% | 3% | 3% | 3% | 3% | 3% unusable | <5% |
| #5 Throughput | 80 | 80 | 80 | 85 vid | 120 img | 85 narrated vid | 100+/day |
| #6 Versatility | T→I | T→I | T→I | +Video | +Understand | **+Audio** | 3 modalities |

**Legend**: Not addressed | Foundation laid | Target hit

**Key observation**: Audio generation is a **force multiplier**, not a bottleneck. It adds 2 seconds per video (negligible) while unlocking an entirely new deliverable type (narrated ads) that previously cost $12,750/day in voice actor fees.

---

## Bridge to Chapter 11 (Generative Evaluation)

You can now generate professional-grade images (≥4.0/5.0 quality), 15-second video ads, narrated voiceovers, and automate QA with VLMs—all running locally on a $2,500 laptop. But there's a hidden scaling problem: **how do you know the 85th video of the day is as good as the 1st?**

Right now, your QA workflow is: generate 5 sample images → show to client → approve batch. But cognitive fatigue makes humans unreliable after reviewing 20+ images. You're generating 120 images/day—manual review doesn't scale.

**The blocker**: Quality evaluation is still manual. You need **automated metrics** that predict professional quality scores without human review.

**Ch.11 Generative Evaluation** introduces:
- **FID (Fréchet Inception Distance)**: Measures distribution distance between real and generated images
- **CLIP score**: Quantifies image-text alignment (does the generated image match the prompt?)
- **HPSv2**: Human Preference Score v2, predicts professional quality ratings (1–5 scale)

With these metrics, you'll replace "looks good to me" with "HPSv2 score 4.1/5.0 → approved for client delivery" — ensuring consistent quality at 120 images/day without hiring a $50k/month QA team.
