# Exercise 05: PixelSmith AI — Multimodal Learning System

> **Learning Goal:** Implement CLIP contrastive learning and image captioning with zero-shot classification  
> **Prerequisites:** Completed [notes/05-multimodal_ai/](../../notes/05-multimodal_ai/)  
> **Time Estimate:** 8-10 hours (coding) + 2 hours (experimentation)  
> **Difficulty:** ⭐⭐⭐⭐ Advanced

> **Note:** This exercise focuses on core multimodal AI implementation. Infrastructure files (Docker, Makefile, monitoring) have been removed to streamline learning. For production deployment patterns, see exercises 06-07 (AI Infrastructure & DevOps).

---

## 0. Grand Challenge: VisualForge Studio

**The Mission:** You're the Lead ML Engineer at **Aperture Creative** — a boutique marketing agency that just lost 60% of its budget (3 freelancers @ $200k/year each). The Creative Director needs to maintain 2,000 assets/year output (product shots, social ads, explainer videos, voiceovers) with a skeleton crew. Your job: build **VisualForge Studio** — a local AI creative pipeline that replaces $600k/year freelancer costs while maintaining professional-grade quality on $5k hardware with zero cloud fees.

**Current Constraint Status:**
| Constraint | Target | Status | Why It's Blocking |
|------------|--------|--------|-------------------|
| **#1 Quality** | ≥4.0/5.0 HPSv2 score | ❌ | No generation capability — can't produce images at all |
| **#2 Speed** | <30s per 512×512 image | ❌ | No generation pipeline to measure |
| **#3 Cost** | <$5k hardware, $0/month cloud | ❌ | No local inference system built yet |
| **#4 Control** | <5% unusable generations | ❌ | Can't control what doesn't exist |
| **#5 Throughput** | 100+ assets/day sustained | ❌ | Zero assets/day (retrieval only) |
| **#6 Versatility** | 4 modalities working | ❌ | No generation — only retrieval and classification |

**What's Blocking Us:**

You've completed the notes for Multimodal Foundations, Vision Transformers, and CLIP. You understand patch embeddings, attention over images, and contrastive learning. But you **cannot generate a single image**. CLIP gives you text-image alignment and zero-shot classification — you can retrieve existing images from a database and classify them into arbitrary categories. That's powerful for search and classification tasks, but it doesn't produce the 2,000 new assets the agency needs.

The Creative Director's response: *"So you can find stock photos and tell me what's in them. Great. But I need you to **create** hero shots for the Q3 campaign. Can your AI do that?"*

Answer: Not yet. You're stuck at retrieval — you have no diffusion model, no latent space compression, no text-to-image generation pipeline. All 6 constraints remain unsatisfied.

**What This Exercise Unlocks:**

Implementing **CLIP contrastive learning** builds the foundation for constraint **#6 (Versatility)** — you'll learn how to:
- Project images and text into the same embedding space (cosine similarity > 0.7)
- Train vision-language models with InfoNCE loss (contrastive objective)
- Perform zero-shot classification without task-specific training
- Measure text-to-image retrieval accuracy (T2I/I2T metrics)

This exercise teaches you the **alignment layer** that will later enable text-conditioned generation in Stable Diffusion (Ch.6) and multimodal LLMs (Ch.10). You're laying the groundwork for the CLIP text encoder that converts "a dog running on a beach at sunset" into the conditioning vector that guides latent diffusion.

**Constraint Progress After This Exercise:** Still ❌❌❌❌❌❌ (no generation yet), but you've built the semantic alignment backbone that enables text→image conditioning in future chapters.

---

## 🎯 **What You'll Implement**

Starting from function stubs and inline TODOs, you'll build a complete vision-language system with:

### **Core Implementation (8-10 hours)**

| File | What You Implement | TODOs | Time |
|------|-------------------|-------|------|
| `src/features.py` | Image preprocessing, text tokenization, data loading | 7 methods | 2.5-3h |
| `src/models.py` | CLIP similarity, zero-shot, captioning, evaluation | 8 methods | 4-5h |
| `src/models.py` | ExperimentRunner for model comparison | 4 methods | 1h |
| `main.py` | Zero-shot demo, captioning demo | 2 sections | 30-60min |

**Interactive Experience:**
- ✅ See CLIP similarity scores immediately after computation
- ✅ Watch zero-shot classification probabilities in real-time
- ✅ Generate image captions with instant feedback
- ✅ Compare models with automatic leaderboards
- ✅ Rich console output with colors, tables, and metrics

**Total:** 8-10 hours of focused multimodal AI coding

---

## 🎓 **What You'll Learn**

### **Vision-Language Concepts**
- 🎯 **Zero-shot learning**: Classify images into ANY categories without task-specific training
- 🔗 **Contrastive learning**: Pull matched image-text pairs together, push unmatched apart
- 🧠 **Cross-attention**: Decoder attends to image features while generating captions
- 📐 **Joint embedding space**: Map images and text to shared representation space
- 🌐 **Vision Transformer (ViT)**: Patch-based image encoding for transformer models

### **Evaluation Metrics**
- **CLIP Score**: Image-text alignment quality (cosine similarity > 0.7)
- **T2I/I2T Accuracy**: Text/Image retrieval accuracy (target: > 70%)
- **BLEU**: N-gram overlap with reference captions (target: > 0.3)
- **CIDEr**: Consensus-based caption quality (target: > 1.0)
- **ROUGE-L**: Longest common subsequence (target: > 0.5)

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

### **2. Run Interactive Demo**

```bash
python main.py
```

**Expected output (after implementing TODOs):**
```
🔧 PREPROCESSING SETUP
  ✓ Image transform built: size=224, augment=False, normalize=True

🎯 ZERO-SHOT CLASSIFICATION DEMO
  Results:
┌─────────────────────────────┬─────────────┐
│ Label                       │ Probability │
│ a dog playing on the beach  │ 0.82        │
│ a dog                       │ 0.65        │
│ a beach                     │ 0.58        │
└─────────────────────────────┴─────────────┘

📝 IMAGE CAPTIONING DEMO
  📷 beach.jpg
  💬 a beautiful beach with turquoise water and white sand

🤖 MODEL COMPARISON
📊 CLIP LEADERBOARD
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ Model           │ T2I Accuracy │ I2T Accuracy │ Matched Sim  │
│ CLIP-ViT-B/16   │ 78.1%        │ 76.3%        │ 0.812        │
│ CLIP-ViT-B/32   │ 75.2%        │ 72.8%        │ 0.768        │
└─────────────────┴──────────────┴──────────────┴──────────────┘

🏆 Best CLIP: CLIP-ViT-B/16 | T2I: 78.1%
```

---

## 📋 **Implementation Guide**

### **Phase 1: Image Preprocessing (45-60 min)**

**File:** `src/features.py` → `ImagePreprocessor` class

**TODO 1: Build Transform Pipeline (20-30 min)**
- Create transform list with augmentation or inference transforms
- Add ToTensor and ImageNet normalization
- Key: RandomResizedCrop for augmentation, CenterCrop for inference

**TODO 2-3: Preprocess Images (25 min total)**
- Load and convert images to RGB
- Apply transforms and stack into batches

**Test:**
```python
from src.features import ImagePreprocessor
preprocessor = ImagePreprocessor(image_size=224, normalize=True)
tensor = preprocessor.preprocess("test.jpg")
print(f"Shape: {tensor.shape}")  # Should be [3, 224, 224]
```

---

### **Phase 2: Text Tokenization (40-50 min)**

**File:** `src/features.py` → `TextTokenizer` class

**TODO 4: Tokenize Text (25-35 min)**
- Lowercase and split text
- Add [SOS] and [EOS] tokens
- Pad to max_length (77 for CLIP)
- Create attention mask (1 for real tokens, 0 for padding)

**TODO 5: Batch Tokenization (15 min)**
- Tokenize multiple texts and stack masks

**Key concept:** Attention masks tell the model which tokens to process

---

### **Phase 3: Data Loading (40-50 min)**

**File:** `src/features.py` → `MultimodalDataLoader` class

**TODO 6: Load Paired Data (40-50 min)**
- Validate image-text alignment
- Check file existence and handle corrupt images
- Return valid image-text pairs

**TODO 7: Create Batches (25-30 min)**
- Stack images and masks into batches
- Keep texts as lists for readability

---

### **Phase 4: CLIP Model (2.5-3 hours)**

**File:** `src/models.py` → `CLIPModel` class

**TODO 8: Compute Similarity (45-60 min)**
- Process images and text with CLIP
- Extract and normalize embeddings
- Compute cosine similarity: `text_embeds @ image_embeds.T`

**TODO 9: Zero-Shot Classification (30-45 min)**
- Process image with multiple text labels
- Apply softmax to get probabilities
- Sort results by probability

**TODO 10: Evaluate CLIP (60-90 min)**
- Compute similarity matrix for test set
- Calculate T2I accuracy (text retrieves correct image)
- Calculate I2T accuracy (image retrieves correct text)
- Compute mean matched similarity

**Success criteria:**
- Similarity > 0.7 for matched pairs
- T2I/I2T accuracy > 70%

---

### **Phase 5: Image Captioning (2-2.5 hours)**

**File:** `src/models.py` → `ImageCaptioningModel` class

**TODO 11: Generate Caption (30-45 min)**
- Load BLIP model
- Generate with beam search
- Decode tokens to text

**TODO 12: Evaluate Captioning (90 min)**
- Generate captions for test images
- Compute BLEU, CIDEr, ROUGE-L metrics
- Install: `pip install nltk pycocoevalcap rouge`

**Success criteria:**
- BLEU > 0.3
- CIDEr > 1.0
- ROUGE-L > 0.5

---

### **Phase 6: Experiment Runner (1 hour)**

**File:** `src/models.py` → `ExperimentRunner` class

**TODO 13-16: Experiment Framework (60 min)**
- Run CLIP experiments and compare models
- Run caption experiments and compare models
- Print leaderboards sorted by metrics
- Color-code results (green for good scores)

---

### **Phase 7: Demos (30-60 min)**

**File:** `main.py`

**TODO 17: Zero-Shot Demo (20-30 min)**
- Create CLIP model and candidate labels
- Run classification and print results table
- Explain zero-shot learning

**TODO 18: Captioning Demo (20-30 min)**
- Create captioning model
- Generate captions for demo images
- Explain evaluation metrics

---

## 📊 **Success Criteria**

Your implementation is successful when:

### **CLIP**
- [x] Similarity scores > 0.7 for matched pairs
- [x] T2I/I2T accuracy > 70%
- [x] Zero-shot works with novel categories
- [x] Immediate feedback shows scores

### **Captioning**
- [x] BLEU > 0.3, CIDEr > 1.0, ROUGE-L > 0.5
- [x] Captions are fluent and descriptive
- [x] Metrics shown immediately

### **System**
- [x] All preprocessing works correctly
- [x] Models load without errors
- [x] Leaderboards sort by metrics
- [x] Colorful console output

---

## 🔬 **Experimentation Ideas**

### **1. Compare CLIP Variants**
- `clip-vit-base-patch32` (fastest)
- `clip-vit-base-patch16` (better quality)
- `clip-vit-large-patch14` (best quality)

**Question:** How much does model size improve accuracy?

### **2. Compare Captioning Models**
- `blip-image-captioning-base`
- `blip-image-captioning-large`
- `microsoft/git-base`

**Question:** Best BLEU vs. speed tradeoff?

### **3. Temperature Tuning**
Try 0.7, 1.0, 1.5 for caption generation

**Question:** How does temperature affect diversity?

---

## 📚 **Resources**

### **Papers**
- **CLIP**: [Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- **BLIP**: [Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)
- **ViT**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

### **Documentation**
- [HuggingFace CLIP](https://huggingface.co/docs/transformers/model_doc/clip)
- [HuggingFace BLIP](https://huggingface.co/docs/transformers/model_doc/blip)
- [PyTorch Vision](https://pytorch.org/vision/stable/transforms.html)

### **Internal Notes**
- [notes/05-multimodal_ai/](../../notes/05-multimodal_ai/)

---

## 🐛 **Troubleshooting**

### **Model Downloads**
```bash
# Set proxy if needed
export HF_ENDPOINT=https://hf-mirror.com
```

### **GPU Memory**
```python
# Use CPU if insufficient GPU memory
config = ModelConfig(device="cpu", batch_size=4)
```

### **Missing Dependencies**
```bash
python -c "import nltk; nltk.download('punkt')"
pip install git+https://github.com/tylin/coco-caption.git
```

---

## ✅ **Completion Checklist**

- [ ] `src/features.py`: 7 TODOs (preprocessing, tokenization, data loading)
- [ ] `src/models.py`: 8 model TODOs (CLIP, captioning)
- [ ] `src/models.py`: 4 ExperimentRunner TODOs (comparison framework)
- [ ] `main.py`: 2 demos (zero-shot, captioning)
- [ ] CLIP scores > 0.7, T2I/I2T > 70%
- [ ] BLEU > 0.3, CIDEr > 1.0
- [ ] Leaderboards work correctly

**Total time:** 8-10 hours

---

## 🎯 **Next Steps**

1. **Explore Applications**: Image search, clustering, anomaly detection
2. **Fine-Tune Models**: Domain-specific CLIP/BLIP
3. **Try New Models**: ALIGN, Florence, LLaVA
4. **Build Apps**: VQA, image-to-text retrieval, text-to-image generation

**Ready to implement multimodal AI? Start with `src/features.py`! 🚀**
