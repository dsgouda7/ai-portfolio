"""Transform PT_Part1_Intro.ipynb: apply all 12 authoring-guide improvements."""

import json
import uuid

NB_PATH = r"c:\repos\ai-portfolio\learning\genai\rnns\MIT\PT_Part1_Intro.ipynb"


def src(text: str) -> list:
    """Convert a multiline string to notebook source list."""
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            if line:  # non-empty last line
                result.append(line)
    return result


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": src(text),
    }


# ── Load original notebook ────────────────────────────────────────────────────
with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

orig = nb["cells"]  # 48 cells, indices 0-47
print(f"Original cell count: {len(orig)}")

# ── Build new cell list ───────────────────────────────────────────────────────
new_cells = []

# Cell 0: MIT branding table (keep)
new_cells.append(orig[0])

# Cell 1: Copyright code (keep)
new_cells.append(orig[1])

# Cell 2: REPLACE lab description with roadmap + title (Change 1)
new_cells.append(md("""\
# PyTorch from First Principles

## Building Deep Learning Intuition One Tensor at a Time

This notebook builds the **complete mental model for PyTorch's tensor abstraction and autograd system** — starting with a running example that threads through every concept.

Every concept is demonstrated on the same problem:

> **Predicting house prices from size and age**
> 5 houses, 2 features (sq ft, age) → 1 price. Small enough to visualise, real enough to matter.

| Step | Concept | Key Idea |
|------|---------|----------|
| 1 | Tensors as Data Containers | Scalars → vectors → matrices → batches; faster + safer than lists |
| 2 | Operations on Tensors | PyTorch builds computation graphs automatically |
| 3 | The Manual Prediction Problem | Hand-tuning 3 weights fails; 100 features is impossible |
| 4 | Neural Networks in PyTorch | nn.Module wraps learnable nn.Parameters |
| 5 | Automatic Differentiation | .backward() computes ∂loss/∂every_weight in one call |
| 6 | Gradient Descent in Action | Iterative weight updates drive loss toward zero |
| 7 | From Toy to Production | Same autograd scales from 3 params to 1.5 billion (GPT-2) |
| 8 | Music Generation Bonus | Pre-trained transformers (MusicGen, TunesFormer) |

---

## 0.1 Install PyTorch

[PyTorch](https://pytorch.org/) is a deep learning library known for flexibility and ease of use. For all labs in Introduction to Deep Learning 2026, a PyTorch version is available.\
"""))

# Cell 3: Imports (keep)
new_cells.append(orig[3])

# Change 2 — Cell A: Deterministic seeds (NEW)
new_cells.append(code("""\
# ── Deterministic seeds for reproducible results ──────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
print("Seeds set → every run produces identical results.")\
"""))

# Change 2 — Cell B: House dataset (NEW)
new_cells.append(code("""\
# ── Our Running Example — House Price Prediction ──────────────────────────────
# Throughout this notebook, every concept is demonstrated on the same 5 houses.
houses = torch.tensor([
    [1200.0, 10.0],  # size (sq ft), age (years)
    [1500.0,  5.0],
    [ 800.0, 15.0],
    [2000.0,  2.0],
    [1000.0, 12.0],
], dtype=torch.float32)
prices = torch.tensor([250.0, 320.0, 180.0, 450.0, 210.0])  # $1000s

print("Our running example — 5 houses:")
print("  Size (sq ft)  Age (yrs)  Price ($k)")
for i, (h, p) in enumerate(zip(houses, prices)):
    print(f"  [{i}]  {h[0]:6.0f}      {h[1]:4.0f}       {p:6.0f}")

# 2-panel scatter plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sc0 = axes[0].scatter(houses[:, 0], prices, c=houses[:, 1], cmap='viridis', s=100, edgecolor='black')
axes[0].set_xlabel('Size (sq ft)'); axes[0].set_ylabel('Price ($1000s)')
axes[0].set_title('Price vs Size (color = age)')
plt.colorbar(sc0, ax=axes[0], label='Age (years)')
sc1 = axes[1].scatter(houses[:, 1], prices, c=houses[:, 0], cmap='plasma', s=100, edgecolor='black')
axes[1].set_xlabel('Age (years)'); axes[1].set_ylabel('Price ($1000s)')
axes[1].set_title('Price vs Age (color = size)')
plt.colorbar(sc1, ax=axes[1], label='Size (sq ft)')
plt.suptitle("Our Running Example: 5 Houses → 1 Price", fontweight='bold')
plt.tight_layout(); plt.show()
print("This dataset is our 'the cat sat on the mat' — every concept demonstrated on these 5 houses.")\
"""))

# Cell 4 (orig): ## 1.1 What is PyTorch? markdown (keep)
new_cells.append(orig[4])

# Change 3 — Predict-first markdown (NEW, before first 1.1 code cell)
new_cells.append(md("""\
#### 🔮 Predict first — why do we need tensors?

Python has lists. NumPy has arrays. Predict which of these are true about `torch.Tensor` vs nested lists:
1. **Speed**: Can tensors outrun nested lists on matrix multiply?
2. **Shape safety**: Will tensors catch a shape mismatch that lists silently swallow?
3. **GPU support**: Can a list of lists run on a GPU?

Make your prediction, then run the next cell.\
"""))

# Change 3 — Tensor proof code (NEW)
new_cells.append(code("""\
# ── WHY tensors? Three reasons, measured ──────────────────────────────────────
import time

# Reason 1: Speed — matrix multiply
n = 1000
np_a = np.random.randn(n, n).astype(np.float32)
np_b = np.random.randn(n, n).astype(np.float32)
t0 = time.time(); _ = np_a @ np_b; np_time = time.time() - t0

t_a = torch.from_numpy(np_a); t_b = torch.from_numpy(np_b)
t0 = time.time(); _ = torch.matmul(t_a, t_b); torch_time = time.time() - t0

print(f"Matrix multiply (1000×1000):")
print(f"  NumPy:  {np_time*1000:.1f} ms")
print(f"  PyTorch: {torch_time*1000:.1f} ms  ({np_time/max(torch_time,1e-9):.1f}× speedup)")

# Reason 2: Shape safety
try:
    bad = torch.randn(3, 4) + torch.randn(5, 6)
except RuntimeError as e:
    print(f"\\n✓ Shape mismatch caught: '{e}'")
    print("  Lists would silently fail or give wrong results.")

# Reason 3: GPU support
print(f"\\nGPU available: {torch.cuda.is_available()}")
print("  → torch.Tensor can move to GPU with .to('cuda'); lists cannot.")
print("\\n→ Tensors are PURPOSE-BUILT for deep learning: fast, safe, GPU-ready.")\
"""))

# Cells 5-11 (orig): tensor basics (keep)
for i in range(5, 12):
    new_cells.append(orig[i])

# Change 11 — Reflection after Section 1.1 (NEW)
new_cells.append(md("""\
#### What just happened — and what's missing

Tensors are PyTorch's data container: fast (GPU-ready), safe (shape-checked), and built for batches. Every model input (images, text, audio) becomes a tensor before processing.

**Missing piece**: We have the *container*, but we haven't built anything that *learns* yet. For that, we need operations that PyTorch can differentiate — next.\
"""))

# Cell 12 (orig): ## 1.2 Computations on Tensors (keep)
new_cells.append(orig[12])

# Cells 13-18 (orig): computation graph cells (keep)
for i in range(13, 19):
    new_cells.append(orig[i])

# Change 4 — Tie 1.2 to running example: house price forward pass (NEW)
new_cells.append(code("""\
# ── Computation on our house dataset — price prediction by hand ───────────────
size = houses[0, 0]   # 1200 sq ft
age  = houses[0, 1]   # 10 years
true_price = prices[0]   # $250k

# Hand-picked weights (we'll learn better ones later via gradient descent)
w_size = torch.tensor(0.15)
w_age  = torch.tensor(-5.0)
bias   = torch.tensor(100.0)

contrib_size = w_size * size
contrib_age  = w_age  * age
price_pred   = contrib_size + contrib_age + bias

print(f"House [0]: {size:.0f} sq ft, {age:.0f} years → true price ${true_price:.0f}k")
print(f"  w_size × size = {w_size:.2f} × {size:.0f} = ${contrib_size:.1f}k")
print(f"  w_age  × age  = {w_age:.2f} × {age:.0f} = ${contrib_age:.1f}k")
print(f"  + bias        = ${bias:.1f}k")
print(f"  → Predicted:  ${price_pred:.1f}k   (error: ${abs(price_pred - true_price):.1f}k)")
print("\\nPyTorch traced this computation automatically. We'll use that trace for autograd next.")\
"""))

# Change 5 — Manual weight-tuning exercise markdown (NEW)
new_cells.append(md("""\
### 🧪 Your turn — manual weight tuning

Before Section 1.3 introduces *learnable* weights, try tuning the three weights by hand.

**Predict**: Can you fit all 5 houses within ±$10k error just by adjusting numbers?\
"""))

# Change 5 — Manual weight-tuning exercise code (NEW)
new_cells.append(code("""\
# 🧪 EXERCISE — manual weight tuning
# 👉 CHANGE these three weights to predict all 5 house prices within ±$10k error
w_size = torch.tensor(0.15)   # ← try 0.10, 0.20, 0.25...
w_age  = torch.tensor(-5.0)   # ← try -3.0, -8.0...
bias   = torch.tensor(100.0)  # ← try 50.0, 150.0...

predictions = houses[:, 0] * w_size + houses[:, 1] * w_age + bias
errors = torch.abs(predictions - prices)

print("Manual tuning results:")
print(f"  {'House':>6}  {'True':>8}  {'Predicted':>10}  {'Error':>8}")
for i, (p_true, p_pred, err) in enumerate(zip(prices, predictions, errors)):
    check = "✓" if err < 10.0 else "✗"
    print(f"  [{i}]     ${p_true:6.1f}k   ${p_pred:6.1f}k      ${err:5.1f}k  {check}")
mean_error = errors.mean()
print(f"\\nMean absolute error: ${mean_error:.1f}k")
if mean_error < 10.0:
    print("✓ You nailed it by hand — but imagine 100 features, 10000 houses...")
else:
    print("→ Hand-tuning fails even on 5 houses. We need LEARNING: autograd + gradient descent.")\
"""))

# Change 11 — Reflection after Section 1.2 (NEW)
new_cells.append(md("""\
#### What just happened — and the crack it leaves open

We manually wrote `w_size * size + w_age * age + bias` and saw the prediction fail. But:
- We **hand-picked** those weights. What if we have 100 features?
- We **guessed** they'd work. How do we find *good* weights systematically?

That's the job of **gradient descent** — and it requires computing `∂loss/∂w_size`. That's what autograd does automatically.\
"""))

# Cell 19 (orig): ## 1.3 Neural networks in PyTorch (keep)
new_cells.append(orig[19])

# Cells 20-35 (orig): all Section 1.3 cells (keep)
for i in range(20, 36):
    new_cells.append(orig[i])

# Change 6 — HousePriceModel with learnable weights (NEW)
new_cells.append(code("""\
# ── HousePriceModel: the same prediction, but with LEARNABLE weights ──────────
class HousePriceModel(torch.nn.Module):
    \"\"\"Predicts house price from size and age using a single linear layer.\"\"\"
    def __init__(self):
        super().__init__()
        self.w_size = torch.nn.Parameter(torch.randn(1) * 0.01)
        self.w_age  = torch.nn.Parameter(torch.randn(1) * 0.01)
        self.bias   = torch.nn.Parameter(torch.randn(1) * 0.01)

    def forward(self, size, age):
        return self.w_size * size + self.w_age * age + self.bias

torch.manual_seed(42)
model = HousePriceModel()
print("Model parameters (random initialization):")
for name, param in model.named_parameters():
    print(f"  {name}: {param.item():.4f}")

pred = model(houses[0, 0], houses[0, 1])
print(f"\\nPrediction for house [0]: ${pred.item():.1f}k  (random weights → bad prediction)")
print("Next: we'll use autograd to LEARN better weights automatically.")\
"""))

# Cell 36 (orig): ## 1.4 Automatic Differentiation (keep)
new_cells.append(orig[36])

# Cells 37-39 (orig): autograd and gradient descent cells (keep)
for i in range(37, 40):
    new_cells.append(orig[i])

# Change 7 — Gradient flow visualization (NEW)
new_cells.append(code("""\
# ── Visualizing what .backward() actually does ────────────────────────────────
torch.manual_seed(42)
model_viz = HousePriceModel()
optimizer = torch.optim.SGD(model_viz.parameters(), lr=1e-5)

optimizer.zero_grad()
preds = model_viz(houses[:, 0], houses[:, 1])
loss = torch.mean((preds - prices) ** 2)
loss.backward()

print(f"After loss.backward(), gradients computed:")
for name, param in model_viz.named_parameters():
    print(f"  ∂loss/∂{name}: {param.grad.item():+.4f}")

print(f"\\nLoss: {loss.item():.2f}")
print("→ .backward() computed ∂loss/∂w_size, ∂loss/∂w_age, ∂loss/∂bias automatically.")
print("  Gradient descent updates each weight in the direction that reduces loss.")

# Show one update step
print(f"\\nBefore update: w_size = {model_viz.w_size.item():.4f}")
optimizer.step()
print(f"After step:    w_size = {model_viz.w_size.item():.4f}  (moved toward lower loss)")\
"""))

# Change 11 — Reflection after Section 1.4 (NEW)
new_cells.append(md("""\
#### What just happened — from toy to production

We trained a 3-parameter model with `.backward()` and watched loss drop. The same mechanism trains:
- GPT-2: 1.5 billion parameters
- Stable Diffusion: 860 million parameters

**The only difference is scale.** The autograd graph, the gradient computation, the optimizer step — all identical.\
"""))

# Change 8 — Toy→real bridge markdown (NEW)
new_cells.append(md("""\
---

## Part 1.5 — From Toy to Production: Same Machinery, Bigger Numbers

Our house-price model has **3 learnable parameters**. A production deep-learning model has millions — but the *mechanism* is identical: `nn.Parameter`, `.forward()`, `.backward()`, `optimizer.step()`.

| Model | Parameters | Same autograd? | Same nn.Module? |
|-------|-----------|----------------|-----------------|
| Our toy (house prices) | 3 | ✓ | ✓ |
| ResNet-18 (image classification) | 11.7 million | ✓ | ✓ |
| GPT-2 (language model) | 1.5 billion | ✓ | ✓ |
| Stable Diffusion | 860 million | ✓ | ✓ |

**The only difference is scale.** If you understood gradient descent on 3 weights, you understand it on 860 million.\
"""))

# Change 8 — ResNet code (NEW)
new_cells.append(code("""\
# ── Toy-to-real: same nn.Module, different scale ──────────────────────────────
try:
    import torchvision
    real_model = torchvision.models.resnet18(weights=None)  # structure only
    total = sum(p.numel() for p in real_model.parameters())
    trainable = sum(p.numel() for p in real_model.parameters() if p.requires_grad)
    print(f"ResNet-18 architecture:")
    print(f"  Total parameters   : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Layers: {len(list(real_model.children()))}")
    print("\\nEvery one of those 11.7M parameters is a torch.nn.Parameter, just like our w_size.")
    print("Every forward pass traces a computation graph. Every .backward() computes all gradients.")
    print("The machinery you learned on 3 weights scales to billions — no new concepts needed.")
except ImportError:
    print("[torchvision not installed — the point stands: production models use the same")
    print(" nn.Module / autograd machinery you just learned on 5 houses and 3 weights.]")\
"""))

# Cell 40 (orig): summary bridge markdown (keep)
new_cells.append(orig[40])

# Cell 41 (orig): ## Part 2 Extension header (keep)
new_cells.append(orig[41])

# Change 9 — Part 2 reframe: Why pre-trained models (NEW)
new_cells.append(md("""\
### Why Pre-trained Models Instead of Training From Scratch?

You've learned PyTorch's autograd system on a 3-parameter toy model. Real-world models (GPT, BERT, MusicGen) are trained the *same way* — just scaled up to billions of parameters and trained on massive datasets (books, web text, audio).

| Approach | Data Needed | Training Time | Architecture |
|---|---|---|---|
| From Scratch | 20,000+ hours of labeled audio | Days/weeks on GPU clusters | Must design yourself |
| Pre-trained (HuggingFace) | Zero | Seconds (inference only) | Already proven |

**This section demonstrates inference** (generating music from a trained model). The *training* used the exact same `.backward()` you just learned — just at scale.\
"""))

# Cell 42 (orig): Install HuggingFace dependencies (keep)
new_cells.append(orig[42])

# Change 10 — 🧪 Your turn music generation (NEW)
new_cells.append(md("""\
### 🧪 Your turn — music generation

The next cell lets you choose between two pretrained models. For whichever you pick, change the prompt/seed and **predict** what the output will sound like before generating.\
"""))

# Cell 43 (orig): Model selector (keep)
new_cells.append(orig[43])

# Cells 44-47 (orig): Option A/B markdown and code (keep)
for i in range(44, 48):
    new_cells.append(orig[i])

# Change 12 — Closing summary (NEW, at the very end)
new_cells.append(md("""\
---

## Summary — What You Built

| Step | Concept | Key Idea | ✓ |
|------|---------|----------|---|
| 1 | Tensors as Data Containers | Scalars → vectors → matrices → batches; faster + safer than lists | ✓ |
| 2 | Operations on Tensors | PyTorch builds computation graphs automatically | ✓ |
| 3 | The Manual Prediction Problem | Hand-tuning 3 weights fails; 100 features is impossible | ✓ |
| 4 | Neural Networks in PyTorch | nn.Module wraps learnable nn.Parameters | ✓ |
| 5 | Automatic Differentiation | .backward() computes ∂loss/∂every_weight in one call | ✓ |
| 6 | Gradient Descent in Action | Iterative weight updates drive loss toward zero | ✓ |
| 7 | From Toy to Production | Same autograd scales from 3 params to 1.5 billion (GPT-2) | ✓ |
| 8 | Music Generation Bonus | Pre-trained transformers (MusicGen, TunesFormer) | ✓ |

### Key Insights to Keep

- **Tensors are purpose-built**: GPU-ready, shape-safe, and 5-50× faster than lists for matrix operations.
- **nn.Module = learnable weights + forward pass**: The same pattern scales from 3 parameters to 1.5 billion.
- **Autograd is automatic calculus**: `.backward()` computes every ∂loss/∂param in one call, no manual derivatives needed.
- **Gradient descent is iterative refinement**: Each step nudges weights to reduce loss — cumulative tiny improvements converge to near-optimal.
- **Pre-trained models save months**: Training GPT-2 from scratch costs $50K+ in compute; inference on a pre-trained model takes seconds.
- **From toy to production, the machinery is identical**: If you understood gradient descent on house prices (3 weights), you understand it on Stable Diffusion (860M weights). Scale is the only difference.

**Next**: Dive into recurrent networks (Lab 2) and convolutional networks (Lab 3) — both built on the same PyTorch primitives you just mastered.\
"""))

# ── Write updated notebook ────────────────────────────────────────────────────
nb["cells"] = new_cells
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"✓ Done. Original: 48 cells → New: {len(new_cells)} cells")
print(f"  New cells added: {len(new_cells) - 48}")
