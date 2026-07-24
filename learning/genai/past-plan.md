# Pre-GenAI Foundations Plan

**Target learner:** Someone who has studied CNNs in intermediate detail (backprop, convolution,
pooling, BN, dropout, ResNet/skip connections) but *without* hands-on production implementation,
and is trying to get to LLMs by working through the `learning/genai/` track.

**Directory to implement:** `learning/pre-genai/` (sits immediately before `learning/genai/`)

**Question this plan answers:** Is the `learning/genai/` transition story sufficient for this
learner? What is critically missing?

---

## Story-Arc Audit

### What the current track assumes at first contact

When a CNN learner opens `00-pytorch-primer/keras-to-pytorch-primer.ipynb`, they can follow
it cleanly. When they open `01-rnns/PT-Part1-Intro.ipynb`, they get DL fundamentals / autograd —
still fine. **The first hard gap appears at `02-transformers/transformers.ipynb`**: the opening
cells refer to RNNs as the architecture Transformers replaced, describe BPTT, and use the
RNN-vs-Transformer comparison table (Keras) as motivation — but RNNs were never taught. A CNN
learner has no mental model of:

- What $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$ means
- Why vanishing gradients are worse in time than in depth
- What BPTT is and why it is expensive
- Why LSTM gating was the solution before attention

The Transformer chapter's "why is this better than what came before?" question is unanswerable
without this background.

### Second gap: tokenization

The `04-llm` chapter works entirely with GPT-2's BPE tokenizer, including `Ġ` (leading-space)
tokens, vocabulary indices, and `tokenize_causal()` vs `tokenize_instruction()` masking patterns.
No prior chapter explains:

- Why we need tokenization at all (strings → discrete tokens → integers → embeddings)
- How BPE works (merge-pair algorithm, why subword beats word and character)
- What `nn.Embedding` does and why it is the first trainable layer in a language model
- What padding and the `-100` ignore index pattern mean at the sequence level

A CNN learner who has worked with image classification has never encountered variable-length
inputs, and the `04-llm` chapter offers no bridge.

### Third gap: the missing "why decoder-only won" forward link

`03-encoder-decoder/encoder-decoder.ipynb` ends with a Tier 3 item ("why decoder-only won —
out of scope") but provides no bridge forward. A learner finishes knowing T5/BART work but not
understanding why `04-llm` opens with decoder-only GPT-2. This is a short cell in `03`, not a
full new chapter.

---

## Absolute Must-Haves

Items ordered by how severely they break the story arc without them.

---

### M-1 · Actual RNN / LSTM Mechanics Notebook

**Why it is a blocker:** `02-transformers/transformers.ipynb` builds its entire motivation
on RNNs as the prior art. Without understanding the hidden-state update equation, BPTT, and
the vanishing-gradient problem, a learner cannot understand *why* any Transformer design
decision is a decision (parallel computation over time, attention over fixed window, positional
encoding instead of implicit recurrence). The folder is named `01-rnns` but contains only DL
foundations — the RNN content is absent entirely.

**What this notebook covers (one notebook, PyTorch-primary, brief Keras mirror):**

| Part | Content | Key proof/demonstration |
|---|---|---|
| 1 | Character-level LM: vocabulary, `nn.Embedding`, `(batch, time, features)` shape contract | Print token-index-to-character mapping; visualize embedding matrix as a heatmap |
| 2 | Vanilla RNN cell from scratch: $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$ | Trace a 5-step unrolled forward pass by hand; verify against `nn.RNN` output |
| 3 | BPTT and vanishing gradients: measure gradient norm vs. timestep depth | Ablation: log-scale gradient-norm plot for depth 5, 20, 50 — show exponential decay |
| 4 | LSTM gating: forget, input, output, cell update equations | Build from scratch as `nn.Module`; verify gate outputs are in [0,1]; train on same character LM |
| 5 | Comparison: plain RNN vs. LSTM on a longer sequence | Side-by-side loss curves, final accuracy; print: "LSTM's cell highway removes the depth penalty" |
| 6 | Toy → real bridge | Parameter table mapping toy dims to real `nn.LSTM`; show `nn.LSTM(hidden_size=256)` output shape |

**Pedagogy requirements (gold-standard parity):**
- Single running example throughout (e.g. short stanza of poetry → character-level LM)
- `🔮 Predict first` before the vanilla RNN forward pass ("what will happen to the gradient at step 50?")
- `🧪 Your turn` exercise: change sequence length, measure vanishing gradient, confirm prediction
- `FuncAnimation` for the vanishing gradient experiment (gradient norm vs. timestep, animated across depth)
- `#### What just happened — and what's missing` after Part 3 to plant the LSTM question
- Toy/real parity table before `nn.LSTM`
- Closing tier-1/2/3 ledger (GRU, bidirectional RNN, and stacked RNN are Tier 3)
- Explicit forward pointer: "in the next chapter, we replace this sequential computation with parallel self-attention"

**Subagent implementation task:**
> Create `learning/pre-genai/01-rnn-sequence-modeling/` with a `rnn-sequence-modeling.ipynb`
> notebook following the gold-standard conventions. Use the MNIST CNN architecture from
> `00-pytorch-primer` as a known anchor, then show why a sequence task (character prediction)
> requires something different. Apply all conventions from `learning/genai/authoring-guide.md`.
> The running example should be a small English text corpus (can be generated). The gradient
> vanishing experiment should produce a log-scale plot and a printed conclusion comparing RNN
> and LSTM. Include images in `images/` matching the `images-plan.md` descriptions for RNN content.

---

### M-2 · Text Tokenization and Embeddings Notebook

**Why it is a blocker:** Every notebook from `02-transformers` onward works with token
sequences, but no chapter explains the text→token→integer→embedding pipeline. Specifically:
- `02-transformers` builds a hand-coded vocabulary (`VOCAB = {"the":0, "cat":1, ...}`) but
  never explains how a real tokenizer produces this
- `04-llm` uses GPT-2's BPE tokenizer extensively (`Ġ` prefix tokens, `input_ids`, padding,
  `-100` masking) and gives a walkthrough in the Code Walkthrough cell — but a learner
  without tokenization foundations will not follow the walkthrough

**What this notebook covers (one notebook, framework-agnostic):**

| Part | Content | Key proof/demonstration |
|---|---|---|
| 1 | Why tokenization exists: string → integer pipeline; why characters are too granular, words have OOV problems | Print vocabulary explosion at character level vs. word level on a small corpus |
| 2 | BPE from scratch: merge-pair algorithm on a toy corpus (5 sentences) | Step-by-step merge table; print the evolution of the vocabulary |
| 3 | Real BPE: GPT-2 tokenizer via `tiktoken`; inspect `Ġ` prefix, multitoken words, "internationalization" example | Compare token count vs. character count for 10 example sentences |
| 4 | `nn.Embedding` as a trainable lookup table: visualize embedding space for a small vocabulary using PCA 2D plot | Show that embeddings start random and become meaningful after training |
| 5 | Sequence padding: variable-length inputs, `pad_token_id`, the `-100` ignore index pattern | Side-by-side: padded batch without and with `attention_mask`; print what CrossEntropy sees |
| 6 | Toy → real bridge: vocabulary sizes (toy=15, GPT-2=50257, LLaMA-3=128k), subword vs. byte-level | Parameter table; note that "the cat" is 2 tokens in GPT-2, 3 in character model |

**Pedagogy requirements:**
- Running example: a 10-sentence English corpus maintained throughout
- `🔮 Predict first` before BPE merge step 10 ("which pair will merge next?")
- `🧪 Your turn`: change the merge budget; measure how vocabulary size affects compression ratio
- Animated BPE merge progression (`FuncAnimation` showing vocabulary evolution over merge steps)
- Closing tier-1/2/3 ledger (SentencePiece, WordPiece, Unigram are Tier 3)
- Forward pointer: "this is the exact pipeline that feeds into `02-transformers`' embedding vectors"

**Subagent implementation task:**
> Create `learning/pre-genai/02-tokenization/` with `tokenization-and-embeddings.ipynb`.
> Apply all gold-standard conventions. The BPE-from-scratch section should print a step-by-step
> merge table. The `nn.Embedding` visualization should use PCA (not t-SNE for reproducibility).
> End with the GPT-2 tokenizer demo using `tiktoken` (fallback: HuggingFace `AutoTokenizer`).

---

### M-3 · "Why Decoder-Only Won" Bridge Cell in `03-encoder-decoder`

**Why it is a blocker:** A learner finishing `03-encoder-decoder` knows how T5/BART work but
has no motivation to open `04-llm`. The chapter explicitly labels "why decoder-only won" as
out-of-scope (Tier 3) — which is correct — but provides no pointer forward. Without a bridge,
the `04-llm` choice of GPT-2 (decoder-only) feels arbitrary.

**What to add:** A single `## What's Next — and Why Decoder-Only` Markdown cell at the end of
`03-encoder-decoder/encoder-decoder.ipynb`, containing:

1. A 3-column comparison table: Encoder-Only | Decoder-Only | Encoder-Decoder
   (task fit, parameter efficiency at scale, generation quality, training objective)
2. 2-sentence framing: "Encoder-decoder excels at fixed-format transductions; at scale,
   decoder-only models turned out to generalize better to diverse tasks without a separate
   encoder — the next chapter works with one of those models."
3. A forward pointer to `04-llm/01-llm-finetuning-data-techniques.ipynb` by name.

**This is a targeted edit to one existing notebook, not a new notebook.**

**Subagent implementation task:**
> Add a `## What's Next — and Why Decoder-Only` Markdown cell to the end of
> `learning/genai/03-encoder-decoder/encoder-decoder.ipynb` (before the Summary section).
> The cell should contain a 3-row comparison table and a 2-sentence forward framing.

---

### M-4 · Wire the 7 Unused `02-transformers` Images

**Why it matters:** All 7 images in `learning/genai/02-transformers/images/` were authored as
companion reference figures but are not embedded in either notebook. They provide orientation
anchors (block overview, architecture families, learning journey roadmap) that the text alone
cannot provide. A learner who opens the notebook in VS Code's notebook viewer has no visual
structure before diving into code.

**What to do (per image):**

| Image | Where to insert | Both notebooks? |
|---|---|---|
| `transformer-learning-journey.png` | Cell 1 intro, after title heading | Yes |
| `attention-qkv-data-flow.png` | Part 4 (Q/K/V section) intro | Yes |
| `positional-encoding-and-rope.png` | Between Part 3a (sinusoidal) and Part 3b (RoPE) | Yes |
| `transformer-block-overview.png` | Part 7 intro (before block assembly) | Yes |
| `toy-to-production-transformers.png` | Toy-to-real bridge (before DistilGPT-2) | Yes |
| `transformer-architecture-families.png` | Part 13 intro (before encoder/decoder/enc-dec section) | Yes |
| `autoregressive-generation-and-kv-cache.png` | Part 9 (autoregressive inference) | Yes |

**Subagent implementation task:**
> For each image in `learning/genai/02-transformers/images/`, insert a Markdown cell with
> `![caption](images/filename.png)` at the listed location in both `transformers.ipynb` and
> `transformers-keras.ipynb`. Use the image filename as a guide to appropriate placement.
> Do not alter any surrounding cells. Verify by reading the cells adjacent to each insertion
> point to ensure the image contextually fits.

---

### M-5 · Validation Split and Loss Curve Plotting in `00-pytorch-primer`

**Why it matters:** The PyTorch manual training loop in `00-pytorch-primer` never shows per-epoch
train/val loss curves. Every subsequent notebook (`01-rnns`, `02-transformers`) expects the
reader to know how to monitor training. A learner who only knows Keras's `validation_split=0.1`
parameter will encounter a gap the first time they write a real PyTorch training loop.

**What to add:**
- A validation split (80/20 from the MNIST training set) added to the manual training loop
- Per-epoch loss and accuracy logged to `train_losses`, `val_losses` lists
- A side-by-side training curve plot (Keras `history.history` vs. PyTorch lists) showing they
  converge identically
- A 1-cell `# 👉 CHANGE` exercise: "change train split to 0.7 — does val loss diverge earlier?"

**This is a targeted addition to `00-pytorch-primer/keras-to-pytorch-primer.ipynb`.**

**Subagent implementation task:**
> Extend the manual training loop in `learning/genai/00-pytorch-primer/keras-to-pytorch-primer.ipynb`
> to include an 80/20 validation split, per-epoch metric logging, and a dual training curve
> plot (train loss + val loss, both frameworks side by side). Add a `🧪 Your turn` exercise
> for split ratio. Do not change anything outside the training loop section.

---

### M-6 · Cache the 3 External MIT Images in `01-rnns`

**Why it matters:** Both PT and TF notebooks load 3 images from MIT's GitHub raw URLs. These
fail offline and will silently break if the upstream repository changes paths.

**What to do:**
1. Download the three images from `raw.githubusercontent.com/MITDeepLearning/introtodeeplearning/master/lab1/img/`
   - `add-graph.png`
   - `computation-graph.png`
   - `computation-graph-2.png`
2. Save to `learning/genai/01-rnns/images/external/`
3. Update all references in both notebooks from the full URL to `images/external/filename.png`
4. Commit the local copies

**Subagent implementation task:**
> Download the three MIT computation graph images and save them locally under
> `learning/genai/01-rnns/images/external/`. Update both
> `PT-Part1-Intro.ipynb` and `TF-Part1-Intro-keras.ipynb` to use relative paths.
> Also: remove or file the orphan `Designer.png` from `01-rnns/images/` (it is never
> referenced in either notebook).

---

## Consistency Improvements (High Value, Not Blockers)

These do not break the story arc but would make the experience consistent with the
gold-standard pedagogy throughout.

### C-1 · FuncAnimation for Gradient Descent in `01-rnns/PT-Part1-Intro.ipynb`

The parabola gradient descent convergence is shown as a static loss-vs-epoch plot. An animation
(x-value stepping down the parabola surface over 500 iterations) would be a dramatically more
effective visual — matching the animation-heavy pedagogy of `02-transformers`.

**Subagent task:** Add a `FuncAnimation` cell to the parabola section showing the x-value
stepping down the $y = x^2$ curve frame by frame. Follow all Section 9.4 conventions
(plt.close before display, print before animation, fps=6).

### C-2 · `🔮` / `🧪` Emoji Normalization in `03-encoder-decoder`

The encoder-decoder notebook uses plain-text "Predict before you run" / "Your turn" headers
(consistent with the Section 3 amendment to the authoring guide). No change strictly needed,
but if the style guide is ever tightened to require emojis, this is the chapter to update.

### C-3 · Designer.png Cleanup in `01-rnns`

`Designer.png` is an orphan. Either wire it into a relevant section or delete it. Keeping
orphan files creates maintenance confusion.

---

## Implementation Parallelization Map

These items can be implemented by independent subagents simultaneously:

| Agent | Task | Target file(s) |
|---|---|---|
| Agent A | M-1: RNN/LSTM notebook | Create `learning/pre-genai/01-rnn-sequence-modeling/` |
| Agent B | M-2: Tokenization notebook | Create `learning/pre-genai/02-tokenization/` |
| Agent C | M-3: Decoder-only bridge cell | Edit `03-encoder-decoder/encoder-decoder.ipynb` |
| Agent D | M-4: Wire 02-transformers images | Edit both `02-transformers/*.ipynb` |
| Agent E | M-5: Validation split + curves | Edit `00-pytorch-primer/keras-to-pytorch-primer.ipynb` |
| Agent F | M-6: Cache external images | Download files, edit both `01-rnns/*.ipynb` |

Agents A and B have no file dependencies on any other agent. C, D, E, F each touch one
existing file or folder. All 6 can run in parallel after this plan is reviewed.

---

## Pre-GenAI Directory Structure (after implementation)

```
learning/
  pre-genai/
    README.md                          # who this is for; how it connects to genai/
    01-rnn-sequence-modeling/
      rnn-sequence-modeling.ipynb     # M-1
      images/
      requirements.txt
    02-tokenization/
      tokenization-and-embeddings.ipynb  # M-2
      images/
      requirements.txt
  genai/
    00-pytorch-primer/  (modified: M-5)
    01-rnns/            (modified: M-6, C-1, C-3)
    02-transformers/    (modified: M-4)
    03-encoder-decoder/ (modified: M-3)
    04-llm/             (unchanged)
```
