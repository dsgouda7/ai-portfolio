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

**Narrative framing (required — Section 8 of the authoring guide):**

The notebook is anchored to one concrete scenario throughout: *a music research team wants to predict the next note in a melody — the same problem that launched sequence modeling in the 1980s.* The running corpus is the opening 40 characters of "Twinkle Twinkle Little Star" encoded as characters (`T`, `w`, `i`, `n`, `k`, `l`, `e`, ` `, ...). This corpus is:
- Small enough to inspect character-by-character (no download required)
- Rich enough to show repeat patterns (the word "twinkle" appears twice — a real test of memory)
- Musical, connecting back to the pretrained music generation demo the learner already saw in `01-rnns/PT-Part1-Intro.ipynb`

Every Part answers a named question for this team: "Does it even know the alphabet?" (Part 1), "Does it remember what it saw 5 steps ago?" (Part 2–3), "Why does it forget the second 'twinkle'?" (Part 3), "What if we give it a memory lane?" (Part 4–5).

**What this notebook covers (one notebook, PyTorch-primary, brief Keras mirror):**

| Part | Content | Key proof/demonstration |
|---|---|---|
| 0 | Challenge: the music team's problem + scope of what changes from a CNN | Print: a CNN classifies one image as one label; an RNN must output one label *per timestep* — the output tensor has a time axis |
| 1 | Character-level LM: vocabulary, `nn.Embedding`, `(batch, time, features)` shape contract | Print token-index-to-character mapping; visualize embedding matrix as a heatmap; **prove**: `model(one_hot_input)` and `model(index_input)` produce identical logits |
| 2 | Vanilla RNN cell from scratch: $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$ | Unroll 5 steps of the melody by hand, printing $h_0, h_1, ..., h_4$; **verify** the manual computation matches `nn.RNN` output to 6 decimal places |
| 3 | BPTT and vanishing gradients: gradient norm vs. timestep distance from loss | **Ablation**: log-scale gradient-norm plot for sequence length 5, 20, 50 on the melody; print: "at depth 50, the gradient at step 1 is {n:.2e} — effectively zero" |
| 4 | LSTM gating: forget, input, output, cell update equations | Build from scratch as `nn.Module`; **prove** gate activations are in [0,1] with `assert ((gates >= 0) & (gates <= 1)).all()`; train on same melody |
| 5 | Comparison: plain RNN vs. LSTM on the second 'twinkle' | Side-by-side loss curves for both; measure whether LSTM correctly predicts the `t` after the space more often than vanilla RNN; **print**: "LSTM got it right N/10 times vs. RNN's M/10" |
| 6 | Toy → real bridge | Parameter table mapping toy dims (`hidden_size=8`) to real `nn.LSTM(hidden_size=256, num_layers=2)`; show GPT-2's actual embedding layer shape |

**Pedagogy requirements (gold-standard parity):**
- **Named scenario and threaded running example throughout** (the music team; the "Twinkle" melody; same characters Part 1–6)
- `🔮 Predict first` before the vanilla RNN forward pass: *"At timestep 50, the gradient signal will be: (a) roughly the same as at timestep 1, (b) about 10× smaller, (c) about 1000× smaller, or (d) actually larger due to accumulation?"* Answer: (c) for vanilla RNN, (a) for LSTM
- `🔮 Predict first` before the RNN-vs-LSTM comparison: *"How many times (out of 10 sampled completions) will each model correctly predict 't' at the start of the second 'twinkle'? Options: RNN<5/LSTM>7, both~5, RNN>LSTM"*
- `🧪 Your turn` exercise: change sequence length from 40 to 10; does vanishing gradient still occur? (Answer: much less — print confirms it)
- `FuncAnimation` for the vanishing gradient experiment (gradient norm vs. timestep, animated adding one layer at a time)
- `#### What just happened — and what's missing` after Part 3: "The RNN can't carry the 'twinkle' memory 15 steps. Next: give it a memory lane with explicit gates."
- Toy/real parity table before `nn.LSTM`
- **Closing decision (Section 8.7):** "For the music team: the LSTM now correctly anticipates the repeat pattern. The cost: 4× more parameters than a vanilla RNN (measured). The benefit: the gradient norm at step 1 went from {rnn_grad:.2e} to {lstm_grad:.2e}. For sequences shorter than ~20 tokens, use RNN; above that, LSTM earns its parameter cost."
- Closing tier-1/2/3 ledger: Tier 1 (vanilla RNN, LSTM), Tier 2 (GRU — same cell highway, fewer gates, explained but not trained), Tier 3 (bidirectional RNN, stacked RNN, attention-augmented RNNs — named with one-line reason)
- Explicit forward pointer: "The music team can predict one note at a time. In the next chapter, we ask: what if the model could attend to *all* past notes simultaneously? That's the Transformer."

**Subagent implementation task:**
> Create `learning/pre-genai/01-rnn-sequence-modeling/` with a `rnn-sequence-modeling.ipynb`
> notebook following ALL gold-standard conventions from `learning/genai/authoring-guide.md`.
> The running corpus is the first 40 characters of "Twinkle Twinkle Little Star" (write it as a
> string constant, no download needed). Every Part answers one named question for the music team.
> The vanilla RNN manual unrolling must print each $h_t$ and verify against `nn.RNN` output.
> The vanishing gradient ablation must produce a log-scale plot and a printed comparison.
> The LSTM from scratch must use `assert` to prove gates are in [0,1].
> The RNN vs. LSTM comparison must sample 10 completions from each and count correct predictions.
> The closing decision must branch on the actual measured gradient norms, not aspirational text.
> Include images in `images/` matching the `images-plan.md` RNN image descriptions.

---

### M-2 · Text Tokenization and Embeddings Notebook

**Why it is a blocker:** Every notebook from `02-transformers` onward works with token
sequences, but no chapter explains the text→token→integer→embedding pipeline. Specifically:
- `02-transformers` builds a hand-coded vocabulary (`VOCAB = {"the":0, "cat":1, ...}`) but
  never explains how a real tokenizer produces this
- `04-llm` uses GPT-2's BPE tokenizer extensively (`Ġ` prefix tokens, `input_ids`, padding,
  `-100` masking) and gives a walkthrough in the Code Walkthrough cell — but a learner
  without tokenization foundations will not follow the walkthrough

**Narrative framing (required — Section 8 of the authoring guide):**

The notebook is anchored to one concrete task: *you're building a multilingual chatbot for a small law firm. Your two problems: (1) legalese has rare compound words that a word-level tokenizer will treat as unknown tokens; (2) the system must also handle French contract clauses. You need a tokenizer that handles rare words without blowing up the vocabulary.* Every technique is framed as answering one of the firm's questions: "Why can't we just split on spaces?" (Part 1), "How does GPT-2 handle 'non-disclosure' without seeing it in training?" (Part 2–3), "Why do 'non-disclosure' and 'nondisclosure' get the same embedding representation?" (Part 4).

**What this notebook covers (one notebook, framework-agnostic):**

| Part | Content | Key proof/demonstration |
|---|---|---|
| 1 | Why tokenization exists: string → integer pipeline; why characters are too granular, words have OOV problems | **Measure** vocabulary sizes on a 20-sentence legal corpus: character-level = 62 unique tokens, word-level = 340 unique tokens, 38 of which appear only once (OOV risk); print the comparison |
| 2 | BPE from scratch: merge-pair algorithm on a toy corpus (5 legal sentences) | Step-by-step merge table printed at every iteration; **prove**: after 20 merges, "non-disclosure" is represented as 2 tokens ("non", "disclosure"), not 15 characters |
| 3 | Real BPE: GPT-2 tokenizer via `tiktoken`; inspect `Ġ` prefix, multitoken words, "indemnification" example | **Measure**: token count vs. character count for 10 legal phrases; print compression ratios; show that French "dommages" is handled without OOV |
| 4 | `nn.Embedding` as a trainable lookup table: visualize embedding space for a small vocabulary using PCA 2D plot | **Prove**: embeddings start random (PCA shows uniform scatter); after 100 training steps on the corpus, legal synonyms cluster; show the before/after PCA side by side |
| 5 | Sequence padding: variable-length inputs, `pad_token_id`, the `-100` ignore index pattern | Side-by-side: padded batch without and with `attention_mask`; print what CrossEntropy sees |
| 6 | Toy → real bridge: vocabulary sizes (toy=15, GPT-2=50257, LLaMA-3=128k), subword vs. byte-level | Parameter table; note that "the cat" is 2 tokens in GPT-2, 3 in character model |

**Pedagogy requirements:**
- **Named scenario threaded throughout** (the law firm; the 20-sentence legal corpus used from Part 1 to Part 4)
- `🔮 Predict first` before BPE merge step 10: *"After 10 merges, what will happen to the token 'non-disclosure': (a) still 15 characters, (b) two tokens 'non' and 'disclosure', (c) one merged token 'non-disclosure'?"* (Answer: b, by step 10 in a legal corpus)
- `🔮 Predict first` before the embedding PCA: *"After 100 training steps on 20 sentences, will 'contract' and 'agreement' be: (a) near each other, (b) far from each other, (c) in random positions?"* (Answer: a — the training signal clusters legal synonyms)
- `🧪 Your turn`: change the merge budget from 20 to 5; measure how vocabulary size and OOV token count change
- `FuncAnimation` for the BPE merge progression (one frame per merge step; watch "non-disclosure" shrink from 15 tokens to 2)
- **Closing decision (Section 8.7):** "For the law firm: BPE with 20k merges handles 'indemnification' in 2 tokens (not OOV), handles French without a separate vocabulary, and produces compression ratio {r:.1f}× vs. character-level. The recommended tokenizer: GPT-2's `tiktoken` encoder, which uses the same algorithm at 50k vocabulary size."
- Closing tier-1/2/3 ledger: Tier 1 (character-level, word-level, BPE from scratch, GPT-2 tiktoken), Tier 2 (WordPiece — same merge algorithm, different scoring; explained but not built), Tier 3 (SentencePiece, Unigram LM, byte-level BPE — named with one-line reason)
- Forward pointer: "This is the exact pipeline that feeds `02-transformers`' `VOCAB` dictionary — the hand-coded vocabulary there is a simplified BPE with exactly these properties."

**Subagent implementation task:**
> Create `learning/pre-genai/02-tokenization/` with `tokenization-and-embeddings.ipynb`.
> Apply ALL gold-standard conventions from `learning/genai/authoring-guide.md`.
> The opening cell states the law firm scenario before the title, following the 04-llm pattern.
> The 20-sentence legal corpus must be written inline as a Python string (no download).
> The BPE-from-scratch section must print a merge table at every step.
> The PCA visualization must show a before (random) and after (clustered) side-by-side panel.
> The closing decision must branch on the actual measured compression ratio, not aspirational text.
> End with GPT-2 tokenizer demo using `tiktoken` (fallback: HuggingFace `AutoTokenizer`).

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
