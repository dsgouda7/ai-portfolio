# Improvement Plan — Text Tokenization and Embeddings

**Audited:** 2026-07-26 | **Audience fit:** 7/10

## Overall Assessment

Strong pedagogical skeleton: the law-firm framing is genuine and specific (two named problems: legalese OOV, French character explosion), every Part ties its technique back to one of those two problems, the predict-first boxes and "What just happened" cadence are effective for engineers learning through code. The primary failure mode is image-corpus mismatch: two of the three static images show generic textbook examples (wrong words, king/queen vectors) that break the thread the prose builds carefully. The secondary failure mode is analogy-after-math ordering in two key introductions.

---

## Strengths (preserve these)

- **Opening scenario** — two named, concrete problems with exact domain words (`indemnification`, `dommages-intérêts`)
- **Per-Part "The firm's question"** — every section opens with the firm's italicised question; each technique feels demanded not arbitrary
- **Predict-first boxes** — vocabulary size, merge-step result, clustering prediction; all well-placed
- **BPE training-loop print table** — step-by-step merge progression; engineers see exactly when `'non'` forms
- **"What just happened — and what's missing" cadence** — structured transition cells prevent false closure
- **Cosine similarity table** — quantifies before/after clustering without relying on visual inference
- **Tier 1/2/3 summary ledger** — honest scope declaration
- **Toy → real bridge table (Part 6)** — parameter comparison across this notebook / GPT-2 / LLaMA-3 is one of the most useful cells
- **Forward pointers** — named, specific cross-references to `02-transformers` and `04-rnn-sequence-modeling`

---

## Gaps & Recommended Changes

### Gap 1 — BPE image shows wrong words, not legal corpus — Priority: High

**Problem:** `images/bpe-merge-steps.png` shows "the lowest" merging to "lowest" in generic steps. The alt text promises `"non-disclosure" shrinks from 14 characters to 2 subword tokens after 20 merges`. The image is placed before the BPE helpers as the first impression of Part 2, signaling "generic content."

**Justification:** The law firm thread works because every visual traces back to the firm's own words. A stock example at this position undermines the bespoke credibility the prose earned.

**Recommendation:** Replace the image. Lowest-effort option: add a `plt.savefig` call that snapshots `nondiscl_history` token-list states at steps 0, 10, 25, 50, building the 3-panel visual from actual training output. This is accurate, corpus-tied, and reproducible.

---

### Gap 2 — Embedding PCA image shows king/queen/man/woman, not legal synonyms — Priority: High

**Problem:** `images/embedding-space-pca.png` shows the canonical word2vec king/queen/man/woman scatter. The alt text promises legal synonym clusters; `images-plan.md` specifies contract/agreement/indemnification clusters. The king/queen image is instantly recognisable as a textbook stock image — every engineer who has touched NLP has seen it — signaling that Part 4 is generic.

**Justification:** This is the most damaging mismatch. It also sets a false expectation: the image shows dramatic, well-separated clusters that the actual code-generated plots (20-sentence corpus, 500 training steps) will not reproduce — creating disappointment the moment the cells run.

**Recommendation:** Remove the placeholder image entirely and rely on the live code-generated PCA plots. The code already produces the correct visual; the placeholder only competes against it with a misleading example.

---

### Gap 3 — `nn.Embedding` intro opens with matrix notation before any plain-English analogy — Priority: Medium

**Problem:** Part 4 begins: "An `nn.Embedding` layer is a matrix $W_e \in \mathbb{R}^{V \times d_e}$..." Matrix notation appears as the first sentence. The lookup-table analogy appears only in the closing Summary.

**Recommendation:** Prepend one sentence before the matrix line: "Think of `nn.Embedding` as a Python dictionary where every token ID maps to a fixed-length list of numbers — except the model updates those numbers during training." Then introduce $W_e$ as the matrix form of that dictionary.

---

### Gap 4 — BPE algorithm description has no single-sentence catchphrase — Priority: Medium

**Problem:** Part 2's intro describes the algorithm as four numbered steps — accurate, but without a one-liner the engineer can recall.

**Recommendation:** Add before the numbered list: "The core idea is deceptively simple: scan all adjacent symbol pairs in the corpus, find the pair that appears most often, collapse it into one symbol, and repeat." The 4 steps follow as precise mechanics.

---

### Gap 5 — `ignore_index=-100` motivation is abstract — Priority: Low

**Problem:** Part 5 states "if `CrossEntropyLoss` counts pad tokens as real predictions, the loss is inflated and misleading" — without using the actual batch numbers (3 sentences of lengths [6, 1, 13] padded to 13) to make the cost visceral.

**Recommendation:** Add: "In the three-sentence batch below — lengths [6, 1, 13] padded to 13 — 12 of 39 total label positions are padding. Without `ignore_index=-100`, those 12 positions inject misleading gradients into every parameter update."

---

### Gap 6 — PCA side-by-side plot has no inline observation after `plt.show()` — Priority: Low

**Problem:** The before/after PCA cell ends with `plt.show()` — no `print()` follows. The before-only PCA cell ends with "→ Random scatter: contract and agreement are nowhere near each other yet." The side-by-side cell has no symmetric "After" observation.

**Recommendation:** Add after `plt.show()`:
```python
print("→ After training: legal synonyms cluster together in embedding space.")
print("  'contract' and 'agreement' are now neighbours; 'indemnification' and 'liability' are near each other.")
print("  The model learned these groupings purely from 500 training steps on 20 sentences.")
```

---

## Do NOT Change

- Opening scenario and per-Part firm questions
- Predict-first boxes in Parts 2, 4, and closing
- BPE training-loop print table (step-by-step is excellent)
- "What just happened — and what's missing" transition cells
- Cosine similarity table (compensates for weak visual signal)
- Tier 1/2/3 summary ledger
- Toy → real bridge table in Part 6
- Forward pointers to `02-transformers` and `04-rnn-sequence-modeling`
- The "When to Use What — Tokenization Decisions" table
