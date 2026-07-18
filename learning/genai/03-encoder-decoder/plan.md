# `encoder-decoder.ipynb` — Parity Plan

Scope: this plan applies only to `03-encoder-decoder/encoder-decoder.ipynb`. It does not
require changes to any other notebook.

## 1 · Current state

38 cells, no execution outputs cached. Topic: build an encoder-decoder transformer
from scratch in PyTorch (`MultiHeadSelfAttention`, `EncoderBlock`, `CrossAttention`,
`DecoderBlock`, `EncoderDecoder`) around a single running example — reversing a
length-4 integer sequence — then bridge to real T5/BART.

Structure (7 numbered Parts + Summary, already following the guide's skeleton):

1. Title + roadmap table, Setup cell.
2. **Part 1** — The Encoder-Decoder Contract (2×2 taxonomy: unidirectional/bidirectional
   × fixed/variable length; cross-attention formula introduced).
3. **Part 2** — The Encoder: bidirectional self-attention. Already contains a strong
   "same weights, one variable changed" experiment (Section 5, point 4 of the guide):
   one `MultiHeadSelfAttention` instance run once with `mask=None` and once with a
   causal mask, heatmapped side by side. A 🔮-style "Predict before you run" cell
   precedes it. A 🧪-style "Your turn" exercise on head-count vs. attention diversity
   follows.
4. **Part 3** — The Bottleneck Problem: RNN-style fixed-vector compression is explained,
   then *measured* — cosine similarity between two different source sequences is
   nearly unaffected at individual encoder positions but collapses after mean-pooling.
   Good "don't assert, demonstrate" example already.
5. **Part 4** — Cross-Attention: the Q/K/V role-split is presented directly, with the
   $(T \times S)$ asymmetry noted, but with **no explicit "Problem 1 / Problem 2"
   numbered callout** motivating why cross-attention (rather than some other design)
   is the fix. This is a real gap — see Section 2 below.
6. **Part 5** — Full model + teacher-forced training loop on the reversal task, with
   loss curve + sequence accuracy. Code Walkthrough cells already exist for the
   dense multi-class code cells but the cells themselves are not yet split
   (Section 10.6 progressive disclosure is not applied — see Section 2).
7. **Part 6** — Cross-attention heatmap proving the learned anti-diagonal routing,
   plus a `FuncAnimation` revealing rows step by step. Matches Section 9.4's
   animation rules (explains before rendering, `plt.close(fig)` before `display`,
   `to_jshtml`) but the animation's grey-vs-colored distinction has no legend
   (Section 10.2 gap).
8. **Part 7** — Toy→real bridge: a full parameter-mapping table (`d_model`, `n_heads`,
   `d_ff`, `n_layers`, vocab, params) against T5-small/BART-base, followed by a live
   (try/except-guarded) T5-small summarisation call. **This already satisfies the
   toy→real bridge-table requirement — no action needed here.**
9. Summary — completed roadmap table + "Key insights to keep" bullets. Matches the
   guide's required closing shape.

Reference-notebook cross-check (`02-transformers/transformers.ipynb`): its actual
"Predict"/"Your turn" headings are plain (`#### Predict first`, `### Your turn - X`)
— it does **not** use the 🔮/🧪 emoji markers the guide's Section 3 describes as
"exact markers to reuse." Since the two gold-standard notebooks are the actual
target of parity (per the guide's own framing), this plan follows the reference
notebooks' real style (plain headings) rather than retrofitting emoji that aren't
actually used anywhere in the collection yet. No renaming needed for
`encoder-decoder.ipynb`'s existing "Predict before you run" / "Your turn" headings —
they already match the gold standard's actual pattern.

The gold standard's cross-attention section *does* use an explicit numbered
callout — `**Problem 1 - The causal mask cuts off the source.**` /
`**Problem 2 - Source and target have different lengths.**` — to motivate cross-
attention over the naive alternative of concatenating encoder output into the
decoder's own self-attention sequence. `encoder-decoder.ipynb` does not yet have
this pattern; Part 3/4 currently only motivate cross-attention against the
*RNN bottleneck* alternative, not the *"just concatenate into self-attention"*
alternative. Both are real, complementary failure modes worth showing.

## 2 · Checklist scoring

**Section 6 (core checklist):**

| Item | Status |
|---|---|
| Title + roadmap table | ✅ |
| Single running example throughout | ✅ (integer reversal, unchanged for all 7 parts) |
| Every claim measured, not asserted | ✅ mostly — encoder/decoder mask heatmap, bottleneck cosine-similarity proof, anti-diagonal cross-attention proof, T≠S shape exercise |
| 🔮 Predict-first before non-obvious reveals | ✅ (4 instances, plain-heading style matching gold standard) |
| 🧪 Your-turn exercises near the concept, `# 👉 CHANGE` comment | ✅ (2 instances: head-count diversity, T≠S shapes) + 1 forward-pointing "change one variable" cell in Part 7 |
| Reflection cells ("what just happened") | ✅ (4 instances, each planting the next part's question) |
| Toy↔real bridge with parameter-mapping table | ✅ (Part 7) |
| Math always glossed in plain English | ✅ |
| Heatmaps/multi-panel over single plots | ✅ (2×2 grid patterns used throughout) |
| Optional interactive viz degrades gracefully | N/A — no plotly used here, matplotlib/seaborn only, which is fine for this topic |
| Section-banner code comments, math-mirroring names | ✅ (`W_Q`, `Q`, `K`, `V`, `d_k`, `attn_w` throughout) |
| Completed roadmap + "Key insights to keep" | ✅ |
| Deterministic seeds wherever numbers are quoted | ✅ (`torch.manual_seed` set before every quoted-number cell) |

**Section 8.8 (narrative framing addendum):** Not applicable in the same way as
`04-llm/01-llm-finetuning.ipynb` — this notebook's scope matches the *mechanistic*
gold standard (`02-transformers/transformers.ipynb`), which itself has no named
business scenario either. Forcing a "named client" framing onto a from-scratch
architecture build would be artificial. No action taken here; this mirrors the
guide's own distinction between mechanistic-depth notebooks and narrative-framing
notebooks.

**Section 9.5 (code-clarity addendum):**

| Item | Status |
|---|---|
| Code Walkthrough after any 30+ line multi-concept cell | ⚠️ Walkthrough cells exist, but the *underlying code cells are not split* — Section 10.6 supersedes this for cells combining several genuinely separate steps (see below). |
| `generate()`-style helpers return only completion tokens | N/A — no autoregressive `.generate()` helper in this notebook (training uses teacher forcing only; T5 demo cell calls `t5_model.generate()` directly and decodes the full output because it's a single illustrative call, not a repeated helper) |
| Combination grid for 2 orthogonal technique axes | N/A — no 2-axis technique comparison in this notebook (single architecture, not a training-strategy grid) |
| Token-position heatmap → `FuncAnimation` | ✅ Part 6 already uses `FuncAnimation` + `to_jshtml` for the step-by-step cross-attention reveal |

**Section 10.8 (navigation/consistency addendum):**

| Item | Status |
|---|---|
| Table of Contents for 40+ cell notebooks | ⚠️ Currently 38 cells (just under threshold) — will cross 40 once progressive-disclosure splits are applied, so **add a TOC**. |
| Per-subplot legends for every color-coded region | ❌ Two gaps: (1) the encoder-vs-decoder mask heatmap (Part 2) sets `cbar=False` on both panels, dropping the continuous-value legend a `sns.heatmap` needs; (2) the Part 6 `FuncAnimation` distinguishes "revealed" vs. "pending" rows by color (categorical) but only explains it in a title string / print statement, not a `Patch`-based legend. |
| Qualitative combination matrix before quantitative one | N/A — no 2-axis technique grid exists in this notebook (see above) |
| State common mental model, then correct it | ⚠️ Partial — Part 4 states the encoder/decoder mask asymmetry as a fact but doesn't first name the common shorthand ("decoder attention is always causal") before correcting it for cross-attention. Cheap to fix while rewriting the Problem 1/2 callout. |
| Ground "expected outcome" claims in real facts | ✅ — every expected-value claim (anti-diagonal position, accuracy threshold, random baselines) is derived from the actual reversal task's definition, not invented |
| Split 30+ line multi-concept cells into small cells + short intros | ❌ Four code cells combine multiple separately-nameable steps in one cell: `MultiHeadSelfAttention` + `FeedForward` + `EncoderBlock` + `sinusoidal_pe`/`MiniEncoder` + sanity check (113 lines, Part 2); `DecoderBlock` + `EncoderDecoder` + architecture inspection (85 lines, Part 5); `CrossAttention` class + demo (67 lines, Part 4); `build_dataset` + loaders + model/optimizer init + training loop (66 lines, Part 5). |
| Cross-reference hygiene — no hardcoded cell distance | ❌ Three "Predict before you run" cells say "run the next cell to see the reveal/find out/measure it" — fragile positional phrasing per Section 10.7's own grep target. |

## 3 · Ordered changes to implement

1. **Fix cross-reference hygiene** (cheapest, do first): reword the three
   "run the next cell..." phrases (row-0 encoder-heatmap predict, mean-pooling
   predict, accuracy predict) to name the target artifact instead of its position.
2. **Add explicit "Problem 1 / Problem 2" numbered callout** motivating
   cross-attention, folded into the existing Part 4 intro cell:
   - Problem 1 — recap the Part-3-proven fixed-vector bottleneck.
   - Problem 2 — new content: naive concatenation of encoder output directly into
     the decoder's self-attention sequence fails for two reasons — the decoder's
     causal mask still blocks it, and source/target length mismatch ($S \ne T$)
     means the two sequences can't be stacked as one set of positions.
   - Fold in a "state the common shorthand, then correct it" beat for the masking
     asymmetry: "decoder attention is always causal" is the common shorthand;
     cross-attention breaks it (no mask at all on the encoder side) because Q and
     K/V come from different sequences, not one sequence being generated in order.
3. **Progressive disclosure** — split the two most severe multi-concept cells first
   (highest step count, highest line count), then the two smaller ones if time
   allows, retitling each existing "Code Walkthrough" as "..., Recapped" per
   Section 10.6's explicit instruction (never delete a walkthrough that still adds
   detail the short intros don't cover, e.g. the shape tables):
   - Part 2 mega-cell → 5 cells: `MultiHeadSelfAttention` / `FeedForward` /
     `EncoderBlock` / `sinusoidal_pe` + `MiniEncoder` / sanity check, each preceded
     by a 2-4 sentence intro. Retitle the walkthrough as
     "Code Walkthrough: MiniEncoder Building Blocks, Recapped."
   - Part 5 mega-cell → 3 cells: `DecoderBlock` / `EncoderDecoder` / architecture
     inspection, each with a short intro. Retitle the walkthrough as
     "Code Walkthrough: Full Encoder-Decoder Architecture, Recapped."
   - Part 4 `CrossAttention` cell → 2 cells: class definition / demo, each with a
     short intro. Retitle the walkthrough as
     "Code Walkthrough: CrossAttention Module, Recapped."
   - Part 5 training-loop cell → up to 3 cells: dataset/loader construction /
     model+optimizer init / training loop itself, each with a short intro.
     Retitle the walkthrough as
     "Code Walkthrough: Teacher-Forced Seq2Seq Training Loop, Recapped."
4. **Fix visualization legends** (Section 10.2):
   - Re-enable colorbars (`cbar=True`, the seaborn default) on both panels of the
     encoder-vs-decoder mask comparison heatmap in Part 2.
   - Add a static `Patch`-based legend ("Revealed step" / "Not yet generated") to
     the Part 6 `FuncAnimation` cross-attention-reveal cell, added once before the
     animation loop starts, per Section 10.2's explicit rule for animations whose
     categorical color is decided inside the per-frame update function.
5. **Add a Table of Contents** cell immediately after the title/roadmap cell, since
   the progressive-disclosure splits above will push the notebook past ~40 cells.
   Numbered list, one entry per `##`-level Part plus the handful of `###`
   subsections worth linking directly (Code Walkthroughs, Predict/Your-turn
   cells that introduce a new named exercise), with the standard outline-panel /
   `Ctrl+F` fallback caveat.
6. **Update the intro roadmap table's "Part" wording only if the split changes a
   Part's shape** — check after edits; expected to remain accurate since splits
   only affect cells *within* existing Parts, not the Part boundaries themselves.

Deliberately **not** doing in this pass (see final report for reasoning): splitting
every code cell to a hard 30-line ceiling (diminishing returns below the four
identified cells); adding a full Section 8 business-narrative scenario (out of
scope for a mechanistic architecture notebook); adding a qualitative/quantitative
combination-grid pair (no 2-axis technique comparison exists in this notebook to
grid).
