# Plan — `transformers-keras.ipynb` parity pass

## 1. Current state

`transformers-keras.ipynb` is the Keras/TensorFlow twin of the gold-standard
`transformers.ipynb` (PyTorch). At 101 cells it already implements essentially the
full mechanistic curriculum the guide describes: 3D toy embeddings → raw attention →
positional encoding (sinusoidal + RoPE, including the "build it crude, then let the
complaint drive the next version" RoPE animation sequence) → Q/K/V → scaled
dot-product attention → multi-head attention → FFN/LayerNorm → full transformer block
→ mini decoder-only LM (train + generate) → toy→real bridge table → W_V-as-filter →
causal triangle/accumulation tower → three architectures (encoder / cross-attention /
encoder-decoder, each trained and measured) → DistilGPT-2 internals → summary.

A direct cell-by-cell diff against `transformers.ipynb` shows the two notebooks are
already at very close parity — same Parts, same proofs, same 🔮/🧪/reflection-cell
conventions, same toy sentence and embedding scheme. In a few places
`transformers-keras.ipynb` is actually *ahead* of the PyTorch reference:

- It has **Code Walkthrough cells** after `MultiHeadAttention`, the RoPE
  implementation, `CrossAttention`, and `DecoderBlockWithCrossAttn` +
  `MiniEncoderDecoder` — the PyTorch notebook has none of these.
- Its RoPE θ-ladder cell visualises 3 frequency pairs (`D_ROPE=6`); the reference
  cell only visualises 1 pair (`D_ROPE=2`).
- It proves the adjacent-pair vs. split-half (`rotate_half`) RoPE conventions are
  numerically equivalent up to a permutation — the reference has no equivalent cell.
- Its summary's "Key insights" list has 2 extra bullets vs. the reference.

So this pass is **not** a "close a large depth gap" pass — it's a targeted pass
against Section 10 of the authoring guide (navigation / progressive disclosure /
cross-reference hygiene), which is new and hasn't been applied to this notebook yet,
plus a couple of small pre-existing internal-consistency bugs the diff surfaced.

## 2. Checklist scoring (Section 6 + 8.8/9.5/10.8 addenda)

Section 6 (core checklist): **passes** — title+roadmap, single running example,
claims are proven not asserted, 🔮/🧪/reflection cells present throughout, toy→real
bridge table present, math is glossed, heatmaps/multi-panel comparisons used
throughout, section-banner comments + math-mirroring names + pedagogical prints,
completed summary table + key-insights list, seeds set deterministically.

Section 8.8 (narrative framing): **not applicable** — this notebook is a mechanistic
walkthrough (like the gold standard itself), not a business-scenario notebook like
`01-llm-finetuning.ipynb`. No changes needed here.

Section 9.5 (code clarity): **passes, exceeds reference** — Code Walkthrough cells
exist for every dense multi-pattern cell (MHA, RoPE, CrossAttention,
DecoderBlockWithCrossAttn). No `generate()`-style helper strips a prompt today, so
9.2 doesn't apply. No orthogonal-axis combination grid exists in either notebook —
9.3 doesn't apply to this topic (no two-orthogonal-technique-axis scenario, unlike
the fine-tuning notebook's data-objective × parameter-strategy grid). The RoPE
animation already follows the `FuncAnimation` + `to_jshtml` + `plt.close(fig)`
+ pre-explanation-print pattern (9.4).

Section 10.8 (navigation & consistency) — **this is where the real, actionable gaps
are**:

- [ ] **10.1 Table of Contents** — MISSING. The notebook has 101 cells, far past the
      ~40-cell threshold, and has no ToC cell today (neither does the PyTorch
      reference, but that's out of scope here).
- [x] **10.2 Per-subplot legends** — mostly compliant (colorbars present on nearly
      every heatmap). Three heatmap cells pass `cbar=False` on a *continuous*
      attention-weight scale (Head 0 vs Head 1 demo, Head P vs Head C proof,
      Exercise 4 mask toggle) — a small, easy compliance gap. (Note: the PyTorch
      reference has the identical `cbar=False` in the same three spots, so this
      is a "go beyond the reference" polish item, not a parity gap.)
- [x] **10.3 Qualitative pros/cons matrix** — not applicable. Nothing in this
      notebook has two *orthogonal* technique axes the way the fine-tuning notebook
      does (data objective × parameter strategy). The closest candidate — Encoder /
      Decoder / Encoder-Decoder — is a single axis with three options, already
      covered by the Part 13 comparison table + the quantitative parameter-count
      table in the same section.
- [x] **10.4 State the common mental model first** — already present: the RoPE
      "production trick" subsection explicitly states the natural assumption
      ("`rotate_half` is the same formula as the adjacent-pair matrix") and then
      shows what's actually true (same rotation, different pairing, proven
      numerically). No new instance forced elsewhere.
- [x] **10.5 Ground expected-outcome claims in real facts** — not a strong fit for
      this notebook (no corpus-derived "expected outcome" prose claims analogous to
      the fine-tuning notebook's chapter-detail grounding); the notebook's
      "expected outcome" statements are already tied to numbers computed earlier in
      the same run (loss/accuracy, cosine similarities, gradient norms).
- [ ] **10.6 Progressive disclosure** — reviewed all cells >30 lines; every
      multi-concept one (MHA, RoPE, CrossAttention, DecoderBlockWithCrossAttn) is
      already followed by a Code Walkthrough. No cell was found combining several
      separately-nameable steps *without* an existing walkthrough that would clearly
      benefit from a split — treating this as **not needed this pass** rather than
      manufacturing a split for its own sake.
- [ ] **10.7 Cross-reference hygiene** — grepped the whole notebook for
      `cell below|cells below|cell above|cells above|next cell|following cell|cells?
      (right )?after`. All 9 hits are "run the next cell" phrases that are always
      immediately adjacent to their target — compliant, but two *real* bugs
      surfaced during the close read instead:
      1. A cell (`Steps 2, 3 and 4 — all at once...`) names a forward target as
         "**Part 4b (RoPE inside attention)**" — but no `4b` heading exists anywhere
         in Part 4. It's a dangling *named* reference (better than a distance
         reference, but still points at nothing).
      2. The roadmap table (see below) lists concepts in an order that does not
         match the notebook's actual physical reading order.

## 3. Concrete changes (ordered)

1. **Fix the roadmap table order** (Section 2: "written last, after the notebook is
   finished, so it's accurate"). Today's 18-row table interleaves concepts in an
   order that doesn't match the body: it lists "Token Generation" as step 17 (next
   to the architecture-comparison rows) when the actual "Inference: Autoregressive
   Token Generation" Part physically appears right after training, long before the
   encoder/cross-attention/encoder-decoder Parts; "RNN Comparison" gets its own row
   even though it's part of the same Part-7 section as "Full Transformer Block".
   Reorder the table to match true physical reading order (merge the RNN-comparison
   row into the transformer-block row it actually shares a section with, and add a
   row for the "Attention: First Contact" section, which currently has no roadmap
   entry at all despite being a fully-developed section with its own animation and
   reflection cell). This only touches the roadmap table's row order/wording — no
   body Part numbering changes (Part 10 stays physically last, mirroring the exact
   same inherited quirk in the gold standard `transformers.ipynb`, which is
   out-of-scope to fix unilaterally in this notebook alone).
2. **Add a Table of Contents cell** (Section 10.1) directly after the title/roadmap
   cell, before the dependency-install cell. Nested list, anchor-linked to every `##`
   Part and the worthwhile `###`/`####` subsections (3a/3b/3d, 4b/4c, 13a-13d),
   ordered to match true physical reading order, with the required Ctrl+F/outline
   fallback caveat line.
3. **Add a missing `#### 4b. RoPE Inside Attention` heading** immediately before the
   "RoPE applied to Q and K inside attention" cell, so the existing "Part 4b" named
   forward-reference actually resolves to something (Section 10.7).
4. **Add colorbars to the three continuous-attention-weight heatmaps** that
   currently pass `cbar=False` (Head 0 vs Head 1 demo, Head P vs Head C proof,
   Exercise 4 mask toggle) — Section 10.2 polish, going one step beyond the
   PyTorch reference.

## 4. Deferred / out of scope for this pass

- Renumbering body `## Part N` headers so Part 10 (GPT-2 internals) is physically
  where its number implies — this exact inconsistency is inherited from
  `transformers.ipynb` itself (its "Part 10 — Cracking Open distilgpt2" is also
  physically last, after Part 13). Fixing it only in the Keras twin would make the
  two notebooks diverge structurally instead of mirroring each other, and touching
  ~15 headers + every internal "Part N" prose reference in a 101-cell notebook is a
  large, cross-cutting change relative to its payoff. Flagging it here in case a
  future pass decides to fix both notebooks together.
- Qualitative pros/cons matrix (10.3) and "ground in real corpus facts" (10.5) —
  not applicable to this notebook's topic/shape, as scored above.
- Progressive-disclosure cell splitting (10.6) — no qualifying cell found.
