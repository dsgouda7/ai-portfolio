# Plan: `03-rag-evaluation.ipynb` → authoring-guide parity

## 1 · Current state

**Topic.** RAG pipeline evaluation: four silent failure modes (correct / coherent-wrong /
hallucination / off-topic), four proxy metrics (context recall & precision, groundedness,
answer relevance, ROUGE-L correctness), a composite per-query dashboard, and a production
LLM-as-judge bridge. 52 cells, none executed (no kernel output persisted in the file).

**Structure (already present).**
- Title cell with a named-scenario brief (Riverside House's knowledge-base assistant),
  framed as a *dry run on a safe stand-in corpus* before the real manuscript catalog —
  a good, specific instance of Section 8.1 narrative framing, not a generic RAG-metrics tour.
- A real 14-document knowledge base + 8 labeled `TEST_CASES`, and a real hybrid
  (BM25 + embedding, RRF-fused) `rag_bot` built once and reused by every metric — metrics are
  computed by actually calling this pipeline, never fabricated "typical" scores (Section 8.2).
- One running question ("How does ReAct combine reasoning and acting?") threaded through
  Parts 1–5, exactly as Section 1's "one running example" principle asks.
- Per-part structure already follows crude-attempt → complaint → refinement (Section 5):
  keyword overlap → embedding cosine (Part 2), Jaccard → ROUGE-L (Part 5), etc.
- "Predict before you run" cells before every non-obvious reveal (Parts 2–5), "Your turn"
  exercise cells with a `# 👉`-style change-this-variable comment, and "What just happened"
  reflection cells closing every part — the functional equivalent of Section 3's 🔮/🧪
  markers, just without the literal emoji (see Section 3 note below).
- 3 "Common Pitfall" cells (Parts 2, 3, 5), each with a Bad/Good pair and a **runnable**
  Quick Health Check cell immediately after — matches Section 8.5 exactly.
- A "Riverside's question for this section" one-liner opens every major Part (Section 8.1).
- Honest, non-cherry-picked results: the closing scorecard (Section 8.7/8.4) branches its
  printed interpretation on the actual `df_dash` numbers from this run and explicitly says
  "that's the honest result of a 5-query toy set" when a metric misses its threshold, rather
  than only showing clean wins.
- Ends with a **Decision** section (not just a recap) scoring what ships against the opening
  brief, mirroring 01-llm-finetuning's Section 8.7 pattern precisely — summary/recap content
  is placed *before* the decision, not after.
- Several dense code cells already have "Code Walkthrough" markdown cells immediately after
  them (retriever/rag_bot cell, groundedness bar+heatmap cell, multi-metric fingerprint cell) —
  Section 9.1 pattern, including a shape/API note in each.
- Part 7's proxy-vs-production table doubles as a qualitative "axis" comparison (metric →
  proxy → true problem → why it breaks), covering the spirit of Section 10.3 even though this
  notebook has no literal M×N technique-choice grid to build one for.

**Conclusion:** this notebook is already close to parity — closer than a typical first pass.
The gaps below are the genuine, remaining ones, not a wholesale rewrite.

## 2 · Checklist scoring (Section 6 + 8.8 + 9.5 + 10.8 addenda)

| Item | Status |
|---|---|
| Title + roadmap table | ✅ present, with a "Riverside's question" column already added |
| One running example throughout | ✅ Q_THREAD used in Parts 1–5 |
| Claims proved, not asserted | ✅ every metric run against real `rag_bot`/`CASES` output |
| 🔮 Predict-first markers | ⚠️ semantically present ("Predict before you run") but no 🔮/🧪 emoji — **intentional, matches sibling `01-llm-finetuning.ipynb` in this same folder, which also omits them**. Not changing, see note below. |
| 🧪 Your turn exercises | ✅ present (Parts 2, 3, 4), no emoji (see above) |
| Reflection cells | ✅ "What just happened" after every part |
| Toy→real bridge table | N/A — no toy/production model split in this topic |
| Math always glossed | ✅ every formula (Recall@k, Precision@k, cosine, ROUGE-L/LCS) has prose gloss |
| Visualisation quality | ⚠️ several categorical-colour panels (pass/fail bars) have **no legend** — Section 10.2 gap |
| Section-banner code comments | ✅ `# ── Title ──` convention used throughout |
| Completed roadmap + insights | ✅ Summary section restates the table + "Key insights to keep" |
| Deterministic seeding | ✅ `np.random.seed(42)` set once, up top |
| **8.1** Named scenario + constraint | ✅ Riverside dry-run framing |
| **8.2** Real numbers only | ✅ |
| **8.3** Verify claims against real diffs | N/A — no trained-weight claims in this notebook |
| **8.4** Honest ambiguous/failed results | ✅ closing scorecard branches on real pass/fail counts |
| **8.5** Pitfalls + runnable health check | ✅ 3 instances |
| **8.6** Ablation framed as a scenario | N/A — no ablation study in this notebook's scope |
| **8.7** Close with decision, not recap | ✅ "The Decision" section after the Summary |
| **9.1** Code Walkthrough for dense cells | ⚠️ 3 dense (>45-line) cells still lack one: ROUGE-L/LCS cell, dual-heatmap grid cell, animated-radar cell |
| **9.2** Completion-only `generate()` | N/A — this notebook's `generate()` is extractive, not an LM completion helper |
| **9.3/10.3** Combination grid (qual + quant) | N/A — no M×N technique axis in this topic; Part 7's proxy table covers the same spirit |
| **9.4** FuncAnimation conventions | ✅ `plt.close(fig)` before `display(HTML(...))`, explanatory print before render, `to_jshtml(fps=...)` |
| **10.1** TOC for >40-cell notebook | ❌ missing — 52 cells, well past the ~40-cell threshold |
| **10.2** Per-subplot legends | ❌ 3 panels use colour-coded categories (pass/fail, grounded/ungrounded, token-match) with no legend |
| **10.4** State common model, then correct | ⚠️ Part 1 implies the "fluent = correct" assumption but never states it as an explicit named claim before disproving it — minor tightening opportunity |
| **10.5** Ground "expected outcome" claims in real facts | ✅ health checks use real corpus documents/tokens, not invented placeholders |
| **10.6** Progressive disclosure / split dense cells | Judged not to need cell-splitting (see §3.4) — dense cells get a Code Walkthrough instead, which is the guide's stated alternative for cells that are single-purpose rather than multi-stage-pipeline |
| **10.7** Cross-reference hygiene | ❌ 2 hardcoded references found: "before running **the next cell**" and "**The cell below** reads back..." |

## 3 · Ordered changes

1. **Add a Table of Contents** (Section 10.1) right after the title/brief cell — numbered list
   of all 7 Parts + Summary + Decision, with the 3 Common Pitfall subsections and the
   Multi-Metric Comparison subsection nested underneath their parent Part, plus the standard
   Ctrl+F/outline-panel caveat line.
2. **Fix the 2 cross-reference hygiene violations** (Section 10.7): reword "before running the
   next cell" (Part 3 predict cell) and "The cell below reads back…" (Decision section) to name
   the target artifact instead of its position.
3. **Add per-subplot legends** (Section 10.2) to the 3 panels that colour-code a category with no
   key: Part 2's per-query retrieval-relevance bar chart (strong/weak), Part 3's per-case
   groundedness bar chart (grounded/ungrounded), and Part 3's token-overlap heatmap
   (token-found/not-found).
4. **Add 3 Code Walkthrough cells** (Section 9.1) after the densest remaining code cells that
   don't already have one: the ROUGE-L/LCS dynamic-programming cell, the dual score/pass-fail
   heatmap grid cell, and the animated-plus-static radar-chart cell (this last one is the
   densest cell in the notebook and the only `FuncAnimation` cell without a walkthrough).
5. **Light-touch Part 1 tightening** (Section 10.4): state the common "a fluent, confident answer
   is probably a correct one" assumption as an explicit claim in the Part 1 opening, immediately
   before the four-failure-mode table disproves it, rather than leaving the assumption implicit.
6. **Do not** add 🔮/🧪 emoji markers — verified `01-llm-finetuning.ipynb` (this notebook's own
   narrative-framing reference, in the same folder) also omits them; adding them here would be
   an inconsistency with the established local convention, not a fix.
7. **Do not** force a toy→real bridge, ablation section, or M×N combination grid — none of these
   structural elements has a natural fit for a metrics-only evaluation notebook with a single
   fixed pipeline; Part 7's proxy/production table already does the equivalent qualitative-axis
   job for this topic.

Scope: this plan and all listed changes apply only to `03-rag-evaluation.ipynb`.
