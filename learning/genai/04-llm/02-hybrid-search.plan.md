# Plan: `02-hybrid-search.ipynb` → Authoring Guide Parity

Scope: this notebook only. No other notebook in `04-llm/` is touched.

## 1 · Current State Summary

**Topic:** Hybrid search for RAG — combining dense/semantic (sentence-transformer embeddings +
cosine similarity) and sparse/lexical (BM25) retrieval, fused via Reciprocal Rank Fusion (RRF) or
normalized weighted-score blending, benchmarked with Recall@K / MRR, and wired into a LangChain
`EnsembleRetriever`-style production pipeline.

**Structure:** 59 cells, 10 numbered Parts plus a Summary + closing "Decision" section, all threaded
through **one running example** (a 10-document medical knowledge base chosen as a stand-in for
Riverside House's manuscript catalog, per the `01-llm-finetuning.ipynb` narrative). Roadmap table in
the opening cell maps each Part to "Riverside's question for this section," exactly matching the
`04-llm/01-llm-finetuning.ipynb` narrative-framing pattern.

**Pedagogy already present (this notebook is unusually far along already — most of Sections 1–9 of
the guide are already satisfied):**

- 🔮-style "Predict first" cells before non-obvious reveals (semantic search on a rare term; BM25 on
  a synonym), each followed by a "Prediction Check" print block that states whether the guess was
  right and why (Section 5/3), though the emoji marker itself is missing (see §2).
- "Your turn" knob-turning exercises for RRF's `k` and the blend `alpha` (Section 5/3), again missing
  the 🧪/👉 emoji markers.
- Per-technique **Common Pitfalls + Quick Health Check** pairs (Section 8.5) after Parts 1, 5, and 7 —
  each health check *runs* code that proves the pitfall is real on this dataset (naive raw-score
  fusion measurably underperforms RRF; the tuned α is checked for generalization on held-out queries
  and the notebook honestly reports when it doesn't clearly win).
- **Honest, non-cherry-picked results** (Section 8.4 style): the BM25 "elevated blood pressure"
  demo branches its printed interpretation on whichever document actually ranks first; the alpha
  sweep explicitly calls out when Recall@5 is flat across all α (an "argmax tie-breaking artifact,"
  not a real preference) rather than asserting a false optimum.
- **Code Walkthrough** cells (Section 9.1) after the BM25 implementation, the RRF-based
  `EnsembleRetriever` pipeline, and the benchmark cell, each with numbered steps, a param table where
  relevant, and a "Shape/API note" callout.
- A real **quantitative combination grid** (Section 9.3): a dual heatmap (Recall@5 and MRR, method ×
  query) with a proper grey-for-untrained-style colorbar-legend on each panel.
- A `FuncAnimation` per-query Recall@5 reveal (Section 9.4) with `to_jshtml`, `plt.close(fig)` before
  `display(HTML(...))`, and an explanatory print before the animation renders.
- A closing **Decision** section (Section 8.7 style) that scores real numbers from cells run earlier
  in the same kernel against the opening brief, and explicitly revisits open/unresolved questions
  (validation-set size, toy-corpus-vs-197-chapters scale) rather than ending on a plain recap.
- Grounded, non-invented facts: the `RIVERSIDE_QUERY_ANALOGY` table's character/place names (Meridian's
  Promise, Kerra Valmont, the six founding families, Wei Lian, Project Drift, Eleanor Vance, the
  Observer) were checked against `01-llm-finetuning.ipynb` and are all real details from that
  notebook's corpus — not invented placeholders (Section 10.5 is already satisfied, no change needed).

## 2 · Checklist Scoring

### Section 6 (core parity checklist)

| Item | Status |
|---|---|
| Title + roadmap table | ✅ Present |
| Single running example | ✅ Medical KB throughout |
| Claims measured, not asserted | ✅ Extensive (Recall@5/MRR, RRF-vs-weighted, naive-fusion health check, α held-out check) |
| 🔮 Predict first | ⚠️ Present in substance, **missing the 🔮 emoji marker** on both instances |
| 🧪 Your turn + `# 👉 CHANGE` | ⚠️ Present in substance, **missing 🧪 and 👉 emoji markers** on both instances |
| Reflection cells ("what just happened") | ✅ Present after Parts 1 and 4 |
| Toy→real bridge table | ✅ Present (Part 10, "Bridging from Toy to Production") |
| Math always glossed | ✅ BM25, RRF, cosine, min-max/z-score all glossed |
| Heatmaps/multi-panel + graceful plotly fallback | ⚠️ Good multi-panel use; **no `HAS_PLOTLY` fallback anywhere** (notebook doesn't use plotly at all — acceptable, matplotlib-only is fine for this topic, not a gap) |
| Section-banner code comments, pedagogical prints | ✅ Present throughout |
| Completed roadmap + "Key insights to keep" | ✅ Present in Summary |
| Deterministic seeds | ✅ `np.random.seed(42)` set once up top |

### Section 8.8 (narrative framing addendum)

| Item | Status |
|---|---|
| Named scenario + concrete ask | ✅ Riverside House, 197 chapters |
| **Real constraint that shapes technical choices** | ❌ **Gap** — the opening cell never restates the privacy/laptop-CPU constraint from `01-llm-finetuning.ipynb`, so the notebook's actual choice of a local `sentence-transformers` model (never a hosted embeddings API) reads as an arbitrary default rather than a constraint-driven decision |
| Each section states which brief-question it answers | ✅ "Riverside's question for this section" callout on every Part |
| Real numbers from real objects, not fabricated | ✅ Every chart/metric comes from cells run earlier in-kernel |
| Weight/parameter change claims verified by diff | N/A — no trained model weights in this notebook (retrieval, not fine-tuning) |
| Honest reporting of ambiguous/failed results | ✅ Alpha-sweep flatness, held-out generalization check |
| Pitfalls → runnable health check | ✅ Present after Parts 1, 5, 7 |
| No dangling "Experiment N" forward references | ✅ N/A, no numbered experiments used |
| Closes with scored decision, recap before it | ✅ Scorecard + Decision section |

### Section 9.5 (code clarity addendum)

| Item | Status |
|---|---|
| Code Walkthrough after any 30+ line multi-call cell | ⚠️ Present after most, but **two cells are dense multi-concept blocks that should be *split* (Section 10.6) rather than only walked-through after the fact** — see below |
| `generate()`-style helpers return only completion | N/A — no generation helpers in this notebook |
| Quantitative combination grid after eval section | ✅ Dual heatmap (Recall@5 × MRR) present |
| Token/step-axis heatmap → `FuncAnimation` | ✅ Per-query Recall@5 animation present |

### Section 10.8 (navigation/consistency addendum)

| Item | Status |
|---|---|
| TOC for notebooks >~40 cells | ❌ **Gap** — 59 cells, no Table of Contents cell |
| Per-subplot legend, no color-coded region without a key | ❌ **Gap** — the 2×2 "Search Gap" figure (Part 1) color-codes green=target-doc vs steelblue/coral=other with **zero legends on any of the 4 subplots**; the animated Recall@5 bar chart's `Patch` legend is only added on the *final* frame instead of existing from the first frame |
| Qualitative combination matrix *before* the quantitative one | ⚠️ **Gap** — a "Best for / Blind spot" qualitative table exists, but it sits in Part 10 immediately *before* the quantitative dual heatmap, not back in Part 1/4 where retrieval-method and query-type are first established as orthogonal axes |
| State common mental model, then correct | ⚠️ **Gap** — no place in the notebook explicitly states a common (slightly-wrong) shorthand about RRF or BM25 before correcting it; the closest existing moment (the IDF-rarity-vs-"feels generic" note in Part 3) is a single print aside, not the full 3-beat structure |
| Ground "expected outcome" claims in real facts | ✅ Already satisfied — verified `RIVERSIDE_QUERY_ANALOGY` names against `01-llm-finetuning.ipynb`, all real |
| Progressive disclosure — split 30+ line multi-concept cells | ❌ **Gap** — the `EnsembleRetriever` production-pipeline cell (~108 lines: class definition → doc conversion → embeddings init → FAISS build → BM25Retriever build → ensemble creation → test loop → Riverside analogy table) is the densest cell in the notebook and combines far more than one nameable step; its own Code Walkthrough cell already enumerates "5 key patterns," which is effectively a ready-made split plan |
| Cross-reference hygiene | ⚠️ **Minor gap** — three hardcoded-distance phrases found by grep: two "...then run the next cell" (Part 1 predictions) and one "the dual heatmap in the next cells" (Part 10 walkthrough) — the first two are *currently* true but fragile; the third is already slightly wrong (it's 2 cells forward, not the immediate next one) |

## 3 · Ordered Changes To Implement

1. **Add a Table of Contents cell** (Section 10.1) directly after the title/roadmap cell, before the
   dependency-install cell. Numbered list of all 10 Parts + Summary/Decision, with `###`-level
   sub-entries for the "Common Pitfalls" callouts and named sub-concepts (Predict-first cells, Your-turn
   exercises) a reader would plausibly jump to. Include the `Ctrl+F`/outline-panel fallback caveat line.
2. **Thread the real constraint into the opening cell** (Section 8.8): add one paragraph explicitly
   recalling Riverside's "no manuscript data ever leaves the building, laptop hardware, not a GPU
   cluster" constraint from `01-llm-finetuning.ipynb`, and use it to justify why this notebook's
   embedding step runs a local `sentence-transformers` model rather than a hosted embeddings API —
   turning an already-made code choice into a constraint-driven decision instead of an unexplained
   default.
3. **Fix emoji conventions** (Section 3) — mechanical find/replace, no content change:
   - `#### Predict first` → `#### 🔮 Predict first` (2 occurrences).
   - `### Your turn — ...` / `# Your turn — ...` → `### 🧪 Your turn — ...` / `# 🧪 Your turn — ...`
     (2 occurrences: RRF `k` exercise, alpha-tuning exercise).
   - `# CHANGE k_val ...` / `# CHANGE alpha_manual ...` → `# 👉 CHANGE k_val ...` /
     `# 👉 CHANGE alpha_manual ...`.
4. **Add per-subplot legends** (Section 10.2):
   - The 2×2 "Search Gap" comparison figure (Part 1): add a `Patch`-handle legend to each of the 4
     subplots distinguishing "Target document" (green) from "Other result" (steelblue for semantic
     panels, coral for lexical panels).
   - The animated Recall@5 bar chart (near the Summary): move the `Patch`-handle legend so it is
     created once *before* `FuncAnimation` starts (draw it directly on `ax_anim` outside
     `_draw_recall_frame`), so it's visible from frame 0, not only the last frame.
5. **Add an early qualitative pros/cons matrix** (Section 10.3) in Part 4 ("Why Hybrid Search Wins"),
   right where retrieval method and query type are established as the two orthogonal axes driving
   the whole notebook — a Markdown table: rows = {Semantic, Lexical (BM25), Hybrid (RRF)}, columns =
   {Rare/exact-term query, Paraphrase/synonym query}, each cell a short Pros/Cons. Follow with 1-2
   sentences naming the pattern (hybrid is the only row with no "Cons" cell reading "misses the
   query entirely") and a forward-pointer ("further down, the Benchmarking section's dual heatmap
   confirms this pattern with real measured Recall@5/MRR numbers") — using distance-free phrasing
   per item 8 below.
6. **Add a "common mental model, then correct" beat** (Section 10.4) in Part 5 where RRF is
   introduced: state the common shorthand ("RRF is often described as just 'averaging the ranks' from
   each system"), show what that would actually imply (a plain rank-average formula), then correct it
   by walking through what the real formula computes (reciprocal, not rank value directly; the `k`
   smoothing constant; why rank 1 vs rank 2 is a much bigger jump than rank 20 vs rank 21) — all in the
   same or an immediately following markdown cell, before the RRF code cell.
7. **Progressive disclosure split** (Section 10.6): split the `EnsembleRetriever` production-pipeline
   cell into ~5 smaller cells, each preceded by a 2-4 sentence intro, following the 5 steps its own
   Code Walkthrough cell already names (ensemble class → LangChain doc conversion → embeddings/FAISS
   build → BM25Retriever build → test loop + Riverside analogy table). Retitle the existing Code
   Walkthrough cell "Code Walkthrough: EnsembleRetriever — Recapped" and reword its opening line so it
   no longer claims the code was "combined into one cell."
8. **Cross-reference hygiene fixes** (Section 10.7): reword the three hardcoded-distance phrases found
   by grep — two "...then run the next cell" → name the artifact instead ("...then run the semantic
   search cell to check" / "...then run the BM25 search cell to check"); "the dual heatmap in the next
   cells" → "the dual heatmap further down." Re-grep after all edits (including the Part 4 forward
   pointer added in item 5 and the cell split in item 7) to confirm no new hardcoded-distance phrases
   were introduced.

**Deferred / out of scope for this pass:**

- Splitting the BM25 Lexical Search Implementation cell (~76 lines) the same way as the
  `EnsembleRetriever` cell — it already has a solid 4-step Code Walkthrough and is lower-effort/lower-
  risk to leave as-is; flagged here for a future pass if the notebook is revisited.
- Adding legends to the 6-panel score-normalization comparison grid (Part 6) — each subplot is
  single-series/single-color, so the mechanical checklist item applies but the pedagogical payoff is
  low relative to the risk of another large multi-replace pass on a cell already dense with subplot
  logic; left for a future pass.
- No `HAS_PLOTLY`/matplotlib-fallback pattern is added — this notebook never uses plotly, so the
  pattern doesn't apply; not treated as a real gap.
