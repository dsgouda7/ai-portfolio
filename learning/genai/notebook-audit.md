# Notebook Audit — `learning/genai/`

**Audit date:** 2026-07-14
**Reference standard:** `learning/genai/02-transformers/transformers.ipynb`
**Authoring guide:** `learning/genai/authoring-guide.md`

---

## Summary Table

| Notebook | Cells (MD/Code) | Running Example | Predict-First | Reflect Cells | Banners | Arrow Prints | Framework | Score |
|---|---|---|---|---|---|---|---|---|
| `01-rnns/MIT/TF_Part1_Intro.ipynb` | 56 (29/27) | Y — Circle vs Square | 2 | 0 | Y (18/27) | Y (3) | Keras/TF | **Good** |
| `01-rnns/MIT/PT_Part1_Intro.ipynb` | 68 (37/31) | Y — Tensors/RNN | 3 | 5 | Y (26/31) | Y (10) | PyTorch | **Good** |
| `02-transformers/transformers.ipynb` | 97 (43/54) | Y — "the cat sat..." | 7 | 1+ | Y (40/54) | Y (30) | Keras/TF | **Excellent** |
| `03-encoder-decoder/encoder_decoder.ipynb` | 0 (empty) | — | — | — | — | — | — | **Poor** |
| `04-llm/hybrid-search.ipynb` | 46 (25/21) | Y — doc corpus | 2 | 1 | Y (12/21) | Y (2) | sklearn/SentenceT | **Good** |
| `04-llm/llm-gateway.ipynb` | 56 (30/26) | Y — mock providers | 5 | 6 | Partial (5/26) | Y (3) | None (pure Python) | **Good** |
| `04-llm/rag-evaluation.ipynb` | 38 (17/21) | Y — failure-mode driven | 0 | 6 | Y (21/21) | Y (10) | None (pure Python) | **Good** |
| `05-llm-tuning/llm_finetuning_deep_dive.ipynb` | 44 (25/19) | Y — distilgpt2 | 0 | 0 | N (0/19) | Y (7) | PyTorch/HuggingFace | **Needs Work** |

**Score key:** Excellent = 10–12 checklist items met | Good = 6–9 | Needs Work = 3–5 | Poor = 0–2

---

## Authoring Guide Checklist Reference

From `authoring-guide.md` Section 6, the 12 items audited:

1. Opens with title + roadmap table
2. Single running example threaded through every section
3. Every non-trivial claim proved with code (not just asserted)
4. `🔮 Predict first` cells before non-obvious reveals
5. `🧪 Your turn` exercises near the concept they drill
6. Reflection cells ("What just happened") close each part
7. Toy → real bridge with explicit parameter-mapping table
8. Math never left un-glossed; every formula has plain-English explanation
9. Visualisations: heatmaps, multi-panel, optional interactive with fallback
10. Code cells: section-banner comments, math-mirroring variable names, pedagogical `print` with `→`
11. Ends with completed roadmap table + "Key insights to keep" bullet list
12. Deterministic seeds set before any cell whose numbers appear in markdown

---

## Detailed Assessments

---

### 1. `01-rnns/MIT/TF_Part1_Intro.ipynb`

**Score: Good**

**Structure:** 56 cells (29 markdown, 27 code). The notebook opens with a lab header ("Lab 1: Intro to TensorFlow/Keras and Music Generation with RNNs") followed by a proper "What You'll Learn" section and a `## 1.0 Our Running Example: Circle vs. Square (2D Binary Classifier)` cell. The Circle-vs-Square example persists through Part 1 before the notebook pivots to music generation with RNNs.

**Checklist assessment:**

| Item | Status | Notes |
|---|---|---|
| 1. Title + roadmap table | PASS | Has title and structured learning objectives |
| 2. Single running example | PASS | Circle vs Square (Part 1), RNN music (Part 2) — two examples, not one threaded through |
| 3. Claims proved with code | PARTIAL | TODO blocks require student completion; some assertions without measurement |
| 4. Predict-first cells | PARTIAL | 2 predict cells present |
| 5. Your turn exercises | FAIL | No `🧪` cells; uses `_TODO_` blocks instead of the guide's pattern |
| 6. Reflection cells | FAIL | Zero "What just happened" cells |
| 7. Toy → real bridge | PASS | Circle/Square toy to real neural net demonstrated |
| 8. Math glossed | PASS | LaTeX equations followed by plain-English explanation |
| 9. Visualisations | PASS | matplotlib plots throughout |
| 10. Section banners + arrow prints | PASS | 18/27 code cells have `# ──` banners; 3 cells with arrow prints |
| 11. Completed roadmap + key insights | FAIL | Last markdown cell is a music-generation options menu, not a summary |
| 12. Deterministic seeds | PARTIAL | Seeds set in some cells |

**Top 2 gaps:**

1. **No reflection cells.** There are zero "What just happened" cells. Every concept transition is abrupt: the notebook jumps from showing a result to the next concept without naming what was just demonstrated or planting the crack that motivates the next step. Add a short reflection markdown cell at the end of each Part (at minimum after the loss/accuracy reveal and after the RNN architecture introduction).

2. **Closing summary is missing.** The final markdown cell describes optional TunesFormer model choices — there is no completed roadmap table and no "Key insights to keep" bullet list. The notebook has no proper landing; a reader who finishes cannot easily recall what they proved. Add a `## Summary` cell with the roadmap restated as a completed journey and three to five one-line quotable takeaways.

---

### 2. `01-rnns/MIT/PT_Part1_Intro.ipynb`

**Score: Good**

**Structure:** 68 cells (37 markdown, 31 code). The strongest of the RNN notebooks. Has 5 reflection cells ("What just happened" pattern), 3 predict-first cells, 26 out of 31 code cells with section-banner comments, 10 cells with arrow `→` prints, and a proper `## Summary — What You Built` table at the end. Framework is pure PyTorch with `mitdeeplearning` helper package.

**Checklist assessment:**

| Item | Status | Notes |
|---|---|---|
| 1. Title + roadmap table | FAIL | First cell is an MIT/Colab badge table, not a title-pitch-roadmap cell |
| 2. Single running example | PASS | Tensors → Sinusoid → RNN music generation, linked throughout |
| 3. Claims proved with code | PASS | Demonstrations with plots at each stage |
| 4. Predict-first cells | PASS | 3 predict-first cells |
| 5. Your turn exercises | FAIL | No `🧪` exercise cells |
| 6. Reflection cells | PASS | 5 reflection cells |
| 7. Toy → real bridge | PASS | Toy sinusoid → real music RNN |
| 8. Math glossed | PASS | Equations followed by prose explanation |
| 9. Visualisations | PASS | matplotlib waveforms, loss curves |
| 10. Section banners + arrow prints | PASS | 26/31 banner cells, 10 arrow-print cells |
| 11. Completed roadmap + key insights | PASS | Summary table present at end |
| 12. Deterministic seeds | PASS | Seeds set at top |

**Top 2 gaps:**

1. **Opening cell is a badge table, not a title + pitch + roadmap.** The first markdown cell is the MIT/Colab/GitHub badge HTML. There is no single opening screen that tells a reader the notebook's topic, one-paragraph pitch, and a `| Step | Concept | Key Idea |` roadmap table. A new reader has no map before diving in. Prepend a proper title cell (or convert the badge cell to include it) that gives the hook, scope, and roadmap in one screen.

2. **No exercise cells.** The notebook has no `🧪 Your turn` cells at any point. A reader passively observes every demonstration without being asked to predict a change or turn a knob. After each major mechanism (tensor ops, gradients, RNN forward pass), add one `🧪` markdown cell posing a change-one-variable question with a `# 👉 CHANGE ...` comment in the adjacent code cell.

---

### 3. `02-transformers/transformers.ipynb`

**Score: Excellent**

**Structure:** 97 cells (43 markdown, 54 code). This is the declared gold standard. "The cat sat on the mat" threads every concept from tokenisation through GPT-2 internals. 7 predict-first cells, 40 of 54 code cells with section banners, 30 cells with pedagogical arrow prints, 3 exercise cells, toy-to-real bridge with parameter-mapping table, completed summary with key insights.

**Checklist assessment:**

| Item | Status | Notes |
|---|---|---|
| 1. Title + roadmap table | PASS | Opens with title, pitch, and `| Part | Concept | Key Idea |` table |
| 2. Single running example | PASS | "the cat sat on the mat" from tokenisation to GPT-2 |
| 3. Claims proved with code | PASS | Every mechanism proved: √dₖ saturation, multi-head specialisation, residual gradients |
| 4. Predict-first cells | PASS | 7 predict cells |
| 5. Your turn exercises | PASS | 3 exercise cells |
| 6. Reflection cells | PARTIAL | 1 explicit "What just happened" cell; others use "So — but…" variant phrasing |
| 7. Toy → real bridge | PASS | 3D toy space → DistilGPT-2, parameter table included |
| 8. Math glossed | PASS | All LaTeX followed by plain-English gloss |
| 9. Visualisations | PASS | Seaborn heatmaps, Plotly 3D with matplotlib fallback, FuncAnimation |
| 10. Section banners + arrow prints | PASS | 40/54 banner cells, 30 arrow-print cells |
| 11. Completed roadmap + key insights | PASS | Summary table + bullet list |
| 12. Deterministic seeds | PASS | Seeds set throughout |

**Top 2 gaps (relative to its own standard):**

1. **Reflection cells are sparse for a 97-cell notebook.** Only 1 cell uses the explicit "What just happened" heading. Many parts close with a transition paragraph embedded in the next cell's markdown, but this is easy for a reader to miss during a quick skim. Adding the `#### What just happened` heading consistently after each Part's reveal cell (even if the prose already exists) would make the structure more scannable.

2. **14 code cells (26%) lack section-banner comments.** The guide requires every code cell to open with `# ── Description ──`. The 14 cells missing banners are presumably the smaller utility/helper cells, but a reader skimming the raw source still hits unmarked territory. A mechanical pass to add minimal banners to the remaining cells would complete the pattern.

---

### 4. `03-encoder-decoder/encoder_decoder.ipynb`

**Score: Poor**

**Structure:** 0 cells. The file exists on disk but contains an empty cell list. The notebook is a stub.

**Assessment:** Not auditable in its current state. All 12 checklist items are unmet by definition.

**Top 2 gaps:**

1. **Notebook is completely empty.** There is no content of any kind — no title, no cells, no code. The encoder-decoder topic (Seq2Seq, cross-attention, Bahdanau attention, encoder vs. decoder masking) is an important bridge between the RNN and full-transformer notebooks and is entirely absent.

2. **No authoring plan exists.** Before any cells can be written, a `plan.md` should be drafted in `03-encoder-decoder/` using the authoring guide's checklist to map out the running example (recommend: English-to-French translation with a short fixed sentence), the toy-to-real bridge (toy Seq2Seq → MarianMT), and the part structure.

---

### 5. `04-llm/hybrid-search.ipynb`

**Score: Good**

**Structure:** 46 cells (25 markdown, 21 code). Opens with a clear title and "Why One Search Alone Isn't Enough" hook. Uses a concrete document corpus throughout; demonstrates failure cases of pure BM25 and pure semantic search before building RRF fusion. 12 of 21 code cells have section banners. Uses sklearn, BM25Okapi, and SentenceTransformers — no deep-learning framework dependency. Closes with a completed roadmap summary table.

**Checklist assessment:**

| Item | Status | Notes |
|---|---|---|
| 1. Title + roadmap table | PASS | Title, pitch, and roadmap table present |
| 2. Single running example | PASS | Fixed document corpus threaded through all comparisons |
| 3. Claims proved with code | PASS | Failure cases proved on concrete queries before fusion is introduced |
| 4. Predict-first cells | PARTIAL | 2 predict cells; fewer than the density the guide recommends |
| 5. Your turn exercises | PARTIAL | 1 exercise cell present |
| 6. Reflection cells | PARTIAL | 1 reflection cell; most Part transitions lack explicit "What just happened" |
| 7. Toy → real bridge | PARTIAL | Shows progression from toy corpus to realistic retrieval; no explicit parameter-mapping table |
| 8. Math glossed | PASS | RRF formula and BM25 formula followed by prose |
| 9. Visualisations | PASS | Seaborn heatmaps for score comparisons, bar charts |
| 10. Section banners + arrow prints | PARTIAL | 12/21 banner cells (57%); 2 arrow-print cells (low) |
| 11. Completed roadmap + key insights | PASS | "## Summary: The Complete Hybrid Search Journey" with completed table |
| 12. Deterministic seeds | PASS | Seeds set for reproducible embedding comparisons |

**Top 2 gaps:**

1. **Thin pedagogical print density.** Only 2 out of 21 code cells contain arrow-print statements that state the takeaway in words. The benchmark comparisons print raw score tables but do not cap them with an explicit `print("  → RRF beats pure BM25 on semantic queries because...")` conclusion. A reader who does not plot the charts still needs the key sentence. Add arrow prints to at least the 5 comparison/measurement cells.

2. **Reflection cells are almost absent — only 1 for a 46-cell notebook.** There is one "What just happened" cell but the notebook has six distinct conceptual transitions (BM25 failure, semantic failure, RRF, weighted fusion, reranking, production considerations). Each transition should end with a short reflection that names the gap the next Part will address. Add at minimum 3 more reflection cells at Part boundaries.

---

### 6. `04-llm/llm-gateway.ipynb`

**Score: Good**

**Structure:** 56 cells (30 markdown, 26 code). Strong on predict-first (5 cells) and reflection (6 cells). The running example is a simulated multi-provider gateway built step by step — routing, load balancing, rate limiting, fallback, caching. Opens with a clear title and "Why a Single API Call Is Never the Whole Story" hook. Has a proper completed summary at the end.

**Checklist assessment:**

| Item | Status | Notes |
|---|---|---|
| 1. Title + roadmap table | PASS | Title, pitch paragraph, and roadmap section present |
| 2. Single running example | PASS | Mock provider simulation threaded through every stage |
| 3. Claims proved with code | PASS | Latency, fallback, cache-hit rate all demonstrated with numbers |
| 4. Predict-first cells | PASS | 5 predict cells |
| 5. Your turn exercises | FAIL | No `🧪` exercise cells |
| 6. Reflection cells | PASS | 6 reflection cells |
| 7. Toy → real bridge | FAIL | Entire notebook is simulation-only; no explicit bridge to a real SDK (LiteLLM, OpenAI, Anthropic) with parameter-mapping table |
| 8. Math glossed | PASS | EMA, token-bucket formulas followed by prose |
| 9. Visualisations | PASS | Time-series plots of latency, cost, cache hit rate |
| 10. Section banners + arrow prints | PARTIAL | Only 5 of 26 code cells (19%) have `# ──` banners; 3 arrow-print cells |
| 11. Completed roadmap + key insights | PASS | "## Summary: The LLM Gateway Mental Model" with completed table |
| 12. Deterministic seeds | PASS | Seeds set for reproducible simulation runs |

**Top 2 gaps:**

1. **Section banners cover only 19% of code cells.** 21 of 26 code cells lack the `# ──` banner. The guide treats this as mandatory for every code cell. A mechanical pass through every unmarked code cell to add a one-line banner would bring this to compliance. Priority cells are the simulation loop bodies, the rate-limiter implementation, and the cache look-up logic, which are currently hard to locate when skimming.

2. **No toy-to-real bridge.** The entire notebook uses simulated providers (`MockProvider` classes). There is no cell that shows the same routing/fallback logic running against a real SDK (even a minimal LiteLLM or `openai` client example), and no parameter-mapping table comparing the simulation's knobs to production equivalents. A reader finishes knowing the concept but cannot directly transfer it to real code. Add a "Bridge to Production" Part (or a minimal appendix cell) that maps `MockProvider(latency=0.2)` → `litellm.completion(model="gpt-4o", ...)` and notes which gateway parameters correspond to which SDK settings.

---

### 7. `04-llm/rag-evaluation.ipynb`

**Score: Good**

**Structure:** 38 cells (17 markdown, 21 code). The most structurally disciplined notebook outside the gold standard: every one of 21 code cells has a `# ──` section-banner comment, 10 cells have arrow-print statements, and 6 reflection cells close the major Parts. The narrative is failure-mode driven — each metric is introduced only after the previous one is shown to be inadequate on a concrete example. Closes with a full roadmap summary.

**Checklist assessment:**

| Item | Status | Notes |
|---|---|---|
| 1. Title + roadmap table | PASS | "RAG Evaluation: Measuring What Your Pipeline Actually Gets Wrong" with table |
| 2. Single running example | PASS | Fixed document corpus and query set threaded through all metric comparisons |
| 3. Claims proved with code | PASS | Every metric failure proved on a concrete failing example before replacement is introduced |
| 4. Predict-first cells | FAIL | Zero predict-first cells — the largest single gap |
| 5. Your turn exercises | FAIL | No `🧪` exercise cells |
| 6. Reflection cells | PASS | 6 reflection cells |
| 7. Toy → real bridge | FAIL | All metrics implemented from scratch; no bridge to production tools (RAGAS, DeepEval, TruLens) |
| 8. Math glossed | PASS | Precision, recall, ROUGE formulas followed by plain-English explanation |
| 9. Visualisations | PASS | Bar charts, confusion-matrix style heatmaps |
| 10. Section banners + arrow prints | PASS | 21/21 banner cells (100%), 10 arrow-print cells |
| 11. Completed roadmap + key insights | PASS | Full journey table and key insights in summary |
| 12. Deterministic seeds | PASS | Seeds set at top |

**Top 2 gaps:**

1. **Zero predict-first cells.** This is the notebook's sharpest gap against the gold standard. Every metric reveal (precision@k, NDCG, faithfulness score) is shown without first posing a concrete, falsifiable prediction question. Given the failure-mode narrative — where each metric breaks in a surprising way — there is natural opportunity for `🔮 Predict first` cells ("Which of these two retrieval lists has higher precision@3?"). Add at minimum one predict cell before each of the four metric family introductions.

2. **No toy-to-real bridge for production eval tools.** The notebook proves every metric from first principles (which is pedagogically strong) but never shows the same metric computed with a real evaluation library. A reader who uses this notebook as a reference cannot directly adopt RAGAS or DeepEval without separate research. Add a final "Bridge to Production Tools" Part that maps the hand-coded metrics to their counterparts in one production framework, with a parameter-mapping table (hand-coded variable → RAGAS field name).

---

### 8. `05-llm-tuning/llm_finetuning_deep_dive.ipynb`

**Score: Needs Work**

**Structure:** 44 cells (25 markdown, 19 code). Uses distilgpt2 as the running model through continued pretraining, instruction tuning, and DPO stages. Has arrow-print statements (7 cells), a roadmap table, math with explanations, and a toy-to-real reference (distilgpt2 as stand-in for production LLMs). However, it is missing three of the guide's highest-impact patterns entirely: section-banner comments (0 of 19 code cells), predict-first cells (0), and reflection cells (0). The final markdown cell is an "Ablation Study" section, not a completed summary.

**Checklist assessment:**

| Item | Status | Notes |
|---|---|---|
| 1. Title + roadmap table | PASS | Title and roadmap table present at opening |
| 2. Single running example | PASS | distilgpt2 threaded through all three tuning stages |
| 3. Claims proved with code | PARTIAL | Some stages assert behaviour without a comparative measurement (e.g. "DPO improves alignment" without a side-by-side output comparison) |
| 4. Predict-first cells | FAIL | Zero predict-first cells |
| 5. Your turn exercises | FAIL | No `🧪` exercise cells |
| 6. Reflection cells | FAIL | Zero "What just happened" reflection cells |
| 7. Toy → real bridge | PARTIAL | distilgpt2 is positioned as a toy stand-in but no explicit parameter-mapping table to LLaMA-3 or Mistral dimensions |
| 8. Math glossed | PASS | LoRA rank/alpha equations followed by prose |
| 9. Visualisations | PASS | Training loss curves present |
| 10. Section banners + arrow prints | PARTIAL | 0/19 banner cells; 7 arrow-print cells (arrows present but banners entirely absent) |
| 11. Completed roadmap + key insights | FAIL | Final cell is "Ablation Study" content, not a summary; key insights list missing |
| 12. Deterministic seeds | PARTIAL | Seeds set for model init but not consistently before all quoted numbers |

**Top 2 gaps:**

1. **No section-banner comments on any of the 19 code cells.** This is the most mechanically straightforward gap: every code cell should open with `# ── Short Description ─────────────────────────────────────────`. Currently, a reader skimming the source cannot orient themselves — all code cells open directly with `import` statements or model training calls. A single pass to prepend a banner to each cell would fix this entirely.

2. **No predict-first or reflection cells anywhere in a 44-cell notebook.** The three-stage tuning pipeline (continued pretraining → instruction tuning → DPO) has at least three natural predict moments: "What happens to perplexity on domain text before vs. after continued pretraining?", "Can the base model follow instructions without instruction tuning?", "Does DPO change the model's output distribution measurably?" None of these are posed before the reveal. Correspondingly, there are no "What just happened" reflection cells connecting the stages. Add at minimum one predict cell and one reflection cell per tuning stage (six new cells total), and replace the final "Ablation Study" cell (which introduces new content) with a proper `## Summary` cell containing the completed roadmap and a "Key insights to keep" bullet list. The ablation study content can remain as an appendix section after the summary.

---

## Priority Remediation Order

Based on gap severity and leverage:

1. **`03-encoder-decoder/encoder_decoder.ipynb`** — Create from scratch; write `plan.md` first.
2. **`05-llm-tuning/llm_finetuning_deep_dive.ipynb`** — Add banners (mechanical), add predict/reflect cells (structural), fix closing summary.
3. **`04-llm/rag-evaluation.ipynb`** — Add predict-first cells (single largest gap for an otherwise strong notebook).
4. **`04-llm/llm-gateway.ipynb`** — Add banners to 21 unmarked code cells (mechanical), add toy→real bridge.
5. **`01-rnns/MIT/TF_Part1_Intro.ipynb`** — Add reflection cells, add closing summary.
6. **`04-llm/hybrid-search.ipynb`** — Add reflection cells at Part boundaries, increase arrow-print density.
7. **`01-rnns/MIT/PT_Part1_Intro.ipynb`** — Fix opening cell (add proper title+roadmap), add exercises.
8. **`02-transformers/transformers.ipynb`** — Add explicit "What just happened" headings, complete remaining 14 banner cells (gold-standard maintenance).
