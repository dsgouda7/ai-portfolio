# Content Remediation & Expansion Plan — `notes/`

> **Purpose.** Make the `notes/` curriculum a coherent, rigorous, hands-on path from "software engineer who has never trained a model" to "hireable AI/ML engineer." The content already spans the right territory across 10 tracks (~120 chapters), but it lacks **consistent depth, formal rigor, and clean structure**, and it under-serves a cluster of modern **LLM-ops** topics (guardrails, gateways, costing/evaluation, LLM-as-judge, local LLM infrastructure, fine-tuning with runnable code).
>
> **How this plan was produced.** Every subdirectory under `notes/` was audited in parallel (one auditor per track) against five dimensions: inventory/notebook substance, theoretical depth & rigor, structural progression, slop & emoji-removal artifacts, and notebook↔prose alignment. The findings below are concrete and file-specific.
>
> **How to read this plan.** Part 1 is the cross-cutting work that touches every track (do it once, mechanically, everywhere). Part 2 is the new LLM-ops content the curriculum is missing (the owner's explicit priorities). Part 3 is the must-know theory coverage matrix. Part 4 is the per-track fix register. Part 5 is sequencing and acceptance criteria.

---

## Part 0 · Executive Summary

**The good news.** The prose is, track-for-track, genuinely strong. The math is mostly correct and motivated; the failure-first pedagogy is real; the running-example "grand challenge" spine is a good idea and largely executed. This is not a rewrite. It is a **consistency, rigor, and completeness pass** plus a **targeted set of new chapters**.

**The four systemic problems** (each appears in nearly every track):

1. **Emoji-removal scar tissue.** A bulk de-emoji script ran across the repo and left artifacts everywhere: trailing spaces where a checkmark used to be (`(>25% )`, `(<5% )`), empty bold markers (`** **`, `****`), constraint-status tables whose entire meaning lived in stripped 🔴🟡🟢 icons (now blank), `Warning — Warning` doubled callouts, and leading-space bullets where an emoji bullet was removed. **The constraint-progression tables — the pedagogical spine — are now unreadable in several tracks.**

2. **Stale structure / rename fallout.** Tracks were renamed and renumbered (PascalCase → kebab, underscores → hyphens, track-number shifts). Internal links, `authoring-guide.md` fingerprints, `grand-solution.md` references, and cross-chapter bridges were **not** updated. Hundreds of dead links; several `authoring-guide.md` files describe a chapter layout that no longer exists.

3. **Broken/duplicated section numbering and topic careening.** Several chapters jump `§0 → §6A → §1`, duplicate whole section bodies (two `## Part 1`, two `## 10`, re-run §8–§12), or cram four numbered constraints onto one em-dash-glued line. A few chapters drift off their running example entirely.

4. **Notebook ↔ prose gaps.** Some notebooks are empty stubs (all 7 multi-agent solution notebooks), some are single 600-line cells, some teach a *different* topic than the chapter's headline (ch02 agentic safety prose is about guardrails; the notebook only does hallucination), and the highest-value hands-on topics (real fine-tuning, local LLM serving, guardrails enforcement) are concept-only.

**The five content gaps** (the owner's priorities), none currently covered with runnable depth:

- **LLM guardrails** as enforced, runnable code (injection/jailbreak detection, PII, output moderation, structured-output validation).
- **LLM gateways** (multi-provider routing, fallback, rate-limiting, semantic caching, spend caps).
- **LLM costing, model comparison & evaluation metrics** as a first-class subject (benchmark methodology, per-task cost comparison, eval harness).
- **LLM-as-a-judge** technique and full pipeline (pointwise vs pairwise, position/verbosity bias mitigation, G-Eval, calibration).
- **LLM infrastructure** taught hands-on via **local mocks/models** (Ollama / llama.cpp / a mock inference server) that lets a learner stand up an endpoint and *measure* TTFT, throughput, batching, and KV-cache effects — instead of reading asserted numbers.
- **LLM fine-tuning with demonstrable code** on the main path (LoRA/QLoRA actually trained, data prep, before/after eval), not deferred to a GPU-gated supplement.

---

## Part 1 · Cross-Cutting Fixes (apply across all tracks)

These are mechanical and high-leverage. Do each one as a single sweep across the whole repo, not track-by-track.

### 1.1 Emoji-removal artifact sweep (P0)

A scripted sweep + manual review. Patterns to find and fix:

| Pattern | Example (real) | Fix |
|---|---|---|
| Trailing space before `)` / `,` where checkmark stripped | `(>25% )`, `(<$15k )`, `Fits in 24GB ,` | Remove the space; restore an explicit text marker where it carried meaning (`[hit]`, `[partial]`, `[blocked]`) |
| Empty bold / bold with leading space | `** 3.2**`, `** $2.5k**`, `**** ` | Remove the empty wrapper; keep the value |
| Constraint-status table cells that were pure emoji | `\| #3 Cost \| \| \| **** \|` | Re-populate with text status words (`Blocked` / `Partial` / `Hit`) |
| Doubled callout word | `> **Warning — Warning:**`, `> ** Updated:**` | Collapse to one label; pick the correct callout type (`Note`/`Tip`/`Warning`) |
| Leading-space bullet (emoji bullet stripped) | `  **CEO feedback:**`, ` **The N×M Integration Explosion**` | Remove leading space |
| Mermaid node labels with ghost emoji space | `A[" Client Request"]`, `X1[ Block PR]` | Strip the leading space inside the bracket/quote |
| Emoji still present in notebook print strings | `⚠️ WARNING`, `✅ GPU detected`, `✓`/`✗`/`⚠` | Replace with text (`WARNING:`, `GPU detected:`, `[ok]`/`[fail]`) per the repo no-emoji rule |
| Empty symbol legends in authoring guides | `` | `` | Key insight | ``, `- = Target hit` | Replace with the text convention the guides now mandate |

**Acceptance:** `grep` for `% )`, `k )`, `s )`, `** `, `****`, `Warning — Warning`, and the emoji unicode ranges returns zero hits in shipped content. The de-emoji rationale and the text-marker convention are documented once in `notes/authoring-guidelines.md` and every track guide points to it.

### 1.2 Link & path repair sweep (P0)

Rename fallout is pervasive. Normalize **all** internal references to the live layout (kebab-case folders, numbered track prefixes `00`–`08`, topic-named `.md` files — not `README.md`, not PascalCase, not underscores).

- Fix intra-track links: `../ch02_mcp` → `../ch02-mcp`, `[CLIP.md](ch03_clip/clip.md)` → `[clip.md](ch03-clip/clip.md)`.
- Fix cross-track links: `../ml/03_neural_networks` → `../01-ml/03-neural-networks`; `.03-ai/` → `../03-ai/`; `notes/06-ai-infrastructure/` → `notes/07-ai-infrastructure/`.
- Fix every `authoring-guide.md` style-fingerprint `canonical_chapters` / `canonical_examples` block to point at real files.
- Fix `grand-solution.md` references to nonexistent `grand_solution.ipynb` / per-chapter notebooks (create or delete).
- Fix notebook "Next:" links that still use old PascalCase folders.

**Acceptance:** `scripts/check-md-links.py` (already in repo) passes clean across `notes/`; add a CI gate.

### 1.3 Structural normalization (P1)

- **Standardize the main-doc filename.** Pick one rule (topic-named `.md`, e.g. `clip.md`) and apply it. Today ch11/ch14 multimodal, ch06 agentic, and several infra chapters use `README.md` while siblings use topic names; ch05 infra has *both*. Rename and re-link.
- **Fix broken/duplicated section numbering.** Renumber monotonic `§0…§N + Bridge`. Specific offenders are listed per-track in Part 4 (e.g. AI ch04 `§0→6A→1`, AI ch08 `§3.2` after `§4`, agentic ch01 two `## 10` + restart at `## 2`).
- **De-cram `§0` constraint lists.** Re-expand em-dash-glued one-liners (`1. **A**: … — 2. **B**: …`) back to separate numbered lines. Use the clean chapters (advanced-DL ch03/ch04) as the template.
- **Reconcile each track's grand-challenge metrics into one monotonic story.** Several tracks have a metric that oscillates chapter to chapter (agentic conversion 8→18→28→30→28→32; AOV `+$2.10` vs `+$2.50`; infra cost `$7,300/mo` vs `$1,095/mo`; multi-agent throughput `24×`/`50×`/`120×` and `1,000`/`1,200`; advanced-DL ch1 mAP `78.2`/`80.2`/`80.4`). Establish a single source-of-truth metrics table per track in `grand-solution.md` and make every chapter reference it.
- **Re-sync each `authoring-guide.md` and track `README.md` to the real chapter list** (statuses, counts, names). Several claim "5 chapters" / "10 chapters" / "IN PROGRESS" for tracks that are complete and have different chapters.

### 1.4 Notebook substance & alignment pass (P1)

- **Fill empty solution notebooks** — all 7 in `06-multi-agent-ai` are `"source": []`. They ship zero runnable reference implementations.
- **Re-cell single-cell notebooks** into interleaved markdown/code (`08-devops-fundamentals/ch07`, `02-advanced-deep-learning/ch10` is a near-bare code dump).
- **Make notebooks teach the chapter's headline topic.** Where prose covers X but the notebook only does Y, add the missing runnable cells (agentic ch02 guardrails; infra ch05 batching).
- **Remove leftover scratch/cruft files** that shouldn't ship: `ch06-advanced-agentic-patterns/README.backup.md`, `ch03_feature_importance/README-WORKFLOW.md`, `AUTHORING-GUIDE-COMPARISON.md`, `ch00b-class-imbalance/IMPLEMENTATION-NOTES.md`, empty `notes/03-ai/archived/deprecated-chapters/`.
- **Fix the broken hint-generation output** (`Compute system_prompt using 99()`, `using items()`) and the dead `scripts/generate-notebooks.py` that targets a nonexistent path layout.

### 1.5 Animation/asset reconciliation (P2)

- Math track ships animation GIFs that are never embedded (filename underscore/hyphen mismatch) and `## Animation` placeholders injected mid-`§0`. Either embed the existing assets (fix filenames) or remove the placeholders.
- Fix the copy-paste Mermaid "progress check" bug in all 7 math chapters (every chapter marks all C1–C7 green; should mark only completed/current/upcoming).
- Add the missing "What you're seeing" captions to generic animation embeds (devops ch05/ch06/ch07).
- Generate or remove the missing interview-guide hero image (`img/AI Interview Primer.png`).

---

## Part 2 · New LLM-Ops Content (the owner's priorities)

This is the substantive expansion. These topics are currently absent or concept-only. Each gets **prose with formal rigor + a runnable notebook that works on a stock laptop (local mocks/models, no paid API required)**. The natural home is the `03-ai` and `05-agentic-ai` tracks, plus a new local-LLM-infra lab in `07-ai-infrastructure`.

### 2.1 LLM serving & infrastructure — hands-on with local mocks/models (NEW)

**Home:** `07-ai-infrastructure` — add a hands-on lab chapter (e.g. `ch12-local-llm-serving-lab`) and a local fallback path through ch05/ch06/ch11.

The track currently teaches infra by **assertion and Azure TODO-stubs**. There is no runnable local endpoint, no measurement. Build:

1. **Local model quick-start.** `ollama run llama3` (or `llama.cpp`, or a tiny HF model on CPU) exposed behind an OpenAI-compatible API — the same interface the ch11 vLLM Dockerfile presents, so the mental model transfers.
2. **A mock inference server** (FastAPI/uvicorn) that emits synthetic tokens with **configurable per-token latency and a fixed prefill cost**, so the learner can *measure* rather than read: TTFT vs TPOT, the effect of batch size, KV-cache reuse on a shared prefix, and a queue under Poisson arrivals.
3. **A load-test notebook** (`requests`/`httpx`/`asyncio`) that produces the latency/throughput numbers the track currently asserts (`12,000 req/day`, `1.2s p95`), turning narrative constants into produced evidence.
4. **An observable continuous-batching / PagedAttention demo** against the mock or a local vLLM, so ch05's headline concepts are actually run.
5. A clear **local ↔ cloud bridge**: the same OpenAI-compatible contract works for `ollama` locally and the gated `vllm/vllm-openai` container in ch11; document the swap.

**Why mocks.** A mock server gives *deterministic, laptop-runnable* control over the exact mechanics (prefill cost, per-token latency, batch coalescing, cache hits) that real GPUs hide behind noise — ideal for building the mental model, then validated against a real local model where possible.

### 2.2 LLM guardrails — enforced, runnable (NEW depth)

**Home:** `05-agentic-ai/ch02-safety-and-hallucination` (prose is already deep; the notebook is hallucination-only). Add runnable cells; consider splitting guardrails into its own chapter if it outgrows ch02.

Cover with code:
- **Input guardrails:** prompt-injection / jailbreak detection (a small classifier or rules + an LLM-judge check), topical/off-scope rejection, PII detection (regex + Presidio).
- **Output guardrails:** content moderation (local model or a mock of Azure AI Content Safety / Llama Guard), structured-output validation (Pydantic / JSON-schema / Guardrails-AI-style retries), groundedness/citation checks.
- **Framework view:** NeMo Guardrails / Guardrails AI taxonomy (input vs output vs dialog rails) — at least one runnable example, the rest as a comparison table.
- **Red-team harness:** a small attack set (DAN, many-shot, indirect injection via tool output) run against the guardrail stack with pass/fail metrics.

### 2.3 LLM gateways (NEW)

**Home:** `05-agentic-ai/ch04-cost-and-latency` (LiteLLM is name-dropped only) or a dedicated short chapter.

Cover with code (against local models + mocks so it runs offline):
- **Multi-provider routing** and a **fallback chain** (primary → cheaper → local) with a LiteLLM-style router.
- **Rate-limiting / spend caps / key management** concepts with a working limiter.
- **Semantic caching** (embed the request, serve a cached response on a similarity hit) — currently one prose paragraph, no code.
- **Observability:** per-request cost/latency/provider logging that feeds the costing chapter.

### 2.4 LLM costing, model comparison & evaluation metrics (NEW depth)

**Home:** `05-agentic-ai/ch03-evaluating-ai-systems` (eval is the strongest existing chapter — extend it) plus the costing in ch04.

- **Cost model:** token accounting (tiktoken), per-task cost across model tiers, caching savings (some of this exists in ch04 — consolidate and make it the canonical reference).
- **Model comparison methodology:** standard benchmarks and what they actually measure (MMLU, MT-Bench, Arena Elo, HELM, GSM8K, HumanEval), their limitations, and how to run a **task-specific** comparison harness on your own data.
- **Evaluation metrics:** retrieval (P@k/R@k/MRR/nDCG — partly present), generation (RAGAS faithfulness/relevance, BLEU/ROUGE/BERTScore caveats), and **why classic metrics fail for open-ended generation** (motivating 2.5).

### 2.5 LLM-as-a-judge — technique + full pipeline (NEW)

**Home:** `05-agentic-ai/ch03-evaluating-ai-systems` already has a pairwise judge with a position-bias swap in the notebook — promote it to a first-class, fully-developed topic.

- **Pointwise vs pairwise vs reference-based** judging; rubric design.
- **Known biases and mitigations:** position bias (swap + average), verbosity bias, self-preference, formatting bias; **G-Eval** (chain-of-thought scoring) and judge **calibration** against human labels (agreement / Cohen's κ).
- **A full pipeline:** dataset → judge prompt → batched scoring (local model via Ollama) → aggregation → regression-threshold gate in CI. Make the whole thing runnable offline.

### 2.6 LLM fine-tuning with demonstrable code (PROMOTE to main path)

**Home:** `05-agentic-ai/ch05-fine-tuning` — the GPU supplement *does* really train (TinyLlama + `get_peft_model` + `SFTTrainer.train()`), but the **main** notebook only defines `LoraConfig` and prints "To train: …". Also `03-ai/ch03-llm-training-pipeline` covers SFT/RLHF/DPO/LoRA conceptually and explicitly defers implementation.

- Make a **runnable end-to-end fine-tune** the centerpiece: data prep (instruction format) → LoRA/QLoRA on a small model that fits a laptop/free-tier GPU → before/after eval (tie to 2.4/2.5) → save/serve the adapter.
- Keep a CPU-only **smoke path** (tiny model, few steps) so the concept is demonstrable without a GPU, with the full run gated.
- Add the **fine-tune-vs-RAG-vs-prompt** decision framing and **catastrophic forgetting** check.
- Reconcile the two homes: `03-ai` teaches the theory/vocabulary, `05-agentic-ai` owns the runnable implementation; fix the dead pointers between them.

---

## Part 3 · Must-Know Theory Coverage Matrix

The owner's bar: *every piece of theory an AI/ML practitioner must hold at the top of their mind, or know well enough to reach for the right library and know when to use it.* This matrix is the rigor checklist. `OK` = covered with rigor; `Thin` = present but under-developed; `GAP` = missing/add.

### Foundations (`00-math-under-the-hood`)
| Topic | Status | Action |
|---|---|---|
| Vectors, dot product, norms (L1/L2/L∞/Lp) | Thin | Formalize norm family (relevant to Lasso, gradient clipping) |
| Matrix multiply, shapes, normal equations | OK | — |
| Derivatives, chain rule, backprop intuition | OK | — |
| Gradient descent + step-size/convergence | OK | — |
| Jacobian / Hessian | OK | — |
| **Eigenvalues / eigen-decomposition** | **GAP** | Add to ch05 (only Hessian eigenvalues name-dropped) |
| **SVD** | **GAP** | Add to ch05 (foundational for PCA/low-rank/conditioning) |
| Convexity (formal) | Thin | Formalize in ch04 (used informally) |
| Probability, expectation/variance, CLT, MLE | OK | — |
| MAP (prior × likelihood) | Thin | Short derivation in ch07 |
| **Entropy / cross-entropy derivation / KL divergence** | **GAP** | Add to ch07 — single biggest theory hole; underpins every classifier loss, RLHF, distillation |
| Distribution zoo (Poisson/Exponential/Beta/Dirichlet) | Thin | Brief treatment beyond Bernoulli/Gaussian |

### Classical & deep ML (`01-ml`)
| Topic | Status |
|---|---|
| Bias-variance, learning curves, CV | OK |
| Regularization (L1/L2 math) | OK |
| Loss functions, MLE view | OK |
| Optimizers (SGD/momentum/RMSProp/Adam/AdamW) | OK |
| Precision/recall/ROC/AUC/F1/PR | OK |
| Class imbalance, threshold tuning | OK (but broken cross-pointers) |
| **Calibration (Platt/isotonic/reliability/Brier)** | **GAP** | Promised but only scattered one-liners; add a dedicated treatment |
| Feature scaling, feature importance, leakage | OK |
| Backprop derivation | OK |
| CNNs / RNNs / LSTMs | OK |
| Attention & Transformers (from scratch) | OK |
| Embeddings | OK |
| Generative (AE/VAE/GAN) | Thin | Normalize notebooks + section order; diffusion lives in `04-multimodal-ai` |

### LLMs (`03-ai`)
| Topic | Status |
|---|---|
| Tokenization / BPE | OK (add a hands-on BPE-merge notebook cell) |
| Attention math, multi-head, positional enc / RoPE | OK |
| KV cache, sampling (temp/top-p/top-k), decoding | OK |
| Pretrain / SFT / RLHF / DPO | OK (concept); implementation → `05-agentic-ai` |
| **Scaling laws (Chinchilla compute-optimal)** | Thin | Add the token/param ratio treatment |
| Embeddings, contrastive/InfoNCE | OK |
| RAG: chunking/retrieval/**reranking** | OK (pull cross-encoder rerank into main prose) |
| Vector indexes (HNSW/IVF/PQ/DiskANN) | OK |
| **Guardrails / gateways / eval / judge / FT code / local serving** | **GAP** | Part 2 |

### Specializations
| Track | Status |
|---|---|
| `04-multimodal-ai` (ViT, CLIP, diffusion, CFG, latent, MLLM, audio) | OK prose; DDIM/DPM-Solver derivation Thin; ch14 orphaned; tables gutted |
| `02-advanced-deep-learning` (ResNet, detection, segmentation, distillation, pruning, **interpretability**) | OK except **ch11 interpretability is a 115-line stub with no notebooks and no Grad-CAM math** |
| `05-agentic-ai` (ReAct, safety, eval, cost, FT, advanced patterns) | OK prose; notebook gaps per Part 2 |
| `06-multi-agent-ai` (messages, MCP, A2A, events, memory, trust, frameworks) | OK prose; **CrewAI named-not-taught**; long-term/semantic agent memory Thin; empty notebooks |
| `07-ai-infrastructure` (GPU, budgets, quant, parallelism, inference, serving, networking, stores, tracking, monitoring, deploy) | OK prose; **no hands-on local serving (Part 2.1)**; numbers asserted not measured; ch08–ch10 narrative drift |
| `08-devops-fundamentals` (Docker, K8s, CI/CD, observability, IaC, networking, secrets) | Content strong; structural/notebook fixes only |

---

## Part 4 · Per-Track Fix Register

Each track lists its specific defects in priority order. Cross-cutting items (Part 1) are not repeated here unless a track-specific instance needs calling out.

### `00-math-under-the-hood`
- **P0 — Notebooks don't exist.** The track is prose-only, yet README/guide/exercises repeatedly promise notebooks and widgets. Decide: author one `notebook.ipynb` per chapter (preferred — it's the foundation track) or strip every notebook/widget reference.
- **P0 — Correctness errors (verified by hand):** ch02 §4.1 worked-example data and §9 progress numbers are wrong; ch03 §3.2.1 `t=0.331` row (`1.08` should be `≈1.61`); ch03 §3.4 velocity `±7.20` should be `±6.5`; ch05 §3.7.1 worked example is internally inconsistent and the "matches perfectly" claim is false. Reconcile the cross-chapter `v₀` constant.
- **P0 — Mermaid progress bug** in all 7 chapters (all nodes green).
- **P1 — ch07 "Code Skeleton" is ch06's gradient/chain-rule content mispasted** into the probability chapter — replace or remove.
- **P2 — Depth:** add eigen-decomposition + SVD (ch05); entropy/cross-entropy/KL + MAP (ch07); formal convexity (ch04); norm family.
- **P2 — Embed the existing animation GIFs** (fix filename mismatch); remove `## Animation` placeholders splitting `§0`.
- **P3 — Artifacts:** leading-space bold in README L25–27, authoring-guide L7/L79–81, grand-solution L11/L16/L18; empty-string print branch; empty callout-symbol table; `00-math_under_the_hood`/`grand_solution` naming drift; `## 9 · The Challenge` typo. Trim repeated "Priority: Intuition over calculation" boilerplate and heavy Phase/DECISION scaffolding in ch04/ch06.

### `01-ml`
- **P0 — Two duplicated chapter bodies:** `08-ensemble-methods/ch01_ensembles` §11–§20 is an entire misplaced SVM/SmartVal chapter (delete ~L272–705); `05-anomaly-detection/ch03_autoencoders` re-runs §7.5 + §13–§17 after the Bridge (delete ~L795–1018).
- **P1 — Calibration gap + broken pointers:** `02-classification/ch03_metrics` forward-points to nonexistent "Anomaly Ch.4 (Calibration)" and "Ch.3 (Threshold Optimization)". Add a calibration treatment (Platt/isotonic/reliability/Brier) and repoint.
- **P2 — `09-generative-models`** breaks the exercise/solution convention (single `notebook.ipynb`), compressed section order, "Track Status: partial" — normalize or document.
- **P3 — Legacy flat format** in `00-data-acquisition` and `10-feature-engineering` (no chNN folder, missing numbered Animation/Progress/Bridge); remove repo-cruft files (`README-WORKFLOW.md`, `AUTHORING-GUIDE-COMPARISON.md`, `IMPLEMENTATION-NOTES.md`); fix the `()` checkmark artifact in the track authoring guide.

### `03-ai`
- **P0 — `ai-primer.md` is duplicated/broken:** two `## Part 1` and two `## Part 2`; the second half is the agentic PizzaBot primer with dead links to a removed layout. Rewrite to the real ch00–ch08.
- **P0 — `authoring-guide.md` is stale** (documents a 5-chapter structure; track has 9). Re-sync.
- **P1 — Section numbering:** ch04 `§0→6A…6G→§7→"§1 Key Distinctions"` (renumber monotonic); ch08 `§3.2`/`§3.3` appear after `§4` (reorder).
- **P1 — Template non-conformance:** add the 3-part opening blockquote to ch05/ch07/ch08; add the missing §Bridge + §Key-Distinctions to ch07 (ends at "Summary").
- **P2 — Depth:** Chinchilla scaling laws (ch03 §4); pull cross-encoder reranking into ch07 main prose; add a hands-on BPE notebook cell (ch01).
- **P2 — Scope handoff note:** add a correctly-linked "what this track defers to 05/07" pointer (guardrails, judge, FT code, gateways, local serving) — currently only reachable via dead links.
- **P3 — README status flags** ("IN PROGRESS" for complete chapters); remove empty `archived/deprecated-chapters/`; fix the `06-ai-infrastructure`→`07-` path in ch07 supplement notebooks.

### `05-agentic-ai`
- **P0 — Stale cross-chapter references** from a prior 10-chapter "LLMOps" layout: ch01 "Before (Ch.5)/After (Ch.6)/Ch.8-10", ch02 "Ch.8", ch03 "Bridge to Chapter 8"/"Ch.7", ch05 "Bridge to Ch.9". Renumber to the 6-chapter scheme.
- **P0 — Emoji-removal artifacts in every `§0` and Progress-Check table** (`(>25% )`, `()`), and the four-constraints-on-one-line cram. Fix track-wide.
- **P1 — Metric reconciliation:** conversion/AOV oscillate across README + all six chapters; pick one monotonic story; fix `+$2.10` vs `+$2.50`.
- **P1 — New runnable content (Part 2):** guardrails cells in ch02; LLM-gateway + semantic-cache cell in ch04; make ch05 main notebook actually train (or route clearly to the GPU supplement); promote LLM-as-judge in ch03.
- **P2 — Structure:** ch01 two `## 10` + restart at `## 2 · Progress Check`; ch02 duplicate `## Interview Checklist` and reversed Bridge/Progress order; ch05 `§1.5` spans >1,000 lines.
- **P2 — Cruft & naming:** delete `ch06/README.backup.md`; rename ch06 `README.md` → `advanced-agentic-patterns.md`.
- **P3 — Notebook hygiene:** fix CamelCase "Next:" links; emojis in ch02/ch03 notebook print strings; garbled exercise hints (`using 99()`, `using items()`).

### `06-multi-agent-ai`
- **P0 — All 7 solution notebooks are empty** (`"source": []`). Populate them — the track ships no runnable reference implementation.
- **P0 — ~32 broken inter-chapter links** (underscore paths vs hyphen folders) + cross-track `../ai/...` links.
- **P1 — Meta files broken:** `authoring-guide.md` has doubled `## The Plan`, triple/contradictory `§0` templates, a corrupted progression table, and stale `04-multi_agent_ai`/`MultiAgentAI` paths; `grand-solution.md` links a nonexistent notebook; `README.md` cites a wrong script path.
- **P1 — Numeric contradictions:** throughput `24×`/`50×`/`120×` and final `1,000` vs `1,200`; baseline `10` vs `24`. Reconcile.
- **P2 — Depth:** CrewAI is named (README/interview/ch5/ch7 stories) but never taught — add to ch07; add an explicit MCP-inside-A2A composition walkthrough; add long-term/semantic agent memory to ch05.
- **P2 — De-duplicate ch07 exercise notebook** (7 identical TODO blocks); fix or delete dead `scripts/generate-notebooks.py`.
- **P3 — Artifacts:** trailing-space-in-bold (ch02 L51–53, ch06 L1324); leading-space callout headers (ch02/ch04/ch06); `** Updated:**`.

### `07-ai-infrastructure`
- **P0 — Planning docs out of sync:** README says "COMPLETE — 10 chapters" but 11 exist and ch11 is omitted; "Chapters 6–8 planned" is stale; `authoring-guide.md` arc names the wrong chapters; `grand-solution` vs guide give contradictory final cost (`$7,300/mo` vs `$1,095/mo`); ch08 (feature stores blocking) contradicts grand-solution (not needed). Reconcile.
- **P1 — User priority (Part 2.1): no hands-on local LLM serving.** Add the local model + mock-server + load-test lab; add a local fallback to ch11's gated GPU path; replace/augment the Azure-only TODO-stub supplements with a local serving lab.
- **P1 — Missing main notebooks** promised by READMEs for ch02/ch03/ch04/ch05 (only Azure stubs exist); standardize main-doc filenames (remove ch05 duplicate `README.md`).
- **P2 — Running-example fracture:** ch08/ch09/ch10 drift to BERT/sentiment/e-commerce — return to the InferenceBase Llama-3 doc-extraction story or justify the pivot. Fix ch06 hardware (A100/Llama-2 → canonical RTX 4090/Llama-3); stop citing TGI/TorchServe in a vLLM/ONNX/TensorRT chapter or actually cover them. Fix GiB/GB sloppiness.
- **P2 — Depth:** numbers are asserted, never measured — the local lab (Part 2.1) is also the fix for the track's own "no benchmark without evidence" red line. Add QAT (ch03 is PTQ-only).
- **P3 — Artifacts:** empty `()` status cells (ch02 L518–519, authoring-guide L961, ch11 table); trailing spaces (ch05 L20/L198, grand-solution L692, ch02 L953, ch03); mermaid node-label ghost spaces; orphaned "Animation placeholder" GIF refs.

### `04-multimodal-ai`
- **P0 — Constraint-progression tables gutted by emoji removal** across ch01–ch13 (status column was pure emoji; now blank/`****`). Restore text status words — the tables are currently meaningless.
- **P0 — All meta-file links broken:** README/grand-solution/authoring-guide use underscore folders + `README.md` names + wrong track number (`05-multimodal_ai`); chapter cross-links use `../ml/...`/`.03-ai/`. Rewrite to live paths.
- **P1 — ch14 is an orphan:** different narrative (in-car EV), no `§0` constraint table, no notebooks, references "Ch.15 planned". Reframe to VisualForge + add notebook, or mark as a standalone appendix. Reconcile chapter count (docs say 12/13, there are 14).
- **P1 — README chapter numbering scrambled** (display-# maps to wrong folders for rows 5/6/7); off-by-one bridges (ch12→itself should be ch13; ch11→ch11 should be ch12); "3 vs 4 modalities" inconsistency.
- **P2 — Emojis still in 9 supplement-solution GPU-check cells** (`⚠️`/`✅`); naming inconsistency (ch11 `README.md`).
- **P2 — Depth:** deepen DDIM/DPM-Solver derivation (ch05) to match ch04's rigor.
- **P3 — Authoring-guide self-inflicted artifacts** (`## 12/13 ·` example headers, stripped legend symbols, stale fingerprint, orphaned `# QA check` line); trailing/leading paren spaces (ch05 L66–68, ch07 L459).

### `02-advanced-deep-learning`
- **P0 — ch11 interpretability is a 115-line stub:** written in Keras prose while the track is PyTorch, no notebooks, no Grad-CAM/saliency/integrated-gradients math, ignores the template. Rewrite as a full chapter with notebooks.
- **P0 — ch10 published AI "thinking out loud":** literal `"Wait, that's too aggressive. Let me recalculate:"` / `"Actually, let's use..."` strings in §1.1/§3.1; and an 80%-sparsity claim whose arithmetic yields ~31%. Delete the slop, fix the math.
- **P1 — Broken links** in README/grand-solution (underscore paths; nonexistent `grand_solution*.ipynb`); duplicate `## Prerequisites`; empty `## Track Position`.
- **P1 — `§0` constraint-cram** in ch01/02/05–10 (use ch03/04 as template); `Warning — Warning` in ch03/04 §6; broken nested-`\text` LaTeX in ch05 §1.
- **P2 — Metric reconciliation** (ch1 mAP, ch4 latency); add markdown narrative to the ch10 code-dump notebook; strip trailing-space residue (ch02 table, ch03, ch06, ch10).

### `08-devops-fundamentals`
- **P1 — ch07 notebooks are single 600-line cells** — re-cell into interleaved markdown/code like the other chapters.
- **P1 — README duplicate Ch.8 entry** + underscore→hyphen link fixes; authoring guide stale `07-devops_fundamentals`/"Track 7" labels.
- **P2 — Constraint-framework inconsistency:** track defines 5 canonical constraints but each chapter `§0` invents its own (ch04 has only 3). Align or explicitly scope.
- **P2 — `Warning — Warning` doubled callouts** (ch01–06, ch08 §1.5; ch07 dropped it — inconsistent); mermaid `[ Block PR]` ghost spaces; emoji residue inside ch08/ch04 `echo`/`raise`/`print` strings; generic animation captions (ch05/06/07); ch04 stray double `---`.

### `interview-guides`
- **P1 — New core-AI/LLM guide** (`ai.md`): house the absent priorities — LLM-as-a-judge, gateways, fine-tuning decision tree, guardrails taxonomy, model-comparison/eval benchmarks. Largest gap.
- **P1 — Missing track guides:** add (or explicitly scope out) guides for `00-math`, `02-advanced-deep-learning`, `08-devops`.
- **P2 — Broken refs:** `interview-guide.md` Next-Steps PascalCase paths; `agentic-ai.md` hero image (run the gen-script or remove); notebook `AgenticAID.md` references; cross-track `../ai/...` links; fingerprint `canonical_examples` paths.
- **P2 — Consistency:** `interview-guide.md` is the lone off-template/off-naming file (rename to `ml.md`, adopt the standard template); `multimodal-ai.md` heading-level drift + missing fingerprint block.
- **P3 — Authoring-guide duplications** ("One wry sentence" block, "Mathematical Style" section, signal-words tables) and emoji-artifact legends.

---

## Part 5 · Sequencing & Acceptance

### Phasing

**Phase A — Mechanical hygiene (unblocks everything, low risk).** Part 1.1 (emoji sweep), 1.2 (links), 1.4 cruft removal, 1.5 assets. Add CI gates: link checker + an artifact linter (the grep patterns in 1.1) + notebook-non-empty check. *Outcome: nothing is broken or scarred; the existing strong prose reads clean.*

**Phase B — Structural correctness (per-track, mechanical-ish).** Part 1.3 (numbering, de-cram, filename standard, metric reconciliation, guide re-sync) and the P0 correctness fixes in Part 4 (math errors, duplicated bodies, ch11 stub, ch10 slop). *Outcome: every chapter has monotonic structure, one metric story, and no factual contradictions.*

**Phase C — Notebook substance.** Fill empty notebooks (multi-agent), re-cell single-cell notebooks (devops ch07, advanced-DL ch10), align notebook↔prose. *Outcome: every chapter ships a runnable, substantive reference notebook.*

**Phase D — New LLM-ops content (the priorities).** Part 2 in order: 2.1 local-serving lab (also fixes infra's "asserted not measured" problem), then 2.2 guardrails, 2.6 fine-tuning, 2.4/2.5 costing+eval+judge, 2.3 gateways. Then Part 3 depth additions (math eigen/SVD/KL, calibration, scaling laws, interpretability) and the interview-guide core-AI guide. *Outcome: the curriculum covers the modern LLM-ops surface area hands-on.*

### Acceptance criteria (definition of done)

A chapter is "done" when:
1. **No artifacts:** passes the emoji/artifact linter and the link checker; no emojis in prose or notebook strings; constraint tables use explicit text status.
2. **Structure:** monotonic `§0…§N + Bridge`; opening 3-part blockquote present; follows the track `authoring-guide.md` (which itself must match reality); one metric story consistent with `grand-solution.md`.
3. **Rigor:** every formula has a verbal gloss within three lines; every must-know topic in its Part 3 row is `OK`; no AI "thinking out loud" or filler.
4. **Hands-on:** a substantive, runnable notebook whose code matches the prose; runs on a stock laptop (local mock/model, no paid API); GPU paths have a CPU/local fallback or a clean GPU guard.
5. **Navigable:** all internal and cross-track links resolve.

### What this plan deliberately does **not** do
- It does not rewrite the strong existing prose — it removes scar tissue, fixes errors, and fills gaps.
- It does not add net-new tracks beyond the LLM-ops content above; the track structure is sound.
- It keeps the local-first, no-paid-subscription constraint: all new labs use Ollama/llama.cpp/mocks so any learner can run them.

---

## Appendix · Issue Tally by Track (triage view)

| Track | P0 items | Headline problem | New content? |
|---|---|---|---|
| 00-math | 3 | No notebooks exist; hand-verified numeric errors | Notebooks + eigen/SVD/KL |
| 01-ml | 1 | Two duplicated chapter bodies | Calibration |
| 03-ai | 2 | `ai-primer.md` + `authoring-guide.md` stale/duplicated | Scaling laws, BPE lab; scope-handoff |
| 04-multimodal | 2 | Gutted constraint tables; all meta links broken | ch14 fix; DDIM depth |
| 02-advanced-DL | 2 | ch11 stub; ch10 AI-slop + bad math | ch11 interpretability |
| 05-agentic | 2 | Stale 10-ch numbering; `§0` artifacts | Guardrails/gateway/judge/FT code |
| 06-multi-agent | 2 | All 7 solution notebooks empty; ~32 dead links | CrewAI; semantic memory |
| 07-infra | 1 | Planning docs vs 11 real chapters; asserted-not-measured | **Local LLM serving lab** |
| 08-devops | 0 | ch07 single-cell notebooks; constraint drift | — |
| interview-guides | 0 | No core-AI/LLM guide; 3 tracks lack guides | Core-AI guide + 3 track guides |
