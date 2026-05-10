# AI Track Rewrite — Parallelized Implementation Plan

## Objective

Rewrite the `03a-ai` track from scenario-driven framing to a **historical walkthrough**: each
chapter covers one era's problem and the invention that solved it. Concepts appear in causal
order — every new idea is a direct response to a limitation of the previous one.

Notebooks are **behavioral only**: same input, different configuration, observe output divergence.
No training loops, no gradient computation, no large downloads.

---

## Track Structure

| Chapter | Historical arc |
|---|---|
| ch01 — LLM Fundamentals | RNNs fail → attention → transformer → GPT/BERT fork → scale → emergent capabilities → alignment |
| ch02 — Prompt Engineering | Base model ≠ instruct model → prompt as behavioral contract → patterns → injection defense |
| ch03 — CoT Reasoning | Single-pass failures → step-by-step → self-consistency → trained reasoning (o1/R1) |
| ch04 — RAG & Embeddings | Hallucination problem → retrieval grounding → encoder role → RAG pipeline → HyDE |
| ch05 — Vector DBs | Naive search doesn't scale → ANN → IVF/HNSW → production index selection |

---

## Execution Model

Three sequential phases, each fully parallelized. Phase N+1 starts only after Phase N is complete.
All agents are independent — no cross-chapter reads, no shared state.

---

## Phase 1 — MD Rewrites (5 agents, parallel)

Each agent reads the existing chapter `.md`, rewrites with the historical arc, writes back.

### Agent brief template

```
Read:  notes/03a-ai/ch{N}-{name}/{name}.md
Write: same file

Arc: [chapter-specific causal chain, see below]

Rules:
- Historical order = causal order (problem → invention → limitation → next invention)
- No scenario/fictional business frame. Concepts stand alone.
- Keep: callout box style, section headers, interview table at end
- Replace DPO full formula + symbol table with 3-sentence plain-English + link to ch10
- Replace entire PEFT deep-dive (LoRA/prefix/prompt tuning math) with:
    > "Three PEFT methods matter here — LoRA, prefix tuning, prompt tuning.
    > Full derivations and tradeoffs in [Ch.10](../ch10-fine-tuning/fine-tuning.md)
    > when you actually run one."
- Do NOT read other chapter files
```

### Ch01 arc brief

RNNs: sequential processing, can't parallelize, vanishing gradient on long sequences →
Bahdanau attention (2014): score every input position against every other, weighted sum →
"Attention Is All You Need" (Vaswani 2017): drop recurrence entirely, pure attention, trains in
parallel → decoder fork (GPT-1 2018): causal attention, predicts next token, generation →
encoder fork (BERT 2018): bidirectional attention, richer representations, ideal for retrieval,
cannot generate → GPT-3 (2020): scale reveals in-context learning nobody designed for →
emergent capabilities as a class of phenomenon → InstructGPT (2022): SFT teaches format,
RLHF aligns to preference → DPO as the simpler replacement → tokenization + context window
as operational facts (cost estimation, lost-in-middle).

### Ch02 arc brief

Base model behavior (text continuation, not instruction following) vs instruct model as the
entry point → the system prompt as a behavioral contract, not a suggestion → zero-shot vs
few-shot as a precision dial → structured output: why natural language fails for machine
consumption and how to force parseable output → prompt injection as the adversarial surface
(user input overriding system intent) → defense layers → templates and versioning as production
hygiene (prompts are code).

### Ch03 arc brief

Models fail on multi-step problems because they compress intermediate steps into one forward
pass → CoT (Wei et al. 2022): "think step by step" forces visible intermediate reasoning,
accuracy improves significantly → self-consistency (Wang et al. 2022): one chain can still be
wrong, run N independent chains at T>0, take majority vote → Tree of Thoughts: branching
exploration for open-ended search, exponential cost → PRM vs ORM: scoring each step vs
scoring the final answer, why dense reward signal matters → o1/RLVR: trained reasoning via
verifiable rewards, test-time compute as a new scaling axis → failure modes: unfaithful
reasoning, sycophancy, overthinking, hallucinated tool results.

### Ch04 arc brief

LLMs hallucinate because the pretraining corpus is frozen and private data was never in it →
retrieval as the grounding fix: don't train the knowledge in, look it up at query time →
embeddings: text → fixed vector where semantic distance is measurable → encoder models
(BERT-family: bidirectional, no generation, ideal for embeddings) vs decoder models (GPT:
causal, generation, weaker for similarity) → contrastive training as what makes embeddings
useful → the RAG pipeline: chunk → embed → store (offline); embed query → ANN search → augment
prompt → generate (runtime) → HyDE: embed a hypothetical answer instead of the raw question,
closes the phrasing gap → failure modes: semantic gap, chunk size, lost-in-middle, unfaithful
generation.

### Ch05 arc brief

Exact nearest-neighbor search: O(N×d) per query, fails at 10M+ vectors, 100M vectors needs
307 GB RAM → kd-trees degrade in high dimensions (curse of dimensionality) → ANN: trade
tiny recall for massive speed → distance metrics (L2 / cosine / dot product, why they agree on
normalized vectors) → IVF: K-means partition, search only the closest nprobe clusters at query
time, tune nprobe for recall vs speed → HNSW: graph-based greedy descent across layers,
dominant in production (ChromaDB, Pinecone, Qdrant), tune M and efSearch → DiskANN: HNSW
graph on SSD for billion-scale → PQ compression: split vectors into sub-vectors, quantize each,
32× compression → index selection guide based on corpus size and constraints.

---

## Phase 2 — Notebook Implementations (5 agents, parallel)

Each agent reads the first 60 lines of the Phase 1 `.md` for section context, reads the existing
`notebook-exercise.ipynb`, and replaces all TODO cells with behavioral experiments.

### Behavioral experiment rules

- Every code cell produces visible output that contrasts with an adjacent comparison
- Same input + different config = different observable output = the lesson
- Ollama + phi3:mini for all LLM calls (local, no API key, ~2 GB)
- `tiktoken`, `sentence-transformers`, `chromadb`, `numpy` for non-LLM work
- No training loops, no gradient computation
- Print output explicitly — student must see the divergence without running the next cell

### Ch01 cells

1. **Tokenizer cross-model comparison**: encode `"gluten-free pepperoni with dairy-free mozzarella"`
   with `cl100k_base` (GPT-3.5/4) and `o200k_base` (GPT-4o) via `tiktoken` → print token
   boundaries side-by-side, count difference, compute cost delta at $0.002/1k vs $0.01/1k

2. **Temperature sweep**: send "Recommend a pizza" 5× at T=0, 5× at T=0.8, 5× at T=1.5 via
   Ollama → print all 15 → student counts unique responses per tier

3. **Lost-in-middle**: inject target fact at 0% / 50% / 100% position in a padded ~2000-token
   context, ask retrieval question, 5 runs per position → print hit/miss table

4. **Base vs instruct framing**: send `"Customer: What sizes do you have?\nAI:"` as raw
   completion vs proper chat message → print both → observe one continues a script, one answers

### Ch02 cells

1. **System prompt delta**: same user query, vague vs structured system prompt → print both,
   highlight format/scope difference

2. **Few-shot precision**: 0 examples vs 3 examples, extract an order as JSON → print both →
   observe compliance rate

3. **Schema enforcement**: ask-for-JSON vs JSON-schema-in-prompt vs Ollama `format="json"` →
   print outputs → measure parse success (`json.loads` try/except)

4. **Injection test**: attack string with and without defense layers in the system prompt → print
   what leaks vs what's blocked

### Ch03 cells

1. **Direct vs CoT**: multi-constraint query sent direct, then with "think step by step" →
   print both → observe intermediate reasoning depth

2. **Self-consistency**: N=1 vs N=5 majority vote on same query at T=0.7 → print all chains +
   voted answer → observe variance reduction

3. **Reasoning model trace**: same arithmetic problem via phi3:mini vs deepseek-r1:8b (Ollama)
   → print response + token count → observe `<think>` block length

4. **Sycophancy demo**: correct answer, then user pushback → print whether model flips →
   observe RLHF alignment side effect

### Ch04 cells

1. **Embedding similarity**: embed 5 pizza descriptions + 1 query with `all-MiniLM-L6-v2` →
   print cosine scores sorted → observe semantic proximity without keyword match

2. **RAG grounding**: same menu question, no context vs top-3 retrieved chunks in prompt →
   print both answers → observe hallucination vs grounded response

3. **HyDE vs raw query**: embed raw query vs LLM-generated hypothetical answer → print top-3
   retrieved chunks for each → observe retrieval precision difference

4. **Lost-in-middle RAG**: allergen chunk at position 1 vs 3 (of 5 retrieved) → run 5× per
   position → print whether model catches the flag each time

### Ch05 cells

1. **Distance metrics**: same 3 vectors, compute L2 / cosine / dot → print all rankings →
   observe they agree when L2-normalized, diverge when not

2. **Brute-force timing**: exact search at 1K / 10K / 100K vectors (numpy) → print time per
   query → observe O(N) growth

3. **IVF nprobe sweep**: FAISS IVF, nprobe=1 vs 4 vs 16 → print recall@10 + query time →
   observe recall-speed tradeoff

4. **HNSW efSearch sweep**: ChromaDB, efSearch=16 vs 64 vs 128 → print recall + query time →
   observe same tradeoff with different characteristics

---

## Phase 3 — GPU Supplement Notebooks (3 agents, parallel)

Creates `notebook-supplement-exercise.ipynb` and `notebook-supplement-solution.ipynb` for
chapters where GPU meaningfully changes what can be observed. Three chapters only.

### GPU guard cell (required, cell 1 in every supplement)

```python
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "No GPU detected — run the CPU notebook instead: notebook-exercise.ipynb\n"
        "To provision a GPU machine: see notes/06-ai-infrastructure/ch01-gpu-architecture/"
    )
device = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {device}  |  VRAM: {vram:.1f} GB")
assert vram >= 10, f"Need ≥10 GB VRAM, got {vram:.1f} GB — use a smaller model tier"
```

### Ch01 supplement — Model scale behavioral comparison (min: RTX 3090 24 GB)

- Same temperature sweep (cell 2 from Phase 2) but phi3:mini vs llama3.2:70b via Ollama →
  observe how model size affects output consistency at T=0 and at T=1.0

- Same lost-in-middle experiment with large context (32k token padding via a long filler
  document) → observe whether larger model degrades less or similarly

### Ch03 supplement — Reasoning depth vs model size (min: RTX 3080 12 GB)

- deepseek-r1:7b vs deepseek-r1:14b on same AIME-style multi-step problem → print reasoning
  trace length, final answer, token count → observe how model size affects reasoning quality

- Overthinking demo: send trivial query ("What is 2+2?") to deepseek-r1:14b → print reasoning
  token count → quantify the compute waste on simple problems

### Ch04 supplement — Embedding model precision comparison (min: any CUDA GPU)

- `all-MiniLM-L6-v2` (384d, CPU-viable) vs `bge-large-en-v1.5` (1024d, GPU-beneficial) →
  embed identical corpus → measure retrieval precision@5 on 10 held-out test queries → print
  comparison table → observe quality vs cost tradeoff

---

## Agent Context Budget

| Phase | Agent reads | Agent writes | Est. tokens/agent |
|---|---|---|---|
| Phase 1 (MD rewrite) | Existing `.md` (~3k tokens) | Rewritten `.md` | ~6k |
| Phase 2 (notebooks) | First 60 lines of `.md` + existing notebook (~2k) | Implemented notebook | ~4k |
| Phase 3 (supplements) | Brief only, no file reads | Two new notebooks | ~3k |

15 agent invocations total. No agent depends on another agent's output within the same phase.

---

## File Map

```
notes/03a-ai/
  ch01-llm-fundamentals/
    llm-fundamentals.md                  ← Phase 1 rewrite
    notebook-exercise.ipynb              ← Phase 2 implement
    notebook-supplement-exercise.ipynb   ← Phase 3 create
    notebook-supplement-solution.ipynb   ← Phase 3 create

  ch02-prompt-engineering/
    prompt-engineering.md                ← Phase 1 rewrite
    notebook-exercise.ipynb              ← Phase 2 implement

  ch03-cot-reasoning/
    cot-reasoning.md                     ← Phase 1 rewrite
    notebook-exercise.ipynb              ← Phase 2 implement
    notebook-supplement-exercise.ipynb   ← Phase 3 create
    notebook-supplement-solution.ipynb   ← Phase 3 create

  ch04-rag-and-embeddings/
    rag-and-embeddings.md                ← Phase 1 rewrite
    notebook-exercise.ipynb              ← Phase 2 implement
    notebook-supplement-exercise.ipynb   ← Phase 3 create
    notebook-supplement-solution.ipynb   ← Phase 3 create

  ch05-vector-dbs/
    vector-dbs.md                        ← Phase 1 rewrite
    notebook-exercise.ipynb              ← Phase 2 implement
```

---

## Out of Scope

- `notes/03-ai/` (PizzaBot arc) — archive/delete decision deferred to user
- Supplement `.md` files (`cot-reasoning-supplement.md`, etc.) — Phase 1 agent flags
  contradictions if found, no proactive rewrite
- Solution notebooks (`notebook-solution.ipynb`) — generated from exercise notebooks after
  Phase 2 in a separate pass
- `notes/03b-agentic-ai/` — this is where a scenario/grand challenge arc belongs; not touched
  by this plan
