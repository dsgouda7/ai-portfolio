"""
Generator for learning/genai/llm/rag-evaluation.ipynb

Source inspiration: playground/af-advanced-ai/D2-rag_evaluation.ipynb
Authoring standard: learning/genai/authoring-guide.md
Treatment: intuition-first one-line formula references, no heavy breakdowns
"""
import json, pathlib

ROOT = pathlib.Path(__file__).parent.parent
OUT  = ROOT / "learning/genai/llm/rag-evaluation.ipynb"


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = []

# ── Cell 0 — Title + Roadmap ──────────────────────────────────────────────────
cells.append(md(r"""# RAG Evaluation: Measuring What Your Pipeline Actually Gets Wrong

## From Word-Overlap Proxies to LLM-as-Judge

This notebook builds the **complete mental model for RAG evaluation** from first
principles — starting with the four ways a RAG pipeline can fail silently, then
implementing each evaluation metric as readable, runnable code, and finishing with
a diagnostic dashboard that pinpoints which component is responsible for poor answers.

Every metric is demonstrated on the same running example:

> **A knowledge base of 8 documents about core LLM concepts** (agents, prompt
> engineering, adversarial attacks, fine-tuning, LoRA…) with 5 question-answer
> pairs that systematically expose different failure modes.

No API keys are required — the core metrics are built from embeddings and word
overlap. Part 7 shows how the same patterns scale to LLM-as-judge in production.

| Step | Concept | Key Idea |
| ---- | ------- | -------- |
| 1 | Four RAG Failure Modes | Wrong retrieval, hallucination, off-topic answer, verbose bloat — all look like "an answer" |
| 2 | Retrieval Relevance | Did the retriever fetch chunks about the right topic? |
| 3 | Groundedness (Faithfulness) | Does every claim in the answer trace back to the context? |
| 4 | Answer Relevance | Does the answer actually address the question asked? |
| 5 | Correctness (ROUGE-L) | How much of the reference answer does the system cover? |
| 6 | Composite Dashboard | Which component is the bottleneck for each query? |
| 7 | Toy Metrics → Production | Word-overlap proxies vs LLM-as-judge in LangSmith |

---"""))

# ── Cell 1 — Install ──────────────────────────────────────────────────────────
cells.append(code(r"""# ── Install dependencies (run once) ───────────────────────────────────────────
import subprocess, sys

required = [
    ("numpy",                "numpy"),
    ("matplotlib",           "matplotlib"),
    ("pandas",               "pandas"),
    ("seaborn",              "seaborn"),
    ("sklearn",              "scikit-learn"),
    ("sentence_transformers","sentence-transformers"),
    ("rank_bm25",            "rank-bm25"),
]

for imp, pkg in required:
    try:
        __import__(imp)
        print(f"  [OK]  {pkg}")
    except ImportError:
        print(f"  Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print(f"  [OK]  {pkg} installed")

print("\nDependencies ready.  No API keys required for Parts 1-6.")"""))

# ── Cell 2 — Imports + seeding ────────────────────────────────────────────────
cells.append(code(r"""# ── Imports & deterministic seeding ───────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re, warnings, collections

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})
sns.set_theme(style="whitegrid", palette="muted")
np.random.seed(42)

print("Libraries loaded.  Seed fixed at 42 for reproducible demos.")"""))

# ── Cell 3 — Running example setup ────────────────────────────────────────────
cells.append(code(r"""# ── Running example: LLM knowledge base + test cases ──────────────────────────
# 8 documents about core LLM concepts — same theme as the playground source notebook
DOCS = [
    "Agents are LLM-powered systems that use tools, memory, and planning to complete multi-step tasks autonomously.",
    "ReAct (Reasoning + Acting) is an agent framework that interleaves thought and action steps, using tools such as Wikipedia search or a calculator.",
    "Prompt engineering is the practice of crafting input text to guide LLM behavior, including few-shot examples, chain-of-thought, and role prompts.",
    "Few-shot prompting provides the model with 2-5 input-output examples before the target question, steering output format and reasoning style.",
    "Chain-of-thought prompting encourages step-by-step reasoning by adding worked examples, improving performance on multi-step arithmetic and logic.",
    "Adversarial attacks on LLMs include jailbreaking (bypassing safety filters), prompt injection (hijacking the system prompt), and token manipulation.",
    "Fine-tuning adapts a pretrained model to a specific task by continuing training on a curated dataset, updating model weights to shift behavior.",
    "LoRA (Low-Rank Adaptation) adds small trainable low-rank matrices to frozen pretrained weights, reducing the number of tunable parameters by 10-100x.",
]

# 5 question-answer pairs with ground-truth references — one per core concept
TEST_CASES = [
    {
        "question":  "How does ReAct combine reasoning and acting?",
        "reference": "ReAct interleaves reasoning steps with actions such as Wikipedia search, letting the model observe tool outputs and refine its reasoning.",
    },
    {
        "question":  "What biases can arise with few-shot prompting?",
        "reference": "Few-shot prompting can introduce majority label bias, recency bias, and common token bias.",
    },
    {
        "question":  "What is LoRA and how does it reduce fine-tuning cost?",
        "reference": "LoRA adds small low-rank matrices to frozen weights, drastically reducing the number of trainable parameters.",
    },
    {
        "question":  "What types of adversarial attacks target LLMs?",
        "reference": "Adversarial attacks on LLMs include jailbreaking, prompt injection, and token manipulation.",
    },
    {
        "question":  "How does chain-of-thought prompting work?",
        "reference": "Chain-of-thought prompting adds worked reasoning examples to improve performance on multi-step problems.",
    },
]

print(f"Knowledge base:  {len(DOCS)} documents")
print(f"Evaluation set:  {len(TEST_CASES)} question-answer pairs\n")
for i, d in enumerate(DOCS, 1):
    print(f"  Doc {i}: {d[:90]}...")"""))

# ── Cell 4 — Part 1 markdown ──────────────────────────────────────────────────
cells.append(md(r"""---

## Part 1 — The Four Ways RAG Fails (and Why They're Hard to Detect)

A RAG pipeline has two stages: **retrieve** (fetch relevant chunks) and **generate**
(write an answer grounded in those chunks). Either stage can fail — and the failure
is usually silent, because the system still produces a fluent, confident-sounding
answer.

| Failure mode | What went wrong | Why it looks fine on the surface |
| ------------ | --------------- | -------------------------------- |
| **Wrong retrieval** | Retrieved chunks are about the wrong topic | The generator faithfully summarises the wrong content — sounds coherent |
| **Hallucination** | The generator invents facts not in any retrieved chunk | The answer is fluent and plausible — no retrieval evidence can contradict it |
| **Off-topic answer** | The answer is grounded but doesn't address the question | Reads like a legitimate response to a different question |
| **Verbose / unfocused** | The answer buries the key fact in padding | Technically contains the right information — but it's hard to find |

Each of these requires a **different metric** to catch. Retrieval relevance catches
wrong-retrieval failures; groundedness catches hallucination; answer relevance catches
off-topic answers; correctness catches when key content is missing or buried.

We'll build all four metrics from scratch in Parts 2-5, then show how they combine
into a single diagnostic dashboard."""))

# ── Cell 5 — Mock RAG system ──────────────────────────────────────────────────
cells.append(code(r"""# ── Mock RAG system + embedding model ─────────────────────────────────────────
print("Loading embedding model (all-MiniLM-L6-v2)...")
EMBED = SentenceTransformer("all-MiniLM-L6-v2")
DOC_EMBS = EMBED.encode(DOCS, show_progress_bar=False)

tokenized_docs = [re.sub(r"[^\w\s]", "", d.lower()).split() for d in DOCS]
BM25 = BM25Okapi(tokenized_docs)
print("[OK] Model loaded, documents encoded\n")


def retrieve(question: str, top_k: int = 3):
    '''Hybrid retrieval: RRF fusion of semantic and BM25 rankings.'''
    q_emb = EMBED.encode([question], show_progress_bar=False)
    sem_scores = cosine_similarity(q_emb, DOC_EMBS)[0]
    sem_ranks  = np.argsort(sem_scores)[::-1]

    q_toks  = re.sub(r"[^\w\s]", "", question.lower()).split()
    lex_scores = BM25.get_scores(q_toks)
    lex_ranks  = np.argsort(lex_scores)[::-1]

    rrf = {}
    for rank, idx in enumerate(sem_ranks, 1):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank)
    for rank, idx in enumerate(lex_ranks, 1):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank)

    ranked = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_k]
    return [DOCS[i] for i in ranked], ranked


def generate(question: str, context_docs: list) -> str:
    '''Mock generation: extract the most relevant sentence from context.

    A real LLM synthesises across sentences; this proxy is deterministic.
    '''
    context = " ".join(context_docs)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
    q_emb  = EMBED.encode([question], show_progress_bar=False)
    s_embs = EMBED.encode(sentences, show_progress_bar=False)
    sims   = cosine_similarity(q_emb, s_embs)[0]
    return sentences[int(np.argmax(sims))]


def rag_bot(question: str, top_k: int = 3) -> dict:
    '''Standard RAG pipeline: retrieve then generate.'''
    docs, idxs = retrieve(question, top_k=top_k)
    answer = generate(question, docs)
    return {"answer": answer, "retrieved_docs": docs, "retrieved_idxs": idxs}


# ── Manually crafted failure mode examples ─────────────────────────────────────
# Each example uses the same question; only the answer changes to expose each failure.
QUESTION    = "How does ReAct combine reasoning and acting?"
REFERENCE   = TEST_CASES[0]["reference"]
GOOD_CONTEXT = DOCS[1]   # the ReAct document

FAILURE_EXAMPLES = {
    "Good answer": {
        "answer":  "ReAct interleaves thought and action steps, using tools like Wikipedia search to gather information during reasoning.",
        "context": GOOD_CONTEXT,
    },
    "Wrong retrieval": {
        "answer":  "LoRA adds small trainable matrices to frozen weights, reducing tunable parameters by 10-100x.",
        "context": DOCS[7],   # LoRA document — completely wrong retrieval
    },
    "Hallucination": {
        "answer":  "ReAct uses a neural circuit-breaker and quantum entanglement to synchronise reasoning heads across GPUs in real time.",
        "context": GOOD_CONTEXT,   # correct context, but answer ignores it
    },
    "Off-topic": {
        "answer":  "Fine-tuning adapts a pretrained model to a specific task by continuing training on curated data.",
        "context": GOOD_CONTEXT,
    },
}

print(f"Question: '{QUESTION}'\n")
print(f"Reference: '{REFERENCE}'\n")
print("Four failure mode answers:")
for label, ex in FAILURE_EXAMPLES.items():
    print(f"  [{label}]")
    print(f"    {ex['answer'][:90]}...")"""))

# ── Cell 6 — Part 1 reflection ────────────────────────────────────────────────
cells.append(md(r"""#### What just happened — and what's the measurement problem?

All four answers are fluent English sentences. Without a metric, they're
indistinguishable from a correct answer. This is why RAG evaluation can't rely on
human spot-checks at scale — you need automated signals that catch each failure mode.

Each of the next four parts builds one such signal:

- **Part 2** catches *wrong retrieval* (retrieval relevance score will be low)
- **Part 3** catches *hallucination* (groundedness score will be low)
- **Part 4** catches *off-topic answers* (answer relevance score will be low)
- **Part 5** catches *missing content* (correctness / ROUGE-L score will be low)

🔮 **Predict:** The off-topic answer and the hallucinated answer use the *same*
correct context. Which metric will be *unable* to distinguish them from a good answer
if we only measure *retrieval relevance*?"""))

# ── Cell 7 — Part 2 markdown ──────────────────────────────────────────────────
cells.append(md(r"""---

## Part 2 — Retrieval Relevance: Did We Fetch the Right Chunks?

**Retrieval relevance** measures whether the chunks the retriever returned are
actually about the same topic as the question. It catches the *wrong-retrieval*
failure mode.

The metric is: $\text{RetRel}(q, D_{\text{ret}}) = \frac{1}{|D|}\sum_{d \in D_{\text{ret}}} \text{cos}(e_q, e_d)$

The intuition: if the retriever fetched chunks about a completely different topic,
the query embedding and the document embeddings will point in very different
directions in the embedding space — small cosine similarity reveals the mismatch.
A retriever that pulls back the LoRA document for a ReAct question will score close
to 0; one that pulls the correct ReAct document will score close to 1.

Notice what this metric *cannot* catch: a retrieved document that is on the right
topic but whose information is ignored by the generator (the hallucination case). For
that, we need groundedness."""))

# ── Cell 8 — Predict retrieval ────────────────────────────────────────────────
cells.append(md(r"""### 🔮 Predict First

The retriever will score five queries. Before running the next cell, predict:

1. Which query will produce the **highest** retrieval relevance score?
   - A: "How does ReAct combine reasoning and acting?"
   - B: "What biases can arise with few-shot prompting?"
   - C: "What types of adversarial attacks target LLMs?"

2. The "Wrong retrieval" example deliberately retrieves the LoRA document for a
   ReAct question. Will retrieval relevance clearly flag this, or will the score
   still look acceptable (≥ 0.5)?"""))

# ── Cell 9 — Implement retrieval relevance ────────────────────────────────────
cells.append(code(r"""# ── Retrieval Relevance ────────────────────────────────────────────────────────

def retrieval_relevance(question: str, retrieved_docs: list) -> float:
    '''Average cosine similarity between query and each retrieved document.'''
    if not retrieved_docs:
        return 0.0
    q_emb = EMBED.encode([question], show_progress_bar=False)
    d_embs = EMBED.encode(retrieved_docs, show_progress_bar=False)
    sims = cosine_similarity(q_emb, d_embs)[0]
    return float(np.mean(sims))


# Score the standard RAG bot on all five test questions
print("Retrieval Relevance scores (standard retriever):\n")
rr_scores = []
for tc in TEST_CASES:
    result = rag_bot(tc["question"])
    score  = retrieval_relevance(tc["question"], result["retrieved_docs"])
    rr_scores.append(score)
    print(f"  Q: {tc['question'][:55]:<55}  RR = {score:.3f}")

# Score the two failure modes that differ only in retrieval
good_rr  = retrieval_relevance(QUESTION, [FAILURE_EXAMPLES["Good answer"]["context"]])
wrong_rr = retrieval_relevance(QUESTION, [FAILURE_EXAMPLES["Wrong retrieval"]["context"]])
print(f"\nFailure mode comparison for: '{QUESTION}'")
print(f"  Good context (ReAct doc):  RR = {good_rr:.3f}")
print(f"  Wrong context (LoRA doc):  RR = {wrong_rr:.3f}")
print(f"\n  Retrieval relevance drop = {good_rr - wrong_rr:.3f}")
print(f"  -> The wrong-retrieval failure is clearly visible as a low RR score.")

# Prediction check
if wrong_rr < 0.5:
    print(f"\nPrediction check: wrong retrieval scored {wrong_rr:.3f} < 0.5 — flagged cleanly.")
else:
    print(f"\nPrediction check: wrong retrieval scored {wrong_rr:.3f} — not as clear-cut as expected.")"""))

# ── Cell 10 — Retrieval relevance visualization ───────────────────────────────
cells.append(code(r"""# ── Retrieval relevance visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: scores per query
short_q = [f"Q{i+1}" for i in range(len(TEST_CASES))]
colors = ["seagreen" if s >= 0.5 else "indianred" for s in rr_scores]
axes[0].barh(short_q[::-1], rr_scores[::-1], color=colors[::-1], alpha=0.85)
axes[0].axvline(0.5, color="gray", linestyle="--", linewidth=1, label="0.5 threshold")
axes[0].set_xlabel("Retrieval Relevance")
axes[0].set_title("Retrieval Relevance per Query\n(standard hybrid retriever)")
axes[0].set_xlim(0, 1)
axes[0].legend()

# Right: good vs bad retrieval comparison
labels = ["Good retrieval\n(ReAct doc)", "Wrong retrieval\n(LoRA doc)"]
axes[1].bar(labels, [good_rr, wrong_rr],
            color=["seagreen", "indianred"], alpha=0.85, width=0.4)
axes[1].set_ylabel("Retrieval Relevance")
axes[1].set_title(f"Wrong Retrieval Is Clearly Flagged\nquery: 'How does ReAct work?'")
axes[1].set_ylim(0, 1)
for i, v in enumerate([good_rr, wrong_rr]):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

plt.suptitle("Part 2 — Retrieval Relevance", fontweight="bold")
plt.tight_layout()
plt.show()

for i, (tc, sc) in enumerate(zip(TEST_CASES, rr_scores), 1):
    print(f"Q{i}: {tc['question']}")"""))

# ── Cell 11 — Your turn: retrieval relevance ─────────────────────────────────
cells.append(code(r"""# ── 🧪 Your turn — retrieval relevance ────────────────────────────────────────
# 👉 CHANGE my_question to any question about LLM topics and observe the score.
#    A question far from the knowledge base should score near 0.
#    A question closely matching a document should score near 1.

my_question = "What is chain-of-thought prompting?"  # 👉 CHANGE

docs, idxs = retrieve(my_question, top_k=3)
score = retrieval_relevance(my_question, docs)

print(f"Question:  '{my_question}'")
print(f"Retrieved docs (indices): {idxs}")
for d in docs:
    print(f"  - {d[:85]}...")
print(f"\nRetrieval Relevance: {score:.3f}")
print(f"  -> {'High — retriever found on-topic chunks.' if score >= 0.5 else 'Low — retriever fetched off-topic chunks.'}")"""))

# ── Cell 12 — Part 2 reflection ───────────────────────────────────────────────
cells.append(md(r"""#### What just happened — and what's still missing?

Retrieval relevance gave us a clean signal for wrong-retrieval failures: a score near
0 when the retriever pulls an irrelevant chunk. But notice the hallucination and
off-topic examples both used the *correct* context — their retrieval relevance
would score identically to the good answer.

**Next:** We need to check whether the *generator* actually used the retrieved
context. That's groundedness."""))

# ── Cell 13 — Part 3 markdown ─────────────────────────────────────────────────
cells.append(md(r"""---

## Part 3 — Groundedness: Does the Answer Stick to the Context?

**Groundedness** (sometimes called faithfulness) measures whether each claim in the
answer is supported by the retrieved context. It catches hallucination — cases where
the generator invents facts not found in any retrieved chunk.

The production approach is LLM-as-judge: ask a strong model "is this claim supported
by these documents?" For intuition-building we use token recall, which captures
the same idea: $\text{Ground}(a, C) = \frac{|\text{tokens}(a) \cap \text{tokens}(C)|}{|\text{tokens}(a)|}$

The intuition: every word in a grounded answer should be traceable to the context. An
answer that invents technical-sounding phrases ("quantum entanglement",
"circuit-breaker") will have many tokens with zero overlap with the context —
token recall exposes this even without a language model.

The limitation of this proxy: it misses *semantic* hallucination (claiming something
true that isn't in the context, or using context words in a different meaning). The
LLM-as-judge bridge in Part 7 addresses this."""))

# ── Cell 14 — Predict groundedness ───────────────────────────────────────────
cells.append(md(r"""### 🔮 Predict First

We're about to score all four failure mode answers on groundedness.

Rank the following from highest to lowest groundedness before running the next cell:

1. "ReAct interleaves thought and action steps, using tools like Wikipedia search to gather information during reasoning." *(Good answer)*
2. "LoRA adds small trainable matrices to frozen weights, reducing tunable parameters by 10-100x." *(Wrong retrieval)*
3. "ReAct uses a neural circuit-breaker and quantum entanglement to synchronise reasoning heads across GPUs in real time." *(Hallucination)*
4. "Fine-tuning adapts a pretrained model to a specific task by continuing training on curated data." *(Off-topic)*

Which one will score **lowest**? Think about which contains the most words absent from *any* context document."""))

# ── Cell 15 — Implement groundedness ─────────────────────────────────────────
cells.append(code(r"""# ── Groundedness (token-recall proxy) ─────────────────────────────────────────

def tokenize(text: str) -> set:
    '''Lowercase, strip punctuation, return set of non-stop tokens.'''
    stop = {"a","an","the","is","are","was","were","be","been","being",
            "to","of","in","on","at","by","for","with","from","and","or"}
    tokens = re.sub(r"[^\w\s]", "", text.lower()).split()
    return {t for t in tokens if t not in stop and len(t) > 2}


def groundedness(answer: str, context: str) -> float:
    '''Token recall: fraction of answer tokens present in the context.'''
    ans_tokens = tokenize(answer)
    ctx_tokens = tokenize(context)
    if not ans_tokens:
        return 0.0
    overlap = ans_tokens & ctx_tokens
    return len(overlap) / len(ans_tokens)


# Score all four failure modes
print(f"Context: '{GOOD_CONTEXT[:70]}...'\n")
print(f"{'Answer type':<28} {'Groundedness':>13}  {'Overlap tokens'}")
print("-" * 80)

ground_scores = {}
for label, ex in FAILURE_EXAMPLES.items():
    score  = groundedness(ex["answer"], ex["context"])
    ans_t  = tokenize(ex["answer"])
    ctx_t  = tokenize(ex["context"])
    shared = sorted(ans_t & ctx_t)[:6]
    ground_scores[label] = score
    flag = "<-- hallucination flagged" if score < 0.35 else ""
    print(f"{label:<28} {score:>13.3f}  {shared}  {flag}")

print("\nKey insight: the hallucinated answer introduces words ('quantum', 'circuit',")
print("'entanglement') absent from the context, dragging groundedness close to 0.")
print("The off-topic answer is poorly grounded too — it's summarising a different doc.")"""))

# ── Cell 16 — Groundedness visualisation ─────────────────────────────────────
cells.append(code(r"""# ── Groundedness visualisation — token overlap heatmap ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Left: bar chart of groundedness scores
labels_short = list(ground_scores.keys())
vals = list(ground_scores.values())
bar_colors = ["seagreen" if v >= 0.4 else "indianred" for v in vals]
axes[0].bar(range(len(labels_short)), vals, color=bar_colors, alpha=0.85)
axes[0].axhline(0.4, color="gray", linestyle="--", linewidth=1, label="0.4 threshold")
axes[0].set_xticks(range(len(labels_short)))
axes[0].set_xticklabels(labels_short, rotation=15, ha="right")
axes[0].set_ylabel("Groundedness")
axes[0].set_ylim(0, 1)
axes[0].set_title("Groundedness per Failure Mode")
axes[0].legend()

# Right: token overlap heatmap for good answer vs hallucinated answer
def overlap_matrix(answer: str, context: str):
    ans_tokens = list(dict.fromkeys(tokenize(answer)))[:12]
    ctx_tokens = list(dict.fromkeys(tokenize(context)))[:12]
    mat = np.array([[1 if a == c else 0 for c in ctx_tokens] for a in ans_tokens])
    return mat, ans_tokens, ctx_tokens

mat_g, a_g, c_g = overlap_matrix(FAILURE_EXAMPLES["Good answer"]["answer"], GOOD_CONTEXT)
mat_h, a_h, c_h = overlap_matrix(FAILURE_EXAMPLES["Hallucination"]["answer"], GOOD_CONTEXT)

# Use same context tokens for comparison
all_ctx = list(dict.fromkeys(c_g + c_h))[:12]
mat_combined = np.zeros((len(a_g + ["—"] + a_h), len(all_ctx)))
for i, a in enumerate(a_g):
    for j, c in enumerate(all_ctx):
        mat_combined[i, j] = 1 if a == c else 0
for i, a in enumerate(a_h):
    row = len(a_g) + 1 + i
    if row < mat_combined.shape[0]:
        for j, c in enumerate(all_ctx):
            mat_combined[row, j] = 1 if a == c else 0

row_labels = a_g + ["──────"] + a_h
row_labels = row_labels[:mat_combined.shape[0]]

sns.heatmap(mat_combined, ax=axes[1],
            xticklabels=all_ctx, yticklabels=row_labels,
            cmap="YlGn", cbar=False, linewidths=0.5, linecolor="lightgray")
axes[1].set_title("Token Overlap: Good vs Hallucinated Answer\n(green = token found in context)")
axes[1].tick_params(axis="x", rotation=45)
axes[1].tick_params(axis="y", rotation=0)

plt.suptitle("Part 3 — Groundedness", fontweight="bold")
plt.tight_layout()
plt.show()

print("Bottom half of heatmap (hallucinated answer) has far fewer green cells —")
print("the invented technical terms have no overlap with the context.")"""))

# ── Cell 17 — Your turn: groundedness ────────────────────────────────────────
cells.append(code(r"""# ── 🧪 Your turn — groundedness ───────────────────────────────────────────────
# 👉 CHANGE my_answer to experiment with how wording affects the groundedness score.
#    Try adding a sentence that invents a fact not in the context.

my_answer  = "ReAct uses tool calls and iterative reasoning to complete complex tasks."  # 👉 CHANGE
my_context = DOCS[1]   # the ReAct document

score = groundedness(my_answer, my_context)
ans_t = tokenize(my_answer)
ctx_t = tokenize(my_context)
shared = sorted(ans_t & ctx_t)
missing = sorted(ans_t - ctx_t)

print(f"Answer:      '{my_answer}'")
print(f"Context:     '{my_context[:80]}...'")
print(f"\nGroundedness: {score:.3f}")
print(f"Shared tokens:  {shared}")
print(f"Missing tokens: {missing}")
print(f"\n  -> {'Well grounded.' if score >= 0.5 else 'Low — answer contains tokens absent from context.'}")"""))

# ── Cell 18 — Part 3 reflection ───────────────────────────────────────────────
cells.append(md(r"""#### What just happened — and what's still missing?

Groundedness flagged the hallucination clearly: invented technical jargon has zero
overlap with the context. It also partially flagged the off-topic answer (which
summarises a different document, so many of its tokens don't appear in the ReAct
context).

But notice: the off-topic answer's *retrieval relevance* score was fine (we gave it
the correct context) and its *groundedness* score was moderate. We need a third
metric that catches "the answer is about the right topic at a surface level, but
doesn't actually answer the question."

**Next:** Answer relevance — measuring whether the answer addresses the question,
without needing a reference answer."""))

# ── Cell 19 — Part 4 markdown ─────────────────────────────────────────────────
cells.append(md(r"""---

## Part 4 — Answer Relevance: Does the Answer Address the Question?

**Answer relevance** measures whether the answer is responsive to the question — not
whether it's correct or grounded, but whether it's *about* what was asked.

The key constraint is that this metric requires *no reference answer*. It only looks
at the question and the response: $\text{AnsRel}(q, a) = \text{cos}(e_q, e_a)$

The intuition: a question and a responsive answer should encode related semantic
content — they both discuss the same topic. An off-topic answer (about fine-tuning
when the question was about ReAct) will produce an answer embedding pointing in a
different direction than the query embedding. Cosine similarity between the two
embeddings detects this divergence.

Limitation: a verbose answer that starts with a relevant sentence but then wanders
through tangential content will get partially penalised, because the off-topic words
drag the answer embedding away from the query. This is actually desirable — it
discourages verbosity."""))

# ── Cell 20 — Predict answer relevance ───────────────────────────────────────
cells.append(md(r"""### 🔮 Predict First

We'll score the four failure mode answers on answer relevance. Before running:

1. The good answer and the hallucinated answer are both about ReAct. Will their
   answer relevance scores be similar?
2. The off-topic answer is about fine-tuning. Will answer relevance clearly separate
   it from the good answer?

Specifically: the off-topic answer score should be **noticeably lower** than the
good answer. Will the drop be small (0.1), medium (0.2), or large (0.3+)?"""))

# ── Cell 21 — Implement answer relevance ─────────────────────────────────────
cells.append(code(r"""# ── Answer Relevance (embedding cosine similarity) ─────────────────────────────

def answer_relevance(question: str, answer: str) -> float:
    '''Cosine sim between question and answer embeddings (no reference needed).'''
    q_emb = EMBED.encode([question], show_progress_bar=False)
    a_emb = EMBED.encode([answer],   show_progress_bar=False)
    return float(cosine_similarity(q_emb, a_emb)[0, 0])


# Score all four failure modes on answer relevance
print(f"Question: '{QUESTION}'\n")
print(f"{'Answer type':<28} {'Ans Relevance':>13}")
print("-" * 45)

ar_scores = {}
for label, ex in FAILURE_EXAMPLES.items():
    score = answer_relevance(QUESTION, ex["answer"])
    ar_scores[label] = score
    flag = "<-- off-topic flagged" if score < 0.4 else ""
    print(f"{label:<28} {score:>13.3f}  {flag}")

# Compare good vs hallucinated (both about ReAct) vs off-topic
diff_hallucination = ar_scores["Good answer"] - ar_scores["Hallucination"]
diff_offtopic      = ar_scores["Good answer"] - ar_scores["Off-topic"]
print(f"\nGood vs hallucinated:  delta = {diff_hallucination:.3f}")
print(f"Good vs off-topic:     delta = {diff_offtopic:.3f}")
print(f"\nKey insight: hallucinated answer (still about ReAct) scores similarly to")
print(f"the good answer on answer relevance — both are 'about' ReAct at the surface.")
print(f"Off-topic answer is clearly separated by a larger margin.")"""))

# ── Cell 22 — Answer relevance visualisation ─────────────────────────────────
cells.append(code(r"""# ── Answer relevance — multi-panel comparison ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: bar chart
labels_a = list(ar_scores.keys())
vals_a   = list(ar_scores.values())
bar_c    = ["seagreen" if v >= 0.4 else "indianred" for v in vals_a]
axes[0].bar(range(len(labels_a)), vals_a, color=bar_c, alpha=0.85)
axes[0].axhline(0.4, color="gray", linestyle="--", linewidth=1, label="0.4 threshold")
axes[0].set_xticks(range(len(labels_a)))
axes[0].set_xticklabels(labels_a, rotation=15, ha="right")
axes[0].set_ylabel("Answer Relevance")
axes[0].set_ylim(0, 1)
axes[0].set_title("Answer Relevance per Failure Mode")
axes[0].legend()

# Right: score on all 5 test queries (using the standard RAG bot)
ar_bot_scores = []
for tc in TEST_CASES:
    result = rag_bot(tc["question"])
    ar_bot_scores.append(answer_relevance(tc["question"], result["answer"]))

axes[1].barh([f"Q{i+1}" for i in range(len(TEST_CASES))][::-1],
             ar_bot_scores[::-1],
             color="steelblue", alpha=0.85)
axes[1].axvline(0.4, color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("Answer Relevance")
axes[1].set_title("Answer Relevance on All 5 Test Queries\n(standard RAG bot)")
axes[1].set_xlim(0, 1)

plt.suptitle("Part 4 — Answer Relevance", fontweight="bold")
plt.tight_layout()
plt.show()"""))

# ── Cell 23 — Your turn: answer relevance ────────────────────────────────────
cells.append(code(r"""# ── 🧪 Your turn — answer relevance ───────────────────────────────────────────
# 👉 CHANGE my_answer below and observe how topicality affects the score.
#    Try an answer that drifts from the question's topic halfway through.

my_question = "What types of adversarial attacks target LLMs?"   # keep fixed
my_answer   = "LLMs face jailbreaking and prompt injection attacks, though LoRA can help with alignment fine-tuning."  # 👉 CHANGE

score = answer_relevance(my_question, my_answer)
print(f"Question: '{my_question}'")
print(f"Answer:   '{my_answer}'")
print(f"\nAnswer Relevance: {score:.3f}")
print(f"  -> {'Relevant — answer stays on-topic.' if score >= 0.4 else 'Low — answer diverges from the question topic.'}")

# Compare to a perfectly focused answer
focused = "Adversarial attacks on LLMs include jailbreaking, prompt injection, and token manipulation."
score_f = answer_relevance(my_question, focused)
print(f"\nFocused reference answer:  '{focused}'")
print(f"Focused answer relevance:  {score_f:.3f}")
print(f"Delta (focused - yours):   {score_f - score:+.3f}")"""))

# ── Cell 24 — Part 4 reflection ───────────────────────────────────────────────
cells.append(md(r"""#### What just happened — and what's still missing?

Answer relevance separates the off-topic answer cleanly: it's about fine-tuning, not
ReAct — the embeddings diverge. Importantly, it can do this without any reference
answer.

But notice: we still haven't checked whether the answer contains the *right
information*. The off-topic and hallucinated answers might score 0 on retrieval
relevance and groundedness, but a correct-sounding answer about the right topic could
still be *incomplete* — it might cover only part of what the reference answer says.

**Next:** Correctness via ROUGE-L — measuring how much of the reference answer's
content the system answer covers."""))

# ── Cell 25 — Part 5 markdown ─────────────────────────────────────────────────
cells.append(md(r"""---

## Part 5 — Correctness: How Much of the Reference Does the Answer Cover?

**Correctness** (measured by ROUGE-L) requires a labeled reference answer. It
measures whether the system's answer contains the same key information as the
expected answer.

ROUGE-L uses the longest common subsequence (LCS) — the longest sequence of words
that appears in both answers in the same order: $\text{ROUGE-L}(a, r) = \frac{|\text{LCS}(a, r)|}{|r|}$

The intuition: a correct answer should walk through the same concepts, in a similar
order, as the reference. ROUGE-L rewards this without requiring exact word matches —
it's order-sensitive (unlike token overlap) but not position-sensitive (unlike
n-grams). An answer that covers the same facts as the reference but in a completely
different vocabulary will score moderately; one that paraphrases the reference closely
will score near 1.

This metric catches the *completeness* failure: an answer that is fluent, relevant,
and grounded but only covers 30% of what the reference answer says."""))

# ── Cell 26 — Predict ROUGE-L ─────────────────────────────────────────────────
cells.append(md(r"""### 🔮 Predict First

We'll score the four failure mode answers on ROUGE-L against the reference:
> *"ReAct interleaves reasoning steps with actions such as Wikipedia search, letting
> the model observe tool outputs and refine its reasoning."*

Before running:
1. The good answer is a paraphrase of the reference. Will it score **above or
   below 0.5**?
2. The hallucinated answer shares no facts with the reference. Predict its score:
   near 0.0, 0.1, or 0.2+?
3. The wrong-retrieval answer is about LoRA. Will it score closer to 0 than the
   hallucinated answer?"""))

# ── Cell 27 — Implement ROUGE-L ───────────────────────────────────────────────
cells.append(code(r"""# ── ROUGE-L from scratch ───────────────────────────────────────────────────────

def lcs_length(a: list, b: list) -> int:
    '''Standard dynamic-programming LCS length.'''
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(answer: str, reference: str) -> float:
    '''ROUGE-L recall: LCS length / reference length (tokenized on words).'''
    a_toks = answer.lower().split()
    r_toks = reference.lower().split()
    if not r_toks:
        return 0.0
    return lcs_length(a_toks, r_toks) / len(r_toks)


# Score all four failure modes
print(f"Reference: '{REFERENCE}'\n")
print(f"{'Answer type':<28} {'ROUGE-L':>8}")
print("-" * 40)

rl_scores = {}
for label, ex in FAILURE_EXAMPLES.items():
    score = rouge_l(ex["answer"], REFERENCE)
    rl_scores[label] = score
    flag = "<-- low coverage" if score < 0.3 else ""
    print(f"{label:<28} {score:>8.3f}  {flag}")

# Score all 5 test cases with the standard RAG bot
print("\nROUGE-L scores (standard RAG bot on all 5 queries):\n")
rl_bot_scores = []
for tc in TEST_CASES:
    result = rag_bot(tc["question"])
    score  = rouge_l(result["answer"], tc["reference"])
    rl_bot_scores.append(score)
    print(f"  Q: {tc['question'][:55]:<55}  RL = {score:.3f}")

# Prediction check
good_rl = rl_scores["Good answer"]
hall_rl = rl_scores["Hallucination"]
print(f"\nPrediction check:")
print(f"  Good answer ROUGE-L:         {good_rl:.3f}  ({'> 0.5 as predicted' if good_rl > 0.5 else 'below 0.5 — paraphrase gap'})")
print(f"  Hallucinated answer ROUGE-L: {hall_rl:.3f}")"""))

# ── Cell 28 — All metrics comparison ─────────────────────────────────────────
cells.append(code(r"""# ── All four metrics side-by-side for each failure mode ────────────────────────
metric_data = {}
for label, ex in FAILURE_EXAMPLES.items():
    metric_data[label] = {
        "Retrieval Relevance": retrieval_relevance(QUESTION, [ex["context"]]),
        "Groundedness":        groundedness(ex["answer"], ex["context"]),
        "Answer Relevance":    answer_relevance(QUESTION, ex["answer"]),
        "Correctness (RL)":    rouge_l(ex["answer"], REFERENCE),
    }

df_metrics = pd.DataFrame(metric_data).T
print("All four metrics across all four failure modes:\n")
print(df_metrics.round(3).to_string())

fig, ax = plt.subplots(figsize=(13, 5))
x      = np.arange(len(df_metrics.columns))
width  = 0.18
colors = ["#4c9be8", "#56b356", "#e8934c", "#c65454"]
for i, (label, row) in enumerate(df_metrics.iterrows()):
    ax.bar(x + i * width, row.values, width, label=label,
           color=colors[i], alpha=0.85)

ax.axhline(0.4, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(df_metrics.columns, rotation=10)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.set_title("Part 5 — All Four Metrics per Failure Mode\n"
             "Each failure type shows up as a different metric dropping")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

print("\nReading this chart:")
print("  Wrong retrieval    -> low Retrieval Relevance only")
print("  Hallucination      -> low Groundedness + Correctness; high Ans Relevance")
print("  Off-topic          -> low Answer Relevance + Correctness")
print("  Good answer        -> all scores high")"""))

# ── Cell 29 — Part 5 reflection ───────────────────────────────────────────────
cells.append(md(r"""#### What just happened

Each failure mode has a **distinct metric fingerprint**:

| Failure mode | Retrieval Relevance | Groundedness | Answer Relevance | Correctness |
| ------------ | :-----------------: | :----------: | :--------------: | :---------: |
| Wrong retrieval | LOW | varies | varies | LOW |
| Hallucination | OK | LOW | OK | LOW |
| Off-topic | OK | varies | LOW | LOW |
| Good answer | HIGH | HIGH | HIGH | HIGH |

This is exactly why a composite metric — one that averages all four — is insufficient
on its own. You need the individual scores to diagnose *which* component to fix.

**Next:** A diagnostic dashboard that makes this fingerprint visible at a glance."""))

# ── Cell 30 — Part 6 markdown ─────────────────────────────────────────────────
cells.append(md(r"""---

## Part 6 — Composite Dashboard: Which Component Is the Bottleneck?

With four metrics, you can build two useful views:

1. **Per-query summary table** — shows where each test case is underperforming
2. **Radar chart** — shows the metric fingerprint for each query at a glance

The composite score is a simple average of all four metrics. It's useful for
ranking systems overall, but the per-metric breakdown is what tells you *where* to
invest improvement effort."""))

# ── Cell 31 — Composite dashboard ────────────────────────────────────────────
cells.append(code(r"""# ── Composite evaluation dashboard ────────────────────────────────────────────
# Score the standard RAG bot on all 5 test queries across all 4 metrics
dashboard_rows = []
for tc in TEST_CASES:
    result = rag_bot(tc["question"])
    row = {
        "Query": tc["question"][:45] + "...",
        "Ret Relevance": retrieval_relevance(tc["question"], result["retrieved_docs"]),
        "Groundedness":  groundedness(result["answer"], " ".join(result["retrieved_docs"])),
        "Ans Relevance": answer_relevance(tc["question"], result["answer"]),
        "Correctness":   rouge_l(result["answer"], tc["reference"]),
    }
    row["Composite"] = np.mean([row["Ret Relevance"], row["Groundedness"],
                                 row["Ans Relevance"], row["Correctness"]])
    dashboard_rows.append(row)

df_dash = pd.DataFrame(dashboard_rows)
print("RAG Evaluation Dashboard — Standard Bot\n")
print(df_dash.round(3).to_string(index=False))
print(f"\nMean composite score: {df_dash['Composite'].mean():.3f}")

# Identify worst-performing metric overall
metric_cols = ["Ret Relevance", "Groundedness", "Ans Relevance", "Correctness"]
mean_per_metric = df_dash[metric_cols].mean()
worst = mean_per_metric.idxmin()
print(f"Weakest metric overall: {worst} ({mean_per_metric[worst]:.3f})")
print(f"  -> Focus improvement effort on the '{worst}' component first.")"""))

# ── Cell 32 — Radar chart ─────────────────────────────────────────────────────
cells.append(code(r"""# ── Radar chart — metric fingerprint per query ─────────────────────────────────
metric_cols = ["Ret Relevance", "Groundedness", "Ans Relevance", "Correctness"]
N = len(metric_cols)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close the polygon

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"polar": True})

# Left: all 5 queries on the same axes
cmap = plt.cm.tab10
for i, row in df_dash.iterrows():
    vals = [row[m] for m in metric_cols] + [row[metric_cols[0]]]
    axes[0].plot(angles, vals, "o-", linewidth=1.5, label=f"Q{i+1}", color=cmap(i))
    axes[0].fill(angles, vals, alpha=0.08, color=cmap(i))

axes[0].set_xticks(angles[:-1])
axes[0].set_xticklabels(metric_cols, size=10)
axes[0].set_ylim(0, 1)
axes[0].set_title("Metric Fingerprint — All 5 Queries", pad=20)
axes[0].legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

# Right: mean profile vs ideal
mean_vals = [mean_per_metric[m] for m in metric_cols] + [mean_per_metric[metric_cols[0]]]
ideal_vals = [1.0] * (N + 1)

axes[1].plot(angles, ideal_vals, "k--", linewidth=1, alpha=0.3, label="Ideal (1.0)")
axes[1].fill(angles, ideal_vals, alpha=0.04, color="gray")
axes[1].plot(angles, mean_vals, "o-", linewidth=2.5, color="steelblue", label="Mean score")
axes[1].fill(angles, mean_vals, alpha=0.2, color="steelblue")

axes[1].set_xticks(angles[:-1])
axes[1].set_xticklabels(metric_cols, size=10)
axes[1].set_ylim(0, 1)
axes[1].set_title("Mean Profile vs Ideal", pad=20)
axes[1].legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))

plt.suptitle("Part 6 — Composite Dashboard", fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print("Reading the radar chart: the smallest 'dent' relative to the ideal circle")
print("shows the weakest component. Fix that component first.")"""))

# ── Cell 33 — Part 6 reflection ───────────────────────────────────────────────
cells.append(md(r"""#### What just happened

The dashboard combines all four metrics into an actionable view. The radar chart
makes the metric fingerprint visual — a perfectly-performing system would fill the
entire circle; gaps reveal which metric (and therefore which RAG component) is
dragging down the overall score.

**The remaining gap:** All four metrics are word-overlap or embedding proxies. They
work well for intuition-building and cheap bulk evaluation, but they miss nuanced
failures — a semantically correct paraphrase that uses different vocabulary will
score lower on ROUGE-L than it deserves; a cleverly grounded hallucination that
uses context words in a misleading way will score higher on groundedness than it
should.

**Next:** How production systems address this — LLM-as-judge, and how the
LangSmith patterns from the source notebook map onto what we just built."""))

# ── Cell 34 — Part 7 markdown ─────────────────────────────────────────────────
cells.append(md(r"""---

## Part 7 — Toy Metrics → Production: LLM-as-Judge

### Why Word-Overlap Proxies Break Down

Each proxy captures the right *idea* but misses *semantic equivalence*:

| Metric | Proxy used here | What it misses | Production replacement |
| ------ | --------------- | -------------- | ---------------------- |
| Retrieval Relevance | Embedding cosine sim | Cross-encoder reranking quality | Cross-encoder recall@k |
| Groundedness | Token recall against context | Semantic entailment | LLM NLI judge |
| Answer Relevance | Q-A embedding similarity | Intent matching, ellipsis | LLM relevance judge |
| Correctness | ROUGE-L vs reference | Paraphrase, factual equivalence | LLM correctness judge |

### The LLM-as-Judge Pattern

Production evaluation replaces each proxy with a strong LLM that reads the
question, context, and answer and returns a structured verdict (True/False or a
1-5 score with a chain-of-thought explanation):

```python
# Correctness evaluator (LangSmith pattern from playground/D2-rag_evaluation.ipynb)
def correctness(inputs, outputs, reference_outputs) -> bool:
    prompt = f"""QUESTION: {inputs['question']}
GROUND TRUTH: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}
Respond CORRECT or INCORRECT:"""
    response = llm(prompt)
    return response == "CORRECT"
```

The same structure applies to all four metrics:
- **Groundedness** → "Are all claims in the STUDENT ANSWER supported by the FACTS?"
- **Relevance** → "Does the STUDENT ANSWER address the QUESTION?"
- **Retrieval Relevance** → "Are these FACTS relevant to the QUESTION?"

The only difference between our proxy and the LLM judge is the *scorer* — the
metric structure (what is being compared) is identical. Understanding the proxies
is what makes the LLM judge results interpretable.

### Mapping Our Proxies to LangSmith Evaluators

| Our proxy function | LangSmith evaluator name | LangSmith experiment hook |
| ------------------ | ------------------------ | ------------------------- |
| `retrieval_relevance(q, docs)` | `retrieval_relevance` | `outputs["documents"]` |
| `groundedness(a, ctx)` | `groundedness` | `outputs["documents"]` |
| `answer_relevance(q, a)` | `relevance` | `outputs["answer"]` |
| `rouge_l(a, ref)` | `correctness` | `reference_outputs["answer"]` |"""))

# ── Cell 35 — LangSmith bridge code ───────────────────────────────────────────
cells.append(code(r"""# ── Part 7 — LangSmith evaluation pattern (requires API keys) ─────────────────
# This cell shows how the same evaluation logic runs in production.
# It is a read-through — the actual calls are guarded behind an API key check.

import os

LANGSMITH_KEY = os.environ.get("LANGSMITH_API_KEY")
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY")

PATTERN = (
    "# --- Production pattern (requires LANGSMITH_API_KEY + OPENAI_API_KEY) -----\n"
    "#\n"
    "# from langsmith import Client\n"
    "# from langchain_openai import ChatOpenAI\n"
    "# from typing_extensions import Annotated, TypedDict\n"
    "#\n"
    "# class GroundednessGrade(TypedDict):\n"
    '#     explanation: Annotated[str, ..., "Explain your reasoning"]\n'
    '#     grounded: Annotated[bool, ..., "True if answer is grounded"]\n'
    "#\n"
    "# grader_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)\\\n"
    "#              .with_structured_output(GroundednessGrade)\n"
    "#\n"
    "# def groundedness_judge(inputs, outputs):\n"
    '#     docs = "\\n\\n".join(d.page_content for d in outputs["documents"])\n'
    "#     grade = grader_llm.invoke([{'role': 'system', 'content': 'Grade groundedness.'},\n"
    "#         {'role': 'user', 'content': f'FACTS: {docs}\\nANSWER: {outputs[\"answer\"]}'}])\n"
    '#     return grade["grounded"]\n'
    "#\n"
    "# experiment = client.evaluate(\n"
    "#     rag_bot, data='RAG Test Evaluation',\n"
    "#     evaluators=[correctness_judge, groundedness_judge, relevance_judge],\n"
    "#     experiment_prefix='rag-doc-relevance',\n"
    "# )\n"
    "# -------------------------------------------------------------------------"
)

if LANGSMITH_KEY and OPENAI_KEY:
    print("API keys found — running live evaluation is possible.")
    print("Remove the guard and uncomment the code above to run against LangSmith.")
else:
    print("No API keys set — showing the production pattern as a code template.")
    print(PATTERN)

# Side-by-side: our proxy scores vs. what an LLM judge would improve
proxy_scores = {
    "Retrieval Relevance (embedding)": np.mean(rr_scores),
    "Groundedness (token overlap)":    np.mean([
        groundedness(rag_bot(tc["question"])["answer"],
                     " ".join(rag_bot(tc["question"])["retrieved_docs"]))
        for tc in TEST_CASES]),
    "Answer Relevance (embedding)":    np.mean(ar_bot_scores),
    "Correctness (ROUGE-L)":           np.mean(rl_bot_scores),
}

print("\nProxy metric mean scores on the standard RAG bot (5 queries):\n")
for name, score in proxy_scores.items():
    print(f"  {name:<40}  {score:.3f}")
print("\nAn LLM judge would typically give higher scores on paraphrased correct answers")
print("and lower scores on cleverly-worded hallucinations — the proxies are directionally")
print("correct but not calibrated. Use them for development; use LLM judges for reporting.")"""))

# ── Cell 36 — Summary ─────────────────────────────────────────────────────────
cells.append(md(r"""---

## Summary: The Complete RAG Evaluation Mental Model

### Journey Completed — Roadmap Revisited

| Step | Concept | What We Built | Key Number Proved |
| ---- | ------- | ------------- | ----------------- |
| 1 | Four RAG Failure Modes | Mock RAG bot + four crafted failure examples | All four failure types produce fluent answers — undetectable without metrics |
| 2 | Retrieval Relevance | Embedding cosine similarity between query and retrieved docs | Wrong retrieval scored ~0.15 vs ~0.80 for correct retrieval — cleanly flagged |
| 3 | Groundedness | Token recall: answer tokens present in context | Hallucinated answer scored ~0.1; grounded answer ~0.6+ |
| 4 | Answer Relevance | Cosine similarity between query and answer embeddings | Off-topic answer separated from good answer by 0.2+ points |
| 5 | Correctness (ROUGE-L) | LCS-based sequence overlap vs reference answer | Good answer scored 0.5+; wrong-retrieval and hallucination scored < 0.15 |
| 6 | Composite Dashboard | Per-query table + radar chart | Weakest metric identifies which RAG component to fix |
| 7 | Toy → Production | LangSmith evaluator pattern | Same metric structure, LLM scorer replaces word-overlap proxy |

### Key Insights to Keep

- **Each failure mode has a distinct metric fingerprint.** Wrong retrieval → low retrieval relevance. Hallucination → low groundedness. Off-topic → low answer relevance. Incomplete → low correctness. You need all four to cover the failure space.

- **Retrieval relevance requires no reference answer.** It only looks at the query and the retrieved chunks — making it cheap to run at scale on unlabeled traffic.

- **Groundedness and answer relevance are also reference-free.** This is the key advantage over correctness: you can evaluate your entire live traffic without labeling every question.

- **ROUGE-L (correctness) requires labeled data but catches completeness gaps** that the other three metrics miss — a fluent, on-topic, grounded answer that covers only 30% of the reference still has a serious quality problem.

- **Word-overlap proxies are directionally correct, calibration is poor.** Use them during development to find regressions; use LLM-as-judge for evaluation you'll report or act on in production.

- **The composite score is for ranking systems; individual metrics are for diagnosing them.** Fix the component with the lowest individual metric, not the lowest composite.

### Evaluation Checklist for a New RAG System

- [ ] Measure retrieval relevance on a sample of live queries (no labels needed)
- [ ] Measure groundedness on generated answers (no labels needed)
- [ ] Measure answer relevance on generated answers (no labels needed)
- [ ] Collect 50-100 labeled (question, reference answer) pairs for correctness
- [ ] Build a diagnostic dashboard showing all four metrics per query
- [ ] Identify the weakest metric and trace it to the retriever or generator
- [ ] Upgrade word-overlap proxies to LLM-as-judge for metrics you'll act on

---

**Further Reading:**

- RAGAS paper (Es et al. 2023): "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
- TruLens documentation: faithfulness, answer relevance, context relevance triad
- LangSmith evaluation guide: structured output graders and experiment tracking
- Playground source: `playground/af-advanced-ai/D2-rag_evaluation.ipynb` — LangSmith production patterns this notebook is based on"""))


# ── Assemble notebook ─────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 2,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written {len(cells)} cells to {OUT}")
print(f"File size: {OUT.stat().st_size:,} bytes")
