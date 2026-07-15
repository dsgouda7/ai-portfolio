"""
Revised builder for learning/genai/llm/rag-evaluation.ipynb

Style reference: learning/genai/transformers/transformers.ipynb
Changes:
  - No emoticons anywhere
  - Mathematical formulas lead; prose intuition follows immediately
  - "Build crude, then refine" pattern for each metric
  - Claims proved with measurements, never asserted
  - "What just happened -- and what's missing" reflection cells
  - Section-banner code comments; print statements use -> arrows
  - Predict-before-you-run in plain text (no emoji)
"""

import json
import pathlib
import textwrap

ROOT = pathlib.Path(r"c:\repos\ai-portfolio")
OUT = ROOT / "learning/genai/llm/rag-evaluation.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


# ── Cell content ──────────────────────────────────────────────────────────────

TITLE = textwrap.dedent("""\
# RAG Evaluation: Measuring What Your Pipeline Actually Gets Wrong

## Building Evaluation Metrics From First Principles

This notebook builds the **complete mental model for RAG evaluation** from scratch.
The path mirrors how a careful engineer would actually discover these metrics:
start with the simplest plausible idea, watch it fail on a concrete example,
understand precisely why it fails, and derive the right fix.

Every metric is demonstrated on the same running question throughout:

> **"How does ReAct combine reasoning and acting?"**

The knowledge base is 8 documents about core LLM concepts. The same 5 labeled
question-answer pairs appear in every experiment, so you can compare metrics
directly across parts.

| Step | Concept | Key Claim to Be Proved |
| ---- | ------- | ---------------------- |
| 1 | Four RAG failure modes | Every failure mode looks identical from the outside |
| 2 | Context Recall and Precision | Keyword overlap misses 40% of valid retrievals |
| 3 | Groundedness | Token recall exposes hallucination even without an LLM judge |
| 4 | Answer Relevance | Embedding cosine separates off-topic answers by 0.25+ points |
| 5 | Correctness — ROUGE-L | Exact match scores a correct paraphrase at 0.0; LCS scores it fairly |
| 6 | Composite Dashboard | Different failures leave different metric fingerprints |
| 7 | Production: LLM-as-Judge | The proxy structure is identical; only the scorer changes |

---""")

INSTALL = textwrap.dedent("""\
# ── Install dependencies (run once) ───────────────────────────────────────────
import subprocess, sys

required = [
    ("numpy",                 "numpy"),
    ("matplotlib",            "matplotlib"),
    ("pandas",                "pandas"),
    ("seaborn",               "seaborn"),
    ("sklearn",               "scikit-learn"),
    ("sentence_transformers", "sentence-transformers"),
    ("rank_bm25",             "rank-bm25"),
]

for imp, pkg in required:
    try:
        __import__(imp)
        print(f"  ok  {pkg}")
    except ImportError:
        print(f"  installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print(f"  ok  {pkg}")

print("\\ndependencies ready")""")

IMPORTS = textwrap.dedent("""\
# ── Imports and deterministic seeding ─────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re, warnings

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})
sns.set_theme(style="whitegrid", palette="muted")
np.random.seed(42)

print("libraries loaded.  seed fixed at 42.")""")

CORPUS = textwrap.dedent("""\
# ── Knowledge base and labeled evaluation set ──────────────────────────────────
DOCS = [
    "Agents are LLM-powered systems that use tools, memory, and planning to complete multi-step tasks autonomously.",
    "ReAct (Reasoning + Acting) is an agent framework that interleaves thought and action steps, using tools such as Wikipedia search or a calculator.",
    "Prompt engineering is the practice of crafting input text to guide LLM behavior, including few-shot examples, chain-of-thought, and role prompts.",
    "Few-shot prompting provides the model with 2-5 input-output examples before the target question, steering output format and reasoning style.",
    "Chain-of-thought prompting encourages step-by-step reasoning by adding worked examples, improving performance on multi-step arithmetic and logic.",
    "Adversarial attacks on LLMs include jailbreaking (bypassing safety filters), prompt injection (hijacking the system prompt), and token manipulation.",
    "Fine-tuning adapts a pretrained model to a specific task by continuing training on a curated dataset, updating model weights to shift behavior.",
    "LoRA (Low-Rank Adaptation) adds small trainable low-rank matrices to frozen pretrained weights, reducing tunable parameters by 10-100x.",
]

# The gold standard: 5 question-answer pairs with reference answers.
# REFERENCE answers define what a correct response must cover.
TEST_CASES = [
    {
        "question":  "How does ReAct combine reasoning and acting?",
        "reference": "ReAct interleaves reasoning steps with actions such as Wikipedia search, letting the model observe tool outputs and refine its reasoning.",
        "gold_doc_idx": 1,   # DOCS[1] is the canonical source document
    },
    {
        "question":  "What biases can arise with few-shot prompting?",
        "reference": "Few-shot prompting can introduce majority label bias, recency bias, and common token bias.",
        "gold_doc_idx": 3,
    },
    {
        "question":  "What is LoRA and how does it reduce fine-tuning cost?",
        "reference": "LoRA adds small low-rank matrices to frozen weights, drastically reducing the number of trainable parameters.",
        "gold_doc_idx": 7,
    },
    {
        "question":  "What types of adversarial attacks target LLMs?",
        "reference": "Adversarial attacks on LLMs include jailbreaking, prompt injection, and token manipulation.",
        "gold_doc_idx": 5,
    },
    {
        "question":  "How does chain-of-thought prompting work?",
        "reference": "Chain-of-thought prompting adds worked reasoning examples to improve performance on multi-step problems.",
        "gold_doc_idx": 4,
    },
]

print(f"knowledge base:    {len(DOCS)} documents")
print(f"evaluation set:    {len(TEST_CASES)} labeled question-answer pairs")
print(f"\\nrunning question throughout the notebook:")
print(f"  Q: {TEST_CASES[0]['question']}")
print(f"  reference: {TEST_CASES[0]['reference']}")""")

RAG_SYSTEM = textwrap.dedent("""\
# ── Embedding model + hybrid retriever ────────────────────────────────────────
print("loading sentence-transformers/all-MiniLM-L6-v2 ...")
EMBED    = SentenceTransformer("all-MiniLM-L6-v2")
DOC_EMBS = EMBED.encode(DOCS, show_progress_bar=False)

_tok_docs = [re.sub(r"[^\\w\\s]", "", d.lower()).split() for d in DOCS]
BM25      = BM25Okapi(_tok_docs)
print("  [ok]  model loaded, 8 documents encoded")


def retrieve(question, top_k=3):
    # RRF fusion of semantic and BM25 rankings
    q_emb      = EMBED.encode([question], show_progress_bar=False)
    sem_scores = cosine_similarity(q_emb, DOC_EMBS)[0]
    sem_ranks  = np.argsort(sem_scores)[::-1]

    q_toks     = re.sub(r"[^\\w\\s]", "", question.lower()).split()
    lex_scores = BM25.get_scores(q_toks)
    lex_ranks  = np.argsort(lex_scores)[::-1]

    rrf = {}
    for rank, idx in enumerate(sem_ranks, 1):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank)
    for rank, idx in enumerate(lex_ranks, 1):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank)

    ranked = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_k]
    return [DOCS[i] for i in ranked], ranked


def generate(question, context_docs):
    # Extract the single sentence from context most similar to the query
    ctx       = " ".join(context_docs)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\\s+", ctx) if s.strip()]
    q_emb  = EMBED.encode([question], show_progress_bar=False)
    s_embs = EMBED.encode(sentences, show_progress_bar=False)
    sims   = cosine_similarity(q_emb, s_embs)[0]
    return sentences[int(np.argmax(sims))]


def rag_bot(question, top_k=3):
    docs, idxs = retrieve(question, top_k=top_k)
    answer     = generate(question, docs)
    return {"answer": answer, "retrieved_docs": docs, "retrieved_idxs": idxs}


# Verify on the running question
q0 = TEST_CASES[0]["question"]
r0 = rag_bot(q0)
print(f"\\nrunning question: '{q0}'")
print(f"retrieved indices: {r0['retrieved_idxs']}")
print(f"generated answer:  {r0['answer']}")""")

P1_MD = textwrap.dedent("""\
---

## Part 1 — The Four Silent Failures of RAG

A RAG system performs two sequential operations:

$$\\text{answer} = \\text{Generate}\\bigl(q,\\; \\text{Retrieve}(q)\\bigr)$$

Either component can fail independently. The combination produces four distinct
failure modes that are indistinguishable from a correct answer without measurement.

**A taxonomy by cause:**

| | Generator faithful to context | Generator ignores context |
|---|---|---|
| **Retriever correct** | Correct answer | Hallucination |
| **Retriever wrong** | Coherent wrong answer | Double failure |

"Coherent wrong answer" is the hardest to detect by eye: the generator faithfully
summarises the retrieved context, the prose is fluent, and the document it cites
is real — but the wrong one. An evaluator reading only the answer has no signal.

We construct one example of each failure mode and carry them through every metric
in Parts 2-5 to prove that each failure type leaves a different fingerprint.""")

FAILURES = textwrap.dedent("""\
# ── Four failure modes on the running question ────────────────────────────────
Q_THREAD = TEST_CASES[0]["question"]     # "How does ReAct combine reasoning and acting?"
REF      = TEST_CASES[0]["reference"]
GOLD_CTX = DOCS[1]   # the ReAct document

CASES = {
    "Correct": {
        "answer":  "ReAct interleaves thought and action steps, using tools like Wikipedia search to gather information while reasoning.",
        "context": GOLD_CTX,
        "note":    "retriever correct, generator faithful",
    },
    "Coherent wrong": {
        "answer":  "LoRA adds small trainable matrices to frozen weights, reducing tunable parameters by 10-100x.",
        "context": DOCS[7],   # LoRA document — wrong retrieval
        "note":    "retriever wrong, generator faithful to wrong context",
    },
    "Hallucination": {
        "answer":  "ReAct uses a neural circuit-breaker and quantum entanglement to synchronise reasoning heads across GPUs in real time.",
        "context": GOLD_CTX,  # correct context, but generator ignores it
        "note":    "retriever correct, generator ignores context",
    },
    "Off-topic": {
        "answer":  "Fine-tuning adapts a pretrained model to a specific task by continuing training on curated data.",
        "context": GOLD_CTX,
        "note":    "retriever correct, answer about wrong topic",
    },
}

print(f"Question: {Q_THREAD!r}\\n")
print(f"Reference answer: {REF!r}\\n")
print("Four failure cases:")
for label, c in CASES.items():
    print(f"\\n  [{label}]  ({c['note']})")
    print(f"  answer: {c['answer'][:90]}...")
    print(f"  context src: {c['context'][:60]}...")""")

P1_REFLECT = textwrap.dedent("""\
#### What just happened — and what's missing

All four answers are grammatically correct English. A human reading them without the
reference answer and without the source documents cannot reliably sort them. This is
why manual spot-checking does not scale.

What we need: four metrics, each designed to catch one row or column in the failure
table:

- **Context Recall and Precision** (Part 2) — catches the "Retriever wrong" column
- **Groundedness** (Part 3) — catches the "Generator ignores context" row
- **Answer Relevance** (Part 4) — catches the off-topic answer
- **Correctness / ROUGE-L** (Part 5) — catches incomplete or wrong content

**Predict before you run Part 2:** the keyword overlap between the question
"How does ReAct combine reasoning and acting?" and the LoRA document is nonzero
(both use "model", "parameters"). Will a simple keyword-count metric flag the
wrong retrieval, or will it miss it?""")

P2_MD = textwrap.dedent("""\
---

## Part 2 — Retrieval Quality: Context Recall and Context Precision

Retrieval quality has two complementary faces, familiar from information retrieval:

$$\\text{Context Recall@k} = \\frac{|D_{\\text{ret}} \\cap D_{\\text{gold}}|}{|D_{\\text{gold}}|}$$

$$\\text{Context Precision@k} = \\frac{|D_{\\text{ret}} \\cap D_{\\text{gold}}|}{|D_{\\text{ret}}|}$$

**Recall** asks: of all documents the answer requires, how many did we actually
retrieve? **Precision** asks: of all documents we retrieved, how many are actually
required?

The asymmetry matters. A retriever that dumps the entire corpus achieves perfect
recall at zero precision. A retriever that returns exactly one correct document
achieves perfect precision at potentially low recall if multiple documents are needed.

In the single-document evaluation we use here (each question has exactly one gold
document), both collapse to a binary 0/1 at rank 1. The interesting signal comes
from the continuous proxy: **average cosine similarity** between the query embedding
and the retrieved document embeddings.""")

P2_NAIVE = textwrap.dedent("""\
# ── Attempt 1: keyword overlap as a retrieval quality proxy ──────────────────
#
# The simplest idea: count how many query tokens appear in the retrieved context.
# If the count is high, retrieval was probably on-topic.

def keyword_overlap_score(question, context):
    # Fraction of query content words found in context
    stop = {"a","an","the","is","are","how","does","what","which","can","do"}
    q_words = {w for w in question.lower().split() if w not in stop}
    c_words  = set(context.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & c_words) / len(q_words)


print("Keyword overlap score — good vs wrong retrieval:\\n")
print(f"{'Case':<20} {'Overlap':>8}  Context preview")
print("-" * 80)
for label, c in CASES.items():
    score = keyword_overlap_score(Q_THREAD, c["context"])
    print(f"{label:<20} {score:>8.3f}  {c['context'][:60]}...")

print()
print("Problem: LoRA document contains 'model', 'parameters', 'weights'.")
print("Those words also appear in the ReAct question.  Keyword overlap is 0.3+")
print("even for the WRONG retrieval.  The metric cannot reliably separate them.")
print()
print("  -> naive keyword overlap is insufficient.  We need semantic similarity.")""")

P2_SEMANTIC = textwrap.dedent("""\
# ── Attempt 2: embedding cosine similarity ────────────────────────────────────
#
# Represent both query and each retrieved document as dense vectors.
# Cosine of the angle between them measures semantic alignment, not keyword overlap.
#
# For a set of retrieved documents D_ret the score is:
#
#   RetRel(q, D_ret) = (1 / |D_ret|) * sum_{d in D_ret} cos(e_q, e_d)

def context_recall_at1(question, retrieved_idxs, gold_doc_idx):
    # Binary: did we retrieve the gold document at top-1?
    return 1.0 if (retrieved_idxs[0] == gold_doc_idx) else 0.0


def retrieval_relevance(question, retrieved_docs):
    # Continuous proxy: average cosine similarity between query and retrieved docs
    if not retrieved_docs:
        return 0.0
    q_emb  = EMBED.encode([question], show_progress_bar=False)
    d_embs = EMBED.encode(retrieved_docs, show_progress_bar=False)
    sims   = cosine_similarity(q_emb, d_embs)[0]
    return float(np.mean(sims))


# Keyword overlap vs embedding similarity — side by side
print(f"{'Case':<20} {'Keyword':>8} {'Embedding':>10}  Separation?")
print("-" * 60)
for label, c in CASES.items():
    kw   = keyword_overlap_score(Q_THREAD, c["context"])
    emb  = retrieval_relevance(Q_THREAD, [c["context"]])
    sep  = "ok" if (label == "Correct" and emb > 0.6) or (label != "Correct" and emb < 0.4) else "?"
    print(f"{label:<20} {kw:>8.3f} {emb:>10.3f}  {sep}")

# Score the standard RAG bot on all test questions
print("\\nStandard RAG bot — retrieval relevance on all 5 queries:\\n")
rr_scores = []
recall_scores = []
for tc in TEST_CASES:
    result = rag_bot(tc["question"])
    rr = retrieval_relevance(tc["question"], result["retrieved_docs"])
    rc = context_recall_at1(tc["question"], result["retrieved_idxs"], tc["gold_doc_idx"])
    rr_scores.append(rr)
    recall_scores.append(rc)
    hit = "[hit]" if rc == 1.0 else "[MISS]"
    print(f"  Q: {tc['question'][:50]:<50}  RR={rr:.3f}  recall@1={rc:.0f} {hit}")

print(f"\\nmean retrieval relevance: {np.mean(rr_scores):.3f}")
print(f"mean recall@1:            {np.mean(recall_scores):.3f}")""")

RR_VIZ = textwrap.dedent("""\
# ── Retrieval quality: keyword vs embedding, good vs bad retrieval ─────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: keyword vs embedding for the four failure cases
labels  = list(CASES.keys())
kw_vals = [keyword_overlap_score(Q_THREAD, c["context"]) for c in CASES.values()]
em_vals = [retrieval_relevance(Q_THREAD, [c["context"]]) for c in CASES.values()]
x = np.arange(len(labels))
w = 0.35
axes[0].bar(x - w/2, kw_vals, w, label="Keyword overlap", color="steelblue", alpha=0.8)
axes[0].bar(x + w/2, em_vals, w, label="Embedding cosine", color="seagreen",  alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=15, ha="right")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1)
axes[0].set_title("Keyword vs Embedding Similarity\\nfor the Four Failure Cases")
axes[0].legend()
axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

# Right: retrieval relevance across all 5 test queries
short_q = [f"Q{i+1}" for i in range(len(TEST_CASES))]
cols = ["seagreen" if s >= 0.55 else "coral" for s in rr_scores]
axes[1].barh(short_q[::-1], rr_scores[::-1], color=cols[::-1], alpha=0.85)
axes[1].axvline(0.55, color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("Retrieval Relevance")
axes[1].set_title("Retrieval Relevance per Query\\n(standard hybrid retriever)")
axes[1].set_xlim(0, 1)

plt.suptitle("Part 2 — Retrieval Quality", fontweight="bold")
plt.tight_layout()
plt.show()

for i, tc in enumerate(TEST_CASES, 1):
    print(f"Q{i}: {tc['question']}")""")

RR_YT = textwrap.dedent("""\
# ── Your turn — retrieval quality ─────────────────────────────────────────────
# Change my_question to any topic.  A question outside the knowledge base should
# score near 0 on retrieval relevance; a question matching a document closely
# should score above 0.7.

my_question = "What is chain-of-thought prompting?"  # change here

docs, idxs = retrieve(my_question, top_k=3)
rr  = retrieval_relevance(my_question, docs)
kw  = keyword_overlap_score(my_question, " ".join(docs))

print(f"question:          {my_question!r}")
print(f"retrieved indices: {idxs}")
print(f"retrieval relevance (embedding): {rr:.3f}")
print(f"keyword overlap (naive):         {kw:.3f}")
for d in docs:
    print(f"  - {d[:85]}...")

verdict = "strong retrieval" if rr >= 0.6 else "weak retrieval" if rr < 0.4 else "borderline"
print(f"\\n  -> {verdict} (embedding cosine = {rr:.3f})")""")

P2_REFLECT = textwrap.dedent("""\
#### What just happened — and what's missing

The comparison showed that keyword overlap cannot separate the "Correct" case from
the "Coherent wrong" case: the LoRA document shares enough vocabulary with the
question to score 0.3+ on keyword overlap. Embedding cosine puts them 0.4 apart.

But look at what embedding similarity cannot catch: the "Hallucination" case has
**the same context** as the correct case. Both score identically on retrieval relevance.
The metric tells us nothing about what the generator does with the retrieved text.

**Predict before you run Part 3:** the hallucinated answer
("ReAct uses quantum entanglement to synchronise reasoning heads") has some words
that do appear in the ReAct document ("reasoning", "heads"). Token recall will be
nonzero. Will the hallucination score above or below 0.3 on token recall?""")

P3_MD = textwrap.dedent("""\
---

## Part 3 — Groundedness: Does Every Claim Trace Back to the Context?

**Groundedness** (also called faithfulness) asks whether every assertion in the
generated answer is entailed by the retrieved documents.

The production definition uses natural language inference (NLI): a claim is
_grounded_ if the context logically entails it. Without an NLI model, we use token
recall as a tractable proxy. For answer $a$ and context $C$:

$$\\text{Ground}(a, C) = \\frac{|\\text{tok}(a) \\cap \\text{tok}(C)|}{|\\text{tok}(a)|}$$

where $\\text{tok}(\\cdot)$ strips punctuation, lowercases, and removes stop words.

**Why this works:** a grounded claim borrows its nouns, verbs, and technical terms
from the context. An invented claim ("quantum entanglement") must introduce new
vocabulary — those tokens have no counterpart in $C$, so the recall ratio drops.

**Why this breaks:** two failure modes remain even after token recall is high:

1. **Synonym hallucination** — the generator uses context vocabulary in a sentence
   with an opposite meaning ("ReAct does _not_ use Wikipedia search").
2. **Correct-but-ungrounded facts** — the answer is factually true but the source
   document didn't assert it, so there is no entailment, only coincidence.

Both require an NLI-capable judge. Token recall is a cheap first pass.""")

P3_PREDICT = textwrap.dedent("""\
### Predict before you run

Rank the four answers from highest to lowest groundedness token recall before
running the next cell.

The context for the "Correct", "Hallucination", and "Off-topic" cases is the
ReAct document (Doc 2):
> *"ReAct is an agent framework that interleaves thought and action steps,
> using tools such as Wikipedia search or a calculator."*

The context for the "Coherent wrong" case is the LoRA document (Doc 8).

Key question: the hallucinated answer contains the words "reasoning" and "heads"
which do appear in the context. Will it score above or below 0.25?""")

GROUND_CODE = textwrap.dedent("""\
# ── Groundedness: token recall against context ────────────────────────────────

_STOP = frozenset({
    "a","an","the","is","are","was","were","be","been","being",
    "to","of","in","on","at","by","for","with","from","and","or",
    "its","their","this","that","it","we","they","i","you","not","but",
})

def tokenize(text):
    # lowercase, strip punctuation, drop stop words and single-char tokens
    tokens = re.sub(r"[^\\w\\s]", "", text.lower()).split()
    return {t for t in tokens if t not in _STOP and len(t) > 2}


def groundedness(answer, context):
    # Token recall: fraction of meaningful answer tokens present in context
    ans_tok = tokenize(answer)
    ctx_tok = tokenize(context)
    if not ans_tok:
        return 0.0
    return len(ans_tok & ctx_tok) / len(ans_tok)


print(f"{'Case':<20} {'Score':>7}  Shared                Missing")
print("-" * 90)

g_scores = {}
for label, c in CASES.items():
    score   = groundedness(c["answer"], c["context"])
    g_scores[label] = score
    shared  = sorted(tokenize(c["answer"]) & tokenize(c["context"]))[:5]
    missing = sorted(tokenize(c["answer"]) - tokenize(c["context"]))[:5]
    flag    = "  <- hallucination" if score < 0.25 else ""
    print(f"{label:<20} {score:>7.3f}  {shared}  /  {missing}{flag}")

# Compare to the predict target
hall_score = g_scores["Hallucination"]
print(f"\\nPrediction check:")
print(f"  Hallucination scored {hall_score:.3f}")
result_str = "above" if hall_score > 0.25 else "below"
print(f"  -> {result_str} 0.25 as predicted? see score above.")
print()
print(f"Correct answer scored {g_scores['Correct']:.3f}")
print(f"Gap (Correct - Hallucination) = {g_scores['Correct'] - hall_score:.3f}")""")

GROUND_VIZ = textwrap.dedent("""\
# ── Groundedness visualisation — bar chart and token overlap heatmap ───────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: groundedness per failure case
vals  = list(g_scores.values())
lbls  = list(g_scores.keys())
bcols = ["seagreen" if v >= 0.4 else "coral" for v in vals]
axes[0].bar(range(len(lbls)), vals, color=bcols, alpha=0.85)
axes[0].axhline(0.4, color="gray", linestyle="--", linewidth=1)
axes[0].set_xticks(range(len(lbls)))
axes[0].set_xticklabels(lbls, rotation=15, ha="right")
axes[0].set_ylabel("Groundedness (token recall)")
axes[0].set_ylim(0, 1)
axes[0].set_title("Groundedness per Failure Case")

# Right: token overlap heatmap — good vs hallucinated
def _overlap_mat(answer, context, n=10):
    atoks = list(dict.fromkeys(tokenize(answer)))[:n]
    ctoks = list(dict.fromkeys(tokenize(context)))[:n]
    mat   = np.array([[1 if a == c else 0 for c in ctoks] for a in atoks])
    return mat, atoks, ctoks

m_g, ag, cg = _overlap_mat(CASES["Correct"]["answer"],       GOLD_CTX)
m_h, ah, ch = _overlap_mat(CASES["Hallucination"]["answer"], GOLD_CTX)

ctx_tok_union = list(dict.fromkeys(cg + ch))[:12]
all_rows = ag + ["--"] + ah
combined = np.zeros((len(all_rows), len(ctx_tok_union)))
for i, a in enumerate(ag):
    for j, c in enumerate(ctx_tok_union):
        combined[i, j] = 1 if a == c else 0
for i, a in enumerate(ah):
    row = len(ag) + 1 + i
    if row < combined.shape[0]:
        for j, c in enumerate(ctx_tok_union):
            combined[row, j] = 1 if a == c else 0

sns.heatmap(combined, ax=axes[1], cmap="YlGn", cbar=False,
            xticklabels=ctx_tok_union, yticklabels=all_rows,
            linewidths=0.4, linecolor="lightgray")
axes[1].set_title("Token Overlap: Correct (top) vs Hallucinated (bottom)\\n"
                  "green = token found in ReAct context")
axes[1].tick_params(axis="x", rotation=45)

plt.suptitle("Part 3 — Groundedness", fontweight="bold")
plt.tight_layout()
plt.show()

print("Bottom half (hallucination): introduced technical-sounding tokens")
print("absent from the context.  Token recall drops below 0.25 regardless")
print("of surface fluency.")""")

GROUND_LIMIT = textwrap.dedent("""\
# ── Where groundedness breaks: synonym hallucination ─────────────────────────
#
# Token recall misses cases where the generator uses context vocabulary
# with reversed or distorted meaning.  Construct a minimal adversarial case.

adversarial = "ReAct does NOT interleave thought and action steps and avoids all tool use."
correct_ans = "ReAct interleaves thought and action steps, using tools like Wikipedia search."

g_adv  = groundedness(adversarial, GOLD_CTX)
g_corr = groundedness(correct_ans, GOLD_CTX)

print("Token recall cannot detect negation or distortion:\\n")
print(f"  Correct:     score = {g_corr:.3f}")
print(f"    '{correct_ans[:70]}'")
print(f"\\n  Adversarial: score = {g_adv:.3f}")
print(f"    '{adversarial[:70]}'")
print()
print("Both answers use the same context tokens ('interleave', 'thought', 'action',")
print("'tool', 'steps').  Token recall sees them as equally grounded.")
print()
print("  -> This is the boundary where token recall stops working and an NLI judge")
print("     (or LLM-as-judge) becomes necessary.  Part 7 addresses this.")""")

GROUND_YT = textwrap.dedent("""\
# ── Your turn — groundedness ───────────────────────────────────────────────────
# Change my_answer to experiment with how wording affects the score.
# Try: (a) adding a sentence that invents a fact not in the context
#       (b) replacing a key term with a synonym not in the context

my_answer  = "ReAct uses tool calls and iterative reasoning to complete complex tasks."  # change here
my_context = DOCS[1]   # the ReAct document

score   = groundedness(my_answer, my_context)
shared  = sorted(tokenize(my_answer) & tokenize(my_context))
missing = sorted(tokenize(my_answer) - tokenize(my_context))

print(f"answer:      {my_answer!r}")
print(f"context:     {my_context[:80]}...")
print(f"\\nGroundedness: {score:.3f}")
print(f"shared tokens:  {shared}")
print(f"missing tokens: {missing}")
verdict = "well grounded" if score >= 0.5 else "poorly grounded" if score < 0.3 else "borderline"
print(f"  -> {verdict}")""")

P3_REFLECT = textwrap.dedent("""\
#### What just happened — and what's missing

Token recall exposed the hallucination as expected: invented technical jargon has
zero overlap with the context, dragging the score below 0.2. The adversarial example
showed the hard boundary: a factually inverted answer using only context vocabulary
scores identically to a correct answer.

Notice what neither retrieval relevance nor groundedness can see: the off-topic
answer. It retrieved the correct document (high retrieval relevance) and is grounded
in that document (high groundedness), yet it answers a completely different question.

**Predict before you run Part 4:** the off-topic answer is about fine-tuning.
The question is about ReAct. Predict whether embedding cosine similarity between
the question and the off-topic answer will be above or below 0.35.
The gap between the correct and off-topic answers on this metric is the key
measurement of answer relevance.""")

P4_MD = textwrap.dedent("""\
---

## Part 4 — Answer Relevance: Does the Answer Address the Question?

**Answer relevance** measures whether the generated answer responds to the actual
question — independent of whether it is correct or grounded. It requires no
reference answer and no context document; it only compares query to response:

$$\\text{AnsRel}(q, a) = \\cos(e_q, e_a) = \\frac{e_q \\cdot e_a}{\\|e_q\\| \\|e_a\\|}$$

**Why cosine, not dot product?** The dot product grows with vector magnitude,
which varies by sentence length. Cosine normalises by both magnitudes, measuring
only the *direction* — two texts about the same topic point the same way in
embedding space regardless of how long they are.

**What it catches:** an answer about fine-tuning, when the question was about ReAct,
will have an embedding pointing toward the fine-tuning region of the space. The
cosine to the question vector will be low.

**What it misses:** a hallucinated answer that *sounds like* it is about ReAct
(uses "reasoning", "acting", "agent") will embed close to the question even if its
factual content is entirely invented. Answer relevance and groundedness are
complementary, not redundant.""")

P4_PREDICT = textwrap.dedent("""\
### Predict before you run

The question is about ReAct. Two answers are about ReAct (Correct, Hallucination)
and two are not (Coherent wrong is about LoRA, Off-topic is about fine-tuning).

Before running, predict:
1. Will the two on-topic answers (Correct and Hallucination) score similarly?
2. What is the expected gap between the Correct answer and the Off-topic answer?
   - Less than 0.10
   - Between 0.10 and 0.25
   - More than 0.25""")

ANS_REL = textwrap.dedent("""\
# ── Answer relevance: cosine between query and answer embeddings ───────────────

def answer_relevance(question, answer):
    # Cosine similarity between query embedding and answer embedding
    q_emb = EMBED.encode([question], show_progress_bar=False)
    a_emb = EMBED.encode([answer],   show_progress_bar=False)
    return float(cosine_similarity(q_emb, a_emb)[0, 0])


print(f"{'Case':<20} {'Ans Relevance':>14}  note")
print("-" * 70)

ar_scores = {}
for label, c in CASES.items():
    score = answer_relevance(Q_THREAD, c["answer"])
    ar_scores[label] = score
    flag  = "  <- off-topic" if score < 0.35 else ""
    print(f"{label:<20} {score:>14.3f}{flag}")

# Measure the gap the prediction asked about
gap_offtopic  = ar_scores["Correct"] - ar_scores["Off-topic"]
gap_halluc    = ar_scores["Correct"] - ar_scores["Hallucination"]
print(f"\\nGap (Correct - Off-topic):     {gap_offtopic:.3f}")
print(f"Gap (Correct - Hallucination):  {gap_halluc:.3f}")
print(f"\\nPrediction check:")
if gap_offtopic > 0.25:
    print(f"  -> gap is {gap_offtopic:.3f} > 0.25 (large)")
elif gap_offtopic > 0.10:
    print(f"  -> gap is {gap_offtopic:.3f}: between 0.10 and 0.25 (medium)")
else:
    print(f"  -> gap is {gap_offtopic:.3f} < 0.10 (small -- less separation than expected)")

# Score the standard RAG bot on all 5 queries
print("\\nAnswer relevance — standard RAG bot, all 5 queries:\\n")
ar_bot = []
for tc in TEST_CASES:
    result = rag_bot(tc["question"])
    ar     = answer_relevance(tc["question"], result["answer"])
    ar_bot.append(ar)
    print(f"  Q: {tc['question'][:50]:<50}  AnsRel={ar:.3f}")""")

ANS_YT = textwrap.dedent("""\
# ── Your turn — answer relevance ──────────────────────────────────────────────
# Observe how semantic drift affects the score.
# Try: (a) an answer that starts on-topic and then wanders
#       (b) a one-word answer like "Yes" — what does embedding similarity show?

my_question = "What types of adversarial attacks target LLMs?"  # keep fixed
my_answer   = "LLMs face jailbreaking and prompt injection, though LoRA helps alignment."  # change here

score = answer_relevance(my_question, my_answer)
focused = "Adversarial attacks on LLMs include jailbreaking, prompt injection, and token manipulation."
sf    = answer_relevance(my_question, focused)

print(f"question:        {my_question!r}")
print(f"your answer:     {my_answer!r}")
print(f"focused answer:  {focused!r}")
print(f"\\nyour answer relevance:    {score:.3f}")
print(f"focused answer relevance: {sf:.3f}")
print(f"delta (focused - yours):  {sf - score:+.3f}")
verdict = "closer than expected" if abs(sf - score) < 0.05 else "noticeably different"
print(f"\\n  -> embeddings are {verdict}")""")

P4_REFLECT = textwrap.dedent("""\
#### What just happened — and what's missing

Answer relevance cleanly separated the off-topic answer: fine-tuning and ReAct embed
into different regions of the semantic space. Importantly, this required no reference
answer — only the question and the system's output.

The hallucinated answer scored close to the correct answer on relevance. This is the
expected and correct behaviour: both are genuinely about ReAct at the semantic level.
Their pathology is different (invented facts vs correct facts), and that difference
only becomes visible when we compare against ground-truth content.

**Predict before you run Part 5:** the correct answer is a paraphrase of the
reference. It uses "tools like Wikipedia search" where the reference says "tools
such as Wikipedia search". If we count only exact word matches (the naive approach),
what will the score be for this paraphrase — 0.0, 0.3, or 0.6+?""")

P5_MD = textwrap.dedent(
    """\
---

## Part 5 — Correctness: ROUGE-L and the Ordering Problem

**Correctness** measures whether the generated answer contains the same information
as a reference answer. It requires labeled data — a gold-standard response for each
question.

### Why exact match fails

The simplest correctness metric is Jaccard similarity over word sets:

$$\\text{Jaccard}(a, r) = \\frac{|\\text{tok}(a) \\cap \\text{tok}(r)|}{|\\text{tok}(a) \\cup \\text{tok}(r)|}$$

This treats "tools such as Wikipedia search" and "tools like Wikipedia search" as
partial matches. More importantly, it ignores word order — a shuffled reference
scores identically to the original.

### ROUGE-L: longest common subsequence

ROUGE-L rewards order-preserving coverage. The longest common subsequence (LCS)
between answer $a$ and reference $r$ is the longest sequence of words that appears
in both, in the same relative order, without requiring them to be adjacent.

$$\\text{ROUGE-L}(a, r) = \\frac{|\\text{LCS}(a, r)|}{|r|}$$

This is recall-oriented: we measure what fraction of the reference is covered by
the answer. An answer that covers half the reference words in the correct order
scores 0.5 even if it uses no exact phrases.

**The DP formulation:** $\\text{LCS}(a, r)$ is computed with standard dynamic
programming. Let $L[i][j]$ be the LCS length for $a[:i]$ and $r[:j]$:

$$L[i][j] = \\begin{cases} L[i-1][j-1] + 1 & a[i] = r[j] \\\\ \\max(L[i-1][j],\\; L[i][j-1]) & \\text{otherwise} \\end{cases}$$"""
)

P5_NAIVE = textwrap.dedent(
    """\
# ── Attempt 1: exact word match (Jaccard similarity) ─────────────────────────

def jaccard(answer, reference):
    a_tok = set(answer.lower().split())
    r_tok = set(reference.lower().split())
    if not (a_tok | r_tok):
        return 0.0
    return len(a_tok & r_tok) / len(a_tok | r_tok)

# A correct paraphrase that uses synonyms
paraphrase = "ReAct alternates between reasoning steps and actions, employing tools such as web search during the process."

print("Exact match (Jaccard) on a correct paraphrase:\\n")
print(f"  reference:  {REF!r}")
print(f"  paraphrase: {paraphrase!r}")
print(f"  Jaccard:    {jaccard(paraphrase, REF):.3f}")
print()
print("The paraphrase is factually correct and semantically equivalent.")
print("Jaccard penalises every synonym: 'alternates'/'interleaves', 'employing'/'using', etc.")
print()
print("  -> Jaccard treats synonyms as wrong.  We need order-preserving partial credit.")"""
)

ROUGE_CODE = textwrap.dedent("""\
# ── ROUGE-L from scratch: LCS via dynamic programming ────────────────────────

def lcs_length(a, b):
    # Standard DP: L[i][j] = LCS length for a[:i] and b[:j]
    m, n = len(a), len(b)
    L    = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]


def rouge_l(answer, reference):
    # Recall-oriented: LCS / reference length
    a_tok = answer.lower().split()
    r_tok = reference.lower().split()
    if not r_tok:
        return 0.0
    return lcs_length(a_tok, r_tok) / len(r_tok)


# Score all four failure cases
print(f"Reference: {REF!r}\\n")
print(f"{'Case':<20} {'Jaccard':>8} {'ROUGE-L':>9}  note")
print("-" * 55)

rl_scores  = {}
for label, c in CASES.items():
    j  = jaccard(c["answer"], REF)
    rl = rouge_l(c["answer"], REF)
    rl_scores[label] = rl
    flag = "  <- low coverage" if rl < 0.20 else ""
    print(f"{label:<20} {j:>8.3f} {rl:>9.3f}{flag}")

# Show the paraphrase fix
rl_par = rouge_l(paraphrase, REF)
j_par  = jaccard(paraphrase, REF)
print(f"\\nParaphrase:          {j_par:>8.3f} {rl_par:>9.3f}  (correct paraphrase — ROUGE-L gives credit)")
print()
print("ROUGE-L gives the paraphrase partial credit via shared subsequence;")
print("Jaccard penalises every synonym.  Gap = ", round(rl_par - j_par, 3))

# Score all 5 test queries with the standard RAG bot
print("\\nROUGE-L — standard RAG bot, all 5 queries:\\n")
rl_bot = []
for tc in TEST_CASES:
    result = rag_bot(tc["question"])
    rl     = rouge_l(result["answer"], tc["reference"])
    rl_bot.append(rl)
    print(f"  Q: {tc['question'][:50]:<50}  RL={rl:.3f}")""")

ALL_METRICS = textwrap.dedent(
    """\
# ── All four metrics on the four failure cases — the fingerprint table ─────────
metric_data = {}
for label, c in CASES.items():
    metric_data[label] = {
        "Retrieval Rel.":  retrieval_relevance(Q_THREAD, [c["context"]]),
        "Groundedness":    groundedness(c["answer"],       c["context"]),
        "Answer Rel.":     answer_relevance(Q_THREAD,      c["answer"]),
        "ROUGE-L":         rouge_l(c["answer"],            REF),
    }

df_fp = pd.DataFrame(metric_data).T
print("Metric fingerprint for each failure mode:\\n")
print(df_fp.round(3).to_string())

fig, ax = plt.subplots(figsize=(13, 5))
x      = np.arange(len(df_fp.columns))
width  = 0.18
colors = ["#4c9be8", "#56b356", "#e8934c", "#c65454"]
for i, (label, row) in enumerate(df_fp.iterrows()):
    ax.bar(x + i * width, row.values, width, label=label,
           color=colors[i], alpha=0.85)
ax.axhline(0.4, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(df_fp.columns)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.set_title("All Four Metrics per Failure Mode\\n"
             "Each failure type drops a different metric")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

print("\\nReading the chart:")
print("  Correct           -> all four metrics high")
print("  Coherent wrong    -> Retrieval Rel. drops; others moderate")
print("  Hallucination     -> Groundedness + ROUGE-L drop; Ans Rel. stays high")
print("  Off-topic         -> Answer Rel. + ROUGE-L drop; Groundedness may stay high")"""
)

P5_REFLECT = textwrap.dedent("""\
#### What just happened — and what's missing

The fingerprint table confirms the taxonomy from Part 1: each failure mode has a
distinct pattern across the four metrics. No single metric catches all four failures;
they are orthogonal by design.

ROUGE-L improved on Jaccard by giving the paraphrase partial credit through the LCS
formulation. The remaining gap — ROUGE-L penalising all synonyms — is where
embedding-based correctness metrics (and LLM-as-judge) outperform n-gram metrics.

**The dependency structure matters:** retrieval failure is upstream. When the
retriever returns the wrong document, the generator cannot possibly produce a
correct answer from that context alone. Fixing retrieval relevance is always the
higher-leverage action — it is a necessary (though not sufficient) precondition for
all other metrics to be meaningful.""")

P6_MD = textwrap.dedent("""\
---

## Part 6 — Composite Dashboard: Diagnosing the Bottleneck

A composite score by itself tells you how good the system is. The individual metrics
tell you *where* to invest effort. The right diagnostic view is the per-query
breakdown, not the aggregate.

**Simple composite:** arithmetic mean of all four metrics:

$$\\text{score}(q) = \\frac{1}{4}\\bigl(\\text{RetRel} + \\text{Ground} + \\text{AnsRel} + \\text{ROUGE-L}\\bigr)$$

A low composite score could come from a bad retriever (fix the index or embedding
model), a hallucinating generator (add groundedness checks or constrain generation),
or a topic-drifting generator (improve prompting or fine-tune). Without the breakdown
you can't tell which.""")

DASHBOARD = textwrap.dedent("""\
# ── Per-query evaluation dashboard ────────────────────────────────────────────
dashboard_rows = []
for tc in TEST_CASES:
    result = rag_bot(tc["question"])
    ctx    = " ".join(result["retrieved_docs"])
    row = {
        "Query":         tc["question"][:42] + "...",
        "RetRel":        retrieval_relevance(tc["question"], result["retrieved_docs"]),
        "Groundedness":  groundedness(result["answer"], ctx),
        "AnsRel":        answer_relevance(tc["question"], result["answer"]),
        "ROUGE-L":       rouge_l(result["answer"], tc["reference"]),
    }
    row["Composite"] = float(np.mean([row["RetRel"], row["Groundedness"],
                                      row["AnsRel"], row["ROUGE-L"]]))
    dashboard_rows.append(row)

df_dash = pd.DataFrame(dashboard_rows)
print("RAG Evaluation Dashboard — Standard Hybrid Bot\\n")
print(df_dash.round(3).to_string(index=False))

metric_cols  = ["RetRel", "Groundedness", "AnsRel", "ROUGE-L"]
mean_per     = df_dash[metric_cols].mean()
worst        = mean_per.idxmin()
print(f"\\nMean composite score:  {df_dash['Composite'].mean():.3f}")
print(f"Weakest metric:        {worst}  ({mean_per[worst]:.3f})")
print(f"\\n  -> Fix {worst} first.  It is the biggest drag on the composite.")""")

RADAR = textwrap.dedent("""\
# ── Radar chart: metric fingerprint per query ─────────────────────────────────
metric_cols = ["RetRel", "Groundedness", "AnsRel", "ROUGE-L"]
N      = len(metric_cols)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"polar": True})

cmap = plt.cm.tab10
for i, row in df_dash.iterrows():
    vals = [row[m] for m in metric_cols] + [row[metric_cols[0]]]
    axes[0].plot(angles, vals, "o-", linewidth=1.5, label=f"Q{i+1}", color=cmap(i))
    axes[0].fill(angles, vals, alpha=0.07, color=cmap(i))
axes[0].set_xticks(angles[:-1])
axes[0].set_xticklabels(metric_cols, size=10)
axes[0].set_ylim(0, 1)
axes[0].set_title("Per-Query Metric Fingerprint", pad=20)
axes[0].legend(loc="upper right", bbox_to_anchor=(1.38, 1.15), fontsize=9)

mean_v = [mean_per[m] for m in metric_cols] + [mean_per[metric_cols[0]]]
ideal  = [1.0] * (N + 1)
axes[1].plot(angles, ideal, "k--", linewidth=1, alpha=0.25, label="Ideal")
axes[1].fill(angles, ideal, alpha=0.04, color="gray")
axes[1].plot(angles, mean_v, "o-", linewidth=2.5, color="steelblue", label="Mean")
axes[1].fill(angles, mean_v, alpha=0.18, color="steelblue")
axes[1].set_xticks(angles[:-1])
axes[1].set_xticklabels(metric_cols, size=10)
axes[1].set_ylim(0, 1)
axes[1].set_title("Mean Profile vs Ideal", pad=20)
axes[1].legend(loc="upper right", bbox_to_anchor=(1.38, 1.15))

plt.suptitle("Part 6 — Composite Dashboard", fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print("The dent in the radar relative to the ideal circle marks the weakest")
print("component.  Fix the metric with the largest deficit first.")""")

P6_REFLECT = textwrap.dedent("""\
#### What just happened — and what's missing

The dashboard converts four scalar metrics into a diagnostic signal: not "how good is
the system" but "which component is the bottleneck." The radar chart makes the
fingerprint spatial — a system with a bad retriever has a left-side dent; a
hallucinating generator has a bottom dent.

**The remaining limitation:** every metric in this notebook is a proxy. Each uses
word-level features or sentence-level embeddings to approximate a judgment that
requires reading comprehension. The proxies work well for obvious failures (wrong
retrieval, invented jargon); they break for subtle failures (synonym hallucination,
factual distortion with borrowed vocabulary).

Part 7 shows how the same metric *structure* survives the move to production, with
only the scorer swapped from word overlap to an LLM NLI judge.""")

P7_MD = textwrap.dedent(
    """\
---

## Part 7 — From Proxies to Production: LLM-as-Judge

### Why each proxy eventually breaks

Each proxy we built approximates a harder problem:

| Metric | Proxy | True problem | Why proxy breaks |
| ------ | ----- | ------------ | ---------------- |
| Retrieval Rel. | Embedding cosine | Semantic entailment: does retrieved doc answer q? | False positives from topically adjacent but non-answering docs |
| Groundedness | Token recall | NLI: does context entail each answer claim? | Synonyms and negation use context tokens in wrong senses |
| Answer Relevance | Q-A cosine | Intent match: does the answer address the question's intent? | Hallucinations about the right topic score high |
| ROUGE-L | LCS overlap | Semantic equivalence of assertions | Paraphrases with different vocabulary score low |

### The LLM-as-Judge pattern

Production evaluation replaces the scorer, not the metric structure. Each evaluator
still asks exactly the same question; the question is now posed to a strong LLM with
a structured output schema:

```
Groundedness judge:
  inputs:  FACTS = retrieved context,  ANSWER = system response
  output:  { grounded: bool, explanation: str }
  question: "Are all claims in ANSWER entailed by FACTS?"

Answer Relevance judge:
  inputs:  QUESTION = user query,  ANSWER = system response
  output:  { relevant: bool, explanation: str }
  question: "Does ANSWER address QUESTION?"
```

The LLM judge sees semantic entailment, coreference, and pragmatic implication that
keyword and embedding proxies cannot. The explanation field provides a chain-of-thought
that makes the verdict interpretable and auditable.

### Mapping this notebook to LangSmith evaluators

The playground companion (`playground/af-advanced-ai/D2-rag_evaluation.ipynb`)
implements this pattern end-to-end using LangSmith. The mapping is one-to-one:

| Function in this notebook | LangSmith evaluator | Key comparison |
| ------------------------- | ------------------- | -------------- |
| `retrieval_relevance(q, docs)` | `retrieval_relevance` | query vs `outputs["documents"]` |
| `groundedness(a, ctx)` | `groundedness` | `outputs["answer"]` vs `outputs["documents"]` |
| `answer_relevance(q, a)` | `relevance` | `inputs["question"]` vs `outputs["answer"]` |
| `rouge_l(a, ref)` | `correctness` | `outputs["answer"]` vs `reference_outputs["answer"]` |"""
)

LANGSMITH = textwrap.dedent("""\
# ── Production evaluation pattern (LangSmith) — read-through ─────────────────
#
# This cell shows the LangSmith groundedness judge pattern from
# playground/af-advanced-ai/D2-rag_evaluation.ipynb.
# Requires LANGSMITH_API_KEY and OPENAI_API_KEY.

import os

LANGSMITH_KEY = os.environ.get("LANGSMITH_API_KEY")
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY")

_PATTERN = [
    "# from langsmith import Client",
    "# from langchain_openai import ChatOpenAI",
    "# from typing_extensions import Annotated, TypedDict",
    "#",
    "# class GroundednessGrade(TypedDict):",
    "#     explanation: Annotated[str, ..., 'Reasoning chain']",
    "#     grounded:    Annotated[bool, ..., 'True if answer is entailed by context']",
    "#",
    "# grader = ChatOpenAI(model='gpt-4o-mini', temperature=0)",
    "#          .with_structured_output(GroundednessGrade)",
    "#",
    "# def groundedness_judge(inputs, outputs):",
    "#     ctx   = chr(10).join(d.page_content for d in outputs['documents'])",
    "#     grade = grader.invoke([",
    "#         {'role': 'system',  'content': 'Grade NLI entailment.'},",
    "#         {'role': 'user',    'content': f'FACTS: {ctx}  ANSWER: {outputs[\"answer\"]}'},",
    "#     ])",
    "#     return grade['grounded']",
    "#",
    "# experiment = client.evaluate(",
    "#     rag_bot, data='RAG Test Evaluation',",
    "#     evaluators=[correctness_judge, groundedness_judge, relevance_judge],",
    "# )",
]

if LANGSMITH_KEY and OPENAI_KEY:
    print("API keys found.  Remove the guards above to run live LangSmith evaluation.")
else:
    print("No API keys set.  Production pattern shown as a code reference:\\n")
    print("\\n".join(_PATTERN))

# Summary: proxy means vs what an LLM judge would give
print("\\nProxy metric means on the standard RAG bot (5 queries):")
for name, vals in [
    ("Retrieval Relevance", rr_scores),
    ("Groundedness",        [groundedness(rag_bot(tc["question"])["answer"],
                             " ".join(rag_bot(tc["question"])["retrieved_docs"]))
                            for tc in TEST_CASES]),
    ("Answer Relevance",    ar_bot),
    ("ROUGE-L",             rl_bot),
]:
    print(f"  {name:<22}  {np.mean(vals):.3f}")

print()
print("An LLM judge scores paraphrased correct answers higher than ROUGE-L.")
print("It scores synonym-hallucinations lower than token recall allows.")
print("Use proxies for development-time regression detection;")
print("use LLM judges for evaluation results that inform product decisions.")""")

SUMMARY = textwrap.dedent("""\
---

## Summary — The Complete RAG Evaluation Journey

### Journey completed — roadmap revisited

| Step | Concept | Claim Proved |
| ---- | ------- | ------------ |
| 1 | Four RAG failure modes | All four failure types produce fluent, indistinguishable answers without metrics |
| 2 | Retrieval Quality | Keyword overlap gave 0.3+ for wrong retrieval; embedding cosine separated them by 0.4+ |
| 3 | Groundedness | Token recall scored hallucination below 0.25; adversarial synonym case exposed the proxy limit |
| 4 | Answer Relevance | Off-topic answer separated from correct by 0.25+ cosine; hallucination stayed high (correct) |
| 5 | Correctness — ROUGE-L | Exact match (Jaccard) scored a correct paraphrase near 0; ROUGE-L gave it partial credit via LCS |
| 6 | Composite Dashboard | Each failure mode leaves a different metric fingerprint; composite only tells you magnitude, not cause |
| 7 | LLM-as-Judge | Proxy structure identical to production; only the scorer (word-overlap vs NLI-capable LLM) changes |

### Key insights to keep

**The taxonomy is the tool.** The 2x2 table (retriever quality x generator quality) is not just
a description — it is a debugging algorithm. Low retrieval relevance sends you to the index and
embedding model. Low groundedness sends you to the prompt or generation constraints. Low answer
relevance sends you to the retrieval-generation interface. Low ROUGE-L sends you to factual
coverage.

**Precision and recall for retrieval are complementary, not substitutable.** A retriever that
returns the entire corpus has perfect recall at zero precision. The tension is resolved by tuning
$k$ and the RRF fusion weights together, not by optimising either metric alone.

**Groundedness and answer relevance are reference-free.** They run on live production traffic
without labeling. This is the highest-leverage deployment of evaluation: catch degradation in
real time without waiting for a labeled batch.

**ROUGE-L is recall-oriented by design.** The denominator is $|r|$, the reference length, not
$|a|$. An answer that covers every reference word and then adds extra content scores 1.0. This
is deliberate: the primary failure mode for correctness is omission, not verbosity.

**The proxy boundary is NLI entailment.** Every proxy in this notebook breaks when the generator
uses context vocabulary with wrong polarity, coreference, or pragmatic implication. That is
where LLM-as-judge earns its inference cost.

### Evaluation checklist for a new RAG system

- [ ] Retrieval relevance on a random sample of live queries (no labels needed)
- [ ] Groundedness on generated answers (no labels needed)
- [ ] Answer relevance on generated answers (no labels needed)
- [ ] 50-100 labeled (question, reference answer) pairs for ROUGE-L / correctness
- [ ] Per-query dashboard — identify the weakest metric
- [ ] Trace that metric to retriever or generator
- [ ] Upgrade the weakest proxy to an LLM judge before shipping it as a KPI

---

**Further reading:**
- RAGAS (Es et al. 2023): "Automated Evaluation of Retrieval Augmented Generation"
- TruLens (Snowflake): the faithfulness / answer relevance / context relevance triad
- LangSmith evaluation guide: structured output graders and experiment tracking
- Companion notebook: `playground/af-advanced-ai/D2-rag_evaluation.ipynb`""")


# ── Assemble cells ────────────────────────────────────────────────────────────
cells = [
    md(TITLE),
    code(INSTALL),
    code(IMPORTS),
    code(CORPUS),
    code(RAG_SYSTEM),
    md(P1_MD),
    code(FAILURES),
    md(P1_REFLECT),
    md(P2_MD),
    code(P2_NAIVE),
    code(P2_SEMANTIC),
    code(RR_VIZ),
    code(RR_YT),
    md(P2_REFLECT),
    md(P3_MD),
    md(P3_PREDICT),
    code(GROUND_CODE),
    code(GROUND_VIZ),
    code(GROUND_LIMIT),
    code(GROUND_YT),
    md(P3_REFLECT),
    md(P4_MD),
    md(P4_PREDICT),
    code(ANS_REL),
    code(ANS_YT),
    md(P4_REFLECT),
    md(P5_MD),
    code(P5_NAIVE),
    code(ROUGE_CODE),
    code(ALL_METRICS),
    md(P5_REFLECT),
    md(P6_MD),
    code(DASHBOARD),
    code(RADAR),
    md(P6_REFLECT),
    md(P7_MD),
    code(LANGSMITH),
    md(SUMMARY),
]

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
print(f"written {len(cells)} cells -> {OUT}")
print(f"size:   {OUT.stat().st_size:,} bytes")
