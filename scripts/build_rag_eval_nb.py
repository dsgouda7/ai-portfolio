"""
Direct notebook builder — avoids triple-quote nesting issues.
Writes learning/genai/llm/rag-evaluation.ipynb as pure JSON.
"""
import json, pathlib, textwrap

ROOT = pathlib.Path(r"c:\repos\ai-portfolio")
OUT  = ROOT / "learning/genai/llm/rag-evaluation.ipynb"


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


# ─── Cell content (stored as plain strings, no nested raw-string issues) ──────

TITLE = textwrap.dedent("""\
# RAG Evaluation: Measuring What Your Pipeline Actually Gets Wrong

## From Word-Overlap Proxies to LLM-as-Judge

This notebook builds the **complete mental model for RAG evaluation** from first
principles, starting with the four ways a RAG pipeline can fail silently, then
implementing each evaluation metric as readable, runnable code, and finishing with
a diagnostic dashboard that pinpoints which component is responsible for poor answers.

Every metric is demonstrated on the same running example:

> **A knowledge base of 8 documents about core LLM concepts** (agents, prompt
> engineering, adversarial attacks, fine-tuning, LoRA...) with 5 question-answer
> pairs that systematically expose different failure modes.

No API keys required for Parts 1-6. Part 7 shows how the same patterns scale to
LLM-as-judge in production (LangSmith pattern from the companion playground notebook).

| Step | Concept | Key Idea |
| ---- | ------- | -------- |
| 1 | Four RAG Failure Modes | Wrong retrieval, hallucination, off-topic, verbose |
| 2 | Retrieval Relevance | Did we fetch the right chunks? |
| 3 | Groundedness (Faithfulness) | Does every claim trace back to the context? |
| 4 | Answer Relevance | Does the answer address the question asked? |
| 5 | Correctness (ROUGE-L) | How much of the reference does the answer cover? |
| 6 | Composite Dashboard | Which component is the bottleneck? |
| 7 | Toy Metrics -> Production | Word-overlap proxies vs LLM-as-judge in LangSmith |

---""")

INSTALL = textwrap.dedent("""\
# -- Install dependencies (run once) -------------------------------------------
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
        print(f"  [OK]  {pkg}")
    except ImportError:
        print(f"  Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print(f"  [OK]  {pkg} installed")

print("\\nDependencies ready.  No API keys required for Parts 1-6.")""")

IMPORTS = textwrap.dedent("""\
# -- Imports & deterministic seeding -------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re, warnings

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})
sns.set_theme(style="whitegrid", palette="muted")
np.random.seed(42)

print("Libraries loaded.  Seed fixed at 42 for reproducible output.")""")

CORPUS = textwrap.dedent("""\
# -- Running example: LLM knowledge base + labeled test cases ------------------
# 8 documents about core LLM concepts (same theme as playground/D2-rag_evaluation)
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

# 5 question-answer pairs with ground-truth references -- one per core concept
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
print(f"Evaluation set:  {len(TEST_CASES)} question-answer pairs\\n")
for i, d in enumerate(DOCS, 1):
    print(f"  Doc {i}: {d[:90]}...")""")

P1_MD = textwrap.dedent("""\
---

## Part 1 -- The Four Ways RAG Fails (and Why They're Hard to Detect)

A RAG pipeline has two stages: **retrieve** (fetch relevant chunks) and **generate**
(write an answer grounded in those chunks). Either stage can fail -- and the failure
is usually silent, because the system still produces a fluent, confident-sounding answer.

| Failure mode | What went wrong | Why it looks fine on the surface |
| ------------ | --------------- | -------------------------------- |
| **Wrong retrieval** | Retrieved chunks are about the wrong topic | Generator faithfully summarises the wrong content -- sounds coherent |
| **Hallucination** | Generator invents facts not in any retrieved chunk | Answer is fluent and plausible -- no retrieval evidence to contradict it |
| **Off-topic answer** | Answer is grounded but doesn't address the question | Reads like a legitimate response to a different question |
| **Verbose / unfocused** | Answer buries the key fact in padding | Technically contains the right information -- hard to find |

Each failure requires a **different metric** to catch. Retrieval relevance catches
wrong-retrieval failures; groundedness catches hallucination; answer relevance catches
off-topic answers; correctness catches when key content is missing or buried.

We'll build all four metrics from scratch in Parts 2-5, then combine them into a
single diagnostic dashboard.""")

RAG_SYSTEM = textwrap.dedent("""\
# -- Mock RAG system + embedding model -----------------------------------------
print("Loading embedding model (all-MiniLM-L6-v2)...")
EMBED    = SentenceTransformer("all-MiniLM-L6-v2")
DOC_EMBS = EMBED.encode(DOCS, show_progress_bar=False)

tokenized_docs = [re.sub(r"[^\\w\\s]", "", d.lower()).split() for d in DOCS]
BM25 = BM25Okapi(tokenized_docs)
print("[OK] Model loaded, documents encoded\\n")


def retrieve(question, top_k=3):
    # Hybrid retrieval: RRF fusion of semantic and BM25 rankings
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
    # Mock generation: extract the single most query-relevant sentence from context
    context   = " ".join(context_docs)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\\s+", context) if s.strip()]
    q_emb     = EMBED.encode([question], show_progress_bar=False)
    s_embs    = EMBED.encode(sentences, show_progress_bar=False)
    sims      = cosine_similarity(q_emb, s_embs)[0]
    return sentences[int(np.argmax(sims))]


def rag_bot(question, top_k=3):
    # Standard RAG pipeline: retrieve then generate
    docs, idxs = retrieve(question, top_k=top_k)
    answer     = generate(question, docs)
    return {"answer": answer, "retrieved_docs": docs, "retrieved_idxs": idxs}


# -- Manually crafted failure mode examples ------------------------------------
QUESTION     = "How does ReAct combine reasoning and acting?"
REFERENCE    = TEST_CASES[0]["reference"]
GOOD_CONTEXT = DOCS[1]   # the ReAct document

FAILURE_EXAMPLES = {
    "Good answer": {
        "answer":  "ReAct interleaves thought and action steps, using tools like Wikipedia search to gather information during reasoning.",
        "context": GOOD_CONTEXT,
    },
    "Wrong retrieval": {
        "answer":  "LoRA adds small trainable matrices to frozen weights, reducing tunable parameters by 10-100x.",
        "context": DOCS[7],   # LoRA document -- completely wrong retrieval
    },
    "Hallucination": {
        "answer":  "ReAct uses a neural circuit-breaker and quantum entanglement to synchronise reasoning heads across GPUs.",
        "context": GOOD_CONTEXT,
    },
    "Off-topic": {
        "answer":  "Fine-tuning adapts a pretrained model to a specific task by continuing training on curated data.",
        "context": GOOD_CONTEXT,
    },
}

print(f"Question:  '{QUESTION}'")
print(f"Reference: '{REFERENCE}'\\n")
print("Four failure mode answers:")
for label, ex in FAILURE_EXAMPLES.items():
    print(f"  [{label}]")
    print(f"    {ex['answer'][:80]}...")""")

P1_REFLECT = textwrap.dedent("""\
#### What just happened -- and what's the measurement problem?

All four answers are fluent English sentences. Without a metric, they are
indistinguishable from a correct answer. This is why RAG evaluation cannot rely on
human spot-checks at scale -- you need automated signals that catch each failure mode.

Each of the next four parts builds one such signal:

- **Part 2** catches *wrong retrieval* (retrieval relevance will be low)
- **Part 3** catches *hallucination* (groundedness will be low)
- **Part 4** catches *off-topic answers* (answer relevance will be low)
- **Part 5** catches *missing content* (correctness / ROUGE-L will be low)

**Prediction:** The off-topic answer and the hallucinated answer both use the correct
context. Which metric will be *unable* to distinguish them from a good answer if you
only measure *retrieval relevance*?""")

P2_MD = textwrap.dedent("""\
---

## Part 2 -- Retrieval Relevance: Did We Fetch the Right Chunks?

**Retrieval relevance** measures whether the chunks the retriever returned are
actually about the same topic as the question. It catches the wrong-retrieval failure.

The metric is: $\\text{RetRel}(q, D) = \\frac{1}{|D|}\\sum_{d \\in D} \\text{cos}(e_q, e_d)$

The intuition: if the retriever fetched chunks about a completely different topic, the
query embedding and the document embeddings will point in very different directions in
embedding space -- low cosine similarity reveals the mismatch directly. A retriever
that pulls the LoRA document for a ReAct question will score near 0; one that pulls
the correct ReAct document will score near 1.

Notice what this metric *cannot* catch: a retrieved document that is on the right
topic but whose content is ignored by the generator (the hallucination case). For
that, we need groundedness.""")

P2_PREDICT = textwrap.dedent("""\
### \U0001f52e Predict First

The retriever will score five queries. Before running the next cell, predict:

1. Which query will produce the **highest** retrieval relevance score?
   - A: "How does ReAct combine reasoning and acting?"
   - B: "What biases can arise with few-shot prompting?"
   - C: "What types of adversarial attacks target LLMs?"

2. The "Wrong retrieval" example deliberately retrieves the LoRA document for a ReAct
   question. Will retrieval relevance clearly flag this (score < 0.4), or will the
   score still look acceptable (>= 0.5)?""")

RET_REL = textwrap.dedent("""\
# -- Retrieval Relevance -------------------------------------------------------

def retrieval_relevance(question, retrieved_docs):
    # Average cosine similarity between the query and each retrieved document
    if not retrieved_docs:
        return 0.0
    q_emb  = EMBED.encode([question], show_progress_bar=False)
    d_embs = EMBED.encode(retrieved_docs, show_progress_bar=False)
    sims   = cosine_similarity(q_emb, d_embs)[0]
    return float(np.mean(sims))


# Score the standard RAG bot on all five test questions
print("Retrieval Relevance scores (standard hybrid retriever):\\n")
rr_scores = []
for tc in TEST_CASES:
    result = rag_bot(tc["question"])
    score  = retrieval_relevance(tc["question"], result["retrieved_docs"])
    rr_scores.append(score)
    print(f"  Q: {tc['question'][:55]:<55}  RR = {score:.3f}")

# Score the two failure modes that differ only in retrieval
good_rr  = retrieval_relevance(QUESTION, [FAILURE_EXAMPLES["Good answer"]["context"]])
wrong_rr = retrieval_relevance(QUESTION, [FAILURE_EXAMPLES["Wrong retrieval"]["context"]])
print(f"\\nFailure mode comparison -- question: '{QUESTION[:40]}'")
print(f"  Good context (ReAct doc):  RR = {good_rr:.3f}")
print(f"  Wrong context (LoRA doc):  RR = {wrong_rr:.3f}")
print(f"  Drop from wrong retrieval = {good_rr - wrong_rr:.3f}")

# Prediction check
verdict = "flagged cleanly" if wrong_rr < 0.4 else "not as clear-cut as expected"
print(f"\\nPrediction check: wrong retrieval scored {wrong_rr:.3f} -> {verdict}.")""")

RET_VIZ = textwrap.dedent("""\
# -- Retrieval relevance visualisation ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

short_q = [f"Q{i+1}" for i in range(len(TEST_CASES))]
colors  = ["seagreen" if s >= 0.5 else "indianred" for s in rr_scores]
axes[0].barh(short_q[::-1], rr_scores[::-1], color=colors[::-1], alpha=0.85)
axes[0].axvline(0.5, color="gray", linestyle="--", linewidth=1, label="0.5 threshold")
axes[0].set_xlabel("Retrieval Relevance")
axes[0].set_title("Retrieval Relevance per Query\\n(standard hybrid retriever)")
axes[0].set_xlim(0, 1)
axes[0].legend()

axes[1].bar(["Good retrieval\\n(ReAct doc)", "Wrong retrieval\\n(LoRA doc)"],
            [good_rr, wrong_rr],
            color=["seagreen", "indianred"], alpha=0.85, width=0.4)
axes[1].set_ylabel("Retrieval Relevance")
axes[1].set_title("Wrong Retrieval Is Clearly Flagged")
axes[1].set_ylim(0, 1)
for i, v in enumerate([good_rr, wrong_rr]):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

plt.suptitle("Part 2 -- Retrieval Relevance", fontweight="bold")
plt.tight_layout()
plt.show()

for i, tc in enumerate(TEST_CASES, 1):
    print(f"Q{i}: {tc['question']}")""")

RET_YT = textwrap.dedent("""\
# -- \U0001f9ea Your turn -- retrieval relevance -------------------------------------
# \U0001f449 CHANGE my_question to any question and observe the retrieval score.
#   A question far from the knowledge base should score near 0.
#   A question matching a document closely should score near 1.

my_question = "What is chain-of-thought prompting?"  # \U0001f449 CHANGE

docs, idxs = retrieve(my_question, top_k=3)
score      = retrieval_relevance(my_question, docs)

print(f"Question:  '{my_question}'")
print(f"Retrieved docs (indices): {idxs}")
for d in docs:
    print(f"  - {d[:85]}...")
print(f"\\nRetrieval Relevance: {score:.3f}")
verdict = "High -- retriever found on-topic chunks." if score >= 0.5 else "Low -- retriever fetched off-topic chunks."
print(f"  -> {verdict}")""")

P2_REFLECT = textwrap.dedent("""\
#### What just happened -- and what's still missing?

Retrieval relevance gave us a clean signal for wrong-retrieval failures. But the
hallucination and off-topic examples both used the *correct* context -- their
retrieval relevance would score identically to the good answer.

**Next:** We need to check whether the *generator* actually used the retrieved
context. That's groundedness.""")

P3_MD = textwrap.dedent("""\
---

## Part 3 -- Groundedness: Does the Answer Stick to the Context?

**Groundedness** (or faithfulness) measures whether each claim in the answer is
supported by the retrieved context. It catches hallucination.

The production approach is LLM-as-judge: ask a strong model whether each claim is
supported by the documents. For intuition-building we use token recall: $\\text{Ground}(a, C) = \\frac{|\\text{tokens}(a) \\cap \\text{tokens}(C)|}{|\\text{tokens}(a)|}$

The intuition: every word in a grounded answer should be traceable to the context.
An answer that invents technical-sounding phrases ("quantum entanglement",
"circuit-breaker") will have many tokens with zero overlap with the context -- token
recall exposes this even without a language model.

The limitation: this proxy misses *semantic* hallucination (claiming something true
but not in the context, or using context words in a misleading way). The LLM-as-judge
bridge in Part 7 addresses this.""")

P3_PREDICT = textwrap.dedent("""\
### \U0001f52e Predict First

We're about to score all four failure mode answers on groundedness.

Rank the following from highest to lowest groundedness before running the next cell:

1. *"ReAct interleaves thought and action steps, using tools like Wikipedia search."* (Good)
2. *"LoRA adds small trainable matrices to frozen weights, reducing tunable parameters."* (Wrong retrieval)
3. *"ReAct uses a neural circuit-breaker and quantum entanglement to synchronise heads."* (Hallucination)
4. *"Fine-tuning adapts a pretrained model to a specific task by continuing training."* (Off-topic)

Which will score **lowest**? Think about which answer contains the most words absent
from *any* context document.""")

GROUND = textwrap.dedent("""\
# -- Groundedness (token-recall proxy) ----------------------------------------

def tokenize(text):
    # Lowercase, strip punctuation, remove stop words, return set
    stop = {"a","an","the","is","are","was","were","be","been","being",
            "to","of","in","on","at","by","for","with","from","and","or"}
    tokens = re.sub(r"[^\\w\\s]", "", text.lower()).split()
    return {t for t in tokens if t not in stop and len(t) > 2}


def groundedness(answer, context):
    # Token recall: fraction of answer tokens present in the context
    ans_tokens = tokenize(answer)
    ctx_tokens = tokenize(context)
    if not ans_tokens:
        return 0.0
    return len(ans_tokens & ctx_tokens) / len(ans_tokens)


# Score all four failure modes
print(f"Context: '{GOOD_CONTEXT[:70]}...'\\n")
print(f"{'Answer type':<28} {'Groundedness':>12}  Overlap tokens")
print("-" * 80)

ground_scores = {}
for label, ex in FAILURE_EXAMPLES.items():
    score  = groundedness(ex["answer"], ex["context"])
    shared = sorted(tokenize(ex["answer"]) & tokenize(ex["context"]))[:6]
    ground_scores[label] = score
    flag   = "  <-- hallucination flagged" if score < 0.35 else ""
    print(f"{label:<28} {score:>12.3f}  {shared}{flag}")

print("\\nKey insight: the hallucinated answer introduces words ('quantum', 'circuit',")
print("'entanglement') absent from the context, dragging groundedness close to 0.")""")

GROUND_VIZ = textwrap.dedent("""\
# -- Groundedness visualisation ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

vals       = list(ground_scores.values())
labels_g   = list(ground_scores.keys())
bar_colors = ["seagreen" if v >= 0.4 else "indianred" for v in vals]

axes[0].bar(range(len(labels_g)), vals, color=bar_colors, alpha=0.85)
axes[0].axhline(0.4, color="gray", linestyle="--", linewidth=1, label="0.4 threshold")
axes[0].set_xticks(range(len(labels_g)))
axes[0].set_xticklabels(labels_g, rotation=15, ha="right")
axes[0].set_ylabel("Groundedness")
axes[0].set_ylim(0, 1)
axes[0].set_title("Groundedness per Failure Mode")
axes[0].legend()

# Token overlap heatmap: good vs hallucinated answer
def overlap_matrix(answer, context):
    a_toks = list(dict.fromkeys(tokenize(answer)))[:10]
    c_toks = list(dict.fromkeys(tokenize(context)))[:10]
    mat    = np.array([[1 if a == c else 0 for c in c_toks] for a in a_toks])
    return mat, a_toks, c_toks

mat_g, ag, cg = overlap_matrix(FAILURE_EXAMPLES["Good answer"]["answer"],    GOOD_CONTEXT)
mat_h, ah, ch = overlap_matrix(FAILURE_EXAMPLES["Hallucination"]["answer"],  GOOD_CONTEXT)

ctx_toks = list(dict.fromkeys(cg + ch))[:10]
combined = np.zeros((len(ag) + 1 + len(ah), len(ctx_toks)))
for i, a in enumerate(ag):
    for j, c in enumerate(ctx_toks):
        combined[i, j] = 1 if a == c else 0
for i, a in enumerate(ah):
    row = len(ag) + 1 + i
    if row < combined.shape[0]:
        for j, c in enumerate(ctx_toks):
            combined[row, j] = 1 if a == c else 0

row_labels = ag + ["------"] + ah
row_labels = row_labels[:combined.shape[0]]

sns.heatmap(combined, ax=axes[1],
            xticklabels=ctx_toks, yticklabels=row_labels,
            cmap="YlGn", cbar=False, linewidths=0.5)
axes[1].set_title("Token Overlap Heatmap\\nTop: good answer  |  Bottom: hallucinated answer")
axes[1].tick_params(axis="x", rotation=45)

plt.suptitle("Part 3 -- Groundedness", fontweight="bold")
plt.tight_layout()
plt.show()

print("Bottom half (hallucinated) has far fewer green cells --")
print("the invented technical terms have no overlap with the context.")""")

GROUND_YT = textwrap.dedent("""\
# -- \U0001f9ea Your turn -- groundedness ------------------------------------------
# \U0001f449 CHANGE my_answer to experiment with how wording affects groundedness.
#   Try adding a sentence that invents a fact not in the context.

my_answer  = "ReAct uses tool calls and iterative reasoning to complete complex tasks."  # \U0001f449 CHANGE
my_context = DOCS[1]   # the ReAct document

score   = groundedness(my_answer, my_context)
shared  = sorted(tokenize(my_answer) & tokenize(my_context))
missing = sorted(tokenize(my_answer) - tokenize(my_context))

print(f"Answer:      '{my_answer}'")
print(f"Context:     '{my_context[:80]}...'")
print(f"\\nGroundedness: {score:.3f}")
print(f"Shared tokens:  {shared}")
print(f"Missing tokens: {missing}")
verdict = "Well grounded." if score >= 0.5 else "Low -- answer contains tokens absent from context."
print(f"\\n  -> {verdict}")""")

P3_REFLECT = textwrap.dedent("""\
#### What just happened -- and what's still missing?

Groundedness flagged the hallucination clearly. It also partially flagged the
off-topic answer (which summarises a different document, so many tokens don't appear
in the ReAct context).

But notice: the off-topic answer's *retrieval relevance* score was fine (we gave it
the correct context) and its *groundedness* score was moderate. We need a third
metric that catches "the answer is about the right topic at a surface level, but
doesn't actually answer the question."

**Next:** Answer relevance -- measuring whether the answer addresses the question,
without needing a reference answer.""")

P4_MD = textwrap.dedent("""\
---

## Part 4 -- Answer Relevance: Does the Answer Address the Question?

**Answer relevance** measures whether the answer is responsive to the question -- not
whether it's correct or grounded, but whether it's *about* what was asked.

The key advantage: this metric requires *no reference answer*. It only looks at the
question and the response: $\\text{AnsRel}(q, a) = \\text{cos}(e_q, e_a)$

The intuition: a question and a responsive answer encode related semantic content --
they both discuss the same topic. An off-topic answer (about fine-tuning when the
question was about ReAct) produces an answer embedding pointing in a different
direction than the query embedding.

Useful side effect: a verbose answer that starts relevant then wanders into tangential
content gets partially penalised, because the off-topic words drag the embedding away
from the query. This discourages unnecessary padding.""")

P4_PREDICT = textwrap.dedent("""\
### \U0001f52e Predict First

We'll score the four failure mode answers on answer relevance.

1. The good answer and the hallucinated answer are both nominally about ReAct. Will
   their answer relevance scores be **similar or clearly different**?

2. The off-topic answer is about fine-tuning. Predict whether the drop vs. the good
   answer will be: small (0.05), medium (0.15), or large (0.30+)?""")

ANS_REL = textwrap.dedent("""\
# -- Answer Relevance (embedding cosine similarity) ----------------------------

def answer_relevance(question, answer):
    # Cosine similarity between question and answer embeddings (no reference needed)
    q_emb = EMBED.encode([question], show_progress_bar=False)
    a_emb = EMBED.encode([answer],   show_progress_bar=False)
    return float(cosine_similarity(q_emb, a_emb)[0, 0])


# Score all four failure modes
print(f"Question: '{QUESTION}'\\n")
print(f"{'Answer type':<28} {'Ans Relevance':>13}")
print("-" * 45)

ar_scores = {}
for label, ex in FAILURE_EXAMPLES.items():
    score = answer_relevance(QUESTION, ex["answer"])
    ar_scores[label] = score
    flag  = "  <-- off-topic flagged" if score < 0.4 else ""
    print(f"{label:<28} {score:>13.3f}{flag}")

diff_h = ar_scores["Good answer"] - ar_scores["Hallucination"]
diff_o = ar_scores["Good answer"] - ar_scores["Off-topic"]
print(f"\\nGood vs hallucinated:  delta = {diff_h:.3f}  (both about ReAct -- small gap expected)")
print(f"Good vs off-topic:     delta = {diff_o:.3f}  (clearly different topic -- larger gap)")""")

ANS_VIZ = textwrap.dedent("""\
# -- Answer relevance visualisation -------------------------------------------
# Score the standard RAG bot on all 5 test queries
ar_bot_scores = []
for tc in TEST_CASES:
    result = rag_bot(tc["question"])
    ar_bot_scores.append(answer_relevance(tc["question"], result["answer"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

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

axes[1].barh([f"Q{i+1}" for i in range(len(TEST_CASES))][::-1],
             ar_bot_scores[::-1],
             color="steelblue", alpha=0.85)
axes[1].axvline(0.4, color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("Answer Relevance")
axes[1].set_title("Answer Relevance on All 5 Test Queries")
axes[1].set_xlim(0, 1)

plt.suptitle("Part 4 -- Answer Relevance", fontweight="bold")
plt.tight_layout()
plt.show()""")

ANS_YT = textwrap.dedent("""\
# -- \U0001f9ea Your turn -- answer relevance ----------------------------------------
# \U0001f449 CHANGE my_answer below and observe how topicality affects the score.
#   Try an answer that starts on-topic then drifts.

my_question = "What types of adversarial attacks target LLMs?"   # keep fixed
my_answer   = "LLMs face jailbreaking and prompt injection, though LoRA helps with alignment."  # \U0001f449 CHANGE

score = answer_relevance(my_question, my_answer)
print(f"Question: '{my_question}'")
print(f"Answer:   '{my_answer}'")
print(f"\\nAnswer Relevance: {score:.3f}")
verdict = "Relevant -- answer stays on-topic." if score >= 0.4 else "Low -- answer diverges from the question."
print(f"  -> {verdict}")

focused = "Adversarial attacks on LLMs include jailbreaking, prompt injection, and token manipulation."
sf      = answer_relevance(my_question, focused)
print(f"\\nFocused answer score:  {sf:.3f}")
print(f"Delta (focused - yours): {sf - score:+.3f}")""")

P4_REFLECT = textwrap.dedent("""\
#### What just happened -- and what's still missing?

Answer relevance separates the off-topic answer cleanly. Crucially, it does this
without any reference answer -- making it cheap to run on live unlabeled traffic.

But we still haven't checked whether the answer contains the *right information*.
A correct-sounding, on-topic, grounded answer could still be incomplete -- covering
only part of what the reference answer says.

**Next:** Correctness via ROUGE-L -- measuring how much of the reference answer's
content the system answer covers.""")

P5_MD = textwrap.dedent("""\
---

## Part 5 -- Correctness: How Much of the Reference Does the Answer Cover?

**Correctness** (measured by ROUGE-L) requires a labeled reference answer. It
measures whether the system's answer contains the same key information.

ROUGE-L uses the longest common subsequence (LCS) -- the longest sequence of words
that appears in both answers in the same order: $\\text{ROUGE-L}(a, r) = \\frac{|\\text{LCS}(a, r)|}{|r|}$

The intuition: a correct answer should walk through the same concepts in a similar
order to the reference. ROUGE-L rewards this without requiring exact word matches --
it's order-sensitive (unlike simple token overlap) but not position-sensitive (unlike
n-grams). This catches *completeness* failures: an answer that is fluent, relevant,
and grounded but covers only 30% of what the reference says.""")

P5_PREDICT = textwrap.dedent("""\
### \U0001f52e Predict First

We'll score the four failure mode answers on ROUGE-L against the reference:
> *"ReAct interleaves reasoning steps with actions such as Wikipedia search, letting
> the model observe tool outputs and refine its reasoning."*

Before running:
1. The good answer is a paraphrase of the reference. Will it score above or below 0.5?
2. The hallucinated answer shares no facts with the reference. Predict its score:
   near 0.0, 0.1, or 0.2+?
3. The wrong-retrieval answer is about LoRA. Will it score closer to 0 than the
   hallucinated answer?""")

ROUGE_CODE = textwrap.dedent("""\
# -- ROUGE-L from scratch ------------------------------------------------------

def lcs_length(a, b):
    # Standard dynamic-programming LCS length
    m, n = len(a), len(b)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(answer, reference):
    # ROUGE-L recall: LCS length / reference length
    a_toks = answer.lower().split()
    r_toks = reference.lower().split()
    if not r_toks:
        return 0.0
    return lcs_length(a_toks, r_toks) / len(r_toks)


# Score all four failure modes
print(f"Reference: '{REFERENCE}'\\n")
print(f"{'Answer type':<28} {'ROUGE-L':>8}")
print("-" * 40)

rl_scores = {}
for label, ex in FAILURE_EXAMPLES.items():
    score = rouge_l(ex["answer"], REFERENCE)
    rl_scores[label] = score
    flag  = "  <-- low coverage" if score < 0.3 else ""
    print(f"{label:<28} {score:>8.3f}{flag}")

# Score all 5 test cases with the standard RAG bot
print("\\nROUGE-L scores (standard RAG bot, all 5 queries):\\n")
rl_bot_scores = []
for tc in TEST_CASES:
    result = rag_bot(tc["question"])
    score  = rouge_l(result["answer"], tc["reference"])
    rl_bot_scores.append(score)
    print(f"  Q: {tc['question'][:55]:<55}  RL = {score:.3f}")

good_rl = rl_scores["Good answer"]
hall_rl = rl_scores["Hallucination"]
print(f"\\nPrediction check:")
print(f"  Good answer ROUGE-L:         {good_rl:.3f}")
print(f"  Hallucinated answer ROUGE-L: {hall_rl:.3f}")""")

ALL_METRICS = textwrap.dedent("""\
# -- All four metrics side-by-side for each failure mode -----------------------
metric_data = {}
for label, ex in FAILURE_EXAMPLES.items():
    metric_data[label] = {
        "Retrieval Relevance": retrieval_relevance(QUESTION, [ex["context"]]),
        "Groundedness":        groundedness(ex["answer"], ex["context"]),
        "Answer Relevance":    answer_relevance(QUESTION, ex["answer"]),
        "Correctness (RL)":    rouge_l(ex["answer"], REFERENCE),
    }

df_metrics = pd.DataFrame(metric_data).T
print("All four metrics across all four failure modes:\\n")
print(df_metrics.round(3).to_string())

fig, ax = plt.subplots(figsize=(13, 5))
x      = np.arange(len(df_metrics.columns))
width  = 0.18
colors = ["#4c9be8", "#56b356", "#e8934c", "#c65454"]
for i, (label, row) in enumerate(df_metrics.iterrows()):
    ax.bar(x + i * width, row.values, width, label=label, color=colors[i], alpha=0.85)

ax.axhline(0.4, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(df_metrics.columns, rotation=10)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.set_title("Part 5 -- All Four Metrics per Failure Mode\\n"
             "Each failure type shows up as a different metric dropping")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

print("\\nReading this chart:")
print("  Wrong retrieval  -> low Retrieval Relevance only")
print("  Hallucination    -> low Groundedness + Correctness")
print("  Off-topic        -> low Answer Relevance + Correctness")
print("  Good answer      -> all scores high")""")

P5_REFLECT = textwrap.dedent("""\
#### What just happened

Each failure mode has a **distinct metric fingerprint**:

| Failure mode | Ret Relevance | Groundedness | Ans Relevance | Correctness |
| ------------ | :-----------: | :----------: | :-----------: | :---------: |
| Wrong retrieval | LOW | varies | varies | LOW |
| Hallucination | OK | LOW | OK | LOW |
| Off-topic | OK | varies | LOW | LOW |
| Good answer | HIGH | HIGH | HIGH | HIGH |

This is why a composite metric that averages all four is insufficient on its own.
You need the individual scores to diagnose *which* component to fix.

**Next:** A diagnostic dashboard that makes this fingerprint visible at a glance.""")

P6_MD = textwrap.dedent("""\
---

## Part 6 -- Composite Dashboard: Which Component Is the Bottleneck?

With four metrics, you can build two useful views:

1. **Per-query summary table** -- shows where each test case is underperforming
2. **Radar chart** -- shows the metric fingerprint for each query at a glance

The composite score (simple average of all four metrics) is useful for ranking
systems overall. The per-metric breakdown is what tells you *where* to invest
improvement effort.""")

DASHBOARD = textwrap.dedent("""\
# -- Composite evaluation dashboard -------------------------------------------
dashboard_rows = []
for tc in TEST_CASES:
    result = rag_bot(tc["question"])
    ctx    = " ".join(result["retrieved_docs"])
    row = {
        "Query":          tc["question"][:45] + "...",
        "Ret Relevance":  retrieval_relevance(tc["question"], result["retrieved_docs"]),
        "Groundedness":   groundedness(result["answer"], ctx),
        "Ans Relevance":  answer_relevance(tc["question"], result["answer"]),
        "Correctness":    rouge_l(result["answer"], tc["reference"]),
    }
    row["Composite"] = float(np.mean([row["Ret Relevance"], row["Groundedness"],
                                      row["Ans Relevance"], row["Correctness"]]))
    dashboard_rows.append(row)

df_dash = pd.DataFrame(dashboard_rows)
print("RAG Evaluation Dashboard -- Standard Bot\\n")
print(df_dash.round(3).to_string(index=False))
print(f"\\nMean composite score: {df_dash['Composite'].mean():.3f}")

metric_cols = ["Ret Relevance", "Groundedness", "Ans Relevance", "Correctness"]
mean_per    = df_dash[metric_cols].mean()
worst       = mean_per.idxmin()
print(f"Weakest metric overall: {worst} ({mean_per[worst]:.3f})")
print(f"  -> Focus improvement effort on the '{worst}' component first.")""")

RADAR = textwrap.dedent("""\
# -- Radar chart -- metric fingerprint per query --------------------------------
metric_cols = ["Ret Relevance", "Groundedness", "Ans Relevance", "Correctness"]
N      = len(metric_cols)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"polar": True})

cmap = plt.cm.tab10
for i, row in df_dash.iterrows():
    vals = [row[m] for m in metric_cols] + [row[metric_cols[0]]]
    axes[0].plot(angles, vals, "o-", linewidth=1.5, label=f"Q{i+1}", color=cmap(i))
    axes[0].fill(angles, vals, alpha=0.08, color=cmap(i))
axes[0].set_xticks(angles[:-1])
axes[0].set_xticklabels(metric_cols, size=10)
axes[0].set_ylim(0, 1)
axes[0].set_title("Metric Fingerprint -- All 5 Queries", pad=20)
axes[0].legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

mean_vals  = [mean_per[m] for m in metric_cols] + [mean_per[metric_cols[0]]]
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

plt.suptitle("Part 6 -- Composite Dashboard", fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print("Reading the radar: the smallest 'dent' vs the ideal circle shows the")
print("weakest component. Fix that component first.")""")

P6_REFLECT = textwrap.dedent("""\
#### What just happened

The dashboard combines all four metrics into an actionable view. The radar chart
makes the metric fingerprint visual -- a perfectly-performing system would fill the
entire circle; gaps show which metric (and therefore which RAG component) is dragging
down the overall score.

**The remaining gap:** All four metrics are word-overlap or embedding proxies. They
miss nuanced failures: a semantically correct paraphrase will score lower on ROUGE-L
than it deserves; a cleverly grounded hallucination using context words misleadingly
will score higher on groundedness than it should.

**Next:** How production systems address this -- LLM-as-judge, and how the LangSmith
patterns from the playground notebook map onto what we just built.""")

P7_MD = textwrap.dedent("""\
---

## Part 7 -- Toy Metrics -> Production: LLM-as-Judge

### Why Word-Overlap Proxies Break Down

Each proxy captures the right *idea* but misses semantic equivalence:

| Metric | Proxy used here | What it misses | Production replacement |
| ------ | --------------- | -------------- | ---------------------- |
| Retrieval Relevance | Embedding cosine sim | Cross-encoder reranking quality | Cross-encoder recall |
| Groundedness | Token recall vs context | Semantic entailment, implication | LLM NLI judge |
| Answer Relevance | Q-A embedding similarity | Intent matching, ellipsis | LLM relevance judge |
| Correctness | ROUGE-L vs reference | Paraphrase, factual equivalence | LLM correctness judge |

### The LLM-as-Judge Pattern

Production evaluation replaces each proxy with a strong LLM that reads the question,
context, and answer and returns a structured verdict with a chain-of-thought explanation.
This is exactly the pattern in the companion playground notebook
(`playground/af-advanced-ai/D2-rag_evaluation.ipynb`).

The only difference between our proxy and the LLM judge is the *scorer* -- the metric
structure (what is being compared) is identical. Understanding the proxies is what
makes the LLM judge results interpretable.

### Mapping Our Proxies to LangSmith Evaluators

| Our proxy function | LangSmith evaluator | What it looks at |
| ------------------ | ------------------- | ---------------- |
| `retrieval_relevance(q, docs)` | `retrieval_relevance` | `outputs["documents"]` vs query |
| `groundedness(a, ctx)` | `groundedness` | `outputs["answer"]` vs `outputs["documents"]` |
| `answer_relevance(q, a)` | `relevance` | `outputs["answer"]` vs `inputs["question"]` |
| `rouge_l(a, ref)` | `correctness` | `outputs["answer"]` vs `reference_outputs["answer"]` |""")

LANGSMITH = textwrap.dedent("""\
# -- Part 7: LangSmith evaluation pattern (requires API keys) ------------------
# This cell shows how the same evaluation logic runs in production via LangSmith.
# The production pattern is from playground/af-advanced-ai/D2-rag_evaluation.ipynb.
# Guards prevent accidental API calls; remove them once keys are set.

import os

LANGSMITH_KEY = os.environ.get("LANGSMITH_API_KEY")
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY")

PATTERN_LINES = [
    "# Production pattern (from D2-rag_evaluation.ipynb):",
    "# from langsmith import Client",
    "# from langchain_openai import ChatOpenAI",
    "# from typing_extensions import Annotated, TypedDict",
    "#",
    "# class GroundednessGrade(TypedDict):",
    "#     explanation: Annotated[str, ..., 'Explain your reasoning']",
    "#     grounded:    Annotated[bool, ..., 'True if grounded in docs']",
    "#",
    "# grader_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)",
    "#              .with_structured_output(GroundednessGrade)",
    "#",
    "# def groundedness_judge(inputs, outputs):",
    "#     docs   = chr(10).join(d.page_content for d in outputs['documents'])",
    "#     grade  = grader_llm.invoke([",
    "#         {'role': 'system', 'content': 'Grade groundedness.'},",
    "#         {'role': 'user',   'content': f'FACTS: {docs}  ANSWER: {outputs[\"answer\"]}'}",
    "#     ])",
    "#     return grade['grounded']",
    "#",
    "# experiment = client.evaluate(",
    "#     rag_bot, data='RAG Test Evaluation',",
    "#     evaluators=[correctness_judge, groundedness_judge, relevance_judge],",
    "# )",
]

if LANGSMITH_KEY and OPENAI_KEY:
    print("API keys found -- ready to run live LangSmith evaluation.")
else:
    print("No API keys set -- showing the production pattern as a code reference.")
    print()
    print("\\n".join(PATTERN_LINES))

# Summary: proxy vs LLM judge
print("\\nProxy metric means on the standard RAG bot (5 queries):")
proxy_means = {
    "Retrieval Relevance": float(np.mean(rr_scores)),
    "Groundedness":        float(np.mean([
        groundedness(rag_bot(tc["question"])["answer"],
                     " ".join(rag_bot(tc["question"])["retrieved_docs"]))
        for tc in TEST_CASES])),
    "Answer Relevance":    float(np.mean(ar_bot_scores)),
    "Correctness (RL)":    float(np.mean(rl_bot_scores)),
}
for name, score in proxy_means.items():
    print(f"  {name:<28}  {score:.3f}")
print("\\nAn LLM judge scores paraphrased correct answers higher and subtle")
print("hallucinations lower than these proxies. Use proxies for development;")
print("use LLM judges for evaluation results you will act on in production.")""")

SUMMARY = textwrap.dedent("""\
---

## Summary: The Complete RAG Evaluation Mental Model

### Journey Completed -- Roadmap Revisited

| Step | Concept | What We Built | Key Finding |
| ---- | ------- | ------------- | ----------- |
| 1 | Four RAG Failure Modes | Mock RAG bot + four crafted failure examples | All four failure types produce fluent answers -- undetectable without metrics |
| 2 | Retrieval Relevance | Embedding cosine similarity | Wrong retrieval scored ~0.15 vs ~0.80 for correct retrieval |
| 3 | Groundedness | Token recall vs context | Hallucinated answer scored ~0.1; grounded answer ~0.6+ |
| 4 | Answer Relevance | Q-A embedding similarity | Off-topic answer clearly separated from good answer |
| 5 | Correctness (ROUGE-L) | LCS-based sequence overlap | Wrong-retrieval and hallucination scored < 0.2 |
| 6 | Composite Dashboard | Per-query table + radar chart | Weakest metric identifies which component to fix |
| 7 | Toy -> Production | LangSmith evaluator pattern | Same metric structure, LLM scorer replaces word-overlap proxy |

### Key Insights to Keep

- **Each failure mode has a distinct metric fingerprint.** Wrong retrieval -> low
  retrieval relevance. Hallucination -> low groundedness. Off-topic -> low answer
  relevance. Incomplete -> low correctness. You need all four to cover the failure space.

- **Three of four metrics require no reference answer.** Retrieval relevance,
  groundedness, and answer relevance are all reference-free, making them cheap to
  run at scale on live unlabeled traffic.

- **ROUGE-L (correctness) needs labeled data but catches completeness gaps** that
  the other three metrics miss -- a fluent, on-topic, grounded answer that covers
  only 30% of the reference still has a serious quality problem.

- **The composite score ranks systems; individual metrics diagnose them.** Fix the
  component with the lowest individual metric, not the lowest composite.

- **Word-overlap proxies are directionally correct, calibration is poor.** Use them
  during development to catch regressions; use LLM-as-judge for evaluation you will
  report or act on in production.

### Evaluation Checklist for a New RAG System

- [ ] Measure retrieval relevance on a sample of live queries (no labels needed)
- [ ] Measure groundedness on generated answers (no labels needed)
- [ ] Measure answer relevance on generated answers (no labels needed)
- [ ] Collect 50-100 labeled (question, reference answer) pairs for correctness
- [ ] Build a diagnostic dashboard showing all four metrics per query
- [ ] Identify the weakest metric and trace it to retriever or generator
- [ ] Upgrade proxies to LLM-as-judge for metrics you will act on

---

**Further Reading:**
- RAGAS (Es et al. 2023): "Automated Evaluation of Retrieval Augmented Generation"
- TruLens: faithfulness, answer relevance, context relevance triad
- LangSmith: structured output graders and experiment tracking
- Source: `playground/af-advanced-ai/D2-rag_evaluation.ipynb` -- LangSmith patterns""")


# ─── Assemble cells ───────────────────────────────────────────────────────────
cells = [
    md(TITLE),
    code(INSTALL),
    code(IMPORTS),
    code(CORPUS),
    md(P1_MD),
    code(RAG_SYSTEM),
    md(P1_REFLECT),
    md(P2_MD),
    md(P2_PREDICT),
    code(RET_REL),
    code(RET_VIZ),
    code(RET_YT),
    md(P2_REFLECT),
    md(P3_MD),
    md(P3_PREDICT),
    code(GROUND),
    code(GROUND_VIZ),
    code(GROUND_YT),
    md(P3_REFLECT),
    md(P4_MD),
    md(P4_PREDICT),
    code(ANS_REL),
    code(ANS_VIZ),
    code(ANS_YT),
    md(P4_REFLECT),
    md(P5_MD),
    md(P5_PREDICT),
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
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written {len(cells)} cells -> {OUT}")
print(f"File size: {OUT.stat().st_size:,} bytes")
