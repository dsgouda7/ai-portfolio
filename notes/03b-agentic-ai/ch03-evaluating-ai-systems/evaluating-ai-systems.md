# Evaluating AI Systems — Measuring What Actually Matters

> **The story.** In 2002, Kishore Papineni and three colleagues at IBM Research handed their manager a single number — 0.38 — and claimed it could predict whether a machine translation was any good. The manager was skeptical: how could a floating-point score capture whether a French paragraph was correctly rendered in English? But when they validated it against thousands of human judgments, the correlation held. They called it **BLEU** — Bilingual Evaluation Understudy — and the NLP field built two decades of evaluation on top of it.
>
> The cracks showed early. BLEU worked when an output had exactly one correct form: a translated sentence, a parsed token, a fixed summary. It broke the moment correct answers could be phrased multiple ways. "The gluten-free base is available" and "We offer a GF crust option" score nearly zero against each other on BLEU — same meaning, different words. **Chin-Yew Lin** patched the recall side with **ROUGE** in 2004; **Tianyi Zhang et al.** dissolved the surface-form problem entirely with **BERTScore** in 2019 by comparing embedding vectors instead of token sequences. But all three still assumed you had a reference answer to compare against.
>
> The LLM era destroyed that assumption. In 2023, **Lianmin Zheng and colleagues** at UC Berkeley ran a bold experiment: let **GPT-4 judge model outputs** rather than hiring human annotators. Their **MT-Bench** paper showed GPT-4 preferences agreed with human preferences ~80% of the time — good enough, cheap enough, fast enough to evaluate hundreds of models at once. That same year, **Shahul Es et al.** productized the idea for RAG pipelines as **RAGAS**: four metrics (faithfulness, answer relevancy, context precision, context recall) that pinpoint exactly which component of a retrieval-generation system is broken.
>
> **Where you are.** [LLM Fundamentals (03a-ai)](../../03a-ai/) gave you prompting, CoT reasoning, RAG, and vector DBs — the foundation. Ch.1 (ReAct & Semantic Kernel) in this track unlocked **28% conversion** via tool orchestration + proactive upselling — beating the 22% phone baseline! But every prompt change risks regression without automated testing. ML had its [own metrics chapter](../../01-ml/02-classification/ch03-metrics) for supervised learning. AI needs one too — because "correctness" in free-form text is fuzzy, context-dependent, and often requires another LLM to evaluate.
>
> **Business context.** You're the Lead AI Engineer at Mamma Rosa's Pizza. Current status: **28% conversion** (target >25% ✅), **+$2.50 AOV** (✅), **~5% error** (✅), **2.5s latency** (✅), **$0.015/conv** (✅). All core targets hit! But the CEO won't ship without automated testing: "One bad regression could wipe out all your conversion gains." You're deploying 2-3 prompt iterations per day, manually testing 3-5 queries each time, and suffering 2-3 regressions per week from changes that "looked fine" in manual tests. No A/B testing framework. No production monitoring. No way to prove a new model version (GPT-4o → GPT-4o-mini) maintains quality. This chapter builds the testing infrastructure that lets you iterate fast without breaking production.
>
> **Notation.** $F \in [0,1]$ — faithfulness (how well the answer is grounded in retrieved context); $P_c \in [0,1]$ — context precision; $R_c \in [0,1]$ — context recall; $R_a \in [0,1]$ — answer relevancy; $\bar{\rho}$ — mean pairwise agreement between judge scores (inter-rater reliability).

***

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Launch **Mamma Rosa's PizzaBot** — a production AI ordering system satisfying 6 constraints:
> 1. **BUSINESS VALUE**: >25% conversion + +$2.50 AOV + 70% labor savings — 2. **ACCURACY**: <5% error — 3. **LATENCY**: <3s p95 — 4. **COST**: <$0.08/conv — 5. **SAFETY**: Zero attacks — 6. **RELIABILITY**: >99% uptime

**What we know so far:**
- ✅ Ch.1-6: All core targets hit! 28% conversion ✅, +$2.50 AOV ✅, <5% error ✅, <3s latency ✅
- ⚡ **Current state**: Production-ready bot with RAG grounding, ReAct orchestration, proactive upselling
- 🎉 **Breakthrough achieved**: Beats phone baseline on both conversion (28% vs. 22%) and AOV ($40.60 vs. $38.50)

**What's blocking us:**

🚨 **No automated testing — regression risk on every code change**

**Current deployment process:**
```
1. Developer makes prompt change ("add more enthusiasm!")
2. Manually test 3-5 sample queries in terminal
3. ✅ "Looks good!" → push to production
4. 🔥 Next day: Customer complaint — "Bot told me Margherita is gluten-free!" (regression)
5. ❌ Rollback, debug, repeat
```

**Problems:**
1. ❌ **No regression detection**: Every prompt tweak risks breaking existing queries
2. ❌ **No quality baseline**: Can't tell if new model version (GPT-4o → GPT-4o-mini) degrades accuracy
3. ❌ **Manual testing doesn't scale**: 500+ menu combinations × 20+ query types = 10,000+ test cases
4. ❌ **No A/B testing framework**: Can't safely test "add garlic bread upsell" vs. "add drink upsell"
5. ❌ **No production monitoring**: Zero visibility into real-world error rates until customers complain

**Business impact:**
- **2-3 regressions per week** from prompt/code changes
- **Each regression**: ~4 hours debugging + rollback + hotfix
- **Customer trust erosion**: "Bot gave wrong info last week, why should I trust it now?"
- **Slow iteration velocity**: Fear of breaking production → conservative changes only
- CEO: "You hit your targets, but I can't ship this without automated testing. One bad regression could wipe out all your conversion gains."

**Why manual testing isn't enough:**

Simple change, hidden regression:
```
Change: Update system prompt to be "more friendly"
Before: "Our Veggie Garden pizza has 540 calories."
After:  "You'll love our Veggie Garden pizza! It's around 500-550 calories." ❌

Problem: "around 500-550" is hallucination! Real value: 540 calories
Manual test: Passed (tester didn't check exact number)
Production impact: 10% of calorie queries now give ranges instead of exact values
```

**What this chapter unlocks:**

🚀 **Automated evaluation framework:**
1. **Golden dataset**: 200 curated query-answer pairs covering all menu scenarios
2. **RAGAS metrics**: Automated faithfulness, answer relevancy, context precision scores
3. **LLM-as-judge**: GPT-4 evaluates answer quality on 1-10 scale
4. **Regression testing**: Every code change runs against full test suite (< 2 min)
5. **A/B testing**: Safe parallel deployment with automatic winner selection
6. **Production monitoring**: Real-time dashboards tracking error rate, latency, conversion

⚡ **Expected improvements:**
- **Regression prevention**: 2-3 regressions/week → **~0.1/week** (95% reduction)
- **Development velocity**: 2 prompt iterations/day → **10+ iterations/day** (safe to experiment)
- **Quality baseline**: Detect model degradation before customer complaints
- **A/B testing confidence**: Launch upsell experiments with statistical significance
- **Metrics**: No change to conversion/AOV (Ch.7 is testing infrastructure, not UX changes)

**Constraint status after Ch.7**:
- #1-6: **All metrics maintained** (28% conversion, 5% error, 2.5s latency, $0.015/conv)
- **Infrastructure**: Automated testing prevents regressions, enables faster iteration
- **Production-ready**: Can now safely deploy updates without manual testing bottleneck

This is a **quality assurance chapter** — no business metric improvements, but essential for production reliability.

---

## 1 · Core Idea

All five blockers in §0 share a structural gap: there is no framework that can tell you, automatically and repeatably, which level of the system broke. Understanding where to measure is the prerequisite to building the regression suite that closes the 2–3-regressions/week problem.

AI system evaluation breaks into three levels:

```
Level 1 — Component evaluation   Does each piece work correctly in isolation?
                                   (embedding model recall, retrieval precision, LLM accuracy on benchmarks)

Level 2 — Pipeline evaluation    Does the assembled system produce good answers?
                                   (RAG faithfulness, agent task completion rate)

Level 3 — User evaluation        Do real users succeed at their goals?
                                   (session success rate, user satisfaction, task completion in A/B tests)
```

Most teams only do Level 1 and are surprised when Level 3 fails. The metrics at each level are different, and passing Level 1 does not guarantee passing Level 3.

> 💡 **Evaluation scope → regression catch rate:** A team testing only Level 1 (components) catches roughly 40% of production regressions — the rest surface from cross-component interactions. For PizzaBot, that means ~1–2 regressions per week still reach customers even with component tests passing. Adding Level 2 pipeline evaluation drops that to ~0.1/week.

---

## 1.5 · The Practitioner Workflow — Your 3-Scope Evaluation Diagnostic

Failure #3 in §0 — manual testing doesn't scale to 10,000+ test cases — is the workflow gap this section closes. The 3-scope framework maps each §0 blocker to a specific decision checkpoint, turning a subjective "looks good" into a reproducible binary: pass or fail.

**What you'll build by the end:** A 3-tier evaluation dashboard covering Component → Pipeline → User metrics, with automated regression tests, LLM-as-judge scoring, and production monitoring — the complete testing infrastructure that prevents the 2–3 regressions/week problem.

```
Scope 1: COMPONENT         Scope 2: PIPELINE           Scope 3: USER
──────────────────────────────────────────────────────────────────────────
Test pieces in isolation:  Test end-to-end pipeline:   Test real user impact:

• Embedding recall@k       • RAGAS faithfulness        • Task success rate
• Retriever precision      • Context precision/recall  • User satisfaction
• LLM perplexity/BLEU      • Answer relevancy          • Time-to-answer
• ROUGE-L for summaries    • Hallucination detection   • A/B test winner
• Semantic similarity      • Trace evaluation          • Inter-annotator κ

→ DECISION:                → DECISION:                 → DECISION:
  Which model/strategy?      Which failure mode?         Ship to production?
  • Recall@10 < 0.7:         • F < 0.9, Pc high:         • Task success < 90%:
    Better embeddings          Fix LLM prompt              More pilot testing
  • BLEU < 0.4:              • F high, Pc < 0.7:         • κ < 0.6:
    Switch LLM                 Fix retriever               Refine eval rubric
  • Perplexity high:         • Both low:                 • p-value > 0.05:
    More training data         System-wide issue           Insufficient data
```

> 💡 **How to use this workflow:** Start with Scope 1 when building/replacing components. Run Scope 2 after every code change (automated CI/CD). Run Scope 3 before major releases or when changing core user experience. All three scopes together form your regression prevention system.

> 📖 **Workflow pattern vs. concept-based chapters:** This is a **procedural chapter** teaching a diagnostic workflow, not a single concept. The 3 evaluation scopes are decision checkpoints, not sequential steps — you iterate between scopes based on what breaks. Compare to Ch.4 (RAG & Embeddings) which is concept-based: one core idea (retrieval augmentation) taught start-to-finish.

---

## 2 · Evaluating RAG Pipelines — RAGAS

Failure #2 in §0 — "no quality baseline" — is what RAGAS addresses. Without it, the team can't tell whether switching from `text-embedding-ada-002` to `text-embedding-3-small` improved or degraded allergen accuracy; with RAGAS scores, faithfulness and context precision give a before/after comparison in under two minutes.

**RAGAS** (Retrieval Augmented Generation Assessment) is the standard framework for evaluating RAG systems without manually labelling every output. It uses an LLM-as-judge approach: a separate LLM scores each component of the RAG response.

### The Four Core RAGAS Metrics

| Metric | What it measures | Formula (conceptual) |
|---|---|---|
| **Faithfulness** | Does the answer contain only claims supported by the retrieved context? | `# claims in answer supported by context / # total claims in answer` |
| **Answer Relevance** | Does the answer actually address the question asked? | Embedding similarity between question and answer (via reverse-generation trick) |
| **Context Precision** | Are the retrieved chunks actually relevant to the question? | `# relevant chunks in top-k / k` |
| **Context Recall** | Did retrieval surface all the chunks needed to answer the question? | `# ground-truth supporting chunks retrieved / # ground-truth chunks total` |

### How RAGAS Works in Practice

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Your RAG system produces — for each question:
#   question: the user query
#   answer:   the LLM's response
#   contexts: the list of retrieved chunks
#   ground_truth: the correct answer (only needed for context_recall)

data = {
    "question": ["Does the Margherita pizza contain gluten?"],
    "answer":   ["The Margherita is available in a gluten-free base option. The standard Margherita contains wheat flour."],
    "contexts": [["allergens.csv: Margherita — gluten (standard base), dairy. GF base available on request."]],
    "ground_truth": ["Standard Margherita contains gluten; gluten-free base available."]
}

result = evaluate(
    Dataset.from_dict(data),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(result)
# {'faithfulness': 1.0, 'answer_relevancy': 0.95, 'context_precision': 1.0, 'context_recall': 1.0}
```

> 💼 **Industry callout — RAGAS in production:** **TruLens** (TruEra, 2023) wraps RAGAS metrics with real-time tracing and dashboard visualization. **LangSmith** (LangChain, 2024) integrates RAGAS evaluation directly into prompt iteration workflows — every prompt version is automatically scored against your golden dataset. Both tools prevent the "test 3 queries manually and hope" anti-pattern. Used at Anthropic (Claude dashboard), OpenAI (eval harness), and Cohere (production monitoring).

### What Each Score Tells You

| Score | Low → | High → |
|---|---|---|
| Faithfulness low | LLM is hallucinating beyond the retrieved context | Answer is grounded in context |
| Answer relevance low | LLM is answering a different question | Answer directly addresses the query |
| Context precision low | Retrieval is noisy — irrelevant chunks in top-k | Retriever is precise |
| Context recall low | Retrieval is missing key chunks | Retriever finds all necessary context |

**The diagnostic matrix:**

| Faithfulness | Context Precision | Root Cause |
|---|---|---|
| Low | High | LLM hallucination — relevant chunks retrieved but LLM ignores them |
| Low | Low | Retriever returns bad chunks AND LLM doesn't stay grounded |
| High | Low | LLM is grounded but working with useless context — lucky or question is easy |
| High | High | System is working correctly |

**PizzaBot query examples — RAGAS diagnostic:**

| Query | Expected Behaviour | Failure Mode if Broken |
|---|---|---|
| "Does the Margherita contain gluten?" | Retrieves allergen chunk; faithfulness = 1.0 | Faithfulness < 1.0 → LLM added information not in chunk |
| "What is the cheapest GF pizza under 600 kcal?" | Retrieves calorie + price chunks; combines both | Context recall < 1.0 → calorie chunk missed |
| "Can I get delivery by 7 pm?" | Checks availability tool, not RAG | Answer relevance low → answered about ingredients instead |

> ⚡ **DECISION CHECKPOINT #1 — Pipeline failure diagnosis:**
>
> **Scenario:** PizzaBot answer quality degrades after switching from `text-embedding-ada-002` to `text-embedding-3-small`
>
> **Metrics before change:** Faithfulness=0.92, Context Precision=0.88, Answer Relevancy=0.91, ROUGE-L=0.68
> **Metrics after change:** Faithfulness=0.42, Context Precision=0.89, Answer Relevancy=0.38, ROUGE-L=0.41
>
> **Diagnosis path:**
> 1. Context Precision stayed high (0.89) → Retriever is still finding relevant chunks
> 2. Faithfulness dropped sharply (0.92 → 0.42) → LLM is now hallucinating beyond retrieved context
> 3. Answer Relevancy dropped (0.91 → 0.38) → LLM is answering different questions than asked
>
> **Root cause:** The embedding model change did NOT break retrieval (precision held), but the LLM is now following the retrieved context poorly. This suggests a **prompt engineering issue** — the new embeddings return chunks in different order/format, and the system prompt wasn't updated to handle the new structure.
>
> **Fix:** Rewrite system prompt to explicitly instruct: "Answer ONLY using facts from the Context section below. If the context doesn't contain the answer, say 'I don't have that information.'" Re-run evals: Faithfulness → 0.89, Answer Relevancy → 0.87, ROUGE-L → 0.66 ✅
>
> **Lesson:** When Context Precision is high but Faithfulness is low, the problem is almost always in the LLM prompt or model choice, not the retriever. Don't waste time retuning embeddings.

---

## 3 · Evaluating Reasoning Agents (ReAct Traces)

Failure #1 in §0 — regression risk on every code change — extends beyond RAG responses to agent planning. A prompt change that causes the ReAct agent to take 10 steps instead of 6 passes manual spot-checks but silently pushes p95 latency from 2.5s toward 4s, crossing the 3s target before any customer complaint surfaces.

For agents that produce Thought → Action → Observation traces, evaluate at the **trace level**, not just the final answer.

### Metrics for Agent Traces

| Metric | Measurement method |
|---|---|
| **Task completion rate** | Did the agent produce a correct final answer? (Binary, requires ground truth) |
| **Step efficiency** | `ideal steps / actual steps` — did the agent take unnecessary loops? |
| **Hallucinated observations** | Did the agent fabricate a tool result without calling the tool? |
| **Tool call accuracy** | Were all tool calls made with correct arguments? |
| **Context window utilisation** | Did the agent run out of context before completing? |
| **Faithfulness of final answer** | Does the final answer follow from the observation trail? |

**PizzaBot order trace targets** (6-step trace for "Large Margherita and Garlic Bread to 42 Maple Street"):

| Metric | Target | How to Measure |
|---|---|---|
| Task completion rate | ≥ 95 % | Run 100 synthetic orders; count successful `FINAL_ANSWER` emissions |
| Step efficiency | ≤ 6 steps | LLM-as-judge on traces; flag traces > 6 steps as over-planning |
| Tool groundedness | 100 % | Every price/availability claim must map to a tool `Observation` token |
| Hallucination rate | < 1 % | Self-consistency sampling: 5 chains per query, flag diverging price claims |

```python
PIZZABOT_EVAL_FIXTURES = [
    {
        "question": "What is the total for a Large Margherita and Garlic Bread delivered to 42 Maple Street?",
        "ground_truth": "£22.96 (Margherita £13.99 + Garlic Bread £3.49 + delivery £1.99)",
        "expected_tools": ["find_nearest_location", "check_item_availability",
                           "retrieve_from_rag", "calculate_order_total"],
    },
    {
        "question": "Which gluten-free pizza has the fewest calories?",
        "ground_truth": "Veggie Feast GF at 490 kcal",
        "expected_tools": ["retrieve_from_rag"],
    },
]
```

Use these fixtures as the `ground_truth` column when running RAGAS `context_recall` evaluations. A passing suite requires faithfulness ≥ 0.95 and context recall ≥ 0.90 on all fixtures.

### Trace Evaluation with LLM-as-Judge

When ground-truth labels aren't available, use a second LLM to score the trace:

```python
TRACE_EVAL_PROMPT = """
You are evaluating an AI agent's reasoning trace.

Question: {question}
Trace:
{trace}
Final Answer: {answer}

Score each dimension from 1–5:
1. Faithfulness: Is the final answer supported by the observations in the trace?
2. Efficiency: Did the agent take unnecessary steps?
3. Groundedness: Did the agent fabricate any observation (marked ⚠️ if tool result is missing)?

Respond in JSON: {"faithfulness": int, "efficiency": int, "groundedness": int, "explanation": str}
"""
```

**LLM-as-judge limitations:** the judge LLM has its own biases (verbosity preference, position bias) and can be inconsistent. Always:
- Use a stronger model as judge than the model being evaluated
- Run each evaluation 3x and take the majority
- Validate the judge's scores against human labels on a sample set

> 💼 **Industry callout — LangSmith for tracing + eval:** **LangSmith** (LangChain, 2024) is the production standard for agent trace evaluation. It automatically captures every Thought→Action→Observation step, computes LLM-as-judge scores, and flags anomalies (hallucinated observations, redundant loops). Integrates with LangChain, LlamaIndex, and Semantic Kernel. **PromptLayer** (2022) is the lightweight alternative focused on prompt versioning with A/B testing. Both used at Zapier (agent automation), Notion (AI workspace), and Replit (code agents). Typical cost: Free tier for <1k traces/month, $50-200/month for production volumes.

> ⚡ **DECISION CHECKPOINT #2 — Agent trace inefficiency diagnosis:**
>
> **Scenario:** PizzaBot order completion trace is taking 8-12 steps instead of the expected 6, causing p95 latency to drift from 2.5s → 4.1s
>
> **Trace analysis (sample order: "Large Margherita to 42 Maple St"):**
> ```
> Step 1: Thought: "Need to find location"        → check_location(postcode="SW1A")
> Step 2: Thought: "Need menu availability"       → check_item_availability("Margherita")
> Step 3: Thought: "What size options exist?"     → retrieve_from_rag("Margherita sizes") ❌ REDUNDANT
> Step 4: Thought: "Already got that info"        → calculate_order_total(items=[...])
> Step 5: Thought: "Wait, forgot delivery fee"    → calculate_order_total(items=[...])  ❌ LOOP
> Step 6: Thought: "Is delivery available?"       → check_location(postcode="SW1A")     ❌ DUPLICATE
> ... (continues)
> ```
>
> **Metrics:**
> - Task completion rate: 94% ✅ (still completing orders)
> - Step efficiency: 6/10 = 0.6 ❌ (ideal 6 steps, actual 10 average)
> - Tool groundedness: 100% ✅ (no fabricated observations)
> - Hallucination rate: <1% ✅
>
> **Root cause:** Agent is **forgetting context from earlier steps** — it re-asks for location data it already retrieved in Step 1, and calls `calculate_order_total` twice because it doesn't remember the first result included delivery fee.
>
> **Fix applied:**
> 1. Updated agent prompt to include **explicit memory instruction**: "Track all tool results in your scratchpad. Before calling a tool, check if you already have that information."
> 2. Reduced context window usage by 30% (removed verbose examples) to ensure full trace history fits
> 3. Added **tool result caching**: if same tool+args called within 5 steps, return cached result
>
> **Metrics after fix:**
> - Step efficiency: 6/6.2 = 0.97 ✅ (near-optimal)
> - p95 latency: 4.1s → 2.6s ✅ (back under 3s target)
> - No change to task completion or accuracy
>
> **Lesson:** High task completion rate does NOT mean efficient traces. Always measure step efficiency separately — redundant tool calls burn latency and API costs even when the final answer is correct.

> 💡 **Step efficiency → latency and cost:** Each redundant tool call adds ~0.3–0.5s. An agent averaging 10 steps instead of the optimal 6 pushes p95 latency from 2.5s toward 4s — past the 3s target — while doubling LLM token costs for those traces. Step efficiency is not a vanity metric: it protects both the latency SLA and the $0.015/conv cost ceiling from §0 simultaneously.

> ➡️ **Ch.10 (Fine-Tuning)** trains the agent’s planning model directly on domain traces, reducing average steps from ~10 to ~6 more reliably than prompt engineering — and closes the “planning regression” class that prompt changes routinely introduce.

---

## 4 · Component-Level Evaluation

Failure #3 in §0 — manual testing doesn't scale — bites hardest at the component level. The embedding model sets the ceiling for every downstream RAGAS score: a Recall@10 of 0.60 means the allergen chunk the retriever silently dropped cannot be recovered by any amount of prompt engineering.

### Embedding Model Evaluation

The retrieval quality is bounded by the embedding model's ability to separate relevant from irrelevant content.

| Metric | Meaning |
|---|---|
| **NDCG@k** | Normalised Discounted Cumulative Gain — ranks correct chunks higher than incorrect ones |
| **MRR** (Mean Reciprocal Rank) | `1 / rank_of_first_correct_chunk` — how early does the first relevant result appear? |
| **Recall@k** | `# relevant chunks in top-k / # total relevant chunks` |

**How to measure:** use a test set of `(query, expected_chunk_ids)` pairs. Run retrieval. Compute the above. MTEB (Massive Text Embedding Benchmark) provides standardised benchmarks for comparing embedding models.

#### Why NDCG@k beats Recall@k for production retrieval

**Recall@k** asks only: "Did we find the relevant chunks somewhere in the top k?" A retriever that buries the allergen chunk at rank 9 out of 10 scores the same Recall@10 as one that returns it first.

**NDCG@k** (Normalised Discounted Cumulative Gain) goes further: it rewards finding relevant chunks *earlier*. Each position is discounted by its rank — position 1 gets full credit; position $i$ contributes $1/\log_2(i+1)$.

$$\text{DCG@k} = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i+1)}$$

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

where $\text{IDCG@k}$ is the ideal DCG — the score you would get if every relevant chunk appeared at the top. Normalising by IDCG brings the metric into $[0, 1]$ regardless of how many relevant chunks exist.

**Plain-English reading.** NDCG@k = 0.90 means your retriever captures 90% of the value of a perfect retriever. A chunk at rank 1 contributes $1/\log_2(2) = 1.0$; the same chunk at rank 4 contributes only $1/\log_2(5) \approx 0.43$ — less than half the credit for the same piece of information.

**PizzaBot grounding.** The query "Does the Margherita contain gluten?" needs the allergen chunk. If it appears at rank 1, NDCG@10 is near 1.0. If it appears at rank 8, Recall@10 is still 1.0 — chunk found! — but NDCG@10 drops to ~0.28. The LLM reads top chunks first; a buried allergen fact is effectively invisible to it. That is the failure mode NDCG@k catches and Recall@k misses entirely.

> 💡 **Business metric consequence.** A retriever with Recall@10 = 1.0 but NDCG@10 = 0.55 will still cause allergen errors — not because the correct chunk is absent, but because it arrives too late for the LLM to weight it correctly. In the pizza ordering context, a buried allergen disclaimer means a customer with a wheat allergy places an order the bot should have warned against. Always report NDCG@k alongside Recall@k when benchmarking your embedding model.

#### Code Snippet #1 — BLEU & ROUGE Calculation for LLM Summaries

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

def evaluate_llm_summary(reference: str, candidate: str) -> dict:
    """
    Evaluate LLM-generated summary against reference using BLEU and ROUGE.

    BLEU measures n-gram overlap (precision-focused).
    ROUGE measures recall of reference n-grams in candidate.
    Both range [0, 1]; higher is better.
    """
    # Tokenize
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    # BLEU-4 with smoothing (handles zero n-gram matches gracefully)
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], cand_tokens,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=smoothing)

    # ROUGE-L (longest common subsequence)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)
    rouge_l = rouge_scores['rougeL'].fmeasure

    # Semantic similarity via embeddings (requires sentence-transformers)
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ref_emb = model.encode(reference, convert_to_tensor=True)
    cand_emb = model.encode(candidate, convert_to_tensor=True)
    semantic_sim = util.cos_sim(ref_emb, cand_emb).item()

    return {
        'bleu_4': bleu,
        'rouge_l': rouge_l,
        'semantic_similarity': semantic_sim
    }

# Example: PizzaBot allergen response evaluation
reference = "The Margherita contains gluten in the standard base. A gluten-free base is available on request."
candidate = "Our Margherita pizza has wheat flour, so it contains gluten. We offer a GF crust option."

scores = evaluate_llm_summary(reference, candidate)
print(f"BLEU-4: {scores['bleu_4']:.3f}")           # → 0.42 (moderate n-gram overlap)
print(f"ROUGE-L: {scores['rouge_l']:.3f}")         # → 0.68 (good recall of key phrases)
print(f"Semantic: {scores['semantic_similarity']:.3f}")  # → 0.91 (semantically equivalent)

# Interpretation:
# - Low BLEU (0.42) but high semantic similarity (0.91) → paraphrased correctly
# - High ROUGE-L (0.68) → covers most reference content
# - This is a GOOD answer despite low BLEU (BLEU penalizes paraphrasing)
```

> 📖 **When to use each metric:**
> - **BLEU**: Translation, code generation (exact syntax matters)
> - **ROUGE**: Summarization (recall of key points matters more than exact wording)
> - **Semantic Similarity**: Conversational AI (meaning preservation > surface form)
> - **All three together**: Comprehensive LLM output evaluation

> 💡 **BLEU and ROUGE scales — reading the numbers.** Both are bounded [0, 1]; higher is better.
>
> | Score | BLEU interpretation | ROUGE-L interpretation |
> |---|---|---|
> | > 0.60 | Strong n-gram overlap (near-reference quality) | Most of the reference content is present |
> | 0.40–0.60 | Moderate overlap — correct paraphrasing likely | Good recall of key phrases |
> | 0.20–0.40 | Weak overlap — investigate output verbosity/truncation | Partial recall; important points may be missing |
> | < 0.20 | Poor overlap — output may be off-topic or hallucinated | Low recall; significant content gaps |
>
> **BLEU penalizes correct paraphrases.** The example above (BLEU=0.42, Semantic=0.91) is typical for conversational AI: semantically equivalent but phrased differently. For PizzaBot allergen answers, semantic similarity is the primary signal; BLEU is a secondary sanity check.
>
> **ROUGE-L baseline for your domain.** On a pizza-domain allergen FAQ, a model that consistently achieves ROUGE-L > 0.65 is covering the key facts from the reference. Below 0.50 warrants inspection of what content is being dropped.

#### Code Snippet #1b — Perplexity Measurement for Language Model Evaluation

Perplexity measures how surprised the model is by a test corpus — lower means the model is a better fit for the language it's evaluating.

$$\text{PP}(W) = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i \mid w_{<i})\right)$$

**Plain-English reading:** Perplexity of 10 means the model is, on average, as uncertain as if it were choosing uniformly among 10 equally likely next tokens at each position. Lower perplexity = the model finds the text more predictable = better language model fit.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(text: str, model_name: str = "gpt2") -> float:
    """
    Compute perplexity of a text string under a causal language model.
    Lower = model is less surprised = better domain fit.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens, labels=tokens["input_ids"])
    loss = outputs.loss  # cross-entropy loss
    return torch.exp(loss).item()

# Example: compare base GPT-2 vs pizza-domain fine-tuned
test_text = "The Margherita pizza contains gluten in the standard base. A gluten-free base is available."

base_pp = compute_perplexity(test_text, "gpt2")
ft_pp = compute_perplexity(test_text, "path/to/pizza-finetuned-gpt2")

print(f"Base GPT-2 perplexity: {base_pp:.1f}")      # → ~42 (unfamiliar domain)
print(f"Fine-tuned perplexity: {ft_pp:.1f}")        # → ~11 (fits pizza domain well)
```

**When to use perplexity:**
- Comparing base language models before task-specific fine-tuning
- Detecting domain drift: if perplexity on production queries rises week-over-week, the language distribution has shifted
- Selecting fine-tuning checkpoints: lower perplexity on held-out domain data = better checkpoint

**Do NOT use perplexity to measure:** instruction-following quality, factual accuracy, or safety — a model can have low perplexity on its domain while still hallucinating facts. Use RAGAS faithfulness + context precision for those.

**Typical reference values:**

| Model | Corpus | Typical Perplexity |
|---|---|---|
| GPT-2 (small) | Penn Treebank | 18–29 |
| GPT-2 (xl) | Penn Treebank | ~10 |
| GPT-4 class | Standard benchmarks | 3–8 |
| Your fine-tuned model | Pizza FAQ corpus | Target < 15 after fine-tuning |

> 💼 **Industry callout — Scale AI for human labeling:** When automated metrics disagree with user satisfaction (e.g., BLEU=0.85 but users say "confusing"), you need **human evaluation at scale**. **Scale AI** (2016, now valued at $7B+) pioneered outsourced annotation with 4-way redundancy and quality control. **Surge AI** (specialized for NLP, 2020) and **Labelbox** (active learning pipelines) are alternatives. Used by OpenAI (RLHF labeling), Anthropic (Constitutional AI feedback), and Google (Bard safety ratings). Typical cost: $0.10–$2.00 per label depending on task complexity.

### Chunking Strategy Evaluation

Different chunking strategies (fixed size, sentence-level, semantic) affect both precision and recall. Evaluate by measuring RAGAS context precision/recall across strategies.

```
Fixed 512-token chunks    → Context precision: 0.71  | Context recall: 0.83
Sentence-level chunks     → Context precision: 0.79  | Context recall: 0.76
Semantic (topic) chunks   → Context precision: 0.85  | Context recall: 0.81
```

There is no universally best strategy — the right one depends on document structure.

#### Code Snippet #2 — Complete RAGAS Evaluation Pipeline

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
import pandas as pd
from typing import List, Dict

class RAGEvaluationPipeline:
    """
    Production RAGAS evaluation pipeline for PizzaBot.
    Runs full suite of metrics and flags regressions.
    """

    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            'faithfulness': 0.90,
            'answer_relevancy': 0.85,
            'context_precision': 0.80,
            'context_recall': 0.85
        }

    def evaluate_batch(self,
                       questions: List[str],
                       answers: List[str],
                       contexts: List[List[str]],
                       ground_truths: List[str]) -> pd.DataFrame:
        """
        Evaluate a batch of RAG responses.

        Args:
            questions: User queries
            answers: LLM-generated responses
            contexts: Retrieved chunks for each query (list of lists)
            ground_truths: Expected correct answers

        Returns:
            DataFrame with per-query scores + pass/fail flags
        """
        # Prepare dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)

        # Run RAGAS evaluation
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy,
                    context_precision, context_recall]
        )

        # Convert to DataFrame
        df = pd.DataFrame(result)

        # Add pass/fail flags
        for metric, threshold in self.thresholds.items():
            df[f'{metric}_pass'] = df[metric] >= threshold

        # Add overall pass column
        pass_cols = [f'{m}_pass' for m in self.thresholds.keys()]
        df['all_pass'] = df[pass_cols].all(axis=1)

        return df

    def flag_regressions(self, current_scores: pd.DataFrame,
                         baseline_scores: pd.DataFrame) -> List[str]:
        """
        Compare current run against baseline, flag significant drops.

        Returns list of regression warnings.
        """
        warnings = []
        metrics = ['faithfulness', 'answer_relevancy',
                  'context_precision', 'context_recall']

        for metric in metrics:
            current_mean = current_scores[metric].mean()
            baseline_mean = baseline_scores[metric].mean()
            drop = baseline_mean - current_mean

            if drop > 0.05:  # >5% drop is regression
                warnings.append(
                    f"REGRESSION: {metric} dropped {drop:.2%} "
                    f"({baseline_mean:.3f} → {current_mean:.3f})"
                )

        return warnings

# Example: PizzaBot golden dataset evaluation
pipeline = RAGEvaluationPipeline()

golden_questions = [
    "Does the Margherita contain gluten?",
    "What's the cheapest GF pizza under 600 kcal?",
    "Can I get delivery by 7 pm to SW1A 1AA?"
]

golden_answers = [
    "The standard Margherita contains gluten. A gluten-free base is available.",
    "The Veggie Feast GF is £10.99 with 490 kcal.",
    "Yes, delivery to SW1A 1AA is available. Current estimated time is 35 minutes."
]

golden_contexts = [
    ["allergens.csv: Margherita — gluten (standard), dairy. GF base available."],
    ["menu.csv: Veggie Feast GF — £10.99, 490 kcal", "menu.csv: Margherita GF — £11.99, 520 kcal"],
    ["delivery_zones.csv: SW1A — Zone 1, 25-40 min typical"]
]

golden_truths = [
    "Standard Margherita contains gluten; GF base available",
    "Veggie Feast GF at 490 kcal for £10.99",
    "Delivery available to SW1A, estimated 35 min"
]

scores = pipeline.evaluate_batch(golden_questions, golden_answers,
                                 golden_contexts, golden_truths)

print(scores[['question', 'faithfulness', 'answer_relevancy',
              'context_precision', 'all_pass']])

# Check for regressions against baseline
baseline = pd.read_csv('baseline_scores.csv')
regressions = pipeline.flag_regressions(scores, baseline)
if regressions:
    print("\n⚠️ REGRESSIONS DETECTED:")
    for warning in regressions:
        print(f"  {warning}")
    # Trigger CI/CD failure
    exit(1)
else:
    print("\n✅ All metrics passed. Safe to deploy.")
```

> 💼 **Industry callout — DeepEval for CI/CD integration:** **DeepEval** (Confident AI, 2024) extends RAGAS with production-ready features: automatic golden dataset generation from production logs, GitHub Actions integration (fail PR if metrics drop >5%), and cost tracking ($0.002/eval with GPT-4o-mini as judge). **TruLens** (TruEra, 2023) offers enterprise features like drift detection and adversarial testing. Both used at Stripe (payment fraud AI), Shopify (merchant support bots), and Airbnb (booking agents). Typical setup: 100-200 golden queries evaluated on every commit, <2 min CI/CD runtime.

---

## 5 · Hallucination Detection

The silent regression in §0 — a friendly-tone prompt change turning "540 calories" into "around 500-550 calories" — passed manual testing and reached production because there was no hallucination detection layer. The 5% error-rate target from §0 is directly determined by how often the LLM fabricates claims beyond its retrieved context.

Hallucination is the biggest reliability problem in deployed LLM systems. Detection strategies:

### Self-consistency sampling

Sample the same question N times at temperature > 0. If the factual claims are consistent across samples, they are likely grounded. If they vary, the model is uncertain and may be hallucinating.

### Toxicity Scoring

Hallucinations are one production failure mode; harmful outputs are another. Toxicity scoring assigns a probability to each model output that it would be considered harmful by a human rater.

**How it works:** A toxicity classifier (trained on human-labeled examples of harmful text) scores each output on a 0–1 scale across categories.

| Category | What it detects | Production use |
|---|---|---|
| **Toxicity** | Hateful, abusive, or threatening language | Primary guardrail |
| **Severe toxicity** | Explicit threats or slurs | Hard block |
| **Identity attack** | Attacks on protected characteristics | Hard block |
| **Insult** | Targeted personal insults | Soft flag |
| **Profanity** | Obscene language | Context-dependent |
| **Threat** | Direct or indirect threats | Hard block |

**Reading the score:** A toxicity score of 0.82 means the classifier assigns an 82% probability that a human rater would consider the output toxic — it is a probabilistic flag, not a legal determination.

**Production thresholds (adjust per use case):**

| Score | Action |
|---|---|
| > 0.90 | Block the response entirely; log for review |
| 0.70–0.90 | Route to human review before delivery |
| 0.50–0.70 | Flag in monitoring dashboard; allow delivery |
| < 0.50 | Pass through; monitor in aggregate |

**Open-source options:**

```python
# Detoxify — runs fully locally, no API key needed
from detoxify import Detoxify

results = Detoxify('original').predict(
    "Your pizza is terrible and so are you!"
)
# {'toxicity': 0.82, 'severe_toxicity': 0.12,
#  'identity_attack': 0.02, 'insult': 0.71,
#  'profanity': 0.31, 'threat': 0.05}

# Apply production threshold
if results['toxicity'] > 0.70 or results['severe_toxicity'] > 0.50:
    print("Route to human review")
elif results['toxicity'] > 0.90:
    print("Block response")
else:
    print("Deliver response")
```

### Self-consistency detection

```python
def detect_hallucination_by_consistency(llm, question, n=5, threshold=0.7):
    answers = [llm.generate(question, temperature=0.7) for _ in range(n)]
    # Extract the key factual claim from each answer
    claims = [extract_claim(a) for a in answers]
    # Check if the majority agree
    from collections import Counter
    most_common, count = Counter(claims).most_common(1)[0]
    return most_common, count / n   # (answer, confidence)
```

### Entailment checking

Run a Natural Language Inference (NLI) model to check if the retrieved context **entails** each claim in the answer:

```
retrieved_context: "allergens.csv: Margherita — gluten (standard), dairy. GF base available."
answer_claim:      "The Margherita contains gluten in the standard base"
NLI label:         ENTAILMENT  →  claim is grounded

answer_claim:      "Home values increased 12% last year"
NLI label:         NEUTRAL     →  claim has no support in context → potential hallucination
```

Open-source NLI models: `cross-encoder/nli-deberta-v3-base`, `vectara/hallucination_evaluation_model`.

> 💡 **Hallucination rate → error rate:** The 5% error-rate target from §0 maps directly to hallucination frequency — 1 fabricated claim per 20 responses = 5% error rate; 1 per 10 = 10%. Self-consistency sampling (5 chains at temperature 0.7) costs ~5× API tokens on flagged queries but catches hallucination clusters before they trigger the 4-hour rollback cycle.

> 💼 **Connection to Ch.7 safety guardrails.** The safety chapter defines what to filter; toxicity scoring is the measurement layer that implements the filter. A guardrail that blocks inputs is only as good as its toxicity score threshold — tune this on your domain's false-positive/false-negative trade-off, just as you tuned precision/recall for FaceAI's Bald classifier. For PizzaBot, severe toxicity and threats are hard blocks; general toxicity > 0.70 routes to a "We're sorry, we can only help with pizza orders" fallback.

---


## 7 · The Evaluation Benchmark Trap

Failure #2 in §0 — "no quality baseline" — tempts teams toward a shortcut: use public MMLU or HumanEval scores to select models. This section explains why a model that aces benchmarks still hallucinates allergen claims in Mamma Rosa's private-data pipeline, and why that shortcut destroys customer trust.

**The trap.** Standard LLM benchmarks — **MMLU** (57-subject knowledge test), **HumanEval** (Python coding), **MATH** (competition math) — measure model capability on fixed, curated test sets. They do not predict application performance. A model that scores 85 on MMLU may still hallucinate your specific domain's terminology at high rates because that terminology was underrepresented in its training data.

**Why they diverge.** Benchmarks are closed-domain, multiple-choice or exact-match tasks compiled once. Your production pipeline is open-domain, free-form, run over private data the model has never seen. The distribution gap is fundamental, not incidental.

**PizzaBot grounding.** GPT-4o scores near the ceiling on MMLU (~86%) and HumanEval (~90%). That tells you it has broad knowledge and writes solid Python. It tells you nothing about whether it faithfully cites allergen data from Mamma Rosa's private menu corpus, stays within the "pizza orders only" persona, or correctly chains `check_item_availability()` for a multi-item order. A smaller model fine-tuned on 500 PizzaBot traces (Ch.8) can easily beat GPT-4o on your RAGAS faithfulness score while sitting 15 points lower on MMLU.

**The rule:** always evaluate on your own data, in your own pipeline, with your own queries. Benchmark scores are useful for initial model shortlisting; they are not a substitute for domain evaluation.

> 💡 **Business metric consequence.** A team that selects models purely by benchmark ranking risks shipping a 90th-percentile MMLU model that fails 12% of allergen queries — destroying customer trust — while rejecting a 75th-percentile model that would have handled them perfectly. Your golden dataset (50–200 PizzaBot queries with RAGAS scores) is a more honest predictor of production quality than any public leaderboard position.

> ⚠️ **Common interview trap.** When asked "how would you choose a base model?", answering "pick the highest MMLU score" signals you haven't shipped production AI. The right answer: shortlist by benchmark, validate on your domain data, and always report RAGAS / task-success rates alongside public leaderboard positions.

---

## 8 · What a Minimal Evaluation Setup Looks Like

All five blockers from §0 can be substantially reduced with a single artefact: a 50–100 question golden dataset run on every commit. This is the minimum viable version that eliminates the "test 3 queries and hope" anti-pattern without requiring a dedicated evaluation engineering team.

For a production RAG pipeline, a minimum viable evaluation setup:

```
1. Curate 50–100 representative questions from real or expected user queries
2. For each: store the correct answer and the expected source chunk(s)
3. Run your full pipeline end-to-end on each question
4. Compute: faithfulness, context precision, context recall, answer relevance (RAGAS)
5. Compute: task completion rate (does the answer match the correct answer?)
6. Set alert thresholds: faithfulness < 0.9 → PagerDuty; context recall < 0.7 → review retriever
7. Re-run after every change to the pipeline (chunk size, embedding model, prompt template)
```

This 50–100 question set is your regression test suite. It catches regressions before they reach production.

#### Code Snippet #3 — Human Evaluation Rubric with Inter-Annotator Agreement

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score
from typing import List, Tuple

class HumanEvaluationRubric:
    """
    Human evaluation framework for PizzaBot responses.
    Measures task success, user satisfaction, and inter-rater reliability.
    """

    RUBRIC = {
        'correctness': {
            0: 'Factually wrong or hallucinated',
            1: 'Partially correct with errors',
            2: 'Mostly correct with minor issues',
            3: 'Completely correct'
        },
        'completeness': {
            0: 'Missing critical information',
            1: 'Addresses question but incomplete',
            2: 'Complete answer to question asked',
            3: 'Complete + proactive helpful extras'
        },
        'tone': {
            0: 'Inappropriate or confusing',
            1: 'Robotic or too formal',
            2: 'Professional and clear',
            3: 'Warm, helpful, on-brand'
        }
    }

    def compute_task_success_rate(self,
                                   correctness_scores: List[int]) -> float:
        """
        Task success = % of responses rated 2 or 3 for correctness.
        Target: ≥90% for production deployment.
        """
        passing = sum(1 for s in correctness_scores if s >= 2)
        return passing / len(correctness_scores)

    def compute_user_satisfaction(self,
                                   correctness: List[int],
                                   completeness: List[int],
                                   tone: List[int]) -> float:
        """
        User satisfaction = average of all dimensions scaled to [0, 1].
        Target: ≥0.85 for production.
        """
        all_scores = correctness + completeness + tone
        return np.mean(all_scores) / 3.0  # Scale from [0-3] to [0-1]

    def compute_inter_annotator_agreement(self,
                                          rater1_scores: List[int],
                                          rater2_scores: List[int]) -> Tuple[float, str]:
        """
        Cohen's Kappa measures inter-rater reliability.

        κ interpretation:
          < 0.40 : Poor agreement (refine rubric)
          0.40-0.60 : Moderate agreement
          0.60-0.80 : Substantial agreement (production-ready)
          > 0.80 : Near-perfect agreement
        """
        kappa = cohen_kappa_score(rater1_scores, rater2_scores)

        if kappa < 0.40:
            interpretation = "Poor agreement — refine rubric definitions"
        elif kappa < 0.60:
            interpretation = "Moderate agreement — usable but noisy"
        elif kappa < 0.80:
            interpretation = "Substantial agreement — production-ready"
        else:
            interpretation = "Near-perfect agreement"

        return kappa, interpretation

# Example: PizzaBot pilot study with 50 test queries, 2 annotators
evaluator = HumanEvaluationRubric()

# Rater 1 scores (correctness dimension, 50 queries)
rater1_correctness = [3, 3, 2, 3, 3, 2, 3, 1, 3, 3,
                       3, 2, 3, 3, 3, 3, 2, 3, 3, 2,
                       3, 3, 3, 2, 3, 3, 3, 3, 2, 3,
                       3, 3, 3, 2, 3, 3, 3, 3, 2, 3,
                       3, 2, 3, 3, 3, 3, 3, 2, 3, 3]

# Rater 2 scores (same queries, independent rating)
rater2_correctness = [3, 3, 2, 3, 3, 2, 3, 2, 3, 3,  # Note: query 8 differs (1→2)
                       3, 2, 3, 3, 3, 3, 2, 3, 3, 2,
                       3, 3, 3, 2, 3, 3, 3, 3, 2, 3,
                       3, 3, 3, 2, 3, 3, 3, 3, 2, 3,
                       3, 2, 3, 3, 3, 3, 3, 2, 3, 3]

# Compute metrics
task_success = evaluator.compute_task_success_rate(rater1_correctness)
print(f"Task Success Rate: {task_success:.1%}")  # → 96% (48/50 rated ≥2)

kappa, interpretation = evaluator.compute_inter_annotator_agreement(
    rater1_correctness, rater2_correctness
)
print(f"Cohen's κ: {kappa:.3f} — {interpretation}")  # → 0.92 — Near-perfect

# Decision: κ > 0.80 and task success > 90% → APPROVED FOR PRODUCTION ✅
```

> ⚡ **DECISION CHECKPOINT #3 — A/B test winner selection:**
>
> **Scenario:** Testing two upsell strategies in parallel:
> - **Variant A (Current):** "Would you like garlic bread with that?"
> - **Variant B (New):** "Popular combo: add garlic bread and a drink for £4.99 (save £1.50)"
>
> **Hypothesis:** Variant B's bundled offer will increase Average Order Value (AOV) without hurting conversion.
>
> **Test setup:**
> - 2000 users per variant (4000 total)
> - Primary metric: AOV
> - Secondary metrics: Conversion rate, task success rate
> - Guardrail: Task success must stay ≥90%
>
> **Results after 1 week:**
>
> **Reading the p-value column:** Each p-value is the probability of seeing a difference this large by random sampling noise alone, if the two variants actually performed identically. p = 0.003 means only a 0.3% chance the AOV lift is noise — it's real. p = 0.08 means an 8% chance the conversion drop is noise — too high to call it significant. (Full p-value treatment: [notes/01-ml/01-regression/ch06-metrics](../../01-ml/01-regression/ch06-metrics/) §8b.)
>
> | Metric | Variant A (Control) | Variant B (Bundle) | Δ | p-value |
> |--------|---------------------|-------------------|---|---------|
> | AOV | £38.50 | £41.20 | +£2.70 (+7.0%) | 0.003 ✅ |
> | Conversion | 28.1% | 26.8% | -1.3% | 0.08 ❌ |
> | Task success | 94.2% | 92.7% | -1.5% | 0.12 ✅ |
>
> **Analysis:**
> 1. **AOV lift is statistically significant** (p=0.003, well below 0.05 threshold) — Variant B generates £2.70 more per order
> 2. **Conversion drop is NOT significant** (p=0.08 > 0.05) — could be random noise, need more data
> 3. **Task success guardrail passes** (92.7% > 90% minimum, p=0.12 means no significant degradation)
>
> **Revenue impact calculation:**
> - Average 500 orders/day
> - Variant B: +£2.70 × 500 = +£1,350/day additional revenue
> - Annual: £1,350 × 365 = **+£492,750/year**
> - But: 1.3% conversion drop → -6.5 orders/day → -£250/day → -£91,250/year
> - **Net gain: £401,500/year** (even with conversion drop, AOV lift dominates)
>
> **Decision:** **SHIP VARIANT B** ✅
>
> **Reasoning:**
> - AOV lift is significant and substantial (+7%)
> - Conversion drop is within noise (p=0.08) and may recover with more data
> - Task success guardrail passes (no quality degradation)
> - Even worst-case scenario (conversion drop is real) still yields +£400k/year
> - Monitor conversion closely for 2 more weeks; if it drops below 26%, revert
>
> **Lesson:** Don't wait for every metric to be "perfect" — if primary metric wins significantly, secondary metric losses are acceptable IF they don't break guardrails AND the revenue math favors shipping. Paralysis from "not enough data" costs more than shipping with 90% confidence.

---

#### Code Snippet #4 — Complete 3-Scope Evaluation Harness

```python
"""
PizzaBot Complete Evaluation Harness
Runs all 3 evaluation scopes: Component → Pipeline → User
Designed for CI/CD integration and production monitoring.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from sklearn.metrics import cohen_kappa_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationThresholds:
    """Production quality gates for PizzaBot deployment."""
    # Scope 1: Component
    embedding_recall_at_10: float = 0.75
    llm_rouge_l: float = 0.60

    # Scope 2: Pipeline
    faithfulness: float = 0.90
    context_precision: float = 0.80
    answer_relevancy: float = 0.85

    # Scope 3: User
    task_success_rate: float = 0.90
    inter_annotator_kappa: float = 0.60

class PizzaBotEvaluationHarness:
    """
    Complete evaluation framework combining all 3 scopes.
    Returns GO/NO-GO decision for production deployment.
    """

    def __init__(self, thresholds: EvaluationThresholds = None):
        self.thresholds = thresholds or EvaluationThresholds()
        self.results = {}

    def run_scope1_component_eval(self,
                                   embedding_model,
                                   llm_model,
                                   test_queries: List[str]) -> Dict:
        """
        Scope 1: Test embedding recall and LLM quality in isolation.
        """
        logger.info("🔬 Running Scope 1: Component Evaluation...")

        # Embedding recall test (mock implementation)
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(embedding_model)

        # Compute recall@10 for 50 test queries against known ground truth chunks
        recall_scores = []
        for query in test_queries[:50]:  # Sample for speed
            # ... retrieval logic ...
            recall_scores.append(0.82)  # Placeholder

        embedding_recall = sum(recall_scores) / len(recall_scores)

        # LLM quality test: ROUGE-L on summarization
        rouge_scores = []
        for query in test_queries[:20]:
            # ... generate summary, compute ROUGE ...
            rouge_scores.append(0.68)  # Placeholder

        llm_rouge = sum(rouge_scores) / len(rouge_scores)

        scope1_pass = (
            embedding_recall >= self.thresholds.embedding_recall_at_10 and
            llm_rouge >= self.thresholds.llm_rouge_l
        )

        result = {
            'embedding_recall@10': embedding_recall,
            'llm_rouge_l': llm_rouge,
            'pass': scope1_pass
        }

        logger.info(f"  Embedding Recall@10: {embedding_recall:.3f} "
                   f"{'✅' if embedding_recall >= self.thresholds.embedding_recall_at_10 else '❌'}")
        logger.info(f"  LLM ROUGE-L: {llm_rouge:.3f} "
                   f"{'✅' if llm_rouge >= self.thresholds.llm_rouge_l else '❌'}")

        self.results['scope1'] = result
        return result

    def run_scope2_pipeline_eval(self,
                                  rag_pipeline,
                                  golden_dataset: Dict) -> Dict:
        """
        Scope 2: Test full RAG pipeline with RAGAS metrics.
        """
        logger.info("🔗 Running Scope 2: Pipeline Evaluation...")

        # Run RAGAS evaluation
        dataset = Dataset.from_dict(golden_dataset)
        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy,
                    context_precision, context_recall]
        )

        df = pd.DataFrame(ragas_result)

        f_mean = df['faithfulness'].mean()
        ar_mean = df['answer_relevancy'].mean()
        cp_mean = df['context_precision'].mean()

        scope2_pass = (
            f_mean >= self.thresholds.faithfulness and
            ar_mean >= self.thresholds.answer_relevancy and
            cp_mean >= self.thresholds.context_precision
        )

        result = {
            'faithfulness': f_mean,
            'answer_relevancy': ar_mean,
            'context_precision': cp_mean,
            'pass': scope2_pass
        }

        logger.info(f"  Faithfulness: {f_mean:.3f} "
                   f"{'✅' if f_mean >= self.thresholds.faithfulness else '❌'}")
        logger.info(f"  Answer Relevancy: {ar_mean:.3f} "
                   f"{'✅' if ar_mean >= self.thresholds.answer_relevancy else '❌'}")
        logger.info(f"  Context Precision: {cp_mean:.3f} "
                   f"{'✅' if cp_mean >= self.thresholds.context_precision else '❌'}")

        self.results['scope2'] = result
        return result

    def run_scope3_user_eval(self,
                             human_ratings: Dict[str, List[int]]) -> Dict:
        """
        Scope 3: Human evaluation with inter-rater agreement.
        """
        logger.info("👥 Running Scope 3: User Evaluation...")

        # Task success rate (% rated ≥2 for correctness)
        rater1 = human_ratings['rater1_correctness']
        task_success = sum(1 for s in rater1 if s >= 2) / len(rater1)

        # Inter-annotator agreement
        rater2 = human_ratings['rater2_correctness']
        kappa = cohen_kappa_score(rater1, rater2)

        scope3_pass = (
            task_success >= self.thresholds.task_success_rate and
            kappa >= self.thresholds.inter_annotator_kappa
        )

        result = {
            'task_success_rate': task_success,
            'inter_annotator_kappa': kappa,
            'pass': scope3_pass
        }

        logger.info(f"  Task Success Rate: {task_success:.1%} "
                   f"{'✅' if task_success >= self.thresholds.task_success_rate else '❌'}")
        logger.info(f"  Cohen's κ: {kappa:.3f} "
                   f"{'✅' if kappa >= self.thresholds.inter_annotator_kappa else '❌'}")

        self.results['scope3'] = result
        return result

    def generate_deployment_decision(self) -> Tuple[bool, str]:
        """
        Final GO/NO-GO decision based on all 3 scopes.
        """
        scope1_pass = self.results.get('scope1', {}).get('pass', False)
        scope2_pass = self.results.get('scope2', {}).get('pass', False)
        scope3_pass = self.results.get('scope3', {}).get('pass', False)

        all_pass = scope1_pass and scope2_pass and scope3_pass

        if all_pass:
            decision = "🚀 GO FOR PRODUCTION"
            reason = "All 3 evaluation scopes passed quality gates."
        elif not scope1_pass:
            decision = "❌ NO-GO"
            reason = "Scope 1 (Component) failed — fix embedding/LLM before pipeline testing."
        elif not scope2_pass:
            decision = "❌ NO-GO"
            reason = "Scope 2 (Pipeline) failed — RAGAS metrics below threshold."
        elif not scope3_pass:
            decision = "⚠️ CONDITIONAL GO"
            reason = ("Scope 3 (User) failed — consider limited rollout "
                     "with close monitoring before full production.")
        else:
            decision = "❌ NO-GO"
            reason = "Multiple scopes failed."

        logger.info(f"\n{'='*60}")
        logger.info(f"DEPLOYMENT DECISION: {decision}")
        logger.info(f"Reason: {reason}")
        logger.info(f"{'='*60}\n")

        return all_pass, reason

# Example: Full evaluation run before production deployment
if __name__ == "__main__":
    harness = PizzaBotEvaluationHarness()

    # Scope 1: Component tests
    test_queries = ["Does Margherita have gluten?", "Cheapest GF pizza?", ...]
    harness.run_scope1_component_eval(
        embedding_model='text-embedding-3-small',
        llm_model='gpt-4o-mini',
        test_queries=test_queries
    )

    # Scope 2: Pipeline tests
    golden_data = {
        'question': ["Does Margherita have gluten?", ...],
        'answer': ["Standard Margherita contains gluten...", ...],
        'contexts': [[...], [...]],
        'ground_truth': ["Standard Margherita contains gluten; GF available", ...]
    }
    harness.run_scope2_pipeline_eval(
        rag_pipeline=None,  # Your RAG pipeline object
        golden_dataset=golden_data
    )

    # Scope 3: Human evaluation
    human_ratings = {
        'rater1_correctness': [3, 3, 2, 3, 3, 2, ...],  # 50 ratings
        'rater2_correctness': [3, 3, 2, 3, 3, 2, ...]   # Same 50, independent
    }
    harness.run_scope3_user_eval(human_ratings)

    # Final decision
    go_decision, reason = harness.generate_deployment_decision()

    # In CI/CD: exit(0) if go_decision else exit(1)
```

**Output example:**
```
🔬 Running Scope 1: Component Evaluation...
  Embedding Recall@10: 0.820 ✅
  LLM ROUGE-L: 0.680 ✅
🔗 Running Scope 2: Pipeline Evaluation...
  Faithfulness: 0.920 ✅
  Answer Relevancy: 0.870 ✅
  Context Precision: 0.850 ✅
👥 Running Scope 3: User Evaluation...
  Task Success Rate: 96.0% ✅
  Cohen's κ: 0.780 ✅

============================================================
DEPLOYMENT DECISION: 🚀 GO FOR PRODUCTION
Reason: All 3 evaluation scopes passed quality gates.
============================================================
```

**Integration with CI/CD:**
```yaml
# .github/workflows/evaluate.yml
name: PizzaBot Evaluation

on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run 3-Scope Evaluation
        run: python evaluate_pizzabot.py
      - name: Fail if NO-GO
        run: |
          if grep -q "NO-GO" eval_results.log; then
            echo "❌ Evaluation failed quality gates"
            exit 1
          fi
```

> 💡 **Production best practices:**
> - **Run Scope 1 on every commit** (fast, catches breaking changes)
> - **Run Scope 2 on every PR** (comprehensive, prevents regressions)
> - **Run Scope 3 before major releases** (expensive, requires human annotators)
> - Store all eval results in **time-series database** (track metric drift over time)
> - Alert on **sustained metric drops** (not single-run noise) via PagerDuty/Slack

---

## 9 · Progress Check — What We Can Solve Now

🎉 **TESTING INFRASTRUCTURE DEPLOYED**: Regression prevention achieved!

**Unlocked capabilities:**
- ✅ **Golden dataset**: 200 curated query-answer pairs covering all menu scenarios
- ✅ **RAGAS metrics**: Automated faithfulness, answer relevancy, context precision/recall scoring
- ✅ **LLM-as-judge**: GPT-4 evaluates answer quality on 1-10 scale
- ✅ **Regression testing**: Every code change runs full test suite in <2 minutes
- ✅ **A/B testing framework**: Safe parallel deployment with statistical significance
- ✅ **Production monitoring**: Real-time dashboards tracking error rate, latency, conversion

**Progress toward constraints:**

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 BUSINESS VALUE | ⚡ **MAINTAINED** | 28% conversion (target >25% ✅), +$2.50 AOV (✅), 70% labor savings (✅) |
| #2 ACCURACY | ✅ **TARGET HIT (maintained)** | ~5% error rate (target <5% ✅) — RAGAS faithfulness score 0.95+ |
| #3 LATENCY | ✅ **EXCELLENT (maintained)** | 2.5s p95 (target <3s ✅) |
| #4 COST | ⚡ **ON TRACK** | $0.015/conv (target <$0.08 ✅) |
| #5 SAFETY | ⚡ **MAINTAINED** | Zero allergen false claims in test suite |
| #6 RELIABILITY | ✅ **IMPROVED!** | Regression detection prevents production failures, uptime >99% |

**What we can solve:**

✅ **Regression prevention (2-3/week → ~0.1/week)**:
```
Before Ch.7: Manual testing only
Scenario: Developer updates system prompt for "friendlier tone"
- Manual test: 3 queries tested, all pass ✅
- Push to production
- Next day: Customer reports "Bot said Margherita is gluten-free!" ❌
- Root cause: Friendly tone prompt caused hallucination on edge case
- Cost: 4 hours debugging + rollback + hotfix

After Ch.7: Automated regression testing
Scenario: Same prompt change
- Pre-commit hook triggers test suite
- 200 queries run in 90 seconds
- RAGAS faithfulness score: 0.89 (down from 0.95) ❌
- Test fails: "Allergen claim not grounded in retrieval context"
- Commit blocked, developer notified with exact failing query
- Fix prompt, re-test, score returns to 0.95 ✅
- Push to production (no regression!)

Result: ✅ Regression caught before production!
        ✅ Zero customer impact
        ✅ Development velocity increased (safe to experiment)
```

✅ **Automated RAGAS evaluation**:
```
Test query: "What's the calorie count for a large Margherita?"

Bot response: "A large Margherita pizza is 920 calories."
Retrieved context: ["Margherita Pizza - Large (14"): 920 calories"]

RAGAS metrics:
- Faithfulness: 1.0 (claim "920 calories" directly supported by context)
- Answer Relevancy: 0.98 (directly answers question, no fluff)
- Context Precision: 1.0 (retrieved chunk is relevant)
- Context Recall: 1.0 (all required info retrieved)

Overall score: 0.995 ✅ (above 0.90 threshold)

---

Negative example (regression caught):
Test query: "What's the calorie count for a large Margherita?"

Bot response: "A large Margherita pizza is approximately 850-900 calories."
Retrieved context: ["Margherita Pizza - Large (14"): 920 calories"]

RAGAS metrics:
- Faithfulness: 0.65 ❌ (claim "850-900" contradicts context "920")
- Answer Relevancy: 0.85 (answers question but with hallucinated range)
- Context Precision: 1.0 (retrieved chunk is relevant)
- Context Recall: 1.0 (all required info retrieved)

Overall score: 0.875 ❌ (below 0.90 threshold)

Verdict: TEST FAILED - Hallucination detected!
```

✅ **A/B testing framework (safe experimentation)**:
```
Experiment: Test "add garlic bread" vs. "add drink" upsell

Control (baseline): Current "add garlic bread" upsell
- 50% of traffic
- Conversion: 28%
- AOV: $40.60
- Sample size: 1,000 visitors

Variant (test): "add drink" upsell
- 50% of traffic
- Conversion: 27.5%
- AOV: $39.80
- Sample size: 1,000 visitors

Statistical analysis:
- Conversion difference: -0.5 percentage points (not significant, p=0.32)
- AOV difference: -$0.80 (significant, p=0.04)
- Winner: Control (garlic bread upsell) ✅

Decision: Keep current upsell, abandon drink experiment
Result: ✅ Data-driven decision, no guesswork!
```

✅ **Production monitoring (real-time alerting)**:
```
Dashboard metrics (real-time):
- Error rate: 4.8% (target <5%) ✅
- Latency p95: 2.4s (target <3s) ✅
- Conversion rate: 28.2% (target >25%) ✅
- RAGAS faithfulness: 0.94 (target >0.90) ✅
- Hallucination incidents: 0/hour ✅

Alert triggered:
🚨 Error rate spike: 4.8% → 7.2% (exceeded 5% threshold)
Timestamp: 2026-04-20 14:32 UTC
Cause: RAG vector DB connection timeout (infrastructure issue)
Action: Auto-fallback to BM25 keyword search triggered
Resolution: Error rate returns to 5.1% within 2 minutes
Incident logged, team notified

Result: ✅ Proactive alerting caught issue!
        ✅ Graceful degradation prevented customer impact
        ✅ Root cause identified without customer complaints
```

**Business metrics update:**
- **Order conversion**: 28% (maintained from Ch.6, target >25% ✅)
- **Average order value**: $40.60 (maintained from Ch.6, +$2.50 vs. baseline ✅)
- **Cost per conversation**: $0.015 (maintained from Ch.6, target <$0.08 ✅)
- **Error rate**: ~5% (maintained from Ch.6, target <5% ✅)
- **Regression rate**: 2-3/week → **~0.1/week** (95% reduction) ✅
- **Development velocity**: 2 prompt iterations/day → **10+ iterations/day** (safe to experiment) ✅
- **Time to detect regressions**: 24 hours (customer complaints) → **<2 minutes** (pre-commit tests) ✅

**❌ What we can't solve yet:**
- **Brand voice inconsistency**: Bot sometimes says "Awesome choice!" (too casual) vs. "Excellent selection" (too formal) — no way to enforce consistent tone without fine-tuning
- **Cost optimization limits**: $0.015/conv is great, but 80% is GPT-4 API calls for answer generation — can't reduce further with current architecture
- **Latency floor**: 2.5s p95 is excellent, but can't break below 2s without model optimization (KV caching, quantization)
- **Adversarial attacks**: No systematic testing for prompt injection or jailbreak attempts

**Why this chapter was critical:**

Ch.7 is the **quality assurance gate** — no business metric improvements, but essential for production reliability:
1. **Regression prevention**: Every code change validated before production
2. **Fast iteration**: Safe experimentation without fear of breaking production
3. **Data-driven decisions**: A/B testing framework enables evidence-based optimization
4. **Proactive monitoring**: Catch issues before customers complain
5. **Compliance**: Automated testing required for enterprise deployment

**Next chapter**: [Fine-Tuning](../ch10_fine_tuning) tackles brand voice consistency and cost reduction via LoRA adapters → **30% conversion, $0.008/conv** (50% cost reduction), consistent Mamma Rosa's tone in every response.

**Key interview concepts from this chapter:**

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| The four RAGAS metrics — what each measures and what a low score implies | Describe the diagnostic matrix: what does low faithfulness + high context precision mean? | Saying accuracy is sufficient for RAG evaluation — accuracy needs ground truth; RAGAS doesn't |
| LLM-as-judge approach — strengths and limitations | How do you evaluate an agent when there is no single correct answer? | Assuming benchmark scores predict application performance |
| Hallucination detection via self-consistency or NLI | What is NDCG@k and why is it better than recall@k for retrieval evaluation? | Confusing context recall (retrieval metric) with answer recall (generation metric) |
| The 3-level evaluation hierarchy | How would you set up a regression test suite for a RAG pipeline? | Evaluating only the LLM component and ignoring retrieval quality |
| **BLEU / ROUGE for generative evaluation:** BLEU measures n-gram precision of generated text against references (primarily translation); ROUGE measures recall-oriented n-gram overlap (primarily summarisation). Both are reference-based and require gold outputs. Key weakness: a factually correct generation using different words scores low | "What are the limitations of BLEU and ROUGE?" | "High BLEU / ROUGE means high quality" — they measure lexical overlap, not factual accuracy or coherence; LLM-as-judge has higher correlation with human preference on open-ended tasks |
| **Human evaluation methodology:** pairwise preference ("A or B?") scales better than absolute rating (avoids anchoring); inter-annotator agreement (Cohen's κ) should be reported; annotators must be blind to system identity. MT-Bench and MMLU are the dominant automated proxy benchmarks for instruction-following and knowledge | "How would you run a rigorous human evaluation for a new LLM?" | "High MT-Bench score means the model is production-ready" — MT-Bench measures generic instruction following on 80 multi-turn problems; domain performance, safety, latency, and cost are not captured |

---

## 10 · Bridge to Chapter 8

Ch.7 unlocked automated testing and regression prevention. But the evaluation suite revealed three systematic gaps:

1. **Brand voice drift**: RAGAS scores 0.95 faithfulness, but tone varies ("Awesome!" vs. "Excellent") — prompt engineering can't enforce consistent style
2. **Cost floor**: $0.015/conv is 80% GPT-4 API calls — RAG and caching already optimized, need model-level optimization
3. **Upsell quality**: A/B testing shows garlic bread upsells work, but conversion gains plateau — need smarter, context-aware suggestions

These aren't retrieval problems (RAG already solves those). They're **generation problems** — the model itself needs adaptation. Chapter 8 (Fine-Tuning) tackles this via **LoRA adapters**: train a lightweight layer on Mamma Rosa's brand voice + successful upsell patterns. Expected impact: **30% conversion** (consistent tone builds trust), **$0.008/conv** (50% cost reduction via smaller fine-tuned model), **2.0s latency** (faster inference).

> *A system you cannot measure is a system you cannot improve. Build the eval suite before you build the application — not after.*

## Illustrations

![Evaluating AI systems — RAGAS radar, reasoning-trace checklist, component-level eval, hallucination gate](img/Evaluating%20AI%20Systems.png)
