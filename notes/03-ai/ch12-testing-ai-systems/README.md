# Ch.12 — Testing AI Systems: Catching the 14% Wrong-Order Bug

**Track**: AI (03-ai) | **Chapter**: 12 | **Grand Challenge**: Mamma Rosa's PizzaBot
**Previous**: [ch11_advanced_agentic_patterns](../ch11_advanced_agentic_patterns/) | **Next**: Production deployment

---

> **A brief history.** In June 2015, a Google Photos engineer opened the app to review a new feature and noticed the autocreated album label: *"Gorillas."* Inside: photos of two Black users. The classifier had labelled real people with an animal category — not because the engineers had trained something malicious, but because the test set had never included enough dark-skinned faces for anyone to notice the failure mode. The fix Google shipped was a search index band-aid: remove "gorilla" from the label vocabulary. There was nothing to prevent the next version from making the same class of mistake, because the *discipline* of testing AI systems for adversarial and underrepresented inputs didn't yet exist.
>
> The pattern repeated. Amazon's hiring AI quietly penalised résumés that included the word "women's" — trained on a decade of historical hiring data, it had learned that women were hired less often and then *optimised for exactly that signal*. Nobody caught it because the evaluation used random holdout, not adversarial test cases. In March 2017, Microsoft's Tay chatbot learned to tweet hate speech within sixteen hours of public availability; the test suite had never simulated users actively trying to poison the model. Each incident shared the same root cause: nobody had checked what the model would do on the inputs it would actually see in production.
>
> The field's answer arrived in 2020. Marco Ribeiro and colleagues at Microsoft Research published *"Beyond Accuracy: Behavioral Testing of NLP Models"* (ACL 2020) — a framework called **CheckList** that gave practitioners three concrete test types: *behavioural* (does the model do the right thing?), *invariance* (does swapping irrelevant words break it?), and *directional* (does more of X produce more of Y?). Simultaneously, the ML Engineering team at Google formalised the **ML Test Score** (Breck et al., 2017) — a 28-point rubric that production ML systems must pass before launch. The conclusion both papers reached: production AI systems need *more* testing infrastructure than traditional software, not less — because the bugs are statistical and silent, not deterministic and loud.
>
> **Where you are in the curriculum.** Ch.1–11 built PizzaBot end-to-end: from tokenization (Ch.1) through prompt engineering (Ch.2), reasoning (Ch.3), RAG grounding (Ch.4–5), tool orchestration (Ch.6), safety guardrails (Ch.7), evaluation metrics (Ch.8), cost/latency optimization (Ch.9), fine-tuning (Ch.10), and advanced agentic patterns (Ch.11). **PizzaBot v2.0 achieves 99.2% edge-case accuracy and $0.18/conversation in Ch.11. The CEO says it's ready to launch.** This chapter asks: *how do you prove that?* Specifically: if production is showing a 14% wrong-order rate and you need to diagnose and fix it before launch, what tests do you write? Ch.8 gave you evaluation metrics — RAGAS, conversion rate, hallucination rate. This chapter turns those metrics into **executable tests** that run in CI/CD on every pull request.
>
> **Notation in this chapter:**
> `assert_output` — test assertion on model output; `retrieval_hit_rate` — fraction of queries where correct document is retrieved; `rag_pipeline` — full chain: ingestion → embedding → retrieval → generation; `pytest.fixture` — reusable test setup; `@hypothesis.given` — property-based test decorator; `wrong_order_rate` — fraction of orders with wrong items/price; `invariant` — test input where output should not change; `directional` — test where output should change monotonically with input.

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Launch **Mamma Rosa's PizzaBot** — satisfying 6 production constraints:
> 1. **BUSINESS VALUE**: >25% conversion + +$2.50 AOV + 70% labor savings
> 2. **ACCURACY**: <5% error rate on menu queries + order placement
> 3. **LATENCY**: <3s p95 response time
> 4. **COST**: <$0.08 per conversation average
> 5. **SAFETY**: Zero successful prompt injections + appropriate refusals
> 6. **RELIABILITY**: >99% uptime + graceful degradation when tools fail

**What we know so far:**

✅ **Ch.1–11 built the full stack:**
- Ch.1: LLM tokenization, context windows, sampling
- Ch.2: Prompt engineering → structured outputs
- Ch.3: Chain-of-thought → multi-constraint queries
- Ch.4–5: RAG + vector DB → <5% error rate ✅ (Constraint #2 ACHIEVED)
- Ch.6: ReAct → tool calling + proactive upselling → 28% conversion ✅ (Constraint #1 ACHIEVED)
- Ch.7: Safety guardrails → zero attacks ✅ (Constraint #5 ACHIEVED)
- Ch.8: Evaluation metrics → RAGAS, conversion, hallucination rate measured
- Ch.9: KV caching + model tiers → <$0.08/conv ✅, <3s p95 ✅ (Constraints #3, #4 ACHIEVED)
- Ch.10: LoRA fine-tuning → brand voice + further cost reduction
- Ch.11: Reflection + debate + hierarchical orchestration → 99.2% accuracy ✅ (Constraint #2 margin improved)

**The CEO says: "Launch it. We've hit all six constraints."**

**But operations is seeing this:**

❌ **Production shows a 14% wrong-order rate since the soft launch last Tuesday.**

```
Week 1 Production Sample (200 orders):
  ✅ 172 orders: correct item, price, delivery zone
  ❌  28 orders: wrong item OR wrong price OR wrong zone
      - Wrong menu item:    19 orders  (68% of failures)
      - Wrong price:         5 orders  (18% of failures)
      - Wrong delivery zone: 4 orders  (14% of failures)

Total wrong-order rate: 28 / 200 = 14%
```

**What's blocking us:**

🚨 **You have no tests.** The evaluation in Ch.8 measured aggregate metrics on a held-out set. But aggregate RAGAS scores don't catch production failure modes:
- The RAG retrieval is returning **the lunch menu instead of the dinner menu** for 14% of evening orders — same embedding distance, wrong temporal context
- No test checks retrieval correctness per time-of-day
- No test verifies that the price in the generated response matches the menu document retrieved
- No CI gate prevents a menu update from breaking pricing retrieval

**Business impact:**
- 28 wrong orders/week × $25 average order = **$700/week in refunds + customer trust damage**
- At 10,000 orders/day (launch scale): 14% wrong-order rate = **1,400 wrong orders/day = $35,000/day in refunds**
- CEO reaction: "You told me this was 99.2% accurate. Pull the plug until it's provably fixed."

**What this chapter unlocks:**

🚀 **A test suite that:**

1. **Diagnoses** the 14% wrong-order bug (retrieval returns wrong menu document for evening orders)
2. **Unit tests** every component of the RAG pipeline in isolation
3. **Integration tests** the full ingestion → retrieval → generation pipeline
4. **Model tests** verify shape, invariance, and directional properties
5. **Property-based tests** generate adversarial inputs automatically with hypothesis
6. **CI/CD gate** (GitHub Actions) runs the full suite on every pull request

**Expected outcome:**
- `wrong_order_rate`: 14% → **0%** (bug found and fixed by retrieval tests)
- Constraint #2 ACCURACY: re-confirmed with test evidence, not just eval metrics
- CI gate catches future menu updates before they reach production

---

## 📽️ Animation Reference

> ⚠️ **Placeholder for animation assets** (to be generated by animation subagents):
>
> **Testing pipeline animation** (see `img/`):
> - `img/testing-pipeline.gif`: Unit → Integration → Model test layers, failures lighting up in red, fix propagating green
>
> Reference inline in the testing implementation section below.

---

## 1 · The Core Idea

The CEO's §0 ultimatum — *"Pull the plug until it's provably fixed"* — exposes a gap: Ch.8's RAGAS context precision of 0.88 never flagged the dinner/lunch retrieval mismatch, because aggregate metrics don't fail when a specific query-document pair breaks. That shift — from *measuring* accuracy to *asserting* correctness — is what this section establishes.

**Plain English:** Traditional software tests check *deterministic* functions — `add(2, 3) == 5`. AI systems are probabilistic: the same input can produce different outputs. But most AI bugs aren't in the randomness — they're in the **pipeline joints**: the transition from document ingestion to retrieval, from retrieval to generation, from generation to order confirmation.

The PizzaBot 14% wrong-order rate isn't a model bug. It's a **retrieval bug**: when a customer orders at 7pm, the RAG retrieval returns lunch-menu documents (because both menus use "pepperoni pizza" and the embedding doesn't capture time-of-day). The fix is one line. The test that would have caught it is four lines. Neither existed.

**The key shift:** Ch.8 measured *accuracy* (aggregate). This chapter tests *correctness* (specific, executable assertions). Accuracy is a number. Tests are runnable proof.

> 💡 **Insight:** A RAGAS context precision of 0.86 doesn't tell you *which* documents are wrong. A `test_retrieval_time_of_day` test tells you exactly which query-document pair breaks.
> **Rule:** For every aggregate metric that drops in production, there should be a targeted test that reproduces it.

---

## 1.5 · The Practitioner Workflow — Your 3-Tier Test Pyramid

The §0 14% wrong-order bug was invisible to aggregate metrics — RAGAS context precision 0.88, above threshold — yet 28 orders per week were wrong. The three-tier pyramid below maps each §0 failure mode to the test tier that would have caught it before the soft launch.

**Before diving into test implementation, understand the three-tier strategy every AI system needs:**

> 🎯 **What you'll build by the end:** A production-ready test suite with 15+ unit tests (component contracts), 8+ integration tests (pipeline correctness), and 200+ property-based adversarial tests (edge-case resilience) — running in CI/CD on every pull request, blocking merges when tests fail.

```
Tier 1: UNIT                  Tier 2: INTEGRATION            Tier 3: ADVERSARIAL
────────────────────────────────────────────────────────────────────────────
Test components in isolation  Test end-to-end pipeline       Generate adversarial inputs

• Ingestion: doc → chunks    • Query at 7pm → dinner menu   • hypothesis generates 200
• Retrieval: query → docs    • Full RAG pipeline correct      pizza orders automatically
• Generation: docs → answer  • Latency < 3s threshold       • Prompt injection attempts
• Parser: text → order       • Price match: doc vs answer   • Edge cases: 0 quantity,
                                                               empty toppings, Unicode

→ DECISION:                   → DECISION:                    → DECISION:
  Component contract broken?    Pipeline joint broken?         System robust to attacks?
  • Coverage ≥ 80%?             • Retrieval hit rate ≥ 90%?    • 0 prompt injections pass?
  • All assertions pass?        • Response price matches doc?  • All valid orders parse?
  • Fast (<1s total)?           • Latency < 3s?                • Pricing monotonic?
```

> 💡 **Test Pyramid Economics:** Unit tests are free and fast (<1s). Integration tests cost ~$0.01 per run (real LLM API calls). Adversarial tests are front-loaded cost (write once, generate hundreds). The pyramid shape (many unit, fewer integration, strategic adversarial) optimizes for **fast feedback** and **low CI cost**.

---

### The Three-Tier Rationale

**Why three tiers, not one "comprehensive test"?**

| Problem | If you only have... | What breaks in production |
|---------|---------------------|---------------------------|
| **Tier 1 missing** | Integration + adversarial only | Individual component bugs (retrieval metadata filter broken) → expensive to debug in full pipeline, slow CI feedback |
| **Tier 2 missing** | Unit + adversarial only | Pipeline joint failures (price in response doesn't match retrieved doc) → components work in isolation but fail when connected |
| **Tier 3 missing** | Unit + integration only | Edge cases (Unicode pizza names, negative quantities, prompt injection) → real users hit untested inputs |

**Real-world example from PizzaBot production:**

```
Week 1 soft launch (no Tier 2 integration tests):
  ✅ Unit tests: All components pass (retrieval works, generation works)
  ✅ Adversarial tests: Property-based tests pass (pricing monotonic)
  ❌ PRODUCTION: 14% wrong-order rate

Root cause: Retrieval returns correct documents (unit test ✅) but generation
uses wrong document because metadata filter wasn't applied (Tier 2 would catch).

Fix: Add test_e2e_dinner_order_returns_correct_price (Tier 2)
      → Bug caught in CI before next deploy
```

**The lesson:** Unit tests verify contracts. Integration tests verify communication. Adversarial tests verify resilience. You need all three.

---

### Progressive Test Coverage Strategy

**How to build the test suite incrementally (don't try to implement all three tiers at once):**

```
Day 1-2: Tier 1 Foundation
─────────────────────────────────
Write 10-15 unit tests covering:
• Ingestion (3 tests)
• Retrieval (4 tests)  ← Priority: test_retrieval_time_of_day catches the 14% bug
• Generation (3 tests)
• Parser (2 tests)

✅ Success criteria: 80% code coverage, all tests <1s, zero API cost
→ Can now refactor components without fear

Day 3-4: Tier 2 Integration
─────────────────────────────────
Add 5-8 E2E pipeline tests:
• test_e2e_dinner_order_returns_correct_price  ← Catches the production bug
• test_e2e_allergen_query_mentions_allergen_source
• test_e2e_out_of_stock_item_not_offered
• test_e2e_latency_under_3s

✅ Success criteria: Wrong-order rate 14% → 0%, CI cost <$0.10/run
→ Can now deploy with confidence

Day 5: Tier 3 Adversarial
─────────────────────────────────
Add property-based tests:
• hypothesis generates 200 pizza orders → all parse without error
• hypothesis generates price scenarios → all monotonic
• Prompt injection test suite (10 attacks)

✅ Success criteria: 0 prompt injections succeed, 0 edge-case crashes
→ System is production-hardened
```

> 💡 **Industry Standard: pytest + CI/CD integration**
> ```python
> # pytest.ini
> [pytest]
> markers =
>     unit: Unit tests (fast, no API cost)
>     integration: Integration tests (slow, costs money)
>     adversarial: Property-based and prompt injection tests
> ```
> ```bash
> # In CI (GitHub Actions):
> pytest -m "unit"              # Every PR (free, <1s)
> pytest -m "integration"       # Push to main only ($0.05/run)
> pytest -m "adversarial"       # Nightly (front-loaded cost)
> ```
> **When to use:** Always. This is the industry-standard pattern for test organization in AI systems.
> **See also:** [pytest markers documentation](https://docs.pytest.org/en/stable/example/markers.html)

---

## 2 · Running Example: Diagnosing the 14% Wrong-Order Bug

You're on call. Operations has flagged the wrong-order rate. The CEO wants a root-cause analysis before 9am tomorrow. You have the production logs.

### Step 1: Reproduce the Bug

```python
# Educational: manual pipeline call to reproduce the failure
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # never hardcode

def query_pizzabot(question: str, context_time: str = "19:00") -> str:
    """Query PizzaBot with explicit time context."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are PizzaBot. Current time: {context_time}."},
            {"role": "user", "content": question}
        ],
        temperature=0,  # deterministic for testing
    )
    return response.choices[0].message.content

# Reproduce the failing case
answer = query_pizzabot("What's your pepperoni pizza?", context_time="19:00")
print(answer)
# Output: "Our pepperoni pizza is $10.99 (lunch special)."
# ❌ WRONG! Dinner price is $14.99. The lunch menu was retrieved at 7pm.
```

You now know the symptom. The price is wrong because the retrieved document is wrong.

### Step 2: Write a Failing Test First

```python
# tests/test_retrieval.py
import pytest

def test_retrieval_returns_dinner_menu_at_7pm(rag_client):
    """Retrieval must return dinner menu for evening queries."""
    docs = rag_client.retrieve(
        query="pepperoni pizza price",
        metadata_filter={"time_of_day": "dinner"}
    )
    assert len(docs) > 0, "No documents retrieved"
    assert docs[0].metadata["menu_type"] == "dinner", (
        f"Expected 'dinner', got '{docs[0].metadata['menu_type']}'"
    )
```

Run it: **RED** — `AssertionError: Expected 'dinner', got 'lunch'`.

That's the bug. The retrieval isn't filtering by `menu_type`. The fix adds one line to the metadata filter. Re-run: **GREEN**.

This is the failure-first pattern for AI testing: **write the failing test, confirm it names the bug exactly, then fix it.**

> ⚠️ **Warning:** The temptation is to write tests *after* fixing the bug. Don't. Write the test first so you can confirm it fails for the right reason — not because of an unrelated error. If your test passes before the fix, it's not testing what you think.
> **Rule:** A test that has never been seen to fail is a test you can't trust.

> 💡 **Failing-test-first → zero wrong orders:** Writing the test before the fix confirmed it failed for the right reason — not for a coincidental error. At §0's launch scale of 10,000 orders/day, a missing metadata filter is a $35,000/day liability. One four-line test, written RED first, eliminates that exposure permanently once the fix turns it GREEN.

---

## 3 · Component Testing — Isolate and Assert Contracts

The §0 retrieval bug — lunch menu returned at 7pm — was invisible at the pipeline level: retrieval returned *a* document, generation produced *a* price, and no aggregate metric failed. Unit tests expose the exact contract each component broke: the metadata filter that was never applied.

**What you're doing in this tier:** Test each component of the RAG pipeline in complete isolation. Retrieval doesn't call the LLM. Generation uses a mocked retrieval response. Ingestion doesn't hit a real database. **Goal: verify component contracts with zero API cost, <1s execution time, 80%+ code coverage.**

Unit tests isolate one function and assert its contract. For a RAG pipeline, the contracts are:

| Component | What it should guarantee |
|-----------|--------------------------|
| **Ingestion** | Every document is chunked, embedded, and stored with correct metadata |
| **Retrieval** | Query returns relevant documents; metadata filters work |
| **Generation** | Response mentions the retrieved item name; price in response matches retrieved document |
| **Reproducibility** | Same query with `temperature=0` returns identical response |

### 3.1 — Fixtures: The Reusable Test Infrastructure

```python
# tests/conftest.py
import pytest
import os
from unittest.mock import MagicMock

@pytest.fixture(scope="session")
def rag_client():
    """
    Real RAG client pointed at test vector DB.
    Using scope='session' so the index is built once, not per test.
    """
    from src.rag_pipeline import RAGClient
    client = RAGClient(
        index_url=os.environ["TEST_VECTOR_DB_URL"],  # separate test index
        embedding_model="text-embedding-3-small",
    )
    yield client

@pytest.fixture
def mock_llm():
    """Deterministic mock LLM — no API calls in unit tests."""
    mock = MagicMock()
    mock.complete.return_value = "Large pepperoni pizza is $14.99 for delivery."
    return mock

@pytest.fixture
def sample_menu_docs():
    """Minimal menu document set for ingestion tests."""
    return [
        {"content": "Pepperoni pizza — large $14.99", "metadata": {"menu_type": "dinner", "item": "pepperoni"}},
        {"content": "Pepperoni pizza — large $10.99 (lunch special)", "metadata": {"menu_type": "lunch", "item": "pepperoni"}},
        {"content": "Margherita pizza — large $12.99", "metadata": {"menu_type": "dinner", "item": "margherita"}},
    ]
```

> 💡 **Insight:** `scope="session"` on the `rag_client` fixture builds the vector index once per test run, not once per test. On a 1,000-doc corpus this saves ~40s per test run.
> **Rule:** Use `scope="session"` for fixtures that are expensive to build (vector indexes, DB connections). Use default (function) scope for anything that mutates state.

### 3.2 — Ingestion Tests

```python
# tests/test_ingestion.py
import pytest

def test_all_menu_items_are_indexed(rag_client, sample_menu_docs):
    """Every document must be retrievable after ingestion."""
    rag_client.ingest(sample_menu_docs)

    for doc in sample_menu_docs:
        item_name = doc["metadata"]["item"]
        results = rag_client.retrieve(query=item_name, top_k=5)
        retrieved_items = [r.metadata["item"] for r in results]
        assert item_name in retrieved_items, (
            f"'{item_name}' was ingested but not retrievable"
        )

def test_ingestion_preserves_metadata(rag_client, sample_menu_docs):
    """Metadata fields must survive ingestion unchanged."""
    rag_client.ingest(sample_menu_docs)
    results = rag_client.retrieve(query="pepperoni", top_k=5)

    for result in results:
        assert "menu_type" in result.metadata, "menu_type metadata dropped during ingestion"
        assert result.metadata["menu_type"] in ("lunch", "dinner"), (
            f"Invalid menu_type: '{result.metadata['menu_type']}'"
        )

def test_chunk_size_within_bounds(rag_client, sample_menu_docs):
    """Chunks must not exceed context window limit."""
    MAX_CHUNK_TOKENS = 512
    chunks = rag_client.chunk(sample_menu_docs, chunk_size=MAX_CHUNK_TOKENS)

    for chunk in chunks:
        # Rough estimate: 1 token ≈ 4 chars
        estimated_tokens = len(chunk.content) // 4
        assert estimated_tokens <= MAX_CHUNK_TOKENS, (
            f"Chunk exceeds {MAX_CHUNK_TOKENS} tokens: {estimated_tokens} estimated"
        )
```

### 3.3 — Retrieval Tests: The Bug We Found

**Retrieval correctness** is measured by **Hit@K** — the fraction of known queries that return the correct document in the top K retrieved results:

$$\text{Hit@K} = \frac{1}{|Q|}\sum_{q \in Q} \mathbf{1}[\text{relevant doc} \in \text{top-}K \text{ results for } q]$$

*Verbal gloss:* For each query $q$, check whether the relevant document appears in the first $K$ retrieved results. Summing the binary hits and dividing by the number of queries gives the fraction that "hit." Hit@1 = 0.90 means 90% of queries surface the correct document at rank 1 — the position that matters most because generation quality degrades when the relevant document isn't first.

> 💡 **Business consequence of Hit@K dropping:** If Hit@1 falls from 90% to 75%, 25% of queries retrieve the wrong menu document before the LLM generates a single token. At 10,000 orders/day that is 2,500 wrong-document retrievals — responses where the quoted price or item may come from the wrong context chunk. Every point of Hit@K below your threshold directly inflates your wrong-order rate.

```python
# tests/test_retrieval.py
import pytest

def test_retrieval_returns_dinner_menu_at_7pm(rag_client, sample_menu_docs):
    """
    The 14% wrong-order bug: retrieval was returning lunch menu at 7pm.
    This test was RED before the fix, GREEN after.
    """
    rag_client.ingest(sample_menu_docs)

    docs = rag_client.retrieve(
        query="pepperoni pizza price",
        metadata_filter={"menu_type": "dinner"}
    )

    assert len(docs) > 0, "No dinner menu documents retrieved"
    assert all(d.metadata["menu_type"] == "dinner" for d in docs), (
        f"Lunch document returned in dinner query: "
        f"{[d.metadata['menu_type'] for d in docs]}"
    )

def test_retrieval_hit_rate(rag_client, sample_menu_docs):
    """
    At least 90% of known queries should retrieve the correct document at rank 1.
    Hit rate < 90% means embeddings or chunking strategy needs rework.
    """
    rag_client.ingest(sample_menu_docs)

    test_queries = [
        ("pepperoni pizza price dinner",    "pepperoni"),
        ("margherita large how much",       "margherita"),
        ("gluten free pepperoni",           "pepperoni"),
    ]

    hits = 0
    for query, expected_item in test_queries:
        results = rag_client.retrieve(query=query, top_k=1)
        if results and results[0].metadata["item"] == expected_item:
            hits += 1

    hit_rate = hits / len(test_queries)
    assert hit_rate >= 0.90, (
        f"Retrieval hit@1 = {hit_rate:.0%} — below 90% threshold"
    )

def test_empty_retrieval_does_not_crash(rag_client):
    """Retrieval with no matching documents must return empty list, not raise."""
    results = rag_client.retrieve(
        query="xyzzy plover zorkmid",  # nonsense — no match expected
        top_k=3
    )
    assert isinstance(results, list)
    assert len(results) == 0
```

> 💡 **What these retrieval tests catch:**
> - **Metadata filter bugs** (test_retrieval_returns_dinner_menu_at_7pm) → Caught the 14% wrong-order bug where lunch menu was served at dinner time
> - **Embedding quality issues** (test_retrieval_hit_rate) → Detects when chunking or embedding strategy degrades below 90% hit rate
> - **Edge case crashes** (test_empty_retrieval_does_not_crash) → Prevents production crashes when users query items not in the menu

### 3.4 — Generation Tests

```python
# tests/test_generation.py
import pytest

def test_generated_price_matches_retrieved_document(mock_llm, rag_client, sample_menu_docs):
    """
    Price in response must come from retrieved document, not hallucinated.
    Checks the joint between retrieval and generation.
    """
    rag_client.ingest(sample_menu_docs)

    from src.rag_pipeline import generate_with_rag

    response, retrieved_docs = generate_with_rag(
        query="How much is a large pepperoni pizza for dinner?",
        rag_client=rag_client,
        metadata_filter={"menu_type": "dinner"},
        llm=mock_llm,
        return_sources=True,
    )

    # Extract price from retrieved doc
    retrieved_price = extract_price(retrieved_docs[0].content)  # "$14.99"

    # Price in response must match
    assert retrieved_price in response, (
        f"Response price doesn't match retrieved doc. "
        f"Retrieved: {retrieved_price}, Response: '{response}'"
    )

def test_response_is_deterministic_at_temperature_zero(rag_client, sample_menu_docs):
    """
    Same query with temperature=0 must produce identical response.
    Non-determinism means you can't write reliable assertions.
    """
    rag_client.ingest(sample_menu_docs)

    from src.rag_pipeline import generate_with_rag
    from src.llm_client import LLMClient

    llm = LLMClient(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini",
        temperature=0,
    )

    query = "What is the price of a large pepperoni pizza?"
    response_1 = generate_with_rag(query, rag_client, llm=llm)
    response_2 = generate_with_rag(query, rag_client, llm=llm)

    assert response_1 == response_2, (
        f"Non-deterministic response at temperature=0:\n"
        f"  Run 1: {response_1}\n"
        f"  Run 2: {response_2}"
    )
```

> 💡 **What these generation tests catch:**
> - **Price hallucination** (test_generated_price_matches_retrieved_document) → Prevents the LLM from inventing prices not in retrieved documents (would cause refunds)
> - **Non-deterministic responses** (test_response_is_deterministic_at_temperature_zero) → Catches when temperature=0 doesn't produce identical outputs (breaks assertion-based testing)
> - **RAG-generation disconnect** → Validates that the LLM actually uses retrieved context instead of generating from memorized training data

> ⚠️ **Warning:** `test_response_is_deterministic_at_temperature_zero` will **fail** for some LLM providers even at `temperature=0`. OpenAI's gpt-4o produces identical outputs at temperature=0 for the same input on the same API version — but Anthropic's Claude and open-source models with batching may not. Always validate this assumption for your provider before building tests that depend on it.
> **Rule:** Mock the LLM in unit tests; reserve real API calls for integration tests only.

---

> 💡 **Unit test verdict:** 15 tests, 89% coverage, <1s run time, zero API cost — `test_retrieval_returns_dinner_menu_at_7pm` isolated and confirmed the 14% wrong-order bug.
> ➡️ Fix any failing unit tests before proceeding; coverage < 80% means uncovered error-handling branches — then move to Tier 2 integration tests.

---

## 4 · Pipeline Testing — End-to-End Correctness

Unit tests in §3 confirmed each component's contract independently — but the §0 bug only appeared when retrieval and generation *communicated*: retrieval returned the right document count, generation used the wrong one because no metadata filter was threaded end-to-end. Pipeline tests are the only tier that catches joint failures.

**What you're doing in this tier:** Fire the full pipeline with real components. No mocks. Ingestion → embedding → vector DB → retrieval → LLM generation → parsed response. **Goal: verify pipeline joints work correctly, catch the bugs that only appear when components communicate (like the 14% wrong-order bug).**

Integration tests fire the full pipeline: ingestion → retrieval → generation. They don't mock anything. They use a real (small) test corpus and real API calls.

> ⚡ **Constraint:** Integration tests cost money (real LLM API calls). At $0.002/1k tokens and 500 tokens/test, 100 integration tests = $0.10. Keep the integration suite under 50 tests. Unit tests should catch 90% of bugs for free.

```mermaid
flowchart LR
    A[Menu docs\n ingest] --> B[Vector DB\n test index]
    B --> C[Query +\n metadata filter]
    C --> D[Retrieved docs\n dinner menu]
    D --> E[LLM\n generate]
    E --> F[Order\n response]
    F --> G{Assert\n price match}
    G -->|✅ pass| H[green]
    G -->|❌ fail| I[red]

    style A fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style B fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style D fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style E fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style F fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style G fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style H fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style I fill:#b91c1c,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

```python
# tests/integration/test_rag_pipeline_e2e.py
import pytest
import os

@pytest.fixture(scope="module")
def e2e_rag():
    """Full integration pipeline using real API."""
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline(
        vector_db_url=os.environ["TEST_VECTOR_DB_URL"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        temperature=0,
    )

def test_e2e_dinner_order_returns_correct_price(e2e_rag):
    """
    Full pipeline: query at 7pm must return dinner price, not lunch price.
    This is the integration reproduction of the 14% wrong-order bug.
    """
    response = e2e_rag.answer(
        query="How much is a large pepperoni pizza?",
        metadata_filter={"menu_type": "dinner"},
    )

    assert "$14.99" in response, (
        f"Expected dinner price $14.99 in response, got:\n{response}"
    )
    assert "$10.99" not in response, (
        f"Lunch price $10.99 leaked into dinner response:\n{response}"
    )

def test_e2e_allergen_query_mentions_allergen_source(e2e_rag):
    """
    Allergen queries must cite the allergen document, not make up information.
    """
    response = e2e_rag.answer(
        query="Is the pepperoni pizza gluten-free?",
    )

    gluten_keywords = ["gluten", "wheat", "crust"]
    assert any(kw in response.lower() for kw in gluten_keywords), (
        f"Allergen response didn't mention gluten/wheat/crust:\n{response}"
    )

def test_e2e_out_of_stock_item_not_offered(e2e_rag):
    """
    PizzaBot must not offer menu items marked out_of_stock=True.
    """
    response = e2e_rag.answer(
        query="Do you have the Volcano Special pizza?",
        metadata_filter={"out_of_stock": False},
    )

    # The Volcano Special is out of stock in our test corpus
    assert "volcano special" not in response.lower() or "not available" in response.lower(), (
        f"Out-of-stock item offered in response:\n{response}"
    )

def test_e2e_latency_under_3s(e2e_rag):
    """
    Constraint #3: p95 latency <3s.
    This integration test measures a single call — use load testing for p95.
    """
    import time

    start = time.perf_counter()
    _ = e2e_rag.answer(query="Large pepperoni pizza for delivery")
    elapsed = time.perf_counter() - start

    assert elapsed < 3.0, (
        f"Response time {elapsed:.2f}s exceeded 3s target"
    )
```

> ➡️ **Forward pointer:** These integration tests run locally but become flaky in CI if the vector DB index isn't stable. [DevOps Ch.4 — CI/CD Pipelines](../../07-devops_fundamentals/) shows how to provision a disposable test index in GitHub Actions using docker-compose, so the integration suite gets a fresh environment on every PR.

---

> 💡 **Integration test verdict:** 8 tests covering dinner-time ordering, allergen queries, out-of-stock, latency (<3s), case invariance — `test_e2e_dinner_order_returns_correct_price` catches the 14% wrong-order production bug.
> ➡️ If the dinner-time test fails, check the metadata filter in `rag_client.retrieve()` — it must include `{"menu_type": "dinner"}` for evening queries; then move to Tier 3 adversarial tests.

> 💡 **Industry Standard: LangChain Callbacks for Test Observability**
> ```python
> from langchain.callbacks import StdOutCallbackHandler
>
> # In integration tests, add callback to see exactly what LLM receives/returns
> llm = ChatOpenAI(
>     model="gpt-4o-mini",
>     temperature=0,
>     callbacks=[StdOutCallbackHandler()],  # Prints every LLM call in test output
> )
> ```
> **When to use:** Integration test debugging. When a test fails and you need to see the exact prompt sent to the LLM and the raw response.
> **Common alternatives:** `langfuse` (cloud tracing), `langsmith` (LangChain-native tracing), `wandb` (experiment tracking)
> **See also:** [LangChain Callbacks documentation](https://python.langchain.com/docs/modules/callbacks/)

---

## 5 · Stress Testing — Shape, Invariance, and Directional Properties

§3–4 diagnosed and fixed the 28 named wrong orders from §0. But the §0 production sample was only 200 orders — at 10,000/day launch scale the real threat is failure modes that never appeared in that sample: users typing in caps, appending their name, or ordering two pizzas when the pricing logic assumed one. Property tests in this section probe that unexplored surface automatically.

**What you're doing in this tier:** Test the system's resilience to edge cases, adversarial inputs, and property violations. Don't hand-write 100 test cases — use **property-based testing** (hypothesis library) to generate them automatically. Verify that the model's behavioral properties hold for *all* inputs, not just your happy-path examples.

**What you just built:**
- ✅ 15 unit tests covering ingestion (3), retrieval (4), generation (3), parsing (2), determinism (1), chunk size (1), metadata preservation (1)
- ✅ 89% code coverage on `src/rag_pipeline.py`, `src/pricing.py`, `src/order_parser.py`
- ✅ Test suite runs in <1s with zero API cost (all LLM calls mocked)
- ✅ `test_retrieval_returns_dinner_menu_at_7pm` — RED before fix, GREEN after (caught the 14% wrong-order bug in isolation)

**What it means:**
- **Component contracts are verified:** Each function in the RAG pipeline does exactly what its signature promises
- **Refactoring is safe:** Can now change implementation details without breaking tests
- **Fast feedback loop:** Developers get test results in <1s on every save
- **Zero CI cost barrier:** Unit tests can run on every commit without budget concerns

**What to do next:**

→ **If any unit test fails:** Fix immediately. Unit test failures mean a broken contract — the component doesn't do what its function signature promises. Do not proceed to integration tests until all unit tests are GREEN.

→ **If coverage < 80%:** Add unit tests for uncovered branches. Common gaps: error handling (`except` blocks), edge cases (empty lists, None values), and alternative code paths (`else` branches).

→ **If any test is slow (>100ms):** You're calling a real API or database. Replace with a mock. Unit tests must be fast or developers will skip them.

→ **Ready to proceed:** Your components work in isolation. Now test if they work **together** → move to Tier 2 integration testing.

---

**Three types:**

| Type | Question | PizzaBot example |
|------|----------|-----------------|
| **Shape** | Does the output have the right structure? | Response is a string, non-empty, <500 chars |
| **Invariance** (INV) | Does irrelevant variation break the output? | "large pepperoni" = "LARGE PEPPERONI" |
| **Directional** (DIR) | Does relevant variation change output monotonically? | More items → higher total price |

### 5.1 — Shape Tests

```python
# tests/test_model_properties.py
import pytest

def test_response_is_non_empty_string(e2e_rag):
    """Model must return a non-empty string for any valid query."""
    response = e2e_rag.answer(query="Large pepperoni for delivery")

    assert isinstance(response, str), f"Expected str, got {type(response)}"
    assert len(response.strip()) > 0, "Response is empty string"

def test_response_length_within_bounds(e2e_rag):
    """
    Response must be ≥ 20 chars (not a refusal) and ≤ 500 chars (not verbose).
    Verbosity > 500 chars correlates with hallucination in our fine-tuned model.
    """
    response = e2e_rag.answer(query="Large pepperoni for delivery")

    assert len(response) >= 20, f"Response too short ({len(response)} chars): '{response}'"
    assert len(response) <= 500, f"Response too long ({len(response)} chars) — hallucination risk"

def test_response_contains_price_for_order_query(e2e_rag):
    """
    Any order confirmation must include a price.
    Missing price = customer can't confirm the order.
    """
    import re
    response = e2e_rag.answer(query="I want a large margherita pizza delivered")

    price_pattern = r"\$\d+\.\d{2}"
    assert re.search(price_pattern, response), (
        f"No price found in order response: '{response}'"
    )
```

> 💡 **Shape tests are the floor.** They tell you the system is alive and producing structurally valid output — non-empty string, valid `$X.XX` price format, length within expected bounds. A response that is `None`, empty, or missing a price in an order confirmation is a hard failure, not a statistical accuracy question.
> **Business consequence:** A missing price means the customer cannot confirm their order — a direct, binary conversion loss. At 10,000 orders/day, even a 0.5% shape failure rate is 50 lost orders per day.

### 5.2 — Invariance Tests

```python
def test_case_invariance(e2e_rag):
    """
    Input case must not affect order outcome.
    LARGE PEPPERONI and large pepperoni should return same price.
    """
    response_lower = e2e_rag.answer(
        query="large pepperoni pizza",
        metadata_filter={"menu_type": "dinner"}
    )
    response_upper = e2e_rag.answer(
        query="LARGE PEPPERONI PIZZA",
        metadata_filter={"menu_type": "dinner"}
    )

    # Both must mention the same price
    price_lower = extract_price(response_lower)
    price_upper = extract_price(response_upper)

    assert price_lower == price_upper, (
        f"Case changed the price: lower='{price_lower}', upper='{price_upper}'"
    )

def test_name_invariance_does_not_change_price(e2e_rag):
    """
    Customer name in query must not affect price.
    "Hi, I'm Maria and I want a large pepperoni" must return the same price
    as "I want a large pepperoni" — name is irrelevant.
    """
    price_without_name = extract_price(
        e2e_rag.answer("I want a large pepperoni pizza")
    )
    price_with_name = extract_price(
        e2e_rag.answer("Hi, I'm Maria and I want a large pepperoni pizza")
    )

    assert price_without_name == price_with_name, (
        f"Customer name changed price: without='{price_without_name}', "
        f"with='{price_with_name}'"
    )

def test_polite_phrasing_invariance(e2e_rag):
    """
    Politeness level must not change the order.
    "Give me a pizza" and "Could I please have a pizza?" should return same item.
    """
    item_rude = extract_item(e2e_rag.answer("Give me a large pepperoni"))
    item_polite = extract_item(e2e_rag.answer("Could I please have a large pepperoni pizza?"))

    assert item_rude == item_polite, (
        f"Phrasing changed item: rude='{item_rude}', polite='{item_polite}'"
    )
```

> 💡 **What invariance tests catch:** Every failing invariance test is a conversion leak. If `"LARGE PEPPERONI"` returns a different price than `"large pepperoni"`, the system silently fails a real fraction of users. These bugs are invisible in aggregate RAGAS scores — a context precision of 0.88 won't surface case-sensitive pricing — but they show up immediately as customer chargebacks when someone typed their order in capitals.
> **Rule:** Write at least one invariance test for each axis of natural user variation: letter case, punctuation, politeness level, name inclusion.

### 5.3 — Directional Expectation Tests

```python
def test_more_items_means_higher_price(e2e_rag):
    """
    Adding items must increase total price — monotonic.
    If 2 pizzas < 1 pizza, pricing logic is broken.
    """
    price_one = extract_price(
        e2e_rag.answer("I want 1 large pepperoni pizza")
    )
    price_two = extract_price(
        e2e_rag.answer("I want 2 large pepperoni pizzas")
    )

    assert price_two > price_one, (
        f"2 pizzas (${price_two}) is not more expensive than 1 pizza (${price_one})"
    )

def test_larger_size_means_higher_price(e2e_rag):
    """Size must be monotonically priced: small < medium < large."""
    price_small = extract_price(e2e_rag.answer("1 small pepperoni pizza"))
    price_medium = extract_price(e2e_rag.answer("1 medium pepperoni pizza"))
    price_large = extract_price(e2e_rag.answer("1 large pepperoni pizza"))

    assert price_small < price_medium < price_large, (
        f"Size pricing not monotonic: small=${price_small}, "
        f"medium=${price_medium}, large=${price_large}"
    )

def test_delivery_adds_to_pickup_price(e2e_rag):
    """Delivery must cost more than pickup — always."""
    price_pickup = extract_price(
        e2e_rag.answer("1 large pepperoni pizza for pickup")
    )
    price_delivery = extract_price(
        e2e_rag.answer("1 large pepperoni pizza for delivery")
    )

    assert price_delivery > price_pickup, (
        f"Delivery (${price_delivery}) is not more than pickup (${price_pickup})"
    )
```

> 💡 **Insight:** Directional tests catch pricing bugs that shape tests miss. If a "buy 2 get 1 free" promo is incorrectly applied to every order, your shape test (price is a valid dollar amount) passes but your directional test (2 > 1) fails.
> **Rule:** Write at least one directional test for every numerical output your system produces.

---

## 6 · Property-Based Testing with hypothesis

§5's invariance and directional tests covered three behavioral properties by hand — case, name, politeness. The §0 wrong-order breakdown (19 wrong-item, 5 wrong-price, 4 wrong-zone) suggests at least a dozen more failure axes in the menu and pricing logic. Property-based testing inverts the approach: write *properties* and let hypothesis generate hundreds of adversarial inputs automatically to falsify them.

Unit tests check specific inputs. Property-based tests ask: *for all inputs satisfying property P, does the output satisfy property Q?* The hypothesis library generates hundreds of adversarial inputs automatically.

```python
# tests/test_properties.py
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from src.order_parser import parse_order, OrderParseError

# Strategy: valid pizza sizes
pizza_sizes = st.sampled_from(["small", "medium", "large", "xl"])

# Strategy: valid topping names from our menu
topping_names = st.sampled_from([
    "pepperoni", "margherita", "supreme", "hawaiian",
    "bbq chicken", "veggie", "mushroom", "olive"
])

# Strategy: valid quantities (1–10)
quantities = st.integers(min_value=1, max_value=10)

@given(
    size=pizza_sizes,
    topping=topping_names,
    quantity=quantities,
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_valid_order_always_parses_without_error(size, topping, quantity):
    """
    For any valid (size, topping, quantity) combination,
    the order parser must return a structured order — never raise.

    This catches: IndexErrors, KeyErrors, AttributeErrors hiding in the parser.
    """
    query = f"I want {quantity} {size} {topping} pizza"

    # Must not raise
    order = parse_order(query)

    assert order is not None
    assert order["quantity"] == quantity
    assert order["size"] == size

@given(
    size=pizza_sizes,
    topping=topping_names,
    quantity=quantities,
)
@settings(max_examples=200)
def test_order_total_is_always_positive(size, topping, quantity):
    """
    For any valid order, total price must be > 0.
    Zero or negative price means pricing logic is broken.
    """
    from src.pricing import calculate_total

    total = calculate_total(size=size, topping=topping, quantity=quantity)

    assert total > 0, (
        f"Negative/zero price for {quantity}x {size} {topping}: ${total:.2f}"
    )

@given(
    size=pizza_sizes,
    topping=topping_names,
    quantity_1=quantities,
    quantity_2=quantities,
)
@settings(max_examples=100)
def test_order_total_monotone_in_quantity(size, topping, quantity_1, quantity_2):
    """
    More pizzas must always cost more (or equal for quantity=0 edge cases).
    This is the property-based version of test_more_items_means_higher_price.
    hypothesis will try to find a counter-example — if it doesn't, we're confident.
    """
    from src.pricing import calculate_total

    if quantity_1 == quantity_2:
        return  # vacuously true

    total_1 = calculate_total(size=size, topping=topping, quantity=quantity_1)
    total_2 = calculate_total(size=size, topping=topping, quantity=quantity_2)

    if quantity_1 < quantity_2:
        assert total_1 < total_2, (
            f"Total not monotone: {quantity_1}x {size} {topping} = ${total_1:.2f}, "
            f"but {quantity_2}x = ${total_2:.2f}"
        )
```

> 📖 **Optional:** hypothesis uses *shrinking* to find the minimal failing example. If `test_order_total_is_always_positive` fails on a generated input like `quantity=7, size="xl", topping="bbq chicken"`, hypothesis automatically shrinks the counterexample to the smallest values that still fail — often revealing the real bug is `quantity=1, size="small"` with a specific topping. This is why property-based tests catch bugs unit tests miss.
> See the [hypothesis docs on shrinking](https://hypothesis.readthedocs.io/en/latest/details.html#shrinking) for rigorous treatment.

> 💡 **Industry Standard: OpenAI Evals Framework**
> ```python
> # For production AI systems, extend property-based testing with OpenAI Evals
> from evals import run_eval
>
> # Define evaluation template (YAML or Python)
> eval_spec = {
>     "eval_name": "pizzabot.order-accuracy",
>     "samples": [
>         {"input": "large pepperoni for delivery", "expected": "price >= 14.99"},
>         {"input": "2 medium margherita pickup", "expected": "price < 30.00"},
>         # ... 100+ samples
>     ],
> }
>
> # Run evaluation against your model
> results = run_eval(
>     model="gpt-4o-mini",
>     eval=eval_spec,
>     output_dir="./eval_results",
> )
> ```
> **When to use:** Regression testing for fine-tuned models. Track accuracy across model versions. Compare base model vs. fine-tuned model on your domain-specific tasks.
> **Common alternatives:** `promptfoo` (CLI-based eval), `langfuse` (tracing + eval), `braintrust` (eval platform)
> **See also:** [OpenAI Evals GitHub](https://github.com/openai/evals)

> 💡 **Industry Standard: Giskard for Automated Test Generation**
> ```python
> import giskard as gsk
>
> # Wrap your RAG model
> wrapped_model = gsk.Model(
>     model=your_rag_pipeline,
>     model_type="text_generation",
>     feature_names=["user_query"],
> )
>
> # Wrap test dataset
> test_dataset = gsk.Dataset(
>     df=test_queries_df,
>     target="expected_response",
> )
>
> # Automatically generate 50+ adversarial tests
> test_suite = gsk.scan(wrapped_model, test_dataset)
> # Generates: prompt injection tests, bias tests, robustness tests,
> #            performance tests, data leakage tests
>
> # Run generated tests
> test_results = test_suite.run()
> test_results.to_html("giskard_report.html")
> ```
> **When to use:** Don't have time to write 100+ adversarial tests manually. Giskard automatically generates test cases for: prompt injections, fairness/bias, hallucination detection, robustness to typos, PII leakage.
> **Common alternatives:** `rebuff` (prompt injection only), `nemo-guardrails` (safety rails), `guardrails-ai` (structured output validation)
> **See also:** [Giskard documentation](https://docs.giskard.ai/)

---

> 💡 **Adversarial test verdict:** 200+ property-based tests confirmed zero crashes, no negative prices, no non-monotonic pricing — all three behavioral invariants (case, name, politeness) hold.
> ➡️ If hypothesis finds a counterexample, the shrunk input reveals the exact edge case; integrate the full suite into CI/CD to block PRs before bugs reach production.

---

## 7 · Putting It Together — The Complete CI/CD Test Pipeline

§3–6 built the test suite that would have caught the §0 wrong-order bug before the soft launch. But the bug reached production precisely because no automated gate blocked the bad retrieval change. This section assembles all three tiers into a GitHub Actions workflow where a failing test blocks a merge — preventing a rerun of the §0 incident.

```bash
# Install test dependencies
pip install pytest pytest-cov hypothesis

# Run unit tests with coverage
pytest tests/unit/ -v --cov=src --cov-report=term-missing

# Run integration tests (costs API money — run sparingly)
pytest tests/integration/ -v -m "integration"

# Run property-based tests
pytest tests/test_properties.py -v

# Run everything
pytest tests/ -v --cov=src --cov-report=html
# Open htmlcov/index.html to see line-by-line coverage
```

**Coverage output (target: ≥80% for src/rag_pipeline.py):**

```
Name                         Stmts   Miss  Cover
-------------------------------------------------
src/rag_pipeline.py             87      9    90%
src/pricing.py                  34      2    94%
src/order_parser.py             52      8    85%
-------------------------------------------------
TOTAL                          173     19    89%
```

>  ⚡ **Constraint:** 89% coverage doesn't mean 89% of bugs are caught. Coverage measures which lines *were run*, not which behaviours were *asserted*. A line can be covered by a test that asserts nothing. Coverage is a floor, not a ceiling.

---

### GitHub Actions Workflow

Every pull request runs the test suite automatically. A failing test blocks the merge.

```yaml
# .github/workflows/test-ai.yml
name: AI Test Suite

on:
  pull_request:
    branches: [main, staging]
  push:
    branches: [main]

jobs:
  unit-tests:
    name: Unit Tests (no API cost)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements-test.txt

      - name: Run unit tests with coverage
        run: |
          pytest tests/unit/ \
            --cov=src \
            --cov-fail-under=80 \
            --cov-report=xml \
            -v
        env:
          # No real API keys needed for unit tests (mocked)
          PYTHONPATH: ${{ github.workspace }}

      - name: Upload coverage report
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml

  model-tests:
    name: Model Behavioural Tests
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements-test.txt

      - name: Spin up test vector DB
        run: docker compose -f docker-compose.test.yml up -d --wait
        # Starts a fresh Chroma instance on port 8001

      - name: Seed test corpus
        run: python scripts/seed_test_corpus.py
        env:
          TEST_VECTOR_DB_URL: http://localhost:8001

      - name: Run model property tests
        run: |
          pytest tests/test_model_properties.py tests/test_properties.py \
            -v \
            --hypothesis-seed=42
        env:
          TEST_VECTOR_DB_URL: http://localhost:8001
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Tear down test DB
        if: always()
        run: docker compose -f docker-compose.test.yml down

  integration-tests:
    name: Integration Tests (API cost ~$0.05 per run)
    runs-on: ubuntu-latest
    needs: model-tests
    # Only run on push to main (not every PR — saves cost)
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements-test.txt

      - name: Spin up test vector DB
        run: docker compose -f docker-compose.test.yml up -d --wait

      - name: Seed test corpus
        run: python scripts/seed_test_corpus.py
        env:
          TEST_VECTOR_DB_URL: http://localhost:8001

      - name: Run integration tests
        run: |
          pytest tests/integration/ \
            -v \
            -m integration \
            --timeout=30
        env:
          TEST_VECTOR_DB_URL: http://localhost:8001
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Tear down test DB
        if: always()
        run: docker compose -f docker-compose.test.yml down
```

**What each job catches:**

| Job | Trigger | Cost | Catches |
|-----|---------|------|---------|
| `unit-tests` | Every PR | $0 | Component contracts, mocked LLM |
| `model-tests` | Every PR (after unit) | ~$0.01 | Shape, invariance, directional, property-based |
| `integration-tests` | Push to main only | ~$0.05 | Full pipeline E2E, latency, retrieval accuracy |

> ⚠️ **Warning:** Store `OPENAI_API_KEY` in GitHub Secrets (`Settings → Secrets → Actions`), never in the workflow YAML. A leaked API key in a public repo can generate thousands of dollars in charges within hours.
> **Rule:** Every secret in CI goes in `${{ secrets.SECRET_NAME }}`. Zero exceptions.

> ➡️ **Forward pointer:** The `docker compose -f docker-compose.test.yml up -d --wait` pattern provisions a disposable Chroma instance per CI run. [DevOps Ch.4 — CI/CD Pipelines](../../07-devops_fundamentals/) covers the full pattern: healthchecks, test isolation, and teardown-on-failure to avoid orphaned containers accumulating cost.

> 💡 **CI/CD gate ROI:** Unit tests run on every PR at $0 and under 1 second. Integration tests cost ~$0.05 per run on push to main only. Adversarial tests run nightly. At 100 PRs/week that is $5/week in test costs — against the §0 baseline of $700/week in wrong-order refunds, or $35,000/day at launch scale. The gate pays for itself in the first wrong order it intercepts.

---

## 8 · What Can Go Wrong: Testing Traps

The §0 wrong-order bug reached production past a team that thought they had working components. The traps below describe the patterns that produce passing tests while production burns — starting with the one most likely responsible for the §0 incident: tests that verify mocks, not the system.

### Trap #1: Testing the Mock, Not the System

```python
# ❌ Bad: This test only confirms the mock works
def test_retrieval(mock_rag_client):
    mock_rag_client.retrieve.return_value = [MagicMock(metadata={"menu_type": "dinner"})]
    results = mock_rag_client.retrieve(query="pepperoni")
    assert results[0].metadata["menu_type"] == "dinner"
    # This passes even if the real RAG client is broken
```

**Fix:** Mock at the *boundary* (external services: LLM API, DB), not at the function under test.

```python
# ✅ Good: Tests real retrieval logic, mocks only the external vector DB API
def test_retrieval(real_rag_client_with_test_corpus):
    results = real_rag_client_with_test_corpus.retrieve(
        query="pepperoni", metadata_filter={"menu_type": "dinner"}
    )
    assert results[0].metadata["menu_type"] == "dinner"
```

### Trap #2: Flaky Tests from Non-Deterministic LLMs

```python
# ❌ Bad: temperature=0.7 → different response each run
def test_mentions_price():
    response = llm.complete("How much is pepperoni?", temperature=0.7)
    assert "$14.99" in response  # Fails randomly (50% of the time)
```

**Fix:** `temperature=0` for all tests. If you need `temperature>0`, test for a *set* of acceptable outputs.

```python
# ✅ Good
def test_mentions_price():
    response = llm.complete("How much is pepperoni?", temperature=0)
    assert "$14.99" in response  # Deterministic at temperature=0
```

### Trap #3: Slow Integration Tests Block Every PR

If integration tests cost $0.05 and take 3 minutes, running them on every PR creates two problems: cost ($0.05 × 100 PRs/week = $5/week) and slow developer feedback (3 min wait on every commit).

**Fix:** Split the test suite with pytest marks.

```python
# tests/integration/test_rag_pipeline_e2e.py
import pytest

@pytest.mark.integration  # Only runs when -m integration is passed
def test_e2e_dinner_order_returns_correct_price(e2e_rag):
    ...
```

```bash
# In CI: integration tests only on main
pytest tests/ -m "not integration"  # unit + model tests (fast, free)
pytest tests/ -m integration         # e2e tests (slow, costs money — main only)
```

### Trap #4: Missing Test for the Exact Production Failure

The most common testing mistake: you run tests after a production incident, they all pass, and you conclude the system is fine. They passed because they don't test the scenario that failed.

**Rule:** Every production bug gets a regression test named after the bug.

```python
def test_retrieval_time_of_day_14pct_wrong_order_bug():
    """
    Regression test for production incident 2024-11-01.
    Wrong-order rate: 14%. Root cause: dinner queries returned lunch docs.
    This test was added after the incident. It was RED before the fix.
    """
    # ... the test from §3.3
```

> 💡 **Testing-trap cost:** Each trap in this section describes a variant of the §0 outcome: passing tests, a live production bug, a 14% wrong-order rate. All four fixes require one-line changes in test architecture — the avoidance cost is near zero. The cost of not avoiding them is $700/week in refunds at soft-launch scale, $35,000/day at full launch.

> ➡️ **Next:** §9 confirms all six production constraints are met by evidence, not just measurement — and identifies the two remaining gaps (p95 load testing, drift detection) that testing alone cannot close.

---

## 9 · Progress Check — PizzaBot Wrong-Order Rate Diagnosed and Quantified

✅ **Unlocked capabilities:**
- **Root cause identified**: retrieval returns lunch menu for dinner queries (metadata filter missing) — `wrong_order_rate: 14% → 0%` ✅
- **Unit test suite**: ingestion, retrieval, generation — 15 unit tests, 89% coverage
- **Model tests**: shape (3 tests), invariance (3 tests), directional (3 tests) — all passing
- **Property-based tests**: 200 hypothesis-generated inputs, no counterexample found for pricing monotonicity
- **CI/CD gate**: GitHub Actions workflow blocks PRs with failing tests; integration tests run on push to main

❌ **Still can't solve:**
- **Load testing** — `test_e2e_latency_under_3s` tests a single call, not p95 under production traffic. 1,400 concurrent orders/day peak is untested.
- **Drift detection** — if the menu is updated and the test corpus is not, tests will pass while production breaks. Continuous monitoring (not covered here) is required.

**Constraint status:**

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 BUSINESS VALUE | ✅ | >28% conversion (Ch.11), lab-confirmed |
| #2 ACCURACY | ✅ | wrong_order_rate: 14% → **0%** (retrieval bug fixed + 15 regression tests) |
| #3 LATENCY | ✅ | <3s single-call (integration test); p95 under load TBD |
| #4 COST | ✅ | <$0.08/conv (Ch.9 + Ch.10) |
| #5 SAFETY | ✅ | Zero attacks (Ch.7) |
| #6 RELIABILITY | ⚡ | >99% uptime manual; CI gate now catches regressions before deploy |

```mermaid
flowchart LR
    C1["Ch.1\n LLM Fundamentals"]
    C2["Ch.2\n Prompt Eng"]
    C3["Ch.3\n CoT Reasoning"]
    C4["Ch.4-5\n RAG + VectorDB\n ✅ ACCURACY"]
    C6["Ch.6\n ReAct\n ✅ CONVERSION"]
    C7["Ch.7\n Safety\n ✅ SAFETY"]
    C9["Ch.9-10\n Cost + Fine-tune\n ✅ COST LATENCY"]
    C11["Ch.11\n Agentic Patterns\n ✅ 99.2% accuracy"]
    C12["Ch.12 ← YOU ARE HERE\n Testing\n ✅ 0% wrong-order rate\n CI gate live"]
    PROD["Production\n Deployment"]

    C1 --> C2 --> C3 --> C4 --> C6 --> C7 --> C9 --> C11 --> C12 --> PROD

    style C12 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C4 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C6 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C7 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C9 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C11 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style PROD fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C1 fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C2 fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C3 fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

---

## 10 · Interview Checklist

### Must Know

| Topic | What the interviewer expects |
|-------|------------------------------|
| **Unit vs. Integration vs. E2E** | Unit = isolated component, mocked dependencies. Integration = two+ real components. E2E = full user journey, real infra. Know the cost/speed trade-off (unit: free/fast, E2E: expensive/slow). |
| **Non-determinism in LLM tests** | `temperature=0` for deterministic tests. Property-based tests handle probabilistic outputs. Never assert exact LLM output unless temperature=0 and provider guarantees determinism. |
| **pytest fixtures** | `scope="session"` for expensive setup (vector indexes). `scope="function"` for anything that mutates state. `conftest.py` for shared fixtures. |
| **Retrieval correctness** | Hit@K metric. A retrieval test must assert *which document* was returned, not just *that something* was returned. |
| **Regression tests** | Every production bug gets a named regression test. Red before fix, green after. |

### Likely Asked

| Question | Strong answer |
|----------|---------------|
| "How do you test a RAG pipeline?" | Three layers: (1) ingestion tests (metadata preserved, all docs indexed), (2) retrieval tests (hit@K, metadata filters, empty-query handling), (3) generation tests (price in response matches retrieved doc). Never mock the retrieval layer in retrieval tests. |
| "What's property-based testing?" | Generate hundreds of inputs from a strategy, verify the output satisfies a property for all of them. hypothesis library. Catches edge cases unit tests miss. Best for: pricing calculations, order parsers, any function with a numerical output and a monotonicity property. |
| "How do you make LLM tests not flaky?" | (1) `temperature=0`, (2) test properties (price is positive, response is non-empty) not exact strings, (3) mock the LLM in unit tests, real calls in integration tests only, (4) `hypothesis` for property-based adversarial tests. |
| "How do you test for hallucination?" | Compare prices in the generated response to prices in the retrieved documents. If the price in the response doesn't appear in any retrieved doc, it's hallucinated. This is a targeted test, not a RAGAS-style aggregate metric. |

### Trap Questions

| Trap | Why it's a trap | Correct answer |
|------|----------------|----------------|
| "100% test coverage means the system is bug-free." | Coverage measures lines run, not behaviours asserted. A line can be covered with no assertion. | "Coverage ≥ 80% is a floor. Directional tests, invariance tests, and property-based tests catch bugs coverage misses entirely." |
| "Just use RAGAS score — if it's above 0.85 you're good." | RAGAS is an aggregate metric. It doesn't tell you *which* query-document pair is wrong. The 14% wrong-order bug had a RAGAS context precision of 0.88 — above threshold. | "RAGAS measures aggregate retrieval quality. For production readiness, you also need targeted tests that reproduce specific failure modes like the time-of-day retrieval bug." |
| "Integration tests should mock the LLM to be deterministic." | Mocking the LLM in integration tests defeats the purpose — you're testing the mock's behaviour, not the LLM's. | "Mock at the external boundary in unit tests. In integration tests, use the real LLM at `temperature=0`. Reserve mocks for flaky external dependencies (slow APIs, rate-limited services)." |

---

## 11 · Bridge to Next Chapter

Ch.12 proved the system is correct — tests say so, CI enforces it. **But tests are run locally or in CI. Production runs at 10,000 orders/day on infrastructure you control.** The next step is production deployment: containerising PizzaBot, managing environment variables securely, setting up load balancing for concurrent traffic, and configuring monitoring so you get alerted *before* the wrong-order rate climbs again. That's the domain of [DevOps Ch.4 — CI/CD Pipelines](../../07-devops_fundamentals/), which takes your tested, CI-gated code and gets it to users at scale.
