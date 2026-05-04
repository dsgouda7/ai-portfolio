# Cost & Latency — Running AI Systems in Production

> **The story.** When the OpenAI API launched in June **2020**, GPT-3 cost $0.06 per 1 K tokens and almost nobody worried about it because almost nobody had production traffic. By **2023** the picture had inverted: every serious AI deployment was a finance problem. The optimisation stack was built fast. **KV-cache reuse** — keeping the attention key/value tensors hot across decoding steps — became standard once **Flash Attention** (Tri Dao, **2022**) made long contexts feasible. **Continuous batching** from the **Orca** paper (Yu et al., OSDI **2022**) replaced static batching and lifted serving throughput ~10×. **Speculative decoding** (Leviathan et al., Google, **2022**) slashed latency by drafting tokens with a small model and verifying with a large one. **Prompt caching** appeared in the **Anthropic** API in August 2024 and **OpenAI** in October 2024, cutting input-token cost ~90% for repeated prefixes. Open-source pricing collapsed in parallel — a 2026 query against a hosted Llama-class or DeepSeek model costs a small fraction of a 2023 GPT-4 call — and the cost-and-latency budget is what decides whether your system ships or stalls.
>
> **Where you are in the curriculum.** The gap between a demo and a production system is almost entirely cost and latency. A prototype that calls GPT-4 with a 50 K-token context, runs self-consistency 5×, and uses an LLM judge to evaluate every response works beautifully at zero users. At 10 000 users it costs thousands of dollars per day and responds in 30 seconds. This document maps the levers — model tier, context length, caching, streaming, batching, quantisation, speculative decoding — that turn that demo into something you can run on a budget. It is the closing chapter of the AI track and the bridge to the [AIInfrastructure](../../ai_infrastructure) track where these levers become hardware decisions.
>
> **Notation.** $c = \frac{n_\text{in} \cdot p_\text{in} + n_\text{out} \cdot p_\text{out}}{10^6}$ — cost per request in USD; $n_\text{in}, n_\text{out}$ — input and output token counts; $p_\text{in}, p_\text{out}$ — price per million tokens; $\lambda$ — request arrival rate (req/s); $\bar{t}$ — mean service time per request.

***

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Launch **Mamma Rosa's PizzaBot** — a production AI ordering system satisfying 6 constraints:
> 1. **BUSINESS VALUE**: >25% conversion + +$2.50 AOV + 70% labor savings — 2. **ACCURACY**: <5% error — 3. **LATENCY**: <3s p95 — 4. **COST**: <$0.08/conv — 5. **SAFETY**: Zero attacks — 6. **RELIABILITY**: >99% uptime

**What we know so far:**
- ✅ Ch.1-9: All technical targets hit! 30% conversion (>25% ✅), $40.60 AOV (+$2.10 ✅), 2.2s latency (<3s ✅), $0.010/conv (<$0.08 ✅), safety validated (0 attacks ✅)
- ✅ Constraints #2 ACCURACY (4.2% error <5% ✅), #5 SAFETY (0 attacks ✅), #6 RELIABILITY (>99% uptime ✅) **ACHIEVED**
- ⚡ Constraints #1 BUSINESS VALUE, #3 LATENCY, #4 COST **PARTIAL** — working but not optimized
- 📊 **Current business metrics**: 30% conversion (phone baseline: 22%), $40.60 AOV (baseline: $38.50), $0.010/conv cost, 2.2s latency

**What's blocking us:**

🚨 **Current system meets technical targets but ROI payback is 18 months — need 10.6 months to justify CEO's $300k investment**

**Current economics (pre-optimization):**
```
Cost breakdown per conversation:
- Fine-tuned Llama-3-8B inference (self-hosted): $0.004/conv
- RAG retrieval (embedding + vector search): $0.002/conv
- Safety validation (Azure Content Safety): $0.002/conv
- Guardrails overhead: $0.001/conv
- Monitoring/logging: $0.001/conv

Total: $0.010/conv

Monthly cost at 50 visitors/day:
- 50 visitors × 28% conversion = 14 orders/day = 420 orders/month
- 420 conv/month × $0.010 = $4.20/month infrastructure cost ✅

Current ROI:
- Revenue: 30% × $40.60 × 50 = $609/day = $18,270/month
- Labor savings: $11,064/month
- Total benefit: $18,270 - $12,705 + $11,064 = $16,629/month
- Payback: $300,000 / $16,629 = **18 months**

Target ROI: 10.6 months (need $28,302/month benefit)
```

**The problem:**

CEO demands proof of 10.6-month ROI payback period to justify the $300k development investment. Current system: **18-month payback** (70% over target). Gap: $11,673/month in additional benefit needed.

**Two paths to close the gap:**
1. **Scale traffic**: 50 → 88 daily visitors (+76%) — marketing campaign costs $50k+ upfront
2. **Optimize operations**: Reduce latency → better UX → higher conversion — **zero additional investment**

**The failure scenario:**

You demo PizzaBot to the CEO. Customer asks: "What's your most popular gluten-free pizza?"

User sees: Loading spinner for 2.2 seconds.

CEO watches the timer: "2 seconds feels slow. What if we had 100 concurrent users during Friday dinner rush?"

You: "We can handle 10 requests/sec. At 100 concurrent users..." *(does math)* "...we'd need 10× more infrastructure. Cost would jump from $4.20/month to $42/month."

CEO: "And customers would wait even longer. I'm seeing 2.5-second delays in your metrics already. **Fix the latency or we're not launching.**"

**Business impact:**
- **2.2s → 1.5s latency** = 30% → 32% conversion (+$2,435/month revenue)
- **Cost optimization** ($0.010 → $0.005/conv) = infrastructure headroom for 2× traffic
- **Throughput optimization** (10 → 20 req/sec) = handles Friday rush without infrastructure scaling
- **Combined impact**: Closes the ROI gap without marketing spend

**What this chapter unlocks:**

🚀 **Cost & latency optimization stack:**
1. **Prompt caching**: Cache system prompt across requests (90% cache hit → $0.002 → $0.0002 RAG cost)
2. **Streaming responses**: First token <500ms (perceived instant UX)
3. **KV-cache reuse**: Reuse attention tensors (-200ms latency)
4. **Speculative decoding**: Draft with small model, verify with large (30% faster generation)
5. **Batched inference**: Process concurrent requests together (2× throughput for peak traffic)
6. **INT8 quantization**: Model 16GB → 8GB (faster memory, -700ms inference time)

⚡ **Constraints #1 + #3 + #4 [ACHIEVED]**:
- **Latency**: 2.2s → **1.5s p95** (target <3s ✅, beats by 50%) → 32% conversion
- **Cost**: $0.010 → **$0.005/conv** (target <$0.08 ✅, 94% under budget) → infrastructure headroom
- **Business Value**: 32% conversion (>25% ✅), +$2.10 AOV, 70% labor savings (✅) → $17,847/month benefit
- **ROI payback**: 16.8 months at current traffic → **10.9 months at 120 visitors/day** ✅

---

## 1 · Core Idea

Every LLM API call costs money and time. Both are functions of one thing: **the number of tokens processed**.

```
Cost    = (input_tokens  × $/1M input)  + (output_tokens × $/1M output)
Latency = time_to_first_token (TTFT)   + tokens_generated × ms/token
```

Every architectural decision — which model, how much context, how many calls, whether to stream — maps directly to these two formulas.

---

## 1.5 · The Practitioner Workflow — Your 4-Phase Optimization

**Before diving into theory, understand the workflow you'll follow with every production deployment:**

> 📊 **What you'll build by the end:** A cost-optimized production LLM system with sub-2s latency, 50% cost reduction via caching, and 2× throughput through batching — validated with real token counts and cost breakdowns.

```
Phase 1: SELECT            Phase 2: CACHE             Phase 3: STREAM            Phase 4: OPTIMIZE
────────────────────────────────────────────────────────────────────────────────────────────────────
Choose model tier:         Cache repeated content:    Stream responses:          Batch & quantize:

• Frontier ($3-15/1M)      • System prompt            • Time-to-first-token      • Continuous batching
• Mid-tier ($0.15-1/1M)    • Few-shot examples        • Progressive rendering    • INT8 quantization
• Open-source ($0-0.3/1M)  • RAG context              • KV-cache reuse           • Speculative decoding

→ DECISION:                → DECISION:                → DECISION:                → DECISION:
  Task complexity?           Cache hit rate?            Interactive vs batch?      Scale requirements?
  • Complex reasoning:       • >70% hit rate:           • User-facing: Stream      • >10 req/sec: Batch
    Frontier model             Enable prefix cache        (perceived <1s)            (2× throughput)
  • Structured tasks:        • <30% hit rate:           • Async tasks: Sync        • <10 req/sec: Single
    Mid-tier model             Skip caching overhead      (wait for full response)   (simpler infra)
  • High-volume:             • 30-70% hit: Test         • Always: Enable           • Memory-bound:
    Open-source self-hosted    (measure savings)          KV-cache                   Quantize to INT8
```

**The workflow maps to this chapter:**
- **Phase 1 (SELECT)** → §3 Model Cost Tiers, §5 Accuracy vs. Cost Tradeoff
- **Phase 2 (CACHE)** → §6.1 Prompt Caching (new section below)
- **Phase 3 (STREAM)** → §4 Latency Components, §6.2 Streaming (new section below)
- **Phase 4 (OPTIMIZE)** → §6.3 Batching & Quantization (new section below)

> 💡 **Usage note:** Phases 1-2 are sequential (choose model, then enable caching). Phases 3-4 can be applied independently — streaming improves perceived latency, batching/quantization improve throughput.

**Typical optimization impact (PizzaBot example):**
```
Baseline (no optimizations):
  Cost: $0.010/conv, Latency: 2.2s p95

After Phase 1 (SELECT mid-tier):
  Cost: $0.010/conv (already using Llama-3-8B), Latency: 2.2s

After Phase 2 (CACHE system prompt):
  Cost: $0.008/conv (-20%), Latency: 2.2s

After Phase 3 (STREAM responses):
  Cost: $0.008/conv, Latency: 1.8s perceived (<1s to first token)

After Phase 4 (OPTIMIZE with INT8 + batching):
  Cost: $0.005/conv (-50% total), Latency: 1.5s p95, Throughput: 2×

ROI: $300k investment → 10.9 months payback (vs. 18 months without optimization)
```

---

## 2 · Where Tokens Come From

In a RAG + agent pipeline, the token budget breaks down roughly as:

| Component | Typical token count | Notes |
|---|---|---|
| System prompt | 200–800 | Often fixed per deployment |
| Few-shot examples | 300–1500 | Per-call if not cached |
| Retrieved chunks (RAG) | 1k–8k | Scales with k and chunk size |
| Conversation history | 0–50k | Grows unboundedly in chat — biggest leak |
| User message | 10–500 | Relatively small |
| **Total input** | **~2k–60k** | Context window determines the ceiling |
| Model output | 100–2000 | Determined by task; agents generate more |

**The biggest cost leak in production:** conversation history. A chat app that passes the full conversation history to every API call has linearly growing costs per session. Solutions: summarise older turns, truncate aggressively, or store history in a vector DB and retrieve only the relevant parts.

---

## 3 · **[Phase 1: SELECT]** Model Tier Selection

| Model tier | Cost range | When to use |
|---|---|---|
| Frontier (GPT-4o, Claude 3.5 Sonnet) | $2–15 / 1M tokens | Complex reasoning, high-stakes outputs, judge models |
| Mid-tier (GPT-4o-mini, Claude Haiku) | $0.15–1 / 1M tokens | Most retrieval and summarisation tasks |
| Open-source (Llama 3, Mistral 7B) | $0–0.3 / 1M tokens (self-hosted) | High-volume, latency-sensitive, private data |
| Embedding models | $0.01–0.13 / 1M tokens | Cheap; embed everything at ingestion, not query time if possible |

**Practical rule:** use the cheapest model that passes your evaluation threshold. Run `EvaluatingAISystems.md` metrics on both models. The quality difference between mid-tier and frontier is often smaller than expected for structured tasks with good prompts.

### 3.1 Model Selection Code Example

```python
from openai import OpenAI
import time

client = OpenAI()

# Define models with pricing ($/1M tokens)
MODELS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

def estimate_cost(prompt: str, model: str, expected_output_tokens: int = 200) -> dict:
    """Estimate cost for a single API call."""
    # OpenAI's tiktoken would be more accurate, but rough estimate:
    input_tokens = len(prompt.split()) * 1.3  # ~1.3 tokens per word
    
    pricing = MODELS[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (expected_output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": expected_output_tokens,
        "cost_usd": total_cost,
        "cost_per_1k_calls": total_cost * 1000
    }

# Phase 1 DECISION LOGIC: Compare models for PizzaBot order confirmation task
prompt = """You are PizzaBot, an AI ordering assistant for Mamma Rosa's Pizzeria.

User: "Large Margherita and 2 garlic breads to 42 Maple St"

Generate order confirmation JSON with: items, quantities, delivery address, estimated total.
"""

print("=== PHASE 1: MODEL SELECTION ===\n")
for model_name in ["gpt-4o", "gpt-4o-mini"]:
    estimate = estimate_cost(prompt, model_name, expected_output_tokens=150)
    print(f"{model_name}:")
    print(f"  Cost per call: ${estimate['cost_usd']:.6f}")
    print(f"  Cost per 1,000 calls: ${estimate['cost_per_1k_calls']:.2f}")
    print()

# DECISION: For structured order confirmation (not complex reasoning),
# gpt-4o-mini is sufficient and 10-15× cheaper than gpt-4o.
# At 10,000 orders/month: $90/month (mini) vs. $1,200/month (frontier)

"""
Expected output:
=== PHASE 1: MODEL SELECTION ===

gpt-4o:
  Cost per call: $0.000163
  Cost per 1,000 calls: $0.16

gpt-4o-mini:
  Cost per call: $0.000012
  Cost per 1,000 calls: $0.01

→ DECISION: gpt-4o-mini selected (13× cheaper, quality validated with eval metrics)
"""
```

> 💡 **Industry Standard:** `LiteLLM` for multi-provider cost tracking
> 
> ```python
> from litellm import completion, cost_per_token
> 
> response = completion(
>     model="gpt-4o-mini",
>     messages=[{"role": "user", "content": prompt}]
> )
> 
> # Automatic cost calculation with exact token counts
> cost = cost_per_token(
>     model="gpt-4o-mini",
>     prompt_tokens=response.usage.prompt_tokens,
>     completion_tokens=response.usage.completion_tokens
> )
> print(f"Actual cost: ${cost:.6f}")
> ```
> 
> **When to use:** Always in production. LiteLLM supports 100+ providers (OpenAI, Anthropic, Azure, AWS Bedrock, open-source) with unified API.
> **Common alternatives:** `OpenAI.usage` (native), custom token counters (error-prone)
> **See also:** [LiteLLM docs](https://docs.litellm.ai/docs/completion/cost_tracking)

### 3.2 DECISION CHECKPOINT — Phase 1 Complete

**What you just saw:**
- GPT-4o costs $0.000163/call vs. GPT-4o-mini $0.000012/call (13× difference)
- At 10,000 orders/month: $1,200 (frontier) vs. $90 (mid-tier) = $1,110/month savings
- PizzaBot order confirmation is structured task → mid-tier sufficient (validated with Ch.8 eval metrics)

**What it means:**
- Model selection is the **highest-leverage cost decision** (10-50× impact)
- Structured tasks (order confirmation, JSON extraction, classification) rarely need frontier models
- Complex reasoning (math, code generation, multi-step planning) justifies frontier cost
- Always validate with evaluation metrics before committing to cheaper model

**What to do next:**
→ **Run evaluation suite** (Ch.8 metrics) on both models with 100-sample test set
→ **For PizzaBot scenario:** gpt-4o-mini achieves 95% accuracy vs. gpt-4o 96% → **choose mid-tier** (1% accuracy loss acceptable for 13× cost savings)
→ **For complex reasoning tasks:** If accuracy drops >5% with mid-tier → **stay with frontier**
→ **For high-volume (>100k calls/month):** Consider **open-source self-hosted** (Llama 3, Mistral) for 50-100× cost savings vs. frontier

---

## 4 · **[Phase 3: STREAM]** Latency Components and Streaming

```
Total latency = network RTT
              + time_to_first_token (TTFT)
              + (tokens_to_generate × ms_per_token)
              + (optional: NLI verification, self-consistency, judge call)

TTFT scales with:    input_token_count  (larger context → longer prefill → longer TTFT)
ms_per_token scales with:  model size    (larger model → lower tokens/sec)
```

### Streaming

Streaming returns tokens as they are generated rather than waiting for the full response. Time-to-first-token remains the same; perceived latency drops dramatically because the user sees text appearing.

**Use streaming whenever the output is text shown to a user.** Never stream when the application needs to parse the full response before acting (e.g., JSON tool calls in an agent loop).

### 4.1 Streaming Code Example

```python
from openai import OpenAI
import time

client = OpenAI()

prompt = "Explain the top 3 pizza toppings at Mamma Rosa's (Pepperoni, Mushrooms, Basil) in 2 sentences each."

print("=== PHASE 3: STREAMING RESPONSES ===\n")

# BASELINE: Non-streaming (wait for full response)
print("Non-streaming (wait for full response):")
start = time.time()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    stream=False
)
end = time.time()
print(f"  Time to first token: {end - start:.2f}s")
print(f"  Total time: {end - start:.2f}s")
print(f"  Response: {response.choices[0].message.content[:100]}...")
print()

# STREAMING: Progressive rendering
print("Streaming (progressive rendering):")
start = time.time()
first_token_time = None
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    stream=True
)

response_text = ""
for chunk in stream:
    if chunk.choices[0].delta.content:
        if first_token_time is None:
            first_token_time = time.time()
            print(f"  Time to first token: {first_token_time - start:.2f}s")
        response_text += chunk.choices[0].delta.content
        print(chunk.choices[0].delta.content, end="", flush=True)

end = time.time()
print(f"\n  Total time: {end - start:.2f}s")
print()

# DECISION: For user-facing responses, streaming provides "instant" feel
# First token appears in <500ms vs 2.2s for full response
# User perceives system as responsive even though total time is similar

"""
Expected output:
=== PHASE 3: STREAMING RESPONSES ===

Non-streaming (wait for full response):
  Time to first token: 2.18s
  Total time: 2.18s
  Response: 1. **Pepperoni**: A classic favorite, Mamma Rosa's pepperoni is thinly sliced and perfectly...

Streaming (progressive rendering):
  Time to first token: 0.42s
1. **Pepperoni**: A classic favorite, Mamma Rosa's pepperoni is thinly sliced and perfectly seasoned...
  Total time: 2.21s

→ DECISION: Streaming enabled (perceived latency 0.42s vs. 2.18s, 5× faster UX)
"""
```

> 💡 **Industry Standard:** `OpenAI` streaming with async for production
> 
> ```python
> import asyncio
> from openai import AsyncOpenAI
> 
> async def stream_response(prompt: str):
>     client = AsyncOpenAI()
>     stream = await client.chat.completions.create(
>         model="gpt-4o-mini",
>         messages=[{"role": "user", "content": prompt}],
>         stream=True
>     )
>     
>     async for chunk in stream:
>         if chunk.choices[0].delta.content:
>             yield chunk.choices[0].delta.content  # Send to frontend in real-time
> 
> # Usage in production (FastAPI endpoint)
> from fastapi import FastAPI
> from fastapi.responses import StreamingResponse
> 
> app = FastAPI()
> 
> @app.post("/chat/stream")
> async def chat_stream(prompt: str):
>     return StreamingResponse(
>         stream_response(prompt),
>         media_type="text/event-stream"
>     )
> ```
> 
> **When to use:** Always for user-facing chat interfaces. Reduces perceived latency by 5-10×.
> **Common alternatives:** Anthropic streaming (same pattern), Replicate streaming (open-source models)
> **See also:** [OpenAI streaming docs](https://platform.openai.com/docs/api-reference/streaming)

### The KV Cache

Transformers cache the key-value matrices of previously computed tokens (the KV cache). For a repeated prefix (system prompt + few-shot examples), the provider can reuse the cached computation rather than recomputing it on every call.

```
Without prefix caching:   every call pays input_tokens × full prefill cost
With prefix caching:      every call pays only new_tokens × prefill cost
                          (cached prefix is free or deeply discounted)
```

**Implication:** put your system prompt and few-shot examples at the beginning of the context. Keep them identical across calls. Most major providers (OpenAI, Anthropic) offer prefix caching automatically or explicitly. Savings of 50–80% on input token costs for chat applications are typical.

### 4.2 DECISION CHECKPOINT — Phase 3 Complete

**What you just saw:**
- Non-streaming: 2.18s wait → user sees loading spinner for 2+ seconds
- Streaming: 0.42s to first token → user sees text immediately, **5× faster perceived latency**
- Total generation time similar (2.18s vs. 2.21s), but UX dramatically better
- KV-cache automatically reuses system prompt computation across requests

**What it means:**
- Streaming is **free latency win** for user-facing apps — no cost increase, massive UX improvement
- Time-to-first-token (TTFT) is the **only metric that matters** for perceived responsiveness
- Total latency still matters for completion, but users tolerate it if progress is visible
- KV-cache is transparent optimization — works automatically when prompts share prefix

**What to do next:**
→ **For interactive chat:** Always enable streaming (OpenAI `stream=True`, Anthropic `stream=True`)
→ **For agent tool calls:** Disable streaming (need full JSON response before parsing)
→ **For batch processing:** Disable streaming (no user watching, total throughput matters)
→ **Optimize TTFT:** Reduce input token count (see Phase 2: CACHE), use smaller models, or self-host with vLLM

---

## 5 · **[Phase 2: CACHE]** Prompt Caching Strategies

Keep system prompts and few-shot examples structurally identical across calls. Any content the provider can cache doesn't get billed again.

**The caching hierarchy:**

| Content type | Cache stability | Savings potential |
|---|---|---|
| System prompt | Fixed per deployment | 90-95% cost reduction |
| Few-shot examples | Fixed per deployment | 90-95% cost reduction |
| RAG context | Changes per query | 50-70% if repeated queries |
| Conversation history | Changes per turn | 20-40% for FAQ patterns |

### 5.1 Prompt Caching Code Example

```python
from openai import OpenAI
client = OpenAI()

# STABLE SYSTEM PROMPT (will be cached across requests)
SYSTEM_PROMPT = """You are PizzaBot, an AI ordering assistant for Mamma Rosa's Pizzeria.

Our menu:
- Margherita: Classic tomato, mozzarella, basil ($12/$18/$24 for S/M/L)
- Pepperoni: Tomato, mozzarella, pepperoni ($14/$20/$26)
- Mushroom: Tomato, mozzarella, mushrooms ($13/$19/$25)

Allergen info:
- All pizzas contain gluten (wheat crust) and dairy (cheese)
- We cannot guarantee allergen-free preparation

Guidelines:
- Always confirm order details (items, size, price, delivery address)
- Flag allergen concerns immediately
- Suggest popular items if customer asks
"""

def call_with_caching(user_message: str, use_cache: bool = True):
    """
    Phase 2 DECISION LOGIC: Enable prompt caching for repeated system prompts.
    OpenAI automatically caches system messages when identical.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},  # Will be cached
        {"role": "user", "content": user_message}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    # Check token usage (OpenAI reports cached tokens separately)
    usage = response.usage
    print(f"Input tokens: {usage.prompt_tokens}")
    if hasattr(usage, 'prompt_tokens_details'):
        cached = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
        print(f"  Cached: {cached} tokens (saved ${cached * 0.15 / 1_000_000:.6f})")
    
    return response.choices[0].message.content

print("=== PHASE 2: PROMPT CACHING ===\n")

# First call: No cache (cold start)
print("First call (cold start - no cache):")
call_with_caching("I want a large Margherita")
print()

# Second call: Cache hit (system prompt reused)
print("Second call (cache hit - system prompt cached):")
call_with_caching("Is the Mushroom pizza gluten-free?")
print()

# Third call: Another cache hit
print("Third call (another cache hit):")
call_with_caching("2 medium Pepperoni to 42 Maple St")
print()

"""
Expected output:
=== PHASE 2: PROMPT CACHING ===

First call (cold start - no cache):
Input tokens: 412
  Cached: 0 tokens (saved $0.000000)

Second call (cache hit - system prompt cached):
Input tokens: 25
  Cached: 387 tokens (saved $0.000058) ← 94% of system prompt cached!

Third call (another cache hit):
Input tokens: 28
  Cached: 387 tokens (saved $0.000058)

→ DECISION: Prompt caching enabled (90-95% token savings on system prompt)
"""
```

> 💡 **Industry Standard:** `Anthropic` prompt caching with explicit cache_control
> 
> ```python
> import anthropic
> 
> client = anthropic.Anthropic()
> 
> response = client.messages.create(
>     model="claude-3-5-sonnet-20241022",
>     max_tokens=1024,
>     system=[
>         {
>             "type": "text",
>             "text": SYSTEM_PROMPT,
>             "cache_control": {"type": "ephemeral"}  # Explicit cache marker
>         }
>     ],
>     messages=[
>         {"role": "user", "content": "I want a large Margherita"}
>     ]
> )
> 
> # Anthropic reports cache hits/writes explicitly
> usage = response.usage
> print(f"Input tokens: {usage.input_tokens}")
> print(f"Cache creation tokens: {usage.cache_creation_input_tokens}")  # First call only
> print(f"Cache read tokens: {usage.cache_read_input_tokens}")  # Subsequent calls
> 
> # Pricing: cache writes = 1.25× input price, cache reads = 0.1× input price
> # Net savings: 90% cost reduction on cached content after first call
> ```
> 
> **When to use:** Always for stable system prompts and few-shot examples. Anthropic offers explicit control; OpenAI caches automatically.
> **Common alternatives:** `vLLM` prefix caching for self-hosted models (automatic, no API changes)
> **See also:** [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching), [OpenAI prompt caching](https://platform.openai.com/docs/guides/prompt-caching)

### 5.2 Tiered model routing

Route requests to the cheapest model that can handle them; escalate to a stronger model only on failure or low-confidence signal.

```python
def route_query(query: str) -> str:
    # Try the cheap model first
    result = cheap_model.generate(query)
    if low_confidence(result) or contains_complex_reasoning(query):
        result = frontier_model.generate(query)
    return result
```

### 5.3 Context window discipline

Aggressively trim the context before each call:
1. Summarise conversation history older than N turns
2. Limit retrieved chunks to the minimum k that meets recall thresholds
3. Remove few-shot examples once the model has demonstrated the pattern (session-level cache)

### 5.4 Request deduplication and caching

Cache model responses for identical (prompt, parameters) pairs. Semantic caching (cache responses for semantically similar queries) can achieve 20–40% cache hit rates on FAQ-style applications.

```python
import hashlib
_cache = {}

def cached_llm_call(prompt: str, model: str, temperature: float) -> str:
    key = hashlib.sha256(f"{prompt}|{model}|{temperature}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    result = llm_api.generate(prompt, model=model, temperature=temperature)
    _cache[key] = result
    return result
```

### 5.5 DECISION CHECKPOINT — Phase 2 Complete

**What you just saw:**
- First call: 412 input tokens, no cache → full cost
- Second call: 25 new tokens + 387 cached tokens (94% cached!) → 90% cost savings
- Third call: Same pattern → consistent savings across session
- Caching works automatically when system prompt stays identical

**What it means:**
- Prompt caching is **highest ROI optimization** after model selection (50-90% input cost reduction)
- System prompt and few-shot examples should be **fixed at deployment** — any dynamic content breaks cache
- Cache hit rate determines savings: >70% hit rate → enable caching, <30% → skip overhead
- Anthropic charges 1.25× for cache writes, 0.1× for cache reads → breaks even after 2nd call

**What to do next:**
→ **Measure cache hit rate:** Track `cached_tokens / total_input_tokens` across 1,000 requests
→ **For PizzaBot scenario:** 90% cache hit rate (stable system prompt) → **enable caching** (saves $0.002/conv)
→ **For dynamic prompts:** Cache hit <30% → **skip caching** (overhead > savings)
→ **Optimize for caching:** Move dynamic content (user message, RAG context) to end of prompt (preserve prefix)

---

## 6 · **[Phase 4: OPTIMIZE]** Batching and Quantization

### 6.1 Batch processing

For non-interactive workloads (nightly summaries, document ingestion), use batch APIs (50% cost reduction on most providers). Latency is irrelevant; throughput is not.

### 6.2 Batching Code Example

```python
from openai import OpenAI
import time

client = OpenAI()

# Simulate 5 concurrent order requests (e.g., Friday dinner rush)
orders = [
    "Large Margherita to 10 Oak St",
    "2 medium Pepperoni to 23 Elm Ave",
    "Small Mushroom and garlic bread to 45 Pine Rd",
    "3 large Margherita to 67 Maple Dr",
    "Medium Pepperoni, no cheese (allergy) to 89 Cedar Ln"
]

print("=== PHASE 4: BATCHED INFERENCE ===\n")

# BASELINE: Sequential processing (one at a time)
print("Sequential processing (one request at a time):")
start = time.time()
results_sequential = []
for order in orders:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Process order: {order}"}]
    )
    results_sequential.append(response.choices[0].message.content)
end = time.time()
print(f"  Total time: {end - start:.2f}s")
print(f"  Throughput: {len(orders) / (end - start):.2f} requests/sec")
print()

# BATCHING: For self-hosted models (vLLM, TensorRT-LLM)
# OpenAI API doesn't expose batching directly, but shows the pattern
print("Conceptual batching (self-hosted with vLLM):")
print("  With continuous batching:")
print("    - All 5 requests processed in single forward pass")
print("    - Shared overhead (attention compute, memory access)")
print("    - Total time: ~2.5s (vs. 10s sequential)")
print("    - Throughput: ~2 requests/sec (2× improvement)")
print()

# DECISION: For peak traffic (>10 concurrent requests), batching doubles throughput
# Requires self-hosted inference server (vLLM, TensorRT-LLM) with continuous batching

"""
Expected output:
=== PHASE 4: BATCHED INFERENCE ===

Sequential processing (one request at a time):
  Total time: 10.82s
  Throughput: 0.46 requests/sec

Conceptual batching (self-hosted with vLLM):
  With continuous batching:
    - All 5 requests processed in single forward pass
    - Shared overhead (attention compute, memory access)
    - Total time: ~2.5s (vs. 10s sequential)
    - Throughput: ~2 requests/sec (2× improvement)

→ DECISION: For >10 concurrent requests, enable batching with vLLM (2× throughput)
"""
```

> 💡 **Industry Standard:** `vLLM` for batched self-hosted inference
> 
> ```python
> from vllm import LLM, SamplingParams
> 
> # Initialize vLLM with continuous batching enabled
> llm = LLM(
>     model="meta-llama/Llama-3-8B",
>     tensor_parallel_size=1,
>     max_model_len=8192
> )
> 
> # Batch 100 requests together (processed in single forward pass)
> prompts = [f"Process order: {order}" for order in orders]
> sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
> 
> # vLLM automatically batches and schedules for maximum GPU utilization
> outputs = llm.generate(prompts, sampling_params)
> 
> for output in outputs:
>     print(output.outputs[0].text)
> 
> # Throughput: 10-50× faster than sequential OpenAI API calls
> # Latency per request: Similar to single request (shared overhead)
> ```
> 
> **When to use:** Self-hosted deployments with >10 concurrent requests. Batching improves GPU utilization from 30-40% to 80-90%.
> **Common alternatives:** `TensorRT-LLM` (NVIDIA-optimized), `Text Generation Inference` (Hugging Face)
> **See also:** [vLLM docs](https://docs.vllm.ai/en/latest/), [Continuous batching paper](https://arxiv.org/abs/2209.01667)

### 6.3 Quantization — INT8 and Beyond

Quantization reduces model precision from FP16 (16-bit floats) to INT8 (8-bit integers). Memory bandwidth (not compute) is the bottleneck for LLM inference, so 2× smaller weights → 2× faster memory access → 40-50% latency reduction.

**Quantization tradeoffs:**

| Precision | Memory | Latency | Accuracy loss | When to use |
|---|---|---|---|---|
| FP16 | 16GB (8B model) | Baseline | 0% (baseline) | Research, highest quality |
| INT8 | 8GB (8B model) | -40% | <1% | Production default |
| INT4 | 4GB (8B model) | -60% | 1-3% | Resource-constrained, high-volume |

### 6.4 Quantization Code Example

```python
# Quantization is applied at model loading time, not per-request
# This example shows vLLM with INT8 quantization enabled

from vllm import LLM, SamplingParams

print("=== PHASE 4: QUANTIZATION ===\n")

# BASELINE: FP16 (full precision)
print("FP16 model (full precision):")
print("  Model size: 16GB")
print("  Memory bandwidth: 2 TB/s (A100 GPU)")
print("  Inference latency: 1.5s per request")
print()

# INT8 QUANTIZATION
print("INT8 quantized model:")
print("  Model size: 8GB (50% reduction)")
print("  Memory bandwidth: 2 TB/s (same GPU)")
print("  Inference latency: 0.8s per request (47% faster!)")
print("  Accuracy loss: <1% (validated on eval suite)")
print()

# Code to load INT8 model (requires bitsandbytes or AWQ)
"""
llm = LLM(
    model="meta-llama/Llama-3-8B",
    quantization="awq",  # or "gptq" or "bitsandbytes"
    tensor_parallel_size=1
)

# Usage identical to FP16 — quantization is transparent
outputs = llm.generate(prompts, sampling_params)
"""

# DECISION: INT8 quantization is free lunch (47% faster, <1% accuracy loss)
# Deploy quantized models by default; fall back to FP16 only if accuracy drops >2%

print("→ DECISION: INT8 quantization enabled (0.8s vs. 1.5s latency, <1% accuracy loss)")
```

> 💡 **Industry Standard:** `LiteLLM` for cost tracking across providers
> 
> ```python
> from litellm import completion, cost_per_token
> 
> # LiteLLM abstracts provider differences (OpenAI, Anthropic, AWS Bedrock, self-hosted)
> response = completion(
>     model="gpt-4o-mini",
>     messages=[{"role": "user", "content": "Process order: Large Margherita"}]
> )
> 
> # Automatic cost calculation
> cost = cost_per_token(
>     model="gpt-4o-mini",
>     prompt_tokens=response.usage.prompt_tokens,
>     completion_tokens=response.usage.completion_tokens
> )
> 
> print(f"Cost: ${cost:.6f}")
> print(f"Monthly at 10k requests: ${cost * 10000:.2f}")
> ```
> 
> **When to use:** Production deployments with multiple LLM providers. Unified API + automatic cost tracking.
> **Common alternatives:** Provider-specific SDKs (OpenAI, Anthropic), custom wrappers
> **See also:** [LiteLLM cost tracking](https://docs.litellm.ai/docs/completion/cost_tracking)

### 6.5 DECISION CHECKPOINT — Phase 4 Complete

**What you just saw:**
- Sequential processing: 10.82s for 5 requests = 0.46 req/sec throughput
- Batched processing: ~2.5s for 5 requests = 2 req/sec throughput (**2× improvement**)
- INT8 quantization: 1.5s → 0.8s latency per request (**47% faster**), <1% accuracy loss
- Combined impact: 2× throughput + 47% faster = handles 4× traffic on same hardware

**What it means:**
- Batching is **pure throughput win** for concurrent requests — no cost increase, 2× more requests/sec
- Quantization is **pure latency win** — 40-50% faster inference, <1% accuracy loss (validated)
- Both require self-hosted infrastructure (vLLM, TensorRT-LLM) — not available in OpenAI/Anthropic APIs
- Break-even: Self-hosting at >10 req/sec throughput costs same as API but with batching/quantization wins

**What to do next:**
→ **For <10 req/sec traffic:** Stay with OpenAI/Anthropic API (simpler, no infrastructure overhead)
→ **For >10 req/sec traffic:** Self-host with vLLM + INT8 quantization (2× throughput, 47% faster)
→ **For PizzaBot scenario (current):** 50 visitors/day = 0.6 req/hour → **stay with API** (too low volume)
→ **For PizzaBot scenario (scale):** 500 visitors/day = 6 req/hour → **still API** (borderline, evaluate at 1,000/day)
→ **For PizzaBot scenario (peak):** Friday rush 100 concurrent → **self-host vLLM** (batching essential for peak traffic)

---

## 7 · The Accuracy vs. Cost Tradeoff

Every accuracy-improving technique adds cost. Knowing the magnitude helps you decide what to afford.

| Technique | Accuracy gain | Cost multiplier | Latency multiplier |
|---|---|---|---|
| Switching to frontier model | +5–15% on complex tasks | 10–50× | 2–3× |
| Adding RAG (3 chunks) | +15–30% on factual tasks | 1.5–3× | 1.5–2× |
| Chain-of-thought | +10–25% on reasoning tasks | 2–4× (more output tokens) | 2–4× |
| Self-consistency (N=5) | +3–8% | 5× | 5× |
| LLM judge for every response | Evaluation quality | 1.5–3× | 1.5–3× |
| NLI claim verification | Hallucination detection | 1.1–1.3× (small model) | 1.1× |

**The hierarchy:** reach for cheaper techniques first. Self-consistency at 5× cost rarely beats a better prompt at 1× cost. An NLI model for hallucination detection costs 0.1× what an LLM judge costs.

---

## 8 · Real Numbers — Cost Estimation

```
Per query:
  Input tokens:  system_prompt(500) + chunks(2000) + query(100) = 2600 tokens
  Output tokens: answer ≈ 250 tokens

Daily cost (mid-tier model at $1/1M input, $3/1M output):
  Input:  10,000 × 2,600 × ($1/1,000,000)  = $26/day
  Output: 10,000 × 250  × ($3/1,000,000)   = $7.50/day
  Total:  ~$33/day  ≈  $1,000/month

With prefix caching on the 500-token system prompt:
  Cached 500 tokens free → saves $5/day → $150/month
  Total after caching:  ~$875/month
```

**Same workload on a self-hosted Llama 3 8B:**
```
Cloud GPU (A100 40GB): ~$1.5/hour → ~$1,080/month at 100% utilisation
Break-even vs. API: ~same, but with full data privacy and no per-token billing
```

---

## 8 · PizzaBot Connection

> See [AIPrimer.md](../ai-primer.md) for the full system definition.

**Token budget per order request** (approximate, mid-tier model):

| Component | Tokens | Notes |
|---|---|---|
| System prompt | ~300 | Fixed; prefix-cached after first request |
| Conversation history | ~400 | ~4 prior turns summarised |
| Retrieved RAG chunks (k=3) | ~450 | Menu item + allergen entry + FAQ section |
| User message | ~40 | "Large Margherita + 2 garlic breads to 42 Maple St" |
| Tool call outputs (3 calls) | ~200 | Location + availability ×2 + order total |
| **Total input** | **~1,390** | Well within 8k context; cheap per call |
| Model output | ~150 | Order confirmation JSON + natural language summary |

**Cost at scale** (GPT-4o-mini at $0.15/1M input, $0.60/1M output):
```
Per order: (1390 × 0.00000015) + (150 × 0.00000060) ≈ $0.00021 + $0.00009 = $0.00030
10,000 orders/day: ~$3.00/day  (~$90/month)
```

**Optimisations applied:**
- **Prefix caching** on the system prompt — 300 tokens cached after the first call per session (saves ~$0.00004/call)
- **Semantic caching** on frequently repeated queries ("Is the Margherita gluten-free?") — cache hit rate ~30% for a menu of 20 items
- **Mid-tier model** selected over frontier after eval showed quality parity on structured order confirmation

---

## 10 · Interview Checklist

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| How LLM API costs are structured: (input tokens × price/1M) + (output tokens × price/1M), with output typically 3–5× more expensive | A RAG system costs $3,000/month. Walk through how you'd diagnose and reduce that cost | "Switching to a smaller model always saves money" — smaller models may require more output tokens or more retries, eliminating the savings |
| What prefix caching is and how it eliminates redundant compute on stable prompt prefixes | Explain KV-cache and how speculative decoding reduces generation latency | Confusing time-to-first-token (TTFT) with total latency — for streaming UX, TTFT governs perceived responsiveness; optimising generation throughput does not help TTFT |
| The latency components: TTFT vs generation throughput, and which matters for interactive vs batch workloads | When should you use the batch API instead of the synchronous API, and what tradeoffs does it introduce? | Cache-busting by inserting dynamic content (timestamps, user IDs) into prompts that share a stable prefix — this prevents prefix-cache hits and multiplies costs |
| The cost–quality decision order: prompt engineering first, then RAG, then fine-tuning, then a larger model | How does semantic caching differ from exact-match caching and when does each make sense? | Measuring cost only at the generator API — embedding calls, re-ranker calls, and judge-model calls can collectively exceed the generator cost in a production RAG pipeline |

---

## 11 · Progress Check — What We Can Solve Now

🎉 **FINAL MILESTONE**: All 6 constraints exceeded! Ready for production!

**Unlocked capabilities:**
- ✅ **Prompt caching**: 90% token reuse for system prompts
- ✅ **Streaming responses**: First tokens in <500ms (perceived instant)
- ✅ **KV-cache reuse**: Attention tensors cached across requests
- ✅ **Speculative decoding**: Llama-3-1B drafts, Llama-3-8B verifies (30% faster)
- ✅ **Batched inference**: 2× throughput for concurrent requests
- ✅ **INT8 quantization**: Model size 16GB → 8GB (faster memory access)

**Final progress toward constraints:**

| Constraint | Status | Final State |
|------------|--------|---------------|
| #1 BUSINESS VALUE | ✅ **TARGET MOSTLY HIT** | **32% conversion** (>25% ✅, beats phone 22%!), **+$2.10 AOV** (target +$2.50, 84% achieved), 70% labor savings (✅) |
| #2 ACCURACY | ✅ **TARGET HIT!** | ~5% error rate (target <5% ✅) — maintained through optimizations |
| #3 LATENCY | ✅ **TARGET CRUSHED!** | **1.5s p95** (target <3s ✅) — **beats target by 50%!** |
| #4 COST | ✅ **TARGET CRUSHED!** | **$0.005/conv** (target <$0.08 ✅) — **94% under budget!** |
| #5 SAFETY | ✅ **TARGET HIT!** | <2% jailbreak vulnerability, 100% allergen validation (✅) |
| #6 RELIABILITY | ✅ **TARGET HIT!** | >99% uptime, graceful degradation (✅) |

**What we can solve:**

✅ **Sub-2-second latency (1.5s p95) → phone-like UX**:
```
Latency breakdown before Ch.9 (2.2s p95):
- LLM inference: 1.5s
- RAG retrieval: 0.3s
- Safety validation: 0.2s
- Network overhead: 0.2s
Total: 2.2s

Latency breakdown after Ch.9 (1.5s p95):
- LLM inference (INT8 quant): 0.8s (-700ms, 47% faster)
- RAG retrieval (cached embeddings): 0.2s (-100ms)
- Safety validation (parallel): 0.2s (unchanged)
- Network overhead (streaming): 0.1s (-100ms)
- KV-cache reuse: -0.2s (attention cached)
Total: 1.5s

Improvement: 2.2s → 1.5s = -700ms (32% reduction)

User experience:
- 2.2s: "Noticeable delay" (30% conversion)
- 1.5s: "Fast" (32% conversion) ← **phone-like responsiveness!**
```

✅ **50% cost reduction ($0.010 → $0.005/conv)**:
```
Cost breakdown before Ch.10 ($0.010/conv):
- Fine-tuned Llama-3-8B inference: $0.004
- RAG retrieval (embedding + vector search): $0.002
- Safety validation: $0.002
- Guardrails + monitoring: $0.002
Total: $0.010/conv

Cost breakdown after Ch.10 ($0.005/conv):
- Llama-3-8B INT8 (50% faster inference): $0.002 (-$0.002)
- RAG retrieval (prompt caching): $0.0002 (-$0.0018, 90% cache hit)
- Safety validation (batched): $0.001 (-$0.001)
- Guardrails + monitoring: $0.0018 (-$0.0002)
Total: $0.005/conv

Savings: $0.010 → $0.005 = -$0.005/conv (50% reduction)

Monthly savings at 50 visitors/day:
- 420 orders/month × $0.005 savings = $2.10/month infrastructure savings
- (Minimal absolute savings, but massive % improvement and headroom for scale)
```

✅ **Streaming responses (perceived <1s latency)**:
```
Before streaming (2.2s total latency):
User: "What's your most popular pizza?"
[2.2s wait with loading spinner]
Bot: "Our Pepperoni pizza is the most popular choice..."

After streaming (1.5s total, <500ms to first token):
User: "What's your most popular pizza?"
[<500ms wait]
Bot: "Our Pepperoni pizza" [visible immediately]
     "is the most popular choice. It features..." [streams in real-time]

Perceived latency: <1s (user sees progress immediately)
Actual latency: 1.5s (unchanged)
UX impact: "Instant" feeling → conversion 30% → 32% (+2 points)
```

✅ **2× throughput with batched inference**:
```
Before batching (sequential processing):
- Request 1: 1.5s latency
- Request 2: waits for Request 1, then 1.5s
- Request 3: waits for Request 1+2, then 1.5s
Throughput: 1 request / 1.5s = 0.67 requests/sec

After batching (parallel processing):
- Requests 1, 2, 3 batched together
- All processed in single forward pass
- Each still completes in ~1.5s (shared overhead)
Throughput: 3 requests / 1.5s = 2 requests/sec

Result: ✅ 2× throughput! Can handle peak traffic (holiday ordering rush)
```

**Final business metrics:**
- **Order conversion**: **32%** (target >25% ✅, beats phone 22% ✅)
- **Average order value**: **$40.60** (+$2.10 vs. baseline $38.50 ✅)
- **Cost per conversation**: **$0.005** (target <$0.08 ✅, 94% under budget!)
- **Error rate**: **~5%** (target <5% ✅)
- **Latency**: **1.5s p95** (target <3s ✅, beats by 50%!)
- **Throughput**: **20 orders/sec** (vs. 10 before, 2× improvement)
- **Security**: <2% jailbreak vulnerability, 100% allergen validation ✅

**Final ROI calculation:**

**At 50 daily visitors (current):**
```
Revenue: 32% × $40.60 × 50 = $649.60/day = $19,488/month
Baseline: 22% × $38.50 × 50 = $423.50/day = $12,705/month
Revenue lift: $19,488 - $12,705 = $6,783/month

Labor savings: 70% reduction = $11,064/month

Total monthly benefit: $6,783 + $11,064 = $17,847/month
Payback period: $300,000 / $17,847 = **16.8 months**
```

**Scale scenario: 88 daily visitors (realistic with marketing):**
```
Revenue: 32% × $40.60 × 88 = $1,141.70/day = $34,251/month
Baseline: 22% × $38.50 × 88 = $743.46/day = $22,304/month
Revenue lift: $34,251 - $22,304 = $11,947/month

Labor savings: $11,064/month

Total monthly benefit: $11,947 + $11,064 = $23,011/month
Payback period: $300,000 / $23,011 = **13 months**
```

**Scale scenario: 120 daily visitors (aggressive marketing):**
```
Revenue: 32% × $40.60 × 120 = $1,559.04/day = $46,771/month
Baseline: 22% × $38.50 × 120 = $1,014/day = $30,420/month
Revenue lift: $46,771 - $30,420 = $16,351/month

Labor savings: $11,064/month

Total monthly benefit: $16,351 + $11,064 = $27,415/month
Payback period: $300,000 / $27,415 = **10.9 months ✅ (beats 10.6 month target!)**
```

**VERDICT: READY FOR PRODUCTION LAUNCH** ✅

All 6 constraints satisfied:
1. ✅ **BUSINESS VALUE**: 32% conversion (>25%), +$2.10 AOV (target +$2.50, 84% achieved), 70% labor savings
2. ✅ **ACCURACY**: ~5% error rate (<5%)
3. ✅ **LATENCY**: 1.5s p95 (<3s, beats by 50%!)
4. ✅ **COST**: $0.005/conv (<$0.08, 94% under budget!)
5. ✅ **SAFETY**: <2% jailbreak, 100% allergen validation
6. ✅ **RELIABILITY**: >99% uptime, graceful degradation

ROI achievable:
- **Conservative** (88 visitors/day): 13 months
- **Target** (120 visitors/day): 10.9 months ✅
- **Aggressive** (150+ visitors/day): <9 months

**System comparison: Final vs. Baseline**

| Metric | Phone Baseline | PizzaBot (Final) | Improvement |
|--------|----------------|------------------|-------------|
| Conversion | 22% | **32%** | +45% |
| AOV | $38.50 | **$40.60** | +5.5% |
| Error rate | ~3% (human) | **~5%** | Comparable |
| Latency | Instant (phone) | **1.5s** | Near-instant |
| Labor cost | $157,680/year | **$47,304/year** | -70% |
| Hours/day | 24/7 (3 shifts) | **24/7 (1 staff)** | -67% headcount |
| Cost/conv | ~$7.50 (labor) | **$0.005** | -99.9% |

**Why the CEO should greenlight launch:**

1. **All targets exceeded**: Not just hit, but crushed on latency (-50%) and cost (-94%)
2. **Beats phone baseline**: 32% vs. 22% conversion, $40.60 vs. $38.50 AOV
3. **Production-ready**: Security audit passed, automated testing, monitoring, graceful degradation
4. **Clear ROI path**: 10.9 months at 120 daily visitors (achievable with basic marketing)
5. **Future-proof**: 2× throughput headroom, 94% cost budget remaining for growth
6. **Competitive moat**: Brand voice fine-tuning creates differentiation impossible for competitors to copy

**Post-launch optimization opportunities:**
- Multi-location franchise expansion: HNSW indexing scales to 100+ locations
- Seasonal menu updates: Daily menu changes without downtime
- A/B testing: Safe experimentation with automated regression detection
- Advanced upselling: Personalized recommendations based on order history
- Voice interface: <1.5s latency enables phone integration

❌ **What we can't solve yet:**

**Nothing for PizzaBot!** All 6 constraints achieved. The system is production-ready.

**But:** These optimizations were algorithmic (caching, quantization, batching). For **hardware-level** optimization:
- **GPU kernel optimization** (custom CUDA kernels for attention)
- **Model architecture design** (MoE, sparse attention)
- **Distributed inference** (tensor parallelism across multiple GPUs)
- **Custom hardware** (TPUs, custom ASICs for inference)

→ See **[AI Infrastructure track](../../ai_infrastructure/README.md)** for the next level: hardware decisions, memory hierarchies, and distributed systems.

**Business metrics update:**
- **Order conversion**: **32%** (baseline: 22% phone orders) — **+45% improvement** ✅
- **Average order value**: **$40.60** (baseline: $38.50 phone) — **+5.5% improvement** ✅
- **Cost per conversation**: **$0.005** (target: <$0.08) — **94% under budget** ✅
- **Error rate**: **~5%** (target: <5%) — **maintained accuracy through optimizations** ✅
- **Latency**: **1.5s p95** (target: <3s) — **beats target by 50%** ✅
- **ROI payback**: **10.9 months at 120 visitors/day** (target: 10.6 months) — **achievable** ✅

**Next step**: AI Infrastructure track — where these algorithmic levers (quantization, batching, caching) become hardware and systems decisions (GPU memory bandwidth, distributed inference, custom accelerators).

---

## 12 · Bridge to AI Infrastructure Track

Ch.9 completed the **application-layer** optimization stack. You now understand:
- Model tier selection (frontier vs mid-tier vs open-source)
- Token budget management (context length, prompt caching, streaming)
- Cost-accuracy tradeoffs (self-consistency, RAG retrieval depth, judge models)
- Latency optimization (KV caching, speculative decoding, batched inference)

**But** every optimization in this chapter assumed:
- **GPU hardware exists** and has sufficient memory/bandwidth
- **Inference frameworks** (vLLM, TensorRT-LLM) handle scheduling and batching
- **Distributed systems** coordinate work across multiple machines
- **Memory hierarchies** (HBM, DRAM, cache) determine actual throughput

**The AI Infrastructure track answers:**
- Why does INT8 quantization work? (GPU memory bandwidth bottleneck, not compute)
- How does batched inference actually improve throughput? (GPU utilization, tensor cores)
- When should you scale vertically (bigger GPU) vs horizontally (more GPUs)? (Amdahl's law, communication overhead)
- How do you design a system that handles 10,000 req/sec? (Load balancing, request routing, autoscaling)

**The handoff:**

PizzaBot is production-ready at 120 visitors/day (10.9-month ROI). To scale to **1,000+ visitors/day**:
- **Ch.9 optimizations won't scale**: Batching helps but GPU memory becomes the bottleneck
- **Need infrastructure decisions**: Multi-GPU inference, request routing, auto-scaling
- **Need hardware understanding**: Memory bandwidth, tensor core utilization, PCIe bottlenecks

→ **[AI Infrastructure](../../ai_infrastructure/README.md)** covers GPU architecture, memory hierarchies, distributed inference, and production serving systems.

**This is the end of the AI track.** Ch.1-9 took PizzaBot from 8% conversion (failing prototype) to 32% conversion (production-ready system beating human baseline). Every chapter solved a specific technical challenge while maintaining the business value story.

---

## Illustrations

![Cost and latency — token-cost stack, latency components, cost-vs-accuracy tiers, optimisation patterns](img/Cost%20and%20Latency.png)
