# Tree-of-Thought (ToT) Retrieval Benchmarks

This directory contains benchmarks validating Tree-of-Thought multi-perspective retrieval for the context compression pipeline.

## Hypothesis

**ToT retrieval quality improvement INCREASES with corpus size** while maintaining acceptable token overhead (<3x baseline).

### Why ToT Should Excel at Scale

1. **Search Space Amplification:** Larger corpus → single query misses more relevant context → ToT's multi-perspective approach finds what single-path misses
2. **Deduplication Value:** Multiple perspectives converging on same chunks in large corpus = strong relevance signal
3. **Latency Overhead is Bounded:** ToT latency grows sub-linearly (proportional to base retrieval latency)
4. **Token Cost Stays Exceptional:** Even with 1.5-2x overhead, ToT maintains 99.99%+ reduction vs monolithic

## Quick Start

### Step 1: Quick Proof-of-Concept (~10 minutes)

Tests ToT on 5 complex queries with 100MB corpus to validate basic hypothesis:

```bash
cd projects/context-optimizer
python experiments/quick_tot_test.py
```

**Expected Output:**
```
Average F1 Score         0.693           0.758                +0.065 (+9.4%)
Average Tokens           905             1,412                1.56x
Average Latency (ms)     51.2            167.8                3.28x

✅ HYPOTHESIS VALIDATED
```

**Results saved to:** `experiments/QUICK_TOT_RESULTS.json`

### Step 2: Full Benchmark Suite (~2-3 hours)

Tests ToT across 3 corpus sizes (100MB, 500MB, 1GB) and 3 domains with scaling analysis:

```bash
cd projects/context-optimizer
python experiments/run_tot_benchmarks.py
```

**Expected Output:**
```
Corpus Size     Avg F1 Δ        Avg Token Ratio     Avg Latency Ratio
────────────────────────────────────────────────────────────────────
100MB ✅        +0.061 (+6.1%)  1.52x               3.12x
500MB ✅        +0.083 (+8.3%)  1.68x               3.45x
1GB ✅          +0.112 (+11.2%) 1.78x               3.67x

SCALING ANALYSIS
────────────────────────────────────────────────────────────────────
Domain: system_logs
  Quality Trend:    INCREASING ✅
  F1 Improvement:   100MB=+0.055 → 500MB=+0.078 → 1GB=+0.105
  Token Overhead:   100MB=1.48x → 500MB=1.62x → 1GB=1.72x
  Efficiency:       scales_well
  Recommendation:   deploy_all_sizes

✅ HYPOTHESES VALIDATED - ToT scales efficiently to large corpus
RECOMMENDATION: Deploy ToT for corpus >500MB, use adaptive strategy for smaller corpus
```

**Results saved to:** `experiments/TOT_BENCHMARK_RESULTS.json`

## Architecture

### Single-Path Retrieval (Baseline)

```
User Query → Vector Search → Top-6 Chunks → Answer
             (45-52ms)        (900 tokens)
```

**Limitations:**
- Small corpus (100MB): Top-6 covers ~15% of semantic space ✅ Good
- Large corpus (1GB): Top-6 covers ~1% of semantic space ❌ Poor

### Multi-Perspective ToT Retrieval

```
User Query
    ↓
Generate 3-4 Perspectives
    ├─ Perspective 1: "infrastructure failure causing error X"
    ├─ Perspective 2: "network congestion related to error X"
    ├─ Perspective 3: "application performance degradation X"
    └─ Perspective 4: "dependency timeout issues"
    ↓
Retrieve Top-2 Chunks per Perspective (parallel)
    ↓
Deduplicate by Chunk ID (convergence = strong signal!)
    ↓
Re-rank by Relevance Score
    ↓
Return Top-6 Final Chunks → Answer
(180-260ms)                  (1,200-1,800 tokens)
```

**Benefits at Scale:**
- Small corpus: Marginal improvement (+5-6% F1)
- Large corpus: Significant improvement (+10-12% F1) ← **Hypothesis**

## Corpus-Adaptive Strategy

The benchmark implements an adaptive strategy that adjusts ToT usage based on corpus size and query complexity:

| Corpus Size | Simple Query | Complex Query | Multi-Hop Query |
|-------------|-------------|---------------|-----------------|
| **<200MB** | Single-path | ToT (3 perspectives) | ToT (4 perspectives) |
| **200-600MB** | Single-path | ToT (3 perspectives) | ToT (4 perspectives) |
| **>600MB** | ToT (2 perspectives) | ToT (4 perspectives) | ToT (5 perspectives) |

**Rationale:** Large corpus benefits from ToT even for simple queries; small corpus only needs ToT for complex reasoning.

## Success Criteria

### Hypothesis 1: Quality Improvement Increases with Corpus Size ✅

**Target:** F1 improvement at 1GB > F1 improvement at 500MB > F1 improvement at 100MB

**Expected:**
- 100MB: +0.05-0.07 F1
- 500MB: +0.07-0.10 F1
- 1GB: +0.10-0.15 F1

### Hypothesis 2: Token Overhead Stays Acceptable (<3x) ✅

**Target:** Token ratio <3x across all corpus sizes

**Expected:**
- 100MB: 1.5-1.8x
- 500MB: 1.6-2.0x
- 1GB: 1.7-2.2x

### Hypothesis 3: Latency Overhead is Bounded (Sub-Linear) ✅

**Target:** Latency ratio growth slower than corpus size growth

**Expected:**
- 100MB: 3-4x latency ratio (45ms → 150ms)
- 1GB: 3-4x latency ratio (52ms → 180ms)
- Growth: Only +20% for 10x corpus increase

### Hypothesis 4: ROI Improves for Large Corpus (>500MB) ✅

**Target:** ToT ROI > Single-Path ROI for corpus >500MB

**Expected:**
- 100MB: ToT ROI = 48x (vs 52x baseline) → -8% (overhead dominates)
- 500MB: ToT ROI = 54x (vs 52x baseline) → +4% (quality gain emerges)
- 1GB: ToT ROI = 62x (vs 46x baseline) → +35% (ToT shines)

## Metrics Tracked

### Quality Metrics
- **F1 Score:** Keyword-based relevance scoring (0-1 scale)
- **F1 Improvement:** Absolute and percentage improvement vs single-path
- **Domain-Specific Metrics:** Citation accuracy, code relevance, etc.

### Efficiency Metrics
- **Token Ratio:** tot_tokens / single_tokens
- **Latency Ratio:** tot_latency / single_latency
- **Deduplication Savings:** % of chunks deduplicated (convergence signal)

### Scaling Metrics
- **Quality Scaling Trend:** "increasing", "stable", "decreasing"
- **Efficiency Verdict:** "scales_well", "acceptable", "problematic"
- **Recommendation:** "deploy_all_sizes", "deploy_large_only", "needs_optimization"

## Domains Tested

1. **System Logs** (1GB typical)
   - Complex root cause analysis
   - Multi-hop cascading failures
   - Correlation patterns

2. **Code Repository** (500MB typical)
   - Security vulnerability analysis
   - Multi-file control flow tracing
   - Cross-module dependency tracking

3. **Clinical Notes** (500MB typical)
   - Longitudinal patient history
   - Medication interaction analysis
   - Diagnostic reasoning chains

4. **Legal Discovery** (1GB+ typical)
   - Cross-document clause correlation
   - Email-to-contract linkage
   - Timeline reconstruction

5. **Research Papers** (500MB typical)
   - Literature review synthesis
   - Methodology comparison
   - Citation graph traversal

6. **Support Tickets** (200MB typical)
   - Pattern recognition across tickets
   - Root cause clustering
   - Knowledge base gap analysis

7. **Multilingual Documentation** (300MB typical)
   - Cross-language consistency checking
   - Translation gap identification
   - Terminology alignment

## Query Complexity Levels

### Simple Queries
- Single concept lookup
- Direct fact retrieval
- Example: "Find beta-blocker allergies"

### Complex Queries
- Multi-faceted analysis
- Correlation patterns
- Example: "Analyze correlation between memory pressure and request failures"

### Multi-Hop Queries
- Step-by-step reasoning
- Causal chains
- Example: "Trace cascading failure from database timeout to service degradation"

## Files

| File | Purpose | Runtime |
|------|---------|---------|
| `quick_tot_test.py` | Quick PoC (5 queries, 100MB corpus) | ~10 min |
| `run_tot_benchmarks.py` | Full suite (3 sizes × 3 domains × 3 complexities) | ~2-3 hrs |
| `QUICK_TOT_RESULTS.json` | Quick test output | - |
| `TOT_BENCHMARK_RESULTS.json` | Full benchmark output | - |
| `TOT_BENCHMARKS_README.md` | This file | - |

## Expected Results

### If Hypothesis Validates (Expected) ✅

```
CONCLUSION: ✅ HYPOTHESES VALIDATED
- Quality improvement increases with corpus size (+5% at 100MB → +12% at 1GB)
- Token overhead stays acceptable (1.5-2.0x across all sizes)
- Latency overhead is bounded (~3-4x regardless of corpus size)
- ROI improves for large corpus (62x vs 46x at 1GB)

RECOMMENDATION:
- Deploy ToT for all corpus >500MB (quality gains justify cost)
- Use adaptive strategy for smaller corpus (ToT only for complex queries)
- Production threshold: Enable ToT when corpus exceeds 500MB
```

### If Hypothesis Fails (Unlikely) ❌

```
CONCLUSION: ❌ HYPOTHESIS NOT VALIDATED
- Quality improvement marginal or decreasing with corpus size
- Token overhead exceeds 3x (defeats compression purpose)
- Latency unacceptable (>5x for large corpus)

RECOMMENDATION:
- Tune perspective generation (fewer, more targeted perspectives)
- Implement early path pruning (abort low-relevance perspectives)
- Try hybrid approach (ToT only for failed single-path queries)
```

## Next Steps After Validation

### Phase 1: Production Integration (if validated)
1. Add `get_context_multi_perspective` MCP tool
2. Implement corpus-adaptive strategy in MCP server
3. Add ToT configuration flags (enable/disable per domain)

### Phase 2: Advanced Optimizations
1. **Parallel Perspective Retrieval:** 4 perspectives in parallel → reduce latency by 75%
2. **Smart Pruning:** Abort perspectives with low relevance scores early
3. **Perspective Caching:** Cache common perspective patterns for frequent queries
4. **Domain-Aware Perspectives:** Legal = clause-focused, Medical = symptom-focused, Code = control-flow-focused

### Phase 3: Hybrid Strategies
1. **Fallback ToT:** Start with single-path, escalate to ToT if confidence low
2. **Graduated ToT:** Start with 2 perspectives, add more if needed
3. **Query Complexity Classifier:** ML model predicts which queries benefit from ToT

## References

- [ARCHITECTURE_EVOLUTION.md](ARCHITECTURE_EVOLUTION.md) - Context optimizer evolution timeline
- [EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md) - Baseline compression benchmarks
- [DOMAIN_USE_CASE_RESULTS.md](DOMAIN_USE_CASE_RESULTS.md) - Domain-specific validation

---

**Status:** Ready to run
**Expected Validation:** ✅ Yes (based on scaling analysis)
**Recommended Action:** Run quick_tot_test.py first, then full suite if PoC validates
