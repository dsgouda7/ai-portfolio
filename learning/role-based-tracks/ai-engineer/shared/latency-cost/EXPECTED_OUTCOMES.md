# Latency and Cost Expected Outcomes

All calculations use the five requests exactly as recorded. A request total must equal the sum of its stage values for both latency and cost.

## Request-Level Metrics

- Success rate: 4 of 5, or 80%.
- Successful-request latency values are 165, 265, 285, and 7 ms. Sorted: 7, 165, 265, 285 ms.
- Nearest-rank p50 successful-request latency: 165 ms.
- Nearest-rank p95 successful-request latency: 285 ms.
- Total billed usage: 600 input tokens and 36 output tokens.
- Total served output: 30 tokens, including 6 tokens served from cache.
- Total observed cost: 6,420 micro-USD, or $0.006420.
- Cost per successful request includes spend from the failed request: 6,420 / 4 = 1,605 micro-USD, or $0.001605.

## Generation Metrics

Use only final successful generation stages for TTFT, TPOT, and uncached generation throughput.

- TTFT values: 40, 60, 50 ms. Nearest-rank p50 = 50 ms; p95 = 60 ms.
- TPOT values: 10, 20, 15 ms. Nearest-rank p50 = 15 ms; p95 = 20 ms.
- Successful uncached output: 6 + 10 + 8 = 24 tokens.
- Successful generation time: 100 + 200 + 120 = 420 ms.
- Successful output throughput: 24 / 0.420 = 57.14 tokens/second.

There are 6 generation attempts across 4 generation-bearing requests, so retry amplification is 6 / 4 = 1.50. Across successful uncached requests only, it is 4 / 3 = 1.33; a report must state which denominator it uses.

## Stage Percentiles

Using all observations of each stage and nearest-rank percentiles:

| Stage | p50 ms | p95 ms |
| --- | ---: | ---: |
| Gateway | 5 | 5 |
| Cache lookup | 2 | 2 |
| Embedding | 8 | 8 |
| Retrieval | 20 | 20 |
| Reranking | 30 | 30 |
| Generation attempt | 100 | 200 |
| Retry backoff | 20 | 20 |

## Cache and Bottleneck

`req-004` reuses `req-001`. The exact hit saves 165 - 7 = 158 ms and 1,030 micro-USD relative to its origin request. Generation has both the highest p50 and p95 stage latency and is the expected first bottleneck diagnosis. Retries amplify both generation cost and tail latency; optimizing retrieval first is not supported by this fixture.
