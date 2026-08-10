# Production Feedback and Drift

> **Evidence banner:** `FIXTURE` inputs, `VALIDATED` deterministic outcomes,
> `OUTPUTS CLEARED`, `UNVALIDATED` production behavior.

This chapter closes Riverside House's production loop without retaining raw
prompts. You compare two six-trace windows, separate seven kinds of drift,
review recurring failures, promote only reviewed privacy-safe summaries into a
versioned evaluation candidate, and choose the component that the evidence
actually implicates.

The notebook makes no provider, model, network, or cloud call. It reads the
shared synthetic fixture directly and writes generated candidate artifacts only
when a learner later runs the relevant cell.

## Start Here

1. Run `setup.ps1` on Windows or `setup.sh` on macOS/Linux.
2. Open `production-feedback-and-drift.ipynb`.
3. Select `Python (AI Engineer 05 Feedback and Drift .venv)`.
4. Run from the top when you are ready to create local fixture evidence.

The setup and notebook completed successfully in the unified FDE environment.
Notebook outputs and execution counts were then cleared for reuse.

## What You Build

- a privacy-safe production sampling policy with must-keep strata;
- a contract-validated baseline/current comparison;
- separate traffic, data, retrieval, quality, latency, cost, and policy drift
  signals;
- proportion uncertainty and explicit small-sample review zones;
- deterministic multi-label failure clusters and a review queue;
- three reviewed evaluation-candidate cases plus a versioned manifest and hash;
- an intervention record covering prompt, retrieval/index, guardrail,
  fine-tuning, and no action;
- an operating-loop checklist for follow-up measurement and rollback decisions.

## Expected Fixture Outcomes

| Signal | Baseline | Current | Change |
|---|---:|---:|---:|
| Security traffic share | 16.7% | 50.0% | +33.3 pp |
| Novel summary-category rate | n/a | 33.3% | Coarse data proxy |
| Retrieval hit rate | 83.3% | 50.0% | -33.3 pp |
| Quality pass rate | 83.3% | 50.0% | -33.3 pp |
| Mean latency | 100 ms | 150 ms | +50% |
| Mean cost | 1,000 micro-USD | 1,500 micro-USD | +50% |
| Latency SLO breach rate | 0.0% | 50.0% | +50.0 pp |
| Policy correctness | 100.0% | 83.3% | -16.7 pp |

All three current quality failures are retrieval misses. One of those failures
is also a policy false allow. The supported immediate actions are therefore a
retrieval/index update and an independent fail-closed guardrail change. Latency
and cost need parallel route-level investigation. Prompt change, fine-tuning,
and no action are not supported as first responses by this fixture.

## Privacy Boundary

The fixture contains synthetic query summaries, stable trace/request/release
IDs, categorical outcomes, integer latency/cost, failure codes, and reviewed
labels. It contains no raw prompts, completions, manuscript text, user identity,
tenant identity, credentials, or production endpoint.

In a real system, collect the minimum fields needed for a stated monitoring and
evaluation purpose. Keep operational metrics content-free and bounded. Put any
approved content review behind access control, retention, deletion, audit, and
sampling policy. Never infer that hashing raw text makes it anonymous.

## Exact Health Checks

The notebook requires these checks before making an intervention decision:

1. Every row validates against the frozen JSON Schema.
2. Trace and request IDs are unique; baseline/current each contain six rows.
3. Recorded failure codes agree with retrieval, quality, latency, and policy facts.
4. Raw-content-like fields are absent from telemetry and candidate cases.
5. Every percentage prints its numerator, denominator, and population.
6. Small-sample proportion metrics include Wilson intervals and a warning.
7. Failure clustering preserves multi-label membership rather than forcing one cause.
8. Only reviewed rows become evaluation candidates.
9. Candidate IDs and source trace IDs are one-to-one and preserve release lineage.
10. Canonical candidate serialization produces one deterministic SHA-256 digest.
11. Policy false allows are critical gates and cannot be averaged away.
12. Fine-tuning is rejected unless retrieved evidence and guardrails are healthy
    and reviewed failures demonstrate a persistent learned-behavior gap.

## Completion Evidence

You have finished this chapter when you have:

- retained the verified fixture version and feedback/schema digests;
- retained seven drift lenses with counts, denominators, populations, and uncertainty;
- preserved multi-label failures and reviewer lineage for every promoted evaluation case;
- produced a versioned candidate manifest and canonical digest without raw content or identity;
- recorded why retrieval/index and guardrail changes are supported before prompt change or fine-tuning;
- linked the iteration decision into a capstone evidence index as `LOCAL_FIXTURE` with production representativeness and causal impact left unvalidated.

## Files

| Path | Purpose |
|---|---|
| `production-feedback-and-drift.ipynb` | Complete failure-first notebook, successfully executed and cleared for reuse |
| `requirements.txt` | Minimal local notebook and schema dependencies |
| `setup.ps1`, `setup.sh` | Chapter-local environment and kernel setup |
| `fixtures/README.md` | Pointer to the immutable shared fixture contract |
| `../shared/feedback-drift/production-feedback.jsonl` | Frozen privacy-safe traces |
| `../shared/feedback-drift/production-feedback.schema.json` | Per-line JSON Schema |
| `../shared/feedback-drift/EXPECTED_OUTCOMES.md` | Independent deterministic answer key |

## Conceptual Owners

- [Hybrid retrieval and index decisions](../../../genai/03-rag/01-hybrid-search.ipynb)
- [RAG evaluation and gold-context diagnosis](../../../genai/03-rag/02-rag-evaluation.ipynb)
- [LLM evaluation, judges, hallucination, and calibration](../../../genai/04-llm-evaluation/README.md)
- [Gateway routing, resilience, caching, cost, and trace control](../../../genai/05-llm-gateway/01-llm-gateway.ipynb)
- [Agent evaluation framework](../../../agentic-ai-system-design/07-agent-evaluation-frameworks.md)
- [Observability, sampling, and health](../../../agentic-ai-system-design/08-observability-tracing-and-health.md)
- [Guardrails and fail-closed policy](../../../agentic-ai-system-design/11-governance-guardrails-and-security.md)
- [Repository authoring standard](../../../../AUTHORING_GUIDE.md)

## Honest Limitations

Twelve synthetic records teach arithmetic, lineage, review, and decision order.
They do not prove that six records represent a production distribution; that
Wilson intervals solve sequential monitoring; that summary novelty measures
semantic data drift; that failure-code groups are discovered clusters; that
user feedback is unbiased; that reviewers agree; or that the proposed index and
guardrail changes will improve production. Production evidence needs a
privacy-approved collection policy, representative windows, calibrated alerts,
reviewer agreement, versioned releases and indexes, controlled evaluation,
follow-up traffic, incident handling, and retained rollback evidence.

## Validation Status

The setup and notebook executed successfully in the unified FDE environment,
including the deterministic calculations, assertions, and hashes. Notebook
outputs were then cleared. No production, provider, or cloud behavior was
validated.
