# Stage 08: LLM Evaluation

**Previous:** [Stage 07 · RAG](../07-rag/README.md)

**Current:** Stage 08 · LLM Evaluation

**Next:** [Stage 09 · LLM Gateway](../09-llm-gateway/README.md)

## Input Contract from Stage 07

Evaluation receives a chain, not one RAG score:

| Evidence link | Stage 08 responsibility |
| --- | --- |
| Retrieval recall | Keep evidence-miss failures separate from answer failures |
| Context support | Test whether the supplied passages entail each answer claim |
| Claim attribution | Retain source-to-claim links and localize unsupported or contradictory claims |
| Confidence and risk routing | Calibrate the decision signal, then choose answer, abstain, or review under explicit costs |

Oracle context remains an interpretation tool: a repaired answer points upstream to retrieval;
an unrepaired answer points to context use, answer construction, or the evaluator.

This four-notebook sequence has one canonical owner for each evaluation question:

| Chapter | Mechanics built and measured | Capability earned |
| --- | --- | --- |
| [Automated Metrics and Benchmarks](01-llm-evaluation-metrics-and-benchmarks.ipynb) · [Theory notes](01-llm-evaluation-metrics-and-benchmarks-theory.md) | Reference/semantic/model-based readings, disagreement records, item and slice results | Choose an instrument for a stated claim and preserve disagreement instead of calling one proxy "quality" |
| [LLM-as-Judge, Safety, and Eval Pipeline](02-llm-as-judge-safety-and-pipeline.ipynb) · [Theory notes](02-llm-as-judge-safety-and-pipeline-theory.md) | Rubrics, order swaps, agreement checks, safety gates, release records | Audit a judge and make release evidence accountable |
| [Hallucination Detection](03-hallucination-detection.ipynb) · [Theory notes](03-hallucination-detection-theory.md) | Atomic claims, support/contradiction labels, cost-sensitive thresholds | Localize unsupported claims and route bounded review |
| [Calibration and Confidence](04-calibration-and-confidence.ipynb) · [Theory notes](04-calibration-and-confidence-theory.md) | Reliability, ECE, temperature scaling, risk-coverage, independent risk gates | Convert confidence into answer/abstain/review policy without confusing confidence with claim support |

1. **Metrics and benchmarks:** what reference, semantic, and model-based metrics can observe.
2. **LLM-as-judge and safety:** how a judge and release pipeline can be trusted.
3. **Hallucination detection:** how unsupported or contradictory claims are localized.
4. **Calibration and confidence:** how confidence becomes abstention policy.

## Shared Evidence Ladder

Each chapter climbs the same ladder: **plain-language failure -> static table or plot -> animation
when a threshold or route changes over time -> one-variable measured comparison -> readable
implementation -> real artifact or policy bridge**. Static evidence remains sufficient when
JavaScript is unavailable. The teaching fixtures prove evaluator mechanics; only representative,
versioned runs can support a model-quality claim.

## Exact Evaluation-to-Gateway Matrix

Stage 09 may enforce only decisions justified by versioned Stage 08 evidence:

| Evaluation evidence | Gateway control | Runtime decision |
| --- | --- | --- |
| Capability/slice benchmark passes its release threshold | Route eligibility for that matching request class | Include the deployment in the approved set; a failing or unmeasured slice is ineligible |
| Safety gate passes | Hard policy eligibility | Dispatch only to releases that passed the required safety gate; never average a safety failure into quality |
| Atomic claim support and attribution pass | Output policy | Return attributed output; otherwise refuse, abstain, or send to bounded review |
| Calibrated confidence is above the risk-cost threshold | Confidence/risk route | Answer on the eligible route; below threshold, abstain, fall back only to another eligible route, or escalate |
| Judge disagreement or evaluator uncertainty exceeds its review threshold | Review gate | Do not convert an uncertain offline verdict into automatic runtime authority |
| Versioned regression or drift gate fails | Release/route state | Disable or roll back that evaluated release and retain the triggering evidence version in the trace |

Latency and cost optimize **inside** the approved set. They do not repair failed quality, safety,
support, confidence, or authority gates.

`evaluation_shared.py` centralizes the Riverside fixture, deterministic seeding, token
normalization, schema checks, and the generic teaching caveat. Each notebook retains its unique
measured evidence and links forward instead of repeating the full metric warning or judge-bias
taxonomy.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this
chapter's `.venv`, installs the adjacent `requirements.txt`, registers the chapter-unique
`learning-08-llm-evaluation` Jupyter kernel, and assigns it to all four notebooks.

## Continue Into Operations

This chapter owns evaluator design, metric limits, hallucination checks, safety evaluation, and
calibration. Continue with:

- [AI Engineer: Prompt Release and Experimentation](../12-role-based-tracks/ai-engineer/02-prompt-release-and-experimentation/README.md)
	for paired comparison, critical-slice gates, uncertainty, shadow design, and rollback evidence;
- [AI Engineer: Production Feedback and Drift](../12-role-based-tracks/ai-engineer/05-production-feedback-and-drift/README.md)
	for privacy-safe monitoring, reviewed failure clusters, and evaluation-set updates;
- [Riverside Evaluation Assets](../../projects/riverside-ai-platform/evaluations/README.md) and
	[evaluation strategy](../../projects/riverside-ai-platform/docs/evaluation-strategy.md) for
	versioned production-oriented gate inputs and evidence policy.

The AI Engineer and FDE operational learning notebooks were executed successfully locally and then
cleared. Riverside evaluation source assets exist, and its non-cloud suite passed 142 tests with 5
cloud tests deselected. That is **operational asset test evidence**, not model-quality evidence.
Model, endpoint, rollout, live Azure evaluation results, and production readiness remain
**live-unvalidated**.
