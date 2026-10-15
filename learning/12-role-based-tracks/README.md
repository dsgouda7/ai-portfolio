# Stage 12 — Role-Based Tracks

**Contents:** [one-line map of this directory](CONTENTS.md)

**Previous:** [Stage 11 — AI Infrastructure](../11-ai-infrastructure/) · **Current:** Stage 12 — Role-Based Tracks · **Next:** complete the chosen role capstone and retain portfolio evidence.

These tracks organize existing technical chapters around the work performed in a particular role.
For a first-time zero-to-hero learner, Stage 12 begins after the numbered route through Stage 11.
The role tracks synthesize that route; they do not replace it, reorder it, or turn Stages 10 and 11
into optional topics.

Use a role track when you want to practice joining concepts across model behavior, data, evaluation, infrastructure, delivery, and operations in one job-shaped workflow.

## Canonical Route and Targeted Entry

```text
Stages 01-09: model and application core
-> Stage 10: bounded agent systems
-> Stage 11: hardware, training, inference, and serving systems
-> Stage 12: role synthesis
-> role capstone
-> supervised practice where the role requires live systems or customer work
```

Prior experience can shorten study, but it does not remove an evidence requirement. A targeted
entrant may challenge out of the advanced stages only when they can attach equivalent evidence:

| Stage | Minimum challenge-out evidence before role synthesis |
|---|---|
| [10 — Agentic AI](../10-agentic-ai/README.md) | A bounded tool contract, explicit state/recovery behavior, an evaluation trace, and proof that unsafe or duplicate action fails closed |
| [11 — AI Infrastructure](../11-ai-infrastructure/README.md) | A measured bottleneck diagnosis, memory/throughput reasoning, and a workload-specific serving decision with stated evidence limits |

Passing a gate changes what you need to study. It does not change the canonical order or what the
role capstone expects you to understand.

| Track | What is this? | When should I use it? | Where do I start? |
|---|---|---|---|
| AI Engineer | A practice route connecting data, model adaptation, retrieval, evaluation, serving, releases, and feedback into one loop. | You want to learn how technical components become an observable, releasable AI application. | [AI Engineer route](ai-engineer/README.md) |
| Forward Deployed Engineer | A practice route for turning an ambiguous customer workflow into a bounded design, rollout, operating model, and handoff. | You already have or are building the AI Engineer core and want to practice customer-facing delivery decisions. | [FDE route](fde/README.md) |

## First-Time Learner Sequence

For each step, learn the concept first, then use the matching role chapter to practice the decision it
supports. This table is a synthesis index, not a substitute for Stages 01-11. Start with the
[AI Engineer baseline](ai-engineer/00-role-baseline-and-route.md); it records evidence for any
prior-experience challenge-out.

| Step | Learn the concept | Then practice it in the role route |
|---:|---|---|
| 1 | [Fine-tuning data techniques](../06-llm-finetuning/01-llm-finetuning-data-techniques.ipynb) | [Training Data Quality and Lineage](ai-engineer/01-training-data-quality-and-lineage/README.md) |
| 2 | [RAG evaluation](../07-rag/02-rag-failure-localization-and-evaluation.ipynb) and [evaluation metrics](../08-llm-evaluation/01-llm-evaluation-metrics-and-benchmarks.ipynb) | [Prompt Release and Experimentation](ai-engineer/02-prompt-release-and-experimentation/README.md) |
| 3 | [LLM gateway](../09-llm-gateway/README.md) and [inference systems](../11-ai-infrastructure/07-inference-systems/inference-systems.ipynb) | [Application Latency and Cost](ai-engineer/03-application-latency-and-cost/README.md) |
| 4 | [Fine-tuning comparison and decision](../06-llm-finetuning/03-llm-finetuning-comparison-and-decision.ipynb) | [Release Registry and Lineage](ai-engineer/04-release-registry-and-lineage/README.md) |
| 5 | [Agent evaluation and observability](../10-agentic-ai/05-agent-evaluation-and-observability/05-agent-evaluation-and-observability.ipynb) | [Production Feedback and Drift](ai-engineer/05-production-feedback-and-drift/README.md) |

After completing Steps 1-5, confirm that Stage 10 and Stage 11 are complete or supported by the
challenge-out evidence above. Then choose the
[AI Engineer capstone](ai-engineer/06-capstone/README.md) or continue to the
[FDE baseline and engagement lifecycle](fde/00-role-baseline-and-engagement-lifecycle.md) for
discovery-to-handoff practice.

Both capstones now use the same graduation loop:

```text
initial decision
-> weakest evidence
-> smallest next experiment
-> re-review
-> one transfer exercise
-> graduation and practicum handoff
```

Finish with the [Stage 12 Graduation and Practicum Handoff](graduation-and-practicum-handoff.md).
It states what your portfolio may support, what remains local or synthetic, what requires
independent review, and what requires supervised live or customer practice.

## Validation and Status Transparency

The notebooks use local or synthetic fixtures unless a chapter explicitly says otherwise. Authored
route notebooks were executed successfully against those fixtures and then had outputs and execution
counts cleared so learners begin from a clean state. Cleared output is not a failed or untested state.

Some linked operational material uses labels such as `HOLD_LOCAL_RELEASE` and
`HOLD_AZURE_PROMOTION`. These are safety gates, not errors: a measured threshold or required validation
has not passed, so release or cloud promotion remains blocked by design. `Live-unvalidated` similarly
means that source or design assets exist without retained evidence from the named cloud or customer
environment. Completing a track demonstrates structured practice, not independent production authority,
customer acceptance, or live-cloud validation.
