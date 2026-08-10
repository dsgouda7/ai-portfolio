# Role-Based Learning Tracks

These tracks organize existing technical chapters around the work performed in a particular role. They are designed to run alongside the subject tracks under `learning/`, not replace them or form a separate prerequisite ladder.

Use a role track when you want to practice joining concepts across model behavior, data, evaluation, infrastructure, delivery, and operations in one job-shaped workflow.

| Track | What is this? | When should I use it? | Where do I start? |
|---|---|---|---|
| AI Engineer | A practice route connecting data, model adaptation, retrieval, evaluation, serving, releases, and feedback into one loop. | You want to learn how technical components become an observable, releasable AI application. | [AI Engineer route](ai-engineer/README.md) |
| Forward Deployed Engineer | A practice route for turning an ambiguous customer workflow into a bounded design, rollout, operating model, and handoff. | You already have or are building the AI Engineer core and want to practice customer-facing delivery decisions. | [FDE route](fde/README.md) |

## First-Time Learner Sequence

For each step, learn the concept first, then use the matching role chapter to practice the decision it
supports. Start with the [AI Engineer baseline](ai-engineer/00-role-baseline-and-route.md); it tells you
which foundations you can challenge out.

| Step | Learn the concept | Then practice it in the role route |
|---:|---|---|
| 1 | [Fine-tuning data techniques](../genai/09-llm-finetuning/01-llm-finetuning-data-techniques.ipynb) | [Training Data Quality and Lineage](ai-engineer/01-training-data-quality-and-lineage/README.md) |
| 2 | [RAG evaluation](../genai/10-rag/02-rag-evaluation.ipynb) and [evaluation metrics](../genai/11-llm-evaluation/01-llm-evaluation-metrics-and-benchmarks.ipynb) | [Prompt Release and Experimentation](ai-engineer/02-prompt-release-and-experimentation/README.md) |
| 3 | [LLM gateway](../genai/12-llm-gateway/01-llm-gateway.ipynb) and [inference systems](../ai-infrastructure/07-inference-systems/inference-systems.ipynb) | [Application Latency and Cost](ai-engineer/03-application-latency-and-cost/README.md) |
| 4 | [Fine-tuning comparison and decision](../genai/09-llm-finetuning/03-llm-finetuning-comparison-and-decision.ipynb) | [Release Registry and Lineage](ai-engineer/04-release-registry-and-lineage/README.md) |
| 5 | [Agent evaluation and observability](../agentic-ai/05-agent-evaluation-and-observability/05-agent-evaluation-and-observability.ipynb) | [Production Feedback and Drift](ai-engineer/05-production-feedback-and-drift/README.md) |

After completing Steps 1-5, choose the [AI Engineer capstone](ai-engineer/06-capstone/README.md)
or continue to the [FDE baseline and engagement lifecycle](fde/00-role-baseline-and-engagement-lifecycle.md)
for discovery-to-handoff practice.

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
