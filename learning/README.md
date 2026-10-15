# Learn AI Systems: Zero to Hero

**Contents:** [one-line map of this directory](CONTENTS.md)

The numbered directories under `learning/` are the active curriculum. Follow them in order when
starting from first principles; enter a later stage only when its prerequisite contracts are
already routine.

The role routes in Stage 12 synthesize Stages 01-11; they do not redefine those stages as optional.
Experienced learners may challenge out of a stage only with the reproducible evidence requested by
the Stage 12 baseline gates.

For original engineering work and reusable data projects, see [projects/](../projects/). For
historical learning artifacts, see [archive/learning-migrations/](../archive/learning-migrations/).
For exploratory third-party course material, see [playground/](../playground/).

## The Route

| Stage | Focus | Measured outcome | Start |
|---:|---|---|---|
| 01 | ML Foundations | Explain optimization, generalization, neural networks, and backpropagation | [Start Stage 01](01-ml-foundations/README.md) |
| 02 | PyTorch Foundations | Implement the reusable tensor, module, data, training, checkpoint, and inference contracts | [Start Stage 02](02-pytorch-foundations/README.md) |
| 03 | Language-Model Foundations | Build sequence-memory, tokenization, embedding, Word2Vec, and encoder-decoder intuition | [Start Stage 03](03-language-model-foundations/README.md) |
| 04 | Transformers | Build attention, position, complete blocks, and language-model architecture families | [Start Stage 04](04-transformers/README.md) |
| 05 | LLM Training | Build a trustworthy token stream and train a resumable base model | [Start Stage 05](05-llm-training/README.md) |
| 06 | LLM Fine-Tuning | Adapt pretrained models with objective-aligned evidence | [Start Stage 06](06-llm-finetuning/README.md) |
| 07 | RAG | Retrieve authorized evidence and localize retrieval versus answer failures | [Start Stage 07](07-rag/README.md) |
| 08 | LLM Evaluation | Measure model behavior, uncertainty, safety, and release readiness | [Start Stage 08](08-llm-evaluation/README.md) |
| 09 | LLM Gateway | Control normalization, routing, resilience, budgets, caching, and telemetry | [Start Stage 09](09-llm-gateway/README.md) |
| 10 | Agentic AI | Build bounded, tool-using, stateful, recoverable agent workflows | [Start Stage 10](10-agentic-ai/README.md) |
| 11 | AI Infrastructure | Understand and optimize hardware, training, inference, and serving systems | [Start Stage 11](11-ai-infrastructure/README.md) |
| 12 | Role-Based Tracks | Synthesize the stack in AI Engineer and Forward Deployed Engineer workflows | [Start Stage 12](12-role-based-tracks/README.md) |

```text
01 ML foundations
-> 02 PyTorch
-> 03 language-model foundations
-> 04 Transformers
-> 05 base-model training
-> 06 adaptation
-> 07 retrieval
-> 08 evaluation
-> 09 request control
-> 10 agents
-> 11 infrastructure
-> 12 role synthesis
```

## Returning-Reader Shortcut

If neural-network mechanics are familiar and only the PyTorch contract is rusty, run
[PyTorch Training Skeleton: Quick Refresher](02-pytorch-foundations/quick-refresher/pytorch-training-skeleton.ipynb).
Its closing readiness check links to the exact Stage 02 chapter for any missing responsibility.
When every check passes, continue at [Stage 03](03-language-model-foundations/README.md).

## Curriculum Boundaries

- The per-notebook pedagogy evaluation is [eval.md](eval.md).
- Numbered directories are required stages in pedagogical order.
- Unnumbered directories inside a stage are shortcuts or optional branches.
- Reusable Riverside House teaching data lives in the
  [Riverside House corpus project](../projects/riverside-house-corpus/README.md).
- Superseded Keras-first and migration artifacts live outside the active curriculum in the
  [learning migration archive](../archive/learning-migrations/README.md).
- The repository-wide authoring standard is [AUTHORING_GUIDE.md](../AUTHORING_GUIDE.md).

## Operational Evidence

Notebook outputs are cleared in source. A cleared notebook is not automatically validated; each
stage README states its evidence boundary. Labels such as `HOLD_LOCAL_RELEASE`,
`HOLD_AZURE_PROMOTION`, and `Live-unvalidated` are deliberate scope gates rather than success-shaped
fallbacks.

Use this evidence ladder across the curriculum:

| Level | What exists | What it supports |
|---:|---|---|
| 1. Implemented source | Inspectable code, configuration, contract, fixture, or planned check | The asset exists; it may not have run |
| 2. Local fixture | A deterministic local or synthetic fixture ran with retained provenance | The named mechanism worked for that fixture |
| 3. Local measured | A named local workload ran on recorded hardware and software | The result applies to that workload and environment |
| 4. Modeled or simulated | A calculation, queue model, or architecture assumption produced a result | Planning and mechanism reasoning, not observed live behavior |
| 5. Customer-validated | An authorized customer role accepted the named artifact and scope | Acceptance within that authority, scope, and date |
| 6. Live-cloud validated | An authorized cloud environment produced retained release-scoped evidence | Behavior in that named environment and window |
| 7. Independently reviewed | A qualified reviewer checked method, provenance, limits, and conclusion | Greater confidence in the bounded claim, not a broader population |

Track-specific schemas may split a level more finely—for example, static validation versus fixture
execution—but they must not promote evidence to a stronger level.

Compact result labels should make the boundary visible:

```text
Mechanics demonstrated
Fixture or measured evidence: <named class>
Capability evidence: available or unavailable
Live behavior: validated or unvalidated
Independent review: present or absent
```

A loss curve, passing test suite, generated sample, or successful simulation can demonstrate a
mechanism without establishing model quality, user outcomes, cloud behavior, or production
readiness.
