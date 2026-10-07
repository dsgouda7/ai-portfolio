# PyTorch-to-GenAI Curriculum Plan

**Status:** Implemented on 2026-10-06. This document records the migration decisions and
acceptance criteria used by the resulting curriculum.

## Second-Pass Decision

The external PyTorch collection should not be appended as ten more prerequisite notebooks. Its
best material should **replace weaker or duplicated implementations**, while the strongest current
visuals, complaint chains, animations, and measured experiments remain.

Create `learning/pytorch-for-llms/` as the canonical code-first spine, but build it from both
sources:

- use the external notebooks as the primary source for concise PyTorch explanations and complete
  training lifecycles;
- merge in the current repository's stronger shape intuition, failure-first explanations,
  animations, and LLM-specific evidence;
- retire duplicated Keras-first and framework-translation notebooks from the required route only
  after their replacements execute successfully and all links have moved;
- preserve optional revision branches for vision, Lightning, representation history, and
  end-to-end Transformer code reading.

This recommendation follows a source-level second pass over all 40 relevant notebooks:

- 10 external PyTorch notebooks;
- 11 current GenAI prerequisite notebooks;
- 8 Transformer notebooks;
- 11 fine-tuning, RAG, evaluation, and gateway notebooks;
- adjacent Transformer and applied-GenAI theory notes and READMEs.

## Final Learning Path

```text
FOUNDATION INTUITION
math -> ML contracts -> neural networks/backprop

PYTORCH-FOR-LLMS SPINE
deep-learning map
-> tensors/autograd
-> models/training loops
-> data/checkpoints/hyperparameters
-> modern training lifecycle
-> sequence memory
-> tokenization/embeddings

GENAI
Transformer mechanisms
-> decoder training and pretraining
-> fine-tuning
-> RAG
-> evaluation
-> gateways
```

Optional branches must not interrupt the required path:

```text
vision: CNNs -> autoencoders -> optional VAE
representation history: one-hot/BoW -> Word2Vec -> GloVe
tooling: native PyTorch -> Lightning
revision: full Transformer code-reading capstone
```

## Learning Artifact Contract

Use three possible artifacts, but do not create all three mechanically.

### 1. Theory notebook

- 30-45 minutes.
- Opens with a visible failure or falsifiable question.
- Uses static evidence, animations when state changes over time, and small measured code.
- Teaches why the mechanism exists and what it does not solve.
- Ends with the unresolved implementation question that opens the lab.

### 2. PyTorch code lab

- 45-75 minutes.
- Builds one artifact corresponding to the theory notebook.
- Makes shapes, dtype, device, state, gradients, modes, masks, and checkpoint boundaries explicit.
- Introduces visible tensor operations before framework helpers.
- Includes assertions, expected outputs, one-variable exercises, and a toy-to-real bridge.
- Targets no more than 30 lines per teaching cell. Longer cells require an immediate walkthrough
  and should rarely exceed 60 lines.

### 3. Handwritten theory note

- Existing `-theory.md` files remain concise revision companions.
- They preserve the complaint, mental model, measured result, and boundary.
- They do not duplicate the full notebook or substitute for executable evidence.

Short, balanced mechanism notebooks should stay integrated. Separate theory and code only when
conceptual and implementation load compete for the learner's attention.

## Canonical Ownership and Condensation Rules

1. One concept has one canonical owner.
2. Later chapters use a short diagnostic recap and link back instead of reteaching it.
3. PyTorch is the only required implementation framework.
4. Keras comparisons are optional translation sidebars, not parallel implementations.
5. Keep consequence experiments before extra derivation or API inventories.
6. Keep current animations that prove a mechanism; remove repeated roadmap and recap prose.
7. Teach a helper's mechanics once before extracting it into reusable track infrastructure.
8. Do not copy long external code cells unchanged; decompose them around visible state transitions.
9. Replacement notebooks must pass execution and link validation before old required notebooks are
   retired.
10. Optional branches cannot be prerequisites for the default LLM route.

## Proposed Directory Shape

```text
learning/
├── genai-prerequisites/
│   ├── 00-math-foundations/
│   ├── 01-ml-basics/
│   ├── 02-neural-networks/
│   └── optional-vision/
├── pytorch-for-llms/
│   ├── 00-deep-learning-map/
│   ├── 01-tensors-and-autograd/
│   ├── 02-models-and-training-loops/
│   ├── 03-data-checkpoints-and-hyperparameters/
│   ├── 04-modern-training-lifecycle/
│   ├── 05-sequence-memory/
│   ├── 06-tokenization-and-embeddings/
│   └── optional/
│       ├── embedding-history/
│       ├── pytorch-lightning/
│       └── transformer-code-reading/
└── genai/
    ├── 01-transformers/
    ├── 02-llm-finetuning/
    ├── 03-rag/
    ├── 04-llm-evaluation/
    └── 05-llm-gateway/
```

## External Notebook Disposition

| External notebook | Decision | Replacement or final home |
|---|---|---|
| `010_DeepLearningIntro.ipynb` | Retain and lightly condense | Required PyTorch Part 00 |
| `020_TensorIntro.ipynb` | Merge with current shape-first material | Required PyTorch Part 01 |
| `030_ModelingExercise.ipynb` | Split and decompose long cells | Required PyTorch Parts 02 and 03 |
| `180_LSTM_FunctionApproximation.ipynb` | Merge with current RNN theory and bridge | Sequence-memory code lab |
| `200_Autoencoders.ipynb` | Retain as optional | Optional vision branch after CNNs |
| `300_TinyTextTransformer.ipynb` | Mine for focused evidence; retire as one core lesson | Transformer Parts 3-5 code labs |
| `300_TextTransformerForAIEngineers.ipynb` | Retain only as optional revision capstone | Post-Transformer engineering bridge |
| `310_ModernPyTorchTraining.ipynb` | Retain, decompose, and make canonical | Required PyTorch Part 04 |
| `310_TextTransformerWithLightning.ipynb` | Retain as optional translation | After manual base-model training |
| `360_EmbeddingEvolution.ipynb` | Split by owner | Small core extraction plus optional history; semantic search moves to RAG |

Do not copy the external `references/` archive by default. Review provenance and redistribution
rights for PDFs, source scripts, datasets, and assets before migration.

## Foundation and PyTorch Replacement Plan

### Foundation 00 - Mathematical intuition

**Current owner:** `genai-prerequisites/00-math-foundations/`

Retain:

- free-kick narrative;
- vectors and coordinates;
- derivative and integration intuition;
- matrix-as-state-machine bridge;
- repeated local updates and uncertainty.

Condense:

- tensor/API detail later owned by the PyTorch spine;
- repeated summaries that do not add a new measured result.

Boundary: no detailed framework instruction.

### Foundation 01 - ML contracts

**Current owner:** `genai-prerequisites/01-ml-basics/`, condensed.

Retain:

- train/validation/test isolation;
- regression, classification, and threshold meaning;
- loss versus metric;
- optimization and learning-rate intuition;
- overfitting, regularization, and generalization.

Replace or remove:

- repeated complete training implementations;
- repeated TensorFlow/PyTorch translations;
- broad parameter sweeps when one controlled comparison proves the claim.

Use only one compact implementation and point to the PyTorch spine for mechanics.

### Foundation 02 - Neural networks and backpropagation

**Current owner:** `02-neural-networks/01-smartval-neural-networks-and-backprop.ipynb`,
condensed into a theory-first chapter.

Retain:

- XOR failure;
- hidden representation and ReLU;
- responsibility reports and gradient verification;
- chain-rule intuition;
- concise depth, width, dropout, and normalization boundaries.

Remove:

- another full training-loop tutorial;
- duplicate autograd instruction;
- equal-weight treatment of advanced extensions in the required introduction.

Move `02-melodyne-backprop-synthesizer.ipynb` to an optional audible capstone. Preserve its
differentiable rendering, four-gradient audit, recovery gates, and audible evidence.

### PyTorch 00 - Deep-learning map

**Primary source:** external `010`.

Keep its short progression:

```text
data -> module -> forward -> loss -> backward -> optimizer -> epoch -> inference
```

Use one tiny learning problem and retain the explicit LLM connection. Do not add DataLoader,
checkpoint, or production abstractions yet.

### PyTorch 01 - Tensors and autograd

**Primary source:** external `020`.

Merge from the current arrays notebook:

- axis naming;
- table and sequence shape contracts;
- visible shape failures;
- reshape, transpose, broadcasting, and matrix-operation intuition.

Retain from external `020`:

- rank, shape, dtype, and device;
- indexing and broadcasting;
- element-wise versus matrix multiplication;
- computation graphs, chain rule, gradient accumulation, detach, and inference mode;
- language-model-shaped tensor example.

Retire `00-arrays-to-model-tensors.ipynb` as a standalone required notebook after these pieces
move. Move its sequence-window cells to Parts 05 and 06.

### PyTorch 02 - Models and training loops

**Primary source:** first half of external `030`.

Teach:

- regression from visible tensor math;
- conversion to `nn.Module`;
- parameters and `forward`;
- loss shapes;
- optimizer state and gradient clearing;
- equivalence of the raw and module implementations.

Split the current 86-line class/training cell into model, forward-contract, loss, update, and
comparison cells.

### PyTorch 03 - Data, checkpoints, and hyperparameters

**Primary source:** second half of external `030`.

Teach:

- manual batches before `Dataset` and `DataLoader`;
- shuffle and batch contracts;
- state dictionaries;
- save/load parity;
- model state versus experiment configuration;
- one-variable hyperparameter comparisons.

Remove duplicate linear-regression and hyperparameter implementations elsewhere.

### PyTorch 04 - Modern training lifecycle

**Primary source:** external `310_ModernPyTorchTraining`.

Retain:

- split-before-normalization discipline;
- train versus validation loops;
- `train()`, `eval()`, and inference mode;
- scheduler and mixed-precision boundaries;
- early stopping;
- best-model versus resumable checkpoints;
- reload and inference in original units.

Decompose the 78-112-line orchestration cells. Keep Lightning mapping and `torch.compile` as an
optional appendix.

Move `01-keras-to-pytorch-antarctic-field-guide.ipynb` out of the required route. Preserve one
optional translation reference containing Keras/PyTorch vocabulary, `fit()` versus explicit-loop
comparison, and evaluation-mode differences.

## Sequence, Tokenization, and Optional Vision

### PyTorch 05A - Sequence-memory theory

Build from the current RNN notebook, not from the external LSTM lab.

Must retain:

- sequence challenge and `(batch, time, feature)`;
- manual vanilla recurrence;
- BPTT and the vanishing-gradient experiment;
- gradient-decay animation;
- LSTM gate/cell-state visual and animation;
- one controlled RNN-versus-LSTM comparison.

Replace the Keras implementation cells with small framework-neutral evidence and links to the
PyTorch lab. Retire the Keras-first notebook only after this theory notebook exists.

### PyTorch 05B - Sequence-memory PyTorch lab

Merge external `180_LSTM_FunctionApproximation` with the strongest sections of the current
PyTorch RNN bridge.

Use external `180` for:

- low-noise forecasting problem;
- sliding windows;
- temporal split;
- `Dataset`/`DataLoader`;
- first `nn.LSTM` model;
- prediction evaluation.

Then extend with the current bridge's:

- token IDs and shifted targets;
- embedding lookup;
- sequence output versus final-state shapes;
- packed sequences;
- vocabulary logits;
- padding-aware cross-entropy;
- causal-invariance proof;
- gradient clipping.

This merged lab replaces the executable Keras sections and the standalone bridge notebook.

### PyTorch 05C - Cinematic piano memory capstone

Keep `02-cinematic-piano-memory.ipynb` as the required sequence capstone.

Preserve:

- original music task;
- audible and visual payoff;
- measured memory-path comparison;
- checkpointed practice;
- one mandatory learner-controlled parameter comparison.

Reuse the canonical sequence lab's model and training helpers instead of reteaching their APIs.

### PyTorch 06A/06B - Tokenization and embeddings

Split the current tokenization notebook into theory and PyTorch lab companions.

Theory must retain:

- why tokenization exists;
- BPE merge animation and compression evidence;
- production tokenizer bridge;
- token IDs as addresses;
- static lookup versus contextual representation boundary;
- padding and masking failure.

Code lab must retain or replace:

- BPE helpers, shortened and walked through;
- GPT-2/tiktoken inspection;
- controlled embedding-row learning rewritten with `torch.nn.Embedding`;
- padded-batch and mask visualization;
- wrong-versus-correct masking experiment;
- loss masking rewritten with `ignore_index` and explicit reduction.

Remove repeated framework comparisons and broad contextual-embedding exposition.

### Optional vision branch

Move the current CNN content out of the required LLM route and rewrite its executable cells in
PyTorch. Preserve convolution, Sobel filters, stride/pooling, feature maps, receptive field,
trained-filter specialization, and the ResNet gradient proof.

Follow it with external `200_Autoencoders` for encoder/decoder, reconstruction, latent-space, and
compression evidence. Keep VAE material as reading only unless a real VAE lab is implemented.

### Optional representation-history branch

Extract from external `360_EmbeddingEvolution`:

- one-hot and bag-of-words;
- Word2Vec;
- GloVe;
- analogies and static geometry;
- static versus contextual boundary.

Remove sentiment classification and generic pipeline demos from this branch. Move semantic search
to retrieval/RAG, where it has a clear job.

## Transformer Integration Plan

The current eight-part Transformer sequence remains the canonical conceptual spine. The external
300-series notebooks become implementation donors, not competing chapters.

### Parts 01-03 - Keep integrated

- Part 1 remains the Transformer-specific handoff from the prerequisite tokenization chapter.
  Reduce its recap after PyTorch Part 06 is complete.
- Part 2 remains the mechanistic gold standard. Preserve failed RoPE attempts, animations, and
  measured position/scaling evidence.
- Part 3 remains the complete block. Import only useful implementation details and shape checks
  from external `300_TinyTextTransformer`.

### Part 04 - Decoder-only theory and code companions

Split because the current notebook combines a manageable conceptual story with more than 1,000
lines of code.

Theory owns:

- causal objective and mask;
- shifted labels;
- parallel teacher-forced training versus serial generation;
- temperature/top-k;
- KV-cache intuition.

Code lab owns:

- explicit input/target alignment;
- attention mask versus padding mask versus ignored loss target;
- external notebook's canonical loss-flattening demonstration;
- greedy and sampled generation;
- stop-token and length-limit behavior;
- recomputation/cache parity where practical.

Do not import full raw-corpus preparation.

### Part 05 - Encoder-decoder theory and code companions

Keep one Part 05 ownership boundary but replace the current oversized single notebook with:

1. theory: source memory, target writer, decoder self-attention, cross-attention, teacher forcing,
   exposure bias, and cache roles;
2. code lab: reversal routing task, mean-pooling bottleneck measurement, `CrossAttention`,
   source/target length mismatch, attention-routing prediction, and teacher-forced versus
   free-running evaluation.

Mine these external `300_TinyTextTransformer` sections rather than retaining that notebook:

- positional-identity/mean-pooling bottleneck;
- `CrossAttention` implementation;
- decoder-query and encoder-key/value shapes;
- routing visualization.

### Part 06 - Keep integrated

Keep RMSNorm, RoPE in context, GQA, SwiGLU, pre-normalization, weight tying, and cache economics.
Consume earlier contracts instead of reteaching basic attention or causal masking.

### Part 07 - Keep integrated

Keep document splits, duplicates, tokenizer-fit boundary, EOD, packing, shards, manifests, hashes,
and reload/decode verification.

Explicitly distinguish:

- sentence `Dataset`/`DataLoader`/dynamic collation from the external labs;
- fixed-length packed pretraining streams and shard manifests owned here.

### Part 08 - Pretraining theory and code companions

Theory owns:

- forward/loss/update lifecycle;
- validation evidence;
- best-model and resume checkpoint semantics;
- artifact lineage;
- limits of a tiny CPU-safe run.

Code lab owns:

- manual training and validation loops;
- aligned logits and targets;
- generation sample;
- checkpoint save/reload/resume;
- tokenizer/config/manifest linkage;
- measured loss and validation result.

Make this the canonical manual training and checkpoint implementation.

### Optional Transformer extensions

- `09-training-a-base-model-with-lightning.ipynb`: translate the understood manual lifecycle into
  `training_step`, `validation_step`, `configure_optimizers`, `Trainer`, and checkpoint callbacks.
  Require shape, loss, and checkpoint parity.
- `10-tiny-transformer-for-ai-engineers.ipynb`: optional systems-oriented revision covering data
  contract, model contract, training evidence, evaluation, artifacts, and limitations.

## Applied GenAI Condensation Plan

### Fine-tuning - retain four notebooks, do not double them

The existing order is correct. Condense rather than create theory/code pairs for every topic.

1. **What Should the Model Learn?** - 75-90 minutes after condensation.
   - Keep objective-selection flow, attention mask versus loss mask, prompt/response/EOS boundary,
     DPO pair invariants, and a short PPO comparison.
   - Keep visible token IDs, `-100` labels, EOS behavior, template boundaries, DPO log-probability
     movement, and malformed-example validation.
   - Remove repeated setup, broad theory surveys, and duplicate provenance lectures.

2. **Where Should the Update Live?** - 60-75 minutes.
   - Keep full tuning, freezing, LoRA, QLoRA, writable-surface comparison, gradient inspection,
     adapter reload parity, and artifact-size evidence.
   - Show the low-rank path manually before PEFT helpers.
   - Do not claim quality from structural measurements.

3. **Evaluation, Comparison, and Decision** - about 60 minutes.
   - Keep candidate isolation, contaminated-versus-valid evidence, gates, cost/rollback comparison,
     lineage, and an executable decision record.
   - Remove metric tutorials owned by the evaluation track.

4. **GPU Practice** - 90-120 minute staged CUDA lab.
   - Stage one novel end to end before all eight.
   - Preserve chronological splits, contamination checks, validation selection, test isolation,
     reload parity, paired bootstrap, retention canary, PASS/FAIL/INCONCLUSIVE gates, manifests,
     adapters, and ledgers.
   - No quick path may bypass test isolation.

### RAG - split only the overloaded retrieval notebook

Final set:

1. `01a-hybrid-retrieval-theory.ipynb` - 35-45 minutes.
2. `01b-hybrid-retrieval-lab.ipynb` - 50-70 minutes.
3. `02-rag-failure-localization-and-evaluation.ipynb` - 75-90 minutes.

Hybrid theory retains lexical-versus-dense failures, overlap, RRF rank movement, Recall@K/MRR,
candidate-depth ceilings, authorization-before-scoring, and support-gate boundaries.

Hybrid lab builds BM25/dense retrieval, RRF/weighted fusion, alpha/k sweeps, reranking,
ACL-before-scoring, abstention, degraded-mode fallback, telemetry, and index manifests.

RAG evaluation narrows to retriever-versus-generator diagnosis, groundedness versus correctness,
oracle-context ablation, metric fingerprints, per-query records, citations, refusal, authorization,
and safety gates. Move judge bias, general hallucination taxonomies, and calibration detail to the
evaluation track.

### Evaluation - retain four distinct owners

Do not create four new companion labs. Tighten the current boundaries:

1. metrics and benchmarks: what reference, semantic, and model-based metrics can observe;
2. LLM-as-judge and safety: how a judge and evaluation pipeline can be trusted;
3. hallucination detection: how unsupported claims are localized;
4. calibration and confidence: how confidence becomes abstention policy.

Centralize shared fixtures, deterministic setup, display helpers, and generic caveats. Do not
repeat "no single metric is enough" or the full evaluator-bias taxonomy in every notebook.

### Gateway - split the overloaded single notebook

Final set:

1. **Gateway Control-Plane Theory** - 45-60 minutes.
2. **Gateway Routing and Resilience Lab** - 75-90 minutes.

Theory owns the gateway-versus-inference-server boundary, normalized contract, request lifecycle,
eligibility before routing, retry versus fallback, cache/limit/budget ordering, and failure-chain
animations.

Lab owns deterministic providers, adapters, routing, least-busy behavior, rate/concurrency/token
limits, budget reserve/settle, exact cache and unsafe semantic-cache evidence, retry/fallback,
telemetry, and cost/latency reporting.

Keep circuit breakers and backpressure as forward links unless they receive measured
implementations. Continuous batching, KV cache management, GPU scheduling, and serving
backpressure remain in the infrastructure track.

## Shared Setup and Helper Policy

The applied track repeats Riverside/Aria introductions, root discovery, device selection, seeds,
model loading, display helpers, artifact directories, and generic evidence disclaimers.

Condense these into:

- one short course-contract cell per track;
- a small tested setup module for non-teaching boilerplate;
- explicit links back to the first explanation.

Helpers may be extracted only after the learner implements the operation once. Good later helpers
include:

- masked-label construction and schema validation;
- parameter counting and frozen-weight verification;
- candidate loading, gate evaluation, and decision records;
- RRF, authorization filters, and retrieval telemetry;
- per-query RAG records and oracle-context runners;
- metric fixtures and judge result schemas;
- provider adapters, routing policies, budget accounting, cache keys, and telemetry.

Do not begin a lab with an opaque end-to-end helper such as `train_everything()`,
`HybridRetriever`, or `Gateway.handle()`.

## Code-Explanation Standard

Every substantial PyTorch implementation should make these ten items visible:

1. input and output shapes;
2. data versus learned state;
3. what `forward` does and does not do;
4. why the loss accepts those shapes and dtypes;
5. where gradients are created, accumulated, cleared, clipped, or disabled;
6. when training, evaluation, and inference modes change behavior;
7. device movement and metadata boundaries;
8. checkpoint contents and missing resume state;
9. teaching implementation versus production abstraction;
10. one printed or plotted check proving the claim.

Prefer short code followed by output interpretation. For complete classes, first introduce the
pieces, then assemble them, then add a shape trace and block-by-block walkthrough.

## Migration Sequence

### Phase 0 - Provenance and dependency map

- Record notebook inputs, outputs, runtime, kernels, artifacts, and inbound links.
- Review external data, assets, source scripts, and PDFs for redistribution.
- Mark every source section as retain, merge, replace, optional, or retire.

### Phase 1 - Replace the foundation implementation path

- Create PyTorch Parts 00-04 from external 010/020/030/310 plus current shape evidence.
- Condense math, ML, and backprop chapters around their conceptual owners.
- Validate replacements before removing current PyTorch-fundamentals notebooks from the required
  route.

### Phase 2 - Replace sequence and tokenization implementations

- Build sequence theory, merged PyTorch lab, and piano capstone.
- Build tokenization theory and PyTorch lab.
- Move CNN/autoencoder and representation history into optional branches.
- Retire Keras-first RNN/tokenization implementations and the standalone bridge only after parity
  checks pass.

### Phase 3 - Integrate external Transformer evidence

- Mine external `300_TinyTextTransformer` into Parts 3-5.
- Split Parts 4, 5, and 8 into theory/code companions.
- Add optional Lightning and AI-engineer revision notebooks after the core route.
- Do not retain a second full Transformer curriculum.

### Phase 4 - Condense applied GenAI

- Condense fine-tuning without doubling its notebook count.
- Split Hybrid Search and Gateway.
- Narrow RAG Evaluation and the four evaluation notebooks to their canonical questions.
- Extract shared setup only after teaching operations remain visible.

### Phase 5 - Route validation

- Execute every notebook from a fresh kernel into temporary copies.
- Verify no error outputs, stable sources, unique cell IDs, local links, image paths, and artifact
  contracts.
- Run entry/exit diagnostics at PyTorch, sequence/tokenization, Transformer, and applied-GenAI
  boundaries.
- Measure representative completion times with a learner and revise split decisions from observed
  load, not file size alone.

## Acceptance Criteria

- The required route is PyTorch-first and contains no mandatory Keras implementation.
- External material replaces or improves weaker content instead of appearing as duplicate lessons.
- Every retired notebook has a validated replacement and updated inbound links.
- The learner enters Transformers already fluent in tensors, autograd, modules, DataLoader,
  train/eval modes, validation, checkpoints, sequence shapes, tokenization, embeddings, and masks.
- Current complaint chains, animations, and consequence experiments survive migration.
- No required concept is fully taught twice.
- Parts 1-3 and 6-7 of Transformers remain focused; overloaded Parts 4, 5, and 8 become companions.
- Fine-tuning remains four notebooks; RAG becomes three; evaluation remains four; gateway becomes
  two.
- Theory notebooks remain useful without JavaScript animation support.
- Code labs remain understandable without rereading the complete theory notebook.
- Optional vision, history, Lightning, and code-reading routes remain available for revision.
- Existing project, role-based, and infrastructure links continue to resolve.

## Non-Goals

- Do not rewrite every notebook at once.
- Do not split every lesson by policy.
- Do not preserve a notebook merely because it already exists; preserve its best evidence in the
  correct owner chapter.
- Do not copy long external code cells unchanged.
- Do not move third-party references without confirmed redistribution rights.
- Do not turn the prerequisite route into a survey of every PyTorch feature or trainer framework.
