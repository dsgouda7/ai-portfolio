# Linked Page Retrieval Lab: Implementation Plan

## Current state

The project is documentation-only. The hypotheses, proposed architecture, benchmark strata, metrics,
and decision gates are defined in [README.md](README.md). No source code, test fixture, index, or
benchmark result exists yet.

## Scope ledger

| Tier | Capability |
|---|---|
| Planned to build and measure | Logical-page construction, FTS5, typed graph, PageRank, local vector baseline, linked-page retrieval, selective and full hybrid retrieval, incremental updates, deterministic retrieval evaluation, local CPU generation, and Microsoft Foundry generation |
| Planned to explain | Why authority differs from relevance, when lexical retrieval is sufficient, how graph expansion helps multi-page questions, and where semantic retrieval remains necessary |
| Named and out of scope | Distributed crawling, managed databases, cloud cost modeling, automatic LLM link extraction, production security, high availability, and web-scale performance |

## Evaluation-first test matrix

The implementation order starts with the evidence required to accept or reject each part of the
value proposition.

| ID | Test case | Primary metric | Public benchmark or proxy | Custom coverage still required |
|---|---|---|---|---|
| T01 | Exact identifier and terminology retrieval | Recall@10, MRR@10 | [BEIR](https://github.com/beir-cellar/beir), especially SciFact and NFCorpus | Version and link metadata |
| T02 | Paraphrased or indirect query | Recall@10, nDCG@10 | BEIR heterogeneous retrieval tasks | Selective-embedding policy |
| T03 | Evidence split across documents | Complete Evidence Rate@10 | [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) and [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) | Typed-edge ablations |
| T04 | Two relevant pages, one canonical | Canonical@1, authority-weighted nDCG@10 | [KILT](https://huggingface.co/datasets/facebook/kilt_tasks) provenance is a partial proxy | Independent authority labels and natural link graph |
| T05 | Current page competing with a superseded page | Current-over-stale win rate, stale exposure@10 | [FreshQA](https://github.com/freshllms/freshqa) is an answer-freshness proxy | Versioned corpus and supersession edges |
| T06 | Unanswerable query | False-positive rate, abstention accuracy | [RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench) labeled RAG examples | Retrieval-threshold behavior |
| T07 | Expansion adds adjacent but irrelevant pages | Context precision@10, edge budget violations | No standard benchmark isolates this | Adversarial graph neighborhoods |
| T08 | One source edit fans out into derived writes | Update amplification, vectors recomputed | No standard benchmark | Fixed add/edit/supersede/delete manifest |
| T09 | Retrieved evidence produces a grounded answer | Required-fact coverage, groundedness, citation precision | RAGBench and [Foundry built-in evaluators](https://learn.microsoft.com/azure/foundry/concepts/built-in-evaluators) | Exact page-ID citation checks |
| T10 | Performance changes with corpus size | p50/p95 latency, index bytes, peak RSS | BEIR supplies heterogeneous corpora, not this scaling contract | Fixed 200-, 2,000-, and 20,000-page tiers |
| T11 | Links are missing, noisy, reversed, or manipulated by a hub | Canonical@1 delta, rank correlation, PageRank lift | No standard RAG benchmark isolates graph trust | Deterministic edge-corruption manifests |
| T12 | Synthetic gains transfer to naturally linked documents | Recall@10, Canonical@1, Complete Evidence Rate@10 | Repository Markdown links provide a local transfer corpus | Small independently judged repository-document set |

### Benchmark use rules

- Public benchmark adapters must pin dataset repository, configuration, revision, split, license,
  and content hash.
- Use a stratified, documented subset for local CPU smoke runs and a larger frozen subset for the
  release benchmark.
- Keep each benchmark's native evidence units. Do not create graph links for a dataset that does
  not contain defensible natural relationships.
- Public benchmark results test retrieval or generation generality. Only the custom versioned graph
  can decide whether PageRank improves canonical authority and whether updates become cheaper.
- Published scores are reference metadata, not directly comparable unless corpus version,
  preprocessing, model, split, and metric implementation match exactly.
- The repository transfer set uses only explicit Markdown links and headings. It does not invent
  semantic edges to make PageRank look useful.

## Baseline ladder

### Retriever-only baselines

These run before any LLM is loaded:

1. random ranking and PageRank-only negative controls;
2. BM25/FTS5;
3. fixed overlapping chunks plus exact dense retrieval, the plain-RAG correctness baseline;
4. fixed overlapping chunks plus a CPU HNSW index, the plain-RAG operational baseline;
5. logical pages plus dense retrieval, isolating segmentation;
6. logical-page BM25 plus dense reciprocal-rank fusion;
7. the same non-graph hybrid plus a pinned CPU cross-encoder reranker;
8. the same candidate and reranker configuration plus PageRank only;
9. the same candidate and reranker configuration plus typed expansion only;
10. the full graph hybrid with expansion and PageRank;
11. the selective graph hybrid;
12. oracle evidence, used only as an upper bound.

Exact cosine remains the retrieval oracle. HNSW prevents a brute-force vector scan from becoming a
straw-man latency baseline. The final graph and non-graph contenders must also run with the same
cross-encoder reranker so any PageRank lift is measured beyond a conventional relevance reranker.

BEIR-style runs report nDCG@10, MAP@100, Recall@100, and precision at the benchmark's standard
cutoffs in addition to this project's Recall@10 and latency metrics. HotpotQA and MultiHop-RAG runs
also report supporting-document recall and Complete Evidence Rate.

### Generator baselines

Every generator model is evaluated against the same five conditions:

1. **closed book:** question only, measuring model-memory contamination;
2. **oracle context:** gold evidence, measuring the generator ceiling;
3. **plain vector RAG:** fixed chunks from the conventional baseline;
4. **page hybrid:** strongest non-graph evidence;
5. **full and selective graph hybrid:** proposed evidence.

The model, system prompt, decoding settings, evidence-token budget, and output schema remain fixed
across conditions. A model-card benchmark such as MMLU is recorded as model provenance but is not a
substitute for these same-model RAG baselines.

Every ML component receives its own immutable manifest: embedding model, cross-encoder reranker,
generator, and judge. A result cannot be grouped with another run when any component revision,
tokenizer, preprocessing rule, quantization, or prompt hash differs.

## How an LLM can affect the metrics

| Metric family | Can the generator LLM change it? | Treatment |
|---|---|---|
| Recall, MRR, nDCG, Canonical@1 | No, when retrieval is deterministic | Compute before generation and treat as the PageRank source of truth |
| Complete Evidence Rate and context precision | No, when expansion is deterministic | Compute before generation |
| Answer correctness and required-fact coverage | Yes | Compare closed-book, retrieved, and oracle-context runs with the same model |
| Groundedness and citation precision | Yes | Require page-ID citations and deterministic support checks before any LLM judge |
| Abstention accuracy | Yes | Score explicit abstention against answerability labels |
| Tokens, answer latency, and cost | Yes | Record per request and aggregate by strategy |
| LLM-judge scores | Yes, through judge bias and version drift | Pin a separate judge deployment, record its metadata, and calibrate against labeled examples |

LLM query rewriting, link generation, or reranking is disabled in the core benchmark because it
would confound PageRank. A later `llm-assisted-retrieval` arm may enable one feature at a time; its
retrieval metrics, tokens, latency, and cost must be reported separately.

## Two evaluation execution profiles

Both profiles consume the same frozen dataset manifests, retriever outputs, prompts, answer schema,
and metric implementation. Retriever-only metrics must be byte-for-byte identical between profiles.
Generation is repeated on a fixed stability subset so non-determinism is measured rather than
assumed away by `temperature=0`.

### Profile A: local CPU with Phi-4 Mini

Use the official
[microsoft/Phi-4-mini-instruct-onnx](https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx)
CPU INT4 artifact with ONNX Runtime GenAI:

- model: `microsoft/Phi-4-mini-instruct`, 3.8B parameters;
- optimized artifact: `cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4`;
- license: MIT;
- runtime: pinned `onnxruntime-genai`;
- execution provider: CPU, concurrency 1;
- generation: temperature 0, fixed seed where supported, fixed maximum output tokens;
- evidence budget: a conservative fixed token budget shared with the cloud profile, not the
  model's entire 128K context window.

Before running, the local adapter writes `model-manifest.json` containing:

- Hugging Face repository and immutable revision;
- base model ID, parameter count, architecture, and declared context length;
- artifact-relative path, quantization, and SHA-256 hashes;
- tokenizer and chat-template hashes;
- license;
- runtime and dependency versions;
- configured evidence/output limits;
- CPU, RAM, operating system, thread count, and execution provider.

The local profile never uses the candidate model as an LLM judge. Primary retrieval metrics and
page-ID citation checks are deterministic. RAGBench labels provide an external check for adherence,
relevance, utilization, and completeness behavior.

### Profile B: Microsoft Foundry deployment

Inputs are the project endpoint, Foundry account resource ID, candidate deployment name, and an
optional separate judge deployment. Authentication uses developer identity or managed identity;
credentials are never written to configuration or result artifacts.

#### Mandatory metadata-first preflight

The pipeline performs no inference until it:

1. resolves the deployment through the Foundry account and records every match;
2. fails on missing, ambiguous, or non-succeeded deployments;
3. resolves the catalog model identity and immutable version behind the deployment;
4. retrieves model details, capabilities, context limits, deployment kind (`ADM` or `Managed`),
   region/SKU information when available, and API surface;
5. retrieves Foundry model benchmark metadata and stores it as descriptive provenance;
6. discovers the available evaluator catalog rather than assuming a preview evaluator exists;
7. writes and hashes a normalized `model-manifest.json`;
8. selects the same-model generator baseline ladder from that manifest.

The current Foundry command surface exposes `model_deployment_get`, `model_details_get`,
`model_benchmark_get`, evaluation suites, and dataset batch evaluations. The implementation may use
the supported SDK or equivalent API, but it must preserve the metadata contract above and must not
infer the model from the deployment name.

Model-relative baseline selection follows these rules:

- always run closed-book, oracle-context, plain-vector, page-hybrid, and graph-hybrid conditions
  with the same candidate deployment;
- cap every strategy to the smallest configured evidence budget supported by all compared models;
- compare a deployment only with results that have the same resolved model name and version unless
  the report explicitly labels a cross-model comparison;
- retain model-catalog benchmark scores for context, but never use unrelated general benchmarks as
  RAG acceptance gates;
- use a distinct judge deployment for model-based evaluators and persist the judge's full manifest.

The Foundry runner sends JSONL rows containing `query`, `response`, `context`, `ground_truth`, page
IDs, required facts, answerability, and strategy metadata. It runs local deterministic metrics
first, then requests available Foundry evaluators such as relevance and groundedness. Retrieval,
document-retrieval, or response-completeness evaluators are enabled only after catalog discovery;
missing evaluators fail that requested evaluation instead of being silently replaced.

Each cloud result also records input/output tokens, request latency, retries, throttling, content
filter outcomes, evaluator identity, candidate deployment identity, and estimated or reported cost.

Before the first request, the Foundry runner produces a dry-run estimate containing rows, candidate
and judge calls, maximum input/output tokens, concurrency, and a configured spending cap. It
requires explicit execution approval outside automated CI when the estimate exceeds that cap. Runs
are resumable and idempotent: response-cache keys include dataset, strategy, evidence, prompt,
deployment, model-version, and decoding hashes.

The runner resolves candidate, judge, and evaluator metadata again after the final request. Any
identity or version drift marks the run invalid instead of merging its results. By default, only the
fictional corpus and public benchmark rows may leave the workstation; a repository or private
transfer corpus requires a separate explicit data-export approval.

#### Tooling preflight discovered during planning

On 2026-10-02 the repository environment had `azd` 1.23.15, which could not install the
`microsoft.foundry` extension; the dependency check reported 1.35.0 as the current stable release.
Before implementing or running the Foundry profile, upgrade `azd` to a compatible supported version,
rerun the dependency check, and pin the working tool versions in the run manifest.

## Definition of done

The POC is complete when a fresh local checkout can:

1. create the deterministic source corpus and locked query suite;
2. download or load pinned public benchmark subsets and verify their hashes;
3. build the plain-RAG baselines and every paired PageRank, expansion, and selective-indexing
   ablation from the same sources;
4. apply the same fixed update manifest to every applicable index;
5. run retrieval, quality, and efficiency evaluations without a cloud API;
6. run the local CPU generator profile and produce its model manifest;
7. expose a Foundry profile that refuses to run until deployment metadata and evaluator
   availability are resolved;
8. write machine-readable metrics plus a Markdown report containing dataset, retriever, model,
   prompt, evaluator, environment, and deployment provenance;
9. reproduce retrieval metrics exactly and latency metrics within a documented tolerance;
10. execute edge-corruption sensitivity tests and the naturally linked transfer set;
11. state whether each preregistered decision gate passed or failed without manually editing the
    report.

The local profile must be executed before the POC can claim local validation. The Foundry profile is
**live-unvalidated** until it completes against an authorized project and retains the remote run
identifiers and normalized manifests.

## Phase 1: Scaffold contracts and CLI

### Deliverables

- `pyproject.toml` with pinned-compatible runtime and development dependencies;
- `src/linked_page_retrieval/` package;
- command entry points for `build`, `evaluate`, and `report`;
- typed models for sources, pages, links, queries, judgments, model manifests, evaluator manifests,
  runs, and metrics;
- provider interfaces for `retrieval-only`, `local-onnx`, and `foundry`;
- immutable component manifests for embedder, ANN index, reranker, generator, and judge;
- resumable run ledger and content-addressed response cache;
- repository-local paths that never write generated artifacts into source fixture directories.

### Contracts

- IDs are stable across repeated builds.
- Dates are timezone-aware ISO 8601 values.
- Every generation run references immutable dataset, retriever, prompt, candidate-model, and
  evaluator-manifest hashes.
- Every generated response is schema-validated; malformed output is retained and scored as a parse
  failure rather than silently repaired.
- Links use a closed initial vocabulary: `references`, `applies_to`, `supersedes`, `defines`, and
  `supported_by`.
- Every derived page retains source ID, source URI, version, and content hash.
- Unknown link types, dangling current-page links, duplicate IDs, and invalid relevance judgments
  fail explicitly.

### Validation

- schema round-trip tests;
- duplicate and malformed fixture rejection tests;
- CLI help and invalid-argument tests;
- deterministic path and seed tests.

## Phase 2: Build the corpus and judgments

### Deliverables

- approximately 80 fictional source records covering products, regions, customer tiers, returns,
  warranties, shipping, billing, and support escalation;
- approximately 200 logical pages split on authored headings and records rather than token counts;
- explicit current, obsolete, and superseding versions;
- graph links authored in fixtures rather than generated by an LLM;
- 80 development queries and 160 locked test queries;
- independently authored canonical-source and authority grades;
- deterministic 2,000-page and 20,000-page distractor tiers around the judged core;
- a small transfer set built from repository Markdown headings and explicit links;
- deterministic missing-edge, spurious-edge, reversed-edge, and link-hub corruption manifests;
- pinned adapters and subset manifests for BEIR, HotpotQA, MultiHop-RAG, KILT provenance, RAGBench,
  and the dated FreshQA snapshot selected for the run;
- a fixed incremental-update manifest containing additions, edits, supersessions, and deletions.
- an edge-acquisition ledger recording automatically extracted, source-authored, and manually
  curated links plus measured authoring and validation time.

### Complaint chain encoded in the data

1. Exact product IDs make vector retrieval unnecessary.
2. Paraphrases make lexical retrieval insufficient.
3. Multi-page questions make isolated top-k chunks insufficient.
4. Obsolete but highly linked policies make raw PageRank unsafe.
5. Unanswerable questions make unconditional retrieval unsafe.

### Validation

- snapshot the corpus and query-suite hashes;
- verify every answerable query has at least one valid relevant page;
- verify every multi-page query declares its complete evidence set;
- verify every stale-page trap has a current replacement;
- verify authority labels are not derived from graph degree or PageRank;
- verify scale-tier distractors do not accidentally satisfy judged queries;
- verify transfer-set judgments were authored without looking at PageRank scores;
- verify every corruption manifest changes only graph edges, not page text or relevance labels;
- verify public benchmark licenses, revisions, splits, and hashes;
- verify benchmark adapters preserve native evidence and answerability labels;
- verify test labels are not imported by tuning code;
- inspect every query and judgment before freezing the first benchmark version.

## Phase 3: Implement deterministic page and graph indexes

### Deliverables

- logical-page builder with stable content hashes;
- SQLite schema for pages, metadata, links, and index provenance;
- FTS5 indexing and BM25 candidate retrieval;
- NetworkX graph projection, raw global PageRank, and eligible typed PageRank;
- one-hop typed-edge expansion;
- incremental page, edge, and rank updates.

The first PageRank implementation may recompute the small graph after a change. The update benchmark
must separately report page/edge writes and rank recomputation so a cheap full recomputation is not
misrepresented as a production incremental algorithm.

### Ranking sequence and initial score

The implementation must preserve this order:

1. apply validity and metadata filters;
2. retrieve query-relevant lexical and/or semantic seeds;
3. expand only configured link types within fixed candidate and edge budgets;
4. rescore every expanded candidate for query relevance;
5. apply a bounded PageRank authority adjustment;
6. select the final evidence set.

The reranker will expose separate normalized terms for:

- lexical relevance;
- semantic relevance;
- PageRank authority;
- metadata-filter match;
- typed-link contribution from a query-relevant seed.

Weights will be selected on the development set and serialized before the locked test run. A page
with no lexical seed path cannot enter the linked-page result solely because it has high PageRank.
Expired, deleted, and ineligible pages cannot receive a score. Raw global PageRank is a naive
control, not the proposed production setting.

The damping factor, link-type weights, authority cap, and expansion budget are part of the frozen
configuration. Development runs include a bounded sensitivity grid; locked-test results are not
used to select them.

### Validation

- deterministic ordering for tied scores;
- graph and SQLite edge parity;
- identical candidate sets for PageRank on/off ranking ablations;
- identical PageRank settings for expansion on/off ablations;
- superseded-page demotion tests;
- one-hop boundary tests;
- candidate and edge-budget enforcement tests;
- damping-factor and link-weight sensitivity tests;
- missing, noisy, reversed, and hub-manipulated edge tests;
- deletion and dangling-edge tests;
- before/after update tests proving that changed pages become visible.

## Phase 4: Implement vector and hybrid baselines

### Deliverables

- pinned Sentence Transformers model ID and revision;
- pinned CPU cross-encoder model ID and revision;
- fixed overlapping-chunk builder for the conventional plain-RAG baseline;
- logical-page vector baseline to isolate segmentation effects;
- batched page and query embedding;
- exact NumPy cosine retrieval;
- CPU HNSW retrieval with frozen construction and search parameters;
- embedding cache keyed by model revision and content hash;
- reciprocal-rank or normalized-score fusion for the strong page-hybrid baseline;
- development-set-only rule for selecting which pages the selective hybrid embeds.

Exact cosine establishes retrieval correctness. HNSW supplies the realistic local latency and index
maintenance baseline. Both use the same embeddings; approximate-search loss is reported explicitly.
The cross-encoder reranker receives the same bounded candidate count for graph and non-graph runs.

### Validation

- embedding dimension and finite-value checks;
- cache invalidation after content or model changes;
- unchanged pages are not re-embedded during incremental updates;
- exact-versus-HNSW Recall@10 and latency comparison;
- deterministic HNSW build under a fixed seed where supported;
- identical cross-encoder candidates for paired PageRank on/off comparisons;
- retrieval ties are stable;
- selective embedding decisions use no locked-test judgments.

## Phase 5: Build the evaluation harness

### Quality evaluation

Compute each metric overall and for every query class:

- Recall@5 and Recall@10;
- MRR@10;
- nDCG@10;
- context precision@10;
- Complete Evidence Rate@10;
- Canonical@1;
- authority-weighted nDCG@10;
- current-over-stale win rate;
- stale exposure@10;
- unanswerable false-positive rate.

For every mechanism, report a paired ablation:

- PageRank on versus off with identical candidates and expansion;
- expansion on versus off with identical retrieval and PageRank settings;
- logical pages versus fixed chunks with the same semantic model;
- selective versus full embedding with the same graph configuration.

Use query-level paired bootstrap 95% confidence intervals for quality deltas. Report the point
estimate, interval, and number of eligible queries for every slice.

Before labels are locked, run a power and minimum-detectable-effect check for each primary slice.
If a slice is too small to support its decision margin, add independently authored queries or mark
that gate exploratory. Freeze the primary metrics, non-inferiority margins, hyperparameter search
budget, and decision rules in a hashed preregistration artifact before opening locked-test results.
Secondary metrics remain diagnostic; they cannot replace a failed primary gate.

Generation evaluation adds:

- answer exact match and token F1 where the imported benchmark defines them;
- required-fact coverage;
- deterministic page-ID citation precision and recall;
- grounded-claim precision;
- answer completeness;
- abstention accuracy;
- structured-output parse-failure rate;
- optional pinned-judge relevance and groundedness.

Run three generations for a fixed 30-query stratified stability subset and report score variance,
parse-failure variance, and pass/fail flips. Before an LLM-judge metric can support a gate, calibrate
it on at least 50 stratified examples labeled independently by two reviewers, report agreement, and
adjudicate disagreements. An uncalibrated judge score remains diagnostic.

### Efficiency evaluation

- run one untimed warm-up before warm-query measurement;
- use the same ordered query list for every strategy;
- repeat timed query runs at least five times;
- report median p50 and p95 rather than the best run;
- run cold builds in new artifact directories;
- apply the same update manifest to every strategy;
- separate embedding-model load time from query time;
- record PageRank computation separately from other indexing work;
- record HNSW construction, search, and update work separately from embedding time;
- record bytes on disk and peak process RSS using one cross-platform measurement implementation;
- record embedding coverage, vectors recomputed, update amplification, candidate counts, traversed
  edges, and evidence tokens;
- record links extracted, links manually curated, edge churn, invalid edges, and authoring or
  validation minutes;
- run the 200-, 2,000-, and 20,000-page scale tiers.

The report must compare efficiency at matched quality and generate a quality/cost Pareto frontier.
Raw latency or storage is not a win when a strategy fails the non-inferiority quality gates.

### Run provenance

Each run record must include:

- Git commit or explicit dirty-worktree marker;
- operating system and CPU;
- logical core count and available memory;
- Python and dependency versions;
- corpus, query-suite, configuration, and update-manifest hashes;
- embedding model ID and revision;
- ANN implementation/version/parameters and reranker model manifest;
- generator and judge model manifests, or an explicit `retrieval-only` marker;
- prompt and output-schema hashes;
- provider profile and remote run IDs when applicable;
- random seed;
- UTC start and end timestamps.

### Validation

- metric unit tests using small hand-calculated rankings;
- model-manifest rejection tests for aliases, missing revisions, ambiguous deployments, and changed
  artifact hashes;
- provider contract tests proving local and Foundry profiles receive identical evidence packets;
- tests proving generator choice cannot mutate stored retriever rankings;
- response-cache and interrupted-run resume tests;
- pre-run versus post-run Foundry metadata drift tests;
- configured request, token, concurrency, retry, and spending-cap tests;
- timer and byte-accounting tests;
- fixture-order randomization tests;
- identical quality metrics across two clean builds;
- report generation from stored metrics without rerunning retrieval.

## Phase 6: Run and interpret the benchmark

### Execution order

1. Freeze version 1 of the corpus and judgments.
2. Run the power check and add data or mark underpowered gates exploratory.
3. Tune weights and abstention thresholds using only the development set and a fixed search budget.
4. Serialize and hash the preregistered metrics, gates, and chosen configuration.
5. Run all paired ablations on the locked test set exactly once per preregistered repeat.
6. Run edge-corruption sensitivity and the naturally linked transfer set.
7. Run the fixed incremental-update benchmark at every scale tier.
8. Repeat the complete benchmark once from a clean artifact directory.
9. Generate confidence intervals, decision gates, and the Pareto frontier.
10. Generate the comparison report.
11. Investigate discrepancies; do not silently average incompatible runs or retune on test data.

### Required report tables

- quality metrics by strategy;
- quality metrics by query class;
- paired PageRank, expansion, segmentation, and selective-indexing deltas with confidence intervals;
- build, query, update, storage, and memory measurements;
- pages embedded and embedding-cache reuse;
- exact-versus-HNSW quality and latency;
- reranker on/off quality, latency, and interaction with PageRank;
- update amplification, candidates scored, edges traversed, and evidence tokens;
- link acquisition/maintenance cost and edge-corruption sensitivity;
- synthetic-versus-natural-link transfer results;
- quality/cost Pareto frontier and scale-tier crossover points;
- decision-gate pass/fail results;
- top retrieval failures with query, expected pages, returned pages, and score components;
- limitations and the next experiment justified by the evidence.

## Phase 7: Run generation through both profiles

This phase begins only after retrieval results are frozen. It does not retune retrieval weights.

### Deliverables

- provider-neutral evidence-packet contract;
- local Phi-4 Mini ONNX Runtime GenAI adapter;
- Foundry deployment metadata resolver and generator adapter;
- Foundry evaluator discovery and dataset batch-evaluation adapter;
- deterministic prompt containing page IDs and citation instructions;
- answer records retaining model, prompt, evidence, output, and timings;
- schema-validation and parse-failure records;
- rule-based required-fact and citation checks;
- a blinded, two-reviewer judge-calibration set with adjudicated labels and agreement metrics;
- separate candidate and judge model manifests;
- dry-run cost plan, budget guard, resumable run ledger, and response cache;
- pre-run and post-run Foundry identity snapshots;
- local-versus-cloud comparison report that never mixes model versions silently.

Generation metrics remain separate from retrieval metrics. A fluent answer cannot turn missing
evidence into a retrieval success.

## Risks and controls

| Risk | Control |
|---|---|
| Synthetic text makes lexical retrieval unrealistically easy | Include reviewed paraphrases and report each query class separately |
| PageRank dominates relevance | Require a query-relevant seed, cap authority contribution, and enforce relevance non-inferiority |
| PageRank receives credit for graph expansion | Use paired PageRank on/off and expansion on/off ablations |
| Logical pages receive credit as PageRank | Compare fixed-chunk vector RAG with logical-page vector RAG |
| Test-set leakage | Separate fixture paths and reject test-label access during tuning |
| Tiny corpus makes all methods look fast | Add 2,000- and 20,000-page tiers and report crossover points without claiming web scale |
| Model download contaminates build timing | Warm/cache the pinned model and report download separately |
| Brute-force vectors make graph latency look artificially good | Keep exact cosine as an oracle and add the same-embedding CPU HNSW baseline |
| Weak non-graph ranking exaggerates PageRank lift | Compare graph variants with the same pinned cross-encoder reranker |
| Link creation cost is hidden | Record automatic, authored, and curated edges plus validation time and edge churn |
| PageRank is fragile to missing or manipulated links | Run fixed edge-drop, edge-noise, reversal, and hub-manipulation tests |
| Selective embeddings are chosen with hindsight | Freeze a label-independent or development-only selection rule |
| Small slices produce unstable pass/fail results | Power-check primary slices and mark underpowered gates exploratory |
| Synthetic graph structure overstates transfer | Require a naturally linked repository-document transfer set |
| Generator quality hides retrieval failure | Keep retriever metrics primary and include an oracle-context ceiling |
| Candidate model judges itself | Require deterministic checks first and a separately identified judge deployment |
| LLM judge disagrees with human labels | Calibrate on a blinded two-reviewer set and keep failed calibration diagnostic only |
| Foundry deployment alias changes model version | Resolve and hash model metadata before every run; reject ambiguous identity |
| Preview evaluator disappears | Discover evaluator catalog and fail explicitly rather than substituting |
| Foundry model changes during a run | Compare pre-run and post-run metadata and invalidate drifted runs |
| Cloud retry duplicates cost or results | Use idempotent cache keys, a run ledger, and bounded retries |
| Private repository content is uploaded unintentionally | Allow only synthetic/public rows by default and require export approval |
| Public benchmark preprocessing changes | Pin repository revision, split, adapter version, and content hash |
| Generated reports become unverifiable claims | Retain metrics, hashes, configuration, and environment provenance |

## Ordered implementation checklist

- [ ] Scaffold package, dependency groups, and CLI.
- [ ] Define and test data contracts.
- [ ] Author deterministic sources, pages, links, queries, and update manifest.
- [ ] Record link acquisition and maintenance effort.
- [ ] Add edge-corruption manifests and the naturally linked transfer set.
- [ ] Add pinned public benchmark adapters and manifests.
- [ ] Generate and validate the 2,000- and 20,000-page scale tiers.
- [ ] Freeze corpus and query-suite version 1 hashes.
- [ ] Implement SQLite FTS5 and lexical baseline.
- [ ] Implement raw and eligible typed PageRank plus negative-control retrieval.
- [ ] Implement linked-page expansion and reranking.
- [ ] Implement fixed-chunk and logical-page vector baselines.
- [ ] Implement exact and HNSW vector retrieval over the same embeddings.
- [ ] Implement the strong page hybrid and pinned cross-encoder reranker without graph features.
- [ ] Implement selective and full hybrid strategies.
- [ ] Power-check slices and freeze a hashed preregistration artifact.
- [ ] Implement quality metrics, paired ablations, confidence intervals, and per-class reports.
- [ ] Implement timing, storage, memory, and update measurements.
- [ ] Implement matched-quality comparisons and the Pareto report.
- [ ] Implement the local Phi-4 Mini CPU profile and model manifest.
- [ ] Implement the Foundry metadata-first profile and evaluator discovery.
- [ ] Implement Foundry budget, drift, privacy, cache, and resume controls.
- [ ] Prove both provider profiles consume identical frozen evidence packets.
- [ ] Calibrate any LLM judge against the blinded human-labeled subset.
- [ ] Tune only on the development split and freeze configuration.
- [ ] Run and reproduce the locked local benchmark.
- [ ] Run and reproduce the local CPU generation evaluation.
- [ ] Run the Foundry profile in an authorized project or retain `live-unvalidated` status.
- [ ] Publish measured results and update the project evidence status.
- [ ] Decide from evidence whether an ANN or optional generation phase is justified.
