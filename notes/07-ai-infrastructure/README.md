# AI Infrastructure — How to Read This Collection

→ Interview prep: [Interview Guide](../interview-guides/ai-infrastructure.md)

> This document is your **entry point and reading map**. It explains the conceptual arc across all chapters, defines the running scenario that threads through every note, shows how each chapter connects to the others, and prescribes reading paths based on your goal.

> **Track Status:** **COMPLETE** — All 10 chapters finished (GPU architecture through production monitoring)

---

## The Central Story in One Paragraph

Every AI model that looks effortless in a demo is actually running on a complex stack of hardware, software, and operational decisions that determine whether it costs $80,000 a month in API bills or $4,000 a month on dedicated compute. **AI Infrastructure is the discipline of understanding that stack from the silicon up**: how a GPU executes a matrix multiply thousands of times per second, how that operation is kept from being memory-starved, how a single model is split across dozens of machines for training, how an inference server batches thousands of user requests into one GPU kernel, and how you build the operational layer that keeps all of it running reliably at scale. The notes in this collection build that understanding from the ground up — starting with a single GPU, ending with a full production AI platform — and they deliberately connect the hardware primitives to the software decisions engineers actually make.

> **Note on running code:** Unlike the ML or AI tracks, this track has no single canonical dataset to compute against. GPU memory, parallelism strategies, and cluster topology are phenomena you observe through profiling, simulation, and calculation — not through training loops on a fixed dataset. Every notebook in this track is a **calculator, simulator, or model estimator**: you bring your own numbers (model size, request rate, hardware budget) and the notebook tells you what to expect before you spend a dollar on cloud compute.

---

## The Running Scenario — InferenceBase

Every note in this track is anchored to a single growing problem: **InferenceBase**, a seed-stage AI startup building a document intelligence API. The product takes enterprise PDF documents, runs them through a self-hosted LLM, and returns structured JSON. The CEO has just forwarded the latest AWS bill — $80,000 in OpenAI API charges for the month — and asked the founding Platform Engineer (you) to evaluate whether self-hosting Llama-3-8B makes sense.

```
InferenceBase after Ch.1:
 Question: What GPU hardware do we even need for Llama-3-8B?
 Answer: Understand CUDA cores, tensor cores, VRAM, and bandwidth — pick a card.

InferenceBase after Ch.2:
 Question: Will the model actually fit? What about training updates?
 Answer: Estimate VRAM precisely — parameters, KV cache, optimizer states.

InferenceBase after Ch.3:
 Question: Can we go smaller without killing quality?
 Answer: Quantize to INT4, benchmark perplexity, decide if the tradeoff holds.

InferenceBase after Ch.4:
 Question: How do we scale out when one GPU isn't enough?
 Answer: Data parallelism, tensor parallelism, ZeRO — pick the right strategy.

InferenceBase after Ch.5:
 Question: How do we serve 10,000 requests/day efficiently?
 Answer: KV cache, continuous batching, PagedAttention — throughput without waste.

InferenceBase after Ch.10:
 Question: How do we know if production models are degrading?
 Answer: Monitor drift, deploy A/B tests, automate rollback on performance drops.

InferenceBase final outcome:
 Original cost: $80,000/month in API bills
 Final cost: $7,300/month for multi-GPU self-hosted infrastructure
 Savings: 91% cost reduction while maintaining production SLAs
```

The key constraint: **InferenceBase has a $15,000/month cloud compute budget to replace $80,000/month in API costs, and the founding Platform Engineer has two weeks**. Every chapter confronts the tradeoffs that constraint forces.

---

## How We Got Here — A Short History of AI Infrastructure

AI infrastructure is one of the few disciplines where you can point at a specific chip, a specific paper, or a specific open-source release and say "before this, you couldn't; after this, you could." **The detailed timeline now lives in [GPUArchitecture](ch01_gpu_architecture) and the other chapter preludes** — each chapter opens with a *"The story"* blockquote that names the people, dates, and bottlenecks behind the breakthrough it teaches.

**The through-line in one paragraph.** Every chapter in this track exists because an earlier bottleneck moved. Compute was the bottleneck → [Tensor Cores](ch01_gpu_architecture) (V100, 2017). Memory was the bottleneck → HBM + ZeRO + Flash Attention (2016–2022) → [MemoryAndComputeBudgets](ch02_memory_and_compute_budgets), [ParallelismAndDistributedTraining](ch04_parallelism_and_distributed_training). Throughput was the bottleneck → continuous batching + PagedAttention (Orca 2022, vLLM 2023) → [InferenceOptimization](ch05_inference_optimization). Cost was the bottleneck → quantisation + speculative decoding + cloud arbitrage (GPTQ/AWQ/GGUF 2023) → [QuantizationAndPrecision](ch03_quantization_and_precision). Operational reliability was the bottleneck → MLOps tools and feature stores (MLflow 2020, Feast 2021) → [MLExperimentTracking](ch09_ml_experiment_tracking), [ProductionMLMonitoring](ch10_production_ml_monitoring).

---

## The Conceptual Architecture

```
Currently Implemented (Ch.1–7):

┌─────────────────────────────────────────────────────────────────────────────┐
│ AI INFRASTRUCTURE STACK │
│ │
│ ┌────────────────────────────────────────────────────────────────────────┐ │
│ │ INFERENCE LAYER (Ch.5) — IMPLEMENTED │ │
│ │ │ │
│ │ KV Cache · Continuous Batching · PagedAttention │ │
│ │ Ch.6: vLLM, ONNX Runtime, TensorRT (Serving Frameworks) │ │
│ └───────────────────────────────┬──────────────────────────────────────┘ │
│ │ │
│ ┌────────────────────────────────▼──────────────────────────────────────┐ │
│ │ PARALLELISM LAYER (Ch.4) — IMPLEMENTED │ │
│ │ │ │
│ │ Data / Tensor / Pipeline Parallelism · ZeRO │ │
│ │ Ch.7: NVLink, InfiniBand, RDMA (Networking) — IMPLEMENTED │ │
│ └───────────────────────────────┬──────────────────────────────────────┘ │
│ │ │
│ ┌────────────────────────────────▼──────────────────────────────────────┐ │
│ │ OPTIMIZATION LAYER (Ch.3) — IMPLEMENTED │ │
│ │ │ │
│ │ BF16 / INT8 / INT4 · GPTQ · AWQ · GGUF · Loss Scaling │ │
│ └───────────────────────────────┬──────────────────────────────────────┘ │
│ │ │
│ ┌────────────────────────────────▼──────────────────────────────────────┐ │
│ │ MEMORY LAYER (Ch.2) — IMPLEMENTED │ │
│ │ │ │
│ │ Parameters · Activations · Optimizer States · KV Cache │ │
│ │ VRAM Budget · Offloading · Flash Attention │ │
│ └───────────────────────────────┬──────────────────────────────────────┘ │
│ │ │
│ ┌────────────────────────────────▼──────────────────────────────────────┐ │
│ │ HARDWARE LAYER (Ch.1) — COMPLETE │ │
│ │ │ │
│ │ CUDA Cores · Tensor Cores · HBM · Memory Bandwidth · Roofline Model │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Chapter Map

### Foundation

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.1 | [GPUArchitecture/](ch01_gpu_architecture) | What does a GPU actually do, and how do its specs translate to AI workloads? |
| Ch.2 | [MemoryAndComputeBudgets/](ch02_memory_and_compute_budgets) | How much VRAM does this model actually need — at inference and at training? |

### Optimization

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.3 | [QuantizationAndPrecision/](ch03_quantization_and_precision) | How do you shrink a model without destroying its quality? |

### Scale-Out

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.4 | [ParallelismAndDistributedTraining/](ch04_parallelism_and_distributed_training) | When one GPU is not enough — how do you split a model across many? |
| Ch.5 | [InferenceOptimization/](ch05_inference_optimization) | How do you serve thousands of requests efficiently without blowing the latency SLA? |

### Infrastructure

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.6 | [ModelServingFrameworks/](ch06_model_serving_frameworks) | vLLM vs ONNX vs TensorRT — which one and when? |
| Ch.7 | [AISpecificNetworking/](ch07_ai_specific_networking) | What happens between the GPUs — bandwidth, latency, and collective operations? |
| Ch.8 | [FeatureStores/](ch08_feature_stores) | How do you serve real-time ML features without killing latency? |

### Operations

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.9 | [MLExperimentTracking/](ch09_ml_experiment_tracking) | How do you run training jobs that survive preemption, track experiments, and version models? |
| Ch.10 | [ProductionMLMonitoring/](ch10_production_ml_monitoring) | What does the complete stack look like when it's running in production for real users? |

---

## Reading Paths

### "I need to know if I can self-host this model"
→ Ch.1 → Ch.2 → Ch.3

*Goal: hardware selection + VRAM sizing + quantization tradeoff.*

### "I need to serve this model at scale"
→ Ch.5 → Ch.6 → Ch.7

*Goal: inference optimization + serving framework selection + cluster networking.*

### "I need to scale training beyond one GPU"
→ Ch.1 → Ch.2 → Ch.4

*Goal: GPU fundamentals + memory budgets + parallelism strategy.*

### "I want the full picture end to end"
→ Ch.1 → Ch.2 → Ch.3 → Ch.4 → Ch.5 → Ch.6 → Ch.7 → Ch.8 → Ch.9 → Ch.10

*Complete infrastructure stack: hardware through production operations.*

---

## Story Arc — How the Concepts Chain Together

```
START HERE
 │
 ▼
Step 0: UNDERSTAND THE HARDWARE
 Ch.1 — GPU Architecture

 Key insight: A GPU is not a fast CPU. It is a massively parallel
 matrix-multiply machine with a narrow memory bandwidth bottleneck.
 Every subsequent decision — quantization, parallelism, serving
 strategy — is an attempt to either feed that machine faster
 or do more work per memory access.
 │
 ▼
Step 1: COUNT YOUR MEMORY BEFORE YOU START
 Ch.2 — Memory & Compute Budgets

 Key insight: Most deployment failures are not code failures —
 they are VRAM failures. A 7B parameter model in FP16 needs
 ~14 GB just for weights. Add optimizer states, activations,
 and the KV cache and you are at 40–80 GB before writing
 a single line of training code.
 │
 ▼
Step 2: SQUEEZE THE MODEL
 Ch.3 — Quantization & Precision

 Key insight: INT4 quantization cuts memory by 8× vs FP32, with
 1–3% perplexity degradation on most models. BF16 training is
 almost always better than FP16. FP8 is the emerging standard
 for H100-class hardware.
 │
 ▼
Step 3: SCALE OUT THE TRAINING
 Ch.4 — Parallelism & Distributed Training

 Key insight: Data parallelism is trivial to implement but hits
 a wall at the per-GPU memory limit. Once the model no longer
 fits on one GPU, you need tensor or pipeline parallelism —
 and the communication cost becomes the dominant constraint.
 │
 ▼
Step 4: OPTIMISE THE INFERENCE PATH
 Ch.5 — Inference Optimization

 Key insight: Training throughput and inference throughput have
 opposite optima. Training wants large batches; inference wants
 low latency. The KV cache, continuous batching, and PagedAttention
 are the mechanisms that reconcile these pressures.
 │
 ▼
Step 5: TRACK YOUR EXPERIMENTS
 Ch.9 — ML Experiment Tracking & Model Registry

 Key insight: At 100+ experiments, you can't rely on file names
 and notebook cells. MLflow tracks hyperparameters and metrics
 across runs. DVC versions datasets. Model registries manage
 deployment lineage — what's in production, what's staged, what failed.
 │
 ▼
Step 6: MONITOR PRODUCTION
 Ch.10 — Production ML Monitoring & A/B Testing

 Key insight: Models degrade silently. Drift detection catches
 distribution shifts. A/B testing validates model improvements
 with real users. Automated rollback prevents bad deployments
 from lasting more than one alert cycle.
```

---

## Chapter Detail — What Each Note Covers

### Ch.1 — GPU Architecture Fundamentals

**Core question:** What does a GPU actually do, and how do its specs translate to AI workloads?

**Concepts covered:**
- CUDA cores vs Tensor Cores vs SM (Streaming Multiprocessor) structure
- Memory hierarchy: registers → L1/L2 cache → HBM (VRAM) → system RAM
- Memory bandwidth vs compute throughput — why AI workloads are bandwidth-bound
- The Roofline Model: the graphical map of compute vs memory limits
- Arithmetic intensity: FLOPs per byte — where your operation lives on the roofline
- Key GPU generations: V100 (32 GB HBM2) → A100 (80 GB HBM2e) → H100 (80 GB HBM3) → consumer (RTX 4090 24 GB)
- GPU specs you actually need to read: TFLOPS (BF16), memory bandwidth (TB/s), VRAM capacity, NVLink bandwidth
- Why a matrix multiply maps naturally to GPU parallelism

**Notebook:** Roofline model calculator — given your GPU's specs and your operation's arithmetic intensity, visualise whether you are compute-bound or memory-bound. Compare representative operations: embedding lookup, attention, linear layer, convolution.

**InferenceBase angle:** Which GPU can run Llama-3-8B inference at 50 tokens/sec? Model the answer before ordering hardware.

---

### Ch.2 — Memory & Compute Budgets

**Core question:** How much VRAM does this model need — and will it fit?

**Concepts covered:**
- Parameter count → memory size: FP32 (4 B/param), FP16/BF16 (2 B/param), INT8 (1 B/param), INT4 (0.5 B/param)
- Inference memory = weights + KV cache + activations
- Training memory = weights + gradients + optimizer states (Adam = 3–4× weights in FP32)
- Mixed-precision training: FP16/BF16 forward + FP32 master weights + loss scaling
- KV cache growth: `2 × layers × heads × head_dim × seq_len × batch_size × bytes_per_element`
- Memory offloading: ZeRO-Offload, CPU offloading, disk offloading
- Flash Attention: recompute from Q/K/V instead of materialising the full $n^2$ attention matrix
- Model parallelism as a memory solution (preview of Ch.4)

**Notebook:** Interactive VRAM budget calculator — input model architecture (layers, heads, hidden dim, seq len, batch size, dtype) and get: inference VRAM, training VRAM with Adam, KV cache size as sequence length grows.

**InferenceBase angle:** Llama-3-8B (8B params, 32 layers, 32 heads, 4096 hidden dim) in BF16 = 16 GB weights + KV cache. Fit analysis across A10G (24 GB), A100 (40 GB / 80 GB), consumer RTX 4090 (24 GB).

---

### Ch.3 — Quantization & Precision

**Core question:** How do you shrink a model without destroying its quality?

**Concepts covered:**
- Floating point formats: FP32, FP16, BF16, FP8 (E4M3 / E5M2), INT8, INT4 — tradeoffs in range, precision, hardware support
- Why BF16 is preferred over FP16 for training (larger exponent range → less overflow)
- Post-training quantization (PTQ): apply after training, no retraining needed; fast but lossy
- Quantization-aware training (QAT): simulate quantization during training; better quality, expensive
- Weight-only quantization vs activation quantization vs KV cache quantization
- GPTQ: layer-wise PTQ with Hessian-based weight adjustment — the dominant format for LLM compression
- AWQ (Activation-aware Weight Quantization): identify salient weights, protect them from quantization
- GGUF / llama.cpp: CPU-friendly format, supports mixed-precision per tensor group
- Perplexity as a quantization quality metric
- The compression tradeoff table: INT4 saves 4–8× VRAM at 1–3% perplexity cost for most models

**Notebook:** NumPy quantization simulator — demonstrate FP32 → INT8 → INT4 rounding error on a weight tensor; plot the distribution of quantization error; compute effective precision loss.

**InferenceBase angle:** Llama-3-8B in GPTQ INT4 → 4.5 GB VRAM. Fits on a single RTX 3080 Ti (12 GB). Does the quality hold for document information extraction tasks?

---

### Ch.4 — Parallelism & Distributed Training

**Core question:** When one GPU is not enough — how do you split a model across many?

**Concepts covered:**
- Why one GPU is not enough: scaling laws, model size wall, memory wall
- **Data parallelism (DP/DDP):** replicate the full model on each GPU; split the batch; all-reduce gradients
- **Tensor parallelism (TP):** split individual weight matrices across GPUs (Megatron-style column/row splitting)
- **Pipeline parallelism (PP):** assign consecutive transformer layers to different GPUs; GPU bubbles
- **Sequence parallelism:** split along the sequence dimension for attention; reduces activation memory
- **ZeRO (Zero Redundancy Optimizer):** stages 1/2/3 — partition optimizer states, gradients, parameters across GPUs; compare to DP
- Gradient accumulation: simulate larger batch sizes without extra VRAM
- Communication collectives: all-reduce, all-gather, reduce-scatter — which parallelism uses which
- Frameworks: PyTorch DDP, FSDP (ZeRO-3 equivalent), DeepSpeed, Megatron-LM
- 3D parallelism: DP × TP × PP combined in large-scale training (GPT-4 class runs)

**Notebook:** Parallelism strategy simulator — model the effective throughput, communication overhead, and pipeline bubble fraction for a given model size, GPU count, and parallelism configuration.

**InferenceBase angle:** Fine-tuning Llama-3-8B on 4× A100s: which strategy is best? ZeRO-2 with DDP vs tensor parallelism vs FSDP — expected VRAM and training time for each.

---

### Ch.5 — Inference Optimization

**Core question:** How do you serve thousands of requests efficiently without blowing the latency SLA?

**Concepts covered:**
- The inference compute graph: prefill phase vs decode phase — different bottlenecks
- **KV cache:** store past-token key/value projections; avoid recomputing them at every decode step
- Why naive inference is inefficient: static batching GPU utilisation patterns
- **Continuous batching (iteration-level scheduling):** retire finished sequences, insert new ones mid-batch; the key vLLM innovation
- **PagedAttention:** store KV cache in non-contiguous memory pages (like virtual memory); eliminates fragmentation; enables large concurrent batch sizes
- **Speculative decoding:** small draft model generates candidates; large verifier model approves many tokens per forward pass
- **Flash Attention:** fused kernel that avoids materialising the full $O(n^2)$ attention matrix; cuts memory from $O(n^2)$ to $O(n)$; 2–4× faster on long sequences
- Throughput vs latency tradeoff curves: batch size ↑ → throughput ↑, latency ↑
- Prefill chunking: break long prompts into chunks to avoid blocking the decode queue
- Prefix caching: cache the KV states of a shared system prompt across all users

**Notebook:** Continuous batching throughput simulator — model requests arriving at a Poisson rate, simulate naive static batching vs continuous batching GPU utilisation; plot latency distribution and GPU idle fraction.

**InferenceBase angle:** At 10,000 document requests/day (avg 2,000 input tokens, 300 output tokens), how many A100s does InferenceBase need under each batching strategy?

---

### Ch.9 — ML Experiment Tracking & Model Registry

**Core question:** How do you run training jobs that survive preemption, track experiments, and version models?

**Concepts covered:**
- Why experiment tracking matters: 100+ hyperparameter combinations, reproducibility, model lineage
- **MLflow:** experiment tracking, parameter logging, metric comparison, model registry
- **Weights & Biases (W&B):** real-time dashboards, hyperparameter sweeps, collaborative team tracking
- **Model registry:** staging/production/archive lifecycle, model versioning, deployment lineage
- **DVC (Data Version Control):** version large datasets and models with Git-like semantics; track pipeline dependencies
- **Checkpointing:** save training state every N steps; resume after spot instance preemption
- **Fault tolerance:** gradient checkpointing, activation recomputation, elastic training (add/remove GPUs mid-run)
- **Cluster schedulers:** SLURM vs Kubernetes — job queues, resource allocation, priority scheduling
- **Hyperparameter tuning:** grid search vs random search vs Bayesian optimization (Optuna, Ray Tune)
- **Experiment organization:** tagging, filtering, comparing runs across teams

**Notebook:** MLflow experiment tracker simulator — log 100+ runs with varying hyperparameters; query best runs by metric; visualize learning curves; demonstrate model registry promotion workflow (candidate → staging → production).

**InferenceBase angle:** InferenceBase runs nightly fine-tuning experiments on new document batches. Track 100+ runs per week. Which hyperparameters actually improved F1 on invoice extraction? What dataset version was used for the production model deployed last Tuesday?

---

### Ch.10 — Production ML Monitoring & A/B Testing

**Core question:** What does the complete stack look like when it's running in production for real users?

**Concepts covered:**
- **Model drift:** input distribution shifts, concept drift, covariate shift — detection via statistical tests (KS test, PSI)
- **Performance monitoring:** latency (p50/p95/p99), throughput (requests/sec), error rate, GPU utilization
- **Data quality monitoring:** null rate, out-of-range values, schema validation, feature correlation changes
- **Prediction monitoring:** output distribution tracking, confidence score calibration, label shift detection
- **A/B testing:** traffic splitting (50/50, 90/10), statistical significance, multi-armed bandits
- **Shadow deployment:** run new model in parallel with production; log predictions without serving users
- **Canary deployment:** route 5% traffic to new model; monitor error rate; roll out or roll back
- **Automated rollback:** trigger rollback on alert: latency > SLA, error rate spike, drift score threshold exceeded
- **Model retraining triggers:** schedule-based (weekly), drift-based (PSI > 0.1), performance-based (accuracy drop > 2%)
- **Observability stack:** Prometheus + Grafana for metrics, ELK/Loki for logs, Jaeger for traces
- **Cost tracking:** per-request inference cost, GPU utilization, spot instance savings

**Notebook:** Drift detection + A/B test simulator — simulate production traffic with gradual distribution shift; run KS test to detect drift; model A/B test with statistical significance calculation; visualize rollback decision flow based on error rate threshold.

**InferenceBase angle:** InferenceBase deploys a new quantized model (Llama-3-8B INT4 vs BF16). A/B test with 10% traffic. New model has 15% lower latency but 2% worse F1 on invoice line-item extraction. Drift detector fires after 3 days — input document format shifted (more scanned PDFs, fewer digital). Should we rollback? Retrain? The monitoring stack answers before the CEO asks.

---

## What Each Chapter Contains

Every chapter in this track follows the same structure (mirroring the ML and AI tracks):

```
notes/AIInfrastructure/ (10 chapters complete)
├── GPUArchitecture/
│ ├── GPUArchitecture.md ← Technical deep-dive: concepts, diagrams, interview checklist
│ └── notebook.ipynb ← Calculator / simulator that makes the concepts concrete
├── MemoryAndComputeBudgets/
│ ├── MemoryAndComputeBudgets.md
│ └── notebook.ipynb
├── QuantizationAndPrecision/
├── ParallelismAndDistributedTraining/
├── InferenceOptimization/
├── ModelServingFrameworks/
├── AISpecificNetworking/
├── FeatureStores/
├── MLExperimentTracking/
│ ├── MLExperimentTracking.md
│ └── notebook.ipynb
└── ProductionMLMonitoring/
 ├── ProductionMLMonitoring.md
 └── notebook.ipynb
```

### Chapter README structure

```
# Ch.N — [Topic Name]

## Core Idea (3–4 sentences, plain English)

## Running Example — InferenceBase in Action
(one paragraph: how does this chapter's concept hit the startup scenario?)

## The Concepts
(key ideas, equations where applicable — every term explained inline)

## How It Works — Step by Step
(numbered walkthrough or decision flow in Mermaid/ASCII)

## The Key Diagram
(Mermaid or ASCII art — minimum 1)

## The Numbers That Matter
(concrete benchmarks, reference specs, order-of-magnitude estimates)

## What Can Go Wrong
(3–5 failure modes, one sentence each)

## Interview Checklist
| Must know | Likely asked | Trap to avoid |

## Bridge to the Next Chapter
```

### Notebook structure

```
[markdown] Chapter title + one-liner
[markdown] ## Core Idea
[markdown] ## The InferenceBase Setup
[code] Define model/hardware parameters (user fills these in)
[markdown] ## The Math / Simulation
[code] Implement the calculator, model, or simulator
[code] Visualise the result
[markdown] ## Sensitivity Analysis
[code] Sweep a key variable — what changes?
[markdown] ## Exercises
[code] 2–3 prompts for the reader to explore independently
```

---

## Interview Checklist — The Track in 90 Seconds (Ch.1–5, 9–10)

| Topic | Must Know | Common Trap |
|-------|-----------|-------------|
| GPU architecture | Tensor Cores do INT/FP matrix multiply; CUDA cores do scalar FP | "More CUDA cores = faster" — no, bandwidth is usually the real limit |
| Memory | 7B model in BF16 = 14 GB weights; add KV cache, activations, optimizer states | Forgetting KV cache growth with sequence length and batch size |
| Quantization | INT4 saves ~8× VRAM vs FP32; GPTQ/AWQ are leading PTQ methods | Quantization is not free: perplexity degrades, especially for reasoning tasks |
| Parallelism | Data parallelism = gradient sync; Tensor parallelism = weight sharding | "More GPUs = linear speedup" — communication overhead breaks this |
| Inference optimization | KV cache avoids recomputing past tokens; continuous batching keeps GPU busy | Confusing prefill (prompt processing) with decode (generation) — very different bottlenecks |
| Experiment tracking | MLflow logs runs, DVC versions data, model registry manages deployments | Tracking everything — track hyperparameters and final metrics, not every step |
| Production monitoring | Drift = distribution shift; A/B test = traffic split; rollback = revert on alert | Deploying without shadow mode — always validate new models on real traffic first |

*(Chapters 6–8, covering serving frameworks, networking, and cloud infrastructure, are planned.)*

---

## Connections to Other Tracks

| This chapter | Connects to |
|---|---|
| Ch.1 (GPU Architecture) | ML Ch.18 (Transformers) — the operations that run on those tensor cores |
| Ch.2 (Memory Budgets) | AI CostAndLatency.md — the VRAM side of the same cost model |
| Ch.3 (Quantization) | AI FineTuning.md — QLoRA is quantization + LoRA combined |
| Ch.4 (Parallelism) | AI FineTuning.md — FSDP and DeepSpeed are used for large fine-tuning runs |
| Ch.5 (Inference Opt.) | AI CostAndLatency.md — KV cache and batching are the mechanism behind the cost numbers |
| Ch.6 (Serving) | AI ReActAndSemanticKernel.md — agents call LLMs via exactly these serving APIs |
| Ch.9 (Experiment Tracking) | ML track — same experiment tracking patterns for traditional ML models |
| Ch.10 (Production Monitoring) | AI EvaluatingAISystems.md — online evaluation is production monitoring |
| Ch.10 (Production) | MultimodalAI LocalDiffusionLab — the same serving patterns apply to diffusion models |

---

> **Status:** Roadmap complete. Chapters are built in order. See individual chapter folders for current state.
