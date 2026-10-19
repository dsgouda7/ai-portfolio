# Curriculum Evaluation

**Contents:** [one-line map of this directory](CONTENTS.md)

This is a read-only pedagogy audit of every learning notebook. Nine reviewers ran in parallel, one per stage group, after commit `802533a` (additive bridges, model-evolution track, focused Transformer copy, contents maps) and `844461c` (generated images). Reviewers parsed notebook sources, theory notes, READMEs, and `CONTENTS.md`; they did not execute notebooks or edit files.

Each notebook gets two scores out of 5:

- **Novice:** can someone new to the concept follow it in plain English?
- **Flow:** does each concept feel necessary before it is named, and do the handoffs hold together?

Every fix proposed here is additive. None asks to remove content from the Stage 04 gold standard.

## 1. Headline

| Stage | Notebooks | Avg novice | Avg flow | One-line verdict |
|---|---:|---:|---:|---|
| 01 ML foundations | 4 | 3.5 | 4.0 | Coherent route; two evidence-contract bugs (test-set timing, stale ReLU gradient). |
| 02 PyTorch foundations | 9 | 3.7 | 3.8 | Solid loop to lifecycle progression; three factual slips and early bridge jargon. |
| 03 Language-model foundations | 10 | 4.1 | 4.1 | Strongest stage; new model-evolution track reads well. 02C is too dense. |
| 04 Transformers (+ focused) | 11 | 3.5 | 3.9 | Gold standard intact; bridges integrate. Focused Part 1 lacks the full attention-path reminder. |
| 05 LLM training | 5 | 3.4 | 4.0 | Clean data to checkpoint to lifecycle; stale "Notebook 08" reference. |
| 06 LLM fine-tuning | 4 | 2.8 | 3.5 | Weakest novice score; monolithic Part 4 runner, title drift, PPO depth. |
| 07 RAG | 3 | 3.7 | 4.0 | Good; toy dense encoder needs an explicit production bridge. |
| 08 LLM evaluation | 4 | 3.5 | 4.0 | Good ownership; heuristic hallucination scores need "proxy" labels. |
| 09 LLM gateway | 2 | 3.5 | 3.5 | Lab is broad; needs a marked short required route. |
| 10 Agentic AI | 11 | 3.0 | 3.8 | Strong evidence ladder; notation-first openings; approval routing vs granting blurred. |
| 11 AI infrastructure | 10 | 2.8 | 3.5 | Densest stage; weak handoff back to Stage 04 attention; model-name and memory-rule errors. |
| 12 Role-based tracks | 15 | 3.0 | 4.0 | Good continuity via Riverside; capstones heavy; CONTENTS miss template dirs. |
| **All** | **88** | **3.3** | **3.9** | Flow is consistently good. Plain-English accessibility drops after Stage 05. |

**The pattern:** flow scores stay near 4 across the curriculum, because the complaint chain and evidence ladder survive everywhere. Novice scores fall from about 4 in Stages 01–04 to about 3 in Stages 06 and 10–12. The later stages open with notation, jargon, or long operational code before a plain-English complaint. That is the main gap against the plan's Section 10.1 hard requirement.

## 2. Cross-cutting inconsistencies

1. **Plain English slips after Stage 05.** Stage 10 openings lead with history and a notation box (Ch 00, 01, 04, 06). Stage 11 chapters front-load jargon (05 distributed, 06 quantization, 08 Triton score 2/5 novice). Stage 06 Parts 1–4 need post-code walkthroughs.
2. **The attention mental model is not reused downstream.** Stage 11 Ch 04 (FlashAttention) and Ch 07 (KV cache) restate attention without pointing back to Stage 04's Q/K/V picture or its "cache stores past K/V; attention over the prefix still happens" distinction.
3. **Running-example drift without a named bridge:**
   - SmartVal changes from rental listings to XOR (Stage 01 Ch 02→03).
   - Sensor becomes fuel (Stage 02 Ch 01→03).
   - Riverside moves from policy retrieval to manuscript evaluation to an editorial gateway (Stages 07→08→09).
   - Stage 10 switches between OrderFlow PO numbers.
   - Stage 06 uses SmolLM2 instead of the Stage 05 tiny checkpoint.

   Each change is defensible, but each needs one sentence saying why the example changed.
4. **Evidence labels are inconsistent.** Stage 12 FDE's `[Measured] / [Modeled] / [Customer-validated]` is the model to copy. Stages 08 (hallucination proxy), 11 (cost, speed, and memory numbers) and 02 ("resumable" checkpoints not tested for exact resume) let toy or reference numbers read as validated.
5. **Stale or drifting names.**
   - Stage 05 "Notebook 08" should be Part 3.
   - Stage 06 Part 2 has two titles.
   - Stage 02 shortcut uses legacy Parts 00–04.
   - Stage 11 has a fictional "LLaMA-3-7B".
   - The Stage 02 refresher claims "CBOW is first", which contradicts Stage 03.
6. **Encoder-decoder appears three times on purpose:** 02C first introduces it, model-evolution 03 runs the measured comparison, and Stage 04 5A covers Transformer cross-attention. No contradiction was found, and teacher forcing is consistent across all three. Readers still need a one-line "which layer is this?" map in 02C and model-evolution 03.
7. **Duplicated bridge text.** The Stage 03 to retrieval-embedding bridge appears nearly verbatim in Stage 07 01a and 01b. Some Stage 10 and 12 prerequisite blocks read as templated.
8. **Gate semantics.** The Stage 10 Chapter 02 notebook still hands off directly to 03, skipping the required 02a. Stage 12 AI-Engineer calls Stage 10 "optional", while the Stage 12 entry gate requires it.
9. **Incomplete `CONTENTS.md` maps.** Stage 12 FDE and AI-Engineer capstone maps omit `templates/`, `cases/`, and `examples/` children, contrary to plan Section 10.3.
10. **Theory-note length.** Stage 04 theory notes for Parts 2–6 run about 1,240–1,410 words, above the roughly 850–1,150 target. This is minor; signpost rather than expand.

## 3. Prioritized fix list (additive)

### P1: confusing or wrong

| # | Where | Fix |
|---|---|---|
| 1 | `01-ml-foundations/02-ml-basics/ml-basics.ipynb` | Defer all test-set reporting until model and threshold choices are final; align the scorecard and coverage ledger. |
| 2 | `01-ml-foundations/03-neural-networks-and-backprop/01-smartval-…ipynb`, "Your turn: close one ReLU gate" | Recompute the forward and backward pass; print measured before/after gradients. |
| 3 | `02-pytorch-foundations` framework translation, Ch 05, quick-refresher | Fix the three-class/two-class mismatch; qualify "resumable" at first use; correct the "CBOW is first" claim. |
| 4 | `03-…/03A-training-skeleton-and-word-objectives.ipynb` opening | Plain-English overview of the three tasks before the shape table; add a "focus here for model-evolution 03" cue. |
| 5 | `03-…/optional/embedding-history/01-…ipynb` §2–3 | Add intuition and result interpretation before the Word2Vec/GloVe code. |
| 6 | `04-transformers/focused/01-…-focused.ipynb` "Before the rotating dials" | Add the full attention-path reminder (Q/K rotate, V does not) and a "checked against canonical revision" note. |
| 7 | `05-llm-training/01-pretraining-data-pipeline.ipynb` and theory | Replace "Notebook 08" with Part 3 (`03-pretrain-a-base-model-lab.ipynb`). |
| 8 | `06-llm-finetuning/04-llm-finetuning-practice.ipynb` | Split the 307-line runner into annotated, learner-sized stages. |
| 9 | `06-llm-finetuning/02-…` and theory | Fix the LoRA concept number; remove the duplicated freezing paragraph. |
| 10 | `07-rag/01b-hybrid-retrieval-lab.ipynb` | Bound the toy dense encoder; add a production embedding/index bridge table. |
| 11 | `08-llm-evaluation/03-hallucination-detection.ipynb` | Label heuristic claim-risk outputs as proxy, not validated attribution. |
| 12 | `09-llm-gateway/02-gateway-routing-resilience-lab.ipynb` | Mark a short required route; demote extensions to optional. |
| 13 | `10-agentic-ai/02-state-context-and-memory` notebook | In-notebook plain-English memory bridge; route the closing handoff through 02a. |
| 14 | `10-agentic-ai/03-durable-workflows-with-langgraph` | Separate "routed to approver" from "approval granted"; test the human-in-the-loop boundary. |
| 15 | `11-ai-infrastructure/04-flash-attention`, `07-inference-systems` Part 1 | Restore the Stage 04 Q/K/V handoff; state what the KV cache stores and what it does not remove. |
| 16 | `11-ai-infrastructure/06-quantization` | Correct "LLaMA-3-7B"; separate real models from hypothetical shapes. |
| 17 | `11-ai-infrastructure/02-mixed-precision`, `05-distributed-training` | Show optimizer/precision/sharding assumptions behind the 4× and 17 GB rules; separate persistent state from activations. |

### P2: notable friction

- Name every running-example switch in one sentence (Section 2, item 3).
- Adopt FDE-style evidence labels (`[Measured]`, `[Reference]`, `[Toy]`, `[Production target]`) in Stages 02, 08 and 11.
- Add a plain-English lead-in before notation in Stage 10 Ch 00, 01, 04 and 06, Stage 11 Ch 05, 06 and 08, and Stage 12 FDE 02, 05 and 07.
- Add a one-line layer map to 02C and model-evolution 03: "first introduction → measured comparison → Transformer replacement".
- Stage 04: distinguish CBOW's center-word lesson from autoregression (Part 1). Explain why numeric reversal isolates address loss (5A). Bound what averaged attention weights can prove (5B).
- Stage 06: reconcile Part 2's title; explain the SmolLM2 substitution; make PPO depth optional.
- Stage 07: keep the Stage 03 embedding bridge once (01a); use a one-line callback in 01b.
- Stage 12: complete the capstone and FDE `CONTENTS.md` maps; clarify that "optional" Stage 10 depth does not waive the Stage 10 gate.
- Show baseline and trial values for the Stage 02 Ch 01 and 03 epoch and learning-rate drills.

### P3: polish

- Visuals the reviewers asked for: a Q/K/V trace for Stage 11 Ch 04/07; a persistent-vs-temporary memory ledger for Stage 11 Ch 02/05; a request-stage timeline (Stage 12 AI-Eng 03); a trusted-context path (Stage 12 FDE 02); a weights/optimizer/progress state visual (Stage 02 Ch 03–05).
- Fix the tab-corrupted `\tau` in Stage 10 Ch 05; label its three-fixture p95 as illustrative.
- Signpost the over-length Stage 04 theory notes; fix the FDE README list numbering.

## 4. What is working (keep)

- Stage 03's new model-evolution track scores 4–4.5. It is framed as a complaint route, not a leaderboard, with LSTM as a side fix and attention shown before Transformers.
- Stage 04's additive bridges (reader map, real RoPE path, block-family order, Stage 03 role reuse) integrate without displacing the gold content. No technical contradictions were found.
- Teacher forcing is explained consistently and in plain English across 02C, model-evolution 03 and Stage 04 4A/5A.
- Stage 12 FDE's evidence labels and the Riverside continuity are the strongest applied-track pattern in the curriculum.

## 5. Per-notebook scores

| Notebook | Novice | Flow |
|---|---:|---:|
| [01-ml-foundations/01-math-foundations/math-foundations-for-ml.ipynb](01-ml-foundations/01-math-foundations/math-foundations-for-ml.ipynb) | 4 | 4 |
| [01-ml-foundations/02-ml-basics/ml-basics.ipynb](01-ml-foundations/02-ml-basics/ml-basics.ipynb) | 4 | 4 |
| [01-ml-foundations/03-neural-networks-and-backprop/01-smartval-neural-networks-and-backprop.ipynb](01-ml-foundations/03-neural-networks-and-backprop/01-smartval-neural-networks-and-backprop.ipynb) | 3 | 4 |
| [01-ml-foundations/03-neural-networks-and-backprop/optional/02-melodyne-backprop-synthesizer.ipynb](01-ml-foundations/03-neural-networks-and-backprop/optional/02-melodyne-backprop-synthesizer.ipynb) | 3 | 4 |
| [02-pytorch-foundations/01-deep-learning-map/01-deep-learning-map.ipynb](02-pytorch-foundations/01-deep-learning-map/01-deep-learning-map.ipynb) | 4 | 4 |
| [02-pytorch-foundations/02-tensors-and-autograd/02-tensors-and-autograd.ipynb](02-pytorch-foundations/02-tensors-and-autograd/02-tensors-and-autograd.ipynb) | 4 | 4 |
| [02-pytorch-foundations/03-models-and-training-loops/03-models-and-training-loops.ipynb](02-pytorch-foundations/03-models-and-training-loops/03-models-and-training-loops.ipynb) | 4 | 4 |
| [02-pytorch-foundations/04-data-checkpoints-and-hyperparameters/04-data-checkpoints-and-hyperparameters.ipynb](02-pytorch-foundations/04-data-checkpoints-and-hyperparameters/04-data-checkpoints-and-hyperparameters.ipynb) | 4 | 4 |
| [02-pytorch-foundations/05-modern-training-lifecycle/05-modern-training-lifecycle.ipynb](02-pytorch-foundations/05-modern-training-lifecycle/05-modern-training-lifecycle.ipynb) | 3 | 4 |
| [02-pytorch-foundations/optional/framework-translation/01-keras-to-pytorch-antarctic-field-guide.ipynb](02-pytorch-foundations/optional/framework-translation/01-keras-to-pytorch-antarctic-field-guide.ipynb) | 3 | 3 |
| [02-pytorch-foundations/optional/vision/01-pytorch-cnns.ipynb](02-pytorch-foundations/optional/vision/01-pytorch-cnns.ipynb) | 4 | 4 |
| [02-pytorch-foundations/optional/vision/02-pytorch-autoencoders.ipynb](02-pytorch-foundations/optional/vision/02-pytorch-autoencoders.ipynb) | 4 | 4 |
| [02-pytorch-foundations/quick-refresher/pytorch-training-skeleton.ipynb](02-pytorch-foundations/quick-refresher/pytorch-training-skeleton.ipynb) | 3 | 3 |
| [03-language-model-foundations/01-sequence-memory/01A-sequence-memory-theory.ipynb](03-language-model-foundations/01-sequence-memory/01A-sequence-memory-theory.ipynb) | 4 | 4 |
| [03-language-model-foundations/01-sequence-memory/01B-sequence-memory-pytorch-lab.ipynb](03-language-model-foundations/01-sequence-memory/01B-sequence-memory-pytorch-lab.ipynb) | 4 | 4 |
| [03-language-model-foundations/01-sequence-memory/01C-cinematic-piano-memory.ipynb](03-language-model-foundations/01-sequence-memory/01C-cinematic-piano-memory.ipynb) | 4 | 4 |
| [03-language-model-foundations/02-tokenization-and-embeddings/02A-tokenization-and-embeddings-theory.ipynb](03-language-model-foundations/02-tokenization-and-embeddings/02A-tokenization-and-embeddings-theory.ipynb) | 4 | 4 |
| [03-language-model-foundations/02-tokenization-and-embeddings/02B-tokenization-and-embeddings-pytorch-lab.ipynb](03-language-model-foundations/02-tokenization-and-embeddings/02B-tokenization-and-embeddings-pytorch-lab.ipynb) | 4.5 | 4.5 |
| [03-language-model-foundations/03-language-models/03A-training-skeleton-and-word-objectives.ipynb](03-language-model-foundations/03-language-models/03A-training-skeleton-and-word-objectives.ipynb) | 3.5 | 3.5 |
| [03-language-model-foundations/03-language-models/03B-counts-and-fixed-window-language-models.ipynb](03-language-model-foundations/03-language-models/03B-counts-and-fixed-window-language-models.ipynb) | 4.5 | 4.5 |
| [03-language-model-foundations/03-language-models/03C-fixed-windows-to-recurrent-language-models.ipynb](03-language-model-foundations/03-language-models/03C-fixed-windows-to-recurrent-language-models.ipynb) | 4.5 | 4 |
| [03-language-model-foundations/03-language-models/03D-recurrent-encoder-decoder-to-attention.ipynb](03-language-model-foundations/03-language-models/03D-recurrent-encoder-decoder-to-attention.ipynb) | 4 | 4.5 |
| [03-language-model-foundations/optional/embedding-history/01-from-counts-to-static-geometry.ipynb](03-language-model-foundations/optional/embedding-history/01-from-counts-to-static-geometry.ipynb) | 3.5 | 3.5 |
| [04-transformers/01-token-embeddings-to-transformers.ipynb](04-transformers/01-token-embeddings-to-transformers.ipynb) | 4 | 4 |
| [04-transformers/02-attention-and-position.ipynb](04-transformers/02-attention-and-position.ipynb) | 4 | 4 |
| [04-transformers/03-complete-transformer-block.ipynb](04-transformers/03-complete-transformer-block.ipynb) | 4 | 4 |
| [04-transformers/04a-decoder-only-language-model-theory.ipynb](04-transformers/04a-decoder-only-language-model-theory.ipynb) | 4 | 4 |
| [04-transformers/04b-decoder-only-language-model-lab.ipynb](04-transformers/04b-decoder-only-language-model-lab.ipynb) | 3 | 4 |
| [04-transformers/05a-encoder-decoder-and-cross-attention-theory.ipynb](04-transformers/05a-encoder-decoder-and-cross-attention-theory.ipynb) | 4 | 4 |
| [04-transformers/05b-encoder-decoder-and-cross-attention-lab.ipynb](04-transformers/05b-encoder-decoder-and-cross-attention-lab.ipynb) | 3 | 4 |
| [04-transformers/06-modern-decoder-only-llm.ipynb](04-transformers/06-modern-decoder-only-llm.ipynb) | 3 | 4 |
| [04-transformers/focused/01-attention-position-and-qkv-focused.ipynb](04-transformers/focused/01-attention-position-and-qkv-focused.ipynb) | 3 | 4 |
| [04-transformers/focused/02-transformer-block-focused.ipynb](04-transformers/focused/02-transformer-block-focused.ipynb) | 3 | 4 |
| [04-transformers/focused/03-decoder-and-encoder-decoder-focused.ipynb](04-transformers/focused/03-decoder-and-encoder-decoder-focused.ipynb) | 3 | 3 |
| [05-llm-training/01-pretraining-data-pipeline.ipynb](05-llm-training/01-pretraining-data-pipeline.ipynb) | 4 | 4 |
| [05-llm-training/02-pretraining-theory.ipynb](05-llm-training/02-pretraining-theory.ipynb) | 4 | 4 |
| [05-llm-training/03-pretrain-a-base-model-lab.ipynb](05-llm-training/03-pretrain-a-base-model-lab.ipynb) | 3 | 4 |
| [05-llm-training/optional/full-model-audit/tiny-transformer-for-ai-engineers.ipynb](05-llm-training/optional/full-model-audit/tiny-transformer-for-ai-engineers.ipynb) | 3 | 4 |
| [05-llm-training/optional/lightning/training-a-base-model-with-lightning.ipynb](05-llm-training/optional/lightning/training-a-base-model-with-lightning.ipynb) | 3 | 4 |
| [06-llm-finetuning/01-llm-finetuning-data-techniques.ipynb](06-llm-finetuning/01-llm-finetuning-data-techniques.ipynb) | 3 | 4 |
| [06-llm-finetuning/02-llm-finetuning-parameter-techniques.ipynb](06-llm-finetuning/02-llm-finetuning-parameter-techniques.ipynb) | 3 | 3 |
| [06-llm-finetuning/03-llm-finetuning-comparison-and-decision.ipynb](06-llm-finetuning/03-llm-finetuning-comparison-and-decision.ipynb) | 3 | 4 |
| [06-llm-finetuning/04-llm-finetuning-practice.ipynb](06-llm-finetuning/04-llm-finetuning-practice.ipynb) | 2 | 3 |
| [07-rag/01a-hybrid-retrieval-theory.ipynb](07-rag/01a-hybrid-retrieval-theory.ipynb) | 4 | 4 |
| [07-rag/01b-hybrid-retrieval-lab.ipynb](07-rag/01b-hybrid-retrieval-lab.ipynb) | 3 | 4 |
| [07-rag/02-rag-failure-localization-and-evaluation.ipynb](07-rag/02-rag-failure-localization-and-evaluation.ipynb) | 4 | 4 |
| [08-llm-evaluation/01-llm-evaluation-metrics-and-benchmarks.ipynb](08-llm-evaluation/01-llm-evaluation-metrics-and-benchmarks.ipynb) | 3 | 4 |
| [08-llm-evaluation/02-llm-as-judge-safety-and-pipeline.ipynb](08-llm-evaluation/02-llm-as-judge-safety-and-pipeline.ipynb) | 4 | 4 |
| [08-llm-evaluation/03-hallucination-detection.ipynb](08-llm-evaluation/03-hallucination-detection.ipynb) | 3 | 4 |
| [08-llm-evaluation/04-calibration-and-confidence.ipynb](08-llm-evaluation/04-calibration-and-confidence.ipynb) | 4 | 4 |
| [09-llm-gateway/01-gateway-control-plane-theory.ipynb](09-llm-gateway/01-gateway-control-plane-theory.ipynb) | 4 | 4 |
| [09-llm-gateway/02-gateway-routing-resilience-lab.ipynb](09-llm-gateway/02-gateway-routing-resilience-lab.ipynb) | 3 | 3 |
| [10-agentic-ai/00-agent-foundations-and-tool-contracts/00-agent-foundations-and-tool-contracts.ipynb](10-agentic-ai/00-agent-foundations-and-tool-contracts/00-agent-foundations-and-tool-contracts.ipynb) | 3 | 4 |
| [10-agentic-ai/01-reasoning-planning-and-control/01-reasoning-planning-and-control.ipynb](10-agentic-ai/01-reasoning-planning-and-control/01-reasoning-planning-and-control.ipynb) | 3 | 4 |
| [10-agentic-ai/02-state-context-and-memory/02-state-context-and-memory.ipynb](10-agentic-ai/02-state-context-and-memory/02-state-context-and-memory.ipynb) | 3 | 3 |
| [10-agentic-ai/02a-skills-discovery-composition-and-versioning/02a-skills-discovery-composition-and-versioning.ipynb](10-agentic-ai/02a-skills-discovery-composition-and-versioning/02a-skills-discovery-composition-and-versioning.ipynb) | 3 | 4 |
| [10-agentic-ai/03-durable-workflows-with-langgraph/03-durable-workflows-with-langgraph.ipynb](10-agentic-ai/03-durable-workflows-with-langgraph/03-durable-workflows-with-langgraph.ipynb) | 3 | 4 |
| [10-agentic-ai/04-agentic-rag-and-self-correction/04-agentic-rag-and-self-correction.ipynb](10-agentic-ai/04-agentic-rag-and-self-correction/04-agentic-rag-and-self-correction.ipynb) | 3 | 4 |
| [10-agentic-ai/05-agent-evaluation-and-observability/05-agent-evaluation-and-observability.ipynb](10-agentic-ai/05-agent-evaluation-and-observability/05-agent-evaluation-and-observability.ipynb) | 3 | 4 |
| [10-agentic-ai/06-safety-human-control-and-governance/06-safety-human-control-and-governance.ipynb](10-agentic-ai/06-safety-human-control-and-governance/06-safety-human-control-and-governance.ipynb) | 3 | 4 |
| [10-agentic-ai/07-mcp-and-agent-interoperability/07-mcp-and-agent-interoperability.ipynb](10-agentic-ai/07-mcp-and-agent-interoperability/07-mcp-and-agent-interoperability.ipynb) | 3 | 3 |
| [10-agentic-ai/08-multi-agent-communication-and-coordination/08-multi-agent-communication-and-coordination.ipynb](10-agentic-ai/08-multi-agent-communication-and-coordination/08-multi-agent-communication-and-coordination.ipynb) | 3 | 4 |
| [10-agentic-ai/09-reliability-recovery-and-production-decisions/09-reliability-recovery-and-production-decisions.ipynb](10-agentic-ai/09-reliability-recovery-and-production-decisions/09-reliability-recovery-and-production-decisions.ipynb) | 3 | 4 |
| [11-ai-infrastructure/01-gpu-hardware/gpu-hardware-foundations.ipynb](11-ai-infrastructure/01-gpu-hardware/gpu-hardware-foundations.ipynb) | 3 | 4 |
| [11-ai-infrastructure/02-mixed-precision/mixed-precision-and-memory.ipynb](11-ai-infrastructure/02-mixed-precision/mixed-precision-and-memory.ipynb) | 3 | 3 |
| [11-ai-infrastructure/03-profiling/pytorch-profiling.ipynb](11-ai-infrastructure/03-profiling/pytorch-profiling.ipynb) | 3 | 4 |
| [11-ai-infrastructure/04-flash-attention/flash-attention-internals.ipynb](11-ai-infrastructure/04-flash-attention/flash-attention-internals.ipynb) | 3 | 3 |
| [11-ai-infrastructure/05-distributed-training/distributed-training.ipynb](11-ai-infrastructure/05-distributed-training/distributed-training.ipynb) | 2 | 3 |
| [11-ai-infrastructure/06-quantization/quantization-in-depth.ipynb](11-ai-infrastructure/06-quantization/quantization-in-depth.ipynb) | 2 | 3 |
| [11-ai-infrastructure/07-inference-systems/inference-systems.ipynb](11-ai-infrastructure/07-inference-systems/inference-systems.ipynb) | 3 | 3 |
| [11-ai-infrastructure/08-triton-kernels/triton-kernels.ipynb](11-ai-infrastructure/08-triton-kernels/triton-kernels.ipynb) | 2 | 4 |
| [11-ai-infrastructure/09-systems-composition/systems-composition-lab.ipynb](11-ai-infrastructure/09-systems-composition/systems-composition-lab.ipynb) | 4 | 4 |
| [11-ai-infrastructure/10-azure-operational-llm-serving/azure-operational-llm-serving.ipynb](11-ai-infrastructure/10-azure-operational-llm-serving/azure-operational-llm-serving.ipynb) | 3 | 4 |
| [12-role-based-tracks/ai-engineer/01-training-data-quality-and-lineage/training-data-quality-and-lineage.ipynb](12-role-based-tracks/ai-engineer/01-training-data-quality-and-lineage/training-data-quality-and-lineage.ipynb) | 3 | 4 |
| [12-role-based-tracks/ai-engineer/02-prompt-release-and-experimentation/prompt-release-and-experimentation.ipynb](12-role-based-tracks/ai-engineer/02-prompt-release-and-experimentation/prompt-release-and-experimentation.ipynb) | 3 | 5 |
| [12-role-based-tracks/ai-engineer/03-application-latency-and-cost/application-latency-and-cost.ipynb](12-role-based-tracks/ai-engineer/03-application-latency-and-cost/application-latency-and-cost.ipynb) | 3 | 4 |
| [12-role-based-tracks/ai-engineer/04-release-registry-and-lineage/release-registry-and-lineage.ipynb](12-role-based-tracks/ai-engineer/04-release-registry-and-lineage/release-registry-and-lineage.ipynb) | 3 | 4 |
| [12-role-based-tracks/ai-engineer/05-production-feedback-and-drift/production-feedback-and-drift.ipynb](12-role-based-tracks/ai-engineer/05-production-feedback-and-drift/production-feedback-and-drift.ipynb) | 3 | 4 |
| [12-role-based-tracks/ai-engineer/06-capstone/capstone-evidence-workbook.ipynb](12-role-based-tracks/ai-engineer/06-capstone/capstone-evidence-workbook.ipynb) | 2 | 4 |
| [12-role-based-tracks/fde/01-discovery-and-success-criteria/discovery-and-success-criteria.ipynb](12-role-based-tracks/fde/01-discovery-and-success-criteria/discovery-and-success-criteria.ipynb) | 4 | 4 |
| [12-role-based-tracks/fde/02-architecture-translation/architecture-translation.ipynb](12-role-based-tracks/fde/02-architecture-translation/architecture-translation.ipynb) | 3 | 4 |
| [12-role-based-tracks/fde/03-data-onboarding-and-contracts/data-onboarding-and-contracts.ipynb](12-role-based-tracks/fde/03-data-onboarding-and-contracts/data-onboarding-and-contracts.ipynb) | 3 | 4 |
| [12-role-based-tracks/fde/04-identity-isolation-and-compliance/identity-isolation-and-compliance.ipynb](12-role-based-tracks/fde/04-identity-isolation-and-compliance/identity-isolation-and-compliance.ipynb) | 3 | 4 |
| [12-role-based-tracks/fde/05-sla-capacity-and-commercials/sla-capacity-and-commercials.ipynb](12-role-based-tracks/fde/05-sla-capacity-and-commercials/sla-capacity-and-commercials.ipynb) | 3 | 4 |
| [12-role-based-tracks/fde/06-rollout-and-change-management/rollout-and-change-management.ipynb](12-role-based-tracks/fde/06-rollout-and-change-management/rollout-and-change-management.ipynb) | 4 | 4 |
| [12-role-based-tracks/fde/07-incident-response-and-recovery/incident-response-and-recovery.ipynb](12-role-based-tracks/fde/07-incident-response-and-recovery/incident-response-and-recovery.ipynb) | 3 | 4 |
| [12-role-based-tracks/fde/08-handoff-and-customer-success/handoff-and-customer-success.ipynb](12-role-based-tracks/fde/08-handoff-and-customer-success/handoff-and-customer-success.ipynb) | 3 | 3 |
| [12-role-based-tracks/fde/09-capstone/capstone-workbook.ipynb](12-role-based-tracks/fde/09-capstone/capstone-workbook.ipynb) | 2 | 4 |

A vendored dependency notebook under `12-role-based-tracks/fde/.venv/` was found and excluded; it is not curriculum.

## 6. Full stage reports

The reviewers' unedited findings follow. Paths are relative to each stage directory unless shown in full.

### 01-ml-foundations

#### Stage summary
- **Overall flow: 4/5.** The required route is coherent: motion and uncertainty → honest ML evidence → representational limits and gradient responsibility.
- One free-kick example carries the math chapter; SmartVal gives the following chapters a shared business setting. Each required notebook names a clear boundary and next handoff.
- The additions generally bridge concepts rather than just listing them. The physics-to-matrix-to-AI bridge is carefully explained; the optional synthesizer-to-sequence-model bridge is relevant but extends beyond the required route.
- Two evidence-contract mismatches need correction: the ML notebook exposes test results before choices are complete, and the ReLU intervention reports a stale upstream gradient.
- No `-theory.md` companions were present in this scope. Chapter and stage READMEs/CONTENTS maps agree on the required route and optional capstone.

#### Per-notebook findings

##### 01-math-foundations/math-foundations-for-ml.ipynb — novice score 4/5, flow score 4/5
- **Covers:** Vectors/dot products, local change, accumulated motion, matrix updates, and probability through one gravity-only free kick.
- **Works well:** Everyday questions precede most terms; the update equations follow a concrete step; 28 exact-acceleration updates are compared with a direct endpoint; the uncertainty section distinguishes one aim from a distribution and states its Gaussian assumption. The next notebook reuses the math vocabulary.
- **Issues:**
  - [P2] **Part 1 — Dot Product as Useful Power Transfer:** “useful speed along the target line” could imply a causal contribution to the ball’s path. It is a geometric projection/alignment score; gravity and the upward component still determine whether the kick reaches the target. Keep that distinction explicit at the point the 19.2 m/s figure is interpreted.
  - [P2] **Part 3 — From a Physics Matrix to an AI Layer:** “normalized [angle, speed]” is not glossed, and the example’s arbitrary input/output units are only clarified later. Explain normalization plainly before the code and say this is a matrix analogy, not a trained layer.
  - [P3] **Integration as Stitching Movie Frames:** the 4/8/16/28 comparison is load-bearing in the GIF, while the notebook code demonstrates only 28 steps. Add a small static, code-generated comparison/storyboard so the idea survives when animation does not render. Existing trajectory/projection assets are present; no additional generated image is needed.

##### 02-ml-basics/ml-basics.ipynb — novice score 4/5, flow score 4/5
- **Covers:** SmartVal train/validation/test roles, regression loss versus metrics, probability-to-decision thresholds, and capacity/generalization.
- **Works well:** It defines the data roles before the experiment, checks row-ID disjointness, changes one main variable at a time, and closes by handing XOR’s linear failure to the next chapter. The bridge from physical math to fitted models is explicit.
- **Issues:**
  - [P1] **§2 — regression fit/report cell (“Fit one compact regression implementation…”):** it predicts on `X_test` and prints test MAE before validation has selected degree/threshold. That contradicts “test rows are the final exam after those choices” and the scorecard’s claim that test is used after choices. Keep test untouched until the final report (or clearly define a separate, frozen demonstration model).
  - [P2] **Scorecard and coverage ledger:** “learning-rate intuition as update-size control” is listed as explained, but the notebook does not teach learning rate. Remove that ledger entry or add a brief prerequisite bridge; likewise distinguish the measured MAE/MSE contribution example from evidence that the metrics actually rank competing models differently.

##### 03-neural-networks-and-backprop/01-smartval-neural-networks-and-backprop.ipynb — novice score 3/5, flow score 4/5
- **Covers:** XOR’s linear-boundary failure, a hand-built ReLU representation, manual gradient responsibility, finite-difference checking, and boundaries for common network tools.
- **Works well:** The failure → hidden-feature fix → gradient question chain is strong. It checks the manual gradient independently and honestly says the XOR weights were hand-set, not learned. The PyTorch handoff matches the README.
- **Issues:**
  - [P1] **Your turn: close one ReLU gate:** after changing `changed_z`, the code multiplies the new gate by `dh1` from the *unmodified* forward pass. Recompute the perturbed forward loss and upstream derivative before reporting the changed gradient; compare baseline and perturbed values so the intervention measures the claim it makes.
  - [P2] **Opening / §1 — Attempt 1: one straight boundary:** the notebook calls the four XOR cases “SmartVal’s next decision” after the rental-regression/classification chapter, but never explains why the business example is being replaced. Add one sentence that makes XOR a deliberately tiny stress test of the same decision-boundary limitation.
  - [P3] **Attempt 1 code print:** “affine score” is unexplained jargon; say “weighted sum plus an offset,” or omit the term.

##### 03-neural-networks-and-backprop/optional/02-melodyne-backprop-synthesizer.ipynb — novice score 3/5, flow score 4/5
- **Covers:** A fixed phrase rendered through four bounded controls; TensorFlow gradient attribution, finite-difference audit, optimization, and audio/spectrum recovery checks.
- **Works well:** The target controls are held out from the loss, all four gradients are checked, and recovery uses independent measures rather than loss alone. The `Image pending` mechanism diagram has a matching pending entry in `optional/images/PROMPTS.md`.
- **Issues:**
  - [P2] **§3 — Loss, Backpropagation, and the Update:** the equation appears before the plain-language account and introduces several unfamiliar symbols at once. Lead with the two observable comparisons (waveform and frequency balance), then show the compact expression as optional notation.
  - [P2] **Opening / §2 — Build the Differentiable Forward Pass:** “differentiable renderer” and FFT/sigmoid terminology arrive in the initial brief/overview before the glossary. Define the core idea in ordinary words first; the later glossary is accurate but compressed.

#### Cross-notebook inconsistencies (within and across stages)
- The Math → ML handoff is strong: the ML opener explicitly recalls vectors, derivatives, and repeated updates. ML → neural keeps the SmartVal name but changes from rental listings to four binary XOR cases without a bridge. Neural → Stage 02 is explicit and consistent: hand-computed responsibility becomes `loss.backward()`.
- The “SmartVal” input contract is not continuous: chapter 02 has three listing features, while chapter 03 has two synthetic signals. This is pedagogically reasonable for visibility, but the transition needs to say so.
- The ML notebook’s test-timing contradiction conflicts with the stage-level promise of fair evidence. Its coverage ledger also overstates learning-rate coverage.
- Maps and READMEs correctly list Math → ML Contracts → Neural Networks, and show the synthesizer as optional. The optional branch returns readers to the PyTorch route in its README; its notebook’s future RNN/LSTM bridge is an enrichment rather than a required handoff.

#### Top 5 prioritized improvements
1. **[P1]** Defer all test-set prediction/reporting in `02-ml-basics/ml-basics.ipynb` until model and threshold choices are finalized; make the scorecard match actual use.
2. **[P1]** Recompute the forward and backward pass in **Your turn: close one ReLU gate**; report measured before/after gradients.
3. **[P2]** Add the explicit rental-task → synthetic-XOR rationale in the opening of `01-smartval-neural-networks-and-backprop.ipynb`.
4. **[P2]** Correct the ML notebook’s coverage ledger and either demonstrate metric-ranking divergence or narrow its claim to the measured contribution comparison.
5. **[P2]** In **Part 3 — A Matrix as a State-Moving Machine**, gloss “normalized” and label the AI-layer example as an analogy; add a code-generated static storyboard alongside the integration animation.

---

### Pedagogy review: PyTorch Foundations

#### PyTorch Foundations

##### Stage summary
- **Overall flow: 4/5.** The required route progresses from a complete loop to named tensor contracts, explicit module ownership, data/checkpoint policy, and finally validation/model-selection lifecycle.
- Measured failures and short runnable interventions generally make concepts feel needed; theory notes preserve the core mental models and honest evidence boundaries.
- Recent language-model bridges are mostly explicit and restrained, but the early tensors chapter front-loads too many future-stage terms, and some bridge prose repeats concepts rather than resuming an unresolved question.
- The route's examples stay in a sensor/fuel domain until optional branches; dataset changes are reasonable but not always explicitly framed as the same problem versus a new demonstration.
- **Images:** no missing image is essential to the core math; the shape traces and small diagrams suffice. A single static visual contrasting model weights, optimizer memory, and experiment/progress state could help in Chapters 03–05, but the state tables already cover most of that job.

##### Per-notebook findings

###### 02-pytorch-foundations/01-deep-learning-map/01-deep-learning-map.ipynb — novice score 4/5, flow score 4/5
- Covers: one nonlinear regression from data through backward/update to inference.
- Works well: the `One update separates responsibilities` evidence distinguishes `.backward()` from `optimizer.step()`; measured loss, curve, and inference make a coherent end-to-end first pass.
- Issues: [P2] `Your turn: change only the epoch budget` prints only the trial loss. The README promises measuring original and trial loss; also print/compare the 300-epoch baseline so the one-variable intervention is self-contained.

###### 02-pytorch-foundations/02-tensors-and-autograd/02-tensors-and-autograd.ipynb — novice score 4/5, flow score 4/5
- Covers: named axes, indexing, reshape/transpose, broadcasting, contraction, gradient accumulation, and a language-shaped graph.
- Works well: every major operation gets a visible failure or prediction, shape/value evidence, and the toy embedding-to-logits bridge states its boundary honestly.
- Issues: [P2] The opening bridge before the title defines token, ID, embedding, hidden state, logit, and head in one paragraph. Move this glossary to `Language-model-shaped evidence` and keep only the learning gap in the opening; terms currently arrive before they are useful. [P3] `Autograd records influence, not intent` uses “chain rule” as a new formal term; gloss it in plain language beside the heading or retain the manual-derivative comparison without naming it.

###### 02-pytorch-foundations/03-models-and-training-loops/03-models-and-training-loops.ipynb — novice score 4/5, flow score 4/5
- Covers: raw tensor learning, `nn.Module`, shape contracts, optimizer state, numerical parity, and a saved model artifact.
- Works well: the same initialization/data/update count makes raw-tensor versus module parity falsifiable; the Part 04 fresh-kernel handoff is concrete.
- Issues: [P2] `Your turn: change only the learning rate` reports the trial trace but not the matched baseline trace or a direct comparison. Print both for the advertised controlled comparison. [P3] The toy switches from the prior generic sensor curve to engine-load/temperature “fuel” measurements without a sentence saying this is a new example chosen to make ownership and persistence concrete.

###### 02-pytorch-foundations/04-data-checkpoints-and-hyperparameters/04-data-checkpoints-and-hyperparameters.ipynb — novice score 4/5, flow score 4/5
- Covers: manual batches, Dataset/DataLoader, inference reload, checkpoint contents, and controlled learning-rate comparisons.
- Works well: the fresh-kernel dependency fails loudly, batch coverage is measured, and prediction parity distinguishes inference state from training continuation.
- Issues: [P2] `A state dictionary is necessary but not sufficient` and the chapter scorecard call the artifact “resumable” before the notebook’s closing evidence-limit cell clarifies that it only writes a bundle and never verifies the next update. State “resume-layout example” at first use and link the limitation there. [P3] The mini-batch training cell trains and saves state without a held-out check; clarify this section is about state layout, not model quality (the later lifecycle chapter supplies that boundary).

###### 02-pytorch-foundations/05-modern-training-lifecycle/05-modern-training-lifecycle.ipynb — novice score 3/5, flow score 4/5
- Covers: leakage-safe preprocessing, train/eval behavior, scheduler/AMP boundary, early stopping, best-model inference, and resume-state layout.
- Works well: validation is separated from updates; uneven batch means are weighted correctly; the ending is unusually clear that key presence does not prove exact recovery.
- Issues: [P1] `Save resumable state separately from the best inference state` omits the AMP scaler, RNG/dropout state, sampler order, and preprocessing state, while calling the result resumable. The later caveat is accurate but arrives after the claim; rename it “resume-state layout example” or save/restore the missing state and test an interrupted-versus-uninterrupted next update. [P2] The lifecycle packs normalization, mode behavior, AMP, scheduler, stopping, checkpoints, and inverse scaling into one notebook; add a brief “must know now / optional performance tool” signpost before AMP so it does not compete with the central validation story. [P3] An image of the two checkpoint jobs could help, but is not necessary given the Mermaid diagram.

###### 02-pytorch-foundations/quick-refresher/pytorch-training-skeleton.ipynb — novice score 3/5, flow score 3/5
- Covers: the core classifier loop, batch/loss contracts, a CBOW shape mapping, readiness checks, and copy/paste skeleton.
- Works well: it clearly labels shortcut audience and excluded lifecycle topics; the readiness table points back to specific chapters.
- Issues: [P1] In `The same skeleton becomes CBOW`, “CBOW [is] the first compact language-shaped module in the track” conflicts with Stage 03’s required Sequence Memory/text lab and the Stage 03 preview route; change to “first compact feed-forward/static-embedding language objective” or equivalent. [P2] The opening calls the full route Parts 00–04, while current Stage 02 contains Parts 01–05; the later note explains old numbering, but move the mapping up front. [P2] `fit coordinates them` is ambiguous because the preceding paragraph says PyTorch does not require `fit`; say “your `fit` function” to separate a user-written helper from Keras’s API.

###### 02-pytorch-foundations/optional/framework-translation/01-keras-to-pytorch-antarctic-field-guide.ipynb — novice score 3/5, flow score 3/5
- Covers: framework vocabulary, data/dtype, dense-weight parity, logits/loss, gradient lifecycle, evaluation, and penguin-classifier ablations.
- Works well: parity and wrong-logits experiments prove framework-specific contracts; the test-set caveat honestly identifies ablation-driven leakage.
- Issues: [P1] In `Antarctic Field Guide model`, the three-species model returns `(batch, 3)`, but its code comment says `(batch, 2)`; the following `What just happened` says the final layer emits two scores. Clarify that two scores belong only to the separate fixed-logits demonstration, and correct the model comment/prose to three. [P2] `Part 6 - Controlled Project Decisions` uses `X_test` for width and feature choices; the warning is good but comes late. Rename it validation data or split a final untouched test set before ablations. [P3] Several Keras reference sections assume familiarity with Keras and jargon-heavy framework terms; preserve their optional status and define terms at first use as needed.

###### 02-pytorch-foundations/optional/vision/01-pytorch-cnns.ipynb — novice score 4/5, flow score 4/5
- Covers: shared filters, convolution, stride/pooling, trained feature maps, receptive field, and residual gradient paths.
- Works well: the prose defines kernel/feature map before code and bounds both receptive-field and residual claims; the manually computed convolution pixel is strong proof.
- Issues: [P2] `Train filters, then inspect their specialization` draws trained filters but does not show a before/after or quantify specialization; compare a fixed Sobel detector with a learned response or frame the plot as inspection, not evidence that filters specialized.

###### 02-pytorch-foundations/optional/vision/02-pytorch-autoencoders.ipynb — novice score 4/5, flow score 4/5
- Covers: convolutional encoder/decoder, reconstruction, latent inspection, compression ratio, and a controlled width comparison.
- Works well: names the VAE boundary explicitly; original/rebuilt image panels and the width-versus-error experiment turn compression into visible evidence.
- Issues: [P2] The class-center distances are printed without explaining what a large/small distance would mean or showing whether class codes separate; gloss this as an exploratory diagnostic, not evidence that the latent has class concepts.

##### Cross-notebook inconsistencies (within and across stages)
- Stage 02's main route is clearly numbered 01–05, but the shortcut still anchors its map to historical Parts 00–04; its own later note partially repairs this, and the first-use wording should be reconciled.
- Refresher's “CBOW is first” statement conflicts with Stage 03's earlier sequence-memory text lab and preview-before-formalization route.
- Checkpoint terminology varies: Chapter 04 and Chapter 05 use “resumable” for bundles that are not tested for exact continuation; both theory companions give the more accurate boundary. Put the same qualification at first use in notebooks and chapter READMEs.
- Main route terminology shifts from the first notebook's generic “sensor/calibrated value” to fuel/engine-load examples in Part 03 onward. This is a modest continuity break, not a technical contradiction; name the transition.
- Optional vision/framework examples are deliberately distinct; their READMEs correctly mark them optional and do not claim they are prerequisites for the Stage 03 language handoff. Contents maps generally match the route.

##### Top 5 prioritized improvements
1. **P1 — Fix factual claims:** correct the three-class/two-class mismatch in framework translation; qualify Chapter 05's “resumable” label before the code; correct the CBOW-first claim in the refresher.
2. **P2 — Complete interventions:** show baseline and trial values for the epoch-budget and learning-rate drills in Chapters 01 and 03.
3. **P2 — Make the shortcut map current:** replace Parts 00–04 with Parts 01–05 and explain any legacy numbering once at the opening.
4. **P2 — Delay bridge terminology until useful:** move the six-term language glossary in Chapter 02 to the LM-shaped bridge; state the fuel-example transition in Chapter 03.
5. **P2 — Keep evidence claims aligned:** surface Chapter 04/05 checkpoint limits at first claim; make framework ablations use validation data or a final untouched test split.

---

### Stage 03 — Language-Model Foundations

#### Stage summary
- **Overall flow: 4/5.** The recommended route moves from recurrent memory and its text-input previews, through token/embedding ownership, to fixed-window and recurrent language prediction, then to the separate reader/writer task and Stage 04.
- Historical order is explicitly not the organizing claim: the new track calls itself a complaint route, marks LSTM as a side refinement, and notes that attention predates Transformers.
- Bridges are mostly explicit and additive. 01B previews token IDs, shifting and padding before Chapter 02 formally owns them; 02C introduces the reader/writer task; evolution Notebook 3 deepens it; Stage 04 says what is reused and what is new.
- The main weakness is density: 02C combines three objectives and a training abstraction in a very long notebook; the optional history notebook has too little plain-language interpretation around sophisticated code.
- No material contradiction found among the new route, 02C, the Stage 04 README and Part 5A. The new Part 5A lead-in explicitly distinguishes its mean-pooling diagnostic from a learned recurrent summary. Images are still pending, but text/plots make the core lessons readable without them.

#### Per-notebook findings

##### `03-language-model-foundations/01-sequence-memory/01A-sequence-memory-theory.ipynb` — novice score 4/5, flow score 4/5
- **Covers:** order, a reused state, fading teaching signals, clipping’s limit, and LSTM gates. The progression from “Order creates a new axis” to “The notecard can forget” to “LSTM creates a controlled memory route” is intuitive and measured.
- **Issue [P2]:** Under “LSTM creates a controlled memory route,” gate values are hand-set to illustrate state arithmetic, not learned by fitting an LSTM. Label that contrast immediately and state that the experiment proves a possible route, not trained-model performance. BPTT and gradient terminology are introduced, but explained in context.

##### `03-language-model-foundations/01-sequence-memory/01B-sequence-memory-pytorch-lab.ipynb` — novice score 4/5, flow score 4/5
- **Covers:** temporal splits, PyTorch batching, packed LSTM states, shifted token targets, loss masking, causal invariance and clipping order. The “Preview checkpoint” clearly admits that token IDs and masks are used before their formal chapter.
- **Issue [P2]:** The notebook changes from continuous forecasting to token prediction and then adds multiple mask/packing contracts. Its preview table and first-use prose help, but a compact close-of-forecast bridge naming why the input task must change would ease the transition. No false claim that packing and loss masking are interchangeable was found.

##### `03-language-model-foundations/01-sequence-memory/01C-cinematic-piano-memory.ipynb` — novice score 4/5, flow score 4/5
- **Covers:** a recurring piano sequence with one opening pitch cue, matched RNN/LSTM training, recorded audio checkpoints, gradient path, and a learner-controlled width trade-off. The audible and numeric evidence continues the sequence-memory question rather than starting an unrelated API tour.
- **Issue [P2]:** “Match everything except the memory cell” is a useful controlled comparison, but the gate families have different parameter counts even at equal hidden width. Keep the reported parameter cost close to the comparison and avoid letting “matched” imply equal capacity. The following text-model transition is explicit.

##### `03-language-model-foundations/02-tokenization-and-embeddings/02A-tokenization-and-embeddings-theory.ipynb` — novice score 4/5, flow score 4/5
- **Covers:** character/word trade-offs, visible BPE merges, token IDs as addresses, static versus contextual representations, and the padding-loss failure. The BPE animation has a static storyboard counterpart; the legal example persists.
- **Issue [P2]:** “Static lookup versus contextualized token representation” uses a deliberately constructed neighbor-mixing toy, not a trained contextual model. The boundary sentence is present; repeat it beside the output so a novice does not read the toy distance as evidence of learned contextual semantics. Correctly places tokenization before model-specific use.

##### `03-language-model-foundations/02-tokenization-and-embeddings/02B-tokenization-and-embeddings-pytorch-lab.ipynb` — novice score 4.5/5, flow score 4.5/5
- **Covers:** readable toy BPE, GPT-2 byte/token inspection and round-trip, embedding-row gradients, padding masks and `ignore_index` parity. “A compact BPE you can read in one minute” leads to the real tokenizer only after the toy mechanism.
- **Issues:** No substantive correctness or ordering concern found. The merge-budget experiment is a meaningful one-variable test; output is explicitly scoped to the small legal corpus. Good handoff from 02A and into Transformer-specific contextual routing.

##### `03-language-model-foundations/03-language-models/03A-training-skeleton-and-word-objectives.ipynb` — novice score 3.5/5, flow score 3.5/5
- **Covers:** CBOW, Skip-gram, a reusable loader/fit contract, then a GRU encoder-decoder with aligned sequence targets, teacher forcing, and greedy generation. It carefully distinguishes objectives, tensor shapes, static rows and the one-state source bottleneck.
- **Issues [P1]:** The opening cell is a `[B, C, D, V, S, T]` shape table before the notebook title and conversational setup; this reverses the plain-intuition-first rule. The 50-cell route also asks readers to traverse vocabulary construction, loading, two Word2Vec objectives, embedding geometry/PCA, and seq2seq. Add an English-first “three different questions” orientation before the shape map and mark the encoder-decoder portion as the relevant bridge for readers approaching Notebook 3. This is an additive fix, not a reason to remove the existing material.

##### `03-language-model-foundations/03-language-models/03B-counts-and-fixed-window-language-models.ipynb` — novice score 4.5/5, flow score 4.5/5
- **Covers:** next-word counts, ordered learned windows, then an identical-input/different-answer proof of the fixed boundary. The `red/blue … rest calmly now` construction makes the impossibility structural, not merely a low benchmark score.
- **Issue [P2]:** The notebook introduces IDs and embedding rows inside “Give each position its own seat” before its code; that is a reasonable first use after Chapter 02, but one sentence distinguishing this next-token model from the later CBOW/Skip-gram objectives would clarify why this is a language model, not another static-vector lesson. The scope ledger and `visible_words` caveat are strong.

##### `03-language-model-foundations/03-language-models/03C-fixed-windows-to-recurrent-language-models.ipynb` — novice score 4.5/5, flow score 4/5
- **Covers:** reuses exactly the eight color-clue sentences, compares a three-token window with a running RNN state, records clue-sensitive state distances, then reconnects to LSTM/GRU memory without reteaching gates.
- **Issue [P2]:** The fixed-window/RNN comparison has unequal parameter counts and differs in input access and update budget; the notebook discloses this, but put that caveat directly beside the accuracy comparison as well as in the prose. Two held-out sentences are explicitly called a tiny task, not a benchmark. Its conclusion correctly separates state difference from useful recall.

##### `03-language-model-foundations/03-language-models/03D-recurrent-encoder-decoder-to-attention.ipynb` — novice score 4/5, flow score 4.5/5
- **Covers:** an explicit task shift to integer reversal; teacher-forced versus free-running inputs; learned recurrent summary versus per-step attention over source states; per-length held-out scores and a diagnostic routing map. The transition from text is explained, and “attention” remains clearly recurrent here.
- **Issue [P2]:** The first implementation calls GRU state a query and uses scaled dot products/softmax; definitions are present, but this is the densest novel jargon in the route. Add one concrete sentence before those names describing the writer’s need to “ask each source note which detail helps this digit.” Model capacity and non-universal results are honestly disclosed.

##### `03-language-model-foundations/optional/embedding-history/01-from-counts-to-static-geometry.ipynb` — novice score 3.5/5, flow score 3.5/5
- **Covers:** one-hot/BOW, local Skip-gram, GloVe-style global co-occurrence fitting, static neighborhoods/analogy-style arithmetic, and the `bank` contextual boundary. It is correctly marked optional and outside the attention prerequisite route.
- **Issue [P1]:** Sections “Word2Vec learns from local prediction” and “GloVe learns from global co-occurrence ratios” move quickly from headings into nontrivial objectives/code with little plain-language setup, prediction prompt, or interpretation of measured neighbors. Add a short intuitive question and what the code will demonstrate before each implementation; define the weighted co-occurrence fit as “predicting how often words meet,” not merely “factorization.”

#### Cross-notebook inconsistencies

- **Encoder-decoder duplication is purposeful but needs an explicit layer label.** 02C teaches a legal-text continuation with a single recurrent state and teacher forcing. Evolution Notebook 3 revisits the reader/writer interface on reversal and compares that fixed summary with recurrent attention. Stage 03 README and Notebook 3 explain the progression; Stage 04 Part 5A explicitly says its mean-pooling test is not a historical learned recurrent summary. Add a one-line “first introduction / measured comparison / Transformer replacement” map in 02C and Notebook 3 to prevent rereading as contradiction.
- **Teacher forcing is consistent:** 02C describes the correct previous target during practice; Notebook 3 states it is a training method, not an architecture requirement; Stage 04 Part 5A then reuses the rule with separate fixed source memory and a growing target prefix. The plain-English gloss precedes the term in the new track, though 02C could surface it sooner.
- **Route and contents maps match at the directory level.** Stage 03 `CONTENTS.md` lists its three route groups; the README labels the core two chapters separately from the recommended model-evolution route and optional branch. The Chapter 02 and model-evolution maps point to the same notebook/theory filenames and correctly identify the optional embedding-history branch. `PROMPTS.md` entries mirror pending placeholders; no missing-artwork link is required for the text to make sense.
- **Running examples change, with reasons:** piano notes establish recurrence, legal text motivates token work, color-clue sentences preserve continuity for fixed-window/RNN, and integer reversal isolates reader/writer addresses. The final two changes are explicitly justified; this is manageable, though the optional `bank` corpus adds another domain only for its own optional lesson.
- **Handoff is sound:** Stage 04 README calls Stage 03 reader/writer roles familiar and says Part 5 opens their internals; Part 5A links both 02C and evolution Notebook 3 and distinguishes the mean-pooling experiment from learned recurrent compression. It does not imply encoder-decoder equals Transformer.

#### Top 5 prioritized improvements

1. **[P1] 02C opening:** precede its initial shape table with plain English explaining the three tasks before introducing symbols; add a small “focus here for Notebook 3” cue and orientation to the 50-cell scope.
2. **[P1] Optional embedding-history notebook, Sections 2–3:** add intuitive setup and measured-result interpretation before Word2Vec/GloVe code; keep the branch optional, but apply the same novice standard.
3. **[P2] 02C → model-evolution Notebook 3 → Stage 04 Part 5A:** add a compact ownership/handoff distinction: first recurrent seq2seq example, then recurrent look-back comparison, then Transformer cross-attention. The existing links and Part 5A caveat make this a small clarification, not a rewrite.
4. **[P2] Sequence-memory 01A and 01C comparison cells:** label synthetic gate trajectories as illustrative and keep parameter-count differences adjacent to “matched” model comparisons.
5. **[P3] `03-language-model-foundations/README.md` and `images/PROMPTS.md`:** when imagery is created, render the first-use map and six model-evolution diagrams (especially the source-note/attention and practice-vs-generation visuals). Pending prompts are complete and no image is required for comprehension, so this is polish rather than a release blocker.

*Audit method: parsed notebook JSON and read extracted cell sources; no notebook execution or repository edits were performed.*

---

### Stage 04 — Transformers

#### Stage summary
- **Overall flow: 4/5.** The canonical track remains a strong one-stop shop: text identity → context routing and position → complete block → future-blind generation → separate-source writing → modern block refinements.
- The additions mostly clarify why a section exists without displacing the original complaint/evidence chain. The Q/K/V and real-RoPE bridge explicitly labels itself a preview, preserving the later full treatment.
- The examples change from `the cat sat on the mat` to `aria heard the signal` and then `[3, 1, 4, 1]`; these are distinct diagnostics, and the README/bridge text mostly explains why.
- No material technical contradiction was found in the block-family order, Q/K-only RoPE, mask contracts, or separate encoder/cross/self-attention cache lifetimes. Static local-link check found no broken links. No notebooks were executed, per scope.
- Three conceptual assets in `images/PROMPTS.md` remain pending (reader map, block-family comparison, real RoPE path); current text placeholders preserve the lesson without them.

#### Per-notebook findings

##### `01-token-embeddings-to-transformers.ipynb` — novice 4/5, flow 4/5
- **Covers / works well:** “The Challenge: Models Cannot Read Strings” begins with an accessible need; “Tokenization,” “Embeddings,” and “Prediction Error” move from stable IDs to learned rows and gradients. “The Ordering Failure” closes with a measured reason for Part 2. The `cat`/`mat` running example is consistent.
- **Issue [P2]:** “A real corpus can supply the lesson; it cannot restore discarded order” adds CBOW after the embedding update, but the shift from next-token supervision to center-word prediction can still feel like a side route. Add one short sentence at “Where did the prediction lesson come from?” distinguishing the objective from the Transformer’s later next-token objective.

##### `02-attention-and-position.ipynb` — novice 4/5, flow 4/5
- **Covers / works well:** “Minimal Attention: Score, Normalize, Retrieve” gives an inspectable retrieval before the position failure; “Add Position Before Attention” and “The complaint points to RoPE” motivate the alternative without claiming sinusoidal position fails. “Before the dials: where the question and label come from” explains Q/K/V in plain language and correctly says values are not rotated.
- **Issue [P2]:** The bridge’s full path introduces head split, masks, softmax and weighted values immediately before the dial preview, before Q/K/V are fully built in Section 4. The explicit preview signpost helps, but a one-line shape trace next to the pending `04-rope-in-real-path.png` would make the order easier to retain. The three pending image prompts are accurately marked, not broken embeds.

##### `03-complete-transformer-block.ipynb` — novice 4/5, flow 4/5
- **Covers / works well:** “The Challenge: Routing Is Not the Whole Computation” follows Part 2 naturally. Separate experiments show why multiple heads can keep incompatible routes, why FFNs add private per-token processing, and how residuals preserve a gradient route; “From Block Outputs to Predictions” reconnects the block to logits and loss.
- **Issue [P2]:** The block-families comparison is in the README rather than adjacent to “From Block Outputs to Predictions.” Add a brief forward pointer there so encoder/decoder variants do not seem like a new block definition in later chapters. The README’s pending comparison image would help.

##### `04a-decoder-only-language-model-theory.ipynb` — novice 4/5, flow 4/5
- **Covers / works well:** Opening plain-English `aria heard` setup explains causal visibility before the term becomes central. “The dishonest shortcut,” the three distinct mask contracts, shifted lessons, serial generation, cache cost, and temperature/top-k have clear boundaries. Cache parity is correctly assigned to Part 4B.
- **Issue [P2]:** The notebook distinguishes the causal score mask, padding-key mask, and ignored target, but “padding key mask” is still compressed jargon for a first-time reader. Add a plain phrase (“hide blank batch slots from attention”) beside its table row.

##### `04b-decoder-only-language-model-lab.ipynb` — novice 3/5, flow 4/5
- **Covers / works well:** Natural-length examples, collation, target alignment, loss flattening, generation stopping limits, and full-prefix/cache parity form a sensible implementation sequence. Walkthroughs preserve the three mask jobs rather than conflating them.
- **Issue [P2]:** Dataset/collator, cache-aware attention and the training loop are dense for a novice, even with the Part 4A bridge. Add a small end-to-end shape trace at “Flatten only the example and position axes,” using one batch row to show which target each flattened logit row is graded against.

##### `05a-encoder-decoder-and-cross-attention-theory.ipynb` — novice 4/5, flow 4/5
- **Covers / works well:** “Familiar roles, new insides” appropriately links Stage 03 while preserving the full canonical explanation. The deliberate `[3,1,4,1]` reversal diagnostic shows why mean pooling loses addresses without claiming that learned recurrent summaries have the same limitation. Cross-attention roles and `(target, source)` routing are explicit.
- **Issue [P2]:** The switch from prose to digit reversal is justified but could still jar. Add a compact mapping at “Why this deliberately crude attempt?” stating that digits stand in for distinguishable source tokens so address loss is visible, not that reversal is a language task.

##### `05b-encoder-decoder-and-cross-attention-lab.ipynb` — novice 3/5, flow 4/5
- **Covers / works well:** Source/target shifts, distinct decoder self-attention and cross-attention, teacher-forced versus free-running evaluation, and learned-routing diagnostics build directly on 5A. Source memory is correctly reused across generation.
- **Issue [P2]:** “Did the learned route match the prediction?” risks making averaged attention weights sound like a complete explanation of the model’s behavior. Add a short boundary note that the routing map is diagnostic evidence, not proof that attention alone caused the correct output.

##### `06-modern-decoder-only-llm.ipynb` — novice 3/5, flow 4/5
- **Covers / works well:** RMSNorm, SwiGLU, GQA and RoPE are introduced as refinements to the same next-token job. Controlled ablations, invariants, diagnostics and the representative configuration bridge make key claims checkable; the RoPE revisit is consistent with Part 2.
- **Issue [P2]:** This is the densest chapter and its many named mechanisms could become a tour for readers who need a refresher. Add a compact “same contract / changed internals” recap immediately before “Toy-to-Real Configuration Bridge,” pointing back to the four measured changes.

##### `focused/01-attention-position-and-qkv-focused.ipynb` — novice 3/5, flow 4/5
- **Covers / works well:** The focused setup restores the familiar sentence locally; the failure experiments, RoPE attempts, invariants, Q/K/V lesson and scaling evidence survive without replaying Part 1. It runs as an independent copy by design.
- **Issue [P2]:** Compared with canonical Part 2, the “Before the rotating dials” bridge is shorter and does not carry forward the full states → projections → head split → rotate Q/K → masks → softmax → weighted V path. Add that one-line path (with the unrotated-V reminder) before the preview, and record the focused copy’s checked-against revision in its provenance.

##### `focused/02-transformer-block-focused.ipynb` — novice 3/5, flow 4/5
- **Covers / works well:** This is a coherent shortcut of Part 3: it retains the incompatible-routing experiment, per-token FFN, normalization, residual-gradient evidence and predictions, with a compact local setup.
- **Issue [P3]:** Its provenance and README establish that it is derived and non-authoritative. Add a short link to the canonical README’s encoder/decoder block comparison so focused readers do not mistake the one demonstrated block for every family’s complete operation order.

##### `focused/03-decoder-and-encoder-decoder-focused.ipynb` — novice 3/5, flow 3/5
- **Covers / works well:** The combined route retains the future-leakage and changed-prefix evidence, then introduces separate-source routing and the three reusable states. “Change the microscope” makes the shift from a text continuation to a numeric reversal visible; the README is clear that this is theory-level, not a replacement for 4B/5B labs.
- **Issue [P2]:** Combining 4A and 5A creates a substantial task/context switch. Add a small comparison card at “Change the microscope” with one growing tape versus a fixed source plus growing target; this would reinforce why the objective changes without repeating either lesson.

#### Cross-notebook inconsistencies
- **Architecture and roles:** The README’s encoder = self-attention + FFN; decoder-only = causal self-attention + FFN; encoder-decoder writer = causal self-attention + cross-attention + FFN order is technically consistent with Parts 4–6. It explicitly says residuals/normalization are omitted from this comparison, not absent. Stage 03’s recurrent reader/writer roles remain distinct from Transformer internals.
- **Teacher forcing and masks:** Parts 4A and 5A correctly reuse “gold previous target during practice, own previous choice during generation,” while 5A adds a fixed source. Causal visibility, padding attention and loss-ignore behavior remain separate contracts. The distinction is consistent with Stage 03 `02C` and model-evolution notebook 3.
- **Examples and handoffs:** The cat sentence, `aria` continuation and digit reversal differ, but each has a stated diagnostic purpose. Stage 05’s data-pipeline handoff matches Part 6’s token-ID-in/logits-out decoder contract. `README.md` and both `CONTENTS.md` maps list the canonical notebooks/theory notes and focused branch coherently; all checked local links resolve.
- **Companion notes:** Theory notes track their notebook concepts and use the same role distinctions, but Parts 2–6 run about 1,238–1,414 words, above the guide’s roughly 850–1,150-word target. This is a mild accessibility/hand-copying concern; prioritize short signposts for the key distinction in each rather than expanding further.

#### Top 5 prioritized improvements
1. **[P1]** In focused Part 1 (`focused/01-attention-position-and-qkv-focused.ipynb`, “Before the rotating dials”), add the missing complete attention-path reminder and mark the focused copy’s checked-against revision; this is the clearest canonical-bridge divergence.
2. **[P2]** At Part 1, “Where did the prediction lesson come from?”, distinguish CBOW’s center-word lesson from the later autoregressive objective.
3. **[P2]** At Part 5A, “Why this deliberately crude attempt?”, explain why numeric reversal isolates address loss and is not a claim about learned recurrent compression.
4. **[P2]** At Part 5B, “Did the learned route match the prediction?”, bound what averaged attention weights can establish.
5. **[P2]** Add the small conceptual diagrams/shape traces in Part 2’s real-RoPE path, Part 3’s block-family pointer, and focused Part 3’s one-tape/two-workspace transition; the corresponding three image prompts remain pending where applicable.

---

### Pedagogy Evaluation: Stages 05–06

#### Stage 05: LLM Training

##### Stage summary
- **Overall flow: 4/5.** The route moves from trusted data to checkpoint contracts to a complete training lifecycle.
- The Stage 04 handoff is explicit and correctly narrows the new work to data and training state rather than teaching the Transformer again.
- Evidence habits recur: measured failures, held-out selection, resume parity, disk reloads, and honest tiny-run boundaries.
- Main friction: the lab reconstructs a fairly sophisticated Stage 04 model and introduces many operational terms; optional branches are appropriately placed after the required route.

##### Per-notebook findings

###### `05-llm-training/01-pretraining-data-pipeline.ipynb` — novice score 4/5, flow score 4/5
- **Covers / works well:** Cell heading “Stage 04 handoff” gives a compact token-ID / next-token-score contract. “Inventory, Then Split by Document” through “Write, Reload, Hash, and Decode” forms an unusually strong complaint chain; diagrams, predictions, measured checks, and `preview_chars` exercise make leakage, EOD, packing, and trust boundaries concrete. Companion `01-pretraining-data-pipeline-theory.md` preserves the same stream-to-model story.
- **Issue [P1]:** “Write, Reload, Hash, and Decode” and the theory companion §8 refer to “Notebook 08” as the digest-refusal consumer, but the required route now names `03-pretrain-a-base-model-lab.ipynb`. Replace the stale handoff with that actual Part 3 link/name.
- **Issue [P2]:** Terms such as shingle similarity, BPE, tokenizer fitting, dtype, and manifest arrive in one dense pipeline. Add a small static “one document → one split → token IDs → block” visual keyed to the Riverside artifact names; keep the threshold animation’s measured static counterpart easy to find.

###### `05-llm-training/02-pretraining-theory.ipynb` — novice score 4/5, flow score 4/5
- **Covers / works well:** “Architecture is capacity; updates are experience” uses a small measured update before “Validation chooses; training loss diagnoses” and a full-state-vs-weights-only resume test. “Artifact lineage” then leads cleanly to Part 3. Companion `02-pretraining-theory.md` summarizes the same causal update, held-out selection, full pause state, and honest boundaries.
- **Issue [P2]:** The opening line-fit experiment is a clean miniature but temporarily leaves the next-token story. Add one sentence mapping its train/validation/update loop back to a shifted token block before the checkpoint section.

###### `05-llm-training/03-pretrain-a-base-model-lab.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** The opening contract flows from artifact refusal to model reconstruction, shifted targets, optimizer policy, validation, resume parity, and packaging. It explicitly says the initial generation is seeded behavior, not learned knowledge; the code walkthroughs help with longer cells.
- **Issue [P2]:** The “Compact TinyModernLM” reconstruction (112-line cell) reopens several Stage 04 mechanisms before training. Mark the cell as a faithful implementation of the already-built model contract, and add a short shape trace from packed IDs to logits and shifted labels; avoid making readers infer that this is a new architecture lesson.

###### `05-llm-training/optional/lightning/training-a-base-model-with-lightning.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** A small model makes the manual-vs-Lightning boundary visible, then checks same-batch loss/shape and reload-logit parity. Its “Optional checkpoint” closes with the correct claim: Lightning wraps orchestration, not the objective.
- **Issue [P2]:** It switches to a synthetic 17-token, 8-position dataset rather than Part 3’s Riverside shards. State plainly that this is an isolated parity micro-example, not a Lightning rerun of the Stage 05 artifact contract; a tiny artifact-map inset would help.

###### `05-llm-training/optional/full-model-audit/tiny-transformer-for-ai-engineers.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** Six explicit audit questions and a read-only scorecard give a useful retrieval pass after Parts 1–3; the final “What this tiny run does not prove” is careful and concrete.
- **Issue [P2]:** The audit expects a fully produced package and speaks in hashes, tensor shapes, and manifests. Keep it optional as documented; add one “start here if an artifact is missing” link to the producing Part 1/3 section so failure does not feel like an unexplained dead end.

#### Stage 06: LLM Fine-Tuning

##### Stage summary
- **Overall flow: 4/5.** Four decisions are ordered well: objective, update strategy, evaluation, then staged GPU evidence.
- The Aria scene in *The Weight of Distant Light* anchors Parts 1–2; Part 3 explicitly distinguishes objective-specific questions; Part 4 turns evaluation discipline into an executable runbook.
- The content is evidence-conscious and repeatedly warns that mechanism runs, fixtures, and tiny samples do not establish capability.
- The main novice cost is scale: Parts 1–3 are long, technically dense notebooks, and Part 4’s 307-line runner is only navigable by its stage map rather than taught linearly.

##### Per-notebook findings

###### `06-llm-finetuning/01-llm-finetuning-data-techniques.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** “Motivation: See the Starting Behavior Before Naming a Technique” leads into CPT → SFT → preference; the same editor request and Aria scene support the complaint chain. Label masking, supervised EOS, contract-matched DPO pairs, and contaminated-evidence warnings are unusually explicit. Companion `01-llm-finetuning-data-techniques-theory.md` carries the same objective distinctions.
- **Issue [P2]:** The 61-cell chapter gives PPO a deep judge/reference/value/old-policy/clipping tour before returning to DPO; several SFT/DPO implementation cells are 150–240 lines. Make PPO a clearly optional depth branch and add short post-cell walkthrough landmarks, so the central CPT/SFT/DPO objective contrast is not buried.

###### `06-llm-finetuning/02-llm-finetuning-parameter-techniques.ipynb` — novice score 3/5, flow score 3/5
- **Covers / works well:** The “total bill” (update state, artifact, regression) is a memorable throughline; full FT → freezing → LoRA → QLoRA follows a genuine constraint ladder, with measured parameter/artifact evidence and explicit limits. Companion `02-llm-finetuning-parameter-techniques-theory.md` gives strong LoRA intuition and correctly labels the CPU low-bit path an analogy.
- **Issue [P1]:** The LoRA section heading says “Concept 3” although the notebook has already numbered full fine-tuning as Concept 4 and freezing as Concept 5; QLoRA is Concept 7. Correct the sequence to Concept 6. The companion §3 also repeats the “Freezing changes update permission, not computation” explanation nearly verbatim; retain one copy.
- **Issue [P2]:** Tensor, optimizer, trainer, quantization, and device concepts stack rapidly. A single static forward/backward diagram showing frozen base vs trainable adapter and the four separate cost lines would ease navigation (some related visuals exist, but the cross-strategy comparison should be one glance).

###### `06-llm-finetuning/03-llm-finetuning-comparison-and-decision.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** “One Riverside Request Becomes an Evaluation” turns baseline/case/scorer/metric/gate into a concrete chain; separate CPT, SFT, DPO, and strategy claims prevent unlike candidates being ranked together. Synthetic records and fixture gates are labeled as mechanics only. Companion `03-llm-finetuning-comparison-and-decision-theory.md` is especially clear about evidence-unavailable decisions and rollback.
- **Issue [P2]:** This is a broad, 39-cell evaluation/control-plane notebook with many terms and long implementation cells. Keep the “one request” walkthrough as the required core and make the parameter-strategy/control-plane implementation an optional appendix or provide a one-page evidence ladder graphic before candidate loading.

###### `06-llm-finetuning/04-llm-finetuning-practice.ipynb` — novice score 2/5, flow score 3/5
- **Covers / works well:** Preflight → one-novel pilot → validation selection/reload parity → one-time test opening → bootstrap/retention/gates → seven more novels is responsibly staged. The README’s CPU-safe preflight and CUDA opt-ins match the notebook. Companion `04-llm-finetuning-practice-theory.md` explains chronological splits and sample-size limits.
- **Issue [P1]:** The core runner is a 307-line function. “Read the Runner as Nine Visible Stages” is helpful, but not a walkthrough of each stage; split the teaching explanation into numbered annotated excerpts (or move runner implementation to a helper module and keep a short inspectable slice in the notebook). Keep stage boundaries and evidence objects visible.
- **Issue [P2]:** Clarify that test isolation is per independent novel run: Parts 1–2 train mechanism examples on the shared 40-chapter Aria corpus. Part 4’s test is withheld from its own candidate’s selection, but is not a globally unseen corpus for a reader following the whole track. Keep the current narrow local-mechanics claims.

##### Cross-notebook inconsistencies
- **Stage 05 stale consumer:** “Notebook 08” appears in Part 1’s lineage diagram/walkthrough and its theory notes, while the current required consumer is Part 3 (`03-pretrain-a-base-model-lab.ipynb`).
- **Stage 06 title drift:** Part 2 is called “Where Should the Update Live?” in the README and Part 1 handoff, but the notebook title and `CONTENTS.md` use “What Must Riverside Pay to Change?”. Choose one title across all three.
- **Stage 05 → 06 handoff:** Stage 06 lists Stage 05 as previous, but its exercises load separate SmolLM2 hub checkpoints rather than Stage 05’s tiny locally trained package. Add a short note that this is an intentional real-base-model substitution; preserve the learning invariant (same next-token model, different objective/update/evidence).
- **Evidence boundary:** Stage 06 Part 3 correctly says Parts 1–2 use all 40 chapters; Part 4’s within-run test protocol is sound, but its audience-facing wording should distinguish “held out from this run” from “never encountered anywhere in the curriculum.”
- Stage 05 required/optional README and `CONTENTS.md` route entries agree. Stage 06’s four-part order and GPU opt-in instructions also agree; no broken or missing companion links were evident in the reviewed route.

##### Top 5 prioritized improvements
1. **[P1]** Replace Stage 05 “Notebook 08” with the actual Part 3 artifact consumer in the pipeline notebook and theory companion.
2. **[P1]** Break down Stage 06 Part 4’s monolithic 307-line runner into annotated, learner-sized stages without hiding the evidence contract.
3. **[P1]** Fix Part 2’s LoRA concept number and remove the duplicated freezing paragraph from its theory companion.
4. **[P2]** Reconcile the Stage 06 Part 2 title in README, `CONTENTS.md`, and Part 1’s handoff; explain why Stage 06 uses SmolLM2 rather than the Stage 05 local tiny checkpoint.
5. **[P2]** Reduce novice load in Stage 06 Parts 1–3: make PPO depth optional, add post-code walkthroughs and a static evidence/strategy map, and distinguish per-run held-out test data from globally unseen data.

---

### RAG, LLM Evaluation, and LLM Gateway

#### Stage 07 — Retrieval-Augmented Generation
- **Overall flow: 4/5.** A clear progression: retrieval failures → a built hybrid pipeline → controlled retriever/generator diagnosis.
- The Riverside policy scenario gives retrieval a concrete authorization constraint, not just a relevance leaderboard.
- The Stage 03 bridge usefully distinguishes a token’s learned input row from a whole-query/passage search vector; its plain distinction is correct and relevant.
- Handoff to Stage 08 preserves retrieval recall, support, attribution, and risk as separate evidence rather than one RAG score.

#### Stage 08 — LLM Evaluation
- **Overall flow: 4/5.** The four-part route moves from what metrics observe, to judge accountability, claim-level checks, and calibrated decision policy.
- Strong cross-part ownership avoids re-teaching every metric in every chapter; READMEs and CONTENTS agree on the route and companions.
- The claim-risk → confidence distinction provides a sound bridge into gateway controls.
- Main continuity cost: the Riverside cast/example changes among pendant/reference metrics, ReAct diagnosis, Booker claims, and synthetic confidence records. All are locally understandable but not one continuous experiment.

#### Stage 09 — LLM Gateway
- **Overall flow: 4/5.** Control-plane theory motivates normalization, eligibility, lifecycle ordering, and retry-versus-fallback; the lab makes these contracts measurable.
- The Stage 08 matrix is translated into route eligibility rather than allowing latency/cost to override safety or quality gates.
- Deterministic adapters, explicit simulation boundaries, production mapping, and a guarded optional live bridge prevent toy results from posing as cloud evidence.
- The lab is broad and long (75–90 minutes); its number of controls can crowd out the novice’s original question unless the required path is made especially visible.

#### Per-notebook findings

##### 07-rag/01a-hybrid-retrieval-theory.ipynb — novice score 4/5, flow score 4/5
- **Covers / works well:** Contrasts lexical and dense misses, then uses RRF, candidate depth, authorization-before-scoring, and a support boundary. The “Attempt 1/2” complaints make fusion necessary; the scorecard answers the opening question. The linked handwritten notes are consistent and preserve the same policy examples.
- **Issues:** [P2] The “One word, two jobs” Stage 03 bridge is a standalone cell before the visible failure, and substantially repeats in 01b. Tie it directly to the question’s retrieval-vector need, and shorten the lab repeat to a callback. Keep the illustrative support check clearly labelled as a toy gate, not a calibrated policy.

##### 07-rag/01b-hybrid-retrieval-lab.ipynb — novice score 3/5, flow score 4/5
- **Covers / works well:** Builds BM25 and a tiny PyTorch dense path, compares fusion, reranks without changing membership, fails closed on ACL uncertainty, records degraded mode, and binds artifacts with a manifest. The scorecard distinguishes retrieval metrics from operational telemetry.
- **Issues:** [P1] Before presenting the toy dense encoder’s results as “semantic,” state exactly what its inspectable feature space can and cannot establish; give a compact toy→production mapping (e.g., learned/pretrained embedding model, vector index, dimensions/version) and one verified real contract. [P2] The six-part lab is dense; add a short “first pass” route so readers can skip the operational extensions without losing the retrieval lesson.

##### 07-rag/02-rag-failure-localization-and-evaluation.ipynb — novice score 4/5, flow score 4/5
- **Covers / works well:** Keeps per-query records, separates retrieval, groundedness, relevance, and correctness, then uses oracle context and independent release gates. The same ReAct question makes the ablation intelligible; theory notes correctly label eight cases as rehearsal, not production evidence.
- **Issues:** [P2] The first section presents a 2×2 retriever/generator table before the per-query diagnostic vocabulary is fully established. Add one sentence walking the reader through the running wrong-answer case and name the owner before generalizing to all four states. Preserve the distinction between “grounded in a wrong source” and “correct.”

##### 08-llm-evaluation/01-llm-evaluation-metrics-and-benchmarks.ipynb — novice score 3/5, flow score 4/5
- **Covers / works well:** Leads with a valid-paraphrase-versus-fluent-error complaint; compares reference overlap, semantic similarity, and language-model surprise; then keeps disagreement and benchmark slices visible. The companion accurately warns that BERTScore and perplexity do not prove truth.
- **Issues:** [P2] BLEU, ROUGE-L, METEOR, BERTScore, and perplexity arrive in quick succession. Put a one-line “question this instrument answers” legend beside the first comparison, and keep the Riverside pendant example as the through-line. Make clear that the tutorial’s transparent semantic space is not BERTScore or an entailment test.

##### 08-llm-evaluation/02-llm-as-judge-safety-and-pipeline.ipynb — novice score 4/5, flow score 4/5
- **Covers / works well:** An unanchored judge score motivates a rubric; position bias motivates swapping; agreement and safety remain separate; a versioned release record closes the chain. The theory companion adds verbosity/self-preference controls and cautions that agreement is not correctness.
- **Issues:** [P2] The visible judge is a deterministic bias proxy, not a live LLM evaluator. Keep that boundary adjacent to its output and add a brief mapping from the proxy test to a production judge’s pinned model, prompt, rubric, and human audit.

##### 08-llm-evaluation/03-hallucination-detection.ipynb — novice score 3/5, flow score 4/5
- **Covers / works well:** Splits answer claims, distinguishes supported/contradicted/unknown, locates entity gaps, and connects threshold choice to review cost. The companion is particularly careful that unsupported does not mean globally false and that entity matching misses relation errors.
- **Issues:** [P1] The simple claim splitter/entity and risk rules can look like a detector despite being teaching heuristics. Put “proxy, not validated NLI/classifier” beside the first result and show which label evidence supports each risk input. [P2] Preserve the example’s “longlisted” versus “won” relation as the clearest demonstration that matching entities do not establish support.

##### 08-llm-evaluation/04-calibration-and-confidence.ipynb — novice score 4/5, flow score 4/5
- **Covers / works well:** Reliability bins precede ECE; temperature scaling is measured with ranking preservation; risk–coverage makes abstention costs visible; claim risk is an independent gate. The next-stage handoff clearly tells the gateway what it may enforce.
- **Issues:** [P2] The synthetic confidence fixture is mechanically useful but the resulting threshold/routing policy may feel more authoritative than warranted. Keep the “teaching fixture” caveat beside the chosen threshold, and distinguish bin-level calibration from an individual answer’s truth.

##### 09-llm-gateway/01-gateway-control-plane-theory.ipynb — novice score 4/5, flow score 4/5
- **Covers / works well:** Failure-first attempts expose provider-shape leakage, optimize-before-eligibility, misplaced cache/budget controls, and retry/fallback confusion. The toy-to-real table and coverage ledger make system boundaries explicit.
- **Issues:** [P2] The Stage 08 evidence bridge is useful but abstract at first. Carry one concrete eligibility decision (for example, an otherwise cheap route that failed a required safety/region gate) through to the normalized request and final route. Keep cache/output-policy caveats visible so “cache first” cannot be read as bypassing current policy.

##### 09-llm-gateway/02-gateway-routing-resilience-lab.ipynb — novice score 3/5, flow score 3/5
- **Covers / works well:** Resumes the same request; measures eligibility, queue-aware routing, multiple admission limits, reserve/settle, safe cache identity, bounded recovery, telemetry, and production mappings. Clear simulation labels and the disabled-by-default live bridge are strengths.
- **Issues:** [P1] The 43-cell, 75–90-minute lab spans enough mechanisms to feel like an operations catalog. Mark a short required route (adapter → eligibility/routing → retry/fallback) and label limiter/cache/budget extensions, preserving all evidence. [P2] Several mechanisms depend on vocabulary (EMA, token bucket, concurrency, TTL); define each in plain English at first use before showing the implementation.

#### Cross-notebook inconsistencies
- The Stage 03 retrieval-embedding bridge is a valuable additive clarification, not a contradiction, but its repeated near-identical cell in 01a and 01b risks feeling bolted on. Use a full explanation once, then a one-line callback in the lab.
- “Riverside” changes from employee policy retrieval (07) to manuscript/award evaluation (08) and an editorial gateway request (09). The shared organization is plausible, but the handoffs do not explain the domain change; identify the product/workflow shift once and reuse a named request where practical.
- Stage 07’s ReAct question is diagnostic, while Stage 08 uses different literary examples. Local chapter continuity is stronger than track-wide example continuity. READMEs and CONTENTS otherwise match the notebooks, companion links, ordering, and stated boundaries.
- The toy/production boundary is consistently acknowledged in prose; the main risk is adjacency: readers can mistake toy semantic features, proxy hallucination scores, or synthetic confidence thresholds for validated measurements.

#### Top 5 prioritized improvements
1. **[P1] 07/01b:** Explicitly bound the toy dense encoder and add a verifiable production embedding/index bridge.
2. **[P1] 08/03:** Label heuristic claim detection/risk as proxy beside the outputs; distinguish it from validated attribution.
3. **[P1] 09/02:** Add a clearly marked short required route; keep operational extensions without making them prerequisites.
4. **[P2] 07/01a + 01b:** Retain the Stage 03 distinction but avoid repeating the full bridge verbatim.
5. **[P2] Cross-stage 07–09:** Explain Riverside’s policy-to-editorial workflow shift and carry one concrete request through the handoffs.

*Review method: parsed all nine notebook JSON sources and read scoped READMEs, CONTENTS, and available theory companions. Notebooks were not executed, per scope.*

---

### Agentic AI (Stage 10)

#### Stage summary

- **Overall flow: 4/5.** The required route generally moves from executable tool contracts to bounded loops, durable state, reusable skills, graphs, retrieval, evaluation, safety, interoperability, specialists, and recovery. OrderFlow keeps these mechanisms in one purchasing domain.
- **Strong evidence ladder:** each notebook has a falsifiable failure, deterministic fixtures, assertions/measurements, and a production-boundary note; chapter images exist at the main mechanism points. The local fixtures are consistently distinguished from production guarantees.
- **Bridges mostly integrate smoothly.** In this track, commit `802533a` added README/CONTENTS routes and two useful first-use bridges (Chapter 02 memory glossary; Chapter 07 protocol vocabulary), not notebook cells. The 13 optional system-design notes also place a plain-language bridge, a hands-on prerequisite, and a lab handoff before dense reference material. The repeated “retain the run, inputs, environment...” sentence is useful but templated.
- **Main novice gap:** many notebook openings put a historical account and a compact notation box before the first concrete complaint. This is a smaller problem on the README-first path, but a real hurdle for direct notebook opens. Internal relative links were checked: none are broken.

#### Per-notebook findings

##### 10-agentic-ai/00-agent-foundations-and-tool-contracts/00-agent-foundations-and-tool-contracts.ipynb — novice score 3/5, flow score 4/5
- Covers registered tools, schema/policy validation, typed retry decisions, an observable loop, and tool/skill/MCP distinctions. The naive-versus-typed comparison and unknown-SKU rejection make the action boundary convincing (`## 1 - The Agency Boundary`, `## 2 - Typed Tool Calls`).
- **[P2]** The opening names ReAct and presents `r`, `a_t`, `o_t`, `T`, and `V` before the challenge supplies plain-language intuition (`# Agent Foundations and Tool Contracts`). Move or gloss notation after the “prose cannot prove a tool exists” complaint; direct readers otherwise meet several terms at once.

##### 10-agentic-ai/01-reasoning-planning-and-control/01-reasoning-planning-and-control.ipynb — novice score 3/5, flow score 4/5
- The repeated inventory check motivates plans, terminal states, cycle detection, then per-tool timeouts; the measured timeout comparison is an especially good toy-to-real bridge (`## 1 - Failure First`, `### Per-Tool Timeouts and Retries Share the Run Budget`). It appropriately leaves the richer error taxonomy to Chapter 00.
- **[P2]** The opening notation and budget equation arrive before a learner has a picture of a finite run (`# Reasoning, Planning, and Bounded Control`, `## 3 - Budgets...`). A short everyday “one run has a step, time, and spending limit” lead-in would reduce that bump; the formula is optional.

##### 10-agentic-ai/02-state-context-and-memory/02-state-context-and-memory.ipynb — novice score 3/5, flow score 3/5
- Compares memory strategies on the same purchase facts, then tests namespace isolation, poisoning, and restart (`## 2 - Compare Memory Strategies`, `## 3 - Namespace Isolation`, `## 4 - Durable Checkpoints`).
- **[P1]** The README glossary is a good bridge *if readers follow the documented route*: it defines working/episodic/semantic/procedural memory and state versus context before linking the notebook. Direct notebook readers instead see unexplained notation first. More importantly, its closing `## Roadmap Checkpoint` sends readers straight to Chapter 03; README/CONTENTS make 02a a required intervening chapter. Add a short notebook lead-in and an explicit 02a handoff.

##### 10-agentic-ai/02a-skills-discovery-composition-and-versioning/02a-skills-discovery-composition-and-versioning.ipynb — novice score 3/5, flow score 4/5
- A floating supplier-tool version changes the recommendation; manifests, gated discovery, pins, evaluation, promotion, and rollback then each answer that failure (`## 0 - The Challenge`, `## 3 - Discover Less`, `## 6 - Pin Behavior`, `## 7 - Evaluate...`). The toy-to-real table and the linked 1,121-word theory companion preserve the same distinctions.
- **[P2]** This is the densest hands-on notebook: tenant, role, risk, status, dependency health, capability surplus, and release state accumulate quickly. Add one compact worked selection trace before the ranking rules. A static dependency/promotion picture could help, though the existing Mermaid and state diagrams already cover the main flows.

##### 10-agentic-ai/03-durable-workflows-with-langgraph/03-durable-workflows-with-langgraph.ipynb — novice score 3/5, flow score 4/5
- Builds a framework-free state machine and fan-out/fan-in before introducing LangGraph, then demonstrates recovery after a crash (`## 1 - Build the State Machine Before the Framework`, `## 2 - Fan-Out and Fan-In`, `## 4 - Crash, Resume...`). This ordering is strong.
- **[P1]** The implementation maps `auto`, `manager`, and `finance` to the same `approve` function, which always sets `approved=True`. The proof establishes that high-value work *visits* Finance, not that a human approved it. Label this as routing-only or model the approval interrupt; otherwise it conflicts with Chapter 06’s human-control and payload-bound approval lesson (`## 3 - Map the Mechanism to LangGraph`).

##### 10-agentic-ai/04-agentic-rag-and-self-correction/04-agentic-rag-and-self-correction.ipynb — novice score 3/5, flow score 4/5
- The stale-$50,000 policy failure leads naturally to trust/revision filtering, a bounded rewrite, fallback, and citation verification (`## 1 - Failure First: Retrieve Is Not Trust`, `## 3 - One Bounded Rewrite`, `## 4 - Conflict Handling...`). It respects Chapter 07 RAG as a prerequisite and avoids claiming a toy index is production retrieval.
- **[P2]** `q`, `D`, `R_k`, and `g` appear in the opening notation before their use is grounded. Define “question,” “policy collection,” and “shortlist” in words first (`# Agentic RAG and Bounded Self-Correction`).

##### 10-agentic-ai/05-agent-evaluation-and-observability/05-agent-evaluation-and-observability.ipynb — novice score 3/5, flow score 4/5
- Same terminal status hides route and argument bugs; spans and trajectory graders localize each seeded defect, then cost/latency attribution completes the arc (`## 1 - Failure First`, `## 2 - Trace Spans`, `## 4 - Regression Harness`).
- **[P2]** The notation line at the top contains a literal tab in `\tau`, so the trajectory symbol is malformed (`# Agent Evaluation, Observability, Cost, and Latency`). Also label the p95 from three cases as an illustrative fixture statistic, not a meaningful production tail estimate (`## 3 - Cost and Latency Attribution`).

##### 10-agentic-ai/06-safety-human-control-and-governance/06-safety-human-control-and-governance.ipynb — novice score 3/5, flow score 4/5
- The supplier injection gives a concrete reason to separate evidence, authority, policy, approval, and audit. Hash-binding changed arguments and checking all five attack fixtures are strong proof points (`## 1 - Failure First`, `## 3 - Policy-as-Code...`, `## 4 - Append-Only Decision Records`).
- **[P2]** The opening notation defines trust and approval with symbols before the story teaches the labels (`# Safety, Human Control, and Governance`). State plainly that the 0/5 result is for the five committed attacks; it demonstrates those controls on this suite, not general immunity to prompt injection.

##### 10-agentic-ai/07-mcp-and-agent-interoperability/07-mcp-and-agent-interoperability.ipynb — novice score 3/5, flow score 3/5
- Discovery, client-side schema rejection, drift review, and a configuration-only provider swap are measurable and useful (`## 2 - Build the Protocol Lifecycle`, `## 3 - Client Discovery...`). The added README bridge defines JSON-RPC and MCP before the local contract, so it fits the route well.
- **[P2]** `### How OpenClaw-Style Personal Agents Work` is a long product/runtime detour inside the protocol lesson. It introduces Gateway, channel adapters, heartbeats, plugins, and sandbox responsibility without advancing OrderFlow’s discovery example. Move it to an optional callout/reference or trim to the one protocol distinction it teaches.

##### 10-agentic-ai/08-multi-agent-communication-and-coordination/08-multi-agent-communication-and-coordination.ipynb — novice score 3/5, flow score 4/5
- It first proves why a generalist fails, then uses typed envelopes, scoped specialists, fan-in, and measured parallel completion (`## 1 - Failure First`, `## 2 - Typed Message Envelopes`, `## 3 - Router versus Supervisor-Worker`). It correctly cautions against splitting agents without a real need.
- **[P2]** `A2A` appears in `## 4 - Context Partitioning, Conflict Handling, and A2A Discovery` without expansion or a clear boundary from MCP/agent cards. Gloss the acronym and say what this local fixture does—and does not—implement. The modeled-latency caveat should remain beside the result.

##### 10-agentic-ai/09-reliability-recovery-and-production-decisions/09-reliability-recovery-and-production-decisions.ipynb — novice score 3/5, flow score 4/5
- Lost response → duplicate commitment → idempotency → durable events/replay → compensation makes a satisfying closing complaint chain (`## 1 - Failure First`, `## 2 - Idempotency Keys...`, `## 4 - Saga Compensation`). It explicitly labels daily capacity as modeled rather than load-tested.
- **[P2]** “Circuit breaker” and “dead-letter” are named quickly; give each a one-sentence plain-language definition before the code/coverage ledger (`## 2 - Idempotency Keys...`, `## 3 - Durable Events...`). Keep the modeled-capacity result clearly separated from the deterministic chaos evidence (`## 5 - Chaos Suite...`).

#### Cross-notebook inconsistencies

- Chapter 02 is the only notebook whose prerequisite glossary and canonical handoff were added to README rather than notebook. That is acceptable for README-first readers, but its notebook’s direct Chapter 03 handoff remains contradictory to the required 02 → 02a → 03 route.
- Chapter 03’s automatic `approved=True` implementation blurs “routed to an approver” with “approval granted,” unlike Chapter 06’s explicit human approval bound to exact payload. This is the most important conceptual mismatch.
- OrderFlow is a strong shared world, but the central request changes (`PO-7293`, `PO-7298`, `PO-7307`, `PO-7311`). That is reasonable for varied fixtures; signpost each as a separate incident so “same running example” does not imply one continuous purchase.
- The 02a theory notes match the notebook’s versioning/authority story and are discoverable from README and CONTENTS. The system-design branch is correctly marked optional and consistently routes all 13 component notes to the corresponding hands-on chapter; it is a dense interview/architecture reference, not novice-first continuation.
- Chapter 05’s tab-corrupted `\tau` is the only confirmed source-encoding/rendering defect found. Chapter titles, README routes, CONTENTS entries, and local image links otherwise align.

#### Top 5 prioritized improvements

1. **[P1]** In Chapter 02, add an in-notebook plain-English memory bridge and route the final checkpoint through 02a before Chapter 03.
2. **[P1]** In Chapter 03, distinguish approval routing from approval completion; make the Finance/HITL boundary explicit and test it.
3. **[P2]** Fix the literal tab in Chapter 05’s `\tau` notation and label p95 from three fixtures as illustrative only.
4. **[P2]** Add direct-open plain-English lead-ins before notation in Chapters 00, 01, 04, and 06; keep formulas as optional evidence, not the opening vocabulary test.
5. **[P2]** Keep Chapter 07’s OpenClaw aside optional, and define A2A at first use in Chapter 08 to protect the core progression.

---

### Stage 11 — AI Infrastructure Pedagogy Evaluation

#### Stage summary
- **Overall flow: 4/5.** The required route has a sound escalation: device limits → memory formats → profiling → attention IO → multi-GPU training → compressed artifacts → serving → kernels → service composition → cloud-shaped operations. Chapters 09–10 make a useful distinction between mechanism evidence and production validation.
- Complaint → prediction → measurement → production boundary is strongest in Chapters 04 and 09–10; the new FlashAttention scorecard improves its local online-softmax transition.
- Novice readiness varies: some openings front-load jargon, chapters are dense, and several claims need qualification. The README target and shortcut checks are appropriate.
- `CONTENTS.md`, the top README, and chapter 09/10 routes agree on chapter order and purpose. All ten notebooks parse as JSON, have unique cell IDs, and have cleared outputs. This was a static content review; no notebook was executed.

#### Per-notebook findings

##### `01-gpu-hardware/gpu-hardware-foundations.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** The “InferenceBase dilemma” motivates capacity and speed; CPU/GPU, bandwidth, roofline, occupancy, and coalescing build toward a concrete GPU choice. Predictions, toy comparisons, and a production readiness gate preserve the toy-to-real distinction.
- **Issues:** **[P2]** The opening paragraph (“Read the memory complaint first”) defines bf16, VRAM, activations, and KV cache before the actual dilemma that follows. Start with the dilemma, then define each term at first use. **[P2]** “Closing Decision” gives a GPU and hourly price; label the price and cost comparison as an illustrative assumption, and separate capacity fit from a real workload/cost qualification.

##### `02-mixed-precision/mixed-precision-and-memory.ipynb` — novice score 3/5, flow score 3/5
- **Covers / works well:** The OOM/overflow complaint motivates memory accounting, number formats, autocast/scaling, checkpointing, profiling, and a final A10G decision. The guided-reading and prediction cells are helpful.
- **Issues:** **[P1]** “What just happened” in Part 1 and “Key insights to keep” assert that training costs “4× parameter memory” and that Adam states are “always fp32.” These are configuration-dependent approximations: optimizer, master weights, trainable/frozen parameters, precision policy, sharding, and activations change the accounting. State the assumed setup beside the rule and show a contrasting LoRA/sharded case. **[P2]** The broad menu (including GradScaler and checkpointing) arrives quickly; use one explicit memory ledger throughout and make the order of weights → gradients → optimizer → activations visible.

##### `03-profiling/pytorch-profiling.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** The 45-second mystery leads naturally to timeline reading, observer effect, compute-versus-memory limits, compile cost, trace regions, and a whole training step. “What just happened” sections and the production-profile contract encourage measurement before optimization.
- **Issues:** **[P2]** The roadmap covers six tools and moves from `torch.profiler` to `autograd.profiler` to `torch.compile`; make the distinction between a profiler, a trace annotation, and a compiler explicit in one compact map before the tour. **[P2]** “End-to-End: Profiling a Real Training Step” is a small representative model, not a real production training workload; rename or prominently qualify “real” as “representative toy step,” and explain transfer limits.

##### `04-flash-attention/flash-attention-internals.ipynb` — novice score 3/5, flow score 3/5
- **Covers / works well:** The sequence from avoidable HBM crossings → tiling → online softmax → IO measurements → SDPA dispatch is coherent. The additive scorecard paragraph before “Part 3 — Online Softmax” is a good plain-English bridge to the running max, weight sum, and accumulated answer. “Anchor 3” rightly separates temporary S×S attention IO from persistent KV-cache memory.
- **Issues:** **[P1]** The opening never explicitly resumes the Stage 04 attention picture. Before the formula in “Part 1 — Standard Attention: The S×S Memory Problem,” say in plain English that each query compares with keys to weight values, and identify the S×S table as those query–key comparisons. Link to Stage 04’s canonical attention chapter and reuse its Q/K/V names. **[P2]** “materializing the full score matrix three separate times” conflates the score and probability intermediates with their HBM reads/writes; distinguish the two S×S arrays and the traffic between them. **[P2]** Label CPU timing comparisons as algorithm/implementation evidence, not GPU FlashAttention speedup; the chapter later correctly notes CPU measurements and dispatch constraints.

##### `05-distributed-training/distributed-training.ipynb` — novice score 2/5, flow score 3/5
- **Covers / works well:** DDP’s communication cost motivates FSDP sharding, then tensor/pipeline parallelism and a strategy decision. Diagrams and toy shape examples support the transitions; production launch and checkpoint actions are gated.
- **Issues:** **[P2]** “Parallelism Selection” jumps from simulated estimates to the recommendation “FSDP + LoRA + NF4: ~17 GB/GPU.” Make the estimate’s included/excluded state explicit (activations, optimizer, runtime/workspace, KV cache is not a training allocation) and state assumptions before the recommendation. **[P2]** DDP, FSDP, TP, PP, all-reduce, and bubbles make a dense sequence; define the shared “what is split / what is copied / what is communicated” comparison before naming each strategy.

##### `06-quantization/quantization-in-depth.ipynb` — novice score 2/5, flow score 3/5
- **Covers / works well:** The MacBook constraint motivates rounding error, dynamic/static PTQ, GPTQ, AWQ, GGUF, then clearly separated QAT and NF4/QLoRA appendices. The four design axes and production-artifact checklist are effective.
- **Issues:** **[P1]** The Riverside story and AWQ section repeatedly say “LLaMA-3-7B.” Meta Llama 3’s released sizes are 8B and 70B; use the real 8B name or explicitly call 7B a hypothetical model. **[P2]** The chapter juxtaposes toy/GPT-2 experiments with production-sounding “+0.3 perplexity,” “~30 tok/s on Apple M3,” and “negligible quality cost.” In “What This Notebook Covered” / “Toy to Production Bridge,” tag each number with model, runtime, device, dataset, and whether measured, modeled, or referenced. **[P2]** The two training appendices arrive after a deployment decision; mark clearly that they answer different questions from inference compression.

##### `07-inference-systems/inference-systems.ipynb` — novice score 3/5, flow score 3/5
- **Covers / works well:** KV cache → continuous batching → paged allocation → speculative decoding → prefill/decode builds an intelligible serving arc. The scheduler timeline, accept/reject trace, and latency clocks make temporal behavior visible.
- **Issues:** **[P1]** “Part 1 — KV Cache: Never Recompute What You’ve Already Computed” and its roadmap do not directly bridge to Stage 04’s attention mental model. Explain: at each decode step the new query scores against cached keys and mixes cached values; only the new token’s K/V is appended. Link the canonical Stage 04 attention chapter, and distinguish “recompute old K/V projections” from the still-required attention over the cached prefix. **[P2]** The opening says this is the “most important single optimization,” which is workload-dependent; replace with a bounded claim and state when cache memory becomes the constraint.

##### `08-triton-kernels/triton-kernels.ipynb` — novice score 2/5, flow score 4/5
- **Covers / works well:** The “performance cliff” motivates the execution-layer distinction; vector add, tiled matmul, fusion, softmax, autotuning, and `torch.compile` progress from small to more realistic kernels. The added paragraph that a Python tiling loop does not prove on-chip placement is especially valuable.
- **Issues:** **[P2]** The title/roadmap’s “87% of peak HBM bandwidth” and the opening “22× slower” are stronger than the toy demonstrations alone support. Label hardware, dtype, shapes, backend, and measurement source; keep the Python-vs-SDPA comparison explicitly illustrative. **[P2]** Several CUDA/Triton terms (PTX, registers, grid, program instance, occupancy) arrive rapidly; retain the primitive map but introduce each at first use in the vector-add example.

##### `09-systems-composition/systems-composition-lab.ipynb` — novice score 4/5, flow score 4/5
- **Covers / works well:** “Attempt 1: add replicas and hope” gives a falsifiable complaint, then tests queue ownership, bounded admission, rollback identity, and toy-to-real mapping. The companion theory notes and README accurately distinguish deterministic `LOCAL_FIXTURE` evidence from production capacity.
- **Issues:** **[P3]** In the scorecard, label scenario figures as fixture outputs and keep rejection rate beside accepted-request p95 whenever the result is discussed; the theory note already explains why dropping rejected work can mislead.

##### `10-azure-operational-llm-serving/azure-operational-llm-serving.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** The contract → readiness → admission → deduplication → deadlines → release → SLO → adapter route is a solid failure-first operational bridge. README and notebook consistently mark the model work as substituted and Azure behavior as unvalidated; each checkpoint motivates the next.
- **Issues:** **[P2]** The opening “story” spends space on the history of circuit breakers before the service complaint; lead with the adapter that works in-process but lacks a safe boundary, then put the pattern citation in a note. **[P2]** “Deadlines, Retries, Circuit Breaking, and Fallback” introduces a compact equation before the more intuitive budget explanation; lead with one request’s queue/attempt/fallback timeline, then retain the equation as optional shorthand.

#### Cross-notebook inconsistencies
- Chapter 04 opens with a worked attention expression and Chapter 07 opens with a KV-cache fix, but neither explicitly points back to Stage 04’s Q/K/V interpretation. Chapter 04’s “Anchor 3” distinguishes attention IO from KV-cache storage; Chapter 07 should reuse that exact distinction at first use.
- Chapter 06’s fictional “LLaMA-3-7B” conflicts with the real 7B/8B naming used elsewhere (e.g. Chapter 04’s LLaMA-3-7B-style model description). Normalize fictional versus real model labels across the track.
- “Measured,” “reference,” “toy,” and “production” claims are handled most carefully in Chapters 09–10, but Chapters 01/05/06/08 have cost, fit, quality, or speed values that could be mistaken for comparable production measurements. Normalize evidence labels throughout.
- The stage map and chapter READMEs/CONTENTS agree; Chapter 09’s theory companion is listed, and Chapter 10’s local/Azure boundary is consistent.

#### Top 5 prioritized improvements
1. **[P1] Restore the Stage 04 attention handoff** in Chapter 04 “Part 1 — Standard Attention” and Chapter 07 “Part 1 — KV Cache”: query-to-key scores weight values; the cache stores past K/V and does not remove attention over that prefix. Link and reuse the canonical Stage 04 example.
2. **[P1] Correct Chapter 06’s model identity** (“LLaMA-3-7B”) and qualify the related AWQ/quantization examples so real models and hypothetical shapes cannot be confused.
3. **[P1] Qualify memory rules** in Chapter 02 and Chapter 05: show optimizer/precision/sharding assumptions, separate persistent state from activations/workspace, and scope the 4×/17-GB estimates.
4. **[P2] Standardize evidence labels** for model quality, latency, bandwidth, and cost across Chapters 01, 04, 05, 06, and 08; distinguish measured GPU, CPU reference, toy simulation, and production target.
5. **[P2] Reduce novice load with additive visuals and first-use bridges:** a Q/K/V trace for Chapters 04/07, a persistent-versus-temporary memory ledger for Chapter 02/05, and a deadline budget timeline for Chapter 10. Preserve existing experiments and add these beside them.

---

### Stage 12 — Role-Based Tracks

**Scope:** Parsed 16 notebooks (15 authored: 6 AI Engineer, 9 FDE; 1 vendored in FDE `.venv`) and reviewed scoped READMEs/CONTENTS. No execution.

#### Stage summary

**AI Engineer — overall flow 4/5**
- One Riverside release-to-feedback story connects training data, prompt experiments, latency/cost, immutable release identity, and feedback into the capstone.
- The strongest chapters force their controls from measured fixture failures; assertions and explicit evidence boundaries curb overclaiming.
- This is synthesis, not a first exposure: Stage 01–11 prerequisites are stated, though the optional-agent wording could be clearer against Stage 10's entry gate.

**Forward Deployed Engineer — overall flow 4/5**
- The same Riverside House customer request persists through discovery, design, data, identity, service commitments, rollout, incidents, operations, and capstone.
- It moves from ambiguity to bounded decisions; challenge-out gates carry concrete artifacts forward.
- `[Measured]`, `[Modeled]`, and `[Customer-validated]` are carefully separated; synthetic success is not represented as live customer competence.

#### Per-notebook findings

##### AI Engineer

###### `12-role-based-tracks/ai-engineer/01-training-data-quality-and-lineage/training-data-quality-and-lineage.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** `## 1 - Immutable Load and Structural Validity` through `## 8 - Evidence Handoff and Completed Roadmap` builds from a parseable-but-unsafe dataset to a fingerprinted, reviewable decision. The 13/13 schema-valid versus 12/13 needing action result is memorable; leakage, rights, PII, coverage, and label issues stay distinct.
- **Issue:** `[P2]` `## 7 - Fingerprint the Candidate and Gate Promotion` / `## 8`—make the one-row curated fixture's insufficiency for training readiness impossible to miss in the notebook's final takeaway. The README does say one row is not readiness. A small raw-file → checks → candidate digest diagram would help.

###### `12-role-based-tracks/ai-engineer/02-prompt-release-and-experimentation/prompt-release-and-experimentation.ipynb` — novice score 3/5, flow score 5/5
- **Covers / works well:** `## 1`–`## 6` carries the 60%→80% aggregate gain into the security-slice regression, underpowered paired evidence, shadow/A-B design, and a justified no-canary decision. The exact result and underpowered-sample boundary are sound.
- **Issue:** `[P2]` `## 3 - Pair the Offline Comparison and Show Its Uncertainty` / `## 4 - Separate Shadow Replay from A/B Assignment`—add a compact picture distinguishing offline replay, shadow (observe, do not expose), and randomized exposure; these terms arrive quickly for a novice.

###### `12-role-based-tracks/ai-engineer/03-application-latency-and-cost/application-latency-and-cost.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** `## 1 - Reconcile the Ledger` to `## 7 - Package the Evidence and Hand Off` cleanly progresses from complete traces to percentiles, TTFT/TPOT, retry/cache accounting, a bottleneck decision, bounded telemetry, and a local-only report. Serial output rate is correctly separated from concurrent throughput.
- **Issue:** `[P2]` `## 1 - Reconcile the Ledger`—the reconciliation equation is a steep first formal step; state “sum the stages to recover this request's total” in words before symbols. A labeled request timeline at `## 3 - Separate TTFT, TPOT, and Useful Throughput` would anchor the new terms.

###### `12-role-based-tracks/ai-engineer/04-release-registry-and-lineage/release-registry-and-lineage.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** `## 1 - A Directory Is Not a Release` through `## 6 - Map the Portable Contract to Azure Evidence` separates paths, mutable aliases, compatibility, rollback graph, registry behavior, and cloud evidence. Schema pass is not mistaken for compatibility.
- **Issue:** `[P2]` `## 4 - Rollback Is a Graph, Not a Label`—a small release-node diagram showing accepted predecessor, dangling edge, and rejected self-edge would make the graph test memorable. The Azure boundary is correctly marked unvalidated.

###### `12-role-based-tracks/ai-engineer/05-production-feedback-and-drift/production-feedback-and-drift.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** `## 1 - Privacy-Safe Sampling Before Analysis` through `## 8 - Operate the Loop, Not the Notebook` preserves privacy, compares windows, distinguishes seven drift lenses, qualifies six-row uncertainty, clusters overlapping failures, versions candidate evaluation, and routes interventions to owners.
- **Issue:** `[P2]` `## 3 - Seven Drift Lenses, Seven Different Owners`—add a one-page lens → evidence → owner map. It would reduce the switching cost without diluting the explicit warning that these synthetic traces do not establish production prevalence.

###### `12-role-based-tracks/ai-engineer/06-capstone/capstone-evidence-workbook.ipynb` — novice score 2/5, flow score 4/5
- **Covers / works well:** `C0 - Scope and Evidence Inventory` through `C6 - Drift Iteration Decision and Unsupported Claims` assembles earlier technical reports into a release review; the README correctly says this is a writing workbook, not executed analysis.
- **Issue:** `[P2]` `C0 - Scope and Evidence Inventory`—link the worked “source-only hold” example at the point where learners start filling the workbook, and label which fields are imported evidence versus learner-authored judgments. Its low novice score is appropriate for a capstone, not a defect in the route.

##### Forward Deployed Engineer

###### `12-role-based-tracks/fde/01-discovery-and-success-criteria/discovery-and-success-criteria.ipynb` — novice score 4/5, flow score 4/5
- **Covers / works well:** `## 1 - See the Solution-First Failure` → stakeholder/workflow mapping → source boundary → forbidden cases → unknowns → design-readiness gate. It demonstrates why “choose a model” is premature and retains unresolved conflicts instead of inventing customer approval.
- **Issue:** `[P3]` `## 8 - What Changed for Riverside`—close with one explicit return question to architecture translation; the handoff is clear in the README but could be echoed in the notebook.

###### `12-role-based-tracks/fde/02-architecture-translation/architecture-translation.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** `## 1 - Start Without AI` → `## 3 - Use a Workflow Before an Agent` → safety boundaries, smallest architecture, ADR, and attack test. It earns complexity rather than treating the requested multi-agent design as a requirement.
- **Issue:** `[P2]` `## 2 - Search, Cited Answers, Drafting, and Fine-Tuning`—briefly define search, RAG, prompt-only generation, and fine-tuning in everyday language before comparing options. A side-by-side workflow/agent sketch would help at `## 3`.

###### `12-role-based-tracks/fde/03-data-onboarding-and-contracts/data-onboarding-and-contracts.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** `## Follow One Record by Hand Before Scaling Out` makes the later authority, mapping, lifecycle, cursor, retrieval-time authorization, deletion, and readiness checks traceable. It distinguishes local simulation from customer/API/IdP proof.
- **Issue:** `[P2]` `## 4 - Do Not Advance the Cursor Past an Unknown Contract`—add a small page-1/page-2 checkpoint diagram showing why the cursor must not advance on schema drift; pagination/cursor semantics are a sharp vocabulary jump.

###### `12-role-based-tracks/fde/04-identity-isolation-and-compliance/identity-isolation-and-compliance.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** `## 1 - First failure: the request lies about authority` → trusted context → retrieval twice → exact tool authorization → minimized audit. The notebook consistently labels code as local-static and disclaims IdP, residency, legal, and cloud proof.
- **Issue:** `[P2]` `## 2 - Build trusted context and stop on the first failed check`—show a static request → trusted lookup → policy → retrieval/tool → audit path; tenant/region/purpose/role all arrive together and merit a visual.

###### `12-role-based-tracks/fde/05-sla-capacity-and-commercials/sla-capacity-and-commercials.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** `## 1 - Break the average-only plan` through scenario, concurrency, cost, sensitivity, and SLA gates turns Riverside's inconsistent inputs into a bounded modeled recommendation, not a quote or achieved SLO.
- **Issue:** `[P2]` `## 1 - Break the average-only plan`—put a unit card (sessions/day ≠ requests/hour ≠ concurrent requests) before the arithmetic. The later explicit limits on p99, bursts, and correlated retries are good.

###### `12-role-based-tracks/fde/06-rollout-and-change-management/rollout-and-change-management.ipynb` — novice score 4/5, flow score 4/5
- **Covers / works well:** `## 2 - Shadow Disagreement Review` → cohort/window gates → abort and communication → owned go/no-go. It correctly teaches that traffic rollback cannot undo an already committed external action.
- **Issue:** `[P3]` `## 3 - Canary Cohorts Are Change Boundaries`—keep the synthetic/authorized-execution distinction adjacent to the first “GO”; a reader may otherwise mistake a passing fixture gate for customer authorization.

###### `12-role-based-tracks/fde/07-incident-response-and-recovery/incident-response-and-recovery.ipynb` — novice score 3/5, flow score 4/5
- **Covers / works well:** `## 1 · Declare and Contain First` → evidence/timeline → severity → safe communication → causal triage → regression → re-enable → postmortem. It prevents fixture facts becoming customer status claims.
- **Issue:** `[P2]` `## 5 · Triage Causally Across Seven Boundaries`—add a compact boundary map connecting the “first missing stable ID” to its owner; seven domains are a lot to hold in working memory.

###### `12-role-based-tracks/fde/08-handoff-and-customer-success/handoff-and-customer-success.ipynb` — novice score 3/5, flow score 3/5
- **Covers / works well:** `Part 1`–`Part 10` advances from folder-presence failure to alert ownership, runbooks, policy change, drills, and recurring backlog. It makes operational acceptance depend on demonstrated capability, not signatures or filenames.
- **Issue:** `[P2]` `## Part 3 - An Alert Fires. Then What?` / `## Parts 7 and 8 - Drills Gate Acceptance and Exit`—add a short route map before these tightly packed modules; one notebook asks the learner to track many handoff artifacts and authorities.

###### `12-role-based-tracks/fde/09-capstone/capstone-workbook.ipynb` — novice score 2/5, flow score 4/5
- **Covers / works well:** `## 0 · The Challenge` through `## 12 · Package Audit and Final Recommendation` preserves eleven conflicts/unknowns while building one traceable customer recommendation. Explicit BLOCKED/HOLD outputs are the right default for missing learner evidence.
- **Issue:** `[P2]` `## 1 · Evidence and Claim Control`—give a one-page “follow one conflict through criterion → architecture → test → owner” map before the full package. The capstone appropriately depends on Chapters 01–08 and AI Engineer foundations.

###### `12-role-based-tracks/fde/.venv/Lib/site-packages/chromadb/experimental/density_relevance.ipynb` — novice score 2/5, flow score 2/5 *(vendored dependency, not a Stage 12 chapter)*
- **Covers / works well:** `## Computing a distribution over distances` sketches a retrieval-relevance heuristic and labels its evaluation preliminary; it is not linked from FDE CONTENTS and should not be treated as a role lesson.
- **Issue:** `[P2]` `### Computing relevance using the density function`—`searchsorted` can return index 0 for a query distance below the first bin, making `cumulative_density[index - 1]` select the final bin. The notebook also installs packages and downloads SciQ, unlike the local-only Stage 12 route; keep it clearly marked as third-party material, not a validated curriculum asset.

#### Cross-notebook inconsistencies

- **Challenge-out integration:** the role-specific `Enter with` / `Challenge out` / `Hand off` blocks are additive and generally personalized to actual evidence (digest, release identity, bounded service envelope, receiving-owner drills). They feel integrated rather than bolted on. Repeated prerequisite language is slightly boilerplate, but useful at independent chapter boundaries.
- **Required versus optional Stage 10:** Stage 12's entry contract requires Stage 10/11 completion or equivalent evidence, while `ai-engineer/00-role-baseline-and-route.md` calls the full Agentic AI branch optional when agents are out of scope. Clarify that “optional” means role-specific agent depth, not waiving the common Stage 10 challenge-out gate.
- **Contents coverage:** links checked all resolve, but FDE chapter CONTENTS maps generally list only README and notebook while the directories also contain `templates/`, `cases/`, or `examples/`; AI Engineer capstone CONTENTS likewise omits `templates/`. This falls short of the curriculum's child-subdirectory map convention and hides useful handoff artifacts. The root maps and one-line chapter descriptions are otherwise clear.
- **Route distinction:** AI Engineer's data chapter audits training-corpus quality; FDE's data chapter governs customer-source onboarding, authority, refresh, and deletion. AI Engineer release/feedback and FDE rollout/incident/handoff overlap in vocabulary but not in role decisions; the shared Riverside setting makes that distinction legible.
- No authored `-theory.md` companions were found. README links to concept-owning earlier stages instead, which is appropriate for these applied synthesis tracks.

#### Top 5 prioritized improvements

1. **[P2] Complete FDE and AI capstone CONTENTS maps** with links to `templates/`, `cases/`, and `examples/` subdirectories; verify each map against its child assets.
2. **[P2] Clarify the Stage 10 gate** in `ai-engineer/00-role-baseline-and-route.md` and its “Optional Specializations” heading so optional depth does not read as optional prerequisite evidence.
3. **[P2] Add two static visuals:** the request-stage timeline in AI Engineer `## 3 - Separate TTFT, TPOT, and Useful Throughput`, and the trusted-context path in FDE `## 2 - Build trusted context and stop on the first failed check`.
4. **[P2] Reduce high-density concept jumps** with short everyday-language/unit lead-ins in FDE `02-architecture-translation` `## 2`, FDE `05-sla-capacity-and-commercials` `## 1`, and FDE `07-incident-response-and-recovery` `## 5`.
5. **[P3] Correct the FDE README list numbering** in “Language-model mechanics and adaptation” and keep the vendored `.venv` retrieval notebook explicitly outside curriculum navigation and validation claims.

No P1 blocker was found; notebooks were not executed.
