# Image Placement Plan — Phase 6

**Trigger:** User drops all generated images into `learning/_generated-images/` (flat directory).

**Goal:** Parallel subagents move each image to its correct `images/` subdirectory and insert a `![caption](images/filename.png)` reference into the right notebook cell.

---

## What Each Subagent Needs to Do

Each agent handles ONE chapter (one source directory):
1. Scan `learning/_generated-images/` for files matching its chapter's expected filenames
2. Create the target `images/` directory if it doesn't exist
3. Copy each matched file to the target directory
4. In the corresponding notebook, find the right markdown cell and insert the image reference
5. Report which images were placed and which were missing

---

## Placement Map (agent task per row)

| Agent | Source files | Target dir | Notebook | Notebook section to embed |
|---|---|---|---|---|
| A | `gradient-descent-convergence.png`, `chain-rule-computation-graph.png`, `free-kick-parabola-constraints.png` | `genai-prerequisites/00-math-foundations/images/` | `math-foundations-for-ml.ipynb` | Cell 1 (challenge), Part 3 intro, Part 5 intro |
| B | `regression-loss-landscape.png`, `overfitting-train-val-curves.png`, `lasso-ridge-coefficients.png` | `genai-prerequisites/01-ml-basics/images/` | `ml-basics.ipynb` | Part 1 header, Part 4 header, Part 6 header |
| C | `xor-not-linearly-separable.png`, `neural-network-forward-pass.png`, `depth-vs-width-decision-boundary.png` | `genai-prerequisites/02-neural-networks/images/` | `neural-networks-and-backprop.ipynb` | Part 1 intro, Part 2 intro, Part 4 intro |
| D | `convolution-filter-operation.png`, `feature-maps-by-layer.png`, `resnet-skip-connection.png` | `genai-prerequisites/03-cnns/images/` | `convolutional-neural-networks.ipynb` | Part 1 intro, Part 2 intro, Part 4 intro |
| E | `rnn-hidden-state-unrolled.png`, `vanishing-gradient-vs-timestep.png`, `lstm-gate-equations.png` | `genai-prerequisites/04-rnn-sequence-modeling/images/` | `rnn-sequence-modeling.ipynb` | Part 2 header, Part 3 header, Part 4 header |
| F | `bpe-merge-steps.png`, `embedding-space-pca.png`, `tokenization-pipeline.png` | `genai-prerequisites/05-tokenization/images/` | `tokenization-and-embeddings.ipynb` | Part 2 header, Part 4 header, Part 6 header |
| G | `gpu-memory-hierarchy.png`, `roofline-model.png`, `warp-simt-execution.png` | `ai-infrastructure/01-gpu-hardware/images/` | `gpu-hardware-foundations.ipynb` | Part 2 header, Part 3 header, Part 4 header |
| H | `memory-footprint-breakdown.png`, `fp32-fp16-bf16-number-line.png`, `gradient-checkpointing-tradeoff.png` | `ai-infrastructure/02-mixed-precision/images/` | `mixed-precision-and-memory.ipynb` | Part 1 header, Part 2 header, Part 4 header |
| I | `profiler-timeline-annotated.png`, `compute-vs-memory-bound.png` | `ai-infrastructure/03-profiling/images/` | `pytorch-profiling.ipynb` | Part 1 header, Part 3 header |
| J | `standard-attention-io.png`, `flash-attention-tiling.png`, `kv-cache-memory-gqa.png` | `ai-infrastructure/04-flash-attention/images/` | `flash-attention-internals.ipynb` | Part 1 header, Part 2 header, Part 6 header |
| K | `ddp-gradient-allreduce.png`, `fsdp-vs-ddp-memory.png`, `parallelism-strategy-matrix.png` | `ai-infrastructure/05-distributed-training/images/` | `distributed-training.ipynb` | Part 1 header, Part 2 header, Part 5 header |
| L | `quantization-rounding-error.png`, `gptq-vs-awq-perplexity.png`, `gguf-quantization-formats.png` | `ai-infrastructure/06-quantization/images/` | `quantization-in-depth.ipynb` | Part 1 header, Part 4 header, Part 5 header |
| M | `kv-cache-mechanism.png`, `continuous-batching-vs-static.png`, `speculative-decoding-accept-reject.png` | `ai-infrastructure/07-inference-systems/images/` | `inference-systems.ipynb` | Part 1 header, Part 2 header, Part 4 header |
| N | `triton-grid-block-thread.png`, `fused-vs-unfused-gelu.png`, `autotune-block-size-sweep.png` | `ai-infrastructure/08-triton-kernels/images/` | `triton-kernels.ipynb` | Part 1 header, Part 3 header, Part 5 header |
| O | `autograd-computation-graph.png`, `training-curves-keras-vs-pytorch.png` | `genai/00-pytorch-primer/images/` | `keras-to-pytorch-primer.ipynb` | Part 4 header, Part 5 header |
| P | `decoder-block-internals.png` | `genai/03-encoder-decoder/images/` | `encoder-decoder.ipynb` | Part 4 intro (after the CrossAttention class cell) |

**Special case — duplicate targets:**
`rnn-hidden-state-unrolled.png`, `vanishing-gradient-vs-timestep.png`, `lstm-gate-equations.png` must be copied to BOTH:
- `genai-prerequisites/04-rnn-sequence-modeling/images/` (Agent E above)
- `genai/01-rnns/images/` (Agent E also handles this copy)

---

## Notebook Embedding Format

Each image should be embedded in a markdown cell immediately before the code cell it illustrates. Use:

```markdown
![Brief description of what the image shows](images/filename.png)
```

The alt text should match the Teaching Job from `generate-all-images.md` — one sentence, plain English, no LaTeX.

**Do NOT** embed images inside existing markdown cells that already have substantial prose — insert a new markdown cell immediately before the relevant Part header cell.

---

## Subagent Instructions (standard for all agents)

Each agent receives:
1. The list of filenames it is responsible for (from the Placement Map above)
2. The source directory: `c:\repos\ai-portfolio\learning\_generated-images\`
3. The target `images/` directory path
4. The notebook path

Agent steps:
1. Use `tool_search` to load `edit_notebook_file` and `copilot_getNotebookSummary`
2. For each expected filename: check if it exists in `learning/_generated-images/`
3. If present: copy (using `run_in_terminal` with `Copy-Item`) to the target `images/` directory
4. Use `copilot_getNotebookSummary` to find the right cell (look for the Part N header markdown cell)
5. Use `edit_notebook_file` with `editType=insert` to add the image markdown cell BEFORE that Part header
6. Report: which images were placed, which were missing from `_generated-images/`

---

## Pre-flight Checks Before Launching Agents

Before running the placement agents, verify:
- `learning/_generated-images/` exists and contains the generated PNG files
- Run: `Get-ChildItem learning/_generated-images/ -Name` to confirm filenames match the expected names exactly (case-sensitive, hyphens not underscores)
- Any filename mismatch must be renamed before running agents — the matching is exact

---

## Commit Strategy

After all 16 agents complete:
1. `git add learning/genai-prerequisites/ learning/ai-infrastructure/ learning/genai/`
2. `git commit -m "Phase 6: add generated images to all chapters and wire notebook references"`

---

## What To Do If an Image Is Missing

If a file is absent from `_generated-images/`:
- The agent logs "MISSING: filename.png — skipping" and continues with the others
- After placement: re-run generation for any missing images, drop them in `_generated-images/`, and re-run only the affected agent(s)
