# All Images — Single Generation Prompt

Drop all generated images into `learning/_generated-images/` (create the folder).
Name each file **exactly** as shown in the filename column — the placement script matches on filename.

## Master Mapping Table (45 images total)

| # | Filename | Target directory | Notebook |
|---|---|---|---|
| 1 | `gradient-descent-convergence.png` | `genai-prerequisites/00-math-foundations/images/` | `math-foundations-for-ml.ipynb` |
| 2 | `chain-rule-computation-graph.png` | `genai-prerequisites/00-math-foundations/images/` | `math-foundations-for-ml.ipynb` |
| 3 | `free-kick-parabola-constraints.png` | `genai-prerequisites/00-math-foundations/images/` | `math-foundations-for-ml.ipynb` |
| 4 | `regression-loss-landscape.png` | `genai-prerequisites/01-ml-basics/images/` | `ml-basics.ipynb` |
| 5 | `overfitting-train-val-curves.png` | `genai-prerequisites/01-ml-basics/images/` | `ml-basics.ipynb` |
| 6 | `lasso-ridge-coefficients.png` | `genai-prerequisites/01-ml-basics/images/` | `ml-basics.ipynb` |
| 7 | `xor-not-linearly-separable.png` | `genai-prerequisites/02-neural-networks/images/` | `neural-networks-and-backprop.ipynb` |
| 8 | `neural-network-forward-pass.png` | `genai-prerequisites/02-neural-networks/images/` | `neural-networks-and-backprop.ipynb` |
| 9 | `depth-vs-width-decision-boundary.png` | `genai-prerequisites/02-neural-networks/images/` | `neural-networks-and-backprop.ipynb` |
| 10 | `convolution-filter-operation.png` | `genai-prerequisites/03-cnns/images/` | `convolutional-neural-networks.ipynb` |
| 11 | `feature-maps-by-layer.png` | `genai-prerequisites/03-cnns/images/` | `convolutional-neural-networks.ipynb` |
| 12 | `resnet-skip-connection.png` | `genai-prerequisites/03-cnns/images/` | `convolutional-neural-networks.ipynb` |
| 13 | `rnn-hidden-state-unrolled.png` | `genai-prerequisites/04-rnn-sequence-modeling/images/` | `rnn-sequence-modeling.ipynb` |
| 14 | `vanishing-gradient-vs-timestep.png` | `genai-prerequisites/04-rnn-sequence-modeling/images/` | `rnn-sequence-modeling.ipynb` |
| 15 | `lstm-gate-equations.png` | `genai-prerequisites/04-rnn-sequence-modeling/images/` | `rnn-sequence-modeling.ipynb` |
| 16 | `bpe-merge-steps.png` | `genai-prerequisites/05-tokenization/images/` | `tokenization-and-embeddings.ipynb` |
| 17 | `embedding-space-pca.png` | `genai-prerequisites/05-tokenization/images/` | `tokenization-and-embeddings.ipynb` |
| 18 | `tokenization-pipeline.png` | `genai-prerequisites/05-tokenization/images/` | `tokenization-and-embeddings.ipynb` |
| 19 | `gpu-memory-hierarchy.png` | `ai-infrastructure/01-gpu-hardware/images/` | `gpu-hardware-foundations.ipynb` |
| 20 | `roofline-model.png` | `ai-infrastructure/01-gpu-hardware/images/` | `gpu-hardware-foundations.ipynb` |
| 21 | `warp-simt-execution.png` | `ai-infrastructure/01-gpu-hardware/images/` | `gpu-hardware-foundations.ipynb` |
| 22 | `memory-footprint-breakdown.png` | `ai-infrastructure/02-mixed-precision/images/` | `mixed-precision-and-memory.ipynb` |
| 23 | `fp32-fp16-bf16-number-line.png` | `ai-infrastructure/02-mixed-precision/images/` | `mixed-precision-and-memory.ipynb` |
| 24 | `gradient-checkpointing-tradeoff.png` | `ai-infrastructure/02-mixed-precision/images/` | `mixed-precision-and-memory.ipynb` |
| 25 | `profiler-timeline-annotated.png` | `ai-infrastructure/03-profiling/images/` | `pytorch-profiling.ipynb` |
| 26 | `compute-vs-memory-bound.png` | `ai-infrastructure/03-profiling/images/` | `pytorch-profiling.ipynb` |
| 27 | `standard-attention-io.png` | `ai-infrastructure/04-flash-attention/images/` | `flash-attention-internals.ipynb` |
| 28 | `flash-attention-tiling.png` | `ai-infrastructure/04-flash-attention/images/` | `flash-attention-internals.ipynb` |
| 29 | `kv-cache-memory-gqa.png` | `ai-infrastructure/04-flash-attention/images/` | `flash-attention-internals.ipynb` |
| 30 | `ddp-gradient-allreduce.png` | `ai-infrastructure/05-distributed-training/images/` | `distributed-training.ipynb` |
| 31 | `fsdp-vs-ddp-memory.png` | `ai-infrastructure/05-distributed-training/images/` | `distributed-training.ipynb` |
| 32 | `parallelism-strategy-matrix.png` | `ai-infrastructure/05-distributed-training/images/` | `distributed-training.ipynb` |
| 33 | `quantization-rounding-error.png` | `ai-infrastructure/06-quantization/images/` | `quantization-in-depth.ipynb` |
| 34 | `gptq-vs-awq-perplexity.png` | `ai-infrastructure/06-quantization/images/` | `quantization-in-depth.ipynb` |
| 35 | `gguf-quantization-formats.png` | `ai-infrastructure/06-quantization/images/` | `quantization-in-depth.ipynb` |
| 36 | `kv-cache-mechanism.png` | `ai-infrastructure/07-inference-systems/images/` | `inference-systems.ipynb` |
| 37 | `continuous-batching-vs-static.png` | `ai-infrastructure/07-inference-systems/images/` | `inference-systems.ipynb` |
| 38 | `speculative-decoding-accept-reject.png` | `ai-infrastructure/07-inference-systems/images/` | `inference-systems.ipynb` |
| 39 | `triton-grid-block-thread.png` | `ai-infrastructure/08-triton-kernels/images/` | `triton-kernels.ipynb` |
| 40 | `fused-vs-unfused-gelu.png` | `ai-infrastructure/08-triton-kernels/images/` | `triton-kernels.ipynb` |
| 41 | `autotune-block-size-sweep.png` | `ai-infrastructure/08-triton-kernels/images/` | `triton-kernels.ipynb` |
| 42 | `autograd-computation-graph.png` | `genai/00-pytorch-primer/images/` | `keras-to-pytorch-primer.ipynb` |
| 43 | `training-curves-keras-vs-pytorch.png` | `genai/00-pytorch-primer/images/` | `keras-to-pytorch-primer.ipynb` |
| 44 | `rnn-hidden-state-unrolled.png` | `genai/01-rnns/images/` | (see note — same image, different track) |
| 45 | `decoder-block-internals.png` | `genai/03-encoder-decoder/images/` | `encoder-decoder.ipynb` |

> **Note on duplicates:** `rnn-hidden-state-unrolled.png`, `vanishing-gradient-vs-timestep.png`, and `lstm-gate-equations.png` appear in both `genai-prerequisites/04-rnn-sequence-modeling/images/` and `genai/01-rnns/images/`. Generate once; the placement script will copy to both locations.

---

## Shared Style Rules (apply to ALL images)

- **Size:** 1600 × 900 px minimum (16:9), PNG
- **Background:** dark graphite `#1e1e2e`
- **Primary text:** `#cdd6f4` (light ivory)
- **Data flow / teal accent:** `#4ecdc4` or `#89dceb`
- **Trainable / amber accent:** `#f5a623` or `#fab387`
- **Error / coral accent:** `#e07b54` or `#f38ba8`
- **No logos, no photorealism, no gradients in fills, no tiny unreadable text**
- **Flat vector / technical diagram aesthetic throughout**

---

## Prompts — Track 1: genai-prerequisites

---

### Image 1 · `gradient-descent-convergence.png`
**Chapter:** P-0 Math Foundations  **Section:** Part 3

Flat vector data visualization, wide 16:9, dark graphite background #1e1e2e. A smooth U-shaped loss curve with x-axis "launch angle (degrees)" and y-axis "penalty". Ten amber (#f5a623) dots step down the left slope toward the minimum, connected by thin lines showing the gradient descent path. The final dot lands near the bottom of the bowl. A small arrow at each dot points in the direction of the negative gradient. Ivory axis labels, coral (#e07b54) dot for starting point, teal (#4ecdc4) dot for ending point. No logos, no photorealism, no gradients, no tiny text.

---

### Image 2 · `chain-rule-computation-graph.png`
**Chapter:** P-0 Math Foundations  **Section:** Part 5

Flat vector technical diagram, wide 16:9, dark graphite background #1e1e2e. Three nodes in a left-to-right chain: circle "x" → amber box "f" → amber box "g" → coral box "loss". Forward arrows in teal (#4ecdc4) labelled "df/dx" and "dg/df". A thick red reverse arrow below the chain, flowing right-to-left, labelled "d(loss)/dx = df/dx × dg/df" with the chain rule product shown explicitly. Ivory labels. No logos, no photorealism.

---

### Image 3 · `free-kick-parabola-constraints.png`
**Chapter:** P-0 Math Foundations  **Section:** Opening challenge

Flat vector physics diagram, wide 16:9, dark graphite background #1e1e2e. 2D side-view of a football free kick: kick origin at left, a defensive wall at 9.15m marked as a teal (#4ecdc4) vertical rectangle labeled "Wall 1.8m", goal at 20m with crossbar marked in amber (#f5a623) labeled "Crossbar 2.44m". A muted coral parabolic trajectory arc passing over the wall and under the crossbar. A shaded green scoring window at the goal. Key dimensions annotated in ivory. No logos, no photorealism.

---

### Image 4 · `regression-loss-landscape.png`
**Chapter:** P-1 ML Basics  **Section:** Part 1

Flat vector data visualization, wide 16:9, dark graphite background #1e1e2e. A 3D-style contour map of an MSE loss bowl over two axes labeled "weight" and "bias". Contour lines in muted teal (#4ecdc4). A spiral path of ten amber (#f5a623) dots descends from a high-loss plateau toward the minimum. The minimum is marked with a coral (#e07b54) star. Ivory axis labels. No logos, no photorealism, no gradients in fills.

---

### Image 5 · `overfitting-train-val-curves.png`
**Chapter:** P-1 ML Basics  **Section:** Part 6

Flat vector data visualization, wide 16:9, dark graphite background #1e1e2e. Two side-by-side line charts. Left panel titled "Training loss only (50 samples)": a teal line falling steeply to near zero. Right panel titled "Train vs. validation": teal train loss falling, coral (#e07b54) val loss forming a U-shape, crossing at epoch 15. A vertical amber dashed line at the crossing marks "Early stop here". Ivory axis labels. No logos, no photorealism.

---

### Image 6 · `lasso-ridge-coefficients.png`
**Chapter:** P-1 ML Basics  **Section:** Part 4

Flat vector bar chart, wide 16:9, dark graphite background #1e1e2e. Two rows of 8 bars each. Top row labeled "Ridge": all bars non-zero, shrunk, amber (#f5a623). Bottom row labeled "Lasso": 3 bars are exactly zero shown as coral (#e07b54), 5 bars are teal (#4ecdc4). Feature names as ivory labels below each bar: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude. Subtitle: "Lasso selects. Ridge shrinks." No logos, no photorealism.

---

### Image 7 · `xor-not-linearly-separable.png`
**Chapter:** P-2 Neural Networks  **Section:** Part 1

Flat vector 2D scatter plot, wide 16:9, dark graphite background #1e1e2e. Four large circular data points: two steel-blue (#45b7d1) circles at (0,0) and (1,1) labeled "0 (not desirable)", two coral (#e07b54) circles at (0,1) and (1,0) labeled "1 (desirable)". A dashed gray diagonal line cuts through the middle, clearly failing to separate the groups. Thin grid lines. Ivory axis labels "x1 (income)" and "x2 (crime, inverted)". Title: "XOR: no straight line separates desirable from not". No logos, no photorealism.

---

### Image 8 · `neural-network-forward-pass.png`
**Chapter:** P-2 Neural Networks  **Section:** Part 2

Flat vector neural network architecture diagram, wide 16:9, dark graphite background #1e1e2e. Three vertical columns: LEFT — two teal (#4ecdc4) circles labeled "x₁" and "x₂". MIDDLE — two amber (#f5a623) circles labeled "h₁" and "h₂" with small "ReLU" text. RIGHT — one coral (#e07b54) circle labeled "ŷ" with "sigmoid" text. Thin ivory arrows connect every input to every hidden node (labeled "W₁"), and every hidden to the output (labeled "W₂"). Bias nodes shown as small "b" circles. No logos, no photorealism.

---

### Image 9 · `depth-vs-width-decision-boundary.png`
**Chapter:** P-2 Neural Networks  **Section:** Part 4

Flat vector two-panel contour plot, wide 16:9, dark graphite background #1e1e2e. Both panels show the same spiral dataset: two interleaved spiral arms, one teal (#4ecdc4) and one coral (#e07b54), ~75 small circular points per arm. LEFT panel titled "Wide (256 neurons, 78% acc)": irregular, angular decision boundary that partially misses the spiral. RIGHT panel titled "Deep (4 layers, 94% acc)": smooth, tight boundary closely following the spiral shape. Semi-transparent region fills. White title text. No logos, no photorealism.

---

### Image 10 · `convolution-filter-operation.png`
**Chapter:** P-3 CNNs  **Section:** Part 1

Flat vector technical diagram, wide 16:9, dark graphite background #1e1e2e. LEFT: a 5×5 grid labeled "Input" with small ivory numbers. CENTER: a 3×3 teal (#4ecdc4) filter/kernel grid with weight values, overlaid on the input with a dashed teal border showing the current sliding position. RIGHT: a 3×3 amber (#f5a623) grid labeled "Feature Map" with output values. A labeled arrow shows "dot product = output value" connecting filter to output cell. White sans-serif labels. No logos, no photorealism.

---

### Image 11 · `feature-maps-by-layer.png`
**Chapter:** P-3 CNNs  **Section:** Part 2

Flat vector three-panel technical diagram, wide 16:9, dark graphite background #1e1e2e. Panel 1 (left): a 28×28 grayscale handwritten digit "3", labeled "Input 1×28×28". Right-pointing teal arrow. Panel 2 (center): a 2×4 grid of 8 small heatmap thumbnails using viridis colormap, labeled "After Conv1 + ReLU". Right-pointing teal arrow. Panel 3 (right): a 2×4 grid of 8 even smaller heatmap thumbnails, labeled "After Conv2 + Pool (7×7)". White sans-serif labels. No logos, no photorealism.

---

### Image 12 · `resnet-skip-connection.png`
**Chapter:** P-3 CNNs  **Section:** Part 4

Flat vector split technical diagram, wide 16:9, dark graphite background #1e1e2e. LEFT titled "Plain Block": vertical stack of rounded boxes: Conv3×3 → BN → ReLU → Conv3×3 → BN. A thin fading coral (#e07b54) dashed arrow runs backward labeled "vanishing gradient". RIGHT titled "Residual Block (ResNet)": same stack, but a thick teal (#4ecdc4) curved arrow bypasses the entire stack labeled "+x", merging at an amber circle ⊕. A bold teal backward arrow labeled "∂L/∂x = 1 + ∂F/∂x". White labels. No logos, no photorealism.

---

### Image 13 · `rnn-hidden-state-unrolled.png`
**Chapter:** P-4 RNN + genai/01-rnns  **Section:** Part 2

Flat vector technical diagram, wide 16:9, dark graphite background #1e1e2e. An RNN unrolled across three timesteps t−1, t, t+1. Each step shows an amber (#f5a623) box labeled "RNN cell" with two inputs: input vector xₜ from below (teal arrow) and hidden state hₜ₋₁ from the left (amber arrow). Each cell outputs hₜ flowing right. The tanh gate shown as a small circle inside each cell. Below the diagram: the equation hₜ = tanh(Wₕ hₜ₋₁ + Wₓ xₜ + b) in large readable ivory text. Right side: a coral warning box "repeat multiplication → vanishing gradient". No logos, no photorealism.

---

### Image 14 · `vanishing-gradient-vs-timestep.png`
**Chapter:** P-4 RNN + genai/01-rnns  **Section:** Part 3

Flat vector side-by-side line chart, wide 16:9, dark graphite background #1e1e2e. Two panels sharing the same x-axis labeled "Distance from loss (timesteps 1→50)". LEFT panel "Vanilla RNN": a steeply decaying coral (#e07b54) curve starting near 1.0 at step 1 and approaching 0.0 by step 20. RIGHT panel "LSTM": a roughly flat teal (#4ecdc4) line staying near 0.8 across all 50 timesteps. Both panels have a y-axis labeled "Gradient norm". Title: "Gating preserves gradient signal over time". No logos, no photorealism.

---

### Image 15 · `lstm-gate-equations.png`
**Chapter:** P-4 RNN + genai/01-rnns  **Section:** Part 4

Flat vector technical infographic, wide 16:9, dark graphite background #1e1e2e. Four aligned LSTM gate equation panels labeled forget gate, input gate, cell state update, output gate. Each panel shows the sigmoid or tanh activation icon, the weight matrices Wf Wi Wc Wo as amber (#f5a623) rectangles, and the resulting gate vector as a colored bar. Arrows show the cell state (teal highway), hidden state (amber), gate activations (coral). All equations in readable symbolic math. Ivory labels. No logos, no photorealism.

---

### Image 16 · `bpe-merge-steps.png`
**Chapter:** P-5 Tokenization  **Section:** Part 2

Flat vector technical infographic, wide 16:9, dark graphite background #1e1e2e. Three panels labeled "Raw", "10 merges", "20 merges". Each panel shows the word "non-disclosure" split into progressively fewer tokens, represented as labeled rounded rectangles. In "Raw": 14 individual character boxes. In "10 merges": 4 boxes. In "20 merges": 2 boxes labeled "non" and "disclosure" in teal (#4ecdc4). Ivory text, amber merge-arrow icons between panels. No logos, no photorealism.

---

### Image 17 · `embedding-space-pca.png`
**Chapter:** P-5 Tokenization  **Section:** Part 4

Flat vector data visualization, wide 16:9, dark graphite background #1e1e2e. Two side-by-side 2D scatter plots labeled "Before training (random)" and "After training (clustered)". Left plot: scattered points with no structure, ivory labels. Right plot: legal synonym clusters — "contract" and "agreement" near each other, "liability" and "indemnity" near each other — with teal (#4ecdc4) ellipses around each cluster. Amber legend. No logos, no photorealism.

---

### Image 18 · `tokenization-pipeline.png`
**Chapter:** P-5 Tokenization  **Section:** Part 6

Flat vector pipeline diagram, wide 16:9, dark graphite background #1e1e2e. Five left-to-right stages connected by teal (#4ecdc4) arrows: raw string box ("non-disclosure"), BPE tokens box (two rounded rectangles "non" + "disclosure"), integer IDs box (numbers 3421, 8872), embedding vectors box (a 2×4 matrix of floats), model input box (transformer icon). Each stage has a short ivory label. No logos, no photorealism.

---

## Prompts — Track 2: ai-infrastructure

---

### Image 19 · `gpu-memory-hierarchy.png`
**Chapter:** Ch1 GPU Hardware  **Section:** Part 2

Flat vector technical diagram, wide 16:9, dark graphite background #1e1e2e. A vertical pyramid with four tiers. Base (largest): "HBM — 80 GB, 2 TB/s" in teal (#4ecdc4). Second tier: "L2 Cache — 40 MB, 12 TB/s" in muted teal. Third tier: "L1 / Shared SRAM — 228 KB/SM" in amber (#f5a623). Apex (smallest): "Registers — 256 KB/SM, fastest" in coral (#e07b54). Each tier labeled with size and bandwidth in ivory. Upward arrows showing "faster, smaller, closer to compute". No logos, no photorealism.

---

### Image 20 · `roofline-model.png`
**Chapter:** Ch1 GPU Hardware  **Section:** Part 3

Flat vector data visualization, wide 16:9, dark graphite background #1e1e2e. Log-log axes: x = "Arithmetic Intensity (FLOP/byte)", y = "Performance (TFLOPS)". A teal (#4ecdc4) roof-line bends at the ridge point labeled "Ridge point — 164 FLOP/byte (RTX 4090)". Left of ridge: sloped "Memory-bound" region shaded in coral (#e07b54). Right: flat "Compute-bound" region in muted teal. A labeled amber dot for "LLM decode (1 token)" sits far left in the memory-bound region. Ivory labels. No logos, no photorealism.

---

### Image 21 · `warp-simt-execution.png`
**Chapter:** Ch1 GPU Hardware  **Section:** Part 4

Flat vector technical diagram, wide 16:9, dark graphite background #1e1e2e. A grid of 32 small squares (threads) grouped into one warp. A branch condition splits them: 20 threads take "if" path (teal, #4ecdc4), 12 threads are grayed out (disabled, #45475a). Two serial passes shown below: Pass 1 executes the teal group, Pass 2 executes the remaining. An arrow shows this doubles execution time. Label: "Warp divergence: serialized execution". Ivory labels. No logos, no photorealism.

---

### Image 22 · `memory-footprint-breakdown.png`
**Chapter:** Ch2 Mixed Precision  **Section:** Part 1

Flat vector stacked bar chart, wide 16:9, dark graphite background #1e1e2e. Two side-by-side bars: "fp32" and "bf16". Each bar stacked with four segments: coral (#e07b54) "Parameters", amber (#f5a623) "Gradients", teal (#4ecdc4) "Optimizer states (fp32 Adam)", muted purple "Activations". Ivory segment labels showing GB. A dashed red horizontal line labeled "A10G limit: 24 GB". The bf16 bar is roughly 2× shorter for params/grads. Title: "GPT-2-Medium (355M) training memory footprint". No logos, no photorealism.

---

### Image 23 · `fp32-fp16-bf16-number-line.png`
**Chapter:** Ch2 Mixed Precision  **Section:** Part 2

Flat vector technical infographic, wide 16:9, dark graphite background #1e1e2e. Three rows: fp32, fp16, bf16. Each row shows a bit-layout diagram (32/16/16 total bits) split into: 1 gray sign bit, orange exponent bits, teal (#4ecdc4) mantissa bits. A vertical coral (#e07b54) line on fp16 marks the overflow threshold labeled "Max: 65,504 — LLM gradients exceed this". A label shows bf16 has the same exponent width as fp32. Ivory labels. No logos, no photorealism.

---

### Image 24 · `gradient-checkpointing-tradeoff.png`
**Chapter:** Ch2 Mixed Precision  **Section:** Part 4

Flat vector dual-axis line chart, wide 16:9, dark graphite background #1e1e2e. X-axis: "Checkpointing frequency (every N layers)". Left y-axis teal (#4ecdc4) line: "Peak memory (GB)", falling from left to right. Right y-axis coral (#e07b54) line: "Compute overhead (%)", rising from left to right. An amber (#f5a623) vertical dashed line marks the sweet spot where memory savings plateau and compute cost begins rising steeply. Title: "Gradient checkpointing: memory vs. compute tradeoff". Ivory labels. No logos, no photorealism.

---

### Image 25 · `profiler-timeline-annotated.png`
**Chapter:** Ch3 Profiling  **Section:** Part 1

Flat vector technical diagram styled as a profiler timeline, wide 16:9, dark graphite background #1e1e2e. A horizontal timeline bar divided into four labeled segments: "data_prep" (teal, narrow), "forward" (amber, medium), "backward" (coral, widest — ~2.5× forward width), "optimizer_step" (muted green, medium). Each segment has a bracket below listing example operator names in small monospace text. A vertical dashed line labeled "Step boundary" at the right. X-axis labeled "Wall time (ms)". Title: "torch.profiler — Single Training Step". Color legend. No logos, no photorealism.

---

### Image 26 · `compute-vs-memory-bound.png`
**Chapter:** Ch3 Profiling  **Section:** Part 3

Flat vector roofline diagram on log-log axes, wide 16:9, dark graphite background #1e1e2e. A teal (#4ecdc4) roof-line curve bends at the ridge point. Two large scatter dots: "matmul (Q@K^T)" in coral (#e07b54) plotted far right near the compute roof, and "softmax (S×S)" in amber (#f5a623) plotted far left on the memory-bound slope. A vertical dashed amber line at the ridge point. Arrows explaining "compute-bound" and "memory-bound". X-axis: "Arithmetic Intensity (FLOP/byte)". Y-axis: "Throughput (TFLOPS)". Ivory labels. No logos, no photorealism.

---

### Image 27 · `standard-attention-io.png`
**Chapter:** Ch4 FlashAttention  **Section:** Part 1

Flat vector systems diagram, wide 16:9, dark graphite background #1e1e2e. Left column: Q, K, V matrices as teal (#4ecdc4) boxes labeled "in HBM". Center: a large coral (#e07b54) box "S = QKᵀ (S×S matrix)" in HBM. Right: P=softmax(S) in HBM as amber box, then O=PV in HBM as teal box. Fat slow red arrows between every HBM step, each labeled with a round-trip count. Title: "Standard attention: O(S²) HBM reads". Ivory labels. No logos, no photorealism.

---

### Image 28 · `flash-attention-tiling.png`
**Chapter:** Ch4 FlashAttention  **Section:** Part 2

Flat vector systems diagram, wide 16:9, dark graphite background #1e1e2e. Left: Q, K, V in HBM (teal boxes). Center: a small amber (#f5a623) SRAM block labeled "Tile (BLOCK×d_head) — fits in SRAM". A fast teal looping arrow inside the SRAM block shows tiled computation; no large S×S matrix appears in HBM. Right: output O written directly to HBM (one arrow). Label: "S×S never touches HBM". Title: "FlashAttention: O(S²/M) HBM reads". Ivory labels. No logos, no photorealism.

---

### Image 29 · `kv-cache-memory-gqa.png`
**Chapter:** Ch4 FlashAttention  **Section:** Part 6

Flat vector bar chart, wide 16:9, dark graphite background #1e1e2e. Three vertical bars: "MHA" (coral, tallest, labeled "32 KV heads, ~2 GB"), "GQA-8" (teal, 8× shorter, labeled "4 KV heads — used in LLaMA-3"), "MQA" (amber, shortest, labeled "1 KV head"). Y-axis: "KV cache (GB) at S=2048". A brace labeled "8× reduction" between MHA and GQA-8 bars. Title: "KV Cache: MHA vs. GQA vs. MQA at S=2048". Ivory labels. No logos, no photorealism.

---

### Image 30 · `ddp-gradient-allreduce.png`
**Chapter:** Ch5 Distributed Training  **Section:** Part 1

Flat vector systems diagram, wide 16:9, dark graphite background #1e1e2e. Four GPU boxes arranged in a ring, connected by teal (#4ecdc4) arrows showing the ring-allreduce communication. Each GPU box shows an amber gradient tensor bar chart (different heights = different local gradients). A circular teal arrow flows clockwise labeled "All-reduce". After the all-reduce: each GPU shows identical teal bars (same height = averaged gradient). Label: "DDP: ring all-reduce → averaged gradients on every GPU". Ivory labels. No logos, no photorealism.

---

### Image 31 · `fsdp-vs-ddp-memory.png`
**Chapter:** Ch5 Distributed Training  **Section:** Part 2

Flat vector comparison diagram, wide 16:9, dark graphite background #1e1e2e. LEFT panel titled "DDP": 4 GPU boxes each containing a full amber (#f5a623) model copy labeled "Full model (840 GB)". RIGHT panel titled "FSDP": 4 GPU boxes each containing only a small teal (#4ecdc4) shard labeled "1/4 model". Coral (#e07b54) all-gather arrows between FSDP boxes labeled "all-gather when needed". Memory labels: "DDP: 840 GB/GPU" (coral warning) vs "FSDP: ~210 GB/GPU" (teal OK). Ivory labels. No logos, no photorealism.

---

### Image 32 · `parallelism-strategy-matrix.png`
**Chapter:** Ch5 Distributed Training  **Section:** Part 5

Flat vector heatmap grid, wide 16:9, dark graphite background #1e1e2e. Y-axis: model size (≤3B, 3–13B, 13–70B, 70B+). X-axis: GPU count (1, 2–4, 8–16, 64+). Each cell contains a strategy label in ivory: "Single GPU + ckpt", "FSDP", "FSDP + LoRA", "3D Parallel". Cells colored light teal (simple) to amber (moderate) to coral (complex). Title: "Parallelism Strategy Selection". No logos, no photorealism.

---

### Image 33 · `quantization-rounding-error.png`
**Chapter:** Ch6 Quantization  **Section:** Part 1

Flat vector technical diagram, wide 16:9, dark graphite background #1e1e2e. A number line from −1.0 to +1.0. Above the line: 256 evenly-spaced teal tick marks (int8 levels). A bell-curve density plot (steelblue fill) shows normal distribution of weights centered at 0 with σ=0.02. Small coral (#e07b54) vertical error bars of uniform height sit atop each tick mark, labeled "max rounding error = scale/2". Title: "int8 quantization: 256 uniform levels, rounding error bounded by scale/2". Ivory labels. No logos, no photorealism.

---

### Image 34 · `gptq-vs-awq-perplexity.png`
**Chapter:** Ch6 Quantization  **Section:** Part 4

Flat vector line chart, wide 16:9, dark graphite background #1e1e2e. X-axis: bit-width (8, 6, 4, 3). Y-axis: "Perplexity delta vs. bf16 baseline" (0 to 8). Two curves: teal (#4ecdc4) solid "GPTQ" with points (8,0.0), (6,0.1), (4,0.8), (3,4.2). Coral (#e07b54) dashed "AWQ" with points (8,0.0), (6,0.1), (4,0.5), (3,2.8). A horizontal amber dashed line at y=1.0 labeled "Acceptable threshold". Title: "GPTQ vs AWQ: perplexity degradation by bit-width (LLaMA-3-7B)". Ivory labels. No logos, no photorealism.

---

### Image 35 · `gguf-quantization-formats.png`
**Chapter:** Ch6 Quantization  **Section:** Part 5

Flat vector comparison table, wide 16:9, dark graphite background #1e1e2e. Four rows: Q4_K_M, Q5_K_M, Q8_0, F16. Columns: format name, bits-per-weight (amber bar showing 4.5 / 5.5 / 8.5 / 16), memory for 7B model (GB label), relative quality (teal bar). The Q4_K_M row is highlighted in coral as the Riverside recommendation. Title: "GGUF formats for local MacBook deployment". Ivory labels. No logos, no photorealism.

---

### Image 36 · `kv-cache-mechanism.png`
**Chapter:** Ch7 Inference Systems  **Section:** Part 1

Flat vector sequence diagram, wide 16:9, dark graphite background #1e1e2e. LEFT section labeled "Prefill": all prompt tokens processed in parallel, teal (#4ecdc4) arrows from each token to the KV cache list (growing teal list). RIGHT section labeled "Decode": one new token per step, a single amber (#f5a623) arrow appending one K/V pair to the growing cache. The cache list visually grows step by step. Label: "Prompt computed once; each new token adds 1 K/V pair". Ivory labels. No logos, no photorealism.

---

### Image 37 · `continuous-batching-vs-static.png`
**Chapter:** Ch7 Inference Systems  **Section:** Part 2

Flat vector timeline diagram, wide 16:9, dark graphite background #1e1e2e. TOP half titled "Static batching": a long request (amber) and a short request (teal) start together; after the short request finishes, its GPU slot shows coral hatching (idle/wasted) while the long request continues. BOTTOM half titled "Continuous batching": as soon as the short request finishes, a new request (amber) immediately fills the slot. Efficiency labels on right: "Static: ~60% GPU util", "Continuous: ~95% GPU util". Ivory labels. No logos, no photorealism.

---

### Image 38 · `speculative-decoding-accept-reject.png`
**Chapter:** Ch7 Inference Systems  **Section:** Part 4

Flat vector pipeline diagram, wide 16:9, dark graphite background #1e1e2e. LEFT: a small teal (#4ecdc4) box "Draft model (70M)" proposes 5 tokens in sequence (shown as 5 small token boxes). RIGHT: a large amber (#f5a623) box "Verifier model (7B)" checks all 5 in one parallel forward pass. Tokens 1–3: teal checkmark "accepted". Tokens 4–5: coral (#e07b54) cross "rejected". Time comparison below: "5 draft calls + 1 verifier call vs. 5 sequential verifier calls". Ivory labels. No logos, no photorealism.

---

### Image 39 · `triton-grid-block-thread.png`
**Chapter:** Ch8 Triton Kernels  **Section:** Part 1

Flat vector hierarchy diagram, wide 16:9, dark graphite background #1e1e2e. Three levels: TOP — a large teal (#4ecdc4) box labeled "CUDA Grid (gridDim.x × gridDim.y)" containing a 4×4 grid of smaller amber (#f5a623) boxes labeled "Thread Block". Each block contains a 4×4 grid of tiny ivory squares labeled "Thread". Bracket annotations show "gridDim", "blockDim", "threadIdx". Label: "Each @triton.jit call = one Thread Block". Ivory labels. No logos, no photorealism.

---

### Image 40 · `fused-vs-unfused-gelu.png`
**Chapter:** Ch8 Triton Kernels  **Section:** Part 3

Flat vector pipeline diagram, wide 16:9, dark graphite background #1e1e2e. TOP half titled "Unfused (2 kernels, 4 HBM accesses)": HBM box → load activations (coral slow arrow) → add bias (amber op) → write to HBM (coral slow arrow) → load again (coral slow arrow) → apply GELU (amber op) → write to HBM (coral slow arrow). BOTTOM half titled "Fused Triton kernel (2 HBM accesses)": HBM box → load (teal fast arrow) → add bias + GELU in SRAM (amber op) → write output (teal fast arrow). Ivory labels. No logos, no photorealism.

---

### Image 41 · `autotune-block-size-sweep.png`
**Chapter:** Ch8 Triton Kernels  **Section:** Part 5

Flat vector bar chart, wide 16:9, dark graphite background #1e1e2e. X-axis: block size (16, 32, 64, 128, 256). Y-axis: throughput in TFLOPS. Bars in muted teal rising to a peak at 128, then falling at 256. The 128 bar highlighted in amber (#f5a623) labeled "autotune winner". A horizontal dashed coral line shows "torch.matmul reference throughput". Title: "Triton tiled matmul: throughput vs. block size". Ivory labels. No logos, no photorealism.

---

### Image 42 · `autograd-computation-graph.png`
**Chapter:** genai/00-pytorch-primer  **Section:** Part 4

Flat vector technical diagram, wide 16:9, dark graphite background #1e1e2e. A left-to-right computation graph: input nodes x1 x2 flow into a multiply node (amber), then an add node with bias (amber), then an MSE loss node (coral). Each edge is a teal arrow. Reverse amber arrows labeled "backward pass" show the chain rule flowing right to left. Each node has a small label showing its operation and its stored gradient value. A small "tape" memory diagram shows the operator list. Ivory labels. No logos, no photorealism.

---

### Image 43 · `training-curves-keras-vs-pytorch.png`
**Chapter:** genai/00-pytorch-primer  **Section:** Part 5

Flat vector data visualization, wide 16:9, dark graphite background #1e1e2e. Two side-by-side loss-vs-epoch line charts. Left titled "Keras model.fit()": a smooth teal (#4ecdc4) loss line descending over 3 epochs with a coral validation loss. Right titled "PyTorch manual loop": identical convergence pattern in the same colors, same axes, same scale. Center annotation: "Same math, different API — identical final loss". Ivory axis labels. No logos, no photorealism.

---

### Image 44 · `decoder-block-internals.png`
**Chapter:** genai/03-encoder-decoder  **Section:** Part 4

Flat vector architecture diagram, wide 16:9, dark graphite background #1e1e2e. A single transformer decoder block shown as a vertical stack of three sublayers. Bottom: "Causal Self-Attention" (masked, teal) with a triangular mask icon. Middle: "Cross-Attention" — two input arrows: teal from encoder output (labeled "K, V from encoder") and amber from causal SA (labeled "Q from decoder"). Top: "Feed-Forward Network" with 4× expansion. Each sublayer has an amber residual bypass arrow on the right. Layer norms shown as small ivory circles. Ivory labels. No logos, no photorealism.

---

## Generation Notes

- Generate all 44 images (items 1–44; items 13–15 and 44 are also referenced in additional locations — generate once)
- **Naming is critical** — filename must match exactly for the placement script to work
- Save all generated images to one flat directory (suggested: `learning/_generated-images/`)
- Recommended resolution: 1600×900 px minimum
- PNG format with dark background (no white backgrounds)
- The placement script in the next phase will auto-sort each file to its correct `images/` subdirectory and add notebook references
