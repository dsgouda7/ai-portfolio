# GenAI Images Plan — All Chapters

This file tracks the image health of every chapter under `learning/genai/`. Chapter
`04-llm` has its own `image-plan.md` with full Perchance prompts; this file focuses
on the remaining four chapters and cross-chapter coordination.

---

## Audit Summary

| Chapter | Images in `images/` | Referenced in notebooks | Orphans | Critical gaps |
|---|---|---|---|---|
| `00-pytorch-primer` | 4 | 4 ✅ | 0 | autograd graph, training curves |
| `01-rnns` | 6 | 5 ✅, 1 ❌ | `Designer.png` | RNN/LSTM diagrams, external deps |
| `02-transformers` | 7 | 0 ❌ | **all 7** | must be wired into notebooks |
| `03-encoder-decoder` | 6 | 6 ✅ | 0 | decoder-block internals |
| `04-llm` | 11 | all referenced | 0 | see `04-llm/image-plan.md` |

---

## Asset Rules (shared with `04-llm/image-plan.md`)

- Store final assets in the chapter's `images/` folder with descriptive lowercase filenames.
- Wide 16:9 composition, at least 1600×900 PNG.
- Dark graphite background, muted teal for data flow, amber for trainable/active components,
  coral for failures/warnings, ivory labels.
- Short labels only — notebook Markdown provides full explanations.
- No logos, photorealistic people, UI mockups, gradients, or unreadably small text.
- Generate with Perchance only. Do not create substitute scripts if Perchance is unavailable.

---

## Chapter 00-pytorch-primer

### Action items (immediate)

#### Wire existing images
All 4 images are already wired. No action needed.

#### Missing images

| Asset | Placement | Teaching job |
|---|---|---|
| `autograd-computation-graph.png` | Before the `loss.backward()` explanation cell | Show how `model(x)` traces operators onto a dynamic graph and how `.backward()` walks it in reverse; the graph for `HousePriceModel`'s 3-node graph specifically |
| `training-curves-keras-vs-pytorch.png` | After the side-by-side accuracy bar chart | Show a side-by-side loss-vs-epoch curve comparison (Keras `history.history['loss']` line vs. PyTorch manual loss list), demonstrating "same convergence, different API" more richly than the final accuracy bar |

### Perchance prompts

#### `autograd-computation-graph.png`

```text
Flat vector technical diagram, wide 16:9, dark graphite background. A left-to-right computation graph for a simple neural network: input nodes x1 x2 flow into a multiply node with weight w, then an add node with bias b, then an MSE loss node. Each edge is a muted teal arrow. Reverse arrows in amber labeled "backward pass" show the chain rule flowing right to left. Each node has a small label showing its operation and its stored gradient value. Bottom section: a small memory diagram showing the "tape" with operator list. Ivory labels, coral warning box noting "calling zero_grad clears these". No logos, no photorealism, no gradients, no tiny text.
```

#### `training-curves-keras-vs-pytorch.png`

```text
Flat vector data visualization diagram, wide 16:9, dark graphite background. Two side-by-side loss-vs-epoch line charts. Left chart labeled "Keras model.fit()" shows a smooth teal loss curve descending over 3 epochs with a coral validation loss curve. Right chart labeled "PyTorch manual loop" shows the identical convergence pattern in the same colors, same axes, same scale. A center annotation states "same math, different API: both reach the same final loss". Axis labels and small epoch markers in ivory. No logos, no photorealism, no gradients, no tiny text.
```

---

## Chapter 01-rnns

### Action items (immediate)

#### Orphan cleanup
`Designer.png` is never referenced in either notebook. Remove it or wire it in. If its content is
not relevant to the chapter, delete it to keep the folder clean.

#### External image dependencies (fragile)
Both PT and TF notebooks load 3 images from `raw.githubusercontent.com/MITDeepLearning/...`:
- `img/add-graph.png` — simple `a + b` node diagram
- `img/computation-graph.png` — multi-step `a,b→c,d→e` graph
- `img/computation-graph-2.png` — perceptron layer diagram

These will silently break if the upstream GitHub path changes or in offline environments.
**Action:** download all three locally into `01-rnns/images/external/`, commit them, update
notebook paths to use relative local references.

#### Missing images

| Asset | Placement | Teaching job |
|---|---|---|
| `rnn-hidden-state-unrolled.png` | Future RNN notebook (when written) | Show the recurrent update equation $h_t = \tanh(W_h h_{t-1} + W_x x_t)$ as a three-step unrolled graph; highlight that depth in time = depth in an equivalent feedforward net |
| `vanishing-gradient-vs-timestep.png` | Future RNN notebook — after BPTT section | Plot gradient norm vs. distance (in timesteps) from the loss; show exponential decay without LSTM and roughly flat line with LSTM |
| `lstm-gate-equations.png` | Future RNN notebook — LSTM section | Show all four gate equations side by side (forget, input, output, cell update) with color-coded data paths: cell state (teal), hidden state (amber), gate activations (coral) |
| `animated-gradient-descent.png` | PT-Part1-Intro.ipynb — parabola section | Static fallback for the gradient descent convergence section showing the x-value tracing down the parabola over 10 key steps — complements the existing loss-vs-epoch convergence plot |

### Perchance prompts

#### `rnn-hidden-state-unrolled.png`

```text
Flat vector technical diagram, wide 16:9, dark graphite background. An RNN unrolled across three timesteps t-1, t, t+1. Each step shows an amber box labeled "RNN cell" with two inputs: input vector x_t from below (muted teal) and hidden state h_{t-1} from the left (amber arrow). Each cell outputs h_t flowing right. The tanh activation gate is shown as a small circle inside each cell. Below the diagram: the recurrent equation h_t = tanh(W_h h_{t-1} + W_x x_t + b) in large readable text. Right side: a deep vertical stack showing "unrolled = depth = vanishing gradients". Ivory labels, coral warning on the depth stack. No logos, no photorealism, no gradients, no tiny text.
```

#### `vanishing-gradient-vs-timestep.png`

```text
Flat vector data visualization, wide 16:9, dark graphite background. Two side-by-side line charts sharing the same x-axis labeled "distance from loss (timesteps)". Left chart "RNN without gating": a steeply decaying coral curve starting near 1.0 at timestep 1 and reaching near 0.0 by timestep 20. Right chart "LSTM": a roughly flat muted teal line staying near 0.8 across all 20 timesteps. Both charts have a y-axis labeled "gradient norm" with range 0 to 1.0. Title in ivory: "Gating preserves gradient signal over time". No logos, no photorealism, no gradients, no tiny text.
```

#### `lstm-gate-equations.png`

```text
Flat vector technical infographic, wide 16:9, dark graphite background. Four aligned LSTM gate equation panels labeled forget gate, input gate, cell state update, output gate. Each panel shows the sigmoid or tanh activation as a small icon, the weight matrices W_f W_i W_c W_o as amber rectangles, and the resulting gate vector as a colored bar. Arrows show: forget gate applied to cell state in teal (with coral "erasure" portion), input gate applied to new candidate in amber, cell state flowing through as a horizontal highway in teal, output gate controlling h_t. All equations in readable symbolic math. Ivory background text, coral for gates that can zero out. No logos, no photorealism, no gradients, no tiny text.
```

---

## Chapter 02-transformers

### Action items (critical — all 7 images are unlinked)

Every image in `02-transformers/images/` was authored as a static reference figure but was never
embedded in either notebook via Markdown `![](images/...)` or `IPython.display.Image`. They must
be wired in.

| Image file | Suggested notebook placement | What to add |
|---|---|---|
| `transformer-learning-journey.png` | Cell 1 (intro, both notebooks) — after the title heading | `![Transformer learning journey: from embedding to GPT-2](images/transformer-learning-journey.png)` |
| `transformer-block-overview.png` | Part 7 intro — before the "Full Transformer Block" assembly cells | An orientation anchor showing the complete block (norm → MHA → residual → norm → FFN → residual) before the reader builds it piece by piece |
| `attention-qkv-data-flow.png` | Part 4 intro (Q/K/V projections) — before the projection code | Shows the soft dictionary lookup: query → dot with all keys → weights → weighted sum of values |
| `positional-encoding-and-rope.png` | Part 3b (RoPE motivation) — between sinusoidal PE and RoPE intro | Compares sinusoidal PE's absolute encoding with RoPE's relative rotation |
| `toy-to-production-transformers.png` | Toy-to-real bridge section (before DistilGPT-2 load) | Complements the comparison table already in the notebook — adds a visual dimension-scaling diagram |
| `transformer-architecture-families.png` | Part 13 intro — before encoder-only/decoder-only/enc-dec section | Shows all three architecture variants side by side before the reader builds each |
| `autoregressive-generation-and-kv-cache.png` | Part 9 (autoregressive inference) or Part 13d (KV cache) | Illustrates the token generation loop and the KV cache savings |

**Implementation task:** add `![caption](images/filename.png)` Markdown cells at each listed
insertion point in both `transformers.ipynb` and `transformers-keras.ipynb`. This is a subagent
task — it requires reading cell positions and inserting without disrupting surrounding pedagogy.

### Missing images (not yet generated)

No new images needed — the 7 existing ones cover all key gaps once wired in.

---

## Chapter 03-encoder-decoder

### Action items

All 6 images are wired correctly. One gap remains.

#### Missing image

| Asset | Placement | Teaching job |
|---|---|---|
| `decoder-block-internals.png` | Part 4 (after CrossAttention class, before training) | Show the three-sublayer decoder stack explicitly: causal SA → cross-attention → FFN, with residuals, each sublayer labeled. The `encoder-decoder-contract.png` shows the high-level flow but not the internal three-sublayer structure |

### Perchance prompt

#### `decoder-block-internals.png`

```text
Flat vector technical architecture diagram, wide 16:9, dark graphite background. A single transformer decoder block shown as a vertical stack of three sublayers. Bottom sublayer: "Causal Self-Attention" with a visible triangular mask icon and label "masked — decoder only sees past tokens". Middle sublayer: "Cross-Attention" with two input arrows: one from the encoder output (muted teal, labeled "K, V from encoder") and one from the causal SA output (amber, labeled "Q from decoder"). Top sublayer: "Feed-Forward Network" with a 4x expansion and projection back. Each sublayer has an amber residual bypass arrow on the right side. Layer norms are shown as small ivory circles before each sublayer. Data flows bottom-to-top in muted teal. Ivory labels, coral warning on the causal mask. No logos, no photorealism, no gradients, no tiny text.
```

---

## Cross-Chapter Notes

### Consistent dark-theme palette
All chapters currently use dark graphite backgrounds with the same color semantics (teal/amber/coral/ivory).
The 7 unlinked `02-transformers` images appear to follow this palette; verify that they match before
wiring them in, as palette inconsistency across chapters is visually jarring.

### Image sizing at notebook content width
All images should be embedded without an explicit `width=` attribute first — allow the notebook
viewer to size them to content width. Only add `width=600` or similar if the image is demonstrably
too large or too small at the default rendering.

### Review rubric (from `04-llm/image-plan.md`, applies to all chapters)

Accept an image only when it passes all of these checks:
1. The mechanism can be explained without relying on text generated inside the image.
2. Shapes, arrows, and short labels remain legible at notebook content width.
3. The visual makes no technical claim that conflicts with its adjacent notebook cell.
4. It uses the shared palette and contains no provider logo or copyrighted character.
5. It adds information that the notebook's existing plots do not already show.
