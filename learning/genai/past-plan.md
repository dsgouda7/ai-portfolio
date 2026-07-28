# GenAI Prerequisites Plan

**Target learner:** Anyone from *zero AI/ML knowledge* to *CNN-familiar but not production-experienced*
who wants to reach `learning/genai/` ready. The track is structured in two phases so a learner
with CNN knowledge can enter at Phase 2 and skip Phase 1.

**Directory to implement:** `learning/genai-prerequisites/`
(name rationale: these notebooks are literally the *prerequisites for the genai learning track* —
"language-foundations" was too narrow; "fundamentals" implies general AI; "genai-prerequisites"
names the relationship unambiguously.)

**Relationship to `notes/`:** Mechanistic content already exists across three notes tracks
(`notes/00-math-under-the-hood/`, `notes/01-ml/03-neural-networks/`, `notes/03-llm/ch00`).
The work is to **promote that content to gold-standard notebooks**: add the pedagogical wrapper
(challenge-before-title, roadmap table, threaded running example, predict-first with candidate
outcomes, closing decision) rather than rediscovering the math.
See per-chapter "Source material from `notes/`" entries below.

**Two phases:**
- **Phase 1 — Zero to CNN-Ready:** covers math foundations → ML basics → neural networks → CNNs
  for learners who have never trained a model. Source: `notes/00-math-under-the-hood/` and
  `notes/01-ml/03-neural-networks/ch01–ch05`.
- **Phase 2 — CNN to LLM-Ready:** covers RNNs + tokenization (the gaps identified in the original
  story-arc audit). Source: `notes/01-ml/03-neural-networks/ch06+ch09` and
  `notes/03-llm/ch00-from-networks-to-language/`.

---

## Story-Arc Audit

### What the current track assumes at first contact

When a CNN learner opens `00-pytorch-primer/keras-to-pytorch-primer.ipynb`, they can follow
it cleanly. When they open `01-rnns/PT-Part1-Intro.ipynb`, they get DL fundamentals / autograd —
still fine. **The first hard gap appears at `02-transformers/transformers.ipynb`**: the opening
cells refer to RNNs as the architecture Transformers replaced, describe BPTT, and use the
RNN-vs-Transformer comparison table (Keras) as motivation — but RNNs were never taught. A CNN
learner has no mental model of:

- What $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$ means
- Why vanishing gradients are worse in time than in depth
- What BPTT is and why it is expensive
- Why LSTM gating was the solution before attention

The Transformer chapter's "why is this better than what came before?" question is unanswerable
without this background.

### Second gap: tokenization

The `04-llm` chapter works entirely with GPT-2's BPE tokenizer, including `Ġ` (leading-space)
tokens, vocabulary indices, and `tokenize_causal()` vs `tokenize_instruction()` masking patterns.
No prior chapter explains:

- Why we need tokenization at all (strings → discrete tokens → integers → embeddings)
- How BPE works (merge-pair algorithm, why subword beats word and character)
- What `nn.Embedding` does and why it is the first trainable layer in a language model
- What padding and the `-100` ignore index pattern mean at the sequence level

A CNN learner who has worked with image classification has never encountered variable-length
inputs, and the `04-llm` chapter offers no bridge.

### Third gap: the missing "why decoder-only won" forward link

`03-encoder-decoder/encoder-decoder.ipynb` ends with a Tier 3 item ("why decoder-only won —
out of scope") but provides no bridge forward. A learner finishes knowing T5/BART work but not
understanding why `04-llm` opens with decoder-only GPT-2. This is a short cell in `03`, not a
full new chapter.

---

## Phase 1 — Zero to CNN-Ready

These chapters are for learners who have never trained a neural network. A learner who
already has solid CNN knowledge (backprop, convolutions, ResNets) skips Phase 1 entirely
and starts at Phase 2.

---

### P-0 · Mathematical Foundations for ML

**Why it exists:** Every ML notebook silently uses vectors, derivatives, and matrix
multiplication. A learner who hasn't seen calculus will memorize steps without
understanding what is being optimised.

**Source material from `notes/`:** `notes/00-math-under-the-hood/` has 7 chapters
(linear algebra, nonlinear algebra, calculus intro, small steps / gradient descent,
matrices, gradient + chain rule, probability/statistics) all using the **"Knuckleball
Free Kick"** running example (projectile motion as constrained optimisation). Both the
content and the scenario are production-ready — the main addition is the gold-standard
pedagogical wrapper.

**What this notebook covers:**

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | Vectors and dot products | Manual dot product of player kicking direction vs. goal direction; measure alignment |
| 2 | Derivatives and rates of change | `sympy.diff()` to find peak height of free kick parabola; verify numerically |
| 3 | Gradient descent on one variable | Minimise `(angle - optimal)^2`; 50-iteration convergence plot |
| 4 | Matrices and linear transforms | Weight matrix as a transformation; visualise basis-vector rotation |
| 5 | Chain rule and backpropagation intuition | Multi-step function differentiated by hand; match `autograd` result |
| 6 | Probability and the Gaussian | Fit a Gaussian to noise-corrupted goal positions; compute P(scoring) |

**Pedagogy requirements:**
- **Named scenario:** the free kick throughout (direct from notes)
- ` Predict first` before gradient descent: *"After 50 steps from angle=80°, will we land: (a) near the optimal, (b) stuck at a local minimum, (c) past the optimal?"*
- Closing decision: *"The kick is scoreable when launch angle is between {lo:.1f}° and {hi:.1f}° — measured by the gradient descent run above."*
- Closing tier-1/2/3 ledger: Tier 1 (vectors, derivatives, gradient descent, chain rule, Gaussian), Tier 3 (Hessians, Taylor series, convexity — named only)

**images-plan.md for this chapter:**

| Asset | Placement | Teaching job |
|---|---|---|
| `gradient-descent-convergence.png` | Part 3 intro | Show a 1D loss bowl with a ball stepping down the slope over 10 iterations; label each step with the loss value; the bottom is the optimal angle |
| `chain-rule-computation-graph.png` | Part 5 intro | Show a 3-node computation graph (f → g → loss); edges labelled with local derivatives; a red reverse-pass arrow showing how they multiply |
| `free-kick-parabola-constraints.png` | Opening challenge cell | Show the parabolic trajectory, the defensive wall at 9.15m, the crossbar at 20m; the scoring window is a shaded green band |

**Perchance prompts (images-plan.md):**
```text
[gradient-descent-convergence.png]
Flat vector data visualization, wide 16:9, dark graphite background. A smooth U-shaped
loss curve with x-axis "launch angle (degrees)" and y-axis "penalty". Ten amber dots
step down the left slope toward the minimum, connected by thin lines showing the path.
The final dot lands near the bottom. A small arrow at each dot points in the direction
of the negative gradient. Ivory axis labels, coral dot for starting point, teal dot
for ending point. No logos, no photorealism, no gradients, no tiny text.

[chain-rule-computation-graph.png]
Flat vector technical diagram, wide 16:9, dark graphite background. Three nodes in a
left-to-right chain: x → amber box "f" → amber box "g" → coral box "loss". Forward
arrows in teal labelled "df/dx" and "dg/df". A thick red reverse arrow below, flowing
right-to-left, labelled "d(loss)/dx = df/dx × dg/df" with the chain rule product
shown explicitly. Ivory labels. No logos, no photorealism, no gradients, no tiny text.

[free-kick-parabola-constraints.png]
Flat vector physics diagram, wide 16:9, dark graphite background. A 2D side-view of a
football free kick: kick origin at left, defensive wall at 9.15m marked with a teal
vertical rectangle, goal at 20m with crossbar marked in amber. A muted-coral parabolic
trajectory arc passing over the wall and under the crossbar. A shaded green window at
the goal showing the scoring region. Key dimensions annotated in ivory. No logos, no
photorealism, no gradients, no tiny text.
```

**Subagent implementation task:**
> Create `learning/genai-prerequisites/00-math-foundations/` with `math-foundations-for-ml.ipynb`.
> **Primary source:** Extract mechanistic content from all 7 chapters of `notes/00-math-under-the-hood/`.
> Keep the Knuckleball free kick running example (it threads through all 7 notes chapters naturally).
> Compress to one notebook covering the 6 Parts above. Apply ALL authoring guide conventions.
> Create an `images-plan.md` from the prompts above.

---

### P-1 · ML Basics: Regression and Classification

**Why it exists:** Before neural networks, a learner needs to see gradient descent train
something real, understand loss functions, and know what "overfitting" looks and feels like.

**Source material from `notes/`:**
- `notes/01-ml/01-regression/` (7 chapters: linear through polynomial regression, regularisation,
  hyperparameter tuning) — uses California Housing dataset
- `notes/01-ml/02-classification/` (5 chapters: logistic regression through SVMs) — uses CelebA

**What this notebook covers:**

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | Linear regression on California Housing | Fit, predict, compute MAE; print: "District 42: predicted ${pred:.0f}k, actual ${actual:.0f}k" |
| 2 | Gradient descent from scratch | Build and train linear regression manually; match `sklearn` output |
| 3 | Loss functions and why MSE penalises outliers | Compare MSE vs. MAE on a dataset with 3 outlier districts; plot residuals |
| 4 | Regularisation: Ridge vs. Lasso | Show coefficient shrinkage; prove Lasso zeros out unimportant features |
| 5 | Binary classification: sigmoid + BCE loss | Classify "high-value district" (>$300k); confusion matrix + decision boundary |
| 6 | Overfitting proof | Train on 50 samples; plot train vs. val loss diverging; fix with L2 |

**images-plan.md for this chapter:**

| Asset | Teaching job |
|---|---|
| `regression-loss-landscape.png` | 2D bowl showing MSE loss surface over w and b; a gradient descent path spiralling to the bottom |
| `overfitting-train-val-curves.png` | Side-by-side: train loss falling, val loss U-shaped; arrow pointing to "sweet spot" |
| `lasso-ridge-coefficients.png` | Bar chart of feature coefficients: Lasso zeros 3/8 features; Ridge shrinks all but none to zero |

**Perchance prompts:** *(in images-plan.md)*
```text
[regression-loss-landscape.png]
Flat vector data visualization, wide 16:9, dark graphite background. A 3D-style
contour map of an MSE loss bowl over two axes labeled "weight" and "bias". A spiral
path of amber dots descends from a high-loss plateau toward the minimum. Contour
lines in muted teal, ivory axis labels. The minimum is marked with a coral star.
No logos, no photorealism, no gradients, no tiny text.

[overfitting-train-val-curves.png]
Flat vector data visualization, wide 16:9, dark graphite background. Two side-by-side
charts. Left: training loss falling smoothly (teal line). Right: training loss falling
(teal) plus validation loss forming a U-shape (coral), crossing at epoch 15. A vertical
amber dashed line at the crossing marks "early stop here". Ivory axis labels. No logos,
no photorealism, no gradients, no tiny text.

[lasso-ridge-coefficients.png]
Flat vector bar chart, wide 16:9, dark graphite background. Two rows of 8 bars each,
one row labeled "Ridge" (all bars non-zero, shrunk, amber), one row labeled "Lasso"
(3 bars are exactly zero and shown as coral, 5 bars are teal). Feature names as ivory
labels below each bar. Subtitle: "Lasso selects. Ridge shrinks." No logos, no
photorealism, no gradients, no tiny text.
```

**Subagent implementation task:**
> Create `learning/genai-prerequisites/01-ml-basics/` with `ml-basics.ipynb`.
> **Primary source:** `notes/01-ml/01-regression/` (California Housing scenario and dataset)
> for Parts 1–4. `notes/01-ml/02-classification/` for Part 5.
> Keep the California Housing dataset (directly importable from `sklearn`; no download).
> Add the gold-standard pedagogical wrapper: challenge cell before title, predict-first questions
> with candidate outcomes, closing decision that branches on actual measured MAE.
> Apply ALL authoring guide conventions. Create `images-plan.md`.

---

### P-2 · Neural Networks and Backpropagation

**Why it exists:** The `learning/genai/00-pytorch-primer` assumes you already know what a
neuron does. This chapter builds that intuition from the XOR problem up through a CNN-ready
architecture, making the primer an acceleration not an introduction.

**Source material from `notes/`:**
`notes/01-ml/03-neural-networks/ch01_xor_problem/` through `ch04_regularisation/`
— all four use the UnifiedAI / California Housing + face classification scenario.

**What this notebook covers:**

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | The XOR problem: why linear models fail | Prove with `np.linalg.solve`; show the 4-point XOR truth table is not linearly separable |
| 2 | One hidden layer + ReLU: XOR solved | Build a 2-2-1 network from scratch; verify it classifies all 4 points correctly |
| 3 | Backpropagation by hand | Compute gradients for the 2-layer XOR network; match `autograd` to 6 decimal places |
| 4 | Universal approximation: why depth beats width | Train width-256 vs. depth-4 on a spiral dataset; compare decision boundaries |
| 5 | Regularisation in neural networks | Demonstrate Dropout (train/eval mode comparison) and BatchNorm; prove each mechanically |
| 6 | Toy → real bridge | Map toy XOR network params to a GPT-2 layer's parameter count; show they are the same operations at scale |

**images-plan.md for this chapter:**

| Asset | Teaching job |
|---|---|
| `xor-not-linearly-separable.png` | 2D grid showing XOR points; failed linear decision boundary |
| `neural-network-forward-pass.png` | 2-3-1 network with annotated forward pass values flowing through each node |
| `depth-vs-width-decision-boundary.png` | Side-by-side spiral dataset: wide shallow net vs. deep narrow net decision boundaries |

**Subagent implementation task:**
> Create `learning/genai-prerequisites/02-neural-networks/` with `neural-networks-and-backprop.ipynb`.
> **Primary source:** `notes/01-ml/03-neural-networks/ch01_xor_problem/notebook.ipynb`
> through `ch04_regularisation/notebook.ipynb`.
> Keep the XOR running example from ch01; use it as the thread through all 6 Parts.
> The backpropagation cell must compute gradients manually and verify against `autograd`.
> Apply ALL authoring guide conventions. Create `images-plan.md`.

---

### P-3 · Convolutional Neural Networks

**Why it exists:** `learning/genai/00-pytorch-primer` builds a CNN from scratch but explains
it as a component; this chapter builds the full spatial-reasoning mental model.

**Source material from `notes/`:**
`notes/01-ml/03-neural-networks/ch05_cnns/` — has both `notebook.ipynb` (TF/Keras) and
`notebook-pytorch.ipynb`. The README explains the spatial feature extraction motivation.

**What this notebook covers:**

| Part | Topic | Key demonstration |
|---|---|---|
| 1 | Convolution as learned spatial filter | Apply a hand-coded edge-detection filter to MNIST; compare to learned conv weights |
| 2 | Stride and pooling: compressing spatial resolution | Visualise feature maps at each layer; show size reduction arithmetic |
| 3 | Receptive field growth with depth | Show that layer-5's neuron "sees" 17×17 pixels of the input; prove with gradient backprop |
| 4 | ResNet skip connections: gradient highway | Measure gradient norm at layer 1 with/without residual connections across 10 layers |
| 5 | Transfer learning: pretrained weights, frozen base | Load ResNet-18; freeze all but final layer; fine-tune on a 2-class MNIST subset; compare to training from scratch |
| 6 | Toy → real bridge | Map our 3-layer CNN dimensions to ResNet-50's architecture; print parameter counts |

**images-plan.md for this chapter:**

| Asset | Teaching job |
|---|---|
| `convolution-filter-operation.png` | 5×5 input, 3×3 filter sliding over it, output feature map with the arithmetic annotated |
| `feature-maps-by-layer.png` | Three panels: raw MNIST image → conv1 feature maps (edges) → conv2 feature maps (curves/corners) |
| `resnet-skip-connection.png` | Two-block comparison: plain block (x → F(x)) vs. residual block (x → F(x)+x); gradient magnitude arrows |

**Subagent implementation task:**
> Create `learning/genai-prerequisites/03-cnns/` with `convolutional-neural-networks.ipynb`.
> **Primary source:** `notes/01-ml/03-neural-networks/ch05_cnns/notebook-pytorch.ipynb`.
> Use MNIST (importable from `torchvision.datasets`; no external download).
> Part 4 (ResNets) must measure actual gradient norms, not assert them.
> Apply ALL authoring guide conventions. Create `images-plan.md`.

---

## Phase 2 — CNN to LLM-Ready

These chapters bridge from CNN-familiar to genai-ready. A learner who already knows CNNs
enters here; a learner who completed Phase 1 continues here.

---


**Why it is a blocker:** `02-transformers/transformers.ipynb` builds its entire motivation
on RNNs as the prior art. Without understanding the hidden-state update equation, BPTT, and
the vanishing-gradient problem, a learner cannot understand *why* any Transformer design
decision is a decision (parallel computation over time, attention over fixed window, positional
encoding instead of implicit recurrence). The folder is named `01-rnns` but contains only DL
foundations — the RNN content is absent entirely.

**Narrative framing (required — Section 8 of the authoring guide):**

The notebook is anchored to one concrete scenario throughout: *a music research team wants to predict the next note in a melody — the same problem that launched sequence modeling in the 1980s.* The running corpus is the opening 40 characters of "Twinkle Twinkle Little Star" encoded as characters (`T`, `w`, `i`, `n`, `k`, `l`, `e`, ` `, ...). This corpus is:
- Small enough to inspect character-by-character (no download required)
- Rich enough to show repeat patterns (the word "twinkle" appears twice — a real test of memory)
- Musical, connecting back to the pretrained music generation demo the learner already saw in `01-rnns/PT-Part1-Intro.ipynb`

Every Part answers a named question for this team: "Does it even know the alphabet?" (Part 1), "Does it remember what it saw 5 steps ago?" (Part 2–3), "Why does it forget the second 'twinkle'?" (Part 3), "What if we give it a memory lane?" (Part 4–5).

**What this notebook covers (one notebook, PyTorch-primary, brief Keras mirror):**

| Part | Content | Key proof/demonstration |
|---|---|---|
| 0 | Challenge: the music team's problem + scope of what changes from a CNN | Print: a CNN classifies one image as one label; an RNN must output one label *per timestep* — the output tensor has a time axis |
| 1 | Character-level LM: vocabulary, `nn.Embedding`, `(batch, time, features)` shape contract | Print token-index-to-character mapping; visualize embedding matrix as a heatmap; **prove**: `model(one_hot_input)` and `model(index_input)` produce identical logits |
| 2 | Vanilla RNN cell from scratch: $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$ | Unroll 5 steps of the melody by hand, printing $h_0, h_1, ..., h_4$; **verify** the manual computation matches `nn.RNN` output to 6 decimal places |
| 3 | BPTT and vanishing gradients: gradient norm vs. timestep distance from loss | **Ablation**: log-scale gradient-norm plot for sequence length 5, 20, 50 on the melody; print: "at depth 50, the gradient at step 1 is {n:.2e} — effectively zero" |
| 4 | LSTM gating: forget, input, output, cell update equations | Build from scratch as `nn.Module`; **prove** gate activations are in [0,1] with `assert ((gates >= 0) & (gates <= 1)).all()`; train on same melody |
| 5 | Comparison: plain RNN vs. LSTM on the second 'twinkle' | Side-by-side loss curves for both; measure whether LSTM correctly predicts the `t` after the space more often than vanilla RNN; **print**: "LSTM got it right N/10 times vs. RNN's M/10" |
| 6 | Toy → real bridge | Parameter table mapping toy dims (`hidden_size=8`) to real `nn.LSTM(hidden_size=256, num_layers=2)`; show GPT-2's actual embedding layer shape |

**Pedagogy requirements (gold-standard parity):**
- **Named scenario and threaded running example throughout** (the music team; the "Twinkle" melody; same characters Part 1–6)
- ` Predict first` before the vanilla RNN forward pass: *"At timestep 50, the gradient signal will be: (a) roughly the same as at timestep 1, (b) about 10× smaller, (c) about 1000× smaller, or (d) actually larger due to accumulation?"* Answer: (c) for vanilla RNN, (a) for LSTM
- ` Predict first` before the RNN-vs-LSTM comparison: *"How many times (out of 10 sampled completions) will each model correctly predict 't' at the start of the second 'twinkle'? Options: RNN<5/LSTM>7, both~5, RNN>LSTM"*
- ` Your turn` exercise: change sequence length from 40 to 10; does vanishing gradient still occur? (Answer: much less — print confirms it)
- `FuncAnimation` for the vanishing gradient experiment (gradient norm vs. timestep, animated adding one layer at a time)
- `#### What just happened — and what's missing` after Part 3: "The RNN can't carry the 'twinkle' memory 15 steps. Next: give it a memory lane with explicit gates."
- Toy/real parity table before `nn.LSTM`
- **Closing decision (Section 8.7):** "For the music team: the LSTM now correctly anticipates the repeat pattern. The cost: 4× more parameters than a vanilla RNN (measured). The benefit: the gradient norm at step 1 went from {rnn_grad:.2e} to {lstm_grad:.2e}. For sequences shorter than ~20 tokens, use RNN; above that, LSTM earns its parameter cost."
- Closing tier-1/2/3 ledger: Tier 1 (vanilla RNN, LSTM), Tier 2 (GRU — same cell highway, fewer gates, explained but not trained), Tier 3 (bidirectional RNN, stacked RNN, attention-augmented RNNs — named with one-line reason)
- Explicit forward pointer: "The music team can predict one note at a time. In the next chapter, we ask: what if the model could attend to *all* past notes simultaneously? That's the Transformer."

**images-plan.md for this chapter:**

| Asset | Teaching job |
|---|---|
| `rnn-hidden-state-unrolled.png` | RNN unrolled over 3 steps; hidden state arrow flowing right; vanishing gradient warning |
| `vanishing-gradient-vs-timestep.png` | Side-by-side line charts: RNN gradient decays exponentially; LSTM stays flat |
| `lstm-gate-equations.png` | 4-panel: forget/input/output/cell gates with color-coded data paths |

**Perchance prompts:** *(in images-plan.md — see `images-plan.md` in `learning/genai/` for shared palette)*
```text
[rnn-hidden-state-unrolled.png]
*(already specified in learning/genai/images-plan.md — copy that prompt here)*

[vanishing-gradient-vs-timestep.png]
*(already specified in learning/genai/images-plan.md — copy that prompt here)*

[lstm-gate-equations.png]
*(already specified in learning/genai/images-plan.md — copy that prompt here)*
```

**Subagent implementation task:**
> Create `learning/genai-prerequisites/04-rnn-sequence-modeling/` with a `rnn-sequence-modeling.ipynb`
> notebook following ALL gold-standard conventions from `learning/genai/authoring-guide.md`.
> **Primary source:** Extract mechanistic content (equations, code structure, BPTT proof, LSTM gate
> implementation) from `notes/01-ml/03-neural-networks/ch06_rnns_lstms/notebook-pytorch.ipynb`
> (PyTorch) and the conceptual framing from
> `notes/01-ml/03-neural-networks/ch06_rnns_lstms/README.md`.
> **Secondary source:** `notes/03-llm/ch00-from-networks-to-language/prerequisites-demo.ipynb`
> for the misconceptions-first opening and the "vanishing gradients were not solved by LSTMs"
> correction frame.
> **What to keep from notes:** The BPTT derivation, the LSTM gate `assert` checks, the vanilla
> RNN vs. LSTM comparison structure. **What to replace:** The housing price running example and
> the UnifiedAI scenario — swap entirely for the music team / "Twinkle Twinkle" scenario specified
> in this plan. All predict-first questions, candidate outcomes, and the closing decision are new.
> The running corpus is the first 40 characters of "Twinkle Twinkle Little Star" (write inline).
> The closing decision must branch on the actual measured gradient norms, not aspirational text.
> Create `images-plan.md` using the prompts above (and those from `learning/genai/images-plan.md`).

---

### P-5 · Text Tokenization and Embeddings *(was M-2)*

**Why it is a blocker:** Every notebook from `02-transformers` onward works with token
sequences, but no chapter explains the text→token→integer→embedding pipeline. Specifically:
- `02-transformers` builds a hand-coded vocabulary (`VOCAB = {"the":0, "cat":1, ...}`) but
  never explains how a real tokenizer produces this
- `04-llm` uses GPT-2's BPE tokenizer extensively (`Ġ` prefix tokens, `input_ids`, padding,
  `-100` masking) and gives a walkthrough in the Code Walkthrough cell — but a learner
  without tokenization foundations will not follow the walkthrough

**Narrative framing (required — Section 8 of the authoring guide):**

The notebook is anchored to one concrete task: *you're building a multilingual chatbot for a small law firm. Your two problems: (1) legalese has rare compound words that a word-level tokenizer will treat as unknown tokens; (2) the system must also handle French contract clauses. You need a tokenizer that handles rare words without blowing up the vocabulary.* Every technique is framed as answering one of the firm's questions: "Why can't we just split on spaces?" (Part 1), "How does GPT-2 handle 'non-disclosure' without seeing it in training?" (Part 2–3), "Why do 'non-disclosure' and 'nondisclosure' get the same embedding representation?" (Part 4).

**What this notebook covers (one notebook, framework-agnostic):**

| Part | Content | Key proof/demonstration |
|---|---|---|
| 1 | Why tokenization exists: string → integer pipeline; why characters are too granular, words have OOV problems | **Measure** vocabulary sizes on a 20-sentence legal corpus: character-level = 62 unique tokens, word-level = 340 unique tokens, 38 of which appear only once (OOV risk); print the comparison |
| 2 | BPE from scratch: merge-pair algorithm on a toy corpus (5 legal sentences) | Step-by-step merge table printed at every iteration; **prove**: after 20 merges, "non-disclosure" is represented as 2 tokens ("non", "disclosure"), not 15 characters |
| 3 | Real BPE: GPT-2 tokenizer via `tiktoken`; inspect `Ġ` prefix, multitoken words, "indemnification" example | **Measure**: token count vs. character count for 10 legal phrases; print compression ratios; show that French "dommages" is handled without OOV |
| 4 | `nn.Embedding` as a trainable lookup table: visualize embedding space for a small vocabulary using PCA 2D plot | **Prove**: embeddings start random (PCA shows uniform scatter); after 100 training steps on the corpus, legal synonyms cluster; show the before/after PCA side by side |
| 5 | Sequence padding: variable-length inputs, `pad_token_id`, the `-100` ignore index pattern | Side-by-side: padded batch without and with `attention_mask`; print what CrossEntropy sees |
| 6 | Toy → real bridge: vocabulary sizes (toy=15, GPT-2=50257, LLaMA-3=128k), subword vs. byte-level | Parameter table; note that "the cat" is 2 tokens in GPT-2, 3 in character model |

**Pedagogy requirements:**
- **Named scenario threaded throughout** (the law firm; the 20-sentence legal corpus used from Part 1 to Part 4)
- ` Predict first` before BPE merge step 10: *"After 10 merges, what will happen to the token 'non-disclosure': (a) still 15 characters, (b) two tokens 'non' and 'disclosure', (c) one merged token 'non-disclosure'?"* (Answer: b, by step 10 in a legal corpus)
- ` Predict first` before the embedding PCA: *"After 100 training steps on 20 sentences, will 'contract' and 'agreement' be: (a) near each other, (b) far from each other, (c) in random positions?"* (Answer: a — the training signal clusters legal synonyms)
- ` Your turn`: change the merge budget from 20 to 5; measure how vocabulary size and OOV token count change
- `FuncAnimation` for the BPE merge progression (one frame per merge step; watch "non-disclosure" shrink from 15 tokens to 2)
- **Closing decision (Section 8.7):** "For the law firm: BPE with 20k merges handles 'indemnification' in 2 tokens (not OOV), handles French without a separate vocabulary, and produces compression ratio {r:.1f}× vs. character-level. The recommended tokenizer: GPT-2's `tiktoken` encoder, which uses the same algorithm at 50k vocabulary size."
- Closing tier-1/2/3 ledger: Tier 1 (character-level, word-level, BPE from scratch, GPT-2 tiktoken), Tier 2 (WordPiece — same merge algorithm, different scoring; explained but not built), Tier 3 (SentencePiece, Unigram LM, byte-level BPE — named with one-line reason)
- Forward pointer: "This is the exact pipeline that feeds `02-transformers`' `VOCAB` dictionary — the hand-coded vocabulary there is a simplified BPE with exactly these properties."

**Subagent implementation task:**
> Create `learning/genai-prerequisites/05-tokenization/` with `tokenization-and-embeddings.ipynb`.
> Apply ALL gold-standard conventions from `learning/genai/authoring-guide.md`.
> **Primary source:** There is no dedicated tokenization chapter in `notes/`. However:
> — The *embeddings as a trainable lookup table* section can draw from
>   `notes/01-ml/03-neural-networks/ch09_sequences_to_attention/notebook.ipynb`.
> — The `Ġ` BPE token explanation from `notes/03-llm/ch00-from-networks-to-language/README.md`.
> **What to build fresh:** The BPE-from-scratch merge algorithm, the legal corpus, the PCA
> before/after visualization, and the `tiktoken` demo have no direct notes equivalent.
> Opening cell states the law firm scenario before the title.
> The 20-sentence legal corpus must be written inline (no download).
> The closing decision must branch on the actual measured compression ratio.
> Create `images-plan.md` using the prompts above.

## Consistency Improvements (High Value, Not Blockers)

### C-0 · "Why Decoder-Only Won" Bridge Cell in `03-encoder-decoder`

Already implemented in the previous session (subagent added the cell). See `learning/genai/03-encoder-decoder/encoder-decoder.ipynb`.

### C-1 · FuncAnimation for Gradient Descent in `01-rnns/PT-Part1-Intro.ipynb`

The parabola gradient descent convergence is shown as a static loss-vs-epoch plot. An animation
(x-value stepping down the parabola surface over 500 iterations) would be a dramatically more
effective visual — matching the animation-heavy pedagogy of `02-transformers`.

**Subagent task:** Add a `FuncAnimation` cell to the parabola section showing the x-value
stepping down the $y = x^2$ curve frame by frame. Follow all Section 9.4 conventions
(plt.close before display, print before animation, fps=6).

### C-2 · `` / `` Emoji Normalization in `03-encoder-decoder`

The encoder-decoder notebook uses plain-text "Predict before you run" / "Your turn" headers
(consistent with the Section 3 amendment to the authoring guide). No change strictly needed,
but if the style guide is ever tightened to require emojis, this is the chapter to update.

### C-3 · Designer.png Cleanup in `01-rnns`

`Designer.png` is an orphan. Either wire it into a relevant section or delete it. Keeping
orphan files creates maintenance confusion.

---

## Implementation Parallelization Map

These items can be implemented by independent subagents simultaneously.
**Phase 1 new chapters** (P-0 through P-3) can all run in parallel with each other and with
**Phase 2** chapters (P-4, P-5):

| Agent | Task | Target | Notes source |
|---|---|---|---|
| Agent A | P-0: Math foundations | Create `learning/genai-prerequisites/00-math-foundations/` | `notes/00-math-under-the-hood/` (all 7 chapters) |
| Agent B | P-1: ML basics | Create `learning/genai-prerequisites/01-ml-basics/` | `notes/01-ml/01-regression/` + `02-classification/` |
| Agent C | P-2: Neural networks | Create `learning/genai-prerequisites/02-neural-networks/` | `notes/01-ml/03-neural-networks/ch01–ch04` |
| Agent D | P-3: CNNs | Create `learning/genai-prerequisites/03-cnns/` | `notes/01-ml/03-neural-networks/ch05_cnns/` |
| Agent E | P-4: RNN/LSTM | Create `learning/genai-prerequisites/04-rnn-sequence-modeling/` | `notes/01-ml/03-neural-networks/ch06+ch09` |
| Agent F | P-5: Tokenization | Create `learning/genai-prerequisites/05-tokenization/` | Partial: `notes/03-llm/ch00/` |
| Agent G | C-4: Cache MIT images | Edit `01-rnns/` notebooks | n/a — download + update paths |

---

---

## GenAI Prerequisites Directory Structure (after implementation)

```
learning/
  genai-prerequisites/             # learning/genai-prerequisites/
    README.md                      # entry guide: Phase 1 vs Phase 2 entry points
    requirements.txt               # torch, torchvision, sklearn, tiktoken
    00-math-foundations/
      math-foundations-for-ml.ipynb  # P-0  (source: notes/00-math-under-the-hood/)
      images/
      images-plan.md
      requirements.txt
    01-ml-basics/
      ml-basics.ipynb               # P-1  (source: notes/01-ml/01-regression+02-class)
      images/
      images-plan.md
      requirements.txt
    02-neural-networks/
      neural-networks-and-backprop.ipynb  # P-2  (source: notes/01-ml/03-nn/ch01–ch04)
      images/
      images-plan.md
      requirements.txt
    03-cnns/
      convolutional-neural-networks.ipynb  # P-3  (source: notes/01-ml/03-nn/ch05)
      images/
      images-plan.md
      requirements.txt
    04-rnn-sequence-modeling/
      rnn-sequence-modeling.ipynb   # P-4  (source: notes/01-ml/03-nn/ch06+ch09)
      images/
      images-plan.md
      requirements.txt
    05-tokenization/
      tokenization-and-embeddings.ipynb  # P-5  (partial source: notes/03-llm/ch00)
      images/
      images-plan.md
      requirements.txt
  genai/                           # unchanged — main genai track
    00-pytorch-primer/
    01-rnns/
    02-transformers/
    03-encoder-decoder/
    04-llm/
  ai-infrastructure/               # future-plan.md track
```

### Notes coverage map (full path)

| Chapter | Notes source | Coverage | What's fresh |
|---|---|---|---|
| P-0 Math foundations | `notes/00-math-under-the-hood/` (all 7 ch) | Full content + Knuckleball scenario | Predict-first Qs, closing decision |
| P-1 ML basics | `notes/01-ml/01-regression/` + `02-classification/` | California Housing + CelebA content | Gold-standard wrapper, images-plan |
| P-2 Neural networks | `notes/01-ml/03-neural-networks/ch01–ch04` | XOR through regularisation notebooks | Backprop-by-hand cell, toy→real bridge |
| P-3 CNNs | `notes/01-ml/03-neural-networks/ch05_cnns/` | Full PyTorch notebook | ResNet gradient proof, images-plan |
| P-4 RNN/LSTM | `notes/01-ml/03-neural-networks/ch06+ch09` | BPTT, LSTM gate proofs | Music scenario, predict-first Qs |
| P-5 Tokenization | `notes/03-llm/ch00` (partial) | Embeddings, `Ġ` prefix explanation | BPE from scratch, law firm scenario |
