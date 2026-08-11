# Learning Notebook Image Audit and Designer Queue

This plan covers every notebook under `learning/genai/` and
`learning/genai-prerequisites/` as of 2026-08-09.

## How to Use This Plan

1. Generate images in the order listed below.
2. Leave Copilot Designer downloads with their default names (`Designer.png`,
   `Designer (1).png`, and so on).
3. Do not manually place or rename them. A follow-up audit will inspect each
   downloaded image, match it to this queue by visible content, rename it to the
   proposed filename, move it beside the target notebook, and add or replace the
   notebook reference.
4. A generated image is accepted only if it shows the requested components and
   their interaction. A collection of prose cards with arrows is not sufficient.
5. Exact measured values must come from the notebook. Designer must not invent
   metrics, tensor dimensions, probabilities, latency, cost, or quality claims.

## Audit Summary

- Notebooks audited: **22**
- Markdown links audited: **137**
- Markdown image references audited: **51**
- Confirmed broken file links after fixes: **0**
- Confirmed broken anchors after fixes: **0**
- Case mismatches after fixes: **0**
- Malformed links after fixes: **0**
- Image files with extension/encoding mismatch after fixes: **0**

### Notebook Disposition

| Notebook | Existing visual audit | Planned action |
| --- | --- | --- |
| `genai-prerequisites/00-math-foundations/math-foundations-for-ml.ipynb` | Three strong geometry/flow visuals | No change |
| `genai-prerequisites/01-ml-basics/ml-basics.ipynb` | Three strong measured technical plots | No Designer image; keep charts executable |
| `genai-prerequisites/02-neural-networks/01-smartval-neural-networks-and-backprop.ipynb` | One duplicated/mislabeled image; backprop flow gap | Queue 1-2 |
| `genai-prerequisites/04-cnns/convolutional-neural-networks.ipynb` | Convolution, feature-map, and residual visuals are strong | Queue 3 |
| `genai-prerequisites/05-rnn-sequence-modeling/rnn-sequence-modeling.ipynb` | Gate-equation image is styled text, not mechanism | Queue 4 |
| `genai-prerequisites/06-tokenization/tokenization-and-embeddings.ipynb` | Training GIF is strong; static reference is text-dense | Remove redundant JPG after final review; no replacement needed |
| `genai-prerequisites/03-pytorch-fundamentals/01-keras-to-pytorch-antarctic-field-guide.ipynb` | Two diagrams are useful but too generic/text-heavy | Queue 5-6 |
| `genai-prerequisites/07-pytorch-rnn-bridge/01-pytorch-rnn-bridge.ipynb` | Core diagrams are accurate; overview and handoff are dense | Queue 7-8 |
| `genai/01-transformers/01-attention-and-transformer-blocks.ipynb` | Four strong component/flow visuals | No change |
| `genai/01-transformers/02-decoder-only-language-model.ipynb` | Autoregressive/KV-cache and scale bridge are strong | No change |
| `genai/01-transformers/03-encoder-decoder-and-cross-attention.ipynb` | Seven strong visuals; two mechanism gaps remain | Queue 9-10 |
| `genai/02-llm-finetuning/01-llm-finetuning-data-techniques.ipynb` | Four strong visuals/animations; DPO construction remains prose-heavy | Queue 11 |
| `genai/02-llm-finetuning/02-llm-finetuning-parameter-techniques.ipynb` | Spectrum, LoRA, and QLoRA visuals are strong | No change |
| `genai/02-llm-finetuning/03-llm-finetuning-comparison-and-decision.ipynb` | Executable charts and Mermaid decision flows are sufficient | No change |
| `genai/02-llm-finetuning/04-llm-finetuning-practice.ipynb` | Runtime dashboard is strong; long-run workflow needs a navigator | Queue 12 |
| `genai/03-rag/01-hybrid-search.ipynb` | Hybrid retrieval storyboard is strong | No change |
| `genai/03-rag/02-rag-evaluation.ipynb` | Failure-location map is strong; live charts cover the metrics | No change |
| `genai/04-llm-evaluation/01-llm-evaluation-metrics-and-benchmarks.ipynb` | Code-first charts are strong; taxonomy is hard to scan | Queue 13 |
| `genai/04-llm-evaluation/02-llm-as-judge-safety-and-pipeline.ipynb` | Generated bias/agreement/regression charts are sufficient | No Designer image; keep measured charts executable |
| `genai/04-llm-evaluation/03-hallucination-detection.ipynb` | Code-first metrics are strong; composed guard path is text-only | Queue 14 |
| `genai/04-llm-evaluation/04-calibration-and-confidence.ipynb` | Reliability/coverage plots are strong; decision sequence is text-only | Queue 15 |
| `genai/05-llm-gateway/01-llm-gateway.ipynb` | Lifecycle image is strong; routing and fallback dynamics need flow | Queue 16-17 |

## Generation Queue

### Queue 1 - Replace the Incorrect Depth-vs-Width Asset

- **Notebook:** `genai-prerequisites/02-neural-networks/01-smartval-neural-networks-and-backprop.ipynb`
- **Anchor:** `## Part 4 — Depth Beats Width: Universal Approximation`
- **Action:** REPLACE
- **Current problem:** `depth-vs-width-decision-boundary.png` is byte-for-byte
  identical to `xor-not-linearly-separable.png`; it does not show depth versus
  width.
- **Target filename:** `depth-vs-width-decision-boundary.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a visual comparison titled "Depth vs. Width on the Same Spiral Problem".

Show the SAME two-class spiral dataset in two large side-by-side coordinate plots.

LEFT: "Wide, shallow network"
- Architecture strip above plot: 2 inputs -> one hidden layer with 256 neurons -> 2 outputs.
- Decision boundary should appear coarse or fragmented in difficult spiral turns.
- Add a small path diagram showing that information passes through one nonlinear transformation.

RIGHT: "Deep, narrower network"
- Architecture strip above plot: 2 inputs -> four hidden layers with 32 neurons each -> 2 outputs.
- Decision boundary should follow the spiral structure more naturally.
- Add a path diagram showing four successive feature transformations.

INTERACTION TO SHOW
- Use identical axes, point locations, class colors, train/test split, and plot scale on both sides.
- Draw a thin central arrow labeled "same data, similar parameter budget, different composition depth".
- Beneath the plots, show a compact transformation rail:
  shallow: input -> one bend -> output
  deep: input -> bend -> bend -> bend -> bend -> output
- The visual claim is representational composition, not that depth always wins.

STYLE
- Match the notebook's light matplotlib aesthetic.
- Class 0 steel blue; Class 1 coral.
- White background, subtle grid, dark gray labels, no decorative frame.
- Large readable plot labels; minimal explanatory prose.

AVOID
- Do not reuse the XOR plot.
- Do not invent accuracy values.
- Do not claim the deep model always generalizes better.
- No text-card collage, 3D surface, stock imagery, or decorative neural-network icons.
```

### Queue 2 - Add the Backpropagation Route

- **Notebook:** `genai-prerequisites/02-neural-networks/01-smartval-neural-networks-and-backprop.ipynb`
- **Anchor:** `## Part 3 — Backpropagation by Hand`
- **Action:** ADD immediately after the forward-pass recap and before the manual
  derivative calculation.
- **Target filename:** `backprop-forward-and-backward-route.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a computational graph titled "One XOR Example: Forward Values, Backward Feedback".

Use the notebook's 2 -> 2 -> 1 XOR network and one input example (x1, x2).

FORWARD FLOW, left to right in blue/teal
- Input nodes x1 and x2.
- Hidden pre-activation z1.
- ReLU hidden activations h1 and h2.
- Output pre-activation z2.
- Sigmoid prediction y_hat.
- Binary cross-entropy loss L compared with target y.
- Show arrows carrying values, not paragraphs.

BACKWARD FLOW, right to left in coral
- Loss sends an error signal to y_hat.
- Signal passes through sigmoid, output weights, ReLU gates, and input weights.
- Label the route with short phrases only:
  "prediction error", "local slope", "credit to W2", "ReLU gate", "credit to W1".
- At a ReLU unit that was inactive, show the backward signal stopping at a closed gate.
- At an active ReLU unit, show the signal continuing.

KEY VISUAL
- Overlay forward arrows above the network and backward arrows below it so the learner sees the same graph traversed in opposite directions.
- Add one compact takeaway below: "Backprop does not move the error as a number; it assigns responsibility through local slopes."

STYLE
- Light background matching existing XOR figures.
- Steel blue forward flow, coral backward flow, amber loss node.
- Flat 2D nodes and arrows, readable mathematical labels.

AVOID
- No page of equations in an image.
- No unexplained derivative notation.
- No decorative brain imagery, 3D network, or dense prose cards.
```

### Queue 3 - Show Frozen Backbone vs. Trainable Head

- **Notebook:** `genai-prerequisites/04-cnns/convolutional-neural-networks.ipynb`
- **Anchor:** `## Part 5 — Transfer Learning`
- **Action:** ADD before the head-only transfer code.
- **Target filename:** `mobilenetv2-frozen-backbone-trainable-head.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create an architecture flow titled "Transfer Learning: Freeze the Visual Extractor, Train a New Decision Head".

LEFT TO RIGHT FLOW
1. Input: a resized 32x32 RGB MNIST digit.
2. MobileNetV2 pretrained backbone represented as a sequence of shrinking feature-map blocks:
   32x32 -> 16x16 -> 8x8 -> 4x4.
3. Global average pooling converts the final feature maps into one feature vector.
4. New Dense(2) head produces logits for digit 0 vs digit 1.

TRAINABILITY INTERACTION
- Backbone blocks are desaturated gray and enclosed by a bracket labeled "ImageNet features - frozen".
- Put small lock symbols directly on backbone blocks.
- Head is bright blue/teal and labeled "new trainable head".
- Draw backward gradient arrows from the loss through the head, stopping visibly at the frozen-backbone boundary.
- Draw forward data arrows through the entire network.

HIERARCHY
- Feature extractor occupies about 65% of width.
- Pooling/head occupies 25%.
- A narrow loss node at the far right sends the backward arrow.

STYLE
- White background and flat block-arrow language matching the ResNet diagram.
- Gray frozen backbone, teal trainable head, coral backward feedback.
- Minimal labels, no layer-by-layer MobileNet inventory.

AVOID
- No claim that ImageNet features are optimal for digits.
- No invented accuracy or parameter counts.
- No text-card collage, gradients, decorative icons, or photorealistic devices.
```

### Queue 4 - Replace Equation Screenshot with an LSTM Mechanism

- **Notebook:** `genai-prerequisites/05-rnn-sequence-modeling/rnn-sequence-modeling.ipynb`
- **Anchor:** `## Part 4 — LSTM Gating: The Cell Highway`
- **Action:** REPLACE `lstm-gate-equations.png`
- **Target filename:** `lstm-cell-state-highway.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a technical diagram titled "LSTM Cell State Highway Across Three Timesteps".

LAYOUT
- Three columns: t-1, t, t+1.
- At every timestep, x_t enters from below and h_{t-1} enters from the left.
- Two horizontal paths travel across all columns:
  thin lower path = hidden state h
  thick upper path = cell state c, labeled "additive memory highway".

INSIDE EACH LSTM CELL
- Forget gate f: controls how much prior cell state continues.
- Input gate i: controls how much new candidate information enters.
- Candidate g: proposed new content.
- Addition node: combines retained old state with gated candidate.
- Output gate o: controls the hidden state exposed from tanh(c_t).
- Use short gate symbols plus one-word labels; do not reproduce all equations.

INTERACTION TO EMPHASIZE
- Forward arrows show c_{t-1} passing through the forget gate, then combining at a plus node with i_t times candidate content.
- A coral backward-gradient arrow travels from t+1 toward t-1 along the thick cell-state highway with little interruption.
- Contrast this visually with smaller branch arrows through gate computations.
- The plus nodes should be the dominant visual explanation for why the path differs from repeated vanilla-RNN tanh updates.

STYLE
- Match the existing blue/coral/green prerequisite palette.
- Flat computational graph, white background, strong arrowheads, large symbols.

AVOID
- No equation sheet, prose paragraphs, or decorative memory metaphor.
- Do not claim gradients never vanish or that LSTMs always solve long context.
- No stock icons or 3D cells.
```

### Queue 5 - Make the Logits Contract Palmer-Specific

- **Notebook:** `genai-prerequisites/03-pytorch-fundamentals/01-keras-to-pytorch-antarctic-field-guide.ipynb`
- **Anchor:** `## Part 3 — Logits First, Probabilities Only When You Need Them`
- **Action:** REPLACE `images/2.png`
- **Target filename:** `raw-logits-training-vs-inference.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a dark technical flow diagram titled "Raw Logits: Training vs. Inference" for the Palmer Penguins three-class classifier.

START
- One classifier output with exactly three raw score bars labeled Adelie, Chinstrap, Gentoo.

SPLIT INTO THREE ROUTES
1. TRAINING - correct
   raw logits -> CrossEntropyLoss + integer species target -> scalar loss -> gradients.
2. INFERENCE - correct
   raw logits -> softmax -> three probabilities -> argmax -> predicted species.
3. TRAINING - incorrect
   raw logits -> softmax too early -> CrossEntropyLoss receives probabilities.
   Mark this branch with a clear stop/error boundary.

INTERACTION
- The first classifier output physically branches into the three routes.
- Training gradient arrows return from loss toward the classifier.
- Inference ends at a species label and has no backward arrow.
- Keep exactly three bars throughout so the graphic matches the dataset.

STYLE
- Match the notebook's existing black background and neon-outline vocabulary.
- Green correct paths, blue operations, amber loss, coral incorrect path.
- Minimum readable body text at notebook width.

AVOID
- No ellipsis implying extra classes.
- No invented logits or probabilities unless clearly schematic.
- No long explanatory cards, stock penguin images, 3D effects, or decorative dashboard frame.
```

### Queue 6 - Simplify the Antarctic Field Guide Overview

- **Notebook:** `genai-prerequisites/03-pytorch-fundamentals/01-keras-to-pytorch-antarctic-field-guide.ipynb`
- **Anchor:** opening visual reference `images/4.png`, directly after the roadmap table
- **Action:** REPLACE `images/4.png`
- **Target filename:** `keras-to-pytorch-antarctic-workflow.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a dark left-to-right workflow titled "Keras to PyTorch: Antarctic Field Guide".

SHOW SEVEN CONNECTED STAGES
1. Palmer Penguins table: bill length, bill depth, flipper length, body mass, species.
2. Typed tensors: X float32, y long integer class IDs.
3. Model: Keras Dense maps to PyTorch nn.Linear inside nn.Module.
4. Raw logits and CrossEntropyLoss: exactly three species scores, no softmax during training.
5. Explicit training cycle: zero_grad -> forward -> loss -> backward -> optimizer.step.
6. Evaluation: eval + no_grad -> probabilities -> predicted species.
7. Controlled ablations: change one variable, compare held-out evidence.

INTERACTION
- A single prominent data arrow connects all stages.
- Beneath stages 3-6, add a thin return arrow labeled "gradients update parameters during training only".
- Use small Keras/PyTorch labels only where terminology changes; do not duplicate every stage as two text cards.

STYLE
- Black background, white text, blue process blocks, green data, amber loss, coral evaluation/decision accents.
- Large labels and generous spacing.

AVOID
- No dense footer legend, tiny prose, decorative penguins, fabricated measurements, or seven separate paragraph cards.
```

### Queue 7 - Make the Symbolic Music Pipeline Faithful and Scannable

- **Notebook:** `genai-prerequisites/07-pytorch-rnn-bridge/01-pytorch-rnn-bridge.ipynb`
- **Anchor:** opening visual reference `images/11.png`, directly after the roadmap
- **Action:** REPLACE `images/11.png`
- **Target filename:** `symbolic-music-rnn-training-and-generation.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a dark technical flow titled "PyTorch RNN Bridge: From Symbolic Motif to Next Token".

USE THE NOTEBOOK'S MOTIF
- Show a compact example containing <BOS>, KEY_C, C, E, G, <BAR>, reversed motif tokens, and <EOS>.

MAIN TRAINING FLOW
symbol tokens -> long token IDs (B,T) -> embedding lookup (B,T,D) -> packed LSTM -> hidden sequence (B,T,H) + final h/c states -> vocabulary head -> raw logits (B,T,V) -> shifted targets with PAD ignored -> cross-entropy loss.

GENERATION INSET
seed prefix -> one LSTM step with carried h/c -> logits -> temperature + sampling -> append next token -> repeat until EOS.

INTERACTION
- Forward training path dominates the top two-thirds.
- Generation loop occupies one clear inset, with a loop arrow from sampled token back to the LSTM input.
- Padding mask appears only at target/loss boundary.
- Use state arrows to show h/c carried through time.

STYLE
- Match existing black background with blue IDs, green embeddings, purple recurrence, amber projection, coral loss, teal generation.
- Large labels; reduce Keras comparison to one small mapping strip or omit it.

AVOID
- No generic C-E-G sequence that omits KEY_C or the reversed bar.
- No dense API comparison cards, tiny tensor prose, 3D blocks, or decorative dice larger than the mechanism.
```

### Queue 8 - Clarify RNN Recurrence vs. Causal Attention

- **Notebook:** `genai-prerequisites/07-pytorch-rnn-bridge/01-pytorch-rnn-bridge.ipynb`
- **Anchor:** `## Part 5 — Transformer Handoff: Keep the Contract, Replace the Path`
- **Action:** REPLACE `images/10.png`
- **Target filename:** `recurrent-state-vs-causal-attention.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a dark side-by-side mechanism comparison titled "Keep the Contract, Replace the Information Path".

LEFT: RNN/LSTM
- Tokens x0, x1, x2, x3 in a row.
- A recurrent state h/c travels strictly left to right through every timestep.
- Emphasize the long sequential path from x0 to the representation at x3.

RIGHT: Decoder-only causal self-attention
- Same tokens and same positions.
- Output z3 receives direct fan-in arrows from x0, x1, x2, x3.
- z2 receives x0, x1, x2 but not x3.
- Include a small lower-triangular causal mask that uses the same allowed/blocked colors as the arrows.

SHARED CONTRACT RAIL
- Token IDs -> embeddings -> (B,T,V) logits -> raw-logit cross-entropy -> autoregressive generation.
- Place this rail beneath both models to show what does not change.

STYLE
- Black background; purple recurrence path; teal causal-attention arrows; amber shared contract; coral blocked future links.
- Use explicit arrowheads and large labels.

AVOID
- No symmetric attention arrows.
- No implication that decoder attention can read future tokens.
- No dense prose cards, decorative icons, or unrelated architecture details.
```

### Queue 9 - Expose the Asymmetric Cross-Attention Matrix

- **Notebook:** `genai/01-transformers/03-encoder-decoder-and-cross-attention.ipynb`
- **Anchor:** `#### Your turn — does cross-attention stay well-defined when T ≠ S?`
- **Action:** ADD after the existing high-level cross-attention bridge
- **Target filename:** `cross-attention-asymmetric-query-source-matrix.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a Riverside dark-theme mechanism diagram titled "Cross-Attention: Decoder Queries, Source Keys and Values".

COMPONENTS
- Decoder states of length T feed a Q projection.
- Fixed encoder outputs of length S fork into K and V projections.
- Q and K meet in a rectangular T-by-S score matrix.
- The softened score matrix mixes V to return one source-aware vector per decoder position.

MATRIX INTERACTION
- Rows labeled decoder step 0 ... T-1 in coral.
- Columns labeled Riverside source tokens in teal/gold, such as Aria, heard, patient, signal, aboard, Meridian.
- Highlight one decoder row attending strongly to patient and signal.
- Show all source columns available for every decoder row; do not draw a causal triangle across the source dimension.

CONTRAST INSET
- Small square self-attention T-by-T causal matrix beside the rectangular cross-attention T-by-S matrix.
- Caption: "causal mask applies to decoder self-attention, not to the already encoded source".

STYLE
- Dark navy Riverside palette, gold encoder, coral decoder, teal cross-attention.
- Shape labels may appear once per path; keep batch/head details secondary.

AVOID
- No full equation page, no claim that padding masks are unnecessary, no dense prose cards, and no symmetric encoder/decoder roles.
```

### Queue 10 - Show Exposure Bias as a Diverging History

- **Notebook:** `genai/01-transformers/03-encoder-decoder-and-cross-attention.ipynb`
- **Anchor:** `### 6a. From Teacher-Forced Proof to Free-Running Generation`
- **Action:** ADD before the measured free-running comparison
- **Target filename:** `teacher-forcing-vs-self-fed-history.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a Riverside dark-theme branching sequence diagram titled "Exposure Bias: Gold History vs. Self-Fed History".

TOP LANE - TRAINING WITH TEACHER FORCING
- Source is encoded once.
- Decoder inputs follow the correct target prefix at every step:
  BOS -> Aria -> detects -> a -> patient -> signal.
- Each step predicts the next gold token.
- Keep the path green/teal and straight.

BOTTOM LANE - FREE-RUNNING INFERENCE
- Start from BOS.
- Step 1 predicts Aria correctly.
- Step 2 predicts a plausible but wrong token, for example heard instead of detects.
- Feed that wrong token back into the next step.
- Show subsequent hidden-history boxes diverging farther from the training lane.
- Keep the divergent branch coral/red.

INTERACTION
- A vertical comparison line at each timestep connects the training prefix and inference prefix.
- Before the first mistake, lines align; after the mistake, the distance visibly grows.
- Cross-attention may still revisit the complete Riverside source in both lanes.

TAKEAWAY
- One short caption: "Training practices on gold prefixes; inference must continue from its own choices."

AVOID
- No fabricated accuracy percentages.
- Do not present scheduled sampling as the universal remedy.
- No probability tables, text-card collage, decorative characters, or stock imagery.
```

### Queue 11 - Make DPO Pair Construction Visible

- **Notebook:** `genai/02-llm-finetuning/01-llm-finetuning-data-techniques.ipynb`
- **Anchor:** `How This Notebook Constructs Its Riverside Pairs`
- **Action:** ADD before sample triplet inspection
- **Target filename:** `riverside-dpo-pair-construction.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a Riverside dark-theme pipeline titled "Building One DPO Preference Triple".

LEFT TO RIGHT COMPONENTS
1. Manuscript context from paragraph i becomes the prompt.
2. First sentence from paragraph i+1 becomes the chosen continuation.
3. A coherent but stalling response is selected from the rejected-response bank.
4. Validation gate checks:
   - complete surface form
   - chosen/rejected length difference within the notebook's tolerance
   - response token budget including EOS
5. Output one record with prompt, chosen, and rejected fields.

INTERACTION
- Show the manuscript splitting into context and authentic continuation.
- Show three rejected candidates entering a selector; only one passes the constraint gate.
- Use token-length rulers below chosen and rejected so their comparison is visual.
- The final triple should show structure, not long copied prose.

STYLE
- Dark Riverside palette: slate context, gold chosen, coral rejected, teal validation/pass.
- Flat arrows, minimal labels, readable at notebook width.

AVOID
- Do not imply that rejected responses are random or factually false by definition.
- Do not invent exact token counts.
- No large prose cards, decorative manuscript imagery, or training-loss charts.
```

### Queue 12 - Add a Long-Run Fine-Tuning Navigator

- **Notebook:** `genai/02-llm-finetuning/04-llm-finetuning-practice.ipynb`
- **Anchor:** `### Read the Runner as Nine Visible Stages`
- **Action:** ADD above the existing Mermaid source; retain Mermaid as accessible text
- **Target filename:** `nine-stage-lora-experiment-workflow.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a dark technical workflow titled "Nine-Stage LoRA CPT Experiment".

FLOW LEFT TO RIGHT IN THREE BANDS
DATA PREPARATION
1. Load frozen split.
2. Build training records.

TRAIN AND SELECT
3. Train candidate checkpoints.
4. Select using validation NLL only.
5. Reload the selected checkpoint and verify it.

OPEN TEST AND DECIDE
6. Open the previously sealed test set.
7. Bootstrap uncertainty.
8. Apply PASS / FAIL / INCONCLUSIVE gates.
9. Write result plus manifest.

INTERACTION
- Use one continuous arrow through all nine stages.
- Place lock icons on validation/test boundaries; the test lock opens only after stage 5.
- Stages 4 and 8 are decision diamonds, not ordinary cards.
- Draw artifact arrows from stages 3, 5, and 9 to checkpoint/manifest files below the flow.

STYLE
- Dark VS Code/Jupyter palette.
- Teal data stages, amber compute/uncertainty, gold decision gates, blue artifacts.
- Large stage numbers, short labels, minimal prose.

AVOID
- No branching paths beyond the two decision points.
- No cloud-service imagery, fabricated runtime, text-card collage, or decorative icons.
```

### Queue 13 - Turn the Evaluation Landscape into a Taxonomy

- **Notebook:** `genai/04-llm-evaluation/01-llm-evaluation-metrics-and-benchmarks.ipynb`
- **Anchor:** `The Full Landscape of LLM Evaluation`
- **Action:** ADD above the detailed table; keep the table for exact scope notes
- **Target filename:** `llm-evaluation-method-taxonomy.png`
- **Target directory:** create `genai/04-llm-evaluation/images/` during integration
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a clean taxonomy diagram titled "LLM Evaluation Methods: What Evidence Do They Use?"

ROOT
LLM evaluation methods

FIRST SPLIT
- Reference-based evidence
- Reference-free evidence
- Human or model judgment
- Task-specific benchmark

SECOND LEVEL
Reference-based:
- surface overlap: BLEU, ROUGE-L, METEOR
- semantic similarity: BERTScore
Reference-free:
- likelihood/perplexity
Human or model judgment:
- LLM-as-judge, pairwise preference, human review, safety review
Task-specific:
- custom MCQ and benchmark harnesses

SCOPE ENCODING
- Solid teal border = built or executed in this notebook.
- Amber border = covered in Part 2.
- Gray dashed border = named only.
- Add a small legend; do not repeat scope prose in every node.

INTERACTION
- Use branching lines that reveal shared evidence sources.
- Place warning tags beside surface-overlap and semantic metrics: "can miss factuality".
- Place a cost/latency scale along the bottom from cheap/deterministic to expensive/judgment-based.

STYLE
- Light neutral background, flat taxonomy tree, concise labels.
- Green/teal, amber, and gray scope colors.

AVOID
- No invented metric rankings.
- Do not imply one branch is universally superior.
- No table screenshot, text-card collage, decorative icons, or tiny leaf text.
```

### Queue 14 - Visualize the Layered Hallucination Guard

- **Notebook:** `genai/04-llm-evaluation/03-hallucination-detection.ipynb`
- **Anchor:** `## Part 6 — The Hallucination-Aware Pipeline`
- **Action:** ADD before the composed guard code
- **Target filename:** `hallucination-guard-layered-routing.png`
- **Target directory:** create `genai/04-llm-evaluation/images/` during integration
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a production flow diagram titled "Layered Hallucination Guard" using the Riverside answer/context example.

LEFT TO RIGHT
1. Retrieved context plus candidate answer.
2. NLI attribution layer asks whether answer claims follow from context.
3. Entity-gap layer extracts people, organizations, dates, and roles, then highlights unsupported entities.
4. Consistency layer samples alternate answers only when earlier evidence is uncertain.
5. Composite decision emits LOW, MEDIUM, or HIGH hallucination risk.

ROUTING INTERACTION
- NLI high-risk result can route directly to HIGH.
- Entity gaps add specific evidence to the review output.
- Borderline composite scores trigger the slower consistency layer asynchronously.
- Low-risk answers bypass the expensive layer.

EXAMPLE
- Show Elena Marchetti as an unsupported entity moving through the entity-gap branch and being surfaced in the final review packet.

STYLE
- Light professional evaluation palette.
- Teal pass, amber review, coral high risk, blue evidence extraction.
- Decision diamonds and explicit bypass arrows; minimal prose.

AVOID
- No invented thresholds or latency numbers.
- Do not imply sampling consistency proves truth.
- No generic shield icon as the main content, text-card collage, or decorative dashboard.
```

### Queue 15 - Show the Two-Gate Confidence Decision Sequence

- **Notebook:** `genai/04-llm-evaluation/04-calibration-and-confidence.ipynb`
- **Anchor:** `## Part 6 — The Confidence-Gated Pipeline`
- **Action:** ADD before the final pipeline implementation
- **Target filename:** `hallucination-and-confidence-two-gate-flow.png`
- **Target directory:** create `genai/04-llm-evaluation/images/` during integration
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a decision flowchart titled "Two Gates Before Riverside Serves an Answer".

START
Candidate answer with hallucination risk and calibrated confidence.

GATE 1 - HALLUCINATION
- Decision: high hallucination risk?
- Yes -> HOLD FOR REVIEW, regardless of confidence.
- No -> continue to confidence gate.

GATE 2 - CALIBRATED CONFIDENCE
- High enough for direct service -> SERVE.
- Middle confidence band -> SERVE WITH CAVEAT.
- Below refusal threshold -> REFUSE.

INTERACTION
- Use explicit yes/no arrows.
- Show a high-confidence but high-hallucination example routed to review to prove confidence cannot override Gate 1.
- Show a low-risk but low-confidence example routed to refusal.
- Put user-facing outcome boxes at the far right.

STYLE
- Light professional palette matching reliability plots.
- Coral hold/review, teal serve, amber caveat, gray refuse.
- Decision diamonds, minimal labels, no exact numeric thresholds unless copied during final integration from notebook output.

AVOID
- No claim that confidence measures factual truth.
- No arbitrary percentages, text-card collage, decorative locks, or stock customer-service imagery.
```

### Queue 16 - Animate Queue-Aware Routing as a Timeline

- **Notebook:** `genai/05-llm-gateway/01-llm-gateway.ipynb`
- **Anchor:** `### Code Walkthrough: Round-Robin vs Least-Busy Load Balancing`
- **Action:** ADD before the measured routing simulation
- **Target filename:** `round-robin-vs-least-busy-routing-timeline.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a dark side-by-side sequence timeline titled "Fixed Rotation vs. Queue-Aware Routing".

LEFT - ROUND ROBIN
- Six request arrivals descend along a time axis.
- Three provider lanes A, B, C.
- Requests route A -> B -> C -> A -> B -> C regardless of current queue.
- Show queue-depth counters changing as requests arrive and completions free capacity.

RIGHT - LEAST BUSY
- Use the same six arrival times and the same completion events.
- Before each dispatch, highlight the current queue counters.
- Route the request to the provider with the smallest current queue.
- Draw the chosen route only after comparing all three queues.

INTERACTION
- Queue depth must visibly update after every arrival/completion.
- A small footer compares the decision rule, not fabricated final counts:
  "fixed schedule" vs. "observe state, then choose".

STYLE
- Match gateway dark slate theme.
- Provider lanes in teal/blue, request arrows amber, completion events gray, selected least-busy provider green.
- Large queue counters and explicit arrowheads.

AVOID
- No invented benchmark totals, latency, or provider names.
- No static prose cards masquerading as a timeline.
- No gradients, 3D servers, cloud logos, or decorative networking icons.
```

### Queue 17 - Show Reliability Multiplying Through Fallback

- **Notebook:** `genai/05-llm-gateway/01-llm-gateway.ipynb`
- **Anchor:** `2. **Multiplicative chain**: Only fails if all three fail simultaneously`
- **Action:** ADD before the reliability calculation
- **Target filename:** `fallback-chain-success-and-failure-cascade.png`
- **Aspect ratio:** 16:9

**Designer prompt**

```text
Create a dark gateway flowchart titled "Fallback Succeeds Unless Every Provider Fails".

VERTICAL CASCADE
1. Incoming request reaches primary provider.
   success branch -> response
   failure branch -> fallback 1
2. Fallback 1.
   success branch -> response
   failure branch -> fallback 2
3. Fallback 2.
   success branch -> response
   failure branch -> complete request failure

INTERACTION
- Green success branches peel off to a shared Response Sent rail.
- Coral failure branches continue down the cascade.
- Beside each failure branch, leave a clearly marked placeholder for the notebook's measured/configured failure probability.
- At the bottom, show the multiplication structure:
  all-fail probability = primary fail × fallback-1 fail × fallback-2 fail.
- Make the all-red path visually narrow compared with the three success exits.

STYLE
- Dark slate gateway palette.
- Teal primary, amber fallback 1, blue fallback 2, green success, coral failure.
- Decision nodes and arrows dominate; minimal prose.

AVOID
- Do not invent provider names, probabilities, costs, or quality levels.
- No claim that failures are independent unless the notebook explicitly states the modeling assumption beside the image.
- No cloud logos, text-card collage, gradients, or 3D effects.
```

## Existing Asset Cleanup After Integration

Do not delete these until the generated replacements have been reviewed and the
notebook references updated.

| Asset | Planned disposition | Reason |
| --- | --- | --- |
| `genai-prerequisites/02-neural-networks/images/depth-vs-width-decision-boundary.png` | Replace via Queue 1 | Duplicate of XOR asset; wrong content |
| `genai-prerequisites/05-rnn-sequence-modeling/images/lstm-gate-equations.png` | Replace via Queue 4 | Equations already exist in Markdown; image adds no mechanism |
| `genai-prerequisites/06-tokenization/images/how-a-token-learns-where-to-live.jpg` | Remove after final visual review | Animated training loop communicates the flow with less text density |
| `genai-prerequisites/06-tokenization/images/bpe-merge-steps.png` | Remove if no future reference is planned | Globally unreferenced and redundant with executable BPE tracing |
| `genai-prerequisites/06-tokenization/images/embedding-space-pca.png` | Remove if no future reference is planned | Globally unreferenced and redundant with executable shared projection |
| Seven unreferenced `genai/02-llm-finetuning/images/*.png` assets | Review during Designer integration | Some may be older candidates; do not delete as a batch without opening each |

## Integration Checklist for the Follow-Up Pass

- [ ] Match each `Designer*.png` by visible content, not download order alone.
- [ ] Verify actual file encoding before choosing `.png`, `.jpg`, or `.gif`.
- [ ] Reject an image if labels are clipped, invented, too small, or inconsistent
      with the target notebook's running example.
- [ ] Rename accepted images to the target filename listed above.
- [ ] Move them into the target chapter's `images/` directory.
- [ ] Add or replace the notebook Markdown reference at the named visible anchor.
- [ ] Preserve descriptive alt text that explains the mechanism rather than the
      image's appearance.
- [ ] Re-run the structured notebook-link audit.
- [ ] Re-run image signature/extension validation.
- [ ] Re-run notebook JSON and Python syntax validation.
- [ ] Confirm no `Designer*.png` files remain after all accepted assets are moved.
