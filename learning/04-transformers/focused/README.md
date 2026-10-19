# Transformer-focused returning-reader route

**Contents:** [one-line map of this directory](CONTENTS.md)

This is an **optional, derived shortcut**, not a replacement for the [canonical Stage 04 route](../README.md). Use it when token IDs, learned vector lookups, recurrence, separate reader/writer roles, and practice with correct previous target tokens are already familiar. The canonical route stays the authoritative one-stop explanation, including its prerequisite recap.

You start with `the cat sat on the mat`. Its vectors identify tokens but cannot yet exchange information or tell first from last. You then build a complete block, stop it from reading future answers, and let a writer look back at a separate source. Integer reversal makes those source addresses visible without changing tokenizers.

## Readiness check

Answer these before taking the shortcut. For any **no**, use the linked explanation and return here.

| Can you explain this in your own words? | If not, return to |
|---|---|
| Why is a token ID an address rather than a numerical meaning? | [Canonical Part 1](../01-token-embeddings-to-transformers.ipynb) and [Stage 03 tokenization](../../03-language-model-foundations/02-tokenization-and-embeddings/README.md) |
| How does an ID retrieve a trainable vector row, and how can loss change it? | [Canonical Part 1](../01-token-embeddings-to-transformers.ipynb) |
| What does a recurrent state remember, and why can a single summary make source details hard to access? | [Stage 03 sequence memory](../../03-language-model-foundations/01-sequence-memory/README.md) and [model evolution](../../03-language-model-foundations/03-language-models/README.md) |
| Why might a reader process a complete source before a writer makes a separate answer? | [Existing seq2seq foundations](../../03-language-model-foundations/03-language-models/03A-training-skeleton-and-word-objectives.ipynb) |
| During practice, why is the correct previous target allowed but the current answer forbidden? What changes during generation? | [Stage 03 model evolution](../../03-language-model-foundations/03-language-models/README.md) and [canonical Part 4A](../04a-decoder-only-language-model-theory.ipynb) |
| Can you follow tensor shapes, a loss, and a backward pass in PyTorch? | [Training skeleton refresher](../../02-pytorch-foundations/quick-refresher/pytorch-training-skeleton.ipynb) |

## Route

| Order | Notebook | Complaint and measured payoff |
|---|---|---|
| 1 | [Attention, position, and queries/keys/values](01-attention-position-and-qkv-focused.ipynb) | Tokens cannot exchange context or recognize order; build retrieval, expose missing position, keep the rotating-dial attempts, and measure score saturation. |
| 2 | [The complete Transformer block](02-transformer-block-focused.ipynb) | One routing table cannot keep conflicting jobs separate; add heads, per-token processing, controlled feature scales, and direct residual paths, then inspect gradients. |
| 3 | [Decoder and encoder-decoder](03-decoder-and-encoder-decoder-focused.ipynb) | A writer can cheat by reading future answers or lose source addresses; measure future-blind masks, repeated prefix work, and retrieval from separate source positions. |

Every notebook runs independently from a fresh kernel. No notebook imports or executes the canonical notebooks. Setup cells restore small local examples, not prerequisite lectures. The sentence changes to `aria heard the signal` for the canonical future-answer test, then to `[3, 1, 4, 1]` for source routing; the third notebook states why these diagnostic tasks differ.

Use the existing [Stage 04 setup](../README.md) and select **Python (Stage 04 Transformers .venv)** (`learning-04-transformers`). The stage setup finds notebooks recursively. No extra environment or downloaded model is needed for these examples. Run all cells in order. Animations have static plots beside them, so the route still teaches in a non-interactive export.

## Provenance and deliberate replacements

All copies derive from pushed revision **`e94a33f31f593e2329c0811226bb7cbbfcf0008c`**, not from later working-tree edits. Each notebook opens with its own source-heading table. Existing images are reused from `../images/`; no new generated images are requested.

| Focused notebook | Canonical source | Source headings retained | What was replaced |
|---|---|---|---|
| 1 | [Part 2](../02-attention-and-position.ipynb) | Minimal Attention; Add Position Before Attention; the projection complaint; RoPE discovery attempts and invariants; Q, K, and V; Real Score Path; Why Scale the Scores; Chapter 2 Checkpoint | The prerequisite recap and its two setup cells become a short recap and one compact setup cell. |
| 2 | [Part 3](../03-complete-transformer-block.ipynb) | Multi-Head Attention and incompatible jobs; FFN; LayerNorm evidence; Residual Paths and Pre-Normalization; gradient-route comparison; From Block Outputs to Predictions; Chapter 3 Checkpoint | The opening recap/setup is focused on known inputs; every Transformer experiment remains. |
| 3 | [Part 4A](../04a-decoder-only-language-model-theory.ipynb) and [Part 5A](../05a-encoder-decoder-and-cross-attention-theory.ipynb) | Dishonest shortcut; three mask jobs; cache cost storyboards/animation; temperature; source pooling collision; rectangular cross-attention routing/animation; changed-prefix trace; three reusable states; both checkpoints | Separate import cells become one setup. The basic prefix/teacher-forcing recaps become brief references while preserving the parallel-training versus serial-generation boundary and changed-prefix evidence. |

Copied lesson sources are unchanged apart from relative link paths. New introductions, bridges, scope tables, and local setup surround them. Historical canonical labels such as “Part 4B” still point to the full implementation labs; the focused route ends with theory-level routing evidence, not those training labs. The canonical handwritten companions remain linked beside their copied lessons; this directory does not duplicate them.

## Continue without losing the boundary

The experiments check mechanics, not language understanding. Hand-written routing scores do not demonstrate learned reversal. A backward pass does not demonstrate a trained language model. Mean pooling deliberately removes source addresses; it is not a substitute for a learned recurrent summary.

For implementation and training, continue to [Part 4B](../04b-decoder-only-language-model-lab.ipynb) and [Part 5B](../05b-encoder-decoder-and-cross-attention-lab.ipynb). Then use [Part 6](../06-modern-decoder-only-llm.ipynb) for modern block refinements. Keep the [canonical theory notes](../README.md) as the detailed reference.

## Maintenance rule

**If canonical changes, update or mark stale.** A change to a copied mechanism must update the affected focused copy in the same change, or clearly mark that copy stale with the last checked revision. Preserve the failure experiment, its consequence, exercises, and static visual evidence together. Re-run structural, fresh-kernel, source-equality, and local-link checks. Do not quietly treat this shortcut as the source of truth or add synchronization machinery.
