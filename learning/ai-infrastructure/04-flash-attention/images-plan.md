# Images Plan — flash-attention-internals

Images are referenced in the notebook but not yet generated.
Use the prompts below with an image generation tool (e.g. Perchance, DALL-E, Midjourney).

---

## standard-attention-io.png

**Used in:** Part 1 (Cell 4/5) — illustrates the HBM round-trips for the standard attention forward pass.

**Perchance prompt:**
```
Technical diagram on a dark background, showing GPU memory hierarchy for standard attention.
Left column labeled "HBM (High Bandwidth Memory)" with three large horizontal bars stacked
vertically, each labeled: "(1) Write S×S scores", "(2) Read scores / Write softmax", "(3) Read softmax → output".
Right column labeled "SRAM (On-chip)" with a small box showing "matmul tiles".
Red arrows connecting HBM bars showing 5–6 round-trips. Caption at bottom: "O(S²) HBM traffic".
Clean white labels, blue/coral color palette, no gradients, minimal style, 16:9 aspect ratio.
```

---

## flash-attention-tiling.png

**Used in:** Part 2 (Cell 7/8) — visualizes the tile-based computation that avoids materializing S×S in HBM.

**Perchance prompt:**
```
Technical diagram showing FlashAttention tiling algorithm on a dark background.
Top row: full Q matrix (S rows × D cols) divided into BLOCK-sized horizontal strips, labeled Q_0, Q_1, Q_2.
Right side: full K and V matrices similarly striped.
Center: a single highlighted BLOCK×BLOCK tile labeled "S_ij tile (SRAM only)" with a glowing border.
Curved green arrow from the tile directly to "Output O" (no arrow passing through HBM).
Faded gray annotation: "S×S never leaves SRAM". Caption: "O(S) HBM traffic".
Steel-blue and green palette, dark background, precise grid lines, 16:9 aspect ratio.
```

---

## kv-cache-memory-gqa.png

**Used in:** Part 6 (Cell 15) — shows the memory footprint comparison between MHA, GQA-8, and MQA.

**Perchance prompt:**
```
Bar chart comparison diagram on a clean white background. Three vertical bars, left to right:
(1) "MHA" — tall coral bar, labeled "32 KV heads, ~2 GB".
(2) "GQA-8" — medium steelblue bar, labeled "4 KV heads, ~0.25 GB", marked "LLaMA-3 / Mistral".
(3) "MQA" — short green bar, labeled "1 KV head, ~0.06 GB".
Each bar has a small "Q heads" section on top in lighter shade showing Q projections are unchanged.
Y-axis labeled "KV cache (GB)", title "KV Cache: MHA vs GQA vs MQA at S=2048".
8× reduction brace between MHA and GQA-8. Minimal flat design, no gradients. 16:9 aspect ratio.
```
