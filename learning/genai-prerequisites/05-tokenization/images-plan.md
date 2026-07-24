# 05-tokenization Image Plan

## Asset Rules

Same conventions as `learning/genai/04-llm/image-plan.md`:
- Dark graphite background (`#1E1E2E`)
- Palette: **teal** `#4ECDC4` / **amber** `#FFD166` / **coral** `#FF6B6B` / **ivory** `#F7F3E9`
- 16:9 aspect ratio
- Generator: Perchance image generator only
- No logos, no photorealism, no gradients, no tiny text
- Flat vector / technical infographic style

---

## Planned Assets

| Asset | Placement | Teaching job |
|-------|-----------|--------------|
| `images/bpe-merge-steps.png` | Part 2 intro (before BPE helpers cell) | 3-panel: raw characters → after 10 merges → after 20 merges; `'non-disclosure'` shrinking from 15 token boxes to 2 |
| `images/embedding-space-pca.png` | Part 4 body (before PCA before-training cell) | Side-by-side PCA scatter: random before training vs. legal synonym clusters after training |
| `images/tokenization-pipeline.png` | Part 1 intro (after opening challenge, before Part 1 header) | Left-to-right pipeline: raw string → BPE tokens → integer IDs → embedding vectors → into model |

---

## Perchance Prompts

### `images/bpe-merge-steps.png`

```
Flat vector technical infographic, wide 16:9, dark graphite background (#1E1E2E). Three panels labeled "Raw characters", "After 10 merges", "After 20 merges". Each panel shows the word "non-disclosure" split into progressively fewer tokens, represented as labeled rounded rectangles in ivory. In "Raw characters": 15 individual small character boxes. In "After 10 merges": 5-6 medium boxes with merged subwords. In "After 20 merges": 2 large boxes labeled "non" and "disclosure" in teal. Teal merge-arrow icons between panels. Panel labels in amber at top. Ivory text throughout. No logos, no photorealism, no gradients, no tiny text below 12pt.
```

### `images/embedding-space-pca.png`

```
Flat vector data visualization, wide 16:9, dark graphite background (#1E1E2E). Two side-by-side 2D scatter plots labeled "Before training" and "After training" in amber at top. Left plot: scattered ivory dots with word labels — no spatial clustering structure visible. Right plot: same words but now showing clear clusters — "contract" and "agreement" grouped inside a teal ellipse, "indemnification" and "damages" inside an amber ellipse, "clause" and "provision" inside a coral ellipse. Small legend in bottom-right corner. Ivory axis labels. No logos, no photorealism, no gradients, no tiny text below 12pt.
```

### `images/tokenization-pipeline.png`

```
Flat vector pipeline diagram, wide 16:9, dark graphite background (#1E1E2E). Five stages connected by thick teal right-arrows: Stage 1 ivory box labeled "Raw string" containing 'non-disclosure', Stage 2 amber box labeled "BPE tokens" containing two rounded rectangles, Stage 3 teal box labeled "Integer IDs" containing numbers like 3421 and 8872, Stage 4 coral box labeled "Embedding vectors" showing a small matrix grid of floats, Stage 5 ivory box labeled "Model input" showing a simple transformer block icon. Short stage description in small ivory text below each box. All elements have clear readable font. No logos, no photorealism, no gradients, no tiny text below 12pt.
```
