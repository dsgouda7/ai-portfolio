# Images Plan — ML Basics Notebook

> **Palette:** Dark graphite background `#1e1e2e`, accent steelblue `#4a9eca`, coral `#e06c75`, muted green `#98c379`, soft white `#abb2bf`.
> Generator: [Perchance AI Image Generator](https://perchance.org/ai-text-to-image-generator)

---

## 1 — `regression-loss-landscape.png`

**Purpose:** Visual aid for Part 1 / Part 2 — show what "minimising MSE" means geometrically before the code proves it numerically.

**Perchance prompt:**
```
dark graphite background, 3D bowl-shaped loss landscape rendered as a wireframe mesh in deep navy blue with glowing steelblue contour lines, a single glowing coral dot labelled "current weights" rolling down the gradient toward the bottom of the bowl labelled "minimum MSE", subtle white arrows indicating gradient direction, minimalist data-science illustration style, no text other than labels, ultra-clean, dark mode
```

**Alt text:** "A 3D bowl-shaped surface representing the MSE loss landscape. A glowing dot rolls down the gradient toward the global minimum."

**Placement:** After the Part 2 gradient descent markdown header cell (Cell 8), before the manual gradient descent code cell.

**Target size:** 800 × 500 px

---

## 2 — `overfitting-train-val-curves.png`

**Purpose:** Visual aid for Part 6 — illustrate conceptually what overfitting looks like before the learning curve code confirms it on real data.

**Perchance prompt:**
```
dark graphite background, two smooth curves on a clean 2D chart: one steelblue line labelled "Train MAE" descending steeply and flattening near zero, one coral line labelled "Val MAE" descending then rising back up as training samples decrease, a dashed green horizontal line labelled "target $40k MAE", x-axis labelled "Training samples (log scale)", y-axis labelled "MAE ($k)", minimal dark-mode data-science diagram, no decorative elements, high contrast labels
```

**Alt text:** "Learning curves showing training MAE decreasing smoothly while validation MAE first decreases then rises as training data shrinks — the overfitting regime."

**Placement:** Before Cell 18 (Part 6 overfitting markdown header), as conceptual preview before the actual learning curve is plotted by Cell 20.

**Target size:** 900 × 500 px

---

## 3 — `lasso-ridge-coefficients.png`

**Purpose:** Visual aid for Part 4 — show conceptually how Lasso zeroes out coefficients while Ridge only shrinks them, before the code grid confirms it for California Housing features.

**Perchance prompt:**
```
dark graphite background, two horizontal bar charts side by side: left chart "Ridge" showing 8 bars of varying steelblue lengths all non-zero, right chart "Lasso" showing same 8 bars but 3 of them are exactly zero (shown as empty/absent), bars labelled with feature names MedInc HouseAge AveRooms AveBedrms Population AveOccup Latitude Longitude, coral highlight on the zero-weight bars in Lasso chart, clean dark-mode data-science infographic, minimalist, no background grid, white axis labels
```

**Alt text:** "Side-by-side bar charts comparing Ridge (all features kept, weights shrunk) vs Lasso (3 features zeroed out, sparse solution) coefficient magnitudes for the 8 California Housing features."

**Placement:** After the Part 4 markdown header cell (Cell 13), before the Ridge vs Lasso code cell (Cell 14).

**Target size:** 1000 × 500 px

---

## Notes

- All images should be saved to `learning/genai-prerequisites/01-ml-basics/` alongside the notebook.
- Images are **supplemental** — all claims in the notebook are proven by live code output. Images serve as visual previews of what the code will demonstrate.
- If images are not available, the notebook runs and teaches correctly without them (no `![]()` embeds in the notebook itself; images are referenced in this plan only).
