# Image Generation Plan — convolutional-neural-networks.ipynb

Planned static images to embed in the notebook (referenced in markdown cells).
Generated offline with Stable Diffusion or DALL-E; committed to `images/` subfolder.

---

## Palette

| Role | Hex | Use |
|---|---|---|
| Background | `#1a1a2e` | Panel fill, page background |
| Surface | `#16213e` | Card / axis background |
| Accent dark | `#0f3460` | Grid lines, borders |
| Filter / kernel | `#4ecdc4` | Teal — sliding filter highlight |
| Output / feature map | `#f5a623` | Amber — output activations |
| Skip connection | `#e07b54` | Coral-red — skip / gradient arrows |
| Text primary | `#f5f5f5` | Titles, labels |
| Text secondary | `#a8a8b3` | Axis ticks, annotations |

---

## Image 1 — `convolution-filter-operation.png`

**Used in:** Part 1 markdown header (Cell 5)
**Size:** 800 × 500 px

### Concept
A 5×5 input grid (representing a small patch of the MNIST digit "3") on the left. A 3×3
teal filter kernel in the centre, shown sliding over the input with a dotted boundary
highlighting the current receptive window. On the right, the resulting 3×3 output
feature map in amber. Arithmetic annotations show one example computation:
`(-1)×0 + (-2)×1 + ... = 4.2` with an arrow pointing to the highlighted output cell.

### Stable Diffusion prompt
```
technical diagram of a 2D convolution operation, dark graphite background #1a1a2e,
left panel: 5x5 grid of small squares representing an input image patch with numeric
values in ivory #f5f5f5 text, center: 3x3 teal #4ecdc4 highlighted filter kernel
with weight values, dotted teal border showing sliding window on input,
right panel: 3x3 amber #f5a623 output feature map grid with computed values,
a curved arrow with arithmetic annotation showing dot-product calculation,
clean flat technical illustration, minimalist scientific diagram style,
dark background, high contrast, ultra-sharp, no photorealism
```

### DALL-E prompt
```
Minimalist technical diagram on dark graphite (#1a1a2e) background. Left: a 5×5 grid
labeled "Input" with small ivory numbers in each cell. Centre: a 3×3 teal (#4ecdc4)
filter/kernel grid with weight numbers, overlaid on the input with a dashed teal border
showing the current position. Right: a 3×3 amber (#f5a623) grid labeled "Feature Map"
with output values. A labeled arrow shows "dot product + bias = output value" connecting
filter to output. White sans-serif labels. Flat, clean, scientific data visualization
style. No gradients, no photorealism.
```

---

## Image 2 — `feature-maps-by-layer.png`

**Used in:** Part 2 markdown header (Cell 8)
**Size:** 900 × 350 px

### Concept
Three panels in a horizontal row, connected by right-pointing arrows:
1. **Panel 1 (left)** — Raw MNIST digit "3" in grayscale, labelled "Input 1×28×28"
2. **Panel 2 (centre)** — A 2×4 grid of 8 feature maps after the first conv layer,
   labelled "After Conv1 + ReLU (16 channels, 28×28)". Each mini-map shows a different
   learned edge detector: some respond to horizontal edges, some to diagonals, some to
   corners.
3. **Panel 3 (right)** — A 2×4 grid of 8 feature maps after conv2+maxpool, labelled
   "After Conv2 + Pool (8 channels, 7×7)". The maps are smaller and more abstract.

Dark graphite palette throughout; feature maps use `viridis` colormap.

### Stable Diffusion prompt
```
technical three-panel diagram, dark graphite background #1a1a2e,
left panel: small grayscale handwritten digit 3 labeled "Input 1×28×28",
center panel: 2x4 grid of eight small feature map thumbnails showing edge patterns
labeled "Conv1 feature maps", each thumbnail uses viridis colormap (blue-green-yellow),
right panel: 2x4 grid of eight smaller blurry abstract feature map thumbnails
labeled "Conv2 feature maps", right-pointing white arrows connecting panels,
ivory white sans-serif labels, minimalist flat scientific diagram, ultra-sharp
```

### DALL-E prompt
```
Clean three-panel technical diagram on dark graphite (#1a1a2e). Panel 1 (left): a 28×28
grayscale handwritten digit "3", labeled "Input 1×28×28". Arrow pointing right. Panel 2
(center): a 2×4 grid of 8 small heatmap thumbnails using viridis colormap, labeled
"After Conv1 + ReLU". Arrow pointing right. Panel 3 (right): a 2×4 grid of 8 even
smaller, blurrier heatmap thumbnails using viridis colormap, labeled "After Conv2 + Pool
(7×7)". White sans-serif labels. Minimalist, flat, scientific visualization. No
gradients, no photorealism.
```

---

## Image 3 — `resnet-skip-connection.png`

**Used in:** Part 4 markdown header (Cell 12)
**Size:** 700 × 450 px

### Concept
Two side-by-side block diagrams comparing a plain residual block (left) vs a ResNet
residual block (right).

**Left block (plain):**
`x` → [Conv 3×3] → [BN] → [ReLU] → [Conv 3×3] → [BN] → `F(x)` → [ReLU] → output
Gradient arrow (thin, fading red) flows backward, shrinking at each layer.

**Right block (ResNet):**
Same forward path as left, but a teal bypass arrow curves around the entire stack,
labelled "+x (identity shortcut)". At the bottom, the teal bypass merges into a coral
addition node `⊕` before the final ReLU. A bold coral gradient arrow flows backward
through the addition node unimpeded, labelled "gradient = 1 + ∂F/∂x always ≥ 1".

### Stable Diffusion prompt
```
technical block diagram comparison, dark graphite background #1a1a2e,
left side: plain neural network residual block with boxes labeled Conv3x3, BN, ReLU
connected top-to-bottom, thin fading red dashed backward arrow showing vanishing gradient,
right side: ResNet residual block with same forward path boxes but with a thick teal
curved bypass arrow going around the entire stack labeled "+x", a coral addition circle
symbol at the bottom merging the bypass, bold coral backward gradient arrow labeled
gradient formula, ivory white labels and connections, clean flat technical diagram style,
minimalist, ultra-sharp
```

### DALL-E prompt
```
Split technical diagram on dark graphite (#1a1a2e). Left side titled "Plain Block": a
vertical stack of rounded boxes labeled Conv3×3 → BN → ReLU → Conv3×3 → BN → Output.
A thin fading red dashed arrow runs backward (bottom to top), labeled "vanishing
gradient". Right side titled "Residual Block (ResNet)": same vertical stack, but a thick
teal (#4ecdc4) curved arrow bypasses the entire stack from input to a coral (#e07b54)
addition circle ⊕ at the bottom, labeled "+x". A bold coral backward arrow runs from
output to input, labeled "∂L/∂x = 1 + ∂F/∂x". White sans-serif labels. Flat, clean,
minimalist scientific diagram. No gradients, no photorealism.
```
