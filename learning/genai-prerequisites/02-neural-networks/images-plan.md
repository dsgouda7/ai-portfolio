# Image Generation Plan — neural-networks-and-backprop.ipynb

Planned static images to embed in the notebook (referenced in markdown cells).
Generated offline with Stable Diffusion or DALL-E; committed to `images/` subfolder.

---

## Palette

| Role | Hex | Use |
|---|---|---|
| Background | `#1a1a2e` | Panel fill, page background |
| Surface | `#16213e` | Card / axis background |
| Accent dark | `#0f3460` | Grid lines, borders |
| Class-0 (negative) | `#45b7d1` | Steel-blue points, region fill |
| Class-1 (positive) | `#e07b54` | Coral points, region fill |
| Highlight / feature | `#4ecdc4` | Teal — hidden neurons, arrows |
| Text primary | `#f5f5f5` | Titles, labels |
| Text secondary | `#a8a8b3` | Axis ticks, annotations |

---

## Image 1 — `xor-not-linearly-separable.png`

**Used in:** Cell 6 output replacement / Part 1 markdown header
**Size:** 600 × 500 px

### Concept
Four XOR points on a 2D plane. Two steel-blue circles at (0,0) and (1,1) (class 0); two
coral circles at (0,1) and (1,0) (class 1). A dashed gray diagonal line ($x_1 + x_2 = 1$)
represents the best possible linear separator — which still misclassifies two points.

### Stable Diffusion prompt
```
minimalist 2D scatter plot, dark graphite background #1a1a2e, four large circular data points,
two steel-blue circles at bottom-left and top-right corners labeled 0, two coral-orange circles
at top-left and bottom-right corners labeled 1, a dashed light-gray diagonal line cutting through
the middle that fails to separate the groups, thin subtle grid lines #0f3460, axis labels in #f5f5f5,
clean data visualization aesthetic, no gradients, flat design, scientific illustration style,
high contrast, ultra-sharp
```

### DALL-E prompt
```
Clean 2D mathematical scatter plot on dark graphite (#1a1a2e) background. Four large circles:
two steel-blue (#45b7d1) circles at coordinates (0,0) and (1,1) labeled "0"; two coral
(#e07b54) circles at (0,1) and (1,0) labeled "1". A dashed gray diagonal line crosses from
lower-left to upper-right, clearly failing to separate blue from coral. White sans-serif axis
labels "x1 (income)" and "x2 (crime)". Minimalist, flat, scientific data visualization.
```

---

## Image 2 — `neural-network-forward-pass.png`

**Used in:** Cell 7 (Part 2 markdown) — architecture diagram
**Size:** 700 × 420 px

### Concept
Architecture diagram of the 2→2→1 XORNet. Left column: two input circles (x1, x2) in teal.
Middle column: two hidden neuron circles (h1, h2) with "ReLU" label in warm orange. Right
column: one output circle (ŷ) in coral with "sigmoid" label. Weighted arrows between each
layer labelled W₁ (layer 1) and W₂ (layer 2). Bias nodes shown as small b₁, b₂ circles.

### Stable Diffusion prompt
```
neural network architecture diagram, dark graphite background #1a1a2e, three vertical columns
of nodes, left column two teal #4ecdc4 glowing circles labeled x1 x2, middle column two warm
orange circles labeled h1 h2 with ReLU text below, right column one coral #e07b54 circle labeled
y-hat with sigmoid text below, thin white arrows connecting all nodes between columns, matrix
labels W1 W2 above the connection groups, clean geometric technical illustration, no photorealism,
minimalist infographic style, high contrast white text
```

### DALL-E prompt
```
Technical neural network diagram on dark graphite (#1a1a2e) background. Three vertical columns:
LEFT — two teal (#4ecdc4) circles labeled "x₁" and "x₂" (inputs). MIDDLE — two orange circles
labeled "h₁" and "h₂" with small "ReLU" text underneath (hidden layer). RIGHT — one coral
(#e07b54) circle labeled "ŷ" with "sigmoid" text (output). Thin white arrows connect every
left circle to every middle circle (labeled "W₁"), and every middle circle to the right circle
(labeled "W₂"). Clean minimalist technical diagram, white labels, no background texture.
```

---

## Image 3 — `depth-vs-width-decision-boundary.png`

**Used in:** Cell 14 output replacement / Part 4 comparison
**Size:** 1100 × 480 px (side-by-side panels)

### Concept
Two-panel contour plot. Both panels show the same spiral validation set (150 points, two
interleaved spiral arms). Left panel: wide network (2→256→1) — blocky, angular decision
boundary that partially follows the spiral. Right panel: deep network (2→8→8→8→1) — smooth,
spiral-following decision boundary with higher accuracy. Background fill in semi-transparent
steel-blue / coral for each class region.

### Stable Diffusion prompt
```
side-by-side scientific visualization, dark graphite background #1a1a2e, two square plot panels,
both showing interleaved spiral data points in steel-blue and coral colors, left panel titled
'Wide 256 neurons' has irregular blocky decision boundary that misses the spiral curve, right
panel titled 'Deep 4 layers' has smooth curved boundary perfectly following the spiral shape,
background region fills in semi-transparent teal and coral, white axis labels, flat contour plot
aesthetic, data science visualization style, ultra sharp, no photorealism
```

### DALL-E prompt
```
Two-panel side-by-side data visualization on dark graphite (#1a1a2e) background. Both panels
show a 2D spiral dataset: two interleaved spiral arms, one steel-blue (#45b7d1) and one coral
(#e07b54), each arm containing ~75 small circular data points. LEFT panel labeled "Wide
(256 neurons)" shows an irregular, angular decision boundary — the colored background regions
don't follow the spiral well. RIGHT panel labeled "Deep (4 layers)" shows a smooth, tight
decision boundary that closely follows the spiral shape. Clean scientific visualization,
minimal axes, white title text.
```

---

## Generation notes

- All images should use the dark graphite palette above — do not use white backgrounds
- Render at 2× resolution then downscale for anti-aliasing
- Export as PNG with transparency disabled (dark background is part of the design)
- Store at `learning/genai-prerequisites/02-neural-networks/images/`
- Reference in notebook markdown as `![alt text](images/filename.png)`
