# Ch.11 — Model Interpretability: Grad-CAM, Saliency, and Integrated Gradients

> **The story.** In 2013, Matthew Zeiler and Rob Fergus published "Visualizing and Understanding Convolutional Networks." They used deconvnets to project feature-map activations back to input pixels — for the first time, humans could see that AlexNet's layer 1 detected edges, layer 2 detected textures, layer 5 responded to faces and text. Ramprasaath Selvaraju et al. followed in 2017 with Grad-CAM: "Gradient-weighted Class Activation Mapping" gave any CNN a one-sentence answer to "where did you look?" without modifying the architecture. These two papers transformed interpretability from academic curiosity into an enterprise deployment gate.
>
> **Where you are.** ProductionCV's shelf-monitoring system achieves mAP 85.2% after Ch.10's pruning pass. The ops team wants to roll out to 50 stores. The lead buyer flags three misdetections from the pilot week — all "shelf gap" events the model missed. Your answer to "why did the model miss that product?" is silence. This chapter gives you four tools to end that silence: saliency maps, Grad-CAM, Guided Grad-CAM, and Integrated Gradients — all in PyTorch, all runnable on the 6.8 MB compressed model.
>
> **Notation.** $A^k \in \mathbb{R}^{H \times W}$ — feature map of channel $k$ in the last convolutional layer; $y^c$ — class score (logit) for class $c$ before softmax; $\frac{\partial y^c}{\partial A^k_{ij}}$ — gradient of class score w.r.t. pixel $(i,j)$ of feature map $k$; $\alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}$ — Grad-CAM importance weight for channel $k$; $L^c \in \mathbb{R}^{H \times W}$ — Grad-CAM localization map; $Z = H \times W$ — number of spatial positions (normalization constant); $x$ — input image tensor; $x'$ — baseline image (black image or mean image) for Integrated Gradients.

---

## 0 · The Challenge — Where We Are

ProductionCV hit mAP 85.2% in the week-two production pilot. Three misdetections were all "shelf gap" events — empty shelf positions the model should have flagged. The ops team pulled the three images and asked: *which part of the shelf was the model looking at?* You have no answer. No accuracy metric tells you this. A model that achieves 85% mAP can still be looking at the shelf label instead of the gap, and it will fail predictably for an explainable reason that no amount of threshold tuning will fix.

**What this chapter adds to the grand challenge:**

| Constraint | Status before Ch.11 | Status after Ch.11 |
|---|---|---|
| mAP >= 85% | ACHIEVED (85.2%) | Unchanged |
| IoU >= 70% | ACHIEVED (71.2%) | Unchanged |
| Latency < 50ms | ACHIEVED (35ms) | Unchanged |
| Model size < 100 MB | ACHIEVED (6.8 MB) | Unchanged |
| Data efficiency < 1000 labels | ACHIEVED (850 labels) | Unchanged |
| **Auditability** | BLOCKED | **ACHIEVED — explain any prediction** |

Auditability is not a soft goal. Enterprise procurement at three of the five pilot retailers explicitly requires it. The system ships when you can answer "why?" for any misdetection.

---

## 1 · Core Idea

A **saliency map** answers: *which input pixels most influenced this prediction?* It does so by measuring the gradient of the class score with respect to the input image — pixels where a small change causes a large change in the score are "important."

A **Grad-CAM map** answers a different, more actionable question: *which spatial regions of the final convolutional layer drove this prediction?* It aggregates gradients over the feature channels rather than over individual pixels, producing a coarse but class-discriminative spatial heatmap.

These are not the same question. The pixel-level saliency map shows local sensitivity; the Grad-CAM map shows where in the image the model's high-level spatial reasoning fired. For the shelf-gap problem, Grad-CAM is the right tool — you want to know "was the model looking at the gap region or the label region?", not "which pixel changed the score most?"

**Two questions, two tools:**

| Question | Tool | Resolution | Computation |
|---|---|---|---|
| Which pixels are most sensitive? | Vanilla Backprop saliency | Input resolution (e.g. 224×224) | One backward pass from input |
| Which spatial regions of the last conv layer drove this prediction? | Grad-CAM | Feature map resolution (e.g. 7×7), upsampled | One backward pass to last conv layer |
| Both: fine-grained spatial + class-discriminative | Guided Grad-CAM | Input resolution | Two passes combined |
| Attribution with completeness guarantee | Integrated Gradients | Input resolution | 50 backward passes |

---

## 2 · Saliency Maps — Vanilla Backprop

The simplest attribution method. Enable gradient tracking on the input image, run a forward pass, call `.backward()` on the target class score, then read the gradient at the input.

$$\text{Saliency}(i, j) = \left| \frac{\partial y^c}{\partial x_{ij}} \right|$$

The absolute value is taken because we care about sensitivity magnitude, not direction.

### 2.1 PyTorch Implementation

```python
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def vanilla_saliency(model: torch.nn.Module,
                     image: torch.Tensor,
                     target_class: int) -> torch.Tensor:
    """
    Compute vanilla backprop saliency map.

    Args:
        model: Trained PyTorch model (eval mode expected)
        image: Input tensor of shape (1, C, H, W), requires_grad will be set
        target_class: Class index for which to compute saliency

    Returns:
        saliency: (H, W) tensor of absolute gradient values
    """
    model.eval()
    image = image.clone().requires_grad_(True)

    # Forward pass
    output = model(image)                        # (1, num_classes)
    class_score = output[0, target_class]        # scalar

    # Backward pass from the target class score
    model.zero_grad()
    class_score.backward()

    # Gradient w.r.t. input: take max over RGB channels for visualization
    saliency = image.grad.data.abs()             # (1, C, H, W)
    saliency, _ = saliency.max(dim=1)            # (1, H, W) — max over channels
    return saliency.squeeze()                    # (H, W)


def visualize_saliency(saliency: torch.Tensor) -> None:
    """Normalize and display saliency map."""
    import matplotlib.pyplot as plt
    s = saliency.cpu().numpy()
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    plt.figure(figsize=(5, 5))
    plt.imshow(s, cmap='hot')
    plt.colorbar()
    plt.title('Vanilla Saliency Map')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
```

### 2.2 What Saliency Maps Reveal (and Don't)

Saliency maps are sensitive but noisy. They show *where a small pixel change would hurt the prediction most*, which is not always *where the model is looking*. A model that has memorised a background artifact will show high saliency on that artifact, correctly revealing the bug — but saliency maps also fire on high-frequency edges and texture boundaries even when those are not semantically meaningful.

**Limitation to keep in mind**: saliency is the *derivative* at one point in input space. It answers "what perturbation here would move the score?" not "what feature here caused the score?" For causal attribution, Grad-CAM and Integrated Gradients are better tools.

---

## 3 · Grad-CAM

Grad-CAM (Selvaraju et al., 2017) uses the gradients flowing into the *last convolutional layer* to produce a coarse but class-discriminative localization map. It answers: **which spatial regions of the final feature map mattered most for predicting class $c$?**

### 3.1 The Math

**Step 1 — Compute the importance weight for each channel:**

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\partial y^c}{\partial A^k_{ij}}$$

Global average pool the gradient of class score $y^c$ over all spatial positions $(i,j)$ of feature map $k$. This gives a single scalar $\alpha_k^c$ per channel: *how much does channel $k$ contribute to predicting class $c$?*

**Verbal gloss:** The gradient $\frac{\partial y^c}{\partial A^k_{ij}}$ tells us "if channel $k$ at position $(i,j)$ increased, how much would the class-$c$ score change?" Averaging over all positions gives the *global* importance of that entire channel for that class — ignoring where in the spatial map it fires.

**Step 2 — Weight the feature maps and apply ReLU:**

$$L^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$

Multiply each feature map $A^k$ by its importance weight $\alpha_k^c$, sum across all channels, then apply ReLU. The ReLU keeps only *positive* contributions — regions that increase the class score. Negative regions (that suppress the class) are zeroed out because we want to highlight *evidence for* the class, not evidence against it.

**Why ReLU?** Without it, $L^c$ would highlight both "looks like a gap" regions (positive) and "looks like a product" regions (negative). The ReLU isolates the localization signal for the target class.

**Step 3 — Upsample to input resolution:**

The feature map $A^k$ has spatial resolution $H' \times W'$ (e.g. $7 \times 7$ for a ResNet-50's last block). $L^c$ has the same spatial resolution. Bicubic-upsample $L^c$ to the original image size ($H \times W$, e.g. $224 \times 224$) and overlay as a heatmap.

### 3.2 PyTorch Implementation with register_hook

```python
import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """
    Grad-CAM implementation using forward and backward hooks.
    Works with any PyTorch model without modifying its architecture.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: Trained PyTorch model
            target_layer: The convolutional layer to attach hooks to.
                          Typically the last conv layer before the classifier.
                          Example: model.layer4[-1].conv2 for ResNet-50
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks to the target layer."""

        def forward_hook(module, input, output):
            # Store the forward activation: shape (batch, C, H', W')
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Store the gradient of the loss w.r.t. the layer's output
            # grad_output[0] has shape (batch, C, H', W')
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self,
                 image: torch.Tensor,
                 target_class: int | None = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            image: Input tensor (1, C, H, W)
            target_class: Class to explain. If None, uses the predicted class.

        Returns:
            cam: (H, W) numpy array with values in [0, 1], ready to overlay
        """
        self.model.eval()

        # Forward pass — activations captured by hook
        output = self.model(image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass — gradients captured by hook
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # alpha_k^c: global-average-pool the gradients over spatial dimensions
        # gradients shape: (1, C, H', W')
        # mean over (H', W') gives (1, C, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps: sum_k( alpha_k^c * A^k )
        # activations shape: (1, C, H', W')
        # cam shape: (1, 1, H', W') after weighted sum and keeping dims
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')

        # Apply ReLU: keep only positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Upsample to input image resolution
        h, w = image.shape[-2], image.shape[-1]
        cam = F.interpolate(cam, size=(h, w), mode='bicubic', align_corners=False)

        return cam.squeeze().cpu().numpy()  # (H, W)


def overlay_cam_on_image(image_np: np.ndarray,
                          cam: np.ndarray,
                          alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        image_np: Original image as (H, W, 3) uint8 array
        cam: Grad-CAM map as (H, W) float array in [0, 1]
        alpha: Heatmap opacity (0 = image only, 1 = heatmap only)

    Returns:
        overlaid: (H, W, 3) uint8 array
    """
    import cv2
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlaid = (1 - alpha) * image_np + alpha * heatmap
    return np.clip(overlaid, 0, 255).astype(np.uint8)
```

### 3.3 Applying Grad-CAM to ProductionCV

```python
import torchvision.models as models

# Load the compressed 6.8 MB ProductionCV model
model = models.mobilenet_v2(pretrained=False)
model.load_state_dict(torch.load('productioncv_pruned.pth'))
model.eval()

# Target layer: last conv layer in MobileNetV2 (features[-1][0])
# This is the 1x1 pointwise conv that produces the 1280-channel output
target_layer = model.features[-1][0]

grad_cam = GradCAM(model=model, target_layer=target_layer)

# Load the problematic shelf image
from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

shelf_img = Image.open('shelf_14C_misdetection.jpg')
input_tensor = transform(shelf_img).unsqueeze(0)  # (1, 3, 224, 224)

# CLASS_NAMES[0] = 'shelf_gap', CLASS_NAMES[1] = 'product_present'
cam = grad_cam(input_tensor, target_class=0)  # explain the 'shelf_gap' class

import numpy as np
import matplotlib.pyplot as plt

img_np = np.array(shelf_img.resize((224, 224)))
overlaid = overlay_cam_on_image(img_np, cam, alpha=0.4)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_np)
axes[0].set_title('Original (shelf 14C)')
axes[1].imshow(cam, cmap='jet')
axes[1].set_title('Grad-CAM heatmap')
axes[2].imshow(overlaid)
axes[2].set_title('Overlay (alpha=0.4)')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig('shelf_14C_grad_cam.png', dpi=150)
plt.show()
```

---

## 4 · Guided Backpropagation and Guided Grad-CAM

### 4.1 Guided Backpropagation

Vanilla backprop passes negative gradients through ReLU activations unchanged. Guided backpropagation modifies the ReLU backward pass: in addition to zeroing gradients where the activation was negative (standard ReLU behavior), it *also* zeros gradients that are negative even if the activation was positive.

**Standard ReLU backward:**
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \hat{x}} \cdot \mathbf{1}[x > 0]$$

**Guided backprop backward (additional gate):**
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \hat{x}} \cdot \mathbf{1}[x > 0] \cdot \mathbf{1}\left[\frac{\partial L}{\partial \hat{x}} > 0\right]$$

The extra gate $\mathbf{1}[\partial L / \partial \hat{x} > 0]$ zeros out negative-gradient flows. This produces sharper, less noisy visualizations — essentially restricting the attribution to features that *increase* the class score.

```python
class GuidedBackprop:
    """
    Guided Backpropagation: modify ReLU backward pass to only pass
    positive gradients through positive activations.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                self._hooks.append(
                    module.register_full_backward_hook(self._guided_relu_hook)
                )

    @staticmethod
    def _guided_relu_hook(module, grad_input, grad_output):
        # Only pass gradients that are: (1) positive AND (2) from positive activations
        return (torch.clamp(grad_output[0], min=0.0),)

    def __call__(self,
                 image: torch.Tensor,
                 target_class: int | None = None) -> torch.Tensor:
        """Returns the guided-backprop saliency map (H, W)."""
        self.model.eval()
        image = image.clone().requires_grad_(True)
        output = self.model(image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()
        guided_saliency = image.grad.data
        guided_saliency, _ = guided_saliency.abs().max(dim=1)
        return guided_saliency.squeeze()

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
```

### 4.2 Guided Grad-CAM

Combine Guided Backprop and Grad-CAM via pointwise multiplication to get the best of both: the spatial localization of Grad-CAM (class-discriminative) and the fine-grained pixel resolution of Guided Backprop.

```python
def guided_grad_cam(grad_cam_map: np.ndarray,
                    guided_bp_map: torch.Tensor) -> np.ndarray:
    """
    Elementwise product of Grad-CAM and Guided Backprop maps.

    Args:
        grad_cam_map: (H, W) Grad-CAM output, already upsampled
        guided_bp_map: (H, W) Guided Backprop output

    Returns:
        ggcam: (H, W) Guided Grad-CAM, normalized to [0, 1]
    """
    ggcam = grad_cam_map * guided_bp_map.cpu().numpy()
    ggcam = (ggcam - ggcam.min()) / (ggcam.max() - ggcam.min() + 1e-8)
    return ggcam
```

**When to use which:**
- Grad-CAM: best for "which part of the image?" — coarse but robust and class-discriminative
- Guided Backprop: best for "which fine-grained textures?" — sharp but can be class-agnostic
- Guided Grad-CAM: best for "which fine-grained features in the relevant region?" — most informative for production auditing

---

## 5 · Integrated Gradients

Gradient-based methods like saliency maps only measure *local* sensitivity — they can give zero attribution to a feature that strongly determined the prediction, if the prediction is in a saturated sigmoid or ReLU-dead zone. Integrated Gradients (Sundararajan, Taly, Yan, 2017) fixes this by accumulating gradients along a linear path from a baseline $x'$ (black image or mean-image) to the input $x$.

### 5.1 The Math

$$\text{IG}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i}\, d\alpha$$

**Verbal gloss:** For each input feature $x_i$ (pixel), compute: "by how much did the prediction change as we linearly interpolated from the baseline $x'$ (no information) to the actual input $x$, weighted by how sensitive the model was to $x_i$ throughout that interpolation?" The integral is approximated by a Riemann sum over $m = 50$ steps.

**Completeness axiom:** $\sum_i \text{IG}_i(x) = F(x) - F(x')$

The sum of all attributions equals the prediction gap between the actual input and the baseline. This is a formal guarantee that vanilla saliency lacks — the attributions *add up to the prediction difference*.

### 5.2 PyTorch Implementation

```python
def integrated_gradients(model: torch.nn.Module,
                          image: torch.Tensor,
                          target_class: int,
                          baseline: torch.Tensor | None = None,
                          steps: int = 50) -> torch.Tensor:
    """
    Compute Integrated Gradients for a single image.

    Args:
        model: Trained PyTorch model (eval mode)
        image: (1, C, H, W) input tensor
        target_class: Class index to explain
        baseline: (1, C, H, W) reference image. Defaults to zero (black image).
        steps: Number of Riemann integration steps (50 is sufficient for most tasks)

    Returns:
        attributions: (C, H, W) tensor — per-pixel, per-channel attribution
    """
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(image)

    # Build interpolation path: x' + alpha * (x - x') for alpha in [0, 1]
    alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1)
    interpolated = baseline + alphas * (image - baseline)  # (steps, C, H, W)
    interpolated.requires_grad_(True)

    # Forward pass on all steps simultaneously
    output = model(interpolated)                          # (steps, num_classes)
    class_scores = output[:, target_class].sum()          # scalar sum for backward

    # Backward pass: dF/dx at each interpolated point
    model.zero_grad()
    class_scores.backward()
    grads = interpolated.grad.data                         # (steps, C, H, W)

    # Riemann sum approximation of the integral
    avg_grads = grads.mean(dim=0)                         # (C, H, W)
    attributions = (image.squeeze() - baseline.squeeze()) * avg_grads

    return attributions  # (C, H, W)


def summarize_attributions(attributions: torch.Tensor) -> torch.Tensor:
    """Sum absolute attributions over channels for visualization: (H, W)."""
    return attributions.abs().sum(dim=0)
```

### 5.3 Verifying Completeness

```python
# Quick sanity check: attributions should sum to F(x) - F(x')
model.eval()
with torch.no_grad():
    score_input    = model(image)[0, target_class].item()
    score_baseline = model(baseline)[0, target_class].item()

attrs = integrated_gradients(model, image, target_class, baseline, steps=50)
attr_sum = attrs.sum().item()

print(f"F(x)          = {score_input:.4f}")
print(f"F(x')         = {score_baseline:.4f}")
print(f"F(x) - F(x')  = {score_input - score_baseline:.4f}")
print(f"sum(IG attrs) = {attr_sum:.4f}")
print(f"Completeness error: {abs(attr_sum - (score_input - score_baseline)):.6f}")
# Expected: Completeness error < 0.001 for steps=50
```

---

## 6 · SHAP for Image Models

SHAP (SHapley Additive exPlanations) provides attributions with theoretical guarantees from cooperative game theory — Shapley values are the unique attribution method satisfying *efficiency, symmetry, dummy,* and *linearity* axioms simultaneously. For images, two implementations exist in the `shap` library.

### 6.1 GradientExplainer

Computes expected gradients over a background dataset — a continuous relaxation of SHAP that approximates Shapley values using gradient information.

```python
import shap
import torch

# Load background images (a small sample of training data)
background = torch.stack([transform(img) for img in background_images[:50]])

explainer = shap.GradientExplainer(model, background)

# Compute SHAP values for the target image
shap_values = explainer.shap_values(input_tensor)
# shap_values: list of (1, C, H, W) arrays, one per class

# Visualize for the target class
shap.image_plot(shap_values[target_class], -input_tensor.numpy())
```

### 6.2 Grad-CAM vs SHAP: When to Use Which

| Criterion | Grad-CAM | SHAP GradientExplainer |
|---|---|---|
| Speed | Fast (one backward pass) | Slow (50+ passes over background) |
| Spatial resolution | Coarse (feature map resolution) | Full resolution |
| Class discriminativity | High — specific to target class | High — Shapley axioms guarantee it |
| Theoretical guarantees | None (heuristic) | Efficiency, symmetry, linearity |
| Requires background data | No | Yes (small sample) |
| Production debugging | Primary choice | When Grad-CAM is ambiguous |

**Practical rule:** Use Grad-CAM for real-time production debugging (fast, interpretable, class-specific). Use SHAP when you need to quantify attribution magnitude for model audits or regulatory reports.

---

## 7 · Failure Modes

### 7.1 Gradient Saturation

Near-saturated ReLU activations have near-zero gradients — the saliency map shows nothing despite the feature being strongly causal. A model confident in its prediction will often show *low* gradients at the relevant features because the sigmoid is flat at high confidence.

**Symptom:** Saliency map is near-uniform or shows high values everywhere.

**Fix:** Use Integrated Gradients instead — it accumulates gradients over the entire interpolation path, not just at the final prediction.

### 7.2 Confirmation Bias in Interpretations

Heatmaps can look plausible but be meaningless. A model that predicts "cat" because of the background carpet can still show a Grad-CAM heatmap that "looks like" it's focusing on the cat, because some cat-texture features overlap with carpet texture.

**Test:** Apply the same attribution method to an adversarial example (same prediction, different input). If the heatmap looks completely different, your interpretation of the original map was not robust.

**Sanity check:** Replace the input image with uniform noise. Run Grad-CAM. If it still shows a "focused" heatmap, the method is responding to model artifacts, not image content.

### 7.3 Adversarial Sensitivity

Grad-CAM can show a completely different heatmap for an adversarially perturbed image even when the model outputs the same class with the same confidence. The perturbation shifts gradients without changing the final prediction — revealing that gradient-based methods are sensitive to the specific path through the computation graph, not just the output.

**Practical implication:** Do not use Grad-CAM as the only evidence for "the model is looking at the right thing." Combine with a behavioral test (does the model's prediction change when you mask the high-attribution region?) for robust auditing.

### 7.4 Layer Selection Sensitivity

Grad-CAM is sensitive to which layer you attach to. The last conv layer before the classifier is the standard choice, but:
- Too-early layers: heatmaps are too generic (edge detectors, not object detectors)
- The last layer: sometimes produces coarser maps than the second-to-last

For MobileNetV2's compressed architecture, `model.features[-1][0]` (the last 1×1 conv) is the correct target. For ResNet-50, use the last layer of `layer4`.

---

## 8 · ProductionCV Debugging Session

The three misdetections from the shelf-gap pilot are investigated using Grad-CAM.

### 8.1 Finding the Root Cause

```python
# Reproduce the misdetection analysis

# Image: shelf_14C — the model predicted "product_present" but should have
# predicted "shelf_gap"
cam_gap    = grad_cam(input_tensor, target_class=CLASS_NAMES.index('shelf_gap'))
cam_prod   = grad_cam(input_tensor, target_class=CLASS_NAMES.index('product_present'))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_np)
axes[0].set_title('Original shelf 14C')
axes[1].imshow(overlay_cam_on_image(img_np, cam_gap),)
axes[1].set_title('Grad-CAM: shelf_gap class')
axes[2].imshow(overlay_cam_on_image(img_np, cam_prod))
axes[2].set_title('Grad-CAM: product_present class')
plt.tight_layout()
```

**Finding:** The `shelf_gap` Grad-CAM concentrates on the shelf label (bottom-left corner of the image), not on the empty space in the center. The `product_present` Grad-CAM concentrates on the same shelf label.

**Root cause:** Both classes are activating on the shelf label feature. In the training dataset, shelf gaps consistently had faded or missing labels — so the model learned *label condition* as a proxy for *gap presence*. This is a dataset bias, not an architecture problem.

**Fix:** Collect 200 additional training examples with:
1. Shelf gaps with bright, undamaged labels
2. Stocked shelves with faded labels

Retrain — do NOT adjust the model architecture or loss function. The model is working correctly; the training data is biased.

**Lesson:** Grad-CAM on three images revealed a root cause that would have required thousands of systematic test images to detect via accuracy metrics alone.

### 8.2 Verifying the Fix

After retraining with the balanced dataset, re-run Grad-CAM on the same three misdetected images:

```python
# Post-fix model
model_fixed.load_state_dict(torch.load('productioncv_v2.pth'))
grad_cam_fixed = GradCAM(model=model_fixed, target_layer=model_fixed.features[-1][0])

cam_gap_fixed = grad_cam_fixed(input_tensor, target_class=CLASS_NAMES.index('shelf_gap'))
# Expected: heatmap now concentrates on the empty shelf space, not the label
```

**Acceptance criterion:** Grad-CAM for `shelf_gap` should concentrate on the central region of the image (the actual empty space) for all three previously misdetected images.

---

## 9 · Progress Check

### What You Can Now Do

| Capability | Tool | When to reach for it |
|---|---|---|
| Identify sensitive input pixels | Vanilla saliency | Quick sanity check; confirm model responds to foreground |
| Locate the spatial region the model used | Grad-CAM | Primary production debugging tool |
| Get fine-grained attribution in the relevant region | Guided Grad-CAM | Detailed audit, high-resolution visualization |
| Attribute with completeness guarantee | Integrated Gradients | Regulatory/compliance reports; saturated networks |
| Quantified per-feature attribution | SHAP GradientExplainer | When Grad-CAM is ambiguous; feature importance ranking |

### Interview Checklist

- Explain the difference between saliency maps and Grad-CAM in one sentence. (Saliency maps measure pixel-level input sensitivity; Grad-CAM measures spatial importance in the last convolutional feature map for a specific class.)
- Why does Grad-CAM apply ReLU to $\sum_k \alpha_k^c A^k$ before upsampling? (To keep only positive contributions — features that increase the class score — and suppress negative contributions that would suppress it.)
- What is the completeness axiom, and which method satisfies it? (Completeness: attributions must sum to $F(x) - F(x')$. Integrated Gradients satisfies it; saliency maps and Grad-CAM do not.)
- How does guided backprop differ from vanilla backprop? (Guided backprop applies an additional gate: it only passes gradients that are positive AND flowing through positive activations, zeroing negative-gradient flows.)
- Give one failure mode of Grad-CAM. (Gradient saturation: near-saturated activations have near-zero gradients, so the map underweights strongly-contributing features in confident predictions.)
- What production workflow does Grad-CAM enable? (On any misdetection: generate the Grad-CAM heatmap, check if the model was looking at the right region, identify spurious correlations, and target data collection to fix them without model architecture changes.)

---

## 10 · Bridge

This chapter completes the ProductionCV system. The grand challenge is satisfied across all six dimensions:
- **mAP >= 85%**: achieved in Ch.3–4 (object detection)
- **IoU >= 70%**: achieved in Ch.5–6 (instance segmentation)
- **Latency < 50ms**: achieved in Ch.2 (efficient architectures)
- **Model size < 100 MB**: achieved in Ch.9–10 (distillation + pruning)
- **Data efficiency < 1000 labels**: achieved in Ch.7–8 (self-supervised pretraining)
- **Auditability**: achieved in Ch.11 (Grad-CAM + Integrated Gradients)

The full compression arc — Ch.9 distillation, Ch.10 pruning, Ch.11 interpretability — demonstrates the production reality: accuracy and efficiency are necessary but not sufficient. Enterprise deployment gates on auditability, and auditability requires tools that translate gradient signals into human-readable spatial explanations.

**What comes after this track:** The [05-agentic-ai](../05-agentic-ai/) track covers LLM-powered agents that use models like ProductionCV as tools — vision models become callable services in a larger agentic pipeline. The [07-ai-infrastructure](../07-ai-infrastructure/) track covers how to serve this 6.8 MB model at production scale: vLLM-style continuous batching for vision models, edge deployment on Jetson Nano, and observability pipelines that log attribution maps alongside predictions for ongoing model monitoring.