# Ch.0 Animation Requirements

## Primary Animation (Header)

**File:** `prerequisites-foundations.gif`
**Purpose:** Show 6 building blocks assembling into transformer architecture
**Placement:** After § 0 Challenge section under `## Animation`

**Sequence:**
1. Frame 1: Embeddings (text → vectors with similarity)
2. Frame 2: RNN (sequential processing with h₁ → h₂ → h₃)
3. Frame 3: Attention (parallel Q/K/V computation with heatmap)
4. Frame 4: Skip connections (+1 gradient path highlighted)
5. Frame 5: Encoder-decoder (bidirectional ↔ causal flow)
6. Frame 6: Training loop (forward → loss → backward → update)
7. Frame 7: All 6 combine into transformer block diagram

**Style:**
- Dark background (#1a1a2e)
- Mermaid color palette (primary: #1e3a8a, success: #15803d)
- 2-3 seconds per frame
- Smooth transitions

## Supporting Animations (Optional Enhancements)

### 1. Vanishing Gradient Decay
**File:** `vanishing-gradient-decay.gif`
**Section:** § 2.2
**Shows:** Bar chart: gradient strength fading Token 1 → Token 100
**Caption:** "After 50 tokens: 0.8^50 = 0.0014% of original gradient"

### 2. Attention Heatmap
**File:** `attention-3token-heatmap.gif`
**Section:** § 3.2
**Shows:** "The river bank" attention matrix with weight values animating
**Caption:** "bank attends to river (38.7%) and itself (41.6%)"

### 3. Skip Connection Gradient Flow
**File:** `skip-connection-gradient-flow.gif`
**Section:** § 4.3
**Shows:** Side-by-side gradient bars through 40 layers (Plain 2% vs ResNet 75%)
**Caption:** "Plain network: 2% at layer 1 | ResNet: 75% at layer 1"

### 4. Encoder-Decoder Information Flow
**File:** `encoder-decoder-flow.gif`
**Section:** § 5.2
**Shows:** Animated arrows showing bidirectional encoder → cross-attention → causal decoder
**Caption:** "Encoder sees all tokens | Decoder generates left-to-right"

### 5. Forward Pass Transformation
**File:** `forward-pass-transformation.gif`
**Section:** § 6.1A
**Shows:** Input (2D) → Layer 1 (4D) → Layer 2 (3D) → Output (2D) with shape annotations
**Caption:** "Each layer transforms representations to build abstractions"
**Note:** Visualization code already exists in notebook cell 26

## Implementation Priority

**Must have:** `prerequisites-foundations.gif` (primary animation)
**Should have:** Vanishing gradient, attention heatmap, skip connection flow
**Nice to have:** Encoder-decoder flow, forward pass transformation

## Generation Scripts

Scripts should follow naming convention:
- `scripts/generate-ch00-[topic].py`
- Use matplotlib with dark background
- Output to `notes/03-ai/ch00-from-networks-to-language/img/`
- Follow existing animation conventions from notes/01-ml

## References

See existing animation implementations:
- `notes/01-ml/*/img/*_generated.gif`
- `scripts/ml_*/*.py` for generation patterns
- Authoring guidelines § 11 for image conventions
