# Ch.0 Animation Concepts

This chapter is diagram-heavy but doesn't include generated animations (unlike notes/01 and notes/02 chapters). All visualizations are **mermaid diagrams** embedded directly in the README.

## Potential Future Animations

If this chapter were to include deterministic animations (following notes/02 style), these would be candidates:

### 1. Gradient Flow Comparison Animation
**Concept:** Show gradient magnitude decay across:
- 100-layer plain RNN (exponential decay to ~0)
- 100-layer ResNet (preserved via skip connections)
- 100-token Transformer (direct paths via attention)

**Visual:** Three parallel timelines with gradient strength meter (green → yellow → red)

### 2. Sequential vs Parallel Processing
**Concept:** Side-by-side animation of:
- LSTM processing tokens one-by-one (sequential bottleneck)
- Transformer processing all tokens simultaneously (parallelism)

**Visual:** Tokens moving through computation graph; clock showing time elapsed

### 3. Attention Mechanism Soft Lookup
**Concept:** Visualize Q/K/V soft dictionary lookup:
- Query token "looks" at all key tokens
- Similarity scores → softmax → weights
- Weighted sum of values → output

**Visual:** Heat map of attention weights, animated value aggregation

### 4. Encoder vs Decoder Attention Masks
**Concept:** Show bidirectional (BERT) vs causal (GPT) attention patterns:
- BERT: full attention matrix (all-to-all)
- GPT: lower triangular matrix (causal mask)

**Visual:** Animated attention matrix with masked regions

### 5. Three-Stage Training Pipeline
**Concept:** Pre-training → SFT → RLHF progression:
- Stage 1: Model learns general language patterns
- Stage 2: Learns instruction-following format
- Stage 3: Aligns with human preferences

**Visual:** Model capability metrics evolving through stages

---

## Current State

All diagrams in Ch.0 are **static mermaid** rendered by GitHub/VS Code markdown preview. No Python generation scripts required.

**Mermaid diagrams included:**
- Section 0: Evolution arc (7 nodes)
- Section 1: RNN sequential processing (4 tokens)
- Section 2: Attention all-to-all (4 tokens)
- Section 3: Transformer block structure
- Section 4: Encoder/decoder fork (BERT vs GPT)
- Section 5: Timeline (2017-2024)
- Section 5: Three-stage training pipeline
- Section 6: Vision transformers evolution
- Section 7: Gradient flow comparison (3 architectures)
- Section 8: Production RAG stack

All mermaid diagrams follow the color scheme from notes/02:
- Gray: Foundation (dense networks, RNNs)
- Red: Problem states (vanishing gradients)
- Orange: Intermediate solutions (LSTMs, attention)
- Green: Breakthroughs (transformers, ResNets)
- Blue: Applications (LLMs, generation)
- Purple: Production (embeddings, RAG)
