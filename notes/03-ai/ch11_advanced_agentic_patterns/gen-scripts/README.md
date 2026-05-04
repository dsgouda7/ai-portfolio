# Chapter 11 Agentic Patterns Animation Scripts

This folder contains Python scripts that generate animated GIFs illustrating advanced agentic patterns for Chapter 11.

## Generated Animations

### 1. Reflection Loop (`gen_ch11_reflection_loop.py`)
**Output:** `../img/ch11_reflection_loop.gif`

Illustrates the **Generate → Critique → Revise** pattern with color-coded stages:
- 🔴 Red: Error detected
- 🟡 Yellow: Critique phase
- 🟢 Green: Successful resolution

**Example:** PizzaBot detecting contradictions in "gluten-free pizza with extra cheese" and self-correcting to "gluten-free pizza with vegan cheese"

**Key Metrics:**
- Shows iteration count
- Token cost: 3× base per iteration
- Demonstrates early-exit detection

---

### 2. Debate Flow (`gen_ch11_debate_flow.py`)
**Output:** `../img/ch11_debate_flow.gif`

Shows **multi-agent debate** with 2 agents proposing solutions and 1 judge arbitrating:
- Agent 1 (Strict): Conservative interpretation
- Agent 2 (Generous): Liberal interpretation  
- Judge: Makes final decision using RAG database lookup

**Example:** Pricing dispute resolution with conflicting discount policies

**Features:**
- Speech bubbles showing agent proposals
- RAG database lookup visualization
- Policy document grounding
- Pitfall warning: groupthink without external validation

---

### 3. Hierarchical Flow (`gen_ch11_hierarchical_flow.py`)
**Output:** `../img/ch11_hierarchical_flow.gif`

Demonstrates **Planner → Workers → Verifier** pattern:
- Planner decomposes complex task into subtasks
- Workers execute in parallel (shows ⚡ parallel execution)
- Verifier validates all constraints

**Example:** Catering order (15 pizzas, 3 delivery times, $200 budget)

**Visual Elements:**
- Tree structure showing task decomposition
- Checkmarks on completed nodes
- Verification phase with constraint checking
- Cost breakdown: 1 planner + N workers + 1 verifier

---

### 4. Tree of Thoughts (`gen_ch11_tree_of_thoughts.py`)
**Output:** `../img/ch11_tree_of_thoughts.gif`

Illustrates **branching reasoning** with multiple paths explored:
- Level 1: Generate 3 candidate approaches (A, B, C)
- Level 2: Expand best branch (B) into 2 refinements
- Level 3: Further refinement of top candidate
- **Best path highlighted** with gold edges after evaluation

**Scoring:**
- Each node shows evaluation score
- Color-coded: Green (good), Blue (neutral), Red (poor)
- Final path: Root → B (7.8) → B1 (8.1) → B1a (8.9)

**Shows:**
- Breadth vs depth exploration tradeoff
- Pruning of low-scoring branches
- Multi-level reasoning optimization

---

### 5. Chain of Verification (`gen_ch11_chain_of_verification.py`)
**Output:** `../img/ch11_chain_of_verification.gif`

Shows **Generate → Verify → Correct** iteration:
- Generate: Create initial claims
- Verify: Systematically check each claim
- Correct: Fix failed claims with verified data

**Example:** Verifying nutritional claims about gluten-free pizza

**Verification States:**
- ○ Pending (gray)
- 🔍 Verifying (orange)
- ✓ Verified (green)
- ✗ Failed (red)

**Metrics:**
- Confidence scores per claim
- Correction tracking
- Token cost: ~4× base (1 gen + 3 verifications + 1 correction)

---

## Running the Scripts

### Prerequisites
```bash
pip install matplotlib imageio numpy
```

### Generate Single Animation
```bash
cd notes/03-ai/ch11_advanced_agentic_patterns/gen_scripts
python gen_ch11_reflection_loop.py
```

### Generate All Animations
```bash
python gen_ch11_reflection_loop.py
python gen_ch11_debate_flow.py
python gen_ch11_hierarchical_flow.py
python gen_ch11_tree_of_thoughts.py
python gen_ch11_chain_of_verification.py
```

### Output
All animations are saved to `../img/` directory as GIF files.

---

## Animation Design Principles

### Timing
- Initial frames: 1.0-1.3s (scene setup)
- Action frames: 1.2-1.5s (transitions, evaluations)
- Final frames: 1.5-2.0s (results, summaries)

### Color Scheme
Consistent across all animations:
- **Purple** (#9B59B6): Coordinators, planners, root nodes
- **Blue** (#3498DB): Workers, agents, standard operations
- **Green** (#27AE60): Success, verification, correctness
- **Red** (#E74C3C): Errors, failures, rejections
- **Orange** (#F39C12): Critique, verification in progress, highlights
- **Gray** (#95A5A6): Inactive, pending states

### Visual Elements
- **Rounded boxes**: Agents, stages, nodes
- **Arrows**: Flow direction, dependencies
- **Checkmarks**: Completed tasks, verified claims
- **Speech bubbles**: Agent communication (debate pattern)
- **Score badges**: Evaluation metrics (tree of thoughts)

---

## File Structure

```
notes/03-ai/ch11_advanced_agentic_patterns/
├── gen_scripts/
│   ├── gen_ch11_reflection_loop.py
│   ├── gen_ch11_debate_flow.py
│   ├── gen_ch11_hierarchical_flow.py
│   ├── gen_ch11_tree_of_thoughts.py
│   ├── gen_ch11_chain_of_verification.py
│   └── README.md (this file)
└── img/
    ├── ch11_reflection_loop.gif
    ├── ch11_debate_flow.gif
    ├── ch11_hierarchical_flow.gif
    ├── ch11_tree_of_thoughts.gif
    └── ch11_chain_of_verification.gif
```

---

## Notes

- All scripts use `matplotlib.use('Agg')` for headless rendering
- Temporary frame PNGs are automatically cleaned up after GIF creation
- Frame counts vary: 5-7 frames per animation
- Total duration: 6-10 seconds per animation
- DPI: 100 (balance between quality and file size)
- Loop: 0 (infinite loop for documentation/presentation use)

---

## Future Enhancements (Part 2)

Part 2 will add:
- Pattern comparison matrix animation
- Cost vs accuracy tradeoff visualization
- Real-world use case decision tree
- Pattern selection flowchart
- Anti-pattern warnings animation

