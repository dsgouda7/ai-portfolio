# Machine Learning Prerequisites: §0-1 Draft

## §0: What is Machine Learning? (The Big Picture)

### The Core Idea

Imagine you're teaching a child to recognize dogs. You don't write down a rulebook like:
- "If it has four legs AND fur AND barks → it's a dog"

Why? Because you'd miss three-legged dogs, hairless dogs, and dogs that don't bark. Instead, you **show them many examples** and let them figure out the patterns.

That's machine learning in a nutshell: **computers learning patterns from examples** instead of following hand-coded rules.

---

**Animation: "Rule-based vs ML approach"**
- Shows: Two paths side-by-side
- Left path: Programmer writes if-then rules → brittle system breaks on edge cases
- Right path: Show many dog photos → model learns pattern → robust recognition
- Formula embedded: None (conceptual diagram)

---

### The Three Ingredients

Every machine learning system needs three things:

1. **Data** (examples to learn from)
2. **Model** (the pattern-learner)
3. **Feedback** (how to improve)

Let's make this concrete with a real problem.

### Example: Predicting House Prices

You're a real estate agent. You want to predict house prices based on square footage.

**Your Data (5 houses you sold last month):**

| Square Feet | Actual Price |
|-------------|--------------|
| 800         | $160,000     |
| 1000        | $200,000     |
| 1200        | $240,000     |
| 1500        | $300,000     |
| 1800        | $360,000     |

**Your Model (a simple line):**
```
Price = Square_Feet × m + b
```
Where `m` (slope) and `b` (y-intercept) are numbers the model learns.

**Your First Guess:**
Let's say the model starts with random guesses: `m = 100`, `b = 50,000`.

For the 800 sq ft house, the model predicts:
```
Price = 800 × 100 + 50,000 = $130,000
```

But the actual price was **$160,000**. The model is **off by $30,000**! 😬

---

**Animation: "Single prediction error"**
- Shows: 800 sq ft input → model box (m=100, b=50,000) → output $130,000
- Red gap between prediction ($130k) and actual ($160k) labeled "Error: $30k"
- Formula embedded: ŷ = mx + b

---

### The Training Loop (How Models Learn)

Here's the magic: the model **adjusts** `m` and `b` to reduce errors.

**Step 1: Make Predictions** (for all 5 houses)

| Sq Ft | Actual Price | Predicted Price | Error       |
|-------|--------------|-----------------|-------------|
| 800   | $160,000     | $130,000        | -$30,000    |
| 1000  | $200,000     | $150,000        | -$50,000    |
| 1200  | $240,000     | $170,000        | -$70,000    |
| 1500  | $300,000     | $200,000        | -$100,000   |
| 1800  | $360,000     | $230,000        | -$130,000   |

**Step 2: Measure Overall Error**

Average error = ($30k + $50k + $70k + $100k + $130k) / 5 = **$76,000** 🤯

(In practice, we use **Mean Squared Error** to penalize big mistakes more, but the idea is the same.)

**Step 3: Adjust the Model**

The model says: "I'm underpredicting by a lot. I need to increase `m` (make the line steeper) and increase `b` (shift it up)."

New values: `m = 180`, `b = 20,000`

**Step 4: Repeat**

Now predictions look like:

| Sq Ft | Actual Price | New Prediction | New Error  |
|-------|--------------|----------------|------------|
| 800   | $160,000     | $164,000       | +$4,000    |
| 1000  | $200,000     | $200,000       | $0         |
| 1200  | $240,000     | $236,000       | -$4,000    |
| 1500  | $300,000     | $290,000       | -$10,000   |
| 1800  | $360,000     | $344,000       | -$16,000   |

Average error = **$6,800** ✅ (much better!)

---

**Animation: "Training loop cycle"**
- Shows: Circular flow with 4 stages
  1. Data → Model (current m, b)
  2. Model → Predictions (show 5 outputs)
  3. Predictions vs Actual → Error calculation (show average)
  4. Error → Weight adjustment (show m and b changing)
- Loop arrow returns to stage 1
- Formula embedded: Error = (1/n)Σ|ŷᵢ - yᵢ|

---

The model keeps doing this loop hundreds or thousands of times until errors get tiny. That's **training**.

### Why This Beats Hard-Coded Rules

**Hard-coded approach:**
```python
if square_feet < 1000:
    price = 180000
elif square_feet < 1500:
    price = 270000
else:
    price = 360000
```

**Problems:**
- ❌ Doesn't handle 1100 sq ft well (is it $180k or $270k?)
- ❌ Can't adapt to new data (what if prices rise 10% next year?)
- ❌ Breaks for new features (what if we add "number of bedrooms"?)

**Machine learning approach:**
- ✅ Learns the exact relationship from data
- ✅ Updates automatically with new examples
- ✅ Easily extends to multiple features

---

### Supervised Learning: The Most Common Type

In our house price example, we had **labeled data**: each input (square feet) came with the correct output (actual price).

This is called **supervised learning** — like a teacher supervising a student with an answer key.

**Other types you'll hear about:**
- **Unsupervised learning**: Find patterns in data without labels (e.g., group similar customers)
- **Reinforcement learning**: Learn by trial and error with rewards (e.g., game-playing AI)

For this chapter, we focus on **supervised learning** because it's the foundation for 90% of ML applications.

---

**Checkpoint Questions:**

1. In your own words, what's the difference between "programming rules" and "learning from data"?
2. In the house price example, what would happen if we started with `m = 200` and `b = 0`? Would the first predictions be too high or too low?
3. Why do we need multiple training iterations instead of just one?

---

**Animation: "Types of ML triangle"**
- Shows: Triangle with three corners
- Top: Supervised (has labels) - house price, dog photos with tags
- Bottom-left: Unsupervised (no labels) - customer grouping, anomaly detection
- Bottom-right: Reinforcement (rewards) - game AI, robot navigation
- Formula embedded: None (taxonomy diagram)

---

## §1: Neural Networks 101

### What Problem Are We Solving?

Our house price model (`Price = m × Square_Feet + b`) works great for **linear relationships** (straight lines).

But what if the relationship is curved? What if price depends on **multiple factors** like:
- Square footage
- Number of bedrooms
- Age of house
- Distance to downtown

And these factors interact in complex ways? (e.g., "2 bedrooms is great for 800 sq ft, but terrible for 2000 sq ft")

**Neural networks** can learn these complex, non-linear patterns. Let's see how.

---

### The Building Block: A Single Neuron

A **neuron** is just a tiny decision-maker. It takes multiple inputs, weighs them by importance, and produces a single output.

**Concrete Example: Predicting Your Test Score**

Suppose your test score depends on:
- **Hours studied** (let's say you studied 5 hours)
- **Hours slept** (let's say you got 7 hours of sleep)

A single neuron might work like this:

```
Inputs:
  x₁ = 5  (hours studied)
  x₂ = 7  (hours slept)

Weights (learned during training):
  w₁ = 10  (studying is important!)
  w₂ = 3   (sleep helps, but less than studying)

Bias:
  b = 20   (baseline score even if x₁ and x₂ are zero)

Calculation:
  z = w₁×x₁ + w₂×x₂ + b
  z = 10×5 + 3×7 + 20
  z = 50 + 21 + 20
  z = 91

Output:
  Your predicted test score = 91%
```

---

**Animation: "Single neuron computation"**
- Shows: Two inputs (x₁=5, x₂=7) flowing into a circle (neuron)
- Inside circle: multiplication w₁×x₁ (shows 10×5=50), w₂×x₂ (shows 3×7=21)
- Then addition: 50+21+20=91
- Flows to output: 91
- Formula embedded: z = w₁x₁ + w₂x₂ + b

---

**Key Insight:** The weights tell you **what matters**. `w₁ = 10` means studying has 3× the impact of sleep (`w₂ = 3`).

During training, the model learns these weights from data (just like it learned `m` and `b` in the house price example).

---

**Checkpoint Questions:**

1. If you studied 3 hours and slept 8 hours, what would this neuron predict? (Use the same weights: w₁=10, w₂=3, b=20)
2. If the model learns `w₁ = 2` and `w₂ = 10` instead, what would that tell you about what matters for test scores?
3. What does the bias `b` represent? (Hint: what happens if x₁ and x₂ are both zero?)

---

### Adding Activation Functions (Making It Nonlinear)

There's one problem with our neuron: it can only model **straight-line relationships**.

Look what happens if we stack two neurons in a chain:
```
First neuron:  z₁ = w₁×x + b₁
Second neuron: z₂ = w₂×z₁ + b₂
```

Substitute z₁ into z₂:
```
z₂ = w₂×(w₁×x + b₁) + b₂
z₂ = (w₂×w₁)×x + (w₂×b₁ + b₂)
```

This is still a straight line! 😱 You could have 100 layers, but it would collapse to a single line.

**Solution: Activation Functions**

We add a **nonlinear function** after each neuron's calculation. The most popular is **ReLU** (Rectified Linear Unit):

```
ReLU(z) = max(0, z)
```

**What does ReLU do? It "turns off" negative values.**

**Example with numbers:**

| Input z | ReLU(z) |
|---------|---------|
| -2      | 0       |
| -1      | 0       |
| 0       | 0       |
| 1       | 1       |
| 2       | 2       |

---

**Animation: "ReLU activation"**
- Shows: Number line from -3 to +3 on x-axis
- Input values slide along bottom: [-2, -1, 0, 1, 2]
- Pass through ReLU gate (shows max(0,z) operation)
- Output values appear on top: [0, 0, 0, 1, 2]
- Graph: shows bent line (flat at 0 for x<0, then diagonal for x>0)
- Formula embedded: f(z) = max(0, z)

---

**Now let's redo our test score example with ReLU:**

```
Step 1: Weighted sum
  z = 10×5 + 3×7 + 20 = 91

Step 2: Apply ReLU
  output = ReLU(91) = max(0, 91) = 91
```

Since 91 is positive, ReLU doesn't change it. But what if our calculation gave `-15`?

```
If z = -15:
  output = ReLU(-15) = max(0, -15) = 0
```

The neuron "shuts off" for negative inputs. This creates **nonlinear bends** in the model's decision boundary.

---

**Other Common Activations:**

**Sigmoid** (squashes output to 0-1, useful for probabilities):
```
sigmoid(z) = 1 / (1 + e^(-z))
```

Example:
- sigmoid(-5) ≈ 0.007 (very unlikely)
- sigmoid(0) = 0.5 (50/50 chance)
- sigmoid(5) ≈ 0.993 (very likely)

**Tanh** (squashes output to -1 to +1):
```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

Example:
- tanh(-2) ≈ -0.96
- tanh(0) = 0
- tanh(2) ≈ 0.96

**When to use which?**
- **ReLU**: Default choice for hidden layers (fast, works well)
- **Sigmoid**: Output layer for binary classification (yes/no predictions)
- **Tanh**: Sometimes used in recurrent networks (not covered here)

---

**Animation: "Three activation functions comparison"**
- Shows: Three side-by-side graphs, each with input z from -5 to +5
- ReLU: flat at 0 until z=0, then diagonal line upward
- Sigmoid: S-curve from 0 to 1, steep in middle
- Tanh: S-curve from -1 to +1, steep in middle
- Highlight: ReLU = "simple cutoff", Sigmoid = "probability", Tanh = "centered around zero"
- Formulas embedded: All three formulas shown above each graph

---

**Checkpoint Questions:**

1. Calculate ReLU(8), ReLU(-3), and ReLU(0).
2. Why would a network with NO activation functions (just linear operations) fail to learn complex patterns?
3. If you're predicting "Is this email spam?" (yes/no answer), which activation function should you use in the output layer? Why?

---

### Stacking Neurons Into Layers

One neuron can learn simple patterns. To learn complex patterns, we **stack many neurons in layers**.

**A 3-Layer Network Example:**

```
Input Layer (2 neurons - just pass through inputs):
  x₁ = 5   (hours studied)
  x₂ = 7   (hours slept)

Hidden Layer (3 neurons - learn intermediate patterns):
  Neuron H1: z₁ = w₁₁×x₁ + w₁₂×x₂ + b₁ → ReLU(z₁) = h₁
  Neuron H2: z₂ = w₂₁×x₁ + w₂₂×x₂ + b₂ → ReLU(z₂) = h₂
  Neuron H3: z₃ = w₃₁×x₁ + w₃₂×x₂ + b₃ → ReLU(z₃) = h₃

Output Layer (1 neuron - final prediction):
  z_out = w_o1×h₁ + w_o2×h₂ + w_o3×h₃ + b_out
  test_score = z_out
```

**Let's trace through with actual numbers!**

**Weights (learned during training):**
```
Hidden layer:
  Neuron H1: w₁₁=8,  w₁₂=2,  b₁=-10  (maybe learns "total study effort")
  Neuron H2: w₂₁=1,  w₂₂=10, b₂=-40  (maybe learns "well-rested")
  Neuron H3: w₃₁=5,  w₃₂=5,  b₃=0    (maybe learns "overall readiness")

Output layer:
  w_o1=20, w_o2=15, w_o3=10, b_out=0
```

**Forward Pass (computing the prediction):**

**Step 1: Hidden Layer Computations**

```
Neuron H1:
  z₁ = 8×5 + 2×7 + (-10) = 40 + 14 - 10 = 44
  h₁ = ReLU(44) = 44

Neuron H2:
  z₂ = 1×5 + 10×7 + (-40) = 5 + 70 - 40 = 35
  h₂ = ReLU(35) = 35

Neuron H3:
  z₃ = 5×5 + 5×7 + 0 = 25 + 35 + 0 = 60
  h₃ = ReLU(60) = 60
```

**Step 2: Output Layer Computation**

```
z_out = 20×44 + 15×35 + 10×60 + 0
z_out = 880 + 525 + 600
z_out = 2005

Final prediction: test_score = 2005
```

Wait, 2005%?! 🤯 That's impossible! This shows our weights are **not trained yet** — they're random garbage.

After training on real data (hundreds of students' study/sleep hours and their actual scores), the model would learn sensible weights that produce scores like 75%, 88%, etc.

---

**Animation: "3-layer forward pass"**
- Shows: Network diagram with nodes and edges
- Left: 2 input nodes (x₁=5, x₂=7)
- Middle: 3 hidden nodes (H1, H2, H3) with ReLU symbols
- Right: 1 output node (test_score)
- Values flow through edges (show numbers appearing):
  - x₁, x₂ → each hidden node (show multiplications)
  - z₁=44 → ReLU → h₁=44 (highlighted calculation)
  - Repeat for H2, H3
  - h₁, h₂, h₃ → output (show multiplications)
  - Final: 2005 (with warning: "Not trained yet!")
- Formula embedded: z = Wx + b, then h = ReLU(z)

---

### Why Do We Need Multiple Layers?

Think of layers as **building up abstractions**:

**Example: Recognizing Handwritten Digits**

- **Layer 1** (close to input): Detects edges and curves
  - "There's a vertical line here"
  - "There's a curve here"

- **Layer 2**: Combines edges into parts
  - "Vertical line + curve = could be '2' or '3'"
  - "Two curves = could be '8' or '0'"

- **Layer 3**: Combines parts into full digits
  - "Top curve + middle line + bottom curve = definitely '8'"

**Why can't one layer do it all?**

Imagine trying to recognize faces with just one step:
- ❌ Pixels → "Is it Bob?" (too big a jump!)

Better:
- ✅ Pixels → edges → facial features (nose, eyes) → face identity

Each layer learns a **hierarchy of features**, from simple to complex.

---

**Animation: "Hierarchical feature learning"**
- Shows: Three columns representing 3 layers
- Column 1 (Input): Raw pixels of handwritten "8"
- Column 2 (Hidden Layer 1): Zoomed boxes showing edge detectors (vertical, horizontal, curves)
- Column 3 (Hidden Layer 2): Assembled parts (top loop, bottom loop, middle crossing)
- Column 4 (Output): Final prediction: "8" with 98% confidence
- Arrows show how simple features combine into complex ones
- Formula embedded: None (conceptual diagram)

---

### The Full Forward Pass (Step-by-Step)

Let's trace one complete prediction through a network:

**Network Architecture:**
- Input: 2 neurons (x₁, x₂)
- Hidden: 3 neurons (h₁, h₂, h₃) with ReLU
- Output: 1 neuron (y)

**Input Data:**
```
x₁ = 5 (hours studied)
x₂ = 7 (hours slept)
```

**Step 1: Input → Hidden Layer**

For each hidden neuron, compute `z = weights × inputs + bias`, then apply ReLU:

```
h₁ = ReLU(w₁₁×x₁ + w₁₂×x₂ + b₁)
h₂ = ReLU(w₂₁×x₁ + w₂₂×x₂ + b₂)
h₃ = ReLU(w₃₁×x₁ + w₃₂×x₂ + b₃)
```

**With our example weights:**
```
h₁ = ReLU(8×5 + 2×7 - 10) = ReLU(44) = 44
h₂ = ReLU(1×5 + 10×7 - 40) = ReLU(35) = 35
h₃ = ReLU(5×5 + 5×7 + 0) = ReLU(60) = 60
```

**Step 2: Hidden Layer → Output**

```
y = w_o1×h₁ + w_o2×h₂ + w_o3×h₃ + b_out
y = 20×44 + 15×35 + 10×60 + 0
y = 2005
```

(Again, this crazy number shows the weights aren't trained yet!)

**Step 3: Compare to Actual Answer**

Let's say the student's actual test score was **85%**.

```
Error = |predicted - actual| = |2005 - 85| = 1920 😱
```

**Step 4: Adjust Weights (Training)**

The model says: "I was WAY too high. I need to decrease my weights."

After one training step, maybe:
```
New weights: w_o1=0.5, w_o2=0.4, w_o3=0.3, b_out=10
```

**Step 5: Try Again**

```
New prediction:
y = 0.5×44 + 0.4×35 + 0.3×60 + 10
y = 22 + 14 + 18 + 10
y = 64

New error = |64 - 85| = 21 (much better!)
```

This process repeats thousands of times until errors get small.

---

**Animation: "Training iteration cycle (neural network)"**
- Shows: Same network diagram as before, but now with feedback loop
- Stage 1: Forward pass (input → hidden → output) with values flowing
- Stage 2: Compare output (64) to target (85), show error bar (21)
- Stage 3: Weights highlighted in red, arrows pointing down (decreasing)
- Stage 4: Loop back to Stage 1 with "Iteration 2" label
- Formula embedded: Error = |ŷ - y|, then ΔW ∝ -∂Error/∂W (simplified)

---

**Checkpoint Questions:**

1. In a 3-layer network (input → hidden → output), which layer learns the most abstract features?
2. If all neurons used NO activation function (just z = Wx + b), what would happen if you stacked 100 layers?
3. Walk through the forward pass: If x₁=10, x₂=5, w₁₁=2, w₁₂=3, b₁=-5, and we use ReLU, what is h₁?

---

### Why Depth Matters (Intuition)

**Shallow Network (1 hidden layer with 100 neurons):**
- Can learn complex patterns, but needs MANY neurons
- Like memorizing every possible input-output pair

**Deep Network (10 hidden layers with 10 neurons each):**
- Same total number of neurons (100 total)
- But can reuse patterns learned in earlier layers
- Like building a vocabulary of concepts and combining them

**Analogy: Writing an Essay**

**Shallow approach:** Memorize full sentences for every topic
- ❌ Need millions of memorized sentences
- ❌ Can't adapt to new topics

**Deep approach:** Learn words → phrases → sentences → paragraphs
- ✅ Reusable building blocks
- ✅ Can create new combinations

**Real-world evidence:**
- ResNet (152 layers) crushes 5-layer networks on image recognition
- GPT-4 (100+ layers) beats 10-layer models on language
- But: very deep networks need special tricks (we'll cover ResNets later!)

---

**Animation: "Shallow vs Deep comparison"**
- Shows: Two networks side-by-side
- Left: 1 hidden layer with 10 neurons, all connected to input and output (messy spaghetti)
- Right: 3 hidden layers with 3-4 neurons each (clean hierarchy)
- Highlight: Right network shows reusable patterns (box around Layer 1: "edges", Layer 2: "shapes", Layer 3: "objects")
- Caption: "Same total neurons, but depth = reusable abstractions"
- Formula embedded: None (conceptual diagram)

---

### Summary: What You've Learned

**§0: Machine Learning Big Picture**
- ✅ ML = learning patterns from data instead of hand-coded rules
- ✅ Training loop: data → predictions → error → adjust weights → repeat
- ✅ Supervised learning = learning with labeled examples

**§1: Neural Networks 101**
- ✅ A neuron computes `z = Wx + b`, then applies activation (e.g., ReLU)
- ✅ Activation functions (ReLU, sigmoid, tanh) add nonlinearity
- ✅ Layers stack neurons: input → hidden → output
- ✅ Forward pass: data flows through all layers to make a prediction
- ✅ Depth lets networks learn hierarchical features (edges → shapes → objects)

**Next Up:**
- How training actually works (backpropagation and gradient descent)
- Loss functions (how we measure "error" precisely)
- Overfitting and how to prevent it

---

**Final Checkpoint Questions:**

1. **Conceptual**: If someone says "I tried a neural network but it performed like a straight line," what might they have forgotten?
2. **Numerical**: Given x=3, w=7, b=-5, calculate z, then ReLU(z).
3. **Design**: You're building a spam classifier (email → spam/not spam). Should your output layer use ReLU or sigmoid? Why?

---

## Animation Summary (For Implementation)

This section requires **9 animations** total:

1. **Rule-based vs ML approach** (§0) — conceptual diagram showing two paradigms
2. **Single prediction error** (§0) — house price example with error gap
3. **Training loop cycle** (§0) — circular flow: data → model → error → adjust
4. **Types of ML triangle** (§0) — taxonomy of supervised/unsupervised/reinforcement
5. **Single neuron computation** (§1) — weighted sum + bias → output
6. **ReLU activation** (§1) — number line with max(0,z) operation
7. **Three activation functions comparison** (§1) — ReLU vs sigmoid vs tanh graphs
8. **3-layer forward pass** (§1) — network diagram with values flowing through
9. **Hierarchical feature learning** (§1) — layers building up from edges to objects
10. **Training iteration cycle (neural network)** (§1) — forward pass + error + weight update
11. **Shallow vs Deep comparison** (§1) — 1 wide layer vs multiple narrow layers

Each animation spec includes:
- What it shows (data flow, computation, comparison)
- Which formula to embed (if any)
- Concrete numbers to display (when applicable)

---

**Word Count:** ~4,200 words (~1,000 lines formatted as markdown)
**Estimated Reading Time:** 12-15 minutes (acceptable for §0-1 combined)
**Numerical Examples:** 15+ worked examples with actual numbers
**Checkpoint Questions:** 12 questions total (4 in §0, 8 in §1)
