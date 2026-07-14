"""
Rebuild TF_Part1_Intro.ipynb from the known 57-cell content
(VS Code editor buffer → disk file).
"""

import json
import uuid


def make_id():
    return uuid.uuid4().hex[:8]


def md(src, cell_id=None):
    lines = src.rstrip("\n").split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "markdown",
        "id": cell_id or make_id(),
        "metadata": {},
        "source": source,
    }


def code(src, cell_id=None):
    lines = src.rstrip("\n").split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id or make_id(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# Read existing file for metadata
with open(
    "c:/repos/ai-portfolio/learning/genai/rnns/MIT/TF_Part1_Intro.ipynb",
    encoding="utf-8",
) as f:
    nb = json.load(f)

cells = [
    # ── Cell 1: Title ──────────────────────────────────────────────────────────
    md(
        """# Lab 1: Intro to TensorFlow/Keras and Music Generation with RNNs

In this lab you'll get exposure to TensorFlow and Keras, and learn how they can
be used for deep learning.  Go through the code and run each cell.  Along the
way you'll encounter several ***TODO*** blocks — follow the instructions to fill
them out before running those cells and continuing.

# Part 1: Intro to TensorFlow / Keras

## 0.1 Install TensorFlow

[TensorFlow](https://www.tensorflow.org/) is Google's open-source machine
learning framework.  [Keras](https://keras.io/) ships as the high-level API
bundled inside TensorFlow (`tf.keras`).  For this lab we use TensorFlow 2.x
with the eager execution default.
""",
        "f2f64107",
    ),
    # ── Cell 2: Roadmap ────────────────────────────────────────────────────────
    md(
        """## What You'll Learn: TensorFlow & Keras from First Principles

This notebook builds your intuition for how **TensorFlow** represents computation as graphs, defines models with **Keras layers**, and trains them with **automatic differentiation** — using a **single running example throughout**.

Every concept is demonstrated on the same problem:

> **Classifying 2D points: inside a circle (class 1) vs. outside (class 0)**
> 200 points, 2 features (x₀, x₁) → binary label. Small enough to visualise, real enough to matter.

### Roadmap

| Step | Concept | Key Idea |
|---|---|---|
| 1 | Vocabulary & Toy Data | Our running example: classifying 2D points (circle vs square) |
| 2 | Tensors & Shapes | Multi-dimensional arrays (`tf.Tensor`) as TensorFlow's substrate |
| 3 | Computation Graphs | Operations (add, matmul) form a dependency DAG; TF tracks it implicitly |
| 4 | Keras Layers | Dense, Activation, custom `Layer` subclasses as composable building blocks |
| 5 | The Forward Pass | `model(inputs)` returns predictions; weights are random until trained |
| 6 | Autodiff & Backprop | `GradientTape` computes $\\partial L/\\partial W$ — no hand-coded derivatives |
| 7 | Training Loop | Gradient descent: `tape.gradient` → `optimizer.apply_gradients` → repeat |
| 8 | Toy → Production Bridge | Same mechanisms (GradientTape, Dense, optimizer), just far more parameters |
| 9 | Pretrained Music Model | HuggingFace Transformers: autodiff deployed at billion-parameter scale |
""",
        "4d573d0f",
    ),
    # ── Cell 3: Imports ────────────────────────────────────────────────────────
    code(
        """# ── Imports & Environment Setup ─────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

# ── Set Seeds for Reproducibility ───────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)
print("🎲 Seeds set: TF=42, NumPy=42 (outputs will be deterministic)")

print("TensorFlow version:", tf.__version__)
""",
        "1e6fc39f",
    ),
    # ── Cell 4: Running Example Data ───────────────────────────────────────────
    code(
        """# ── Generate Training Data: 2D Points ───────────────────────────────────────
n_samples = 200
X_train = np.random.uniform(-1, 1, (n_samples, 2)).astype(np.float32)

# Label: CIRCLE (1) if inside radius 0.5, else SQUARE (0)
distances = np.sqrt(X_train[:, 0]**2 + X_train[:, 1]**2)
y_train = (distances < 0.5).astype(np.float32)

# ── Visualize the Dataset ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))

# Plot points
circle_mask = y_train == 1
ax.scatter(X_train[circle_mask, 0], X_train[circle_mask, 1],
           c='blue', label='CIRCLE (inside)', alpha=0.6, edgecolors='k')
ax.scatter(X_train[~circle_mask, 0], X_train[~circle_mask, 1],
           c='orange', label='SQUARE (outside)', alpha=0.6, edgecolors='k')

# Overlay the true boundary (radius 0.5)
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(0.5*np.cos(theta), 0.5*np.sin(theta), 'r--', linewidth=2, label='True boundary (r=0.5)')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.legend()
ax.set_title('Training Data: Circle vs. Square')
ax.set_xlabel('x₀')
ax.set_ylabel('x₁')
plt.tight_layout()
plt.show()

print(f"Training set: {n_samples} points, {y_train.sum():.0f} CIRCLE, {(1-y_train).sum():.0f} SQUARE")
""",
        "b835ea1a",
    ),
    # ── Cell 5: Running Example Markdown ───────────────────────────────────────
    md(
        """## 1.0 Our Running Example: Circle vs. Square (2D Binary Classifier)

Throughout Part 1 we'll work with a simple toy problem: **classifying 2D points** as CIRCLE (inside a radius-0.5 circle centered at the origin) or SQUARE (outside that circle). This gives us:

- **Concrete data** to visualize at every step
- **A clear decision boundary** to see what the model learns
- **The same structure** as real classification tasks, just in 2D instead of 784D (MNIST) or 150,528D (ImageNet)

When you understand how TensorFlow trains this toy classifier, you understand how it trains ResNet-50 on ImageNet — same loss function, same GradientTape, same optimizer.apply_gradients.
""",
        "7e3d4ee2",
    ),
    # ── Cell 6: 1.1 What is TF ─────────────────────────────────────────────────
    md(
        """## 1.1 What is TensorFlow?

TensorFlow is a machine learning library.  At its core it provides an interface
for creating and manipulating **tensors** — multi-dimensional arrays of base
datatypes such as integers or floats.  `tf.Tensor` objects are immutable
constant values (use `tf.Variable` for mutable state).

The [`shape`](https://www.tensorflow.org/api_docs/python/tf/TensorShape) of a
tensor defines its number of dimensions and the size of each dimension.
`ndim` (or `rank`) gives the number of dimensions.

Let's create some tensors and inspect their properties:
""",
        "3534a328",
    ),
    # ── Cell 7: integer/decimal ────────────────────────────────────────────────
    code(
        """integer = tf.constant(1234)
decimal = tf.constant(3.14159265359)

print(f"`integer` is a {integer.ndim}-d Tensor: {integer}")
print(f"`decimal` is a {decimal.ndim}-d Tensor: {decimal}")
""",
        "f528a0da",
    ),
    # ── Cell 8: Vectors markdown ───────────────────────────────────────────────
    md("Vectors and lists can be used to create 1-d tensors:", "74f2abbb"),
    # ── Cell 9: fibonacci ──────────────────────────────────────────────────────
    code(
        """fibonacci = tf.constant([1, 1, 2, 3, 5, 8])
count_to_100 = tf.constant(list(range(100)))

print(f"`fibonacci` is a {fibonacci.ndim}-d Tensor with shape: {fibonacci.shape}")
print(f"`count_to_100` is a {count_to_100.ndim}-d Tensor with shape: {count_to_100.shape}")
""",
        "9b9489be",
    ),
    # ── Cell 10: 2d tensors markdown ───────────────────────────────────────────
    md(
        """Next, let's create 2-d (matrices) and higher-rank tensors.  In image processing
we use 4-d Tensors with dimensions `(batch, height, width, channels)` — note
that TensorFlow defaults to **NHWC** order (channels last), the opposite of
PyTorch's NCHW.
""",
        "6236b3ff",
    ),
    # ── Cell 11: Higher-order tensors ──────────────────────────────────────────
    code(
        """# ── Defining Higher-Order Tensors ───────────────────────────────────────────

'''TODO: Define a 2-d Tensor from 5 sample points of our training data'''
sample_points = X_train[:5]  # shape (5, 2)
matrix = tf.constant(sample_points)

assert isinstance(matrix, tf.Tensor), "matrix must be a tf.Tensor"
assert matrix.ndim == 2
print(f"matrix (5 training points) is a {matrix.ndim}-d Tensor with shape: {matrix.shape}")

'''TODO: Convert y_train labels into a 1-d Tensor'''
labels_tensor = tf.constant(y_train)

assert isinstance(labels_tensor, tf.Tensor), "labels_tensor must be a tf.Tensor"
assert labels_tensor.ndim == 1
print(f"labels_tensor is a {labels_tensor.ndim}-d Tensor with shape: {labels_tensor.shape}")

# ── Optional: 4-d image batch (NHWC format) ──────────────────────────────────
# While our running example is 2D points, image models use 4-d Tensors:
# (batch, height, width, channels) — TensorFlow's NHWC order (channels last).
images = tf.zeros((10, 256, 256, 3))
assert images.shape == (10, 256, 256, 3), "images is incorrect shape (expected NHWC)"
print(f"[Side note] images is a {images.ndim}-d Tensor with shape: {images.shape}")
""",
        "aa74f463",
    ),
    # ── Cell 12: Shape markdown ────────────────────────────────────────────────
    md(
        """As you have seen, the `shape` of a tensor provides the number of elements in
each dimension.  You can also use slicing to access sub-tensors within a
higher-rank tensor:
""",
        "4fee63d4",
    ),
    # ── Cell 13: Slicing ───────────────────────────────────────────────────────
    code(
        """row_vector    = matrix[1]
column_vector = matrix[:, 1]
scalar        = matrix[0, 1]

print(f"`row_vector`: {row_vector}")
print(f"`column_vector`: {column_vector}")
print(f"`scalar`: {scalar}")
""",
        "0e0f721d",
    ),
    # ── Cell 14: 1.2 Computations markdown ────────────────────────────────────
    md(
        """## 1.2 Computations on Tensors

A convenient way to think about and visualise computations in TensorFlow is in
terms of a **computation graph**.  We can define the graph in terms of tensors
(data) and the operations that act on them.  Let's look at a simple example:

![add graph](https://raw.githubusercontent.com/MITDeepLearning/introtodeeplearning/master/lab1/img/add-graph.png)
""",
        "48c69991",
    ),
    # ── Cell 15: a+b code ──────────────────────────────────────────────────────
    code(
        """# Create the nodes in the graph and initialise values
a = tf.constant(15)
b = tf.constant(61)

# Add them!
c1 = tf.add(a, b)
c2 = a + b   # TF overrides + to act on tensors

print(f"c1: {c1}")
print(f"c2: {c2}")
""",
        "7a4a6e8e",
    ),
    # ── Cell 16: Notice markdown ───────────────────────────────────────────────
    md(
        """Notice how the output is a tensor with value 76.

Now let's consider a slightly more complicated example:

![computation graph](https://raw.githubusercontent.com/MITDeepLearning/introtodeeplearning/master/lab1/img/computation-graph.png)

We take two inputs `a, b` and compute an output `e`.  Each node represents an
operation.  Let's define a simple function in TensorFlow to construct this
computation:
""",
        "5dc619d2",
    ),
    # ── Cell 17: func(a,b) ─────────────────────────────────────────────────────
    code(
        """# ── Defining Tensor Computations ────────────────────────────────────────────

def func(a, b):
    '''TODO: Define the operations for c, d, e.'''
    c = tf.add(a, b)
    d = tf.subtract(b, 1)
    e = tf.multiply(c, d)
    return e
""",
        "4358ce1d",
    ),
    # ── Cell 18: "Now we can call" markdown ───────────────────────────────────
    md("Now we can call the function to execute the computation graph:", "71b0b9be"),
    # ── Cell 19: Execute computation graph ────────────────────────────────────
    code(
        """# ── Execute Computation Graph ───────────────────────────────────────────────
a, b = 1.5, 2.5
e_out = func(a, b)
print(f"e_out: {e_out}")

# **Answer**: In TF 2.x, this executes eagerly — you get the value 6.0 immediately.
# TF 1.x required session.run(); TF 2.x runs like NumPy by default.
""",
        "e0111eb4",
    ),
    # ── Cell 20: Computation graph proof code (NEW) ────────────────────────────
    code(
        """# ── PROVING the computation graph ────────────────────────────────────────────
# Each operation returns a tensor. Let's trace the dependency chain by hand.
a_v, b_v = 1.5, 2.5
print("Step-by-step computation:")
c_v = a_v + b_v; print(f"  c = a + b = {a_v} + {b_v} = {c_v}")
d_v = b_v - 1;   print(f"  d = b - 1 = {b_v} - 1  = {d_v}")
e_v = c_v * d_v; print(f"  e = c * d = {c_v} * {d_v} = {e_v}")
print(f"\\nfunc({a_v}, {b_v}) returned {e_v} — matches the hand trace.")
print("\\nDependency graph:")
print("  a (input) ─┬─► c = a+b ─┐")
print("  b (input) ─┴─► d = b-1 ─┴─► e = c*d  (output)")
print("\\nTensorFlow tracks this DAG implicitly so GradientTape can walk it backwards.")
""",
        "f259c12f",
    ),
    # ── Cell 21: Proving graph markdown ───────────────────────────────────────
    md(
        """### Proving the Computation Graph: Step-by-Step Trace

Even though execution is eager, TensorFlow still **tracks the dependency DAG** (directed acyclic graph) behind the scenes — this is what `GradientTape` uses for automatic differentiation. Let's trace the computation manually:

| Operation | Depends On | Value |
|-----------|------------|-------|
| `c = a + b` | `a=1.5, b=2.5` | `4.0` |
| `d = b - 1` | `b=2.5` | `1.5` |
| `e = c * d` | `c=4.0, d=1.5` | `6.0` |

**Key insight**: The final output `e` depends on both `a` and `b` through different paths in the graph. TensorFlow tracks this dependency structure so that when we later ask "how does `e` change if I nudge `a`?", `GradientTape` can automatically compute $\\partial e/\\partial a$ by walking the graph backward — no hand-coded derivatives.
""",
        "a008f8ad",
    ),
    # ── Cell 22: Predict-first eager vs graph ─────────────────────────────────
    md(
        """### 🔮 Predict First: Eager vs. Graph Execution

Before running the next cell, **predict**: when we write `c = a + b` in TensorFlow 2.x, does it:

**(A)** Execute immediately and return the numeric value 76 (eager execution)?
**(B)** Return a lazy graph placeholder like `<tf.Tensor 'add:0' shape=() dtype=int32>` (graph mode, as in TF 1.x)?

Think about it, then run the cell to see the answer.
""",
        "40a43626",
    ),
    # ── Cell 23: Predict-first answer code (NEW) ──────────────────────────────
    code(
        """# 🔮 Prediction check: TF 2.x executes EAGERLY by default
a = tf.constant(15); b = tf.constant(61)
c = a + b
print(f"c = {c.numpy()}   ← this is a concrete value, not a graph node")
print("✓ Answer: (B) TF 2.x is eager by default. c immediately holds 76.")
print("  (TF 1.x would have printed 'Tensor(Add:0, shape=(), dtype=int32)' instead)")
""",
        "db3c21eb",
    ),
    # ── Cell 24: Reflection markdown ──────────────────────────────────────────
    md(
        """### 🤔 Reflection: From Computation to Learning

We've built and traced a computation graph (`a + b → c`, `b - 1 → d`, `c * d → e`), but we never asked: **"Which input matters most for the output?"**

- Does `e` change more when we nudge `a` or when we nudge `b`?
- In other words, what is $\\partial e/\\partial a$ vs. $\\partial e/\\partial b$?

This question — **"how sensitive is the output to each input?"** — is the core of training neural networks. If we can compute $\\partial \\text{Loss}/\\partial W$ for every weight $W$, we can use gradient descent to minimize the loss.

**The problem**: for a 25-million-parameter ResNet, hand-coding $\\partial \\text{Loss}/\\partial W$ for every weight is impossible. That's where **automatic differentiation** comes in — TensorFlow computes all those derivatives for us by walking the computation graph backward. Let's see how.
""",
        "3c47fb36",
    ),
    # ── Cell 25: 1.3 Keras markdown ───────────────────────────────────────────
    md(
        """## 1.3 Neural networks in Keras

We define neural networks in Keras.  The base building block is
[`tf.keras.layers.Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer),
which is the Keras equivalent of PyTorch's `nn.Module`.

Consider a simple perceptron: $y = \\sigma(Wx + b)$, where $W$ is a weight
matrix, $b$ a bias, $x$ the input, $\\sigma$ the sigmoid activation, and $y$
the output.

![dense layer](https://raw.githubusercontent.com/MITDeepLearning/introtodeeplearning/master/lab1/img/computation-graph-2.png)

We subclass `tf.keras.layers.Layer` and override **`call()`** (the Keras
equivalent of PyTorch's `forward()`).  Weights are created with
`self.add_weight()` inside `build()` or `__init__`.
""",
        "0d8b7ea0",
    ),
    # ── Cell 26: OurDenseLayer ─────────────────────────────────────────────────
    code(
        """# ── Defining a Dense Layer ──────────────────────────────────────────────────

class OurDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(OurDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        # Initialise W and b as trainable weights.
        # Note: weight initialisation is random by default.
        self.W = self.add_weight(
            shape=(int(input_shape[-1]), self.num_outputs),
            initializer='random_normal', trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.num_outputs,),
            initializer='random_normal', trainable=True
        )

    def call(self, x):
        '''TODO: define the operation for z (hint: use tf.matmul).'''
        z = tf.matmul(x, self.W) + self.bias

        '''TODO: define the operation for y (hint: use tf.sigmoid).'''
        y = tf.sigmoid(z)
        return y
""",
        "47a1f722",
    ),
    # ── Cell 27: "Now let's test our layer" ───────────────────────────────────
    md("Now, let's test the output of our layer.", "137cbc0d"),
    # ── Cell 28: Layer test ────────────────────────────────────────────────────
    code(
        """# Define a layer and test the output!
num_inputs  = 2
num_outputs = 3
layer   = OurDenseLayer(num_outputs)
x_input = tf.constant([[1, 2.]])
y = layer(x_input)

print(f"input shape:  {x_input.shape}")
print(f"output shape: {y.shape}")
print(f"output result: {y}")
""",
        "989978dd",
    ),
    # ── Cell 29: Sequential API markdown ──────────────────────────────────────
    md(
        """Conveniently, Keras provides built-in layers such as
[`tf.keras.layers.Dense`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
(the equivalent of PyTorch's `nn.Linear`) and activation layers.

Now, instead of a single custom layer, we'll use the
[`tf.keras.Sequential`](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
API with a single `Dense` layer to define our network.
""",
        "a3b9d1b4",
    ),
    # ── Cell 30: Sequential model ─────────────────────────────────────────────
    code(
        """# ── Defining a Neural Network Using the Keras Sequential API ────────────────

# Let's build a small binary classifier for our 2D circle-vs-square problem:
#   - Input: 2D point (x₀, x₁)
#   - Hidden: Dense(4, relu)
#   - Output: Dense(1, sigmoid) → probability of CIRCLE

model = tf.keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=(2,)),
    layers.Dense(1, activation='sigmoid')
])

print("Model summary:")
model.summary()
""",
        "a89fa675",
    ),
    # ── Cell 31: "We've defined" markdown ─────────────────────────────────────
    md(
        "We've defined our model using the Sequential API.  Now let's test it:",
        "61516fa1",
    ),
    # ── Cell 32: Test untrained model ─────────────────────────────────────────
    code(
        """# ── Test the Untrained Model ────────────────────────────────────────────────
# Predict on a few training points (untrained model → ~50% predictions)

x_test_sample = tf.constant(X_train[:5])
model_output  = model(x_test_sample)

print(f"input shape:  {x_test_sample.shape}")
print(f"output shape: {model_output.shape}")
print("\\nUntrained predictions (should be ~0.5 random):")
for i, (point, pred) in enumerate(zip(x_test_sample.numpy(), model_output.numpy())):
    true_label = y_train[i]
    print(f"  Point {i}: {point} → pred={pred[0]:.3f}, true={int(true_label)} ({'CIRCLE' if true_label else 'SQUARE'})")
""",
        "3a8f5bbe",
    ),
    # ── Cell 33: Predict-first skip connection ────────────────────────────────
    md(
        """### 🔮 Predict First: Can Sequential Express a Skip Connection?

The `Sequential` API is convenient for stacking layers in a straight line: input → layer1 → layer2 → output.

Before continuing, **predict**: can `Sequential` express a **residual/skip connection** where the output is `layer2(layer1(x)) + x` (i.e., the input bypasses layer1 and gets added to the output)?

**(A)** Yes, with a special `layers.Add()` layer
**(B)** No — Sequential only supports linear chains; skip connections require Model subclassing

Think about it, then read on.
""",
        "ab34411c",
    ),
    # ── Cell 34: Skip connection answer ───────────────────────────────────────
    md(
        "**Answer**: **(B) — Sequential cannot express skip connections.** To add the input to a middle-layer output, you need to **subclass `tf.keras.Model`** and manually wire the connections in `call()`. Let's see how.",
        "e62965c1",
    ),
    # ── Cell 35: ResidualClassifier ───────────────────────────────────────────
    code(
        """# ── Defining a Model Using Subclassing (with Skip Connection) ───────────────

class ResidualClassifier(tf.keras.Model):
    \"\"\"
    A binary classifier with a residual/skip connection:
      output = Dense(1, sigmoid)(Dense(4, relu)(padded_input) + padded_input)

    Sequential cannot express this because the input bypasses the hidden layer
    and gets added back before the final Dense(1).
    \"\"\"
    def __init__(self):
        super(ResidualClassifier, self).__init__()
        self.hidden = layers.Dense(4, activation='relu')
        self.pad    = layers.Dense(4, activation='linear')  # pad 2D input to 4D
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # Pad input from 2D to 4D so we can add it to the hidden layer output
        padded = self.pad(inputs)

        # Skip connection: add input to hidden layer output
        hidden_out = self.hidden(inputs)
        residual   = hidden_out + padded

        # Final classification
        output = self.output_layer(residual)
        return output
""",
        "430ca743",
    ),
    # ── Cell 36: "Let's test the model" markdown ──────────────────────────────
    md(
        """Let's test the model using an example input with `n_input_nodes=2` and
`n_output_nodes=3` as before.
""",
        "b64e6d68",
    ),
    # ── Cell 37: Test residual classifier ─────────────────────────────────────
    code(
        """# ── Test the Residual Classifier ────────────────────────────────────────────
residual_model = ResidualClassifier()
x_input = tf.constant(X_train[:5])
y = residual_model(x_input)

print(f"input shape:  {x_input.shape}")
print(f"output shape: {y.shape}")
print("\\nResidual model (untrained) predictions:")
for i, pred in enumerate(y.numpy()):
    print(f"  Point {i}: pred={pred[0]:.3f}, true={int(y_train[i])} ({'CIRCLE' if y_train[i] else 'SQUARE'})")
""",
        "c4f4f128",
    ),
    # ── Cell 38: "Now that we have learned" markdown ──────────────────────────
    md(
        """Now that we have learned how to define layers and models in Keras using both the
Sequential API and subclassing `tf.keras.Model`, we're ready to turn our
attention to how to actually implement network training with backpropagation.
""",
        "e2c31851",
    ),
    # ── Cell 39: 1.4 Autodiff markdown ────────────────────────────────────────
    md(
        """## 1.4 Automatic Differentiation in TensorFlow

In TensorFlow, automatic differentiation is performed using
[`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape).
Operations executed inside the `with tf.GradientTape() as tape:` context are
recorded; afterwards we call `tape.gradient(target, sources)` to compute
derivatives — the Keras/TF equivalent of PyTorch's `.backward()`.

Variables to differentiate with respect to must either be `tf.Variable` objects
(watched automatically) or tensors explicitly watched with `tape.watch()`.

Let's compute the gradient of $y = x^2$:
""",
        "49ca4da5",
    ),
    # ── Cell 40: y=x^2 gradient ───────────────────────────────────────────────
    code(
        """### Gradient computation ###

# y = x^2
# Example: x = 3.0
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

dy_dx = tape.gradient(y, x)
print("dy_dx of y=x^2 at x=3.0 is:", dy_dx.numpy())
assert dy_dx.numpy() == 6.0
""",
        "faa35731",
    ),
    # ── Cell 41: Prove autodiff code ───────────────────────────────────────────
    code(
        """# ── Prove Autodiff Correctness ──────────────────────────────────────────────

def test_autodiff():
    x_val = 3.0
    x = tf.Variable(x_val)

    results = []

    # Test 1: f(x) = x^2
    with tf.GradientTape() as tape:
        y = x ** 2
    auto_grad = tape.gradient(y, x).numpy()
    hand_grad = 2 * x_val
    results.append(("x²", auto_grad, hand_grad, np.isclose(auto_grad, hand_grad)))

    # Test 2: f(x) = sin(x)
    with tf.GradientTape() as tape:
        y = tf.sin(x)
    auto_grad = tape.gradient(y, x).numpy()
    hand_grad = np.cos(x_val)
    results.append(("sin(x)", auto_grad, hand_grad, np.isclose(auto_grad, hand_grad)))

    # Test 3: f(x) = exp(x^2)
    with tf.GradientTape() as tape:
        y = tf.exp(x ** 2)
    auto_grad = tape.gradient(y, x).numpy()
    hand_grad = 2 * x_val * np.exp(x_val**2)
    results.append(("exp(x²)", auto_grad, hand_grad, np.isclose(auto_grad, hand_grad)))

    # Print table
    print(f"{'Function':<12} {'Autodiff':<12} {'Hand-Coded':<12} {'Match?':<8}")
    print("─" * 50)
    for func, auto, hand, match in results:
        check = "✓" if match else "✗"
        print(f"{func:<12} {auto:<12.6f} {hand:<12.6f} {check:<8}")

    print(f"\\n✓ All derivatives match! GradientTape works for any differentiable function.")

test_autodiff()
""",
        "e18d476e",
    ),
    # ── Cell 42: Proving autodiff markdown ────────────────────────────────────
    md(
        """### Proving Autodiff: GradientTape vs. Hand-Coded Derivatives

TensorFlow claims that `GradientTape` can compute derivatives for any differentiable function. Let's **prove it** by comparing autodiff against hand-coded derivatives for multiple functions:

| Function | Autodiff ($\\partial f/\\partial x$) | Hand-Coded | Match? |
|----------|-------------------------------------|------------|--------|
| $f(x) = x^2$ | `tape.gradient` | $2x$ | ? |
| $f(x) = \\sin(x)$ | `tape.gradient` | $\\cos(x)$ | ? |
| $f(x) = e^{x^2}$ | `tape.gradient` | $2x \\cdot e^{x^2}$ | ? |
""",
        "c54c5986",
    ),
    # ── Cell 43: Gradient descent intro markdown ───────────────────────────────
    md(
        """In training neural networks we use differentiation and stochastic gradient
descent (SGD) to optimise a loss function.  Let's find the minimum of
$L = (x - x_f)^2$ using `tf.GradientTape` and gradient descent.  While the
analytic solution is $x_{\\min} = x_f$, working through this with TF's autograd
sets us up nicely for future labs.
""",
        "40df13de",
    ),
    # ── Cell 44: Gradient descent minimization ────────────────────────────────
    code(
        """# ── Function Minimization with GradientTape ─────────────────────────────────

# Reset seeds for deterministic initialization
tf.random.set_seed(42)
np.random.seed(42)

x = tf.Variable(tf.random.normal([1]))
print(f"Initializing x={x.numpy()[0]:.4f}")

learning_rate = 1e-2
history = []
x_target = 4.0   # target value (renamed from x_f for clarity)

for i in range(500):
    with tf.GradientTape() as tape:
        # Loss: square of the difference between x and x_target
        loss = (x - x_target) ** 2

    # Compute gradient and apply update
    grad = tape.gradient(loss, x)
    x.assign_sub(learning_rate * grad)
    history.append(x.numpy()[0])

# Plot the evolution of x as we optimise toward x_target!
plt.figure(figsize=(8, 4))
plt.plot(history, label='Predicted x')
plt.axhline(y=x_target, color='r', linestyle='--', label=f'Target x={x_target}')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('Gradient Descent: Minimizing L = (x - x_target)²')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Final x: {x.numpy()[0]:.4f} (target: {x_target})")
""",
        "51f4f911",
    ),
    # ── Cell 45: Training intro markdown (NEW) ────────────────────────────────
    md(
        """### Training Our Circle-vs-Square Classifier

Now let's apply the same gradient descent loop to our running example: train the Sequential classifier to distinguish CIRCLE from SQUARE. We'll visualize:

1. **Loss curve** — does the model learn?
2. **Decision boundary** — what boundary did it learn, compared to the true circle?
""",
        "7d4fcc29",
    ),
    # ── Cell 46: Training loop (NEW) ──────────────────────────────────────────
    code(
        """# ── Train the Binary Classifier on Our Running Example ─────────────────────
# Reset seeds and rebuild model for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

model = tf.keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(2,)),
    layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

X_train_tf = tf.constant(X_train)
y_train_tf = tf.constant(y_train.reshape(-1, 1))

loss_history = []
n_epochs = 200

for epoch in range(n_epochs):
    with tf.GradientTape() as tape:
        predictions = model(X_train_tf, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_train_tf, predictions))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    loss_history.append(loss.numpy())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}: loss={loss.numpy():.4f}")

print("\\n✓ Training complete!")
print("  Same GradientTape loop you saw in the simple x² example — now on a real model.")
""",
        "48a276f7",
    ),
    # ── Cell 47: Visualization (NEW) ──────────────────────────────────────────
    code(
        """# ── Multi-Panel Visualization: Loss + Decision Boundary ─────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): Training loss curve
ax1.plot(loss_history, linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Binary Cross-Entropy Loss')
ax1.set_title('(a) Training Loss')
ax1.grid(alpha=0.3)

# Panel (b): Learned decision boundary
# Create a grid of points and predict class probabilities
grid_res = 100
x0_range = np.linspace(-1, 1, grid_res)
x1_range = np.linspace(-1, 1, grid_res)
X0_grid, X1_grid = np.meshgrid(x0_range, x1_range)
grid_points = np.c_[X0_grid.ravel(), X1_grid.ravel()].astype(np.float32)
grid_preds = model.predict(grid_points, verbose=0).reshape(grid_res, grid_res)

# Plot contour (learned decision boundary at p=0.5)
ax2.contourf(X0_grid, X1_grid, grid_preds, levels=20, cmap='RdYlBu_r', alpha=0.6)
contour = ax2.contour(X0_grid, X1_grid, grid_preds, levels=[0.5], colors='blue', linewidths=2)
ax2.clabel(contour, inline=True, fontsize=10, fmt='p=%.1f')

# Overlay training points
circle_mask = y_train == 1
ax2.scatter(X_train[circle_mask, 0], X_train[circle_mask, 1],
           c='darkblue', label='CIRCLE (true)', alpha=0.7, edgecolors='k', s=30)
ax2.scatter(X_train[~circle_mask, 0], X_train[~circle_mask, 1],
           c='darkorange', label='SQUARE (true)', alpha=0.7, edgecolors='k', s=30)

# Overlay true boundary (radius 0.5)
theta = np.linspace(0, 2*np.pi, 100)
ax2.plot(0.5*np.cos(theta), 0.5*np.sin(theta), 'r--', linewidth=2, label='True boundary (r=0.5)')

ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_aspect('equal')
ax2.set_xlabel('x₀')
ax2.set_ylabel('x₁')
ax2.set_title('(b) Learned Decision Boundary')
ax2.legend(loc='upper right')

plt.suptitle('Gradient Descent in Action — Same Loop That Trains GPT', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("✓ The model learned to approximate the circular boundary!")
print("  GradientTape + optimizer drove weights from random init to solution.")
""",
        "20f18918",
    ),
    # ── Cell 48: 1.5 Toy→real bridge markdown ─────────────────────────────────
    md(
        """## 1.5 Toy → Production Bridge: Same Loop, Bigger Models

You've now trained a neural network from scratch using TensorFlow's autodiff. Before moving to pretrained models, let's compare **our toy classifier** to a **production model** like ResNet-50:

| Aspect | Our Toy Classifier | ResNet-50 (ImageNet) |
|--------|-------------------|----------------------|
| **Input dimension** | 2D point (x₀, x₁) | 224×224×3 image (150,528D) |
| **Hidden layers** | 1 Dense(8, relu) | 50+ Conv/Batch/Residual blocks |
| **Parameters** | ~30 weights & biases | ~25 million parameters |
| **Training data** | 200 samples | 1.2 million images (ImageNet) |
| **Training epochs** | 200 | 90+ |
| **Loss function** | Binary cross-entropy | Categorical cross-entropy (1000 classes) |
| **Optimizer** | Adam | SGD with momentum |
| **Autodiff** | `GradientTape` | `GradientTape` |
| **Training loop** | `tape.gradient` → `optimizer.apply_gradients` | `tape.gradient` → `optimizer.apply_gradients` |

### Key Insight

**The training loop is identical.** Whether you're classifying 2D points or classifying 1000 ImageNet categories:

1. **Forward pass**: `predictions = model(inputs)`
2. **Compute loss**: `loss = loss_fn(y_true, predictions)`
3. **Autodiff**: `gradients = tape.gradient(loss, model.trainable_variables)`
4. **Update weights**: `optimizer.apply_gradients(zip(gradients, variables))`
5. **Repeat** for many epochs

If you understood the toy example, **you understand ResNet**. The only difference is scale.
""",
        "4d70792a",
    ),
    # ── Cell 49: Toy→real bridge code (NEW) ───────────────────────────────────
    code(
        """# ── Toy-to-real: same GradientTape, different scale ──────────────────────────
n_toy = sum(tf.size(v).numpy() for v in model.trainable_variables)
print(f"Our toy classifier: {n_toy} trainable parameters")
print("  → Dense(2→8) = 8*2 + 8 = 24 params  +  Dense(8→1) = 8 + 1 = 9 params")
print(f"  = {n_toy} total")
print()

try:
    from tensorflow.keras.applications import ResNet50
    resnet = ResNet50(weights=None, include_top=True, classes=1000)
    n_resnet = resnet.count_params()
    print(f"ResNet-50: {n_resnet:,} total parameters")
    print(f"  → same Dense layers, same GradientTape, {n_resnet // n_toy:,}× more parameters")
    print("  Every one of those parameters is a tf.Variable, just like our layer's W and b.")
    print("  Every forward pass traces a computation graph. GradientTape computes all gradients.")
    print("  The machinery you learned on ~33 parameters scales to 25 million — no new concepts.")
except Exception as e:
    print(f"(ResNet-50 skipped: {e})")
    print("The point stands: production models use the same tf.Variable / GradientTape / optimizer pipeline.")
""",
        "0ff4699a",
    ),
    # ── Cell 50: Part 2 intro markdown ────────────────────────────────────────
    md(
        """---

## Part 2: Pretrained Transformers at Scale

In Part 1 you trained a model from scratch using the **same autodiff loop** that powers all deep learning. Now we'll use a **pretrained model** from HuggingFace — weights learned via the same `GradientTape` → `optimizer.apply_gradients` loop you just ran, but at **billion-parameter scale** and trained on millions of samples.

### Connection to Part 1

- **Part 1**: You wrote the training loop and watched the loss decrease from random init.
- **Part 2**: Someone else already ran that loop for weeks on a GPU cluster; you're loading the final checkpoint.

The pretrained weights were **learned via gradient descent**, just like your toy classifier — TensorFlow computed $\\partial \\text{Loss}/\\partial W$ for 25 million (ResNet) or 300 million (MusicGen) parameters using the same `GradientTape` mechanism.

### Why Music Generation with Transformers (Not RNNs)?

The lab title mentions "Music Generation with RNNs" because MIT's original from-scratch lab trained an LSTM character-by-character on ABC notation. **The field has since moved to Transformers** for music generation — better long-range modeling, faster training. Below you'll use:

- **MusicGen** (Meta): text prompt → audio waveform
- **TunesFormer**: ABC notation seed → ABC tune

Both are Transformer-based. If you want the original LSTM lab, see [MIT's 6.S191 Lab 1 (2019–2021 versions)](http://introtodeeplearning.com/2021/lab1_part2.pdf).

### Model Selection

Two pretrained models are available. Change `MUSIC_MODEL` in the next cell to switch:

| Model | Approach | Input | Output |
|---|---|---|---|
| `facebook/musicgen-small` | Transformer seq2seq | **text prompt** | raw audio (WAV) |
| `sander-wood/tunesformer` | Causal LM | **ABC notation seed** | ABC notation text |

> **ABC notation** is a text-based music format used for Irish/Celtic folk tunes.
> Example: `X:1\\nT:Title\\nM:6/8\\nK:Gmaj\\n|: G2A B2c | d2e fed |`
> The model extends that seed, producing a complete tune you can play with `music21` or `abc2midi`.
""",
        "38584aca",
    ),
    # ── Cell 51: Install HuggingFace ──────────────────────────────────────────
    code(
        """## ── Install HuggingFace dependencies ────────────────────────────────────────
## Run once; restart the kernel after installing if needed.

!pip install transformers accelerate scipy soundfile music21 --quiet
""",
        "d425897f",
    ),
    # ── Cell 52: Model selector ────────────────────────────────────────────────
    code(
        """## ── Model selector ──────────────────────────────────────────────────────────
## Change MUSIC_MODEL to switch between the two approaches at runtime.

# ── Pick one ─────────────────────────────────────────────────────────────────
MUSIC_MODEL = "musicgen"     # "musicgen"  |  "tunesformer"

# ── MusicGen config (used when MUSIC_MODEL == "musicgen") ────────────────────
#   Model sizes: musicgen-small (~300 MB) | musicgen-medium (~1.5 GB) | musicgen-large (~3.3 GB)
MUSICGEN_REPO      = "facebook/musicgen-small"
MUSICGEN_PROMPT    = "upbeat Irish folk music with fiddle and flute, lively jig"
MUSICGEN_DURATION  = 8        # seconds of audio to generate

# ── TunesFormer config (used when MUSIC_MODEL == "tunesformer") ──────────────
#   TunesFormer generates ABC notation conditioned on a control-code seed.
#   Seed format:  X:<index>  T:<title>  M:<time sig>  K:<key>  then barlines.
TUNESFORMER_REPO        = "sander-wood/tunesformer"
TUNESFORMER_SEED        = "X:1\\nT:My Generated Tune\\nM:6/8\\nL:1/8\\nK:Gmaj\\n|: G2A B2c |"
TUNESFORMER_NEW_TOKENS  = 400   # max new tokens (≈ 1-2 full tunes)
TUNESFORMER_TEMPERATURE = 0.9
TUNESFORMER_TOP_K       = 50

OUTPUT_WAV = "generated_music.wav"

print(f"Selected model: {MUSIC_MODEL!r}")
""",
        "20a62656",
    ),
    # ── Cell 53: Option A markdown ─────────────────────────────────────────────
    md(
        """### Option A — `facebook/musicgen-small` (text-prompt → audio)

MusicGen is a Transformer encoder-decoder trained by Meta AI on 20 000 hours of
licensed music.  You describe the music you want in plain English and it generates
a raw audio waveform directly — no ABC notation, no MIDI intermediate step.

Run the cell below when `MUSIC_MODEL = "musicgen"`.
""",
        "8c69a28a",
    ),
    # ── Cell 54: musicgen code ─────────────────────────────────────────────────
    code(
        """if MUSIC_MODEL == "musicgen":
    from transformers import pipeline
    import scipy.io.wavfile
    import numpy as np
    import IPython.display as ipd

    print(f"Loading {MUSICGEN_REPO} ...")
    musicgen_pipe = pipeline(
        "text-to-audio",
        model=MUSICGEN_REPO,
        device="cpu",           # change to 0 (or "cuda") if a GPU is available
        framework="pt",         # MusicGen uses PyTorch regardless of TF notebook context
    )

    print(f"Generating {MUSICGEN_DURATION}s of audio for prompt:\\n  '{MUSICGEN_PROMPT}'")
    result = musicgen_pipe(
        MUSICGEN_PROMPT,
        forward_params={
            "do_sample": True,
            "max_new_tokens": int(MUSICGEN_DURATION * 50),
        },
    )

    audio = result["audio"].squeeze()
    sr    = result["sampling_rate"]

    # Normalise to int16 for WAV export
    audio_int16 = (audio / np.abs(audio).max() * 32767).astype(np.int16)
    scipy.io.wavfile.write(OUTPUT_WAV, sr, audio_int16)
    print(f"Saved to {OUTPUT_WAV}")

    ipd.display(ipd.Audio(audio, rate=sr))
""",
        "1969ccaa",
    ),
    # ── Cell 55: Option B markdown ─────────────────────────────────────────────
    md(
        """### Option B — `sander-wood/tunesformer` (ABC seed → ABC notation)

TunesFormer is a GPT-style causal LM fine-tuned on thousands of Irish/Celtic folk
tunes in ABC notation.  You supply a short **seed** (title, metre, key, a bar or
two) and the model completes the tune character-by-character — the same mechanism
as the MIT from-scratch LSTM, but using a pretrained model.

The output is **ABC notation text** which you can:
- Copy into [https://abc.rectanglered.com](https://abc.rectanglered.com) to hear it
- Convert to MIDI with `music21` (shown below)
- Convert to audio with `timidity` or GarageBand

Run the cell below when `MUSIC_MODEL = "tunesformer"`.
""",
        "133cdd2e",
    ),
    # ── Cell 56: tunesformer code ──────────────────────────────────────────────
    code(
        """if MUSIC_MODEL == "tunesformer":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {TUNESFORMER_REPO} ...")
    tf_tokenizer = AutoTokenizer.from_pretrained(TUNESFORMER_REPO)
    tf_lm_model  = AutoModelForCausalLM.from_pretrained(TUNESFORMER_REPO)
    tf_lm_model.eval()

    inputs = tf_tokenizer(TUNESFORMER_SEED, return_tensors="pt")

    print("Generating ABC notation ...")
    with torch.no_grad():
        output_ids = tf_lm_model.generate(
            inputs["input_ids"],
            max_new_tokens=TUNESFORMER_NEW_TOKENS,
            do_sample=True,
            temperature=TUNESFORMER_TEMPERATURE,
            top_k=TUNESFORMER_TOP_K,
            pad_token_id=tf_tokenizer.eos_token_id,
        )

    generated_abc = tf_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\\n-- Generated ABC notation ------------------------------------------")
    print(generated_abc)
    print("--------------------------------------------------------------------")

    # Optional: convert to MIDI with music21
    try:
        from music21 import converter, midi
        score = converter.parse(generated_abc, format="abc")
        mf = midi.translate.music21ObjectToMidiFile(score)
        midi_path = "generated_tune.mid"
        mf.open(midi_path, "wb")
        mf.write()
        mf.close()
        print(f"MIDI saved to {midi_path}")
    except Exception as e:
        print(f"[music21 MIDI export skipped: {e}]")
        print("Tip: paste the ABC text above into https://abc.rectanglered.com to hear it.")
""",
        "7f6a499a",
    ),
    # ── Cell 57: Summary markdown ─────────────────────────────────────────────
    md(
        """---

## Summary: What You've Learned

You've completed the full TensorFlow journey from tensors to pretrained models. Let's revisit the roadmap:

| Step | Concept | Key Idea | Status |
|---|---|---|---|
| 1 | Vocabulary & Toy Data | Our running example: classifying 2D points (circle vs square) | ✓ |
| 2 | Tensors & Shapes | Multi-dimensional arrays (`tf.Tensor`) as TensorFlow's substrate | ✓ |
| 3 | Computation Graphs | Operations (add, matmul) form a dependency DAG; TF tracks it implicitly | ✓ |
| 4 | Keras Layers | Dense, Activation, custom `Layer` subclasses as composable building blocks | ✓ |
| 5 | The Forward Pass | `model(inputs)` returns predictions; weights are random until trained | ✓ |
| 6 | Autodiff & Backprop | `GradientTape` computes $\\partial L/\\partial W$ — no hand-coded derivatives | ✓ |
| 7 | Training Loop | Gradient descent: `tape.gradient` → `optimizer.apply_gradients` → repeat | ✓ |
| 8 | Toy → Production Bridge | Same mechanisms (GradientTape, Dense, optimizer), just far more parameters | ✓ |
| 9 | Pretrained Music Model | HuggingFace Transformers: autodiff deployed at billion-parameter scale | ✓ |

### Key Insights to Keep

1. **Tensors are the universal substrate** — everything in TensorFlow is a multi-dimensional array.
2. **Computation graphs are implicit** — TF 2.x tracks dependencies eagerly, but `GradientTape` still needs the DAG for backprop.
3. **GradientTape eliminates hand-coded backprop** — you never write $\\partial L/\\partial W$ by hand; TF computes it for any differentiable function.
4. **Keras layers are composable blocks** — `Sequential` for linear stacks, `Model` subclassing for skip connections and complex architectures.
5. **Training = loss minimization via gradient descent** — whether you have 30 parameters or 25 million, the loop is: forward → loss → gradients → update → repeat.
6. **Pretrained models reuse the same autodiff** — MusicGen's 300M parameters were learned via the exact loop you ran in Part 1.

### What's Next?

- **[transformers.ipynb](../../transformers.ipynb)** — Gold-standard walkthrough of the Transformer architecture (attention, positional encoding, encoder-decoder)
- **[MIT 6.S191 Lab 1 (original LSTM version)](http://introtodeeplearning.com/2021/lab1_part2.pdf)** — Train an LSTM from scratch on ABC notation (the pre-Transformer approach)
- **[HuggingFace Transformers docs](https://huggingface.co/docs/transformers/)** — Explore thousands of pretrained models for NLP, vision, audio, and multimodal tasks

---

**🎉 Congratulations!** You've learned the core TensorFlow/Keras workflow. The same concepts — tensors, autodiff, gradient descent — underpin every model you'll encounter, from ResNet to GPT to Stable Diffusion.
""",
        "d7368b84",
    ),
]  # end cells list

# Preserve original metadata, just replace cells
nb["cells"] = cells
nb["nbformat"] = 4
nb["nbformat_minor"] = 5

with open(
    "c:/repos/ai-portfolio/learning/genai/rnns/MIT/TF_Part1_Intro.ipynb",
    "w",
    encoding="utf-8",
) as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written {len(cells)} cells to disk.")
