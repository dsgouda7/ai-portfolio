# Pre-Requisites — Mathematical Foundations for ML

> The on-ramp to the rest of the repo. If `notes/01-ml/` opens with $\hat{y} = wx + b$ and the symbols already feel slippery, start here. Seven short chapters that walk from the equation of a line to the matrix chain rule — the mathematics that every later chapter silently assumes.

---

## Who this is for

- You have written Python and seen basic algebra.
- You have *heard* of derivatives, matrices, and gradients but they do not feel like tools you reach for.
- You want to read an ML paper and have $\nabla_\theta \mathcal{L}$ mean something concrete.

If that is you, read these seven chapters in order, run every notebook. If not, skip to [`notes/01-ml/`](../01-ml/README.md) and come back only when a symbol bites.

---

## The running thread — "The Perfect Knuckleball Free Kick"

Every chapter uses the *same* real-world problem: a football (soccer) striker lines up a direct free kick 20 m from goal, aiming to clear a defensive wall and dip the ball under the 2.44 m crossbar. Because it's struck as a **knuckleball** — almost zero spin, Juninho/Pirlo/Ronaldo style — the ball's path is dominated by gravity alone, so we can model it as a clean 2D parabola and leave the Magnus curve out of the story entirely.

### **The Grand Challenge**

**Can we score this goal?** To succeed, the ball must satisfy **THREE constraints simultaneously**:

1. **🧱 Wall Clearance**: At the wall position (9.15 m horizontal distance, ~0.6s flight time), the ball must be **above 1.8 m** (wall height)
2. **🎯 Crossbar Clearance**: At the goal line (20 m horizontal distance, ~1.2s flight time), the ball must be **below 2.44 m** (crossbar height)
3. **⚡ Keeper-Beating Speed**: The ball must arrive fast enough (or with short enough flight time) that the goalkeeper cannot react

These are **competing constraints** — a high launch angle clears the wall easily but might sail over the crossbar. A low angle beats the keeper but hits the wall. Finding the right launch speed v₀ and angle θ is a **constrained multi-objective optimization problem** — exactly what real ML does!

This thread was picked deliberately — projectile motion is the problem that *forced* Newton and Leibniz to invent calculus in the 1660s–80s, so the mathematics and the example grew up together. Each chapter gives you ONE new mathematical tool to solve ONE piece of the challenge.

### **Progressive Capability Unlock**

| Chapter | Math Tool | What It Unlocks | Challenge Progress |
|---|---|---|---|
| Ch.1 Linear Algebra | Lines, slopes, intercepts | Predict height during first 0.1s (linear approx) | ❌ Can't model full curve yet |
| Ch.2 Non-linear Algebra | Polynomials, parabolas | Model full trajectory h(t) = v₀ᵧt - ½gt² | ❌ Can't find peak/crossings yet |
| Ch.3 Calculus Intro | Derivatives, rate of change | Find apex (h'=0), compute wall/goal heights | ✅ **Can check constraints 1 & 2!** |
| Ch.4 Small Steps | Gradient descent, iteration | Optimize ONE parameter (angle for max range) | ✅ Can find best single-variable solution |
| Ch.5 Matrices | Multi-variable systems | Handle v₀ AND θ AND wind simultaneously | ❌ Can't optimize multi-dim yet |
| Ch.6 Chain Rule | Gradients, Jacobians | Optimize MULTIPLE parameters at once | ✅ **Can solve full challenge!** |
| Ch.7 Probability | Distributions, noise | Handle striker fatigue, ball variance | ✅ Can reason about success rate |

**The narrative arc**: We move from "can we predict?" (Ch.1-3) → "can we optimize one thing?" (Ch.4) → "can we optimize everything?" (Ch.5-6) → "can we handle uncertainty?" (Ch.7).

---

## Roadmap — chapters and what each delivers

### [Ch.1 — Linear Algebra: Lines, Weights, Biases](ch01_linear_algebra/README.md)
**Mental model:** a line is a two-parameter object. The parameters are a weight (slope) and a bias (offset). Vectors are lists of numbers; the dot product is a weighted sum.
**Artifact:** an interactive plot where you drag $w$ and $b$ and watch the line re-orient, plus a scatter of (time, height) samples the reader fits *by hand*.

### [Ch.2 — Non-linear Algebra: Polynomials and the Feature-Expansion Trick](ch02_nonlinear_algebra/README.md)
**Mental model:** $ax^2 + bx + c$ is non-linear in $x$ but linear in $(a, b, c)$. Substitute $x_1 = x^2, x_2 = x$ and you've turned a curve into a plane in 2-D feature space. This single trick is how "linear" models fit non-linear data.
**Artifact:** slider-driven parabola fitting a real projectile, plus a 3-D view of the same parabola as a flat plane in $(x_1, x_2, y)$.

### [Ch.3 — Calculus: Derivatives and Integrals from Scratch](ch03_calculus_intro/README.md)
**Mental model:** a derivative is the slope of a curve at one point; an integral is the area under it. Both are limits — a secant collapsing into a tangent, rectangles shrinking to the curve. The Fundamental Theorem says they're inverses of each other.
**Artifact:** a secant-to-tangent animation and a Riemann-sum accumulator on the free-kick trajectory.

### [Ch.4 — Small Steps on a Curve](ch04_small_steps/README.md)
**Mental model:** when you can't solve $f'(x) = 0$ analytically, walk downhill. The update $x \leftarrow x - \eta f'(x)$ converges if the step size is right — and if the landscape has only one valley. This is gradient descent one dimension early, with all the warts (step-size tuning, non-convexity, local optima) visible.
**Artifact:** start-angle and step-size sliders on a long-goal-kick range curve, plus a basin-of-attraction map on a wind-affected non-convex version.

### [Ch.5 — Matrices, Linear Systems, and Matrix Calculus](ch05_matrices/README.md)
**Mental model:** a matrix is a linear map. $A\mathbf{x} = \mathbf{b}$ has three views — row (intersecting hyperplanes), column (weighted sum of columns), transformation (warp of space). Normal equations $\hat{\mathbf{w}} = (X^\top X)^{-1}X^\top \mathbf{y}$ are just high-dimensional line-fitting.
**Artifact:** a $2 \times 2$ matrix-warping widget with live determinant, plus the full free-kick parabola fitted in one `lstsq` call with physics constants read straight off the weight vector.

### [Ch.6 — Gradient + Matrix Chain Rule](ch06_gradient_chain_rule/README.md)
**Mental model:** the gradient $\nabla f$ packs every partial derivative into a vector that points uphill. The matrix chain rule $\nabla_\mathbf{x}(g \circ f) = J_f^\top \nabla g$ is the single equation behind every `.backward()` call in PyTorch — reverse-mode autodiff is just that product evaluated right-to-left.
**Artifact:** a 2-D descent widget with tunable $(\theta_1^{(0)}, \theta_2^{(0)}, \eta, \text{iters})$, a one-layer neural-net shape drill, finite-difference and PyTorch `autograd` cross-checks, plus a forward-vs-reverse-mode benchmark showing reverse mode's $d$-fold speed-up on deep stacks.

### [Ch.7 — Probability & Statistics](ch07_probability_statistics/README.md)
**Mental model:** expectation, variance, likelihood. Mean-squared error isn't a design choice — it's what Gaussian noise mathematically demands. Change the noise model and you change the loss: Gaussian→MSE, Laplace→MAE, Bernoulli→cross-entropy. Every supervised loss in ML Ch.15 falls out of this one principle.
**Artifact:** an interactive CLT widget (switch source distribution and batch size, watch the sample-mean histogram morph to a Gaussian), Gaussian MLE by closed form and grid search agreeing exactly, a confirmation that OLS on the free-kick parabola equals its Gaussian MLE, and a mean-vs-median robustness demo showing why swapping noise models swaps loss functions.

---

## Historical and chronological evolution

Reading the mathematics in the order it was *discovered* makes it stick. Every concept below solved a problem that had no prior answer — understanding *which* problem is half the intuition. **The detailed timeline now lives in each chapter's own prelude** — every chapter opens with a *"The story"* blockquote that names the mathematicians, dates, and problems behind the idea.

**Story arc in one paragraph.** Linear equations (al-Khwārizmī, c. 820) → curves (Descartes & Fermat, 1630s) → calculus that tames curves (Newton & Leibniz, 1665–1684) → matrices that pack many curves together (Cayley & Sylvester, 1850s) → gradients that navigate them (Cauchy, 1847; stochastic version Kiefer & Wolfowitz, 1951) → probability that accepts the noise (Kolmogorov, 1930s). The recursive matrix chain rule from Rumelhart, Hinton & Williams (1986) is where this Pre-Reqs book hands off to [ML Ch.5](../ml/03_neural_networks/ch03_backprop_optimisers). Every step addresses a specific problem that had no satisfactory prior answer. Keep that arc in your head as you read.

---

## Authoring conventions

See [AUTHORING_GUIDE.md](authoring-guide.md) for chapter folder layout, README/notebook style rules, section order, and the running-thread convention.

---

## External references

- **Jon Krohn — *Linear Algebra for Machine Learning* (YouTube course).** <https://www.jonkrohn.com/posts/2021/5/9/linear-algebra-for-machine-learning-complete-math-course-on-youtube> — primary video companion for Ch.1 and Ch.5. Free, paced for self-study, covers exactly the subset of linear algebra that appears in neural networks.
- **3Blue1Brown — *Essence of Linear Algebra* and *Essence of Calculus* (YouTube playlists).** The visual companion to Ch.1, Ch.3, and Ch.5. If a concept here still feels abstract, watch the matching 3B1B video.
- **Gilbert Strang — *Introduction to Linear Algebra* (MIT OCW 18.06).** The definitive long-form text for Ch.5.
- **Goodfellow, Bengio, Courville — *Deep Learning*, Chs. 2–4.** The gap-filler between Pre-Reqs and the ML book.

---

## Where to go next

Once Ch.1–7 feel comfortable, open [`notes/01-ml/01_regression/ch01_linear_regression/`](../01-ml/01_regression/ch01_linear_regression/README.md). The pointers run *forward* from this track — every Pre-Req chapter ends with a **Where This Reappears** section listing the ML chapters that reuse the material — so use those as your map back when an ML symbol feels unfamiliar.
