# Mathematical Foundations for ML

**Contents:** [one-line map of this directory](CONTENTS.md)



This chapter keeps the free-kick visual story as the conceptual owner of vectors, local change,
accumulation, matrix state updates, repeated updates, and uncertainty.

## Required Notebook

- [Mathematical Foundations for Machine Learning](math-foundations-for-ml.ipynb)

The chapter intentionally stops before framework tensor APIs. Continue to
[ML Contracts](../02-ml-basics/ml-basics.ipynb) for data roles, loss, metrics, thresholds, and
generalization.

## Setup

- Windows: `./setup.ps1`
- Linux or macOS: `bash ./setup.sh`

The committed notebook stays output-free; the static assets in `images/` preserve the visual
teaching evidence before execution.

## Required Bounded Intervention

In the notebook's **“Your turn — cost of poor consistency”** section, predict the effect of changing
only `typical_spread` from `8.0` to `1.0`, run both values, and record expected scores per 100 and
surprise penalty. Diagnose which metric moved and explain whether this bounded evidence supports
the claim that a tighter assumed kick distribution raises the probability of landing inside the
same legal window.
