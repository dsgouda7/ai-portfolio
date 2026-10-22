# ML Basics

**Start:** [ML Basics: A good fit is not yet a useful prediction](ml-basics.ipynb).
One SmartVal dataset separates fair tests, loss, metrics, thresholds, and generalization.
Theory and scikit-learn experiments are together in the notebook.

Read one listing as three features and one target, follow the train-only scaler into a
Ridge prediction, and compare absolute versus squared misses. Then hold probabilities
fixed while moving a flagging threshold, and vary polynomial degree on size alone.
The printed test MAE belongs to the fixed Ridge baseline, not the later degree selection.

**Before this:** [Math Foundations](../01-math-foundations/README.md).
**Contents:** [lesson map](CONTENTS.md).

## Setup

- Windows: `.\setup.ps1`
- Linux or macOS: `bash ./setup.sh`

Use the `stage01-ml-basics` kernel. Framework training loops wait until Stage 02.

**Next:** [SmartVal: Why Hidden Layers and Backpropagation Exist](../03-neural-networks-and-backprop/01-smartval-neural-networks-and-backprop.ipynb).
Fair evidence helps judge a model; it cannot make a straight boundary express XOR.
