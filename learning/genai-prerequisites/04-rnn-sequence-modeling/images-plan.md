# 04-rnn-sequence-modeling Image Plan

## Asset Rules
- Store in `images/` with descriptive lowercase filenames.
- Wide 16:9 composition, at least 1600×900 PNG.
- Dark graphite background, muted teal for data flow, amber for trainable components, coral for failures/warnings, ivory labels.
- Generate with Perchance only.

## Planned Assets

| Asset | Notebook placement | Teaching job |
|---|---|---|
| `rnn-hidden-state-unrolled.png` | Part 1 intro (before vocab code) | Show RNN unrolled over 3 steps with hidden state flowing right |
| `vanishing-gradient-vs-timestep.png` | Part 3 intro (before BPTT ablation) | Side-by-side: RNN gradient decays exponentially; LSTM stays flat |
| `lstm-gate-equations.png` | Part 4 intro (before LSTMCellFromScratch) | 4-panel: forget/input/output/cell gates with color-coded data paths |

## Perchance Prompts

### `rnn-hidden-state-unrolled.png`
```text
Flat vector technical diagram, wide 16:9, dark graphite background. An RNN unrolled across three timesteps t-1, t, t+1. Each step shows an amber box labeled "RNN cell" with two inputs: input vector x_t from below (muted teal) and hidden state h_{t-1} from the left (amber arrow). Each cell outputs h_t flowing right. The tanh activation gate is shown as a small circle inside each cell. Below the diagram: the recurrent equation h_t = tanh(W_h h_{t-1} + W_x x_t + b) in large readable text. Right side: a deep vertical stack showing "unrolled = depth = vanishing gradients". Ivory labels, coral warning on the depth stack. No logos, no photorealism, no gradients, no tiny text.
```

### `vanishing-gradient-vs-timestep.png`
```text
Flat vector data visualization, wide 16:9, dark graphite background. Two side-by-side line charts sharing the same x-axis labeled "distance from loss (timesteps)". Left chart "RNN without gating": a steeply decaying coral curve starting near 1.0 at timestep 1 and reaching near 0.0 by timestep 20. Right chart "LSTM": a roughly flat muted teal line staying near 0.8 across all 20 timesteps. Both charts have a y-axis labeled "gradient norm" with range 0 to 1.0. Title in ivory: "Gating preserves gradient signal over time". No logos, no photorealism, no gradients, no tiny text.
```

### `lstm-gate-equations.png`
```text
Flat vector technical infographic, wide 16:9, dark graphite background. Four aligned LSTM gate equation panels labeled forget gate, input gate, cell state update, output gate. Each panel shows the sigmoid or tanh activation as a small icon, the weight matrices W_f W_i W_c W_o as amber rectangles, and the resulting gate vector as a colored bar. Arrows show: forget gate applied to cell state in teal (with coral erasure portion), input gate applied to new candidate in amber, cell state flowing through as a horizontal highway in teal, output gate controlling h_t. All equations in readable symbolic math. Ivory background text, coral for gates that can zero out. No logos, no photorealism, no gradients, no tiny text.
```

## Embedding Convention
Reference images as `![Alt text](images/filename.png)` in a dedicated markdown cell immediately before the relevant code section.
