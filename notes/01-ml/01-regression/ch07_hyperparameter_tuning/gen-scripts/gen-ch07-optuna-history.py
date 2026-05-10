"""
gen_ch07_optuna_history.py
Generates: ../img/ch07-optuna-history.png
Optuna convergence history: per-trial MAE scatter + running best line.
Uses real Optuna run if optuna is installed, else synthetic fallback data.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BG     = '#1a1a2e'
GRID_C = '#1e293b'
TEXT_C = '#e2e8f0'
WHITE  = '#f8fafc'

N_TRIALS = 100

# ── Try real Optuna run, fall back to synthetic data ────────────────────
try:
    import optuna
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline

    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    trial_maes:    list[float] = []
    trial_degrees: list[int]   = []

    def objective(trial: optuna.Trial) -> float:
        degree   = trial.suggest_int('degree', 1, 3)
        alpha    = trial.suggest_float('alpha', 1e-4, 10, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        pipe = Pipeline([
            ('poly',   PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model',  ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=5,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
        mae = -scores.mean() * 100_000
        trial_maes.append(mae)
        trial_degrees.append(degree)
        return -scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)

    trial_maes    = np.array(trial_maes)
    trial_degrees = np.array(trial_degrees)
    print('Used real Optuna run.')

except Exception as e:
    print(f'Optuna unavailable ({e}), using synthetic data.')
    rng = np.random.default_rng(42)
    # Simulate: first ~20 trials random, then Bayesian steering kicks in
    trial_degrees = rng.integers(1, 4, size=N_TRIALS)
    # Degree-2 tends to produce MAE ~37-41k, others higher
    base = np.where(trial_degrees == 2, 38.5, np.where(trial_degrees == 1, 55.0, 41.0))
    noise = rng.normal(0, 2.5, size=N_TRIALS)
    # Bayesian steering: gradually improve after trial 25
    cooling = np.clip(1.0 - np.arange(N_TRIALS) / 80, 0.3, 1.0)
    trial_maes = base + noise * cooling

# ── Running best ─────────────────────────────────────────────────────────
running_best = np.minimum.accumulate(trial_maes)

# ── Degree colours ───────────────────────────────────────────────────────
deg_colors = {1: '#f87171', 2: '#38bdf8', 3: '#4ade80'}
trial_nums = np.arange(1, N_TRIALS + 1)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)

for deg, color, label in [(1, '#f87171', 'degree=1 trials'),
                           (2, '#38bdf8', 'degree=2 trials'),
                           (3, '#4ade80', 'degree=3 trials')]:
    mask = trial_degrees == deg
    ax.scatter(trial_nums[mask], trial_maes[mask],
               color=color, alpha=0.55, s=30, zorder=3, label=label)

# Running best line
ax.plot(trial_nums, running_best, color=WHITE, linewidth=2.5, zorder=5,
        label='Running best MAE')

# Annotations
ax.axhline(38_000, color='#fb923c', linestyle='--', linewidth=1.2, alpha=0.7,
           label='Ch.5 baseline ($38k)')
ax.axhline(40_000, color='#4ade80', linestyle=':', linewidth=1.2, alpha=0.7,
           label='SmartVal target ($40k)')

# Arrow where Bayesian steering converges
conv_trial = int(np.argmin(np.abs(running_best - running_best[-1] - 0.5)) + 1)
conv_trial = min(conv_trial, N_TRIALS - 5)
ax.annotate('Bayesian steering\nconverges here',
            xy=(conv_trial, running_best[conv_trial - 1]),
            xytext=(conv_trial + 8, running_best[conv_trial - 1] + 2500),
            color=TEXT_C, fontsize=8.5,
            arrowprops=dict(arrowstyle='->', color=TEXT_C, lw=1.2))

ax.set_xlabel('Trial number', color=TEXT_C, fontsize=12)
ax.set_ylabel('CV MAE ($)', color=TEXT_C, fontsize=12)
ax.set_title('Optuna Convergence — 100 Trials, Smart Not Random',
             color=TEXT_C, fontsize=13, fontweight='bold')
ax.tick_params(colors=TEXT_C)
for sp in ax.spines.values():
    sp.set_edgecolor(GRID_C)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}k'))
ax.grid(True, color=GRID_C, linewidth=0.6, linestyle='--')
ax.legend(fontsize=8.5, framealpha=0.3, facecolor=BG, labelcolor=TEXT_C,
          loc='upper right', ncol=2)

fig.tight_layout()

out = Path(__file__).parent.parent / 'img' / 'ch07-optuna-history.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'Saved: {out}')
