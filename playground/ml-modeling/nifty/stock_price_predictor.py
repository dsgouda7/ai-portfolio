import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import root_mean_squared_error

DATA_FILE  = 'nifty500_5yr_data.csv'
KAGGLE_DS  = 'shreyashautomation/nifty500-companies-5-years-stock-market-data'

if not os.path.exists(DATA_FILE):
    print(f"{DATA_FILE} not found, downloading from kaggle...")
    # kaggle looks for credentials in kaggle.json (this dir) or ~/.kaggle/kaggle.json
    if os.path.exists('kaggle.json'):
        os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_DS, path='.', unzip=True)
    print("done")

N_SPLITS   = 5
MAX_DEGREE = 10
VOL_WINDOW = 20
N_TICKERS  = 5

raw = pd.read_csv('nifty500_5yr_data.csv', parse_dates=['Date'])

# --- ticker quality filters ---
# we want tickers that a lag-based model can actually learn from.
# a 2-feature polynomial model trained on 3-4 years of daily closes is
# already a fragile setup — feeding it noisy, illiquid, or structurally
# broken tickers just produces garbage predictions and muddies the results.

ticker_stats = raw.groupby('Ticker').agg(
    avg_volume=('Volume', 'mean'),
    trading_days=('Date', 'count'),
)

# filter 1: drop lowest-10% by average volume
# thin stocks (low volume) have wide bid-ask spreads and sparse trading days.
# their price moves are often driven by a single large order rather than
# any discoverable pattern — the lag features pick up noise, not signal.
# the 10th-percentile threshold is a data-driven cutoff rather than an
# arbitrary absolute number, so it scales if the dataset changes.
vol_threshold = ticker_stats['avg_volume'].quantile(0.10)
eligible = ticker_stats[ticker_stats['avg_volume'] > vol_threshold].index

# filter 2: drop tickers with extreme intra-year price swings (>±50%)
# a stock that halves or doubles in a single calendar year has likely
# undergone a structural event (rights issue, debt restructuring, M&A,
# regulatory action). the distribution before the event is completely
# different from the distribution after it — training on the pre-event
# period and predicting into the post-event period is pointless at best
# and actively misleading at worst (see: IDEA's -151 metric spike).
# we check every 12-month rolling window, not just the 5-year span,
# because a blowup in year 2 can still corrupt training data for year 3.
def max_annual_swing(grp):
    grp = grp.sort_values('Date')
    annual_return = grp['Close'].pct_change(periods=252)   # ~1 trading year
    return annual_return.abs().max()

annual_swings = raw[raw['Ticker'].isin(eligible)].groupby('Ticker').apply(
    max_annual_swing, include_groups=False
)
stable = annual_swings[annual_swings <= 0.50].index

# pick top-N from the filtered pool by average volume
# volume is still the right ranking signal: high-volume tickers have tight
# spreads and reflect genuine price discovery, which is what the metric
# (typical_price × ln(vol/vol_avg)) is trying to capture.
top_tickers = (
    ticker_stats.loc[stable, 'avg_volume']
    .nlargest(N_TICKERS)
    .index.tolist()
)

print(f"eligible after volume filter : {len(eligible)}")
print(f"eligible after swing  filter : {len(stable)}")
print(f"top tickers: {top_tickers}")

results = {}
for ticker in top_tickers:
    df = raw[raw['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)

    df['typical'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['vol_avg'] = df['Volume'].rolling(VOL_WINDOW).mean()
    # metric = typical_price * ln(vol / vol_avg)
    # wanted a single number that picks up on both price range and volume activity
    # typical price uses H+L+C so it's not just the close — covers the full day
    # ln(vol/vol_avg) tells me how unusual the day's volume was vs the rolling baseline
    # zero on a normal day, positive when things get busy, negative when it's quiet
    # multiplying them together means the metric only spikes when price AND volume
    # are both doing something interesting — not just one of them
    # keeps the feature space small (just lag1, lag2) which makes CV more stable
    df['metric'] = df['typical'] * np.log(df['Volume'] / df['vol_avg'])
    df['lag1'] = df['metric'].shift(1)
    df['lag2'] = df['metric'].shift(2)
    df = df.dropna(subset=['metric', 'lag1', 'lag2'])

    cutoff = df['Date'].max() - pd.DateOffset(years=1)
    train = df['Date'] < cutoff
    print(f"  train: {df['Date'][train].min().date()} -> {df['Date'][train].max().date()}  |  test: {cutoff.date()} -> {df['Date'].max().date()}")
    X, y = df[['lag1', 'lag2']].values, df['metric'].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_scores = [
        cross_val_score(
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('poly', PolynomialFeatures(degree=d, include_bias=False)),
                ('scaler', StandardScaler()),
                ('model', LinearRegression()),
            ]),
            X[train], y[train], cv=tscv, scoring='r2'
        ).mean()
        for d in range(1, MAX_DEGREE + 1)
    ]
    best_deg = int(np.argmax(cv_scores)) + 1

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=best_deg, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression()),
    ])
    pipe.fit(X[train], y[train])

    y_pred = pipe.predict(X[~train])
    y_test = y[~train]
    r2   = pipe.score(X[~train], y[~train])
    rmse = root_mean_squared_error(y_test, y_pred)

    results[ticker] = {
        'dates': df['Date'].values[~train],
        'y_test': y_test,
        'y_pred': y_pred,
        'r2': r2,
        'rmse': rmse,
        'deg': best_deg,
    }
    print(f"  {ticker}  deg={best_deg}  R²={r2:.4f}  RMSE={rmse:,.4f}")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(
    f'Nifty 500 top-{N_TICKERS} by volume — Year 5 Hold-out\n'
    f'metric=((H+L+C)/3)·ln(vol/vol_{VOL_WINDOW}d)  |  features: lag1, lag2',
    fontsize=11,
)

for ax, ticker in zip(axes.flatten(), top_tickers):
    r = results[ticker]
    ax.plot(r['dates'], r['y_test'], color='steelblue', linewidth=0.9, label='Actual')
    ax.plot(r['dates'], r['y_pred'], color='tomato', linewidth=0.9, linestyle='--', label='Predicted')
    ax.set_title(f'{ticker}  (deg={r["deg"]}, R²={r["r2"]:.4f}, RMSE={r["rmse"]:,.4f})', fontsize=9)
    ax.set_ylabel('Metric', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

axes.flatten()[-1].set_visible(False)

plt.tight_layout()
plt.savefig('nifty500_top5.jpg', dpi=150)
plt.show()





