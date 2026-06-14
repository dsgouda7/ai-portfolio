# nifty500 stock price predictor

playing around with building a single composite metric for OHLCV data and seeing how well a simple lag model can predict it.

## the metric

```
metric = ((H + L + C) / 3) * ln(vol / vol_20d_avg)
```

wanted something that captures both price action and volume context in one number rather than feeding raw OHLCV separately. typical price covers the full intraday range instead of just close. the log-ratio on volume is zero on an average day, positive when volume picks up, negative when it's quiet — so the metric only spikes when *both* price and volume are doing something. keeps the feature space small which makes cross-validation more stable.

## what it does

- loads nifty500 5-year daily OHLCV data
- picks the top 5 tickers by average volume
- for each ticker: builds the metric, creates lag1/lag2 features, runs polynomial CV (degrees 1–10) using TimeSeriesSplit to find the best degree without leaking future data
- trains on years 1–4, predicts year 5
- plots actual vs predicted for all 5 tickers in a single figure

## running it

```bash
.\setup.ps1        # creates venv, installs deps, checks for kaggle creds
```

first run will auto-download the dataset from kaggle. needs credentials — either copy `~/.kaggle/kaggle.json` into this directory or create one:

```json
{ "username": "your_kaggle_username", "key": "your_api_key" }
```

get your key at https://www.kaggle.com/settings → Account → API → Create New Token. `kaggle.json` is gitignored.

```bash
python stock_price_predictor.py
```

## findings

R² on the year-5 holdout varies by ticker but is generally low — expected. a 2-feature lag model isn't going to beat the market. the point was to see if the custom metric carries any predictive signal at all and to practice the full pipeline: feature design → CV → holdout evaluation → visualization.
