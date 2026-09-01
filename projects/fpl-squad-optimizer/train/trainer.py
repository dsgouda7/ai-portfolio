"""
trainer.py — pure training logic, no I/O side-effects.

Extracted from train.py so it can be imported by the pipeline step
(pipeline/steps/train.py) without triggering the full local orchestration
(git pull, ingest, FPL API calls, etc.) that train.py performs at module level.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from utils import POS_FEATURES, MODEL_NAMES

N_ESTIMATORS = 44   # walk-forward CV (tune_n_estimators.py, 5-fold) showed
                    # early stopping halts at 14–47 trees per position;
                    # p90 across all positions = 44. 200 was overfitting.


def train_models(
    df: pd.DataFrame,
    completed_internal_index: int,
) -> tuple[dict, dict]:
    """
    Train one XGBRegressor per position through the completed-data cutoff.

    Each model uses only the features relevant to its position (POS_FEATURES),
    removing cross-position noise (e.g. saves for outfielders, threat for GKs).
    Returns (models, metrics) dicts keyed by position; metrics includes r2,
    rmse, n, top_feature, model_name, and n_features.
    """
    train = df[
        (df['Game_Week'] <= completed_internal_index) & df['target'].notna()
    ].copy()
    models: dict = {}
    metrics: dict = {}

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_features = POS_FEATURES[pos]
        pos_train = train[train['element_type'] == pos]
        X = pos_train[pos_features].copy()
        # FPL API and vaastav can return team/opponent_team as object dtype.
        # Cast any remaining object columns to int so XGBoost accepts them.
        for _c in X.select_dtypes(include='object').columns:
            X[_c] = pd.to_numeric(X[_c], errors='coerce')
        y = pos_train['target']

        model = XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)
        models[pos] = model

        y_pred = model.predict(X)
        absolute_errors = np.abs(y.to_numpy(dtype=float) - y_pred)
        top_feat = pos_features[model.feature_importances_.argmax()]
        metrics[pos] = {
            'r2':          round(float(r2_score(y, y_pred)), 4),
            'rmse':        round(float(root_mean_squared_error(y, y_pred)), 4),
            'n':           len(pos_train),
            'top_feature': top_feat,
            'model_name':  MODEL_NAMES[pos],
            'n_features':  len(pos_features),
            'error_p80':   round(float(np.quantile(absolute_errors, 0.80)), 4),
            'error_p95':   round(float(np.quantile(absolute_errors, 0.95)), 4),
        }
        m = metrics[pos]
        print(f"  [{MODEL_NAMES[pos]}]")
        print(f"    n={m['n']:,}  features={m['n_features']}  "
              f"R\u00b2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  top={top_feat}")

    return models, metrics
