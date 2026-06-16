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

N_ESTIMATORS = 200  # boosting rounds; 200 converges well at lr=0.1 on ~20k rows


def train_models(df: pd.DataFrame, game_week: int) -> tuple[dict, dict]:
    """
    Train one XGBRegressor per position on all GWs before game_week-1.

    Each model uses only the features relevant to its position (POS_FEATURES),
    removing cross-position noise (e.g. saves for outfielders, threat for GKs).
    Returns (models, metrics) dicts keyed by position; metrics includes r2,
    rmse, n, top_feature, model_name, and n_features.
    """
    train = df[(df['Game_Week'] < game_week - 1) & df['target'].notna()].copy()
    models: dict = {}
    metrics: dict = {}

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_features = POS_FEATURES[pos]
        pos_train = train[train['element_type'] == pos]
        X = pos_train[pos_features].fillna(0)
        # FPL API and vaastav can return team/opponent_team as object dtype.
        # Cast any remaining object columns to int so XGBoost accepts them.
        for _c in X.select_dtypes(include='object').columns:
            X[_c] = pd.to_numeric(X[_c], errors='coerce').fillna(0).astype(int)
        y = pos_train['target']

        model = XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)
        models[pos] = model

        y_pred = model.predict(X)
        top_feat = pos_features[model.feature_importances_.argmax()]
        metrics[pos] = {
            'r2':          round(float(r2_score(y, y_pred)), 4),
            'rmse':        round(float(root_mean_squared_error(y, y_pred)), 4),
            'n':           len(pos_train),
            'top_feature': top_feat,
            'model_name':  MODEL_NAMES[pos],
            'n_features':  len(pos_features),
        }
        m = metrics[pos]
        print(f"  [{MODEL_NAMES[pos]}]")
        print(f"    n={m['n']:,}  features={m['n_features']}  "
              f"R\u00b2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  top={top_feat}")

    return models, metrics
