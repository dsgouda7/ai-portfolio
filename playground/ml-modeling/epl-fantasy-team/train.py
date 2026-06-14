"""
train.py -- ingest the FPL dataset and train one XGBRegressor per position.

Run this whenever the dataset is updated (the GitHub repo updates daily):
    python train.py

Saves trained models and per-position quality metrics to models.joblib.
team_generator.py loads that file at selection time, so you never need to
re-train just to pick a team for the current GW.
Delete fantasy_football.db to force a full re-ingest of the CSV files.
"""
import joblib
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from utils import (
    DB_FILE, MODELS_FILE, PLAYERS_DIR, RAW_DATA_PATH, GAME_WEEK, FEATURES,
    ingest, build_features, register_trained_players,
)

N_ESTIMATORS = 200  # boosting rounds; 200 converges well at lr=0.1 on ~20k rows


def train_models(df, game_week):
    """
    Train one XGBRegressor per position on all GWs before game_week-1.

    Separate models per position because GK features (saves, clean_sheets)
    are noise for outfield players and vice versa. Returns (models, metrics)
    dicts keyed by position; metrics includes r2, rmse, n, top_feature.
    """
    train = df[(df['Game_Week'] < game_week - 1) & df['target'].notna()].copy()
    models = {}
    metrics = {}

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_train = train[train['element_type'] == pos]
        X = pos_train[FEATURES].fillna(0)
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

        # in-sample so optimistic, but R² near 0 means no signal found for that position
        y_pred = model.predict(X)
        metrics[pos] = {
            'r2':   round(float(r2_score(y, y_pred)), 4),
            'rmse': round(float(root_mean_squared_error(y, y_pred)), 4),
            'n':    len(pos_train),
            'top_feature': FEATURES[model.feature_importances_.argmax()],
        }
        m = metrics[pos]
        print(f"  {pos}: n={m['n']:,}  R\u00b2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  top={m['top_feature']}")

    return models, metrics


ingest(PLAYERS_DIR, RAW_DATA_PATH, DB_FILE)

print("Building features...")
all_data = build_features(DB_FILE)

print("Training XGBoost models (one per position)...")
models, metrics = train_models(all_data, GAME_WEEK)

joblib.dump({'models': models, 'metrics': metrics}, MODELS_FILE)
print(f"Models + metrics saved to {MODELS_FILE}")

print("Registering trained players in DB...")
register_trained_players(DB_FILE, all_data)
