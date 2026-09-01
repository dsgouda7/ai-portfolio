from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRanker

from utils import MODEL_NAMES, POS_FEATURES


def _numeric_frame(rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return rows[features].apply(pd.to_numeric, errors='coerce')


@dataclass
class CatBoostPositionModel:
    model: CatBoostRegressor
    features: list[str]
    categorical: tuple[str, ...] = ('team', 'opponent_team')

    def _frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        frame = rows[self.features].copy()
        for column in self.features:
            if column in self.categorical:
                frame[column] = frame[column].fillna(-1).astype(int).astype(str)
            else:
                frame[column] = pd.to_numeric(frame[column], errors='coerce')
        return frame

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self._frame(rows)), dtype=float)


@dataclass
class CalibratedRanker:
    ranker: XGBRanker
    calibrator: IsotonicRegression
    features: list[str]

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        raw = self.ranker.predict(_numeric_frame(rows, self.features))
        return np.asarray(self.calibrator.predict(raw), dtype=float)


def _metrics(position: str, targets: pd.Series, predictions: np.ndarray, features: list[str], name: str) -> dict:
    absolute_errors = np.abs(np.asarray(targets, dtype=float) - predictions)
    return {
        'r2': round(float(r2_score(targets, predictions)), 4),
        'rmse': round(float(root_mean_squared_error(targets, predictions)), 4),
        'n': len(targets),
        'top_feature': name,
        'model_name': f'{MODEL_NAMES[position]} ({name})',
        'n_features': len(features),
        'error_p80': round(float(np.quantile(absolute_errors, 0.80)), 4),
        'error_p95': round(float(np.quantile(absolute_errors, 0.95)), 4),
    }


def train_catboost_models(df: pd.DataFrame, cutoff: int) -> tuple[dict, dict]:
    train = df[(df['Game_Week'] <= cutoff) & df['target'].notna()].copy()
    models, metrics = {}, {}
    for position in ('GK', 'DEF', 'MID', 'FWD'):
        features = list(POS_FEATURES[position])
        rows = train[train['element_type'] == position]
        wrapper = CatBoostPositionModel(
            CatBoostRegressor(
                iterations=100,
                depth=5,
                learning_rate=0.05,
                loss_function='RMSE',
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            ),
            features,
        )
        inputs = wrapper._frame(rows)
        categorical_indexes = [
            inputs.columns.get_loc(column)
            for column in wrapper.categorical if column in inputs
        ]
        wrapper.model.fit(inputs, rows['target'], cat_features=categorical_indexes)
        predictions = wrapper.predict(rows)
        models[position] = wrapper
        metrics[position] = _metrics(position, rows['target'], predictions, features, 'CatBoost')
    return models, metrics


def _relevance_by_gameweek(rows: pd.DataFrame) -> pd.Series:
    percentile = rows.groupby('Game_Week')['target'].rank(method='average', pct=True)
    return np.floor(percentile.mul(5).clip(upper=4.999)).astype(int)


def train_lambdarank_models(df: pd.DataFrame, cutoff: int) -> tuple[dict, dict]:
    train = df[(df['Game_Week'] <= cutoff) & df['target'].notna()].copy()
    models, metrics = {}, {}
    for position in ('GK', 'DEF', 'MID', 'FWD'):
        features = list(POS_FEATURES[position])
        rows = train[train['element_type'] == position].copy()
        rows['Game_Week'] = pd.to_numeric(rows['Game_Week'], errors='raise').astype(int)
        rows = rows.sort_values(['Game_Week', 'id']).reset_index(drop=True)
        gameweeks = sorted(rows['Game_Week'].unique())
        calibration_count = max(1, min(5, len(gameweeks) // 5))
        calibration_start = gameweeks[-calibration_count]
        fit_rows = rows[rows['Game_Week'] < calibration_start].reset_index(drop=True)
        calibration_rows = rows[rows['Game_Week'] >= calibration_start].reset_index(drop=True)
        if fit_rows.empty:
            fit_rows = rows.copy()
            calibration_rows = rows.copy()
        groups = fit_rows.groupby('Game_Week', sort=True).size().tolist()
        if sum(groups) != len(fit_rows):
            raise ValueError(f'LambdaRank groups do not cover all {position} rows.')
        ranker = XGBRanker(
            objective='rank:pairwise',
            n_estimators=80,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        fit_inputs = _numeric_frame(fit_rows, features).reset_index(drop=True)
        relevance = _relevance_by_gameweek(fit_rows).reset_index(drop=True)
        ranker.fit(fit_inputs, relevance, group=groups)
        calibration_inputs = _numeric_frame(calibration_rows, features).reset_index(drop=True)
        calibration_raw = ranker.predict(calibration_inputs)
        calibrator = IsotonicRegression(out_of_bounds='clip').fit(
            calibration_raw, calibration_rows['target'].reset_index(drop=True)
        )
        wrapper = CalibratedRanker(ranker, calibrator, features)
        predictions = wrapper.predict(rows)
        models[position] = wrapper
        metrics[position] = _metrics(position, rows['target'], predictions, features, 'LambdaRank')
    return models, metrics
