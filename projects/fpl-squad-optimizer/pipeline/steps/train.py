"""
Pipeline Step 2 — Train
=======================
Reads the SQLite database produced by the ingest step, builds rolling
features, trains four XGBRegressors (one per position), and writes
models + metrics to a joblib checkpoint.

I/O contract (env vars)
-----------------------
  FPL_DB_FILE       full path to fantasy_football.db (input)
                    Default: <project_root>/fantasy_football.db
  FPL_MODELS_FILE   full path to write models.joblib (output)
                    Default: <project_root>/models.joblib

In an Azure ML pipeline this step would:
  - Mount the DB output from the ingest step at the directory containing FPL_DB_FILE
  - Write to a pipeline Output at the directory containing FPL_MODELS_FILE

Local run (no container):
  python pipeline/steps/train.py

Container run:
  docker run --rm \\
    -e FPL_DB_FILE=/data/fantasy_football.db \\
    -e FPL_MODELS_FILE=/data/models/models.joblib \\
    -v pipeline_data:/data \\
    fpl-train
"""
import pathlib
import sys

import joblib

# Make project root importable regardless of cwd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from utils import (
    DB_FILE, MODELS_FILE, GAME_WEEK,
    POS_FEATURES, MODEL_NAMES,
    build_features, register_trained_players,
)
from eligibility import get_epl_members
from train.trainer import train_models


def main() -> None:
    print(f"[train] DB          : {DB_FILE}")
    print(f"[train] Models out  : {MODELS_FILE}")

    # Ensure the output directory exists
    pathlib.Path(MODELS_FILE).parent.mkdir(parents=True, exist_ok=True)

    # Step 2a — build rolling features from the DB
    print("[train] Building features …")
    all_data = build_features(DB_FILE)

    # Step 2b — train one XGBRegressor per position
    print("[train] Training XGBoost models (one per position) …")
    models, metrics = train_models(all_data, GAME_WEEK)

    # Step 2c — snapshot the current EPL member set for inference-time filtering
    try:
        epl_members = get_epl_members()
        print(f"[train] EPL snapshot: {len(epl_members)} players")
    except Exception as exc:
        print(f"[train] Warning: EPL snapshot failed ({exc}) — saving without")
        epl_members = None

    # Step 2d — build per-position feature metadata for the validation report
    feature_metadata = {
        pos: {
            'features':     POS_FEATURES[pos],
            'importances':  dict(zip(
                POS_FEATURES[pos],
                models[pos].feature_importances_.tolist(),
            )),
            'top_feature':  POS_FEATURES[pos][
                models[pos].feature_importances_.argmax()
            ],
        }
        for pos in models
    }

    # Step 2e — persist everything in a single joblib checkpoint
    joblib.dump({
        'models':           models,
        'metrics':          metrics,
        'epl_members':      epl_members,
        'model_names':      MODEL_NAMES,
        'pos_features':     {pos: list(feats) for pos, feats in POS_FEATURES.items()},
        'feature_metadata': feature_metadata,
    }, MODELS_FILE)
    print(f"[train] Saved to {MODELS_FILE}")

    # Step 2f — record which players were in this training run (used by ingest
    #           to identify new arrivals that need GW history backfilled)
    register_trained_players(DB_FILE, all_data)
    print("[train] Done.")


if __name__ == "__main__":
    main()
