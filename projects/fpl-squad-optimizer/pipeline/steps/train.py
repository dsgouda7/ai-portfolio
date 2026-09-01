"""
Pipeline Step 2 — Train
=======================
Reads the SQLite database produced by the ingest step, builds rolling
features, trains selectable XGBoost and/or GRU position models, and writes
separate joblib checkpoints with temporal cutoff manifests.

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
import argparse
import os
import pathlib
import sys

# Make project root importable regardless of cwd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from utils import DB_FILE, build_features, get_runtime_context, register_trained_players
from eligibility import get_epl_members
from feature_cache import load_or_build_feature_cache
from train.model_training import train_and_save_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('xgboost', 'catboost', 'lambdarank', 'rnn', 'ensemble', 'all'),
        default=os.environ.get('FPL_MODEL_TYPE', 'all'),
    )
    args = parser.parse_args()
    print(f"[train] DB          : {DB_FILE}")

    # Step 2a — build rolling features from the DB
    print("[train] Building features …")
    all_data, cache_hit = load_or_build_feature_cache(DB_FILE, build_features)
    print(f"[train] Feature cache: {'hit' if cache_hit else 'rebuilt in SQLite'}")

    # Step 2b — snapshot the current EPL member set for inference-time filtering
    try:
        epl_members = get_epl_members()
        print(f"[train] EPL snapshot: {len(epl_members)} players")
    except Exception as exc:
        print(f"[train] Warning: EPL snapshot failed ({exc}) — saving without")
        epl_members = None

    runtime = get_runtime_context(DB_FILE)
    train_and_save_models(
        all_data,
        runtime['completed_internal_index'],
        args.model,
        epl_members,
        DB_FILE,
    )

    # Step 2f — record which players were in this training run (used by ingest
    #           to identify new arrivals that need GW history backfilled)
    register_trained_players(DB_FILE, all_data)

    print("[train] Done.")


if __name__ == "__main__":
    main()
