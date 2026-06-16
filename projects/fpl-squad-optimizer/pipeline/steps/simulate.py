"""
Pipeline Step 3 — Simulate
===========================
Runs the walk-forward GW simulation: for each test GW it builds the feature
snapshot, selects a squad with the trained models, and records the predicted
vs actual points alongside the oracle-optimal squad.

I/O contract (env vars)
-----------------------
  FPL_DB_FILE       full path to fantasy_football.db (input)
                    Default: <project_root>/fantasy_football.db
  FPL_MODELS_FILE   full path to models.joblib (input)
                    Default: <project_root>/models.joblib
  FPL_RESULTS_DIR   directory to write simulation_results.csv / player_rows.csv
                    Default: <project_root>/simulations/results
  FPL_SIM_FROM      first GW to evaluate (int, default 15)
  FPL_SIM_TO        last  GW to evaluate (int, default 37)

In an Azure ML pipeline this step would:
  - Mount DB and models outputs from previous steps
  - Write to a pipeline Output at FPL_RESULTS_DIR

Local run (no container):
  python pipeline/steps/simulate.py

  # or with explicit GW range:
  FPL_SIM_FROM=20 FPL_SIM_TO=30 python pipeline/steps/simulate.py

Container run:
  docker run --rm \\
    -e FPL_DB_FILE=/data/fantasy_football.db \\
    -e FPL_MODELS_FILE=/data/models/models.joblib \\
    -e FPL_RESULTS_DIR=/data/results \\
    -e FPL_SIM_FROM=15 \\
    -e FPL_SIM_TO=37 \\
    -v pipeline_data:/data \\
    fpl-simulate
"""
import os
import pathlib
import sys

# Make project root importable regardless of cwd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from simulations.simulate_season import run_simulation


def main() -> None:
    test_from = int(os.environ.get('FPL_SIM_FROM', '15'))
    test_to   = int(os.environ.get('FPL_SIM_TO',   '37'))

    print(f"[simulate] GW range : {test_from}–{test_to}")
    print(f"[simulate] Results  : {os.environ.get('FPL_RESULTS_DIR', '(default)')}")

    run_simulation(test_from=test_from, test_to=test_to, overwrite=True)
    print("[simulate] Done.")


if __name__ == "__main__":
    main()
