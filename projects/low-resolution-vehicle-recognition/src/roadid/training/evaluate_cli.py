"""RoadID bundle evaluation report command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roadid.training.trainer import evaluate_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and print a RoadID evaluation report.")
    parser.add_argument("--bundle", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(evaluate_bundle(arguments.bundle), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
