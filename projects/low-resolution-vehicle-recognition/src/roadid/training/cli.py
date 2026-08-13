"""RoadID dataset preparation and training command line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from roadid.training.config import load_training_config
from roadid.training.trainer import prepare_dataset, train_pipeline


def prepare_main(argv: list[str] | None = None) -> int:
    parser = _parser("Prepare a leakage-safe RoadID dataset.")
    arguments = parser.parse_args(argv)
    config = load_training_config(arguments.config)
    root = prepare_dataset(config, smoke=arguments.smoke, output_root=arguments.output_root)
    print(root)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser("Train, calibrate, evaluate, and package RoadID.")
    parser.add_argument("--prepared-root", type=Path)
    arguments = parser.parse_args(argv)
    config = load_training_config(arguments.config)
    prepared = arguments.prepared_root
    if prepared is None:
        prepared = prepare_dataset(
            config,
            smoke=arguments.smoke,
            output_root=arguments.output_root / "dataset" if arguments.output_root else None,
        )
    model_root = arguments.output_root / "models" if arguments.output_root else None
    bundle = train_pipeline(
        config,
        prepared_root=prepared,
        smoke=arguments.smoke,
        output_root=model_root,
    )
    print(bundle)
    return 0


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path("configs/train_resnet50.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=Path)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
