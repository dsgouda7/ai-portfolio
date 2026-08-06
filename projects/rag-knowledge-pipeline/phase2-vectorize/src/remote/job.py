"""Python wheel entry point for the Databricks index update job."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from .runner import run_remote_vectorization


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update Riverside's Databricks vector index")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--chunk-table", required=True)
    parser.add_argument("--vector-record-table", required=True)
    parser.add_argument("--quality-table", required=True)
    parser.add_argument("--vector-search-endpoint", required=True)
    parser.add_argument("--vector-search-endpoint-type", default="standard")
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--index-version", required=True)
    parser.add_argument("--pipeline-version", required=True)
    parser.add_argument("--chunk-strategy-version", required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=128)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-model-revision", required=True)
    parser.add_argument("--embedding-endpoint", required=True)
    parser.add_argument("--embedding-dimensions", type=int, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--since")
    parser.add_argument("--run-id")
    return parser


def main(**overrides: Any) -> dict[str, Any]:
    config = vars(_parser().parse_args()) if not overrides else overrides
    logging.basicConfig(level=logging.INFO)
    return run_remote_vectorization(config, logger=logging.getLogger("riverside.indexing"))


if __name__ == "__main__":
    main()
