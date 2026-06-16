"""
Phase 1: Ingestion Pipeline
Raw corpus → Delta Lake

This module loads a dataset (Wikipedia, PubMed, etc.) and writes it to Delta Lake
with ACID guarantees. The output is used by Phase 2 for vectorization.
"""

import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config_loader import load_config, get_local_config, get_mode
from shared.logging_config import setup_logging, get_logger
from shared.constants import DELTA_LAKE_PATH, RAW_DATA_PATH

from loaders.wikipedia import WikipediaLoader


def run_ingestion():
    """Execute the ingestion pipeline."""

    # Load configuration
    config = load_config("config.yaml")
    mode = get_mode(config)
    local_config = get_local_config(config)

    # Setup logging
    log_level = config.get("logging", {}).get("level", "INFO")
    setup_logging(level=log_level)
    logger = get_logger(__name__)

    logger.info(f"Starting ingestion pipeline in {mode} mode")

    if mode == "remote":
        logger.error("Remote mode not yet implemented")
        raise NotImplementedError("Databricks remote ingestion requires workspace configuration")

    # Local mode: Load dataset and write to Delta Lake
    dataset_name = local_config.get("dataset", "wikipedia")
    sample_size = local_config.get("sample_size", 1000)
    delta_path = local_config.get("delta_path", str(DELTA_LAKE_PATH))

    logger.info(f"Dataset: {dataset_name}, Sample size: {sample_size}")
    logger.info(f"Delta Lake path: {delta_path}")

    # Load dataset
    if dataset_name == "wikipedia":
        loader = WikipediaLoader()
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    logger.info("Loading corpus...")
    df = loader.load(sample_size=sample_size)
    logger.info(f"Loaded {len(df)} documents")

    # Write to Delta Lake
    logger.info("Writing to Delta Lake...")
    loader.write_to_delta(df, delta_path)
    logger.info("Delta Lake write complete")

    # Verify
    row_count = loader.verify_delta(delta_path)
    logger.info(f"Verification complete: {row_count} rows in Delta Lake")

    logger.info("Phase 1 (Ingestion) completed successfully")


if __name__ == "__main__":
    try:
        run_ingestion()
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        sys.exit(1)
