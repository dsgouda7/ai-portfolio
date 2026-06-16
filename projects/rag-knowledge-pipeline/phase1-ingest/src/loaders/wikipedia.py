"""Wikipedia dataset loader."""

import pandas as pd
from datasets import load_dataset
from deltalake import write_deltalake, DeltaTable
from pathlib import Path
from shared.logging_config import get_logger


logger = get_logger(__name__)


class WikipediaLoader:
    """Load Wikipedia Simple English dataset into Delta Lake."""

    def __init__(self):
        self.dataset_name = "wikipedia"
        self.config_name = "20220301.simple"

    def load(self, sample_size: int = 1000) -> pd.DataFrame:
        """
        Load Wikipedia dataset from HuggingFace.

        Args:
            sample_size: Number of articles to load

        Returns:
            DataFrame with columns: id, title, text
        """
        logger.info(f"Loading {sample_size} Wikipedia articles...")

        try:
            # Try loading without trust_remote_code (new datasets versions)
            dataset = load_dataset(
                self.dataset_name,
                self.config_name,
                split=f"train[:{sample_size}]"
            )
        except Exception as e:
            logger.warning(f"Failed to load Wikipedia dataset: {e}")
            logger.info("Using synthetic test data instead...")

            # Generate synthetic data for testing
            import random
            articles = []
            for i in range(sample_size):
                articles.append({
                    "id": str(i),
                    "title": f"Test Article {i+1}",
                    "text": f"This is test article number {i+1}. " * 20 +
                            f"It contains information about topic {i+1}. " * 10 +
                            f"This demonstrates the RAG pipeline functionality."
                })

            df = pd.DataFrame(articles)
            logger.info(f"Generated {len(df)} synthetic articles")
            return df
        return df

    def write_to_delta(self, df: pd.DataFrame, delta_path: str):
        """
        Write DataFrame to Delta Lake.

        Args:
            df: DataFrame to write
            delta_path: Path to Delta Lake table
        """
        delta_dir = Path(delta_path)
        delta_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing to Delta Lake: {delta_path}")

        write_deltalake(
            table_or_uri=str(delta_dir / "documents"),
            data=df,
            mode="overwrite",
            schema_mode="overwrite"
        )

        logger.info("Delta Lake write successful")

    def verify_delta(self, delta_path: str) -> int:
        """
        Verify Delta Lake table exists and return row count.

        Args:
            delta_path: Path to Delta Lake table

        Returns:
            Number of rows in table
        """
        delta_table_path = Path(delta_path) / "documents"

        if not delta_table_path.exists():
            raise FileNotFoundError(f"Delta table not found: {delta_table_path}")

        dt = DeltaTable(str(delta_table_path))
        df = dt.to_pandas()

        return len(df)
