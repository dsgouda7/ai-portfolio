"""Unit tests for ingestion pipeline."""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase1_ingest.src.loaders.wikipedia import WikipediaLoader


def test_wikipedia_loader_initialization():
    """Test WikipediaLoader can be instantiated."""
    loader = WikipediaLoader()
    assert loader.dataset_name == "wikipedia"
    assert loader.config_name == "20220301.simple"


def test_wikipedia_loader_load_small_sample():
    """Test loading a tiny Wikipedia sample."""
    loader = WikipediaLoader()
    df = loader.load(sample_size=10)

    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 10
    assert "id" in df.columns
    assert "title" in df.columns
    assert "text" in df.columns
    assert df["text"].str.len().min() > 0  # No empty texts


@pytest.mark.skipif(not Path("../../data/delta_lake/documents").exists(),
                    reason="Delta table does not exist")
def test_delta_verification():
    """Test Delta Lake verification (if table exists)."""
    loader = WikipediaLoader()
    row_count = loader.verify_delta("../../data/delta_lake")
    assert row_count > 0
