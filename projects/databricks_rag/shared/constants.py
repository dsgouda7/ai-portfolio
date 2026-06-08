"""Shared constants for the RAG pipeline."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_ROOT / "raw"
DELTA_LAKE_PATH = DATA_ROOT / "delta_lake"
CHROMA_DB_PATH = DATA_ROOT / "chroma_db"

# Dataset configurations
WIKIPEDIA_CONFIG = {
    "name": "wikipedia",
    "dataset_path": "wikipedia",
    "config": "20220301.simple",
    "split": "train",
    "text_column": "text",
    "title_column": "title"
}

PUBMED_CONFIG = {
    "name": "pubmed",
    "dataset_path": "pubmed",
    "config": None,
    "split": "train",
    "text_column": "abstract",
    "title_column": "title"
}

# Default embedding models
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATABRICKS_EMBEDDING_MODEL = "databricks-gte-large-en"

# Chunking defaults
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128

# Retrieval defaults
DEFAULT_TOP_K = 6
DEFAULT_LAMBDA_MULT = 0.25
