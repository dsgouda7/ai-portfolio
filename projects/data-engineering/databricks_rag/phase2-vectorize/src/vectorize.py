"""
Phase 2: Vectorization Pipeline
Delta Lake → ChromaDB embeddings

This module reads documents from Delta Lake, generates embeddings,
and populates ChromaDB for semantic search.
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config_loader import load_config, get_local_config, get_mode
from shared.logging_config import setup_logging, get_logger
from shared.constants import DELTA_LAKE_PATH, CHROMA_DB_PATH

from embeddings.embedding_manager import EmbeddingManager


def run_vectorization():
    """Execute the vectorization pipeline."""

    # Load configuration
    config = load_config("config.yaml")
    mode = get_mode(config)
    local_config = get_local_config(config)

    # Setup logging
    log_level = config.get("logging", {}).get("level", "INFO")
    setup_logging(level=log_level)
    logger = get_logger(__name__)

    logger.info(f"Starting vectorization pipeline in {mode} mode")

    # Check device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if mode == "remote":
        logger.error("Remote mode not yet implemented")
        raise NotImplementedError("Databricks remote vectorization requires workspace configuration")

    # Local mode: Read from Delta Lake, generate embeddings, write to ChromaDB
    delta_path = local_config.get("delta_path", str(DELTA_LAKE_PATH))
    vector_store_path = local_config.get("vector_store_path", str(CHROMA_DB_PATH))
    embedding_model = local_config.get("embedding_model")
    chunk_size = local_config.get("chunk_size", 1024)
    chunk_overlap = local_config.get("chunk_overlap", 128)

    logger.info(f"Delta Lake path: {delta_path}")
    logger.info(f"Vector store path: {vector_store_path}")
    logger.info(f"Embedding model: {embedding_model}")
    logger.info(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")

    # Initialize embedding manager
    manager = EmbeddingManager(
        embedding_model=embedding_model,
        device=device,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Load documents from Delta Lake
    logger.info("Loading documents from Delta Lake...")
    documents = manager.load_from_delta(delta_path)
    logger.info(f"Loaded {len(documents)} documents")

    # Create text chunks
    logger.info("Splitting documents into chunks...")
    chunks = manager.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    # Generate embeddings and populate ChromaDB
    logger.info("Generating embeddings and populating ChromaDB...")
    vectorstore = manager.create_vectorstore(chunks, vector_store_path)
    logger.info("ChromaDB populated successfully")

    # Verify
    collection_count = vectorstore._collection.count()
    logger.info(f"Verification complete: {collection_count} vectors in ChromaDB")

    logger.info("Phase 2 (Vectorization) completed successfully")


if __name__ == "__main__":
    try:
        run_vectorization()
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Vectorization failed: {str(e)}", exc_info=True)
        sys.exit(1)
