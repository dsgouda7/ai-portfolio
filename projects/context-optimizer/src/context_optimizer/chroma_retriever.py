"""
ChromaDB-backed Retriever for Compressed Chunks

Stores compressed summaries with embeddings in local ChromaDB for semantic similarity search.
Much better than keyword matching for conceptual queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import chromadb
from chromadb.config import Settings

from .compressor import CompressedChunk


class ChromaCompressedRetriever:
    """
    Store and retrieve compressed chunks using ChromaDB with embeddings.

    Better than DualStorageRetriever because:
    - Uses semantic similarity (embeddings) instead of keyword matching
    - Persistent storage (survives restarts)
    - Scales to millions of chunks
    - Built-in metadata filtering
    """

    def __init__(
        self,
        collection_name: str = "compressed_chunks",
        persist_directory: str = "./chroma_db",
        embedding_function: str = "default"
    ):
        """
        Initialize ChromaDB retriever

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Where to store ChromaDB data
            embedding_function: "default" (ChromaDB's built-in) or custom
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        print(f"[ChromaRetriever] Initialized with collection '{collection_name}'")
        print(f"[ChromaRetriever] Storage: {self.persist_directory}")
        print(f"[ChromaRetriever] Current size: {self.collection.count():,} chunks")

    def add_chunks(self, chunks: list[CompressedChunk], batch_size: int = 100) -> None:
        """
        Add compressed chunks to ChromaDB with embeddings

        Args:
            chunks: List of compressed chunks to store
            batch_size: How many to add at once
        """
        if not chunks:
            print(f"[ChromaRetriever] No chunks to add")
            return

        print(f"[ChromaRetriever] Adding {len(chunks):,} chunks in batches of {batch_size}...")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.compressed_summary for chunk in batch]  # ChromaDB will embed these
            metadatas = [
                {
                    "entities": ",".join(chunk.entities),
                    "keywords": ",".join(chunk.keywords),
                    "original_tokens": chunk.original_tokens,
                    "compressed_tokens": chunk.compressed_tokens,
                    "compression_ratio": chunk.compression_ratio,
                    **chunk.metadata  # Include any custom metadata
                }
                for chunk in batch
            ]

            # Store raw text separately (too large for metadata)
            # We'll store chunk_id -> raw_text mapping in a separate dict/file

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            if (i + batch_size) % 500 == 0:
                print(f"  [Progress] Added {min(i + batch_size, len(chunks)):,}/{len(chunks):,} chunks")

        print(f"[ChromaRetriever] [OK] Added {len(chunks):,} chunks")
        print(f"[ChromaRetriever] Total collection size: {self.collection.count():,} chunks")

    def search(
        self,
        query: str,
        top_k: int = 5,
        where_filter: dict[str, Any] | None = None
    ) -> list[dict]:
        """
        Semantic search using embeddings

        Args:
            query: Search query (will be embedded automatically)
            top_k: Number of results to return
            where_filter: Optional metadata filters (e.g., {"source": "books"})

        Returns:
            List of results with chunk_id, summary, metadata, similarity score
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )

        # Format results
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "chunk_id": results["ids"][0][i],
                "compressed_summary": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None,
                "entities": results["metadatas"][0][i]["entities"].split(",") if results["metadatas"][0][i].get("entities") else [],
                "keywords": results["metadatas"][0][i]["keywords"].split(",") if results["metadatas"][0][i].get("keywords") else [],
            })

        return hits

    def get_chunk_by_id(self, chunk_id: str) -> dict | None:
        """Retrieve a specific chunk by ID"""
        results = self.collection.get(ids=[chunk_id])

        if not results["ids"]:
            return None

        return {
            "chunk_id": results["ids"][0],
            "compressed_summary": results["documents"][0],
            "metadata": results["metadatas"][0],
            "entities": results["metadatas"][0]["entities"].split(",") if results["metadatas"][0].get("entities") else [],
            "keywords": results["metadatas"][0]["keywords"].split(",") if results["metadatas"][0].get("keywords") else [],
        }

    def delete_collection(self) -> None:
        """Delete the entire collection (for testing/cleanup)"""
        self.client.delete_collection(self.collection.name)
        print(f"[ChromaRetriever] Deleted collection '{self.collection.name}'")

    def get_stats(self) -> dict:
        """Get collection statistics"""
        count = self.collection.count()

        return {
            "collection_name": self.collection.name,
            "total_chunks": count,
            "storage_path": str(self.persist_directory),
            "embedding_function": "default (ChromaDB built-in)"
        }


def load_chunks_from_chroma(
    collection_name: str = "compressed_chunks",
    persist_directory: str = "./chroma_db"
) -> ChromaCompressedRetriever:
    """
    Load an existing ChromaDB collection

    Args:
        collection_name: Name of the collection to load
        persist_directory: Where ChromaDB data is stored

    Returns:
        ChromaCompressedRetriever instance
    """
    retriever = ChromaCompressedRetriever(
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    if retriever.collection.count() == 0:
        print(f"[WARNING] Collection '{collection_name}' is empty!")

    return retriever
