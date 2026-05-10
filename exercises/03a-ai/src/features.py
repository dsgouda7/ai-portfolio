"""Text preprocessing and embedding generation for AI models

This module provides:
- TextPreprocessor: Clean and tokenize text (with TODOs)
- EmbeddingGenerator: Create vector embeddings (with TODOs)
- VectorDatabase: Store and retrieve embeddings (with TODOs)
- Immediate feedback with rich console output

Learning objectives:
1. Implement text cleaning and tokenization for LLM input
2. Generate embeddings with sentence transformers
3. Build vector database with ChromaDB or FAISS
4. Perform semantic search and retrieval
5. Measure embedding quality and retrieval accuracy
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console

logger = logging.getLogger("pizzabot")
console = Console()


@dataclass
class PreprocessConfig:
    """Configuration for text preprocessing."""
    lowercase: bool = True
    remove_punctuation: bool = False
    remove_stopwords: bool = False
    min_length: int = 3
    max_length: int = 512


class TextPreprocessor:
    """Text preprocessing for LLM input preparation.

    Handles:
    - Tokenization and cleaning
    - Length normalization
    - Special character handling
    - Stopword removal (optional)
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        """Initialize preprocessor with configuration."""
        self.config = config or PreprocessConfig()
        self.stopwords = None

    def preprocess(self, text: str) -> str:
        """
        TODO: Implement text preprocessing (clean, lowercase, handle punctuation/stopwords per config)

        📖 See: notes/03a-ai/ch05-prompt-engineering/ (text normalization)
                notes/03a-ai/ch07-rag-and-embeddings/ (preparing text for embeddings)
        🎯 Unlocks: Constraint #2 ACCURACY (clean text → better embeddings → accurate retrieval)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement text preprocessing")

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        TODO: Preprocess list of texts with progress tracking

        📖 See: notes/03a-ai/ch07-rag-and-embeddings/ (batch processing for RAG)
        🎯 Unlocks: Constraint #2 ACCURACY (efficient preprocessing for large document sets)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement batch preprocessing")


class EmbeddingGenerator:
    """Generate vector embeddings for text using sentence transformers.

    Sentence transformers create dense vectors that capture semantic meaning:
    - Similar texts → similar vectors (high cosine similarity)
    - Different texts → different vectors (low similarity)
    - Vector dimension typically 384-768 for efficient models

    Popular models:
    - all-MiniLM-L6-v2: Fast, 384-dim, good for general use
    - all-mpnet-base-v2: Slower, 768-dim, better quality
    - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A retrieval
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """Initialize embedding generator.

        Args:
            model_name: Sentence transformer model name
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = None

    def load(self):
        """
        TODO: Load sentence transformer model and determine embedding dimension

        📖 See: notes/03a-ai/ch07-rag-and-embeddings/ (sentence transformers, embedding models)
        🎯 Unlocks: Constraint #2 ACCURACY (semantic embeddings enable accurate retrieval)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement model loading")

    def encode(self, text: str) -> np.ndarray:
        """
        TODO: Generate normalized embedding for single text

        📖 See: notes/03a-ai/ch07-rag-and-embeddings/ (vector embeddings, cosine similarity)
        🎯 Unlocks: Constraint #2 ACCURACY (semantic search for menu facts)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement single text encoding")

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        TODO: Generate normalized embeddings for batch of texts with progress bar

        📖 See: notes/03a-ai/ch07-rag-and-embeddings/ (batch embedding generation)
        🎯 Unlocks: Constraint #3 LATENCY (batch processing reduces overhead)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement batch encoding")


class VectorDatabase:
    """Vector database for semantic search using ChromaDB or FAISS.

    Stores embeddings and enables fast similarity search:
    - Add documents with embeddings
    - Query by text or embedding
    - Return top-k most similar documents
    - Measure retrieval accuracy

    ChromaDB vs FAISS:
    - ChromaDB: Easier API, persistent storage, metadata support
    - FAISS: Faster for large collections (1M+ docs), more memory efficient
    """

    def __init__(
        self,
        db_type: str = "chromadb",
        collection_name: str = "documents",
        persist_directory: Optional[str] = None
    ):
        """Initialize vector database.

        Args:
            db_type: Database type ('chromadb' or 'faiss')
            collection_name: Name for document collection
            persist_directory: Directory to save database
        """
        self.db_type = db_type
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.db = None
        self.collection = None  # For ChromaDB
        self.documents = []
        self.embeddings = None
        self.metadata = []  # For FAISS

    def create(self, embedding_dim: int):
        """
        Create ChromaDB or FAISS vector database index

        📖 See: notes/03a-ai/ch08-vector-dbs/ (ChromaDB, FAISS, indexing strategies)
        🎯 Unlocks: Constraint #2 ACCURACY + #3 LATENCY (fast similarity search)
        """
        console.print(f"Creating {self.db_type} vector database...")

        if self.db_type == "chromadb":
            # Create ChromaDB client and collection
            import chromadb
            from chromadb.config import Settings

            if self.persist_directory:
                self.db = chromadb.Client(Settings(
                    anonymized_telemetry=False,
                    persist_directory=self.persist_directory
                ))
            else:
                self.db = chromadb.Client(Settings(anonymized_telemetry=False))

            # Create or get collection
            self.collection = self.db.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            console.print("  ✓ ChromaDB collection created (HNSW index)", style="green")

        elif self.db_type == "faiss":
            # Create FAISS index (HNSW for best recall/speed balance)
            import faiss

            # Use HNSW index for production-quality search
            M = 16  # Number of connections per node
            self.db = faiss.IndexHNSWFlat(embedding_dim, M)
            self.db.hnsw.efConstruction = 64  # Build quality
            self.db.hnsw.efSearch = 40  # Query quality
            console.print("  ✓ FAISS HNSW index created", style="green")

        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add documents with embeddings to database (ChromaDB or FAISS)

        📖 See: notes/03a-ai/ch08-vector-dbs/ (indexing documents, metadata storage)
        🎯 Unlocks: Constraint #2 ACCURACY (build knowledge base for retrieval)
        """
        console.print(f"Adding {len(documents)} documents to {self.db_type}...")

        if self.db_type == "chromadb":
            # ChromaDB expects embeddings as list of lists
            ids = [f"doc_{i}" for i in range(len(documents))]

            if metadata:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadata
                )
            else:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=documents
                )
            console.print(f"  ✓ {len(documents)} documents added to ChromaDB", style="green")

        elif self.db_type == "faiss":
            # FAISS stores vectors only, we need to track documents separately
            self.documents = documents
            self.metadata = metadata or [{} for _ in documents]

            # Normalize embeddings for cosine similarity (using inner product)
            embeddings_normalized = embeddings.copy()
            norms = np.linalg.norm(embeddings_normalized, axis=1, keepdims=True)
            embeddings_normalized = embeddings_normalized / norms

            # Add to FAISS index
            self.db.add(embeddings_normalized.astype(np.float32))
            console.print(f"  ✓ {len(documents)} documents added to FAISS", style="green")

        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents and return top-k results with similarity scores

        📖 See: notes/03-ai/ch05_vector_dbs/ (similarity search, HNSW, IVF)
                notes/03-ai/ch04_rag_and_embeddings/ (retrieval metrics)
        🎯 Unlocks: Constraint #2 ACCURACY (retrieve relevant context → ground LLM answers)
        """
        if self.db_type == "chromadb":
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Format results
            return [
                {
                    "document": doc,
                    "similarity": 1 - dist,  # ChromaDB returns distance, convert to similarity
                    "metadata": meta
                }
                for doc, dist, meta in zip(
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0] if results["metadatas"][0] else [{}] * top_k
                )
            ]

        elif self.db_type == "faiss":
            # Normalize query for cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)

            # Search FAISS
            distances, indices = self.db.search(
                query_norm.reshape(1, -1).astype(np.float32),
                top_k
            )

            # Format results
            return [
                {
                    "document": self.documents[idx],
                    "similarity": float(dist),  # Inner product = cosine similarity on normalized vectors
                    "metadata": self.metadata[idx]
                }
                for dist, idx in zip(distances[0], indices[0])
                if idx >= 0  # FAISS returns -1 for empty slots
            ]

        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

    def query_text(
        self,
        query_text: str,
        embedding_generator: EmbeddingGenerator,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (convenience wrapper for query)

        📖 See: notes/03-ai/ch04_rag_and_embeddings/ (text-to-vector retrieval)
        🎯 Unlocks: Constraint #2 ACCURACY (end-to-end retrieval pipeline)
        """
        # Generate embedding for query text
        query_embedding = embedding_generator.encode(query_text)

        # Search using embedding
        return self.query(query_embedding, top_k)

    def evaluate_retrieval(
        self,
        test_queries: List[Dict[str, Any]],
        embedding_generator: EmbeddingGenerator,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate retrieval accuracy by checking if expected documents appear in top-k results

        📖 See: notes/03-ai/ch08_evaluating_ai_systems/ (retrieval metrics: precision@k, recall@k, MRR)
                notes/03-ai/ch04_rag_and_embeddings/ (RAG evaluation)
        🎯 Unlocks: Constraint #6 RELIABILITY (measure and monitor retrieval quality)

        Expected format for test_queries:
        [
            {"query": "What is the price of pepperoni pizza?", "expected_docs": [3, 7, 12]},
            ...
        ]
        """
        console.print(f"\nEvaluating retrieval on {len(test_queries)} test queries...")

        total_precision = 0.0
        total_recall = 0.0
        total_mrr = 0.0

        for query_data in test_queries:
            query_text = query_data["query"]
            expected_docs = set(query_data.get("expected_docs", []))

            # Perform search
            results = self.query_text(query_text, embedding_generator, top_k)

            # For FAISS, we need to track doc indices; for ChromaDB, extract from IDs
            if self.db_type == "chromadb":
                retrieved_indices = [
                    int(result.get("id", "doc_-1").split("_")[1])
                    for result in results
                ]
            else:
                # For FAISS, we track indices directly
                retrieved_indices = list(range(len(results)))

            retrieved_set = set(retrieved_indices[:top_k])

            # Calculate metrics
            if expected_docs:
                # Precision@k: fraction of retrieved docs that are relevant
                true_positives = len(retrieved_set & expected_docs)
                precision = true_positives / top_k if top_k > 0 else 0.0

                # Recall@k: fraction of relevant docs that were retrieved
                recall = true_positives / len(expected_docs) if expected_docs else 0.0

                # MRR: reciprocal rank of first relevant doc
                mrr = 0.0
                for rank, idx in enumerate(retrieved_indices[:top_k], start=1):
                    if idx in expected_docs:
                        mrr = 1.0 / rank
                        break

                total_precision += precision
                total_recall += recall
                total_mrr += mrr

        n_queries = len(test_queries)
        metrics = {
            "precision@k": total_precision / n_queries if n_queries > 0 else 0.0,
            "recall@k": total_recall / n_queries if n_queries > 0 else 0.0,
            "mrr": total_mrr / n_queries if n_queries > 0 else 0.0
        }

        console.print(f"  Precision@{top_k}: {metrics['precision@k']:.3f}", style="cyan")
        console.print(f"  Recall@{top_k}: {metrics['recall@k']:.3f}", style="cyan")
        console.print(f"  MRR: {metrics['mrr']:.3f}", style="cyan")

        return metrics

    def save(self):
        """
        Save database to disk (ChromaDB or FAISS index + documents)

        📖 See: notes/03-ai/ch05_vector_dbs/ (persistence, index serialization)
        🎯 Unlocks: Constraint #6 RELIABILITY (persist index for production)
        """
        if not self.persist_directory:
            console.print("  ⚠ No persist_directory set, skipping save", style="yellow")
            return

        from pathlib import Path
        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        if self.db_type == "chromadb":
            # ChromaDB handles persistence automatically if persist_directory is set
            console.print(f"  ✓ ChromaDB persisted to {self.persist_directory}", style="green")

        elif self.db_type == "faiss":
            import faiss
            import pickle

            # Save FAISS index
            index_path = persist_path / f"{self.collection_name}.faiss"
            faiss.write_index(self.db, str(index_path))

            # Save documents and metadata
            docs_path = persist_path / f"{self.collection_name}_docs.pkl"
            with open(docs_path, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "metadata": self.metadata
                }, f)

            console.print(f"  ✓ FAISS index saved to {index_path}", style="green")
            console.print(f"  ✓ Documents saved to {docs_path}", style="green")

        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

    def load(self):
        """
        Load database from disk (FAISS index + documents)

        📖 See: notes/03-ai/ch05_vector_dbs/ (loading persisted indexes)
        🎯 Unlocks: Constraint #6 RELIABILITY (restore index across restarts)
        """
        if not self.persist_directory:
            console.print("  ⚠ No persist_directory set, skipping load", style="yellow")
            return

        from pathlib import Path
        persist_path = Path(self.persist_directory)

        if self.db_type == "chromadb":
            # ChromaDB handles loading automatically when client is created
            import chromadb
            from chromadb.config import Settings

            self.db = chromadb.Client(Settings(
                anonymized_telemetry=False,
                persist_directory=self.persist_directory
            ))
            self.collection = self.db.get_collection(name=self.collection_name)
            console.print(f"  ✓ ChromaDB loaded from {self.persist_directory}", style="green")

        elif self.db_type == "faiss":
            import faiss
            import pickle

            # Load FAISS index
            index_path = persist_path / f"{self.collection_name}.faiss"
            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {index_path}")

            self.db = faiss.read_index(str(index_path))

            # Load documents and metadata
            docs_path = persist_path / f"{self.collection_name}_docs.pkl"
            if docs_path.exists():
                with open(docs_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data["documents"]
                    self.metadata = data["metadata"]

            console.print(f"  ✓ FAISS index loaded from {index_path}", style="green")
            console.print(f"  ✓ Documents loaded from {docs_path}", style="green")

        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")
