"""Feature engineering for FlixAI

Provides: User and item embedding generation for neural CF
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger("flixai")


class EmbeddingGenerator:
    """Embedding generator for user and item representations.
    
    Creates dense vector representations for users and items
    for use in neural collaborative filtering or similarity search.
    
    Attributes:
        embedding_dim: Dimension of embedding vectors
        user_embeddings: Learned user embeddings
        item_embeddings: Learned item embeddings
        user_scaler: Scaler for user features
        item_scaler: Scaler for item features
    
    Example:
        >>> generator = EmbeddingGenerator(embedding_dim=50)
        >>> user_emb, item_emb = generator.fit_transform(ratings_train)
        >>> user_vec = generator.get_user_embedding(user_id=123)
    """
    
    def __init__(
        self,
        embedding_dim: int = 50,
        random_state: int = 42
    ):
        """Initialize embedding generator.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            random_state: Random seed for reproducibility
        
        Raises:
            ValueError: If embedding_dim < 1
        """
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        
        self.user_embeddings = None
        self.item_embeddings = None
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        self._fitted = False
        
        logger.info(f"Initialized EmbeddingGenerator (dim={embedding_dim})")
    
    def fit_transform(
        self,
        ratings: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit embedding generator and create embeddings.
        
        Args:
            ratings: DataFrame with user_id, item_id, rating columns
        
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        
        Example:
            >>> user_emb, item_emb = generator.fit_transform(train_ratings)
        """
        logger.info("Generating user and item embeddings")
        
        # Get unique users and items
        unique_users = ratings['user_id'].unique()
        unique_items = ratings['item_id'].unique()
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        logger.info(f"Generating embeddings for {n_users} users and {n_items} items")
        
        # Create random embeddings (to be learned by model)
        np.random.seed(self.random_state)
        self.user_embeddings = np.random.randn(n_users, self.embedding_dim) * 0.1
        self.item_embeddings = np.random.randn(n_items, self.embedding_dim) * 0.1
        
        # Normalize embeddings
        self.user_embeddings = self.user_scaler.fit_transform(self.user_embeddings)
        self.item_embeddings = self.item_scaler.fit_transform(self.item_embeddings)
        
        self._fitted = True
        
        logger.info(
            f"Embeddings created: users={self.user_embeddings.shape}, "
            f"items={self.item_embeddings.shape}"
        )
        
        return self.user_embeddings, self.item_embeddings
    
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get embedding vector for a specific user.
        
        Args:
            user_id: User ID
        
        Returns:
            User embedding vector
        
        Raises:
            RuntimeError: If not fitted
            KeyError: If user_id not found
        """
        if not self._fitted:
            raise RuntimeError("EmbeddingGenerator not fitted. Call fit_transform() first.")
        
        if user_id >= len(self.user_embeddings):
            raise KeyError(f"User {user_id} not found in embeddings")
        
        return self.user_embeddings[user_id]
    
    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Get embedding vector for a specific item.
        
        Args:
            item_id: Item ID
        
        Returns:
            Item embedding vector
        
        Raises:
            RuntimeError: If not fitted
            KeyError: If item_id not found
        """
        if not self._fitted:
            raise RuntimeError("EmbeddingGenerator not fitted. Call fit_transform() first.")
        
        if item_id >= len(self.item_embeddings):
            raise KeyError(f"Item {item_id} not found in embeddings")
        
        return self.item_embeddings[item_id]
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = 'cosine'
    ) -> float:
        """Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine' or 'euclidean')
        
        Returns:
            Similarity score
        
        Example:
            >>> user_emb = generator.get_user_embedding(123)
            >>> item_emb = generator.get_item_embedding(456)
            >>> sim = generator.compute_similarity(user_emb, item_emb)
        """
        if metric == 'cosine':
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        elif metric == 'euclidean':
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert distance to similarity (higher is better)
            return 1.0 / (1.0 + distance)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
