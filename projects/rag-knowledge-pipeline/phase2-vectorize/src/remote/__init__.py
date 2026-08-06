"""Contract-native remote vectorization for Azure Databricks."""

from .contracts import AuthorizationContext, EmbeddingSpec, RemoteSettings
from .pipeline import RemoteVectorizationPipeline

__all__ = [
    "AuthorizationContext",
    "EmbeddingSpec",
    "RemoteSettings",
    "RemoteVectorizationPipeline",
]
