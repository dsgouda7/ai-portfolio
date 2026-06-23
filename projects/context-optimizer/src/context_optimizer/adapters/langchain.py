"""
LangChain adapter — drop-in BaseRetriever backed by CorpusIndex.

Usage::

    from context_optimizer.adapters.langchain import ContextOptimizerRetriever
    from langchain.chains import RetrievalQA

    retriever = ContextOptimizerRetriever.from_texts(
        my_log_lines,
        model="llama3.2:3b",
        persist_dir="./my_index",
    )
    chain = RetrievalQA.from_chain_type(llm=your_llm, retriever=retriever)

Requires langchain-core:

    pip install 'context-optimizer[langchain]'
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

# Soft import — do not hard-fail at import time so users without langchain
# can still import other parts of the package.
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

if TYPE_CHECKING:
    from langchain_core.documents import Document as DocumentType


def _require_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain-core is required for the LangChain adapter. "
            "Install with: pip install 'context-optimizer[langchain]'"
        )


if _LANGCHAIN_AVAILABLE:

    class ContextOptimizerRetriever(BaseRetriever):
        """
        LangChain ``BaseRetriever`` backed by :class:`~context_optimizer.CorpusIndex`.

        Can be used anywhere LangChain accepts a retriever (RetrievalQA,
        ConversationalRetrievalChain, LCEL pipelines …).

        Quick-start::

            retriever = ContextOptimizerRetriever.from_texts(docs)
            # or
            retriever = ContextOptimizerRetriever.from_documents(langchain_docs)
        """

        # Pydantic fields — BaseRetriever uses pydantic v1 model
        index: Any          # CorpusIndex — typed Any to avoid pydantic compat issues
        collection: str = "default"
        top_k: int = 6

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: "CallbackManagerForRetrieverRun",
        ) -> list["DocumentType"]:
            result = self.index.query(
                query, collection=self.collection, top_k=self.top_k
            )
            return [Document(page_content=s) for s in result.evidence]

        @classmethod
        def from_texts(
            cls,
            texts: list[str],
            *,
            model: str = "llama3.2:3b",
            collection: str = "default",
            top_k: int = 6,
            **index_kwargs: Any,
        ) -> "ContextOptimizerRetriever":
            """Ingest plain-text lines and return a ready-to-use retriever."""
            from context_optimizer.index import CorpusIndex

            index = CorpusIndex(compression_model=model, **index_kwargs)
            index.ingest(texts, collection=collection)
            return cls(index=index, collection=collection, top_k=top_k)

        @classmethod
        def from_documents(
            cls,
            documents: list["DocumentType"],
            *,
            model: str = "llama3.2:3b",
            collection: str = "default",
            top_k: int = 6,
            **index_kwargs: Any,
        ) -> "ContextOptimizerRetriever":
            """Ingest LangChain ``Document`` objects and return a retriever."""
            return cls.from_texts(
                [d.page_content for d in documents],
                model=model,
                collection=collection,
                top_k=top_k,
                **index_kwargs,
            )

else:

    class ContextOptimizerRetriever:  # type: ignore[no-redef]
        """Stub raised when langchain-core is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_langchain()

        @classmethod
        def from_texts(cls, *args: Any, **kwargs: Any) -> "ContextOptimizerRetriever":
            _require_langchain()
            raise RuntimeError("unreachable")

        @classmethod
        def from_documents(cls, *args: Any, **kwargs: Any) -> "ContextOptimizerRetriever":
            _require_langchain()
            raise RuntimeError("unreachable")
