"""
RAG pipeline logic.

This module encapsulates the core RAG functionality including:
- LLM initialization
- Vector store loading
- Query execution
"""

import os
import torch
from pathlib import Path
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxLLM

from shared.logging_config import get_logger


logger = get_logger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self, config: dict):
        """
        Initialize RAG pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.chat_history = []

        logger.info(f"Initializing RAG pipeline on device: {self.device}")

        # Initialize components
        self._init_llm()
        self._init_embeddings()
        self._load_vectorstore()
        self._build_qa_chain()

    def _init_llm(self):
        """Initialize the language model."""
        llm_config = self.config.get("llm", {})

        model_id = os.getenv(
            "WATSONX_MODEL_ID",
            llm_config.get("model_id", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8")
        )
        watsonx_url = os.getenv(
            "WATSONX_URL",
            "https://us-south.ml.cloud.ibm.com"
        )
        project_id = os.getenv(
            "WATSONX_PROJECT_ID",
            "skills-network"
        )

        logger.info(f"Initializing WatsonxLLM: {model_id}")

        model_parameters = {
            "max_new_tokens": llm_config.get("max_tokens", 512),
            "temperature": llm_config.get("temperature", 0.1),
        }

        self.llm = WatsonxLLM(
            model_id=model_id,
            url=watsonx_url,
            project_id=project_id,
            params=model_parameters
        )

        logger.info("WatsonxLLM initialized")

    def _init_embeddings(self):
        """Initialize embedding model."""
        local_config = self.config.get("local", {})
        embedding_model = local_config.get(
            "embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        logger.info(f"Initializing embeddings: {embedding_model}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": self.device}
        )

        logger.info("Embeddings initialized")

    def _load_vectorstore(self):
        """Load ChromaDB vector store."""
        local_config = self.config.get("local", {})
        vector_store_path = local_config.get("vector_store_path", "./data/chroma_db")

        vector_path = Path(vector_store_path)

        if not vector_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {vector_store_path}. "
                "Run Phase 2 (vectorization) first."
            )

        logger.info(f"Loading ChromaDB from {vector_store_path}")

        self.vectorstore = Chroma(
            persist_directory=str(vector_path),
            embedding_function=self.embeddings
        )

        collection_count = self.vectorstore._collection.count()
        logger.info(f"ChromaDB loaded: {collection_count} vectors")

    def _build_qa_chain(self):
        """Build the QA retrieval chain."""
        retrieval_config = self.config.get("retrieval", {})

        search_type = retrieval_config.get("search_type", "mmr")
        top_k = retrieval_config.get("top_k", 6)
        lambda_mult = retrieval_config.get("lambda_mult", 0.25)

        logger.info(f"Building QA chain: {search_type}, k={top_k}")

        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={'k': top_k, 'lambda_mult': lambda_mult}
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            input_key="question"
        )

        logger.info("QA chain built successfully")

    def query(self, question: str, temperature: Optional[float] = None) -> dict:
        """
        Execute a RAG query.

        Args:
            question: The user's question
            temperature: Optional temperature override

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")

        # Update temperature if specified
        if temperature is not None and temperature != self.llm.params.get("temperature"):
            logger.debug(f"Overriding temperature: {temperature}")
            self.llm.params["temperature"] = temperature

        # Execute query
        result = self.qa_chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        })

        answer = result["result"]

        # Update chat history
        self.chat_history.append((question, answer))

        logger.info(f"Query complete. Answer length: {len(answer)} chars")

        return {
            "answer": answer,
            "question": question,
            "sources_count": self.config.get("retrieval", {}).get("top_k", 6)
        }

    def reset_history(self):
        """Clear chat history."""
        logger.info("Clearing chat history")
        self.chat_history = []

    def get_status(self) -> dict:
        """Get pipeline status."""
        return {
            "llm": "initialized" if self.llm else "not initialized",
            "embeddings": "initialized" if self.embeddings else "not initialized",
            "vectorstore": f"{self.vectorstore._collection.count()} vectors" if self.vectorstore else "not loaded",
            "device": self.device
        }
