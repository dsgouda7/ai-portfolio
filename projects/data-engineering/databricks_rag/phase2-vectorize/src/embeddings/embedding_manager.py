"""Embedding generation and ChromaDB management."""

from pathlib import Path
from typing import List
from deltalake import DeltaTable
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from shared.logging_config import get_logger


logger = get_logger(__name__)


class EmbeddingManager:
    """Manage document embedding and vector store operations."""

    def __init__(
        self,
        embedding_model: str,
        device: str = "cpu",
        chunk_size: int = 1024,
        chunk_overlap: int = 128
    ):
        """
        Initialize embedding manager.

        Args:
            embedding_model: HuggingFace model name
            device: Device for embeddings (cpu or cuda:0)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model = embedding_model
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embeddings
        logger.info(f"Initializing embeddings: {embedding_model} on {device}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def load_from_delta(self, delta_path: str) -> List[Document]:
        """
        Load documents from Delta Lake.

        Args:
            delta_path: Path to Delta Lake table

        Returns:
            List of LangChain Document objects
        """
        delta_table_path = Path(delta_path) / "documents"

        if not delta_table_path.exists():
            raise FileNotFoundError(f"Delta table not found: {delta_table_path}")

        logger.info(f"Reading from Delta Lake: {delta_table_path}")
        dt = DeltaTable(str(delta_table_path))
        df = dt.to_pandas()

        # Convert to LangChain Documents
        documents = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=row["text"],
                metadata={
                    "id": str(row["id"]),
                    "title": row["title"]
                }
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from Delta Lake")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        """
        logger.info(f"Splitting {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(
        self,
        documents: List[Document],
        persist_directory: str
    ) -> Chroma:
        """
        Create ChromaDB vector store from documents.

        Args:
            documents: List of document chunks
            persist_directory: Path to persist ChromaDB

        Returns:
            ChromaDB vector store instance
        """
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating ChromaDB at {persist_directory}")
        logger.info(f"Embedding {len(documents)} chunks (this may take a few minutes)...")

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(persist_path)
        )

        logger.info("ChromaDB creation complete")
        return vectorstore
