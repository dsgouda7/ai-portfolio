# Context Optimizer Core Package
__version__ = "0.1.0"

from context_optimizer.benchmark import BenchmarkResult
from context_optimizer.compressor import ingest_file_blocks, split_into_sub_chunks
from context_optimizer.index import CorpusIndex, IngestStats, QueryResult
from context_optimizer.protocols import Retriever
from context_optimizer.raw_index import BlockIndex, BlockPointer, RawHit, RawIndex
from context_optimizer.tot_reasoner import Branch, ToTReasoner, ToTResult

__all__ = [
    # High-level facade
    "CorpusIndex",
    "QueryResult",
    "IngestStats",
    # Benchmark comparison
    "BenchmarkResult",
    # Lower-level building blocks
    "ToTReasoner",
    "ToTResult",
    "Branch",
    "Retriever",
    # Raw content stores
    "RawIndex",
    "RawHit",
    # Block-pointer index (large corpus, no data duplication)
    "BlockIndex",
    "BlockPointer",
    # Ingestion utilities
    "split_into_sub_chunks",
    "ingest_file_blocks",
]
