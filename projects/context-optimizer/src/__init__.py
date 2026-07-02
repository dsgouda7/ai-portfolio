# Context Optimizer Core Package
__version__ = "0.1.0"

from context_optimizer.benchmark import BenchmarkResult
from context_optimizer.index import CorpusIndex, IngestStats, QueryResult
from context_optimizer.protocols import Retriever
from context_optimizer.raw_index import RawHit, RawIndex
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
    # Raw content indexer
    "RawIndex",
    "RawHit",
]
