# Context Optimizer Core Package
__version__ = "0.1.0"

from context_optimizer.index     import CorpusIndex, IngestStats, QueryResult
from context_optimizer.benchmark import BenchmarkResult
from context_optimizer.tot_reasoner import Branch, Retriever, ToTReasoner, ToTResult

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
]
