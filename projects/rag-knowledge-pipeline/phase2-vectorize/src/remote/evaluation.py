"""Versioned, content-free retrieval evaluation for an authorized vector index."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

from .contracts import AuthorizationContext, RemoteSettings, is_record_authorized, stable_identifier, utc_now
from .interfaces import EmbeddingProvider, VectorIndex


EVALUATOR_VERSION = "1.0.0"


def _case_metrics(relevant: set[str], retrieved: Sequence[str]) -> dict[str, float]:
    if not relevant:
        raise ValueError("Retrieval evaluation cases require at least one relevant chunk ID")
    hits = relevant.intersection(retrieved)
    recall = len(hits) / len(relevant)
    reciprocal_rank = 0.0
    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), len(retrieved))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "recall_at_k": recall,
        "reciprocal_rank": reciprocal_rank,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "case_count": 0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "authorization_leakage_count": 0,
        }
    count = len(rows)
    return {
        "case_count": count,
        "recall_at_k": sum(float(row["recall_at_k"]) for row in rows) / count,
        "mrr": sum(float(row["reciprocal_rank"]) for row in rows) / count,
        "ndcg_at_k": sum(float(row["ndcg_at_k"]) for row in rows) / count,
        "authorization_leakage_count": sum(
            int(row["authorization_leakage_count"]) for row in rows
        ),
    }


class RetrievalEvaluator:
    def __init__(
        self,
        *,
        settings: RemoteSettings,
        embedder: EmbeddingProvider,
        index: VectorIndex,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.index = index
        self.clock = clock

    def evaluate(
        self,
        cases: Sequence[Mapping[str, Any]],
        *,
        dataset_id: str,
        dataset_version: str,
        report_id: str | None = None,
    ) -> dict[str, Any]:
        vectors = list(self.embedder.embed([str(case["query"]) for case in cases]))
        if len(vectors) != len(cases):
            raise ValueError("Embedding provider did not return one vector per evaluation query")

        case_results: list[dict[str, Any]] = []
        by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case, vector in zip(cases, vectors):
            context = AuthorizationContext(
                tenant_id=str(case["tenant_id"]),
                region=str(case["region"]),
                classifications=tuple(case["classifications"]),
                principal_ids=tuple(case.get("principal_ids", ())),
                group_ids=tuple(case.get("group_ids", ())),
            )
            top_k = int(case["top_k"])
            records = list(self.index.query_authorized(vector, context, top_k=top_k))
            leakage_count = sum(not is_record_authorized(record, context) for record in records)
            metrics = _case_metrics(
                set(case["relevant_chunk_ids"]),
                [str(record["chunk_id"]) for record in records],
            )
            result = {
                "query_id": str(case["query_id"]),
                "slice": str(case["slice"]),
                "top_k": top_k,
                "returned_count": len(records),
                "authorization_leakage_count": leakage_count,
                **metrics,
            }
            case_results.append(result)
            by_slice[result["slice"]].append(result)

        generated_at = self.clock()
        resolved_report_id = report_id or stable_identifier(
            "retrieval-eval",
            (
                dataset_id,
                dataset_version,
                self.settings.index_name,
                self.settings.index_version,
                generated_at,
            ),
        )
        aggregate = _aggregate(case_results)
        return {
            "report_version": "1.0.0",
            "kind": "retrieval_evaluation_report",
            "report_id": resolved_report_id,
            "generated_at": generated_at,
            "dataset": {"id": dataset_id, "version": dataset_version},
            "evaluator": {"id": "riverside-retrieval-evaluator", "version": EVALUATOR_VERSION},
            "index": {
                "name": self.settings.index_name,
                "version": self.settings.index_version,
                "embedding": self.settings.embedding.descriptor(),
            },
            "aggregate": aggregate,
            "slices": {name: _aggregate(rows) for name, rows in sorted(by_slice.items())},
            "cases": case_results,
            "decision": "pass" if aggregate["authorization_leakage_count"] == 0 else "fail",
        }
