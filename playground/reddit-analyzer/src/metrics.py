from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


@dataclass
class DriftMetrics:
    baseline_lexicon_size: int
    discovered_keywords_size: int
    overlap_count: int
    coverage: float
    coverage_gap: float
    detection_drift: float
    missing_baseline_terms: List[str]
    novel_terms: List[str]


def compute_detection_drift(
    cluster_keywords: Dict[int, List[str]],
    baseline_lexicon: Set[str],
) -> DriftMetrics:
    discovered = {kw.lower() for kws in cluster_keywords.values() for kw in kws}
    overlap = discovered & baseline_lexicon
    missing = sorted(baseline_lexicon - discovered)
    novel = sorted(discovered - baseline_lexicon)

    coverage = len(overlap) / max(1, len(baseline_lexicon))
    coverage_gap = 1.0 - coverage
    detection_drift = len(novel) / max(1, len(discovered))

    return DriftMetrics(
        baseline_lexicon_size=len(baseline_lexicon),
        discovered_keywords_size=len(discovered),
        overlap_count=len(overlap),
        coverage=coverage,
        coverage_gap=coverage_gap,
        detection_drift=detection_drift,
        missing_baseline_terms=missing,
        novel_terms=novel,
    )
