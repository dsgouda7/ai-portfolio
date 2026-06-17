from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PipelineConfig:
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs")
    report_dir: Path = Path("reports")

    subreddits: List[str] = field(default_factory=lambda: ["politics", "news"])
    keywords: List[str] = field(
        default_factory=lambda: [
            "immigrant",
            "crime",
            "minority",
            "religion",
            "gender",
            "race",
        ]
    )

    limit_per_subreddit: int = 300
    sentiment_threshold: float = -0.20
    min_cluster_size: int = 12
    min_samples: int = 4
    max_keywords_per_cluster: int = 200

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_cpu_only: bool = True
