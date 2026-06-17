from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.analysis import ClusterConfig, cluster_comments, extract_cluster_keywords
from src.config import PipelineConfig
from src.ingestion import (
    HFDumpConfig,
    IngestionConfig,
    LocalDumpConfig,
    fetch_comments,
    fetch_from_hf_dumps,
    fetch_from_local_dumps,
    load_latest_raw,
    save_raw,
)
from src.lexicon import LexiconManager
from src.metadata_enrichment import enrich_with_metadata
from src.metrics import compute_detection_drift
from src.moderation import default_moderation_config_from_env, label_clusters
from src.preprocessing import filter_negative, preprocess
from src.reporting import build_safety_audit_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reddit Social Safety Audit Pipeline")
    parser.add_argument(
        "--source",
        choices=["latest", "praw", "hf", "local", "mixed"],
        default="hf",
        help="Data source mode: latest (reuse), praw (Reddit API), hf (Hugging Face dumps), local (local files), mixed (HF + local).",
    )
    parser.add_argument("--subreddits", nargs="*", default=["politics", "news"])
    parser.add_argument("--keywords", nargs="*", default=["immigrant", "race", "religion"])
    parser.add_argument("--limit-per-subreddit", type=int, default=300)
    parser.add_argument("--hf-datasets", nargs="*", default=[])
    parser.add_argument("--hf-max-datasets", type=int, default=12)
    parser.add_argument("--hf-max-rows", type=int, default=25000)
    parser.add_argument("--local-dump-paths", nargs="*", default=[])
    parser.add_argument("--sentiment-threshold", type=float, default=-0.2)
    parser.add_argument("--min-cluster-size", type=int, default=12)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--max-keywords", type=int, default=200)
    parser.add_argument("--disable-llm-labeling", action="store_true")
    parser.add_argument("--hf-lexicon", default="SEACrowd/tgl_profanity")
    return parser.parse_args()


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        subreddits=args.subreddits,
        keywords=args.keywords,
        limit_per_subreddit=args.limit_per_subreddit,
        sentiment_threshold=args.sentiment_threshold,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        max_keywords_per_cluster=args.max_keywords,
    )

    for d in [cfg.raw_dir, cfg.processed_dir, cfg.output_dir, cfg.report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.source == "latest":
        raw_df = load_latest_raw(cfg.raw_dir)
    elif args.source == "praw":
        ingestion_cfg = IngestionConfig(
            subreddits=cfg.subreddits,
            keywords=cfg.keywords,
            limit_per_subreddit=cfg.limit_per_subreddit,
        )
        raw_df = fetch_comments(ingestion_cfg)
    elif args.source == "hf":
        raw_df = fetch_from_hf_dumps(
            HFDumpConfig(
                dataset_ids=args.hf_datasets,
                max_datasets=args.hf_max_datasets,
                max_rows_per_dataset=args.hf_max_rows,
            )
        )
    elif args.source == "local":
        raw_df = fetch_from_local_dumps(LocalDumpConfig(dump_paths=args.local_dump_paths))
    else:
        hf_df = fetch_from_hf_dumps(
            HFDumpConfig(
                dataset_ids=args.hf_datasets,
                max_datasets=args.hf_max_datasets,
                max_rows_per_dataset=args.hf_max_rows,
            )
        )
        local_df = fetch_from_local_dumps(LocalDumpConfig(dump_paths=args.local_dump_paths))
        raw_df = hf_df if local_df.empty else local_df if hf_df.empty else pd.concat([hf_df, local_df], ignore_index=True)

    if raw_df.empty:
        raise RuntimeError(
            "No records ingested. For --source hf, allow discovery or provide --hf-datasets. "
            "For --source local, provide --local-dump-paths."
        )

    save_raw(raw_df, cfg.raw_dir)

    lexicon_manager = LexiconManager()
    lexicon = lexicon_manager.build_lexicon(
        include_hf_seed=True,
        dataset_name=args.hf_lexicon,
    )

    enriched = preprocess(raw_df, lexicon)
    enriched.to_parquet(cfg.processed_dir / "comments_enriched.parquet", index=False)

    negative = filter_negative(enriched, threshold=cfg.sentiment_threshold)
    negative.to_parquet(cfg.processed_dir / "comments_negative.parquet", index=False)

    clustered = cluster_comments(
        negative,
        ClusterConfig(
            embedding_model=cfg.embedding_model,
            min_cluster_size=cfg.min_cluster_size,
            min_samples=cfg.min_samples,
            max_keywords_per_cluster=cfg.max_keywords_per_cluster,
        ),
    )
    clustered = enrich_with_metadata(clustered)
    clustered.to_parquet(cfg.processed_dir / "comments_clustered.parquet", index=False)

    cluster_keywords = extract_cluster_keywords(
        clustered,
        max_keywords_per_cluster=cfg.max_keywords_per_cluster,
    )
    save_json(cluster_keywords, cfg.output_dir / "cluster_keywords.json")

    moderation_cfg = default_moderation_config_from_env(
        use_llm=not args.disable_llm_labeling
    )
    cluster_labels = label_clusters(clustered, cluster_keywords, moderation_cfg)
    save_json(cluster_labels, cfg.output_dir / "cluster_labels.json")

    drift = compute_detection_drift(cluster_keywords, lexicon)
    save_json(asdict(drift), cfg.output_dir / "detection_drift_metrics.json")

    build_safety_audit_report(
        clustered_df=clustered,
        cluster_keywords=cluster_keywords,
        cluster_labels=cluster_labels,
        drift_metrics=drift,
        report_path=cfg.report_dir / "safety_audit_report.md",
    )

    print("Pipeline complete.")
    print(f"Report: {cfg.report_dir / 'safety_audit_report.md'}")


if __name__ == "__main__":
    main()
