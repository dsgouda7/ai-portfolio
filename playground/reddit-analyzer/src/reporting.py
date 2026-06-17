from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
from scipy.stats import kruskal


def _hour_bucket(hour: int) -> str:
    if 0 <= hour <= 5:
        return "night"
    if 6 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 17:
        return "afternoon"
    return "evening"


def _safe_kruskal(groups: List[pd.Series]) -> float:
    valid = [g for g in groups if len(g) > 0]
    if len(valid) < 2:
        return float("nan")
    try:
        stat = kruskal(*valid)
        return float(stat.pvalue)
    except ValueError:
        # SciPy raises when all compared values are identical.
        return float("nan")


def build_safety_audit_report(
    clustered_df: pd.DataFrame,
    cluster_keywords: Dict[int, List[str]],
    cluster_labels: Dict[int, Dict],
    drift_metrics,
    report_path: Path,
) -> str:
    work = clustered_df.copy()
    work = work[work["cluster_id"] != -1].copy()

    if work.empty:
        content = "# Safety Audit Report\n\nNo clusters identified after negative sentiment filtering."
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(content, encoding="utf-8")
        return content

    work["hour_bucket"] = work["hour_of_day"].map(_hour_bucket)

    by_cluster = (
        work.groupby("cluster_id")
        .agg(
            comments=("comment_id", "count"),
            avg_sentiment=("sentiment_compound", "mean"),
            avg_depth=("depth", "mean"),
            avg_score=("score", "mean"),
            peak_hour=("hour_of_day", lambda x: int(pd.Series.mode(x).iloc[0])),
        )
        .sort_values("comments", ascending=False)
    )

    depth_groups = [g["depth"] for _, g in work.groupby("cluster_id")]
    sentiment_groups = [g["sentiment_compound"] for _, g in work.groupby("cluster_id")]
    depth_p = _safe_kruskal(depth_groups)
    sentiment_p = _safe_kruskal(sentiment_groups)

    corr = work[["cluster_id", "depth", "sentiment_compound", "hour_of_day"]].corr(
        method="spearman"
    )

    lines = []
    lines.append("# Safety Audit Report")
    lines.append("")
    lines.append(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Total clustered negative comments: {len(work):,}")
    lines.append(f"- Clusters discovered: {work['cluster_id'].nunique()}")
    lines.append("")
    lines.append("## Cluster-Metadata Correlation")
    lines.append("")
    lines.append("Spearman correlation matrix:")
    lines.append("")
    lines.append(corr.to_markdown())
    lines.append("")
    lines.append("Kruskal-Wallis p-values across clusters:")
    lines.append("")
    lines.append(f"- Thread depth variance by cluster: {depth_p:.4g}")
    lines.append(f"- Sentiment variance by cluster: {sentiment_p:.4g}")
    lines.append("")
    lines.append("## Detection Drift / Coverage Gap")
    lines.append("")
    lines.append(f"- Baseline lexicon size: {drift_metrics.baseline_lexicon_size}")
    lines.append(f"- Cluster-discovered keyword size: {drift_metrics.discovered_keywords_size}")
    lines.append(f"- Overlap count: {drift_metrics.overlap_count}")
    lines.append(f"- Coverage: {drift_metrics.coverage:.2%}")
    lines.append(f"- Coverage gap: {drift_metrics.coverage_gap:.2%}")
    lines.append(f"- Detection drift: {drift_metrics.detection_drift:.2%}")
    lines.append("")
    lines.append("## Cluster Summaries")
    lines.append("")

    for cid, row in by_cluster.head(15).iterrows():
        meta = cluster_labels.get(int(cid), {})
        label = meta.get("cluster_label", "unlabeled")
        summary = meta.get("risk_summary", "")
        kws = cluster_keywords.get(int(cid), [])[:25]
        lines.append(f"### Cluster {int(cid)} - {label}")
        lines.append("")
        lines.append(f"- Comments: {int(row['comments'])}")
        lines.append(f"- Avg sentiment: {float(row['avg_sentiment']):.3f}")
        lines.append(f"- Avg depth: {float(row['avg_depth']):.2f}")
        lines.append(f"- Avg score: {float(row['avg_score']):.2f}")
        lines.append(f"- Peak hour (UTC): {int(row['peak_hour'])}")
        if summary:
            lines.append(f"- Risk summary: {summary}")
        lines.append(f"- Top keywords: {', '.join(kws)}")
        lines.append("")

    lines.append("## Coverage Gap Highlights")
    lines.append("")
    lines.append(
        "- Missing baseline terms (sample): "
        + ", ".join(drift_metrics.missing_baseline_terms[:50])
    )
    lines.append(
        "- Novel terms not in baseline (sample): "
        + ", ".join(drift_metrics.novel_terms[:50])
    )
    lines.append("")

    report = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report
