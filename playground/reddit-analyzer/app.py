from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")

OUTPUT_DIR = Path("outputs")
PROCESSED_DIR = Path("data/processed")

# Cache for loaded data
_cache = {}


def get_cluster_data():
    """Load and cache cluster data."""
    if "cluster_data" in _cache:
        return _cache["cluster_data"]

    keywords_path = OUTPUT_DIR / "cluster_keywords.json"
    labels_path = OUTPUT_DIR / "cluster_labels.json"
    clustered_path = PROCESSED_DIR / "comments_clustered.parquet"

    with keywords_path.open(encoding="utf-8") as f:
        cluster_keywords = json.load(f)
        cluster_keywords = {int(k): v for k, v in cluster_keywords.items()}

    with labels_path.open(encoding="utf-8") as f:
        cluster_labels = json.load(f)
        cluster_labels = {int(k): v for k, v in cluster_labels.items()}

    clustered_df = pd.read_parquet(clustered_path)
    cluster_counts = clustered_df[clustered_df["cluster_id"] != -1].groupby("cluster_id").size()

    data = {
        "cluster_keywords": cluster_keywords,
        "cluster_labels": cluster_labels,
        "cluster_counts": cluster_counts.to_dict(),
        "clustered_df": clustered_df,
    }
    _cache["cluster_data"] = data
    return data


def compute_word_frequencies(keywords: list[str]) -> list[dict]:
    """Compute frequencies for keywords."""
    counter = Counter(keywords)
    return [
        {"word": word, "count": count}
        for word, count in counter.most_common(50)
    ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/clusters")
def get_clusters():
    """Get all clusters with metadata."""
    data = get_cluster_data()
    result = []

    for cluster_id, keywords in data["cluster_keywords"].items():
        if cluster_id == -1:
            continue
        result.append({
            "id": cluster_id,
            "label": data["cluster_labels"].get(cluster_id, "Unlabeled"),
            "count": data["cluster_counts"].get(cluster_id, 0),
            "keywords": keywords[:50],
        })

    return jsonify(sorted(result, key=lambda x: x["count"], reverse=True))


@app.route("/api/cluster/<int:cluster_id>/keywords")
def get_cluster_keywords(cluster_id):
    """Get keywords for a specific cluster."""
    data = get_cluster_data()
    keywords = data["cluster_keywords"].get(cluster_id, [])
    frequencies = compute_word_frequencies(keywords)

    return jsonify({
        "cluster_id": cluster_id,
        "label": data["cluster_labels"].get(cluster_id, "Unlabeled"),
        "keywords": frequencies,
    })


@app.route("/api/cluster/<int:cluster_id>/metadata")
def get_cluster_metadata(cluster_id):
    """Get metadata distributions for a cluster."""
    data = get_cluster_data()
    df = data["clustered_df"]

    # Filter to this cluster
    cluster_df = df[df["cluster_id"] == cluster_id]

    if cluster_df.empty:
        return jsonify({"error": "Cluster not found"}), 404

    metadata = {
        "cluster_id": cluster_id,
        "count": len(cluster_df),
        "sources": cluster_df["source"].value_counts().to_dict() if "source" in cluster_df.columns else {},
        "subreddits": cluster_df["subreddit"].value_counts().to_dict() if "subreddit" in cluster_df.columns else {},
        "languages": cluster_df["language"].value_counts().to_dict() if "language" in cluster_df.columns else {},
        "months": cluster_df["month"].value_counts().sort_index().to_dict() if "month" in cluster_df.columns else {},
        "time_periods": cluster_df["time_period"].value_counts().to_dict() if "time_period" in cluster_df.columns else {},
        "engagement_tiers": cluster_df["engagement_tier"].value_counts().to_dict() if "engagement_tier" in cluster_df.columns else {},
        "depth_tiers": cluster_df["depth_tier"].value_counts().to_dict() if "depth_tier" in cluster_df.columns else {},
        "avg_sentiment": float(cluster_df["sentiment_compound"].mean()) if "sentiment_compound" in cluster_df.columns else 0,
        "avg_score": int(cluster_df["score"].mean()) if "score" in cluster_df.columns else 0,
        "avg_depth": int(cluster_df["depth"].mean()) if "depth" in cluster_df.columns else 0,
    }

    return jsonify(metadata)


@app.route("/api/keyword-correlations")
def get_keyword_correlations():
    """Get correlations between keywords and metadata features."""
    data = get_cluster_data()
    df = data["clustered_df"]

    # Get language filter if specified
    language = request.args.get("language")
    source = request.args.get("source")
    time_period = request.args.get("time_period")

    if language and language != "all":
        df = df[df.get("language") == language]
    if source and source != "all":
        df = df[df["source"] == source]
    if time_period and time_period != "all":
        df = df[df.get("time_period") == time_period]

    # Count keyword appearances by metadata
    correlations = {}

    for cluster_id, keywords in data["cluster_keywords"].items():
        if cluster_id == -1:
            continue

        cluster_df = df[df["cluster_id"] == cluster_id]
        if cluster_df.empty:
            continue

        correlations[cluster_id] = {
            "label": data["cluster_labels"].get(cluster_id, "Unlabeled"),
            "count": len(cluster_df),
            "language_dist": cluster_df.get("language", pd.Series()).value_counts().head(5).to_dict(),
            "source_dist": cluster_df["source"].value_counts().head(5).to_dict(),
            "time_period_dist": cluster_df.get("time_period", pd.Series()).value_counts().to_dict(),
            "engagement_dist": cluster_df.get("engagement_tier", pd.Series()).value_counts().to_dict(),
        }

    return jsonify(correlations)


@app.route("/api/available-filters")
def get_available_filters():
    """Get available filter options for the UI."""
    data = get_cluster_data()
    df = data["clustered_df"]

    filters = {
        "languages": sorted([l for l in df.get("language", pd.Series()).unique() if pd.notna(l)]),
        "sources": sorted(df["source"].unique().tolist()),
        "time_periods": sorted([t for t in df.get("time_period", pd.Series()).unique() if pd.notna(t)]),
        "subreddits": sorted(df["subreddit"].unique().tolist()),
        "months": sorted([m for m in df.get("month", pd.Series()).unique() if pd.notna(m)]),
        "years": sorted([y for y in df.get("year", pd.Series()).unique() if pd.notna(y)]),
    }

    return jsonify(filters)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

