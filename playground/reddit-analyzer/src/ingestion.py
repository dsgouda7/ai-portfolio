from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import praw
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import list_datasets
from praw.models import MoreComments


@dataclass
class IngestionConfig:
    subreddits: Iterable[str]
    keywords: Iterable[str]
    limit_per_subreddit: int = 300
    time_filter: str = "month"


@dataclass
class HFDumpConfig:
    dataset_ids: Optional[Iterable[str]] = None
    max_datasets: int = 12
    max_rows_per_dataset: int = 25000


@dataclass
class LocalDumpConfig:
    dump_paths: Iterable[str]


def _build_reddit_client() -> praw.Reddit:
    load_dotenv()
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "reddit-safety-audit:v1.0")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Reddit API credentials. Fill .env from .env.example before ingestion."
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )


def _flatten_comment(comment, submission, keyword: str) -> Dict:
    body = (comment.body or "").strip()
    created = datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)
    return {
        "comment_id": comment.id,
        "post_id": submission.id,
        "post_title": submission.title,
        "subreddit": str(submission.subreddit).lower(),
        "keyword": keyword,
        "body": body,
        "score": int(comment.score or 0),
        "depth": int(getattr(comment, "depth", 0) or 0),
        "author": str(comment.author) if comment.author else "[deleted]",
        "created_utc": created.isoformat(),
        "hour_of_day": int(created.hour),
        "permalink": f"https://www.reddit.com{comment.permalink}",
    }


def fetch_comments(config: IngestionConfig) -> pd.DataFrame:
    reddit = _build_reddit_client()
    rows: List[Dict] = []
    seen = set()

    for subreddit_name in config.subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for keyword in config.keywords:
            for submission in subreddit.search(
                query=keyword,
                sort="new",
                time_filter=config.time_filter,
                limit=config.limit_per_subreddit,
            ):
                submission.comments.replace_more(limit=0)
                for c in submission.comments.list():
                    if isinstance(c, MoreComments):
                        continue
                    if c.id in seen:
                        continue
                    payload = _flatten_comment(c, submission, keyword)
                    if payload["body"]:
                        rows.append(payload)
                        seen.add(c.id)

    return pd.DataFrame(rows)


def discover_hf_reddit_datasets(max_datasets: int = 12) -> List[str]:
    discovered: List[str] = []
    for ds in list_datasets(search="reddit", limit=300):
        ds_id = ds.id
        tags = {t.lower() for t in (ds.tags or [])}
        looks_reddit = ("reddit" in ds_id.lower()) or ("reddit" in tags)
        if looks_reddit:
            discovered.append(ds_id)
        if len(discovered) >= max_datasets:
            break
    return discovered


def _guess_text(row: Dict) -> str:
    for key in (
        "body",
        "text",
        "comment",
        "comment_text",
        "comment_body",
        "content",
        "selftext",
        "post_text",
        "prompt",
        "completion",
        "chosen",
        "rejected",
        "message",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    conversations = row.get("conversations")
    if isinstance(conversations, list):
        parts = []
        for item in conversations:
            if isinstance(item, dict):
                text_val = item.get("value") or item.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        if parts:
            return " ".join(parts)

    return ""


def _guess_subreddit(row: Dict) -> str:
    for key in (
        "subreddit",
        "subreddit_name",
        "subreddit.name",
        "community",
        "forum",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return "unknown"


def _guess_created(row: Dict) -> str:
    for key in ("created_utc", "created_at", "timestamp", "date"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, (int, float)) and value > 0:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    return datetime.now(tz=timezone.utc).isoformat()


def _normalize_rows(records: Iterable[Dict], source_name: str) -> List[Dict]:
    rows: List[Dict] = []
    for row in records:
        body = _guess_text(row)
        if not body:
            continue
        created = _guess_created(row)
        try:
            hour = datetime.fromisoformat(created.replace("Z", "+00:00")).hour
        except Exception:
            hour = 0

        rows.append(
            {
                "comment_id": str(row.get("id") or row.get("comment_id") or ""),
                "post_id": str(row.get("link_id") or row.get("post_id") or ""),
                "post_title": str(row.get("title") or row.get("post_title") or ""),
                "subreddit": _guess_subreddit(row),
                "keyword": str(row.get("keyword") or "hf_dump"),
                "body": body,
                "score": int(row.get("score") or row.get("ups") or 0),
                "depth": int(row.get("depth") or row.get("level") or 0),
                "author": str(row.get("author") or "unknown"),
                "created_utc": created,
                "hour_of_day": int(hour),
                "permalink": str(row.get("permalink") or ""),
                "source": source_name,
            }
        )
    return rows


def fetch_from_hf_dumps(config: HFDumpConfig) -> pd.DataFrame:
    dataset_ids = list(config.dataset_ids or [])
    if not dataset_ids:
        dataset_ids = discover_hf_reddit_datasets(max_datasets=config.max_datasets)

    all_rows: List[Dict] = []
    seen = set()
    split_candidates = ("train", "validation", "test")

    for dataset_id in dataset_ids[: config.max_datasets]:
        records: List[Dict] = []
        loaded = False
        for split in split_candidates:
            try:
                stream = load_dataset(dataset_id, split=split, streaming=True)
                for idx, row in enumerate(stream):
                    if idx >= config.max_rows_per_dataset:
                        break
                    row_id = str(row.get("id") or f"{dataset_id}:{split}:{idx}")
                    if row_id in seen:
                        continue
                    seen.add(row_id)
                    records.append(dict(row))
                loaded = True
                break
            except Exception:
                continue

        if loaded and records:
            all_rows.extend(_normalize_rows(records, source_name=f"hf:{dataset_id}"))

    return pd.DataFrame(all_rows)


def _load_local_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported dump format: {path}")


def fetch_from_local_dumps(config: LocalDumpConfig) -> pd.DataFrame:
    all_rows: List[Dict] = []
    for p in config.dump_paths:
        path = Path(p)
        if not path.exists() or not path.is_file():
            continue
        try:
            df = _load_local_file(path)
            all_rows.extend(_normalize_rows(df.to_dict(orient="records"), source_name=str(path)))
        except Exception:
            continue
    return pd.DataFrame(all_rows)


def save_raw(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"reddit_comments_{ts}.jsonl"
    parquet_path = output_dir / f"reddit_comments_{ts}.parquet"

    with json_path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    df.to_parquet(parquet_path, index=False)
    return {"jsonl": json_path, "parquet": parquet_path}


def load_latest_raw(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("reddit_comments_*.parquet"))
    if not files:
        raise FileNotFoundError("No raw parquet files found in data/raw.")
    return pd.read_parquet(files[-1])
