from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


def add_language_detection(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Add language detection to dataframe."""
    if not HAS_LANGDETECT:
        df["language"] = "unknown"
        return df

    def detect_lang(text: str) -> str:
        if not isinstance(text, str) or len(text) < 3:
            return "unknown"
        try:
            return detect(text)
        except (LangDetectException, Exception):
            return "unknown"

    df["language"] = df[text_col].map(detect_lang)
    return df


def add_temporal_features(df: pd.DataFrame, timestamp_col: str = "created_utc") -> pd.DataFrame:
    """Add temporal features from timestamp."""
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    df["year"] = df[timestamp_col].dt.year.astype("Int64")
    df["month"] = df[timestamp_col].dt.month.astype("Int64")
    df["day_of_week"] = df[timestamp_col].dt.day_name()
    df["date"] = df[timestamp_col].dt.date.astype(str)

    df["time_period"] = df["hour_of_day"].map(_categorize_time_period)
    return df


def _categorize_time_period(hour: int) -> str:
    """Categorize hour into time period."""
    if 0 <= hour <= 5:
        return "night"
    if 6 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 17:
        return "afternoon"
    return "evening"


def add_engagement_tier(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Categorize comments by engagement score."""
    if score_col not in df.columns:
        df["engagement_tier"] = "neutral"
        return df

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    quartiles = df[score_col].quantile([0.25, 0.5, 0.75])

    def tier(score):
        if score < quartiles[0.25]:
            return "low"
        if score < quartiles[0.5]:
            return "medium-low"
        if score < quartiles[0.75]:
            return "medium-high"
        return "high"

    df["engagement_tier"] = df[score_col].map(tier)
    return df


def add_depth_tier(df: pd.DataFrame, depth_col: str = "depth") -> pd.DataFrame:
    """Categorize comments by thread depth."""
    if depth_col not in df.columns:
        df["depth_tier"] = "root"
        return df

    def tier(depth):
        if depth == 0:
            return "root"
        if depth <= 2:
            return "shallow"
        if depth <= 5:
            return "medium"
        return "deep"

    df["depth_tier"] = df[depth_col].map(tier)
    return df


def enrich_with_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all metadata enrichment steps."""
    df = add_language_detection(df)
    df = add_temporal_features(df)
    df = add_engagement_tier(df)
    df = add_depth_tier(df)
    return df
