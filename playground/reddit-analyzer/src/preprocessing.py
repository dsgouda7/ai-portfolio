from __future__ import annotations

import re
from typing import Iterable, List, Set

import pandas as pd
from nltk import download as nltk_download
from nltk.sentiment import SentimentIntensityAnalyzer

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


def _ensure_vader() -> None:
    nltk_download("vader_lexicon", quiet=True)


def clean_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_tokens(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(text)]


def preprocess(df: pd.DataFrame, lexicon: Set[str]) -> pd.DataFrame:
    _ensure_vader()
    analyzer = SentimentIntensityAnalyzer()

    work = df.copy()
    work["clean_text"] = work["body"].fillna("").map(clean_text)
    work["tokens"] = work["clean_text"].map(extract_tokens)

    scores = work["clean_text"].map(analyzer.polarity_scores)
    work["sentiment_neg"] = scores.map(lambda s: float(s["neg"]))
    work["sentiment_neu"] = scores.map(lambda s: float(s["neu"]))
    work["sentiment_pos"] = scores.map(lambda s: float(s["pos"]))
    work["sentiment_compound"] = scores.map(lambda s: float(s["compound"]))

    work["lexicon_hits"] = work["tokens"].map(
        lambda toks: sorted({t for t in toks if t in lexicon})
    )
    work["lexicon_hit_count"] = work["lexicon_hits"].map(len)
    return work


def filter_negative(df: pd.DataFrame, threshold: float = -0.2) -> pd.DataFrame:
    return df[df["sentiment_compound"] <= threshold].reset_index(drop=True)
