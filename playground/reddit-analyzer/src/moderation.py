from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd
import requests


@dataclass
class ModerationConfig:
    use_llm: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M"
    timeout_seconds: int = 45


def _heuristic_label(cluster_keywords: List[str]) -> Dict:
    head = cluster_keywords[:6]
    label = " | ".join(head[:2]) if head else "uncategorized toxicity"
    return {
        "cluster_label": label,
        "risk_summary": "Heuristic label fallback (LLM unavailable).",
        "stereotype_terms": head,
    }


def _call_ollama(prompt: str, config: ModerationConfig) -> Dict:
    url = f"{config.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": config.ollama_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
        },
    }
    resp = requests.post(url, json=payload, timeout=config.timeout_seconds)
    resp.raise_for_status()
    data = resp.json()
    return json.loads(data.get("response", "{}"))


def label_clusters(
    clustered_df: pd.DataFrame,
    cluster_keywords: Dict[int, List[str]],
    config: ModerationConfig,
) -> Dict[int, Dict]:
    labels: Dict[int, Dict] = {}

    if not config.use_llm:
        for cid, kws in cluster_keywords.items():
            labels[cid] = _heuristic_label(kws)
        return labels

    for cid, kws in cluster_keywords.items():
        sample_comments = (
            clustered_df[clustered_df["cluster_id"] == cid]["clean_text"]
            .head(8)
            .tolist()
        )
        prompt = (
            "You are a social media safety auditor. Return strict JSON with keys "
            "cluster_label (string), risk_summary (string), stereotype_terms (array of strings).\n"
            "Use neutral wording. No slurs in output unless already in keywords.\n"
            f"Cluster ID: {cid}\n"
            f"Top keywords: {kws[:40]}\n"
            f"Sample comments: {sample_comments}\n"
        )

        try:
            labels[cid] = _call_ollama(prompt, config)
            if not isinstance(labels[cid], dict) or "cluster_label" not in labels[cid]:
                labels[cid] = _heuristic_label(kws)
        except Exception:
            labels[cid] = _heuristic_label(kws)

    return labels


def default_moderation_config_from_env(use_llm: bool = True) -> ModerationConfig:
    return ModerationConfig(
        use_llm=use_llm,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M"),
    )
