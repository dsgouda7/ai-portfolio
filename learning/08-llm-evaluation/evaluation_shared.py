"""Shared deterministic fixtures for the four-notebook evaluation track."""

from __future__ import annotations

import random
import re

import numpy as np
import torch

SEED = 23

COURSE_CAVEAT = (
    "The local scorers are transparent teaching instruments on a small labeled fixture. "
    "Their thresholds are not production guarantees; validate the real evaluator on representative data."
)

RIVERSIDE_CASES = [
    {
        "id": "pendant",
        "query": "Why does Mei-Lin keep the pendant?",
        "reference": "The pendant hides a letter proving Mei-Lin's lineage and lets her reclaim the family trade route.",
        "candidate": "A letter inside the pendant confirms Mei-Lin's family line, allowing her to recover the trade route.",
        "fluent_wrong": "The pendant is a jade heirloom that symbolizes her family's social status.",
        "context": "Inside the pendant, Mei-Lin finds a letter proving her lineage and restoring her claim to the family trade route.",
        "slice": "rare entity",
    },
    {
        "id": "award",
        "query": "What recognition did Elena Marchetti receive in 2019?",
        "reference": "Elena Marchetti was longlisted for the Booker Prize in 2019.",
        "candidate": "In 2019, Elena Marchetti made the Booker Prize longlist.",
        "fluent_wrong": "Elena Marchetti won the Booker Prize in 2019 for The Silence of Bridges.",
        "context": "Elena Marchetti was longlisted for the Booker Prize in 2019.",
        "slice": "relation",
    },
    {
        "id": "react",
        "query": "How does ReAct combine reasoning and acting?",
        "reference": "ReAct interleaves thought and action steps and uses tools such as search or a calculator.",
        "candidate": "ReAct alternates reasoning with tool actions such as search and calculation.",
        "fluent_wrong": "ReAct stores every reasoning step in a vector database before acting.",
        "context": "ReAct interleaves thought and action steps and can call tools such as search or a calculator.",
        "slice": "technical",
    },
    {
        "id": "rights",
        "query": "What does RIGHTS-17 require?",
        "reference": "RIGHTS-17 requires a rights review before translated excerpts are licensed externally.",
        "candidate": "A rights review is required before licensing translated excerpts to an external publisher.",
        "fluent_wrong": "RIGHTS-17 allows automatic licensing after a translation quality check.",
        "context": "RIGHTS-17 requires a rights review before translated excerpts are licensed to an external publisher.",
        "slice": "critical policy",
    },
]

SEMANTIC_GROUPS = {
    "lineage": "family-proof",
    "family": "family-proof",
    "letter": "document",
    "document": "document",
    "reclaim": "recover",
    "recover": "recover",
    "restoring": "recover",
    "longlisted": "longlist",
    "longlist": "longlist",
    "reasoning": "reason",
    "thought": "reason",
    "actions": "act",
    "acting": "act",
    "tools": "tool",
    "search": "tool",
    "calculator": "tool",
    "calculation": "tool",
    "rights": "rights",
    "licensing": "license",
    "licensed": "license",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalized_terms(text: str) -> list[str]:
    return [SEMANTIC_GROUPS.get(token, token) for token in tokenize(text)]


def assert_fixture_contracts() -> None:
    ids = [case["id"] for case in RIVERSIDE_CASES]
    assert len(ids) == len(set(ids))
    required = {"id", "query", "reference", "candidate", "fluent_wrong", "context", "slice"}
    assert all(required <= set(case) for case in RIVERSIDE_CASES)
