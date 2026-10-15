"""Shared deterministic fixtures for the RAG learning sequence."""

from __future__ import annotations

import random
import re
from typing import Iterable

import numpy as np
import torch

SEED = 17

COURSE_CAVEAT = (
    "These fixtures prove pipeline mechanics on a tiny deterministic corpus. "
    "They do not establish production thresholds, model quality, or security readiness."
)

DOCUMENTS = [
    {
        "id": "policy-ai-usage-r4",
        "title": "Public AI Usage",
        "text": "Public AI tools may receive public material only. Draft manuscripts and personal data are prohibited.",
        "tenant": "riverside",
        "groups": {"employee"},
        "revision": 4,
    },
    {
        "id": "policy-rights-17-r3",
        "title": "Copyright and Licensing",
        "text": "RIGHTS-17 requires a rights review before translated excerpts are licensed to an external publisher.",
        "tenant": "riverside",
        "groups": {"legal"},
        "revision": 3,
    },
    {
        "id": "policy-parental-leave-r2",
        "title": "Parental Leave",
        "text": "Employees welcoming a new child receive sixteen weeks of paid parental leave.",
        "tenant": "riverside",
        "groups": {"employee"},
        "revision": 2,
    },
    {
        "id": "policy-benefits-r5",
        "title": "Employee Benefits",
        "text": "Support is available for health insurance, dental care, and retirement contributions.",
        "tenant": "riverside",
        "groups": {"employee"},
        "revision": 5,
    },
    {
        "id": "policy-procurement-r7",
        "title": "Procurement",
        "text": "FIN-42 requires three supplier quotes for purchases above ten thousand dollars.",
        "tenant": "riverside",
        "groups": {"finance"},
        "revision": 7,
    },
    {
        "id": "policy-remote-work-r6",
        "title": "Remote Work",
        "text": "Remote employees may work outside the country for up to twenty business days with manager approval.",
        "tenant": "riverside",
        "groups": {"employee"},
        "revision": 6,
    },
]

RETRIEVAL_QUERIES = [
    {
        "id": "q-rights-code",
        "question": "What does RIGHTS-17 require?",
        "gold_ids": {"policy-rights-17-r3"},
        "groups": {"legal"},
        "kind": "exact code",
    },
    {
        "id": "q-parental-paraphrase",
        "question": "What support is available when welcoming a baby?",
        "gold_ids": {"policy-parental-leave-r2"},
        "groups": {"employee"},
        "kind": "paraphrase",
    },
    {
        "id": "q-public-ai",
        "question": "Can I paste a draft manuscript into a public chatbot?",
        "gold_ids": {"policy-ai-usage-r4"},
        "groups": {"employee"},
        "kind": "policy wording",
    },
    {
        "id": "q-procurement",
        "question": "How many supplier quotes does FIN-42 require?",
        "gold_ids": {"policy-procurement-r7"},
        "groups": {"finance"},
        "kind": "code plus detail",
    },
    {
        "id": "q-remote",
        "question": "How long may an employee work abroad?",
        "gold_ids": {"policy-remote-work-r6"},
        "groups": {"employee"},
        "kind": "paraphrase",
    },
    {
        "id": "q-unsupported",
        "question": "What is the dental deductible for dependants?",
        "gold_ids": set(),
        "groups": {"employee"},
        "kind": "unsupported",
    },
]

RAG_EVAL_CASES = [
    {
        "id": "react-tools",
        "question": "How does ReAct combine reasoning and acting?",
        "gold_context": "ReAct interleaves thought and action steps and can call tools such as search or a calculator.",
        "reference": "It alternates reasoning with actions and uses tools such as search or a calculator.",
        "required_points": {"interleaves reasoning and actions", "uses tools"},
    },
    {
        "id": "rights-review",
        "question": "What must happen before translated excerpts are licensed?",
        "gold_context": DOCUMENTS[1]["text"],
        "reference": "A rights review must happen before licensing translated excerpts.",
        "required_points": {"rights review", "before licensing"},
    },
    {
        "id": "parental-duration",
        "question": "How much paid parental leave is available?",
        "gold_context": DOCUMENTS[2]["text"],
        "reference": "Employees welcoming a new child receive sixteen weeks of paid parental leave.",
        "required_points": {"sixteen weeks", "paid parental leave"},
    },
    {
        "id": "public-ai-boundary",
        "question": "May a draft manuscript be sent to a public AI tool?",
        "gold_context": DOCUMENTS[0]["text"],
        "reference": "No. Draft manuscripts are prohibited in public AI tools.",
        "required_points": {"no", "draft manuscripts prohibited"},
    },
]

SEMANTIC_FEATURES = {
    "baby": "family",
    "child": "family",
    "welcoming": "family",
    "parental": "family",
    "leave": "leave",
    "weeks": "leave",
    "chatbot": "ai",
    "ai": "ai",
    "public": "ai",
    "manuscript": "ai",
    "copyright": "rights",
    "licensing": "rights",
    "licensed": "rights",
    "translated": "rights",
    "quotes": "procurement",
    "supplier": "procurement",
    "purchases": "procurement",
    "abroad": "remote",
    "country": "remote",
    "remote": "remote",
    "health": "benefits",
    "dental": "benefits",
    "retirement": "benefits",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def authorized_documents(groups: Iterable[str], tenant: str = "riverside") -> list[dict]:
    group_set = set(groups)
    return [
        document
        for document in DOCUMENTS
        if document["tenant"] == tenant and bool(document["groups"] & group_set)
    ]


def retrieval_case(case_id: str) -> dict:
    return next(case for case in RETRIEVAL_QUERIES if case["id"] == case_id)


def assert_fixture_contracts() -> None:
    document_ids = [document["id"] for document in DOCUMENTS]
    assert len(document_ids) == len(set(document_ids))
    assert all(case["gold_ids"] <= set(document_ids) for case in RETRIEVAL_QUERIES)
    assert all(case["required_points"] for case in RAG_EVAL_CASES)
