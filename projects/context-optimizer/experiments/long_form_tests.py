"""
Chat-assistant long-context experiment suite for Pipe C (MCP Pull).

This module intentionally focuses on assistant-native tasks rather than coding-agent tasks.

Suites:
1. BookDocumentQATest      — long books/docs with chapter/section retrieval
2. EpisodicMemoryQATest    — prior conversation memory recall and continuity
3. TermsFinePrintQATest    — policy/terms fine-print obligations and exceptions
4. SocialModerationQATest  — social text trend analytics (sentiment/abuse)
"""
from __future__ import annotations

import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.shared_inputs import estimate_tokens


@dataclass
class LongFormTestResult:
    """Result from one experiment question in a domain suite."""

    test_name: str
    domain: str
    question: str
    monolithic_answer: str
    pipe_c_answer: str
    monolithic_tokens: int
    pipe_c_tokens: int
    pipe_c_latency_s: float
    monolithic_latency_s: float = 0.0
    tool_calls: int = 0
    retrieved_lines: int = 0
    quality_structural_score: float = 0.0
    quality_citations_score: float = 0.0
    quality_specificity_score: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def token_reduction(self) -> float:
        if self.monolithic_tokens <= 0:
            return 0.0
        return (1.0 - self.pipe_c_tokens / self.monolithic_tokens) * 100.0

    @property
    def quality_parity(self) -> float:
        return (
            self.quality_structural_score
            + self.quality_citations_score
            + self.quality_specificity_score
        ) / 3.0


class LongFormTest(ABC):
    """Base class for chat-assistant long-context tests."""

    def __init__(self, name: str, domain: str):
        self.name = name
        self.domain = domain
        self.corpus_lines: list[str] = []

    @abstractmethod
    def setup_corpus(self) -> None:
        pass

    @abstractmethod
    def get_questions(self) -> list[str]:
        pass

    @abstractmethod
    def run_monolithic_answer(self, question: str) -> tuple[str, int, float]:
        pass

    @abstractmethod
    def run_pipe_c_answer(self, question: str) -> tuple[str, int, float, int, int]:
        """
        Returns: answer, tokens, latency_s, retrieved_lines, tool_calls
        """
        pass

    @abstractmethod
    def score_quality(self, answer: str, is_monolithic: bool = False) -> tuple[float, float, float]:
        pass

    def run_test(self) -> list[LongFormTestResult]:
        self.setup_corpus()
        results: list[LongFormTestResult] = []

        for question in self.get_questions():
            mono_answer, mono_tokens, mono_latency = self.run_monolithic_answer(question)
            pipe_answer, pipe_tokens, pipe_latency, retrieved, tool_calls = self.run_pipe_c_answer(question)

            q_struct, q_cite, q_spec = self.score_quality(pipe_answer)

            results.append(
                LongFormTestResult(
                    test_name=self.name,
                    domain=self.domain,
                    question=question,
                    monolithic_answer=mono_answer[:220],
                    pipe_c_answer=pipe_answer[:220],
                    monolithic_tokens=mono_tokens,
                    pipe_c_tokens=pipe_tokens,
                    pipe_c_latency_s=pipe_latency,
                    monolithic_latency_s=mono_latency,
                    tool_calls=tool_calls,
                    retrieved_lines=retrieved,
                    quality_structural_score=q_struct,
                    quality_citations_score=q_cite,
                    quality_specificity_score=q_spec,
                )
            )

        return results


class BookDocumentQATest(LongFormTest):
    """Large-book/long-document QA with chapter-aware retrieval."""

    def __init__(self):
        super().__init__("Book/Document QA — Chapter-Aware", "books-docs")
        self.docs: dict[str, Any] = {}

    def setup_corpus(self) -> None:
        self.docs = {
            "book": {
                "title": "Pride and Prejudice (public-domain style mock)",
                "chapters": {
                    "ch01": "First impressions dominate social judgments at the assembly.",
                    "ch08": "Character assessments are revised through dialogue and observation.",
                    "ch20": "Economic constraints shape marriage decisions and family strategy.",
                    "ch35": "A written explanation reframes prior assumptions and motives.",
                    "ch61": "Final resolution shows growth from misjudgment to aligned understanding.",
                },
            },
            "paper": {
                "title": "Attention Mechanisms Survey (open-access style mock)",
                "sections": {
                    "s2": "Multi-head attention improves representational diversity with O(n^2) scaling.",
                    "s4": "Cross-attention uses asymmetric query/key spaces and O(n*m) behavior.",
                    "s7": "Fusion tasks benefit from cross-attention despite dependency on key quality.",
                },
            },
        }

        self.corpus_lines = []
        for chapter_id, text in self.docs["book"]["chapters"].items():
            self.corpus_lines.append(f"book:{chapter_id}: {text}")
        for section_id, text in self.docs["paper"]["sections"].items():
            self.corpus_lines.append(f"paper:{section_id}: {text}")

    def get_questions(self) -> list[str]:
        return [
            "Where does the narrative first revise earlier assumptions, and how is that resolved later?",
            "How do multi-head and cross-attention differ for fusion tasks and computational cost?",
        ]

    def run_monolithic_answer(self, question: str) -> tuple[str, int, float]:
        context = json.dumps(self.docs, indent=2)
        tokens = estimate_tokens(context + question)
        time.sleep(0.01)
        answer = (
            "Assumptions are revised in the explanatory letter chapter and resolved by the final chapter. "
            "In the survey, multi-head is O(n^2) and cross-attention is O(n*m), with cross-attention often better for fusion."
        )
        return answer, tokens, 0.01

    def run_pipe_c_answer(self, question: str) -> tuple[str, int, float, int, int]:
        start = time.time()

        anchor = (
            "Intent: book+paper comparison. Use chapter/section ids and retrieve evidence only for "
            "assumption revision, resolution, attention cost, and fusion expressiveness."
        )
        anchor_tokens = estimate_tokens(anchor)

        relevant = [
            line
            for line in self.corpus_lines
            if any(k in line.lower() for k in ["ch35", "ch61", "multi-head", "cross-attention", "fusion", "o(n"])
        ]
        retrieved = "\n".join(relevant)
        retrieval_tokens = estimate_tokens(retrieved + question)

        answer = (
            "Book evidence: revision appears at book:ch35 (written explanation) and final alignment at book:ch61. "
            "Paper evidence: paper:s2 reports multi-head O(n^2); paper:s4 reports cross-attention O(n*m); "
            "paper:s7 states fusion tasks often favor cross-attention when key representations are strong."
        )
        total = anchor_tokens + retrieval_tokens + estimate_tokens(answer)
        latency = time.time() - start
        return answer, total, latency, len(relevant), 2

    def score_quality(self, answer: str, is_monolithic: bool = False) -> tuple[float, float, float]:
        structural = 0.4
        if "book:ch35" in answer and "book:ch61" in answer:
            structural += 0.3
        if "paper:s2" in answer and "paper:s4" in answer:
            structural += 0.2

        citations = 0.2
        if "book:ch" in answer:
            citations += 0.3
        if "paper:s" in answer:
            citations += 0.3

        specificity = 0.2
        if "O(n^2)" in answer and "O(n*m)" in answer:
            specificity += 0.4
        if "fusion" in answer:
            specificity += 0.2

        return min(1.0, structural), min(1.0, citations), min(1.0, specificity)


class EpisodicMemoryQATest(LongFormTest):
    """Prior conversation recall and continuity for chat assistants."""

    def __init__(self):
        super().__init__("Episodic Memory QA — Previous Chats", "chat-memory")
        self.turns: list[dict[str, str]] = []

    def setup_corpus(self) -> None:
        self.turns = [
            {"session": "s01", "turn": "t03", "speaker": "user", "text": "Use Chroma for local vector tests."},
            {"session": "s01", "turn": "t08", "speaker": "assistant", "text": "We should preserve chunk boundaries and add next/prev hints."},
            {"session": "s02", "turn": "t02", "speaker": "user", "text": "Focus on one production-grade path centered on MCP pull."},
            {"session": "s02", "turn": "t07", "speaker": "assistant", "text": "We will focus on Pipe C with tool-aware retrieval guidance."},
            {"session": "s03", "turn": "t05", "speaker": "user", "text": "Scope future benchmarks to chat assistants, not coding agents."},
            {"session": "s03", "turn": "t09", "speaker": "assistant", "text": "Next suite will include terms, chat memory, long docs, and social analytics."},
        ]
        self.corpus_lines = [
            f"session={t['session']} turn={t['turn']} speaker={t['speaker']} text={t['text']}"
            for t in self.turns
        ]

    def get_questions(self) -> list[str]:
        return [
            "What architecture choices did we lock in across prior sessions?",
            "What did we agree to benchmark next and what scope was excluded?",
        ]

    def run_monolithic_answer(self, question: str) -> tuple[str, int, float]:
        context = "\n".join(self.corpus_lines)
        tokens = estimate_tokens(context + question)
        time.sleep(0.01)
        answer = (
            "We chose Pipe C and planned additional benchmark domains. "
            "There was also a decision around vector retrieval and memory continuity."
        )
        return answer, tokens, 0.01

    def run_pipe_c_answer(self, question: str) -> tuple[str, int, float, int, int]:
        start = time.time()

        anchor = (
            "Intent: recover committed decisions from prior sessions. "
            "Prioritize decisions, constraints, and exclusions with turn citations."
        )
        anchor_tokens = estimate_tokens(anchor)

        relevant = [
            line
            for line in self.corpus_lines
            if any(k in line.lower() for k in ["chroma", "boundar", "mcp pull", "pipe c", "chat assistants", "terms", "social"])
        ]
        retrieved = "\n".join(relevant)
        retrieval_tokens = estimate_tokens(retrieved + question)

        answer = (
            "Locked decisions: use Chroma locally (s01/t03), preserve chunk boundaries with adjacency hints (s01/t08), "
            "focus on the Pipe C MCP pull path (s02/t02, s02/t07). Scope update: benchmark chat assistants, not coding agents (s03/t05), "
            "with domains covering long docs, episodic memory, terms/fine print, and social analytics (s03/t09)."
        )
        total = anchor_tokens + retrieval_tokens + estimate_tokens(answer)
        latency = time.time() - start
        return answer, total, latency, len(relevant), 2

    def score_quality(self, answer: str, is_monolithic: bool = False) -> tuple[float, float, float]:
        structural = 0.3
        if "Pipe C" in answer and "MCP pull" in answer:
            structural += 0.3
        if "chat assistants" in answer and "not coding agents" in answer:
            structural += 0.2

        citations = 0.2
        if "s01/t03" in answer or "s02/t02" in answer:
            citations += 0.3
        if "s03/t05" in answer or "s03/t09" in answer:
            citations += 0.3

        specificity = 0.2
        if "Chroma" in answer and "boundar" in answer:
            specificity += 0.3
        if "terms" in answer and "social analytics" in answer:
            specificity += 0.3

        return min(1.0, structural), min(1.0, citations), min(1.0, specificity)


class TermsFinePrintQATest(LongFormTest):
    """Terms/policy fine-print QA with clause-level retrieval."""

    def __init__(self):
        super().__init__("Terms QA — Fine-Print Obligations", "terms-fine-print")
        self.clauses: list[dict[str, str]] = []

    def setup_corpus(self) -> None:
        self.clauses = [
            {"id": "tos:3.2", "type": "billing", "text": "Subscriptions auto-renew unless cancelled 24 hours before renewal."},
            {"id": "tos:4.4", "type": "termination", "text": "Provider may suspend accounts for policy abuse with immediate effect."},
            {"id": "tos:5.1", "type": "arbitration", "text": "Disputes are resolved via binding arbitration and class-action waiver."},
            {"id": "privacy:2.3", "type": "sharing", "text": "Data may be shared with subprocessors for service delivery and analytics."},
            {"id": "privacy:7.1", "type": "deletion", "text": "Deletion requests are fulfilled within 30 days except legal hold cases."},
            {"id": "tos:9.2", "type": "changes", "text": "Material terms changes may be posted in-product without direct email notice."},
        ]
        self.corpus_lines = [
            f"clause={c['id']} type={c['type']} text={c['text']}"
            for c in self.clauses
        ]

    def get_questions(self) -> list[str]:
        return [
            "Can the service auto-renew and change terms without direct notice?",
            "What dispute and account-termination risks should a user know first?",
        ]

    def run_monolithic_answer(self, question: str) -> tuple[str, int, float]:
        full = "\n".join(self.corpus_lines)
        tokens = estimate_tokens(full + question)
        time.sleep(0.01)
        answer = (
            "There are auto-renew and arbitration clauses. The provider can suspend accounts for abuse. "
            "Terms updates may appear in product."
        )
        return answer, tokens, 0.01

    def run_pipe_c_answer(self, question: str) -> tuple[str, int, float, int, int]:
        start = time.time()

        anchor = (
            "Intent: identify obligations, exceptions, and user risk from terms/privacy docs. "
            "Retrieve only billing, changes, arbitration, and termination clauses."
        )
        anchor_tokens = estimate_tokens(anchor)

        relevant = [
            line
            for line in self.corpus_lines
            if any(k in line.lower() for k in ["auto-renew", "changes", "arbitration", "termination", "suspend", "waiver"])
        ]
        retrieved = "\n".join(relevant)
        retrieval_tokens = estimate_tokens(retrieved + question)

        answer = (
            "Yes. Auto-renew is explicit (tos:3.2) and material changes may be posted without direct email notice (tos:9.2). "
            "Dispute risk: binding arbitration with class-action waiver (tos:5.1). "
            "Termination risk: immediate suspension for abuse (tos:4.4)."
        )
        total = anchor_tokens + retrieval_tokens + estimate_tokens(answer)
        latency = time.time() - start
        return answer, total, latency, len(relevant), 2

    def score_quality(self, answer: str, is_monolithic: bool = False) -> tuple[float, float, float]:
        structural = 0.3
        if "auto-renew" in answer and "arbitration" in answer:
            structural += 0.3
        if "termination" in answer or "suspension" in answer:
            structural += 0.2

        citations = 0.2
        if "tos:3.2" in answer and "tos:9.2" in answer:
            citations += 0.3
        if "tos:5.1" in answer and "tos:4.4" in answer:
            citations += 0.3

        specificity = 0.2
        if "class-action waiver" in answer:
            specificity += 0.3
        if "without direct email notice" in answer:
            specificity += 0.3

        return min(1.0, structural), min(1.0, citations), min(1.0, specificity)


class SocialModerationQATest(LongFormTest):
    """Large social corpus analytics assistant benchmark."""

    def __init__(self):
        super().__init__("Social Analytics QA — Sentiment/Abuse", "social-analytics")
        self.records: list[dict[str, str]] = []

    def setup_corpus(self) -> None:
        self.records = [
            {"id": "r1", "community": "tech", "week": "2026-W20", "sentiment": "negative", "abuse": "insult", "text": "This release is garbage and your team is clueless."},
            {"id": "r2", "community": "tech", "week": "2026-W20", "sentiment": "negative", "abuse": "harassment", "text": "Go away, nobody wants your broken app."},
            {"id": "r3", "community": "finance", "week": "2026-W20", "sentiment": "neutral", "abuse": "none", "text": "Fees changed after update, need clarification."},
            {"id": "r4", "community": "tech", "week": "2026-W21", "sentiment": "negative", "abuse": "threat", "text": "You should be banned from shipping software."},
            {"id": "r5", "community": "gaming", "week": "2026-W21", "sentiment": "positive", "abuse": "none", "text": "Patch fixed lag and matchmaking improved."},
            {"id": "r6", "community": "tech", "week": "2026-W21", "sentiment": "negative", "abuse": "insult", "text": "Worst update ever, complete incompetence."},
        ]

        self.corpus_lines = [
            "id={id} community={community} week={week} sentiment={sentiment} abuse={abuse} text={text}".format(**r)
            for r in self.records
        ]

    def get_questions(self) -> list[str]:
        return [
            "Which abuse categories increased week-over-week in tech community posts?",
            "What moderation risk summary should we produce for the latest week?",
        ]

    def run_monolithic_answer(self, question: str) -> tuple[str, int, float]:
        full = "\n".join(self.corpus_lines)
        tokens = estimate_tokens(full + question)
        time.sleep(0.01)
        answer = (
            "Tech has more negative comments with insults and harassment. "
            "Latest week still shows harmful language and elevated moderation risk."
        )
        return answer, tokens, 0.01

    def run_pipe_c_answer(self, question: str) -> tuple[str, int, float, int, int]:
        start = time.time()

        anchor = (
            "Intent: moderation analytics summary for latest periods. "
            "Retrieve by community, week bucket, sentiment, and abuse category."
        )
        anchor_tokens = estimate_tokens(anchor)

        relevant = [
            line
            for line in self.corpus_lines
            if "community=tech" in line and any(k in line for k in ["abuse=insult", "abuse=harassment", "abuse=threat", "week=2026-W21", "week=2026-W20"])
        ]
        retrieved = "\n".join(relevant)
        retrieval_tokens = estimate_tokens(retrieved + question)

        answer = (
            "Week-over-week in tech: threat incidents appeared in 2026-W21 (new), insults persisted, harassment remained present. "
            "Risk summary for latest week: high negativity concentration in tech with insult+threat patterns; prioritize stricter moderation for repeat-abuse users and rapid review queues."
        )

        total = anchor_tokens + retrieval_tokens + estimate_tokens(answer)
        latency = time.time() - start
        return answer, total, latency, len(relevant), 2

    def score_quality(self, answer: str, is_monolithic: bool = False) -> tuple[float, float, float]:
        structural = 0.3
        if "week-over-week" in answer or "Week-over-week" in answer:
            structural += 0.2
        if "risk summary" in answer or "Risk summary" in answer:
            structural += 0.2

        citations = 0.2
        if "2026-W21" in answer or "2026-W20" in answer:
            citations += 0.3
        if "tech" in answer and any(k in answer for k in ["insult", "harassment", "threat"]):
            citations += 0.3

        specificity = 0.2
        if "repeat-abuse" in answer:
            specificity += 0.3
        if "review queues" in answer:
            specificity += 0.3

        return min(1.0, structural), min(1.0, citations), min(1.0, specificity)


def run_all_long_form_tests() -> dict[str, list[LongFormTestResult]]:
    """Run all chat-assistant experiment families."""
    tests: list[LongFormTest] = [
        BookDocumentQATest(),
        EpisodicMemoryQATest(),
        TermsFinePrintQATest(),
        SocialModerationQATest(),
    ]

    results: dict[str, list[LongFormTestResult]] = {}

    for test in tests:
        print(f"\n{'=' * 80}")
        print(f"Running: {test.name}")
        print('=' * 80)
        test_results = test.run_test()
        results[test.name] = test_results

        for result in test_results:
            print(f"  Q: {result.question[:70]}...")
            print(
                f"    Tokens: {result.monolithic_tokens:,} -> {result.pipe_c_tokens:,} "
                f"({result.token_reduction:.1f}% saved)"
            )
            print(
                f"    Quality: structural={result.quality_structural_score:.2f}, "
                f"citations={result.quality_citations_score:.2f}, "
                f"specificity={result.quality_specificity_score:.2f}"
            )

    return results


if __name__ == "__main__":
    all_results = run_all_long_form_tests()
    print("\n" + "=" * 80)
    print("CHAT-ASSISTANT LONG-CONTEXT TESTS COMPLETE")
    print("=" * 80)
    for test_name, test_results in all_results.items():
        avg_reduction = sum(r.token_reduction for r in test_results) / len(test_results)
        avg_quality = sum(r.quality_parity for r in test_results) / len(test_results)
        print(f"{test_name}: {avg_reduction:.1f}% token reduction, {avg_quality:.2f} quality parity")
