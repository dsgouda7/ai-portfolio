"""
CodeReasoningAgent — navigates a CodeTreeIndex to answer code queries.

Extends the tree-navigation pattern from TreeReasoningAgent with
code-specific actions:

  search_cluster(cluster_id)   — expand L2/L3 cluster to child summaries
  fetch_source(chunk_id)       — fetch exact source lines (CodePointer)
  answer(text)                 — produce final answer with file:line citation

The agent appends ``[file:start-end]`` citations to its final answer
so callers know exactly where in the codebase the answer comes from.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .code_index import CodeTreeIndex


_TOOL_PROMPT = """\
You are a code-search assistant navigating a hierarchical index of source code.

AVAILABLE TOOLS:
  search_cluster(cluster_id)  - expand a cluster to see child function/class summaries
  fetch_source(chunk_id)      - read the actual source code for a specific function/class
  answer(text)                - give the final answer (cite file and line numbers)

CURRENT CONTEXT:
{context}

QUESTION: {question}

Rules:
- Use search_cluster to drill down when summaries look relevant but lack detail.
- Use fetch_source when you need exact code, line numbers, or to verify behaviour.
- Include [file:start_line-end_line] citations in your answer.
- Prefer fetch_source over guessing from summaries.

Decide the next action. Respond with ONLY valid JSON:
{{"action": "search_cluster", "cluster_id": "cluster_XXXX"}}
{{"action": "fetch_source", "chunk_id": "CHUNK_ID"}}
{{"action": "answer", "text": "your answer with [file:line] citations"}}
JSON:"""

_SYNTHESIS_PROMPT = """\
Answer the question using ONLY the code context below.
Include [file:start_line-end_line] citations for every claim.
Be concise (2-4 sentences). If context is insufficient say "Insufficient context."

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class CodeAgentStep:
    action: str
    target_id: str = ""
    result_tokens: int = 0
    latency_ms: float = 0.0
    citation: str = ""  # "file:start-end" for fetch_source steps


@dataclass
class CodeQueryResult:
    query: str
    answer: str
    steps: list[CodeAgentStep] = field(default_factory=list)
    context_tokens: int = 0
    total_latency_ms: float = 0.0
    citations: list[str] = field(default_factory=list)  # all file:line refs used


class CodeReasoningAgent:
    """
    Multi-step tool-calling agent over a :class:`~context_optimizer.code.code_index.CodeTreeIndex`.

    Parameters
    ----------
    index:
        The populated CodeTreeIndex.
    llm:
        Ollama / HF LLM for tool decisions and synthesis.
    top_clusters:
        How many top-level clusters to include in the initial context.
    top_chunks_per_cluster:
        How many L1 chunks to surface per cluster.
    max_rounds:
        Max tool calls before forcing synthesis.
    """

    def __init__(
        self,
        index: "CodeTreeIndex",
        llm: Any | None = None,
        *,
        top_clusters: int = 2,
        top_chunks_per_cluster: int = 3,
        max_rounds: int = 4,
    ) -> None:
        self._index = index
        self._llm = llm
        self._top_clusters = top_clusters
        self._top_chunks = top_chunks_per_cluster
        self._max_rounds = max_rounds

    def reason(self, query: str) -> CodeQueryResult:
        t_start = time.perf_counter()
        steps: list[CodeAgentStep] = []
        citations: list[str] = []

        # Initial hierarchical search
        cluster_hits = self._index.search(
            query,
            top_clusters=self._top_clusters,
            top_blocks_per_cluster=self._top_chunks,
        )

        context_parts: list[str] = []
        for ch in cluster_hits:
            context_parts.append(f"[Cluster {ch.cluster_id}]\n{ch.super_summary}")
            for bh in ch.block_hits:
                ptr = self._index.get_pointer(bh.block_id)
                cite = ptr.citation() if ptr else bh.block_id
                context_parts.append(
                    f"  [Function {bh.block_id} @ {cite}]\n  {bh.summary[:300]}"
                )
        context = "\n\n".join(context_parts)

        if self._llm is None:
            return CodeQueryResult(
                query=query,
                answer=context[:1000] if context else "No relevant code found.",
                context_tokens=len(context) // 4,
                total_latency_ms=(time.perf_counter() - t_start) * 1000,
            )

        answer = ""
        for _ in range(self._max_rounds):
            prompt = _TOOL_PROMPT.format(context=context[:3500], question=query)
            t_call = time.perf_counter()
            try:
                resp = self._llm.invoke(prompt)
                resp_text = (
                    resp.content if hasattr(resp, "content") else str(resp)
                ).strip()
                call_ms = (time.perf_counter() - t_call) * 1000

                # Strip markdown code fences if present
                if "```" in resp_text:
                    import re

                    resp_text = re.sub(r"```[^\n]*\n?", "", resp_text).strip()

                decision = json.loads(resp_text)
                action = decision.get("action", "answer")

                if action == "answer":
                    answer = decision.get("text", "")
                    steps.append(CodeAgentStep(action="answer", latency_ms=call_ms))
                    break

                elif action == "search_cluster":
                    cid = decision.get("cluster_id", "")
                    child_hits = self._index.expand_cluster(
                        cid, query=query, top_k=self._top_chunks
                    )
                    new_parts = []
                    for bh in child_hits:
                        ptr = self._index.get_pointer(bh.block_id)
                        cite = ptr.citation() if ptr else bh.block_id
                        new_parts.append(
                            f"  [Function {bh.block_id} @ {cite}]\n  {bh.summary[:300]}"
                        )
                    context += (
                        "\n\n[Expanded cluster " + cid + "]\n" + "\n".join(new_parts)
                    )
                    steps.append(
                        CodeAgentStep(
                            action="search_cluster",
                            target_id=cid,
                            result_tokens=len("\n".join(new_parts)) // 4,
                            latency_ms=call_ms,
                        )
                    )

                elif action == "fetch_source":
                    cid = decision.get("chunk_id", "")
                    source = self._index.get_raw_chunk(cid)
                    ptr = self._index.get_pointer(cid)
                    cite = ptr.citation() if ptr else cid
                    if source:
                        context += f"\n\n[Source @ {cite}]\n```\n{source[:3000]}\n```"
                        citations.append(cite)
                    steps.append(
                        CodeAgentStep(
                            action="fetch_source",
                            target_id=cid,
                            result_tokens=len(source) // 4,
                            latency_ms=call_ms,
                            citation=cite,
                        )
                    )

            except (json.JSONDecodeError, Exception):
                steps.append(CodeAgentStep(action="answer", latency_ms=0))
                break

        # Synthesis if loop didn't produce an answer
        if not answer:
            synth = _SYNTHESIS_PROMPT.format(context=context[:4000], question=query)
            t_synth = time.perf_counter()
            try:
                resp = self._llm.invoke(synth)
                answer = (
                    resp.content if hasattr(resp, "content") else str(resp)
                ).strip()
            except Exception:
                answer = "Synthesis failed."
            steps.append(
                CodeAgentStep(
                    action="answer",
                    latency_ms=(time.perf_counter() - t_synth) * 1000,
                )
            )

        return CodeQueryResult(
            query=query,
            answer=answer,
            steps=steps,
            context_tokens=len(context) // 4,
            total_latency_ms=(time.perf_counter() - t_start) * 1000,
            citations=citations,
        )
