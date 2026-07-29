"""
TreeReasoningAgent — multi-step reasoning over a hierarchical TreeIndex.

The agent is given three tools and uses them autonomously:

  search_cluster(cluster_id)   → expands a Level-2 cluster to its Level-1
                                  block summaries.  Use when a cluster summary
                                  looks relevant but lacks enough detail.

  fetch_raw_block(block_id)    → reads the full raw text for a block from disk
                                  via BlockIndex file pointer.  Use when a
                                  block summary hints at the answer but lacks
                                  the exact words, numbers, or names.

  (implicit stop)              → when context is sufficient, the LLM produces
                                  a final answer without calling a tool.

Reasoning loop
--------------
1. Search L2: get top-N cluster super-summaries.
2. Present super-summaries + top block summaries to the LLM.
3. LLM decides:
     a) Answer now from available summaries.
     b) Call search_cluster(id) to get finer-grained L1 summaries.
     c) Call fetch_raw_block(id) to get full text.
4. Execute tool call, add result to context, repeat up to max_rounds.
5. Return final answer + what was retrieved.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_optimizer.tree_index import BlockHit, ClusterHit, TreeIndex


# ── Tool-decision prompt ──────────────────────────────────────────────────────
# The LLM sees context accumulated so far and must decide the next action.

_TOOL_PROMPT = """\
You are answering a question using a hierarchical text index.

AVAILABLE TOOLS:
  search_cluster(cluster_id)  - expand a cluster to see its individual block summaries
  fetch_raw_block(block_id)   - read the full text of a specific block from disk
  answer(text)                - give the final answer (use when you have enough information)

CURRENT CONTEXT:
{context}

QUESTION: {question}

Decide the next action. Respond with ONLY valid JSON:
{{"action": "search_cluster", "cluster_id": "cluster_XXXX"}}
{{"action": "fetch_raw_block", "block_id": "some_block_XXXXXX"}}
{{"action": "answer", "text": "your concise answer here"}}

Choose "answer" if the context already contains sufficient information.
Prefer "fetch_raw_block" over "search_cluster" when you need exact quotes, numbers, or names.
JSON:"""


_SYNTHESIS_PROMPT = """\
Answer the question using ONLY the retrieved context below.
Be concise (1-3 sentences). If context is insufficient say "Insufficient context."

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class AgentStep:
    """Records one step in the reasoning loop."""

    action: str  # "search_cluster" | "fetch_raw_block" | "answer"
    target_id: str = ""  # cluster_id or block_id
    result_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class TreeQueryResult:
    """Final result from a TreeReasoningAgent.reason() call."""

    query: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    context_tokens: int = 0
    total_latency_ms: float = 0.0
    used_raw_fallback: bool = False
    kw_recall: float = 0.0


class TreeReasoningAgent:
    """
    Multi-step reasoning agent over a :class:`~context_optimizer.tree_index.TreeIndex`.

    Parameters
    ----------
    tree:
        The populated :class:`TreeIndex`.
    llm:
        LLM for the tool-decision and answer-synthesis steps.
    top_clusters:
        How many Level-2 clusters to retrieve in the initial search.
    top_blocks_per_cluster:
        How many Level-1 blocks to surface per cluster.
    max_rounds:
        Maximum number of tool calls before forcing a final answer.
    fallback_threshold:
        Cosine distance threshold.  If the best L2 hit distance exceeds this
        value, a raw-block fetch is triggered immediately (low confidence).
    """

    def __init__(
        self,
        tree: "TreeIndex",
        llm: Any | None = None,
        *,
        top_clusters: int = 2,
        top_blocks_per_cluster: int = 3,
        max_rounds: int = 3,
        fallback_threshold: float = 0.50,
    ) -> None:
        self._tree = tree
        self._llm = llm
        self._top_clusters = top_clusters
        self._top_blocks = top_blocks_per_cluster
        self._max_rounds = max_rounds
        self._fallback_threshold = fallback_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def reason(self, query: str) -> TreeQueryResult:
        """
        Run the multi-step tool-calling loop for *query*.

        Returns a :class:`TreeQueryResult` with the final answer, all steps
        taken, and token/latency statistics.
        """
        t_start = time.perf_counter()
        steps: list[AgentStep] = []
        used_raw = False

        # ── Step 1: initial hierarchical search ───────────────────────────────
        cluster_hits = self._tree.search(
            query,
            top_clusters=self._top_clusters,
            top_blocks_per_cluster=self._top_blocks,
        )

        # Build initial context string
        context_parts: list[str] = []
        for ch in cluster_hits:
            context_parts.append(
                f"[Cluster {ch.cluster_id} — dist={ch.distance:.3f}]\n"
                f"{ch.super_summary}"
            )
            for bh in ch.block_hits:
                context_parts.append(
                    f"  [Block {bh.block_id} — dist={bh.distance:.3f}]\n"
                    f"  {bh.summary[:300]}"
                )
        context = "\n\n".join(context_parts)

        if self._llm is None:
            # No LLM: aggregate summaries as the answer
            answer = context[:1000] if context else "No relevant content found."
            return TreeQueryResult(
                query=query,
                answer=answer,
                steps=steps,
                context_tokens=len(context) // 4,
                total_latency_ms=(time.perf_counter() - t_start) * 1000,
                used_raw_fallback=False,
            )

        # ── Step 2: tool-calling loop ─────────────────────────────────────────
        answer = ""
        for round_idx in range(self._max_rounds):
            prompt = _TOOL_PROMPT.format(context=context[:3000], question=query)
            t_call = time.perf_counter()
            try:
                resp = self._llm.invoke(prompt)
                resp_text = (
                    resp.content if hasattr(resp, "content") else str(resp)
                ).strip()
                call_ms = (time.perf_counter() - t_call) * 1000

                # Parse JSON decision
                decision = json.loads(resp_text)
                action = decision.get("action", "answer")

                if action == "answer":
                    answer = decision.get("text", "")
                    steps.append(AgentStep(action="answer", latency_ms=call_ms))
                    break

                elif action == "search_cluster":
                    cid = decision.get("cluster_id", "")
                    # Use expand_cluster() so depth>2 trees can navigate
                    # one level deeper (L3->L2, L2->L1) as needed.
                    child_hits = self._tree.expand_cluster(
                        cid, query=query, top_k=self._top_blocks
                    )
                    if child_hits:
                        new_parts = [
                            f"  [Item {bh.block_id}]\n  {bh.summary}"
                            for bh in child_hits
                        ]
                        context += (
                            "\n\n[Expanded cluster "
                            + cid
                            + "]\n"
                            + "\n".join(new_parts)
                        )
                    else:
                        # Fallback: search in pre-fetched cluster_hits (depth=2 compat)
                        cluster_result = next(
                            (ch for ch in cluster_hits if ch.cluster_id == cid), None
                        )
                        if cluster_result:
                            new_parts = [
                                f"  [Block {bh.block_id}]\n  {bh.summary}"
                                for bh in cluster_result.block_hits
                            ]
                            context += (
                                "\n\n[Expanded cluster "
                                + cid
                                + "]\n"
                                + "\n".join(new_parts)
                            )
                    steps.append(
                        AgentStep(
                            action="search_cluster",
                            target_id=cid,
                            result_tokens=(
                                len("\n".join(new_parts)) // 4 if cluster_result else 0
                            ),
                            latency_ms=call_ms,
                        )
                    )

                elif action == "fetch_raw_block":
                    bid = decision.get("block_id", "")
                    raw = self._tree.get_raw_block(bid)
                    if raw:
                        context += f"\n\n[Raw block {bid} — full text]\n{raw[:3000]}"
                        used_raw = True
                    steps.append(
                        AgentStep(
                            action="fetch_raw_block",
                            target_id=bid,
                            result_tokens=len(raw[:3000]) // 4 if raw else 0,
                            latency_ms=call_ms,
                        )
                    )

            except (json.JSONDecodeError, Exception) as exc:
                # Malformed JSON or LLM error → force synthesis
                steps.append(AgentStep(action="answer", latency_ms=0))
                break

        # ── Step 3: synthesise final answer if tool loop didn't produce one ───
        if not answer:
            synth_prompt = _SYNTHESIS_PROMPT.format(
                context=context[:4000], question=query
            )
            t_synth = time.perf_counter()
            try:
                resp = self._llm.invoke(synth_prompt)
                answer = (
                    resp.content if hasattr(resp, "content") else str(resp)
                ).strip()
            except Exception:
                answer = "Synthesis failed."
            steps.append(
                AgentStep(
                    action="answer",
                    latency_ms=(time.perf_counter() - t_synth) * 1000,
                )
            )

        return TreeQueryResult(
            query=query,
            answer=answer,
            steps=steps,
            context_tokens=len(context) // 4,
            total_latency_ms=(time.perf_counter() - t_start) * 1000,
            used_raw_fallback=used_raw,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def tree(self) -> "TreeIndex":
        return self._tree
