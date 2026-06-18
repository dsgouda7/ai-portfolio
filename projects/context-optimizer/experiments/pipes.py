"""
Pipeline implementations for Context Optimizer experiments.

Pipe A     — monolithic baseline (raw prompt + full logs).
Pipe OOTB  — standard LangChain LCEL RAG; what most developers build first.
Pipe C     — proposed MCP-pull architecture with structured shell (SOLUTION).
"""
from __future__ import annotations

import json
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from context_optimizer_benchmark import (
    CompressedIncident,
    run_compression_step,
    run_pipeline_a,
    run_pipeline_c,
)
from experiments.shared_inputs import MOCK_ANSWER_PIPE_A, MOCK_ANSWER_PIPE_OOTB, MOCK_ANSWER_PIPE_C, estimate_tokens
from experiments.retriever import SemanticVectorRetriever, SimpleRetriever


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PipeResult:
    pipe_name: str
    answer: str
    latency_s: float
    # Token counts (estimated at ~4 chars/token when not using real tokenizer)
    prompt_tokens_sent: int = 0
    tool_call_count: int = 0
    retrieved_lines: int = 0
    compression_latency_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_latency_s(self) -> float:
        return self.latency_s + self.compression_latency_s


# ---------------------------------------------------------------------------
# Pipe A — monolithic baseline (re-uses existing implementation)
# ---------------------------------------------------------------------------


def run_pipe_a(
    reasoning_llm: Any,
    raw_prompt: str,
    log_corpus: list[str],
    provider: str,
) -> PipeResult:
    """Pipe A: raw prompt + full log dump sent to the reasoning model."""
    if provider == "mock":
        logs_blob = "\n".join(log_corpus)
        t0 = time.perf_counter()
        answer = MOCK_ANSWER_PIPE_A
        latency = time.perf_counter() - t0
        prompt_tokens = estimate_tokens(raw_prompt + logs_blob)
        return PipeResult(
            pipe_name="Pipe A — Monolithic (baseline)",
            answer=answer,
            latency_s=latency,
            prompt_tokens_sent=prompt_tokens,
            retrieved_lines=len(log_corpus),
        )

    answer, latency, lines = run_pipeline_a(reasoning_llm, raw_prompt, log_corpus, provider)
    logs_blob = "\n".join(log_corpus)
    return PipeResult(
        pipe_name="Pipe A — Monolithic (baseline)",
        answer=answer,
        latency_s=latency,
        prompt_tokens_sent=estimate_tokens(raw_prompt + logs_blob),
        retrieved_lines=lines,
    )


# ---------------------------------------------------------------------------
# Pipe OOTB — standard LangChain LCEL RAG (no custom compression)
# ---------------------------------------------------------------------------

_OOTB_SYSTEM = (
    "You are a senior SRE and distributed-systems incident responder. "
    "Use only the context provided to diagnose the incident. "
    "Identify root cause, supporting evidence, mitigations, and next checks."
)


def run_pipe_ootb(
    reasoning_llm: Any,
    raw_prompt: str,
    log_corpus: list[str],
    provider: str,
    k: int = 5,
) -> PipeResult:
    """
    Pipe OOTB: standard out-of-the-box LangChain RAG.

    This is the pattern most developers write when first building a
    context-augmented LLM application — no compression, no schema,
    just retrieve top-k chunks and send with the raw user query.
    """
    retriever = SimpleRetriever(log_corpus)

    if provider == "mock":
        t0 = time.perf_counter()
        chunks = retriever.retrieve(raw_prompt, k=k)
        context = "\n---\n".join(chunks)
        answer = MOCK_ANSWER_PIPE_OOTB
        latency = time.perf_counter() - t0
        prompt_tokens = estimate_tokens(raw_prompt + context)
        return PipeResult(
            pipe_name="Pipe OOTB — Standard LangChain RAG",
            answer=answer,
            latency_s=latency,
            prompt_tokens_sent=prompt_tokens,
            retrieved_lines=context.count("\n") + 1,
        )

    # Standard LangChain LCEL chain — the "out of the box" pattern.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _OOTB_SYSTEM),
            (
                "human",
                "Retrieved log context:\n{context}\n\n"
                "Incident report:\n{question}",
            ),
        ]
    )

    def fetch_context(query: str) -> str:
        chunks = retriever.retrieve(query, k=k)
        return "\n---\n".join(chunks)

    chain = (
        {"context": RunnableLambda(fetch_context), "question": RunnablePassthrough()}
        | prompt
        | reasoning_llm
        | StrOutputParser()
    )

    chunks = retriever.retrieve(raw_prompt, k=k)
    context = "\n---\n".join(chunks)

    t0 = time.perf_counter()
    answer = chain.invoke(raw_prompt)
    latency = time.perf_counter() - t0

    return PipeResult(
        pipe_name="Pipe OOTB — Standard LangChain RAG",
        answer=answer,
        latency_s=latency,
        prompt_tokens_sent=estimate_tokens(raw_prompt + context),
        retrieved_lines=context.count("\n") + 1,
    )


# ---------------------------------------------------------------------------
# Pipe C — MCP pull architecture with structured shell
# ---------------------------------------------------------------------------

_PIPE_C_PERSONA = (
    "You are a principal SRE incident analyst with deep expertise in distributed systems. "
    "You operate with a strict token budget. "
    "Use the retrieve_context tool to pull targeted evidence. "
    "Do not request raw logs. Build a focused, evidence-backed diagnosis."
)

_PIPE_C_TOOL_DECLARATIONS = [
    {
        "name": "retrieve_context",
        "description": (
            "Retrieve semantically matched context from a chunked vector store. "
            "The server embeds the query with the same embedding model used at index time, "
            "hybrid-ranks vector similarity with lexical relevance, and returns scored chunks. "
            "Each chunk preserves its original boundary metadata and includes continuation hints "
            "when the evidence appears truncated at the start or end of the chunk. "
            "Use explicit technical identifiers when possible; use broader symptom phrasing first "
            "when the root cause is unclear."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language retrieval query, error code, service pair, or hypothesis to test"
                },
                "depth": {
                    "type": "string",
                    "enum": ["brief", "detailed", "exhaustive"],
                    "description": "brief=top-3, detailed=top-6, exhaustive=top-10 before budget trim",
                },
                "service": {
                    "type": "string",
                    "description": "Optional service filter such as order-service or ingress-nginx"
                },
                "severity": {
                    "type": "string",
                    "enum": ["ERROR", "WARN", "INFO"],
                    "description": "Optional severity filter to reduce noise"
                },
            },
            "required": ["query"],
        },
    }
]


class _MCPServer:
    """
    In-process MCP server simulation.

    In production this would be a separate FastMCP process with a pre-compressed
    vector DB backend. Here it is an in-process class with the same typed contract
    so experiments can run without a running MCP daemon.

    The store holds pre-compressed context: the log corpus is chunked and each
    chunk is semantically indexed before storage, simulating write-time
    compression plus vectorization.
    """

    def __init__(self, log_corpus: list[str], small_llm: Any = None, provider: str = "mock"):
        self._retriever = SemanticVectorRetriever(log_corpus, provider=provider)
        self._token_budget_used: int = 0
        self._call_count: int = 0

    def retrieve_context(
        self,
        query: str,
        depth: str = "brief",
        service: str | None = None,
        severity: str | None = None,
    ) -> str:
        k = {"brief": 3, "detailed": 6, "exhaustive": 10}.get(depth, 3)
        filters = {
            key: value
            for key, value in {"service": service, "severity": severity}.items()
            if value
        }
        hits = self._retriever.retrieve(query=query, k=k, filters=filters)
        payload = {
            "status": "success" if hits else "empty",
            "query": query,
            "depth": depth,
            "backend": self._retriever.backend_name,
            "embedding_backend": self._retriever.embedding_name,
            "ranking": {
                "vector_weight": 0.7,
                "lexical_weight": 0.3,
                "same_embedding_space": True,
            },
            "guidance": {
                "interpretation": "Relevance scores indicate semantic closeness, not proof. Corroborate across chunks before concluding.",
                "next_query_hint": "Refine with concrete identifiers returned in top chunks when confidence is low.",
                "boundary_hint": "If a chunk sets needs_prev_chunk or needs_next_chunk=true, fetch corroborating context before finalizing causality.",
            },
            "chunks": [
                {
                    "rank": index + 1,
                    "chunk_id": hit.chunk_id,
                    "summary": hit.summary,
                    "context": hit.text,
                    "metadata": hit.metadata,
                    "relevance_score": round(hit.combined_score, 4),
                    "vector_score": round(hit.vector_score, 4),
                    "lexical_score": round(hit.lexical_score, 4),
                }
                for index, hit in enumerate(hits)
            ],
        }
        result = json.dumps(payload, indent=2)
        self._token_budget_used += estimate_tokens(result)
        self._call_count += 1
        return result

    @property
    def tokens_consumed(self) -> int:
        return self._token_budget_used

    @property
    def call_count(self) -> int:
        return self._call_count


def _build_structured_shell(compressed: CompressedIncident) -> str:
    """Build the fixed structured shell prompt for Pipe C."""
    return textwrap.dedent(
        f"""
        PERSONA: {_PIPE_C_PERSONA}

        AVAILABLE TOOLS:
        {json.dumps(_PIPE_C_TOOL_DECLARATIONS, indent=2)}

        INSTRUCTIONS:
        - Use retrieve_context to gather evidence before concluding.
        - The tool performs semantic vector search plus lexical reranking over chunked context.
        - Stored chunks preserve original context boundaries and expose continuation hints.
        - Start with a broad hypothesis query when uncertain, then refine with identifiers from returned chunks.
        - Relevance scores indicate closeness, not certainty; corroborate with at least two aligned signals before concluding.
        - If a retrieved chunk reports needs_prev_chunk=true or needs_next_chunk=true, treat it as incomplete local evidence and fetch adjacent corroboration before making a causal claim.
        - Prefer brief queries first; escalate to detailed only when confidence remains low.
        - Use service or severity filters to narrow search when the evidence already points to a subsystem.
        - Stop when you have enough evidence to provide a confident root-cause analysis.
        - Return: (1) root cause, (2) evidence citations, (3) immediate mitigations, (4) next checks.

        COMPRESSED TASK ANCHOR:
        core_issue: {compressed.core_issue}
        symptoms: {'; '.join(compressed.observed_symptoms)}
        identifiers: {', '.join(compressed.technical_identifiers)}
        """
    ).strip()


def _simulate_mcp_tool_loop(
    mcp: _MCPServer,
    compressed: CompressedIncident,
    max_calls: int = 4,
    token_ceiling: int = 3000,
) -> tuple[list[str], int]:
    """
    Simulate what a reasoning model would do: issue targeted MCP calls
    based on the compressed identifiers, accumulate context, then stop.

    In production the reasoning LLM drives this loop via tool calling.
    In mock mode we pre-define the retrieval queries from the compressed
    identifiers so the experiment is deterministic.
    """
    retrieved_contexts: list[str] = []

    # Derive retrieval queries from the compressed identifiers.
    # A real reasoning model would generate these from the task anchor.
    priority_topics = [
        t for t in compressed.technical_identifiers
        if any(kw in t.lower() for kw in ["cosmos", "21012", "ingress", "timeout", "504"])
    ][:max_calls]

    # Fall back to generic topics if identifiers are sparse
    if len(priority_topics) < 2:
        priority_topics = ["CosmosDB timeout 21012", "upstream timed out", "order-service error"]

    for topic in priority_topics[:max_calls]:
        if mcp.tokens_consumed >= token_ceiling:
            break
        result = mcp.retrieve_context(query=topic, depth="brief")
        retrieved_contexts.append(result)

    return retrieved_contexts, mcp.call_count


def run_pipe_c(
    small_llm: Any,
    reasoning_llm: Any,
    raw_prompt: str,
    log_corpus: list[str],
    provider: str,
    max_mcp_calls: int = 4,
    token_ceiling: int = 3000,
) -> PipeResult:
    """
    Pipe C: MCP-pull architecture with structured shell.

    Fixed structured shell + compressed task anchor. Reasoning model calls
    the in-process MCP server for targeted evidence retrieval. Context budget
    manager enforces a ceiling on total tool-response tokens.
    """
    # Stage 1: compress the input
    compressed, comp_latency = run_compression_step(small_llm, raw_prompt, provider)

    # Stage 2: build structured shell + start MCP server
    mcp = _MCPServer(log_corpus, small_llm=small_llm, provider=provider)
    shell = _build_structured_shell(compressed)

    if provider == "mock":
        t0 = time.perf_counter()
        retrieved_contexts, call_count = _simulate_mcp_tool_loop(
            mcp, compressed, max_calls=max_mcp_calls, token_ceiling=token_ceiling
        )
        answer = MOCK_ANSWER_PIPE_C
        latency = time.perf_counter() - t0
        prompt_tokens = estimate_tokens(shell) + mcp.tokens_consumed
        return PipeResult(
            pipe_name="Pipe C — MCP Pull (structured shell)",
            answer=answer,
            latency_s=latency,
            compression_latency_s=comp_latency,
            prompt_tokens_sent=prompt_tokens,
            tool_call_count=call_count,
            retrieved_lines=mcp.tokens_consumed * 4 // 80,  # rough line estimate
            extra={
                "mcp_tokens_consumed": mcp.tokens_consumed,
                "shell_tokens": estimate_tokens(shell),
                "total_context_tokens": estimate_tokens(shell) + mcp.tokens_consumed,
            },
        )

    # Real LLM path: use LangChain tool binding on the reasoning model.
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def retrieve_context(
        query: str,
        depth: str = "brief",
        service: str = "",
        severity: str = "",
    ) -> str:
        """Retrieve semantically matched log context from the MCP vector store."""
        return mcp.retrieve_context(
            query=query,
            depth=depth,
            service=service or None,
            severity=severity or None,
        )

    model_with_tools = reasoning_llm.bind_tools([retrieve_context])

    messages = [
        SystemMessage(content=shell),
        HumanMessage(
            content=(
                "Begin your investigation. Use retrieve_context to gather evidence, "
                "then provide your root-cause analysis."
            )
        ),
    ]

    t0 = time.perf_counter()
    for _ in range(max_mcp_calls):
        ai_msg = model_with_tools.invoke(messages)
        messages.append(ai_msg)
        tool_calls = getattr(ai_msg, "tool_calls", None) or []

        if not tool_calls:
            # Model decided it has enough context
            break

        for tc in tool_calls:
            args = tc.get("args", {})
            result = mcp.retrieve_context(
                query=args.get("query", ""),
                depth=args.get("depth", "brief"),
                service=args.get("service") or None,
                severity=args.get("severity") or None,
            )
            messages.append(
                ToolMessage(content=result, tool_call_id=tc.get("id", "unknown"))
            )

        if mcp.tokens_consumed >= token_ceiling:
            # Budget manager: stop and ask for final answer
            messages.append(
                HumanMessage(content="Token budget reached. Provide your final analysis now.")
            )
            break

    final = reasoning_llm.invoke(messages)
    latency = time.perf_counter() - t0

    return PipeResult(
        pipe_name="Pipe C — MCP Pull (structured shell)",
        answer=str(final.content),
        latency_s=latency,
        compression_latency_s=comp_latency,
        prompt_tokens_sent=estimate_tokens(shell) + mcp.tokens_consumed,
        tool_call_count=mcp.call_count,
        retrieved_lines=mcp.tokens_consumed * 4 // 80,
        extra={
            "mcp_tokens_consumed": mcp.tokens_consumed,
            "shell_tokens": estimate_tokens(shell),
            "total_context_tokens": estimate_tokens(shell) + mcp.tokens_consumed,
        },
    )
