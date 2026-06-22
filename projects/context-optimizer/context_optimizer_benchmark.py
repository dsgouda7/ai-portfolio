from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:  # pragma: no cover
        ChatOllama = None

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover
    ChatGroq = None

# ToTReasoner is baked into src/context_optimizer — import it from there.
sys.path.insert(0, str(Path(__file__).parent / "src"))
from context_optimizer.tot_reasoner import ToTReasoner  # noqa: E402


COMPRESSION_SYSTEM_PROMPT = (
    "You are a highly efficient Token Compression Filter for a distributed systems debugging agent. "
    "Your job is to strip out all human rambling, conversational fluff, and emotional context from "
    "the user's prompt. Convert the input into a dense, robotic, engineering brief optimized for a "
    "downstream reasoning LLM. Retain all technical identifiers exactly (IPs, service names like AKS, "
    "CosmosDB, metrics, error codes). Do not answer the user's problem."
)

MOCK_INCIDENT_PROMPT = textwrap.dedent(
    """
    Hey team, sorry this is a bit all over the place because I have been on this for hours and this has
    turned into a full-on fire. Since around 02:13 UTC the checkout flow has been intermittently timing out
    and support is flooded. We run on AKS, ingress-nginx in front, api-gateway then order-service and
    payment-service. Clients report 504 then sometimes 499. Prometheus shows p95 latency climbed from 220ms
    to 8.7s, error_rate is at 17.6%, and CPU on a few pods is normal which is weird. CosmosDB dependency
    calls in application insights look bad. I keep seeing timeout error code 21012 and a lot of retries.

    We had a deployment at 01:55 UTC but only for recommendation-service so I do not think it should impact
    checkout, but maybe noisy neighbors? Also there was a weird spike in ingress warnings around
    "upstream timed out while reading response header" on aks-prod-eastus nodepool np-user-03. One trace
    references 10.42.7.19 and another mentions 10.42.8.44. I also saw a stack trace from order-service:
    System.TimeoutException at CosmosClient.ReadItemAsync, then downstream call cancellation in
    PaymentConnector.SubmitAsync.

    I am honestly not sure if this is network, CosmosDB RU starvation, bad retry policy, or something in
    ingress connection handling. Can you help figure out what is likely happening and what to check first?
    """
).strip()


class CompressedIncident(BaseModel):
    """Schema returned by the Token Compression Engine."""

    core_issue: str = Field(description="Single-sentence statement of the core technical problem.")
    observed_symptoms: list[str] = Field(
        description="List of concrete observations (metrics, errors, system behavior)."
    )
    technical_identifiers: list[str] = Field(
        description="Exact technical tokens extracted from input, including names/IPs/error codes."
    )


@dataclass
class ModelConfig:
    provider: str
    small_model: str
    reasoning_model: str
    temperature: float = 0.0


def default_model_names(provider: str) -> tuple[str, str]:
    if provider == "groq":
        return "llama-3.1-8b-instant", "llama-3.3-70b-versatile"
    if provider == "mock":
        return "mock-paraphraser", "mock-reasoner"
    return "phi4:mini", "qwen3"


def init_chat_model(provider: str, model_name: str, temperature: float = 0.0) -> BaseChatModel | None:
    """Initialize a chat model with a provider abstraction for Ollama, Groq, or mock mode."""
    normalized = provider.strip().lower()
    if normalized == "ollama":
        if ChatOllama is None:
            raise ImportError(
                "Ollama provider selected but ChatOllama is unavailable. Install langchain-ollama or langchain-community."
            )
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if normalized == "groq":
        if ChatGroq is None:
            raise ImportError(
                "Groq provider selected but langchain-groq is unavailable. Install with: pip install langchain-groq"
            )
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Export it before running with --provider groq."
            )
        return ChatGroq(model=model_name, temperature=temperature)
    if normalized == "mock":
        return None
    raise ValueError("provider must be one of: ollama, groq, mock")


def build_mock_log_cache(total_lines: int = 1050, seed: int = 73) -> list[str]:
    """Create an in-memory log corpus with realistic AKS/Cosmos/stack trace noise."""
    random.seed(seed)
    logs: list[str] = []
    start = datetime(2026, 6, 16, 1, 45, 0)

    services = [
        "api-gateway",
        "order-service",
        "payment-service",
        "recommendation-service",
        "inventory-service",
    ]
    pods = [
        "order-service-7f4b9d7b9f-k2m8q",
        "order-service-7f4b9d7b9f-r5vpl",
        "payment-service-69c57c6b9b-dj2nr",
        "api-gateway-6f9dddc75f-n7k4m",
        "ingress-nginx-controller-5f89d4c4bf-v8z2s",
    ]

    for i in range(total_lines):
        ts = (start + timedelta(seconds=i * 2)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        service = random.choice(services)
        pod = random.choice(pods)
        request_id = f"req-{i:06d}"

        if i % 89 == 0:
            logs.append(
                f"{ts} ERROR {service} [{pod}] [{request_id}] "
                "System.TimeoutException: Request timed out at CosmosClient.ReadItemAsync "
                "status=408 substatus=21012 region=eastus2"
            )
            logs.append(
                f"{ts} ERROR {service} [{pod}] [{request_id}] "
                "at Checkout.Repository.OrderStore.GetOrderById(String id) in /src/OrderStore.cs:line 214"
            )
            logs.append(
                f"{ts} ERROR {service} [{pod}] [{request_id}] "
                "at PaymentConnector.SubmitAsync(CancellationToken ct) in /src/PaymentConnector.cs:line 87"
            )
            continue

        if i % 53 == 0:
            logs.append(
                f"{ts} WARN ingress-nginx [{pod}] [{request_id}] "
                'upstream timed out while reading response header from upstream, client: 10.42.7.19, '
                'server: checkout.api.internal, request: "POST /v1/checkout HTTP/1.1", '
                'upstream: "http://10.42.8.44:8080/v1/checkout", host: "checkout.example.com"'
            )
            continue

        if i % 47 == 0:
            logs.append(
                f"{ts} WARN {service} [{pod}] [{request_id}] "
                "CosmosDB query timeout after 5000ms retries=3 ru_charge=128.44 "
                "partition=tenant-445 operation=ReadItem"
            )
            continue

        if i % 61 == 0:
            logs.append(
                f"{ts} ERROR api-gateway [{pod}] [{request_id}] "
                "HTTP 504 upstream timeout route=/v1/checkout p95=8.7s error_rate=17.6%"
            )
            continue

        logs.append(
            f"{ts} INFO {service} [{pod}] [{request_id}] "
            "request completed status=200 latency_ms="
            f"{random.randint(55, 480)}"
        )

    return logs[:total_lines]


MOCK_LOG_CACHE = build_mock_log_cache(total_lines=1050)


@tool
def query_log_cache(keyword: str, lines_context: int = 5) -> str:
    """Search in-memory logs and return matching lines with neighboring context."""
    needle = keyword.strip().lower()
    if not needle:
        return "query_log_cache error: keyword must be non-empty"

    ctx = max(0, min(lines_context, 25))
    max_hits = 8
    hits: list[str] = []

    for idx, line in enumerate(MOCK_LOG_CACHE):
        if needle in line.lower():
            start = max(0, idx - ctx)
            end = min(len(MOCK_LOG_CACHE), idx + ctx + 1)
            excerpt_lines = []
            for j in range(start, end):
                marker = ">" if j == idx else " "
                excerpt_lines.append(f"{marker} [{j:04d}] {MOCK_LOG_CACHE[j]}")
            hits.append("\n".join(excerpt_lines))
            if len(hits) >= max_hits:
                break

    if not hits:
        return f"No matches for keyword='{keyword}'."

    return (
        f"Found {len(hits)} match window(s) for keyword='{keyword}' with lines_context={ctx}.\n\n"
        + "\n\n---\n\n".join(hits)
    )


def _mock_compress(raw_prompt: str) -> CompressedIncident:
    identifiers = sorted(
        {
            match
            for match in re.findall(
                r"\b(?:AKS|CosmosDB|ingress-nginx|api-gateway|order-service|payment-service|"
                r"recommendation-service|error_rate|p95|\d+\.\d+\.\d+\.\d+|\d{4,6})\b",
                raw_prompt,
                flags=re.IGNORECASE,
            )
        }
    )
    return CompressedIncident(
        core_issue=(
            "Checkout pipeline experiences cascading timeouts across ingress and CosmosDB-backed dependencies "
            "during the active incident window."
        ),
        observed_symptoms=[
            "Checkout requests intermittently return 504/499.",
            "p95 latency rose from 220ms to 8.7s.",
            "Error rate reached 17.6% while CPU remained near baseline.",
            "Timeout code 21012 observed in CosmosDB-dependent path.",
            "Ingress reported upstream response-header timeouts.",
        ],
        technical_identifiers=identifiers,
    )


def _mock_reason_raw(raw_prompt: str, full_logs: list[str]) -> str:
    blob = "\n".join(full_logs).lower()
    timeout_hits = blob.count("timeout")
    cosmos_hits = blob.count("cosmos")
    ingress_hits = blob.count("upstream timed out")
    return (
        "Most likely root cause: CosmosDB-read latency propagated through order-service and ingress timeout limits.\n"
        f"Evidence: timeout_hits={timeout_hits}, cosmos_hits={cosmos_hits}, ingress_timeout_hits={ingress_hits}.\n"
        "Immediate mitigations: bounded retries, temporary timeout budget increase, ingress upstream tuning.\n"
        "Next checks: RU saturation, retry fan-out, partition hot-keys, cancellation waterfall, ingress queue depth.\n"
        f"Prompt chars processed: {len(raw_prompt)}; log lines processed: {len(full_logs)}."
    )


def _mock_reason_optimized(compressed: CompressedIncident) -> tuple[str, int, int]:
    search_terms = ["CosmosDB", "21012", "upstream timed out", "504"]
    retrieved_lines = 0
    for term in search_terms:
        snippet = query_log_cache.invoke({"keyword": term, "lines_context": 3})
        retrieved_lines += snippet.count("\n") + 1

    summary = (
        "Most likely root cause: timeout amplification from CosmosDB latency plus ingress upstream limits.\n"
        f"Retained identifiers: {', '.join(compressed.technical_identifiers[:8])}.\n"
        "Immediate mitigations: cap retries, add jitter, tune ingress timeouts, isolate noisy workloads.\n"
        "Next checks: dependency latency by partition, upstream header timing, retry burst metrics."
    )
    return summary, len(search_terms), retrieved_lines


def _mock_reason_tot(compressed: CompressedIncident) -> tuple[str, list[dict[str, Any]], str, int, int]:
    """Thin wrapper: build a log-search retriever and delegate to ToTReasoner."""

    class _LogCacheRetriever:
        """Adapts query_log_cache (langchain @tool) to the Retriever protocol."""

        def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
            snippet = query_log_cache.invoke({"keyword": query, "lines_context": 2})
            if "No matches" in snippet:
                return []
            return [{"compressed_summary": snippet}]

    reasoner = ToTReasoner(retriever=_LogCacheRetriever(), top_k_per_term=1)
    branch_specs = [
        {
            "id": "cosmos",
            "title": "CosmosDB / RU saturation branch",
            "search_terms": ["CosmosDB", "21012", "ru_charge"],
        },
        {
            "id": "ingress",
            "title": "Ingress / upstream timeout branch",
            "search_terms": ["upstream timed out", "504", "ingress-nginx"],
        },
        {
            "id": "retry",
            "title": "Retry / cancellation waterfall branch",
            "search_terms": ["retry", "cancellation", "timeout"],
        },
    ]
    result = reasoner.reason(compressed, branch_specs=branch_specs)
    branches_dicts = [
        {
            "id": b.id,
            "title": b.title,
            "search_terms": b.search_terms,
            "score": b.score,
            "evidence_hits": b.evidence_hits,
            "evidence_snippets": b.evidence_snippets,
        }
        for b in result.branches
    ]
    return (
        result.selected_summary,
        branches_dicts,
        result.selected_branch_id,
        len(result.branches),
        result.total_retrieved_lines,
    )


def run_pipeline_c(
    reasoning_llm: BaseChatModel | None,
    compressed: CompressedIncident,
    provider: str,
    max_tool_rounds: int = 3,
) -> tuple[str, float, list[dict[str, Any]], str, int, int]:
    """Tree-of-Thought reasoning via ToTReasoner (baked into src/context_optimizer)."""
    start = time.perf_counter()

    if provider == "mock" or reasoning_llm is None:
        output, branches, selected_branch, _, retrieved_lines_total = _mock_reason_tot(compressed)
        latency = time.perf_counter() - start
        return output, latency, branches, selected_branch, len(branches), retrieved_lines_total

    # LLM-driven path: build branch specs from incident then reason via ToTReasoner.
    branch_prompts = [
        ("cosmos",  "Create a concise hypothesis about CosmosDB or RU saturation causing the incident, plus one search term."),
        ("ingress", "Create a concise hypothesis about ingress or upstream timeout causing the incident, plus one search term."),
        ("retry",   "Create a concise hypothesis about retry storms or cancellation waterfalls causing the incident, plus one search term."),
    ]

    branch_specs = []
    for branch_id, prompt in branch_prompts:
        response = reasoning_llm.invoke(
            [HumanMessage(content=f"{prompt}\n\nIncident brief:\n{compressed.model_dump_json(indent=2)}")]
        )
        branch_text = str(response.content)
        search_terms = [t.strip() for t in branch_text.split(",") if t.strip()][:2] or [branch_id]
        branch_specs.append({"id": branch_id, "title": branch_id, "search_terms": search_terms})

    class _LogCacheRetriever:
        def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
            snippet = query_log_cache.invoke({"keyword": query, "lines_context": 1})
            if "No matches" in snippet:
                return []
            return [{"compressed_summary": snippet}]

    reasoner = ToTReasoner(retriever=_LogCacheRetriever(), top_k_per_term=1)
    result = reasoner.reason(compressed, branch_specs=branch_specs)

    branches_dicts = [
        {
            "id": b.id,
            "title": b.title,
            "search_terms": b.search_terms,
            "score": b.score,
            "evidence_hits": b.evidence_hits,
            "evidence_snippets": b.evidence_snippets,
        }
        for b in result.branches
    ]
    latency = time.perf_counter() - start
    return result.selected_summary, latency, branches_dicts, result.selected_branch_id, len(result.branches), result.total_retrieved_lines


def run_compression_step(
    paraphraser_llm: BaseChatModel | None,
    raw_prompt: str,
    provider: str,
) -> tuple[CompressedIncident, float]:
    """Run the Edge Filter chain with structured output validation."""
    start = time.perf_counter()

    if provider == "mock" or paraphraser_llm is None:
        result = _mock_compress(raw_prompt)
        latency = time.perf_counter() - start
        return result, latency

    try:
        structured_llm = paraphraser_llm.with_structured_output(CompressedIncident)
        result = structured_llm.invoke(
            [
                SystemMessage(content=COMPRESSION_SYSTEM_PROMPT),
                HumanMessage(content=raw_prompt),
            ]
        )
        latency = time.perf_counter() - start
        return result, latency
    except Exception as exc:
        logging.warning("Primary structured output path failed: %s", exc)

    parser = PydanticOutputParser(pydantic_object=CompressedIncident)
    fallback_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COMPRESSION_SYSTEM_PROMPT),
            (
                "human",
                "Compress the following incident report.\n"
                "Return only the required JSON structure.\n"
                "{format_instructions}\n\n"
                "Incident:\n{raw_prompt}",
            ),
        ]
    )
    fallback_chain = fallback_prompt | paraphraser_llm | parser
    result = fallback_chain.invoke(
        {
            "raw_prompt": raw_prompt,
            "format_instructions": parser.get_format_instructions(),
        }
    )
    latency = time.perf_counter() - start
    return result, latency


def run_pipeline_a(
    reasoning_llm: BaseChatModel | None,
    raw_prompt: str,
    full_logs: list[str],
    provider: str,
) -> tuple[str, float, int]:
    """Baseline: send full prompt + full logs directly to the reasoning model."""
    if provider == "mock" or reasoning_llm is None:
        start = time.perf_counter()
        output = _mock_reason_raw(raw_prompt, full_logs)
        latency = time.perf_counter() - start
        return output, latency, len(full_logs)

    logs_blob = "\n".join(full_logs)
    reasoning_prompt = textwrap.dedent(
        f"""
        You are a senior distributed systems incident responder.

        Analyze this raw user report and full log dump.
        Provide:
        1) Most likely root cause
        2) Why this evidence supports it
        3) Immediate mitigation steps
        4) Next 5 targeted log/metric queries

        Raw user prompt:
        {raw_prompt}

        Full logs ({len(full_logs)} lines):
        {logs_blob}
        """
    ).strip()

    start = time.perf_counter()
    response = reasoning_llm.invoke([HumanMessage(content=reasoning_prompt)])
    latency = time.perf_counter() - start
    return str(response.content), latency, len(full_logs)


def _execute_tool_call(tool_call: dict[str, Any]) -> str:
    name = tool_call.get("name", "")
    args = tool_call.get("args", {})
    if name != "query_log_cache":
        return f"Unsupported tool call: {name}"

    if isinstance(args, dict):
        return query_log_cache.invoke(args)
    if isinstance(args, str):
        return query_log_cache.invoke({"keyword": args})
    return "Invalid args for query_log_cache"


def run_pipeline_b(
    reasoning_llm: BaseChatModel | None,
    compressed: CompressedIncident,
    provider: str,
    max_tool_rounds: int = 4,
) -> tuple[str, float, int, int]:
    """Optimized: use compressed prompt + dynamic retrieval from in-memory log cache."""
    start = time.perf_counter()

    if provider == "mock" or reasoning_llm is None:
        output, tool_calls, retrieved_lines = _mock_reason_optimized(compressed)
        latency = time.perf_counter() - start
        return output, latency, tool_calls, retrieved_lines

    model_with_tools = reasoning_llm.bind_tools([query_log_cache])

    messages = [
        SystemMessage(
            content=(
                "You are a principal SRE incident analyst. Use query_log_cache whenever you need evidence "
                "from logs. Do not ask for full logs. Build a focused diagnosis."
            )
        ),
        HumanMessage(
            content=(
                "Compressed incident brief JSON:\n"
                f"{compressed.model_dump_json(indent=2)}\n\n"
                "You may call query_log_cache multiple times. Then return:\n"
                "1) most likely root cause\n"
                "2) supporting evidence\n"
                "3) immediate mitigations\n"
                "4) next observability checks"
            )
        ),
    ]

    tool_calls_total = 0
    retrieved_lines_total = 0

    for _ in range(max_tool_rounds):
        ai_message = model_with_tools.invoke(messages)
        messages.append(ai_message)
        tool_calls = getattr(ai_message, "tool_calls", None) or []

        if not tool_calls:
            latency = time.perf_counter() - start
            return str(ai_message.content), latency, tool_calls_total, retrieved_lines_total

        tool_calls_total += len(tool_calls)
        for tc in tool_calls:
            tool_output = _execute_tool_call(tc)
            retrieved_lines_total += tool_output.count("\n") + 1
            messages.append(ToolMessage(content=tool_output, tool_call_id=tc.get("id", "unknown")))

    final_message = reasoning_llm.invoke(messages)
    latency = time.perf_counter() - start
    return str(final_message.content), latency, tool_calls_total, retrieved_lines_total


def load_logs(log_file: str | None, max_log_lines: int = 6000) -> list[str]:
    """Load logs from a text file, else generate a mock corpus at the requested size."""
    if not log_file:
        return build_mock_log_cache(total_lines=max_log_lines)

    path = Path(log_file)
    if not path.exists():
        raise FileNotFoundError(f"log file not found: {log_file}")

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [line.rstrip("\n") for line in handle]

    if not lines:
        raise ValueError(f"log file is empty: {log_file}")

    return lines[:max_log_lines]


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _keyword_quality_proxy(output: str, keywords: list[str]) -> tuple[float, float, float]:
    normalized_output = output.lower()
    answer_words = set(re.findall(r"[a-z0-9]+", normalized_output))
    matched = sum(1 for keyword in keywords if keyword.lower() in normalized_output)
    precision = matched / len(answer_words) if answer_words else 0.0
    recall = matched / len(keywords) if keywords else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def build_comparison_metrics(
    raw_prompt: str,
    compressed: CompressedIncident,
    pipe_a_output: str,
    pipe_c_output: str,
    pipe_a_latency: float,
    pipe_c_latency: float,
    pipe_a_log_lines: int,
    pipe_c_retrieved_lines: int,
    pipe_c_tool_calls: int,
) -> dict[str, Any]:
    """Build experiment-style metrics for comparing Pipe A vs Pipe C."""
    reference_keywords = [
        "cosmos",
        "ingress",
        "timeout",
        "retry",
        "aks",
        "21012",
        "latency",
        "dependency",
        *compressed.technical_identifiers[:8],
    ]

    pipe_a_precision, pipe_a_recall, pipe_a_kw_f1 = _keyword_quality_proxy(pipe_a_output, reference_keywords)
    pipe_c_precision, pipe_c_recall, pipe_c_kw_f1 = _keyword_quality_proxy(pipe_c_output, reference_keywords)

    raw_prompt_tokens = _estimate_tokens(raw_prompt)
    compressed_prompt_tokens = _estimate_tokens(compressed.model_dump_json(indent=2))
    token_reduction_pct = (
        max(0.0, (raw_prompt_tokens - compressed_prompt_tokens) / raw_prompt_tokens * 100.0)
        if raw_prompt_tokens
        else 0.0
    )

    return {
        "pipe_a_prompt_tokens": raw_prompt_tokens,
        "pipe_c_prompt_tokens": compressed_prompt_tokens,
        "token_reduction_pct": token_reduction_pct,
        "pipe_a_latency_s": pipe_a_latency,
        "pipe_c_latency_s": pipe_c_latency,
        "pipe_a_kw_f1": pipe_a_kw_f1,
        "pipe_c_kw_f1": pipe_c_kw_f1,
        "pipe_a_judge_score": pipe_a_recall,
        "pipe_c_judge_score": pipe_c_recall,
        "pipe_a_precision": pipe_a_precision,
        "pipe_c_precision": pipe_c_precision,
        "pipe_a_log_lines": pipe_a_log_lines,
        "pipe_c_retrieval_lines": pipe_c_retrieved_lines,
        "pipe_c_tool_calls": pipe_c_tool_calls,
    }


def print_comparison_report(metrics: dict[str, Any]) -> None:
    print_section("Pipe A vs Pipe C Comparison")
    print(f"pipe_a_prompt_tokens:     {metrics['pipe_a_prompt_tokens']}")
    print(f"pipe_c_prompt_tokens:     {metrics['pipe_c_prompt_tokens']}")
    print(f"token_reduction_pct:      {metrics['token_reduction_pct']:.2f}%")
    print(f"pipe_a_latency_s:         {metrics['pipe_a_latency_s']:.4f}")
    print(f"pipe_c_latency_s:         {metrics['pipe_c_latency_s']:.4f}")
    print(f"pipe_a_kw_f1:             {metrics['pipe_a_kw_f1']:.3f}")
    print(f"pipe_c_kw_f1:             {metrics['pipe_c_kw_f1']:.3f}")
    print(f"pipe_a_judge_score:       {metrics['pipe_a_judge_score']:.3f}")
    print(f"pipe_c_judge_score:       {metrics['pipe_c_judge_score']:.3f}")
    print(f"pipe_a_log_lines:         {metrics['pipe_a_log_lines']}")
    print(f"pipe_c_retrieval_lines:   {metrics['pipe_c_retrieval_lines']}")
    print(f"pipe_c_tool_calls:        {metrics['pipe_c_tool_calls']}")


def print_telemetry(
    raw_prompt: str,
    compressed: CompressedIncident,
    compression_latency_s: float,
    baseline_reasoning_latency_s: float,
    optimized_reasoning_latency_s: float,
    tot_reasoning_latency_s: float,
    tool_calls: int,
    raw_log_lines_processed: int,
    optimized_log_lines_retrieved: int,
    tot_log_lines_retrieved: int,
) -> None:
    raw_chars = len(raw_prompt)
    compressed_payload = compressed.model_dump_json()
    compressed_chars = len(compressed_payload)
    saved_chars = raw_chars - compressed_chars
    saved_pct = (saved_chars / raw_chars * 100.0) if raw_chars else 0.0

    print_section("Telemetry")
    print(f"raw_char_count:        {raw_chars}")
    print(f"compressed_char_count: {compressed_chars}")
    print(f"char_savings:          {saved_chars} ({saved_pct:.2f}% reduction)")
    print(f"compression_latency_s: {compression_latency_s:.4f}")
    print(f"pipe_a_reasoning_s:    {baseline_reasoning_latency_s:.4f}")
    print(f"pipe_b_reasoning_s:    {optimized_reasoning_latency_s:.4f}")
    print(f"pipe_c_reasoning_s:    {tot_reasoning_latency_s:.4f}")
    print(f"pipe_b_tool_calls:     {tool_calls}")
    print(f"pipe_a_log_lines:      {raw_log_lines_processed}")
    print(f"pipe_b_log_lines:      {optimized_log_lines_retrieved}")
    print(f"pipe_c_log_lines:      {tot_log_lines_retrieved}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Token Compression + In-Memory Log Retrieval (Pipe A vs Pipe B)."
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "groq", "mock"],
        default=os.getenv("LLM_PROVIDER", "mock"),
    )
    parser.add_argument("--small-model", default=None, help="Paraphraser model name")
    parser.add_argument("--reasoning-model", default=None, help="Reasoning model name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--pipeline",
        choices=["both", "raw", "optimized", "tot", "all", "compare"],
        default=os.getenv("PIPELINE_MODE", "compare"),
        help="Run baseline, optimized, and/or ToT-style reasoning flows. 'compare' runs only Pipe A and Pipe C.",
    )
    parser.add_argument(
        "--log-file",
        default=os.getenv("LOG_FILE"),
        help="Optional path to an external log file for large-context testing.",
    )
    parser.add_argument(
        "--max-log-lines",
        type=int,
        default=int(os.getenv("MAX_LOG_LINES", "2000")),
        help="Maximum number of log lines to load from --log-file. Defaults to a medium corpus size.",
    )
    parser.add_argument(
        "--metrics-json",
        default=os.getenv("METRICS_JSON_PATH"),
        help="Optional output path for structured metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    small_default, reasoning_default = default_model_names(args.provider)
    config = ModelConfig(
        provider=args.provider,
        small_model=args.small_model or os.getenv("SMALL_MODEL", small_default),
        reasoning_model=args.reasoning_model or os.getenv("REASONING_MODEL", reasoning_default),
        temperature=args.temperature,
    )

    print_section("Configuration")
    print(f"provider:        {config.provider}")
    print(f"small_model:     {config.small_model}")
    print(f"reasoning_model: {config.reasoning_model}")
    print(f"pipeline_mode:   {args.pipeline}")

    active_logs = load_logs(args.log_file, max_log_lines=args.max_log_lines)
    global MOCK_LOG_CACHE
    MOCK_LOG_CACHE = active_logs
    print(f"log_cache_lines: {len(active_logs)}")
    if args.log_file:
        print(f"log_file:        {args.log_file}")

    paraphraser_llm = init_chat_model(config.provider, config.small_model, config.temperature)
    reasoning_llm = init_chat_model(config.provider, config.reasoning_model, config.temperature)

    print_section("Input Prompt")
    print(MOCK_INCIDENT_PROMPT)

    compressed: CompressedIncident | None = None
    compression_latency = 0.0
    pipe_a_latency = 0.0
    pipe_b_latency = 0.0
    pipe_c_latency = 0.0
    tool_calls = 0
    raw_lines = 0
    optimized_lines = 0
    tot_lines = 0

    include_raw = args.pipeline in {"both", "raw", "optimized", "tot", "all", "compare"}
    include_optimized = args.pipeline in {"both", "optimized", "tot", "all"}
    include_tot = args.pipeline in {"tot", "all", "compare"}
    need_compressed = include_optimized or args.pipeline == "compare"

    if need_compressed:
        compressed, compression_latency = run_compression_step(paraphraser_llm, MOCK_INCIDENT_PROMPT, config.provider)
        print_section("Compressed Incident")
        print(compressed.model_dump_json(indent=2))

    if include_raw:
        pipe_a_output, pipe_a_latency, raw_lines = run_pipeline_a(
            reasoning_llm,
            MOCK_INCIDENT_PROMPT,
            active_logs,
            config.provider,
        )
        print_section("Pipe A - Baseline (Raw Prompt + Full Logs)")
        print(pipe_a_output)

    if include_optimized:
        if compressed is None:
            raise RuntimeError("compressed payload is required for optimized pipeline")
        pipe_b_output, pipe_b_latency, tool_calls, optimized_lines = run_pipeline_b(
            reasoning_llm,
            compressed,
            config.provider,
        )
        print_section("Pipe B - Optimized (Compressed Prompt + query_log_cache Tool)")
        print(pipe_b_output)

    if include_tot:
        if compressed is None:
            raise RuntimeError("compressed payload is required for ToT pipeline")
        pipe_c_output, pipe_c_latency, _, _, _, tot_lines = run_pipeline_c(
            reasoning_llm,
            compressed,
            config.provider,
        )
        print_section("Pipe C - ToT-Enhanced Reasoning")
        print(pipe_c_output)

    if compressed is None:
        compressed = CompressedIncident(
            core_issue="N/A (raw-only run)",
            observed_symptoms=[],
            technical_identifiers=[],
        )

    comparison_metrics: dict[str, Any] | None = None
    if args.pipeline == "compare":
        comparison_metrics = build_comparison_metrics(
            raw_prompt=MOCK_INCIDENT_PROMPT,
            compressed=compressed,
            pipe_a_output=pipe_a_output,
            pipe_c_output=pipe_c_output,
            pipe_a_latency=pipe_a_latency,
            pipe_c_latency=pipe_c_latency,
            pipe_a_log_lines=raw_lines,
            pipe_c_retrieved_lines=tot_lines,
            pipe_c_tool_calls=tool_calls,
        )
        print_comparison_report(comparison_metrics)

    print_telemetry(
        raw_prompt=MOCK_INCIDENT_PROMPT,
        compressed=compressed,
        compression_latency_s=compression_latency,
        baseline_reasoning_latency_s=pipe_a_latency,
        optimized_reasoning_latency_s=pipe_b_latency,
        tot_reasoning_latency_s=pipe_c_latency,
        tool_calls=tool_calls,
        raw_log_lines_processed=raw_lines,
        optimized_log_lines_retrieved=optimized_lines,
        tot_log_lines_retrieved=tot_lines,
    )

    metrics = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "provider": config.provider,
        "small_model": config.small_model,
        "reasoning_model": config.reasoning_model,
        "pipeline_mode": args.pipeline,
        "log_line_count": len(active_logs),
        "raw_char_count": len(MOCK_INCIDENT_PROMPT),
        "compressed_char_count": len(compressed.model_dump_json()),
        "compression_latency_s": compression_latency,
        "pipe_a_reasoning_s": pipe_a_latency,
        "pipe_b_reasoning_s": pipe_b_latency,
        "pipe_c_reasoning_s": pipe_c_latency,
        "pipe_b_tool_calls": tool_calls,
        "pipe_a_log_lines": raw_lines,
        "pipe_b_log_lines": optimized_lines,
        "pipe_c_log_lines": tot_lines,
    }
    if comparison_metrics is not None:
        metrics["comparison_metrics"] = comparison_metrics

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"metrics_json_path: {metrics_path}")


if __name__ == "__main__":
    main()
