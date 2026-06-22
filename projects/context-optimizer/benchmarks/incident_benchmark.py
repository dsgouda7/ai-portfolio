"""
Context Optimizer E2E Benchmark — Pipe A (raw) vs Pipe B (compressed+tools) vs Pipe C (ToT).

Requires:
  --provider  ollama|groq
  --log-file  path to a real log file (structured text, one entry per line)

Usage:
    python benchmarks/incident_benchmark.py \
        --provider ollama \
        --log-file /path/to/incident.log \
        --pipeline compare

Environment variables:
    LLM_PROVIDER       ollama|groq           (overrides --provider)
    SMALL_MODEL        model name for compression LLM
    REASONING_MODEL    model name for reasoning LLM
    OLLAMA_BASE_URL    default: http://localhost:11434
    GROQ_API_KEY       required when provider=groq
    LOG_FILE           path to log file      (overrides --log-file)
    MAX_LOG_LINES      max lines to load     (overrides --max-log-lines)
    METRICS_JSON_PATH  output path for JSON  (overrides --metrics-json)
    PIPELINE_MODE      compare|all|raw|...   (overrides --pipeline)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
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
        from langchain_community.chat_models import ChatOllama  # type: ignore[no-redef]
    except ImportError:
        ChatOllama = None  # type: ignore[assignment]

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover
    ChatGroq = None  # type: ignore[assignment]

# ToTReasoner is baked into src/context_optimizer.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from context_optimizer.tot_reasoner import ToTReasoner  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

COMPRESSION_SYSTEM_PROMPT = (
    "You are a highly efficient Token Compression Filter for a distributed systems debugging agent. "
    "Your job is to strip out all human rambling, conversational fluff, and emotional context from "
    "the user's prompt. Convert the input into a dense, robotic, engineering brief optimized for a "
    "downstream reasoning LLM. Retain all technical identifiers exactly (IPs, service names like AKS, "
    "CosmosDB, metrics, error codes). Do not answer the user's problem."
)

# Reference incident prompt — a realistic example used as the benchmark input.
INCIDENT_PROMPT = textwrap.dedent(
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

# Module-level log cache — set by load_logs() / main().
# The query_log_cache @tool closes over this variable.
_active_log_cache: list[str] = []


# ── Data models ───────────────────────────────────────────────────────────────

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


# ── Provider helpers ──────────────────────────────────────────────────────────

def default_model_names(provider: str) -> tuple[str, str]:
    if provider == "groq":
        return "llama-3.1-8b-instant", "llama-3.3-70b-versatile"
    # ollama default
    return "phi4:mini", "qwen3"


def init_chat_model(provider: str, model_name: str, temperature: float = 0.0) -> BaseChatModel:
    """Initialise a real chat model. Supports ollama and groq."""
    normalized = provider.strip().lower()
    if normalized == "ollama":
        if ChatOllama is None:
            raise ImportError(
                "Ollama provider selected but ChatOllama is unavailable. "
                "Install with: pip install langchain-ollama"
            )
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if normalized == "groq":
        if ChatGroq is None:
            raise ImportError(
                "Groq provider selected but langchain-groq is unavailable. "
                "Install with: pip install langchain-groq"
            )
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Export it before running with --provider groq."
            )
        return ChatGroq(model=model_name, temperature=temperature)
    raise ValueError(f"Unknown provider '{provider}'. Supported: ollama, groq")


# ── Log-cache MCP tool ────────────────────────────────────────────────────────

@tool
def query_log_cache(keyword: str, lines_context: int = 5) -> str:
    """Search incident logs and return matching lines with neighbouring context."""
    needle = keyword.strip().lower()
    if not needle:
        return "query_log_cache error: keyword must be non-empty"

    ctx = max(0, min(lines_context, 25))
    max_hits = 8
    hits: list[str] = []

    for idx, line in enumerate(_active_log_cache):
        if needle in line.lower():
            start = max(0, idx - ctx)
            end = min(len(_active_log_cache), idx + ctx + 1)
            excerpt = []
            for j in range(start, end):
                marker = ">" if j == idx else " "
                excerpt.append(f"{marker} [{j:04d}] {_active_log_cache[j]}")
            hits.append("\n".join(excerpt))
            if len(hits) >= max_hits:
                break

    if not hits:
        return f"No matches for keyword='{keyword}'."
    return (
        f"Found {len(hits)} match window(s) for keyword='{keyword}' "
        f"with lines_context={ctx}.\n\n"
        + "\n\n---\n\n".join(hits)
    )


# ── Pipeline functions ────────────────────────────────────────────────────────

def run_compression_step(
    paraphraser_llm: BaseChatModel,
    raw_prompt: str,
) -> tuple[CompressedIncident, float]:
    """Run the Edge Filter chain — compresses raw incident prose into structured JSON."""
    start = time.perf_counter()
    try:
        structured_llm = paraphraser_llm.with_structured_output(CompressedIncident)
        result: CompressedIncident = structured_llm.invoke(
            [
                SystemMessage(content=COMPRESSION_SYSTEM_PROMPT),
                HumanMessage(content=raw_prompt),
            ]
        )
        return result, time.perf_counter() - start
    except Exception as exc:
        logging.warning("Primary structured output failed (%s). Falling back to parser.", exc)

    parser = PydanticOutputParser(pydantic_object=CompressedIncident)
    fallback = ChatPromptTemplate.from_messages(
        [
            ("system", COMPRESSION_SYSTEM_PROMPT),
            (
                "human",
                "Compress the following incident.\n"
                "Return only the JSON.\n{format_instructions}\n\nIncident:\n{raw_prompt}",
            ),
        ]
    )
    result = (fallback | paraphraser_llm | parser).invoke(
        {"raw_prompt": raw_prompt, "format_instructions": parser.get_format_instructions()}
    )
    return result, time.perf_counter() - start


def run_pipeline_a(
    reasoning_llm: BaseChatModel,
    raw_prompt: str,
    full_logs: list[str],
) -> tuple[str, float, int]:
    """Pipe A — Baseline: inject the full raw prompt and all log lines into the LLM."""
    prompt = textwrap.dedent(
        f"""
        You are a senior distributed systems incident responder.
        Analyse this raw user report and full log dump.

        1) Most likely root cause
        2) Why this evidence supports it
        3) Immediate mitigation steps
        4) Next 5 targeted queries

        Raw user prompt:
        {raw_prompt}

        Full logs ({len(full_logs)} lines):
        {chr(10).join(full_logs)}
        """
    ).strip()

    start = time.perf_counter()
    response = reasoning_llm.invoke([HumanMessage(content=prompt)])
    return str(response.content), time.perf_counter() - start, len(full_logs)


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
    reasoning_llm: BaseChatModel,
    compressed: CompressedIncident,
    max_tool_rounds: int = 4,
) -> tuple[str, float, int, int]:
    """Pipe B — Optimised: compressed prompt + dynamic retrieval via query_log_cache tool."""
    model_with_tools = reasoning_llm.bind_tools([query_log_cache])
    messages: list[Any] = [
        SystemMessage(
            content=(
                "You are a principal SRE incident analyst. Use query_log_cache whenever you need evidence "
                "from logs. Do not ask for full logs. Build a focused diagnosis."
            )
        ),
        HumanMessage(
            content=(
                f"Compressed incident brief JSON:\n{compressed.model_dump_json(indent=2)}\n\n"
                "Call query_log_cache as needed, then return:\n"
                "1) most likely root cause\n"
                "2) supporting evidence\n"
                "3) immediate mitigations\n"
                "4) next observability checks"
            )
        ),
    ]

    tool_calls_total = 0
    retrieved_lines_total = 0
    start = time.perf_counter()

    for _ in range(max_tool_rounds):
        ai_msg = model_with_tools.invoke(messages)
        messages.append(ai_msg)
        tcs = getattr(ai_msg, "tool_calls", None) or []
        if not tcs:
            return str(ai_msg.content), time.perf_counter() - start, tool_calls_total, retrieved_lines_total

        tool_calls_total += len(tcs)
        for tc in tcs:
            out = _execute_tool_call(tc)
            retrieved_lines_total += out.count("\n") + 1
            messages.append(ToolMessage(content=out, tool_call_id=tc.get("id", "unknown")))

    final = reasoning_llm.invoke(messages)
    return str(final.content), time.perf_counter() - start, tool_calls_total, retrieved_lines_total


def run_pipeline_c(
    reasoning_llm: BaseChatModel,
    compressed: CompressedIncident,
) -> tuple[str, float, list[dict[str, Any]], str, int, int]:
    """Pipe C — Tree-of-Thought reasoning via ToTReasoner."""
    branch_prompts = [
        ("cosmos",  "Concise hypothesis about CosmosDB or RU saturation causing the incident, plus one search term."),
        ("ingress", "Concise hypothesis about ingress or upstream timeout causing the incident, plus one search term."),
        ("retry",   "Concise hypothesis about retry storms or cancellation waterfalls, plus one search term."),
    ]
    branch_specs: list[dict[str, Any]] = []
    for branch_id, prompt in branch_prompts:
        resp = reasoning_llm.invoke(
            [HumanMessage(content=f"{prompt}\n\nIncident brief:\n{compressed.model_dump_json(indent=2)}")]
        )
        terms = [t.strip() for t in str(resp.content).split(",") if t.strip()][:2] or [branch_id]
        branch_specs.append({"id": branch_id, "title": branch_id, "search_terms": terms})

    class _LogCacheRetriever:
        def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
            snippet = query_log_cache.invoke({"keyword": query, "lines_context": 1})
            return [] if "No matches" in snippet else [{"compressed_summary": snippet}]

    start = time.perf_counter()
    result = ToTReasoner(retriever=_LogCacheRetriever(), top_k_per_term=1).reason(
        compressed, branch_specs=branch_specs
    )
    branches_dicts = [
        {"id": b.id, "title": b.title, "search_terms": b.search_terms,
         "score": b.score, "evidence_hits": b.evidence_hits, "evidence_snippets": b.evidence_snippets}
        for b in result.branches
    ]
    return (
        result.selected_summary,
        time.perf_counter() - start,
        branches_dicts,
        result.selected_branch_id,
        len(result.branches),
        result.total_retrieved_lines,
    )


# ── Log loading ───────────────────────────────────────────────────────────────

def load_logs(log_file: str, max_log_lines: int = 6000) -> list[str]:
    """Load log lines from a file. Trims to max_log_lines."""
    path = Path(log_file)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if not lines:
        raise ValueError(f"Log file is empty: {log_file}")
    return lines[:max_log_lines]


# ── Metrics and reporting ─────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _keyword_quality_proxy(output: str, keywords: list[str]) -> tuple[float, float, float]:
    normalized = output.lower()
    answer_words = set(re.findall(r"[a-z0-9]+", normalized))
    matched = sum(1 for kw in keywords if kw.lower() in normalized)
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
    """Build experiment-style metrics for Pipe A vs Pipe C comparison."""
    reference_keywords = [
        "cosmos", "ingress", "timeout", "retry", "aks", "21012", "latency", "dependency",
        *compressed.technical_identifiers[:8],
    ]
    pipe_a_precision, pipe_a_recall, pipe_a_kw_f1 = _keyword_quality_proxy(pipe_a_output, reference_keywords)
    pipe_c_precision, pipe_c_recall, pipe_c_kw_f1 = _keyword_quality_proxy(pipe_c_output, reference_keywords)

    raw_tokens        = _estimate_tokens(raw_prompt)
    compressed_tokens = _estimate_tokens(compressed.model_dump_json(indent=2))
    token_reduction   = max(0.0, (raw_tokens - compressed_tokens) / raw_tokens * 100.0) if raw_tokens else 0.0

    return {
        "pipe_a_prompt_tokens":   raw_tokens,
        "pipe_c_prompt_tokens":   compressed_tokens,
        "token_reduction_pct":    token_reduction,
        "pipe_a_latency_s":       pipe_a_latency,
        "pipe_c_latency_s":       pipe_c_latency,
        "pipe_a_kw_f1":           pipe_a_kw_f1,
        "pipe_c_kw_f1":           pipe_c_kw_f1,
        "pipe_a_judge_score":     pipe_a_recall,
        "pipe_c_judge_score":     pipe_c_recall,
        "pipe_a_precision":       pipe_a_precision,
        "pipe_c_precision":       pipe_c_precision,
        "pipe_a_log_lines":       pipe_a_log_lines,
        "pipe_c_retrieval_lines": pipe_c_retrieved_lines,
        "pipe_c_tool_calls":      pipe_c_tool_calls,
    }


def print_comparison_report(metrics: dict[str, Any]) -> None:
    print_section("Pipe A vs Pipe C Comparison")
    for key, val in metrics.items():
        print(f"  {key:<30} {val:.4f}" if isinstance(val, float) else f"  {key:<30} {val}")


def print_telemetry(
    raw_prompt: str,
    compressed: CompressedIncident,
    compression_latency_s: float,
    pipe_a_latency_s: float,
    pipe_b_latency_s: float,
    pipe_c_latency_s: float,
    pipe_b_tool_calls: int,
    pipe_a_log_lines: int,
    pipe_b_log_lines: int,
    pipe_c_log_lines: int,
) -> None:
    raw_chars = len(raw_prompt)
    comp_chars = len(compressed.model_dump_json())
    saved_pct = ((raw_chars - comp_chars) / raw_chars * 100.0) if raw_chars else 0.0

    print_section("Telemetry")
    print(f"  raw_chars / compressed_chars: {raw_chars} / {comp_chars} ({saved_pct:.1f}% reduction)")
    print(f"  compression_latency_s:        {compression_latency_s:.4f}")
    print(f"  pipe_a_reasoning_s:           {pipe_a_latency_s:.4f}")
    print(f"  pipe_b_reasoning_s:           {pipe_b_latency_s:.4f}")
    print(f"  pipe_c_reasoning_s:           {pipe_c_latency_s:.4f}")
    print(f"  pipe_b_tool_calls:            {pipe_b_tool_calls}")
    print(f"  pipe_a/b/c log_lines:         {pipe_a_log_lines} / {pipe_b_log_lines} / {pipe_c_log_lines}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Context Optimizer E2E benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "groq"],
        default=os.getenv("LLM_PROVIDER", "ollama"),
    )
    parser.add_argument("--small-model",     default=None)
    parser.add_argument("--reasoning-model", default=None)
    parser.add_argument("--temperature",     type=float, default=0.0)
    parser.add_argument("--log-level",       default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--pipeline",
        choices=["both", "raw", "optimized", "tot", "all", "compare"],
        default=os.getenv("PIPELINE_MODE", "compare"),
    )
    parser.add_argument(
        "--log-file",
        default=os.getenv("LOG_FILE"),
        required=not bool(os.getenv("LOG_FILE")),
        help="Path to log file (one entry per line). Required.",
    )
    parser.add_argument(
        "--max-log-lines",
        type=int,
        default=int(os.getenv("MAX_LOG_LINES", "2000")),
    )
    parser.add_argument("--metrics-json", default=os.getenv("METRICS_JSON_PATH"))
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
    print(f"  provider={config.provider}  small={config.small_model}  "
          f"reasoning={config.reasoning_model}  pipeline={args.pipeline}")

    active_logs = load_logs(args.log_file, max_log_lines=args.max_log_lines)
    global _active_log_cache
    _active_log_cache = active_logs
    print(f"  log_file={args.log_file}  lines={len(active_logs)}")

    paraphraser_llm = init_chat_model(config.provider, config.small_model,    config.temperature)
    reasoning_llm   = init_chat_model(config.provider, config.reasoning_model, config.temperature)

    print_section("Input Prompt")
    print(INCIDENT_PROMPT)

    compressed: CompressedIncident | None = None
    compression_latency = pipe_a_latency = pipe_b_latency = pipe_c_latency = 0.0
    tool_calls = raw_lines = optimized_lines = tot_lines = 0
    pipe_a_output = pipe_c_output = ""

    need_compressed = args.pipeline in {"both", "optimized", "tot", "all", "compare"}

    if need_compressed:
        compressed, compression_latency = run_compression_step(paraphraser_llm, INCIDENT_PROMPT)
        print_section("Compressed Incident")
        print(compressed.model_dump_json(indent=2))

    if args.pipeline in {"both", "raw", "optimized", "tot", "all", "compare"}:
        pipe_a_output, pipe_a_latency, raw_lines = run_pipeline_a(reasoning_llm, INCIDENT_PROMPT, active_logs)
        print_section("Pipe A — Baseline")
        print(pipe_a_output)

    if args.pipeline in {"both", "optimized", "all"}:
        if compressed is None:
            raise RuntimeError("Pipe B requires a compressed payload (run compression first)")
        pipe_b_output, pipe_b_latency, tool_calls, optimized_lines = run_pipeline_b(reasoning_llm, compressed)
        print_section("Pipe B — Optimised (Compressed + query_log_cache)")
        print(pipe_b_output)

    if args.pipeline in {"tot", "all", "compare"}:
        if compressed is None:
            raise RuntimeError("Pipe C requires a compressed payload (run compression first)")
        pipe_c_output, pipe_c_latency, _, _, _, tot_lines = run_pipeline_c(reasoning_llm, compressed)
        print_section("Pipe C — ToT-Enhanced Reasoning")
        print(pipe_c_output)

    if compressed is None:
        compressed = CompressedIncident(
            core_issue="N/A (raw-only run)", observed_symptoms=[], technical_identifiers=[]
        )

    comparison_metrics: dict[str, Any] | None = None
    if args.pipeline == "compare":
        comparison_metrics = build_comparison_metrics(
            raw_prompt=INCIDENT_PROMPT, compressed=compressed,
            pipe_a_output=pipe_a_output, pipe_c_output=pipe_c_output,
            pipe_a_latency=pipe_a_latency, pipe_c_latency=pipe_c_latency,
            pipe_a_log_lines=raw_lines, pipe_c_retrieved_lines=tot_lines,
            pipe_c_tool_calls=tool_calls,
        )
        print_comparison_report(comparison_metrics)

    print_telemetry(
        raw_prompt=INCIDENT_PROMPT, compressed=compressed,
        compression_latency_s=compression_latency,
        pipe_a_latency_s=pipe_a_latency, pipe_b_latency_s=pipe_b_latency, pipe_c_latency_s=pipe_c_latency,
        pipe_b_tool_calls=tool_calls, pipe_a_log_lines=raw_lines,
        pipe_b_log_lines=optimized_lines, pipe_c_log_lines=tot_lines,
    )

    metrics: dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "provider": config.provider,
        "small_model": config.small_model, "reasoning_model": config.reasoning_model,
        "pipeline_mode": args.pipeline, "log_line_count": len(active_logs),
        "compression_latency_s": compression_latency,
        "pipe_a_reasoning_s": pipe_a_latency, "pipe_b_reasoning_s": pipe_b_latency,
        "pipe_c_reasoning_s": pipe_c_latency, "pipe_b_tool_calls": tool_calls,
        "pipe_a_log_lines": raw_lines, "pipe_b_log_lines": optimized_lines, "pipe_c_log_lines": tot_lines,
    }
    if comparison_metrics:
        metrics["comparison_metrics"] = comparison_metrics

    if args.metrics_json:
        out = Path(args.metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\n  metrics saved -> {out}")


if __name__ == "__main__":
    main()
