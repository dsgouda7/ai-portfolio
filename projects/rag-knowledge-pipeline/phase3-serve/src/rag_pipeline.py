"""Provider-neutral retrieval-augmented generation orchestration."""

from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from shared.contract_adapters import AuthorizationContext, RetrievedRecord, utc_now
from shared.logging_config import get_logger

from .providers import GenerationResult, build_generation_provider
from .retrieval import build_retriever


logger = get_logger(__name__)
_CITATION_MARKER = re.compile(r"<cite:([A-Za-z0-9][A-Za-z0-9._:-]{0,127})>")


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    citations: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    retry_count: int
    deployment: Mapping[str, Any]
    sources_count: int


class RAGPipeline:
    """Compose independently selectable retrieval and generation providers."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.mode = str(config["mode"])
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.retriever = build_retriever(config, self.device)
        self.generator = build_generation_provider(config)
        self.chat_history: list[tuple[str, str]] = []
        self._started_at = utc_now()
        logger.info("RAG pipeline initialized in %s mode on %s", self.mode, self.device)

    def _context_message(self, records: Sequence[RetrievedRecord]) -> dict[str, str]:
        blocks = "\n\n".join(
            f'<source id="{record.chunk_id}">\n{record.content}\n</source>'
            for record in records
        )
        instructions = (
            "Use only the source blocks below. After every supported sentence, append a marker "
            "in the exact form <cite:chunk_id>. Do not invent identifiers. If the sources do not "
            "support an answer, say that the authorized sources do not contain the answer."
        )
        return {"role": "system", "content": f"{instructions}\n\n{blocks}"}

    def _resolve_citations(
        self,
        raw_answer: str,
        records: Sequence[RetrievedRecord],
    ) -> tuple[str, list[dict[str, Any]]]:
        by_id = {record.chunk_id: record for record in records}
        output: list[str] = []
        spans: dict[str, list[dict[str, int]]] = {}
        cursor = 0
        for match in _CITATION_MARKER.finditer(raw_answer):
            output.append(raw_answer[cursor : match.start()])
            current = "".join(output)
            end = len(current.rstrip())
            boundary = max(current.rfind(". ", 0, max(0, end - 1)), current.rfind("\n", 0, end))
            start = boundary + (2 if boundary >= 0 and current[boundary : boundary + 2] == ". " else 1)
            while start < end and current[start].isspace():
                start += 1
            chunk_id = match.group(1)
            if chunk_id in by_id and end > start:
                spans.setdefault(chunk_id, []).append({"start": start, "end": end})
            cursor = match.end()
        output.append(raw_answer[cursor:])
        answer = "".join(output)
        citations = []
        for index, record in enumerate(records, start=1):
            if record.chunk_id not in spans:
                continue
            citations.append(
                {
                    "contract_version": "1.0.0",
                    "kind": "citation",
                    "citation_id": f"cite-{index:03d}",
                    "chunk_id": record.chunk_id,
                    "document_id": record.document_id,
                    "source_uri": record.source_uri,
                    "source_version": record.source_version,
                    "content_hash": record.content_hash,
                    "title": record.title,
                    "answer_spans": spans[record.chunk_id],
                    "retrieved_at": utc_now(),
                }
            )
        citations.sort(key=lambda citation: citation["answer_spans"][0]["start"])
        return answer, citations

    def _deployment(
        self,
        generation: GenerationResult,
        records: Sequence[RetrievedRecord],
        authorization: AuthorizationContext,
    ) -> Mapping[str, Any]:
        if generation.deployment:
            return generation.deployment
        serving = self.config["serving"]
        configured = serving.get("deployment", {})
        generation_config = serving["generation"]
        provider = str(generation_config["provider"])
        index_version = records[0].index_version if records else str(
            self.config["remote"].get("index_version", "0.1.0-legacy-local")
        )
        return {
            "contract_version": "1.0.0",
            "kind": "deployment_metadata",
            "environment": str(configured.get("environment", "dev")),
            "release_id": str(configured.get("release_id", "local-development")),
            "model_alias": str(generation_config["model_alias"]),
            "deployment_name": str(configured.get("deployment_name", provider.replace("_", "-"))),
            "deployment_slot": str(configured.get("deployment_slot", "blue")),
            "region": str(configured.get("region", authorization.region)),
            "runtime": str(configured.get("runtime", "rag-phase3-serve")),
            "runtime_version": str(configured.get("runtime_version", "0.2.0")),
            "index_version": index_version,
            "deployed_at": str(configured.get("deployed_at", self._started_at)),
            "source_commit": str(configured.get("source_commit", "0000000")),
        }

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        authorization: AuthorizationContext,
        retrieval_enabled: bool = True,
        top_k: int | None = None,
        search_type: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_input_tokens: int | None = None,
        retrieval_filters: Mapping[str, Any] | None = None,
    ) -> PipelineResult:
        generation_config = self.config["serving"]["generation"]
        retrieval_config = self.config.get("retrieval", {})
        question = next(
            (str(message["content"]) for message in reversed(messages) if message["role"] == "user"),
            None,
        )
        if question is None:
            raise ValueError("A RAG request requires at least one user message")

        records: Sequence[RetrievedRecord] = []
        if retrieval_enabled:
            scoped_authorization = authorization
            if retrieval_filters:
                requested_region = retrieval_filters.get("region")
                if requested_region and requested_region != authorization.region:
                    raise ValueError("Retrieval region cannot differ from the trusted region")
                requested_classifications = retrieval_filters.get("classification")
                if requested_classifications:
                    allowed = tuple(
                        value
                        for value in authorization.classifications
                        if value in requested_classifications
                    )
                    if not allowed:
                        raise ValueError("Retrieval classifications are outside the trusted scope")
                    scoped_authorization = AuthorizationContext(
                        tenant_id=authorization.tenant_id,
                        region=authorization.region,
                        classifications=allowed,
                        principal_ids=authorization.principal_ids,
                        group_ids=authorization.group_ids,
                    )
            records = self.retriever.retrieve(
                question,
                top_k=top_k or int(retrieval_config.get("top_k", 6)),
                search_type=search_type or str(retrieval_config.get("search_type", "mmr")),
                authorization=scoped_authorization,
                filters=retrieval_filters,
            )
        generation_messages = [self._context_message(records), *messages] if retrieval_enabled else list(messages)
        generation = self.generator.complete(
            generation_messages,
            temperature=float(
                generation_config.get("temperature", 0.1) if temperature is None else temperature
            ),
            max_tokens=int(generation_config["max_tokens"] if max_tokens is None else max_tokens),
            max_input_tokens=int(
                generation_config["max_input_tokens"]
                if max_input_tokens is None
                else max_input_tokens
            ),
        )
        answer, citations = self._resolve_citations(generation.content, records)
        return PipelineResult(
            answer=answer,
            citations=citations,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            finish_reason=generation.finish_reason,
            retry_count=generation.retry_count,
            deployment=self._deployment(generation, records, authorization),
            sources_count=len(records),
        )

    def query(
        self,
        question: str,
        *,
        authorization: AuthorizationContext,
        top_k: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        result = self.complete(
            [{"role": "user", "content": question}],
            authorization=authorization,
            top_k=top_k,
            temperature=temperature,
        )
        self.chat_history.append((question, result.answer))
        return {
            "answer": result.answer,
            "question": question,
            "sources_count": result.sources_count,
        }

    def reset_history(self) -> None:
        self.chat_history = []

    def get_status(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "retrieval": self.retriever.status(),
            "generation": self.generator.status(),
            "device": self.device,
        }

    def contract_response(self, result: PipelineResult, model_alias: str) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-{secrets.token_urlsafe(12)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_alias,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.answer},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
            "citations": result.citations,
            "trace": {"trace_id": secrets.token_hex(16), "retry_count": result.retry_count},
            "deployment": result.deployment,
        }
