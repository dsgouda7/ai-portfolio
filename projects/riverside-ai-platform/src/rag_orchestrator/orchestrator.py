from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from endpoint_client import AsyncEndpointClient, ChatCompletionRequest, ChatCompletionResponse, Citation
from endpoint_client.models import AnswerSpan, AssistantMessage, ChatMessage, CompletionChoice, RetrievalOptions

from .models import OrchestratorConfig, RetrievedChunk, SearchQuery, TrustedAuthContext
from .protocols import NoopTelemetryHooks, SearchIndex, TelemetryHooks


_CITATION_MARKER = re.compile(r"<cite:([A-Za-z0-9][A-Za-z0-9._:-]{0,127})>")


def _bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 10:
        return "6-10"
    return "11-20"


class RAGOrchestrator:
    def __init__(
        self,
        search_index: SearchIndex,
        endpoint_client: AsyncEndpointClient,
        *,
        config: OrchestratorConfig | None = None,
        telemetry: TelemetryHooks | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._search_index = search_index
        self._endpoint_client = endpoint_client
        self._config = config or OrchestratorConfig()
        self._telemetry = telemetry or NoopTelemetryHooks()
        self._clock = clock

    def _authorized(self, chunk: RetrievedChunk, auth: TrustedAuthContext) -> bool:
        if chunk.tenant_id != auth.tenant_id or chunk.deletion_status != "active":
            return False
        if chunk.acl.visibility == "tenant":
            return True
        return auth.principal_id in chunk.acl.principals or bool(auth.group_ids & chunk.acl.groups)

    def _query_text(self, request: ChatCompletionRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content
        raise ValueError("a RAG request must contain a user message")

    async def _retrieve(
        self,
        request: ChatCompletionRequest,
        auth: TrustedAuthContext,
    ) -> list[RetrievedChunk]:
        retrieval = request.retrieval
        requested_top_k = retrieval.top_k if retrieval is not None else self._config.default_top_k
        top_k = min(requested_top_k, self._config.max_top_k)
        query = SearchQuery(
            text=self._query_text(request),
            top_k=top_k,
            search_type=(retrieval.search_type if retrieval and retrieval.search_type else self._config.default_search_type),
            filters=retrieval.filters if retrieval else None,
            auth=auth,
        )
        try:
            raw_results = await self._search_index.search(query)
        except Exception:
            await self._telemetry.failed(tenant_tier=auth.tenant_tier, category="retrieval_failure")
            raise
        authorized = [chunk for chunk in raw_results[:top_k] if self._authorized(chunk, auth)]
        await self._telemetry.retrieval_finished(
            tenant_tier=auth.tenant_tier,
            requested_top_k=top_k,
            result_count_bucket=_bucket(len(authorized)),
            outcome="success" if authorized else "empty",
        )
        return authorized

    def _context_message(self, chunks: Sequence[RetrievedChunk]) -> ChatMessage:
        remaining = self._config.max_context_characters
        blocks: list[str] = []
        for chunk in chunks:
            content = chunk.content[: min(self._config.max_chunk_characters, remaining)]
            if not content:
                break
            blocks.append(f"<source id=\"{chunk.chunk_id}\">\n{content}\n</source>")
            remaining -= len(content)
            if remaining <= 0:
                break
        context = "\n\n".join(blocks)
        instructions = (
            "Use only the authorized source blocks below. After every supported sentence, append "
            "a marker in the exact form <cite:chunk_id> immediately after the final punctuation. "
            "Do not invent identifiers. If the sources do not support an answer, refuse.\n\n"
        )
        return ChatMessage(role="system", content=f"{instructions}{context}")

    def _generation_request(
        self,
        request: ChatCompletionRequest,
        chunks: Sequence[RetrievedChunk],
    ) -> ChatCompletionRequest:
        if len(request.messages) >= 128:
            raise ValueError("RAG context requires one available message slot")
        return request.model_copy(
            update={
                "messages": [self._context_message(chunks), *request.messages],
                "stream": False,
                "retrieval": RetrievalOptions(enabled=False, top_k=1),
            }
        )

    def _resolve_citations(
        self,
        raw_answer: str,
        chunks: Sequence[RetrievedChunk],
    ) -> tuple[str, list[Citation]]:
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        output: list[str] = []
        spans: dict[str, list[AnswerSpan]] = {}
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
                spans.setdefault(chunk_id, []).append(AnswerSpan(start=start, end=end))
            cursor = match.end()
        output.append(raw_answer[cursor:])
        answer = "".join(output)
        retrieved_at = self._clock()
        citations = [
            Citation(
                citation_id=f"cite-{index:03d}",
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_uri=chunk.source_uri,
                source_version=chunk.source_version,
                content_hash=chunk.content_hash,
                title=chunk.title,
                answer_spans=spans[chunk.chunk_id],
                retrieved_at=retrieved_at,
            )
            for index, chunk in enumerate(chunks, start=1)
            if chunk.chunk_id in spans
        ]
        citations.sort(key=lambda citation: citation.answer_spans[0].start)
        covered = [False] * len(answer)
        for citation in citations:
            for span in citation.answer_spans:
                for position in range(span.start, span.end):
                    covered[position] = True
        if any(not character.isspace() and not covered[index] for index, character in enumerate(answer)):
            return answer, []
        return answer, citations

    def _refusal(self, response: ChatCompletionResponse) -> ChatCompletionResponse:
        message = AssistantMessage(content=self._config.refusal_message, refusal=self._config.refusal_message)
        return response.model_copy(
            update={
                "choices": [CompletionChoice(message=message, finish_reason="stop")],
                "citations": [],
            }
        )

    async def complete(
        self,
        request: ChatCompletionRequest,
        auth: TrustedAuthContext,
    ) -> ChatCompletionResponse:
        if request.retrieval is not None and not request.retrieval.enabled:
            return await self._endpoint_client.complete(request)
        chunks = await self._retrieve(request, auth)
        if not chunks:
            response = await self._endpoint_client.complete(
                self._generation_request(request, [])
            )
            refused = self._refusal(response)
            await self._telemetry.generation_finished(
                tenant_tier=auth.tenant_tier,
                outcome="refusal",
                citation_count_bucket="0",
                finish_reason="stop",
            )
            return refused
        try:
            response = await self._endpoint_client.complete(self._generation_request(request, chunks))
        except Exception:
            await self._telemetry.failed(tenant_tier=auth.tenant_tier, category="generation_failure")
            raise
        answer, citations = self._resolve_citations(response.choices[0].message.content, chunks)
        if not citations:
            normalized = self._refusal(response)
            outcome = "refusal"
        else:
            normalized = response.model_copy(
                update={
                    "choices": [
                        CompletionChoice(
                            message=AssistantMessage(content=answer, refusal=response.choices[0].message.refusal),
                            finish_reason=response.choices[0].finish_reason,
                        )
                    ],
                    "citations": citations,
                }
            )
            outcome = "success"
        await self._telemetry.generation_finished(
            tenant_tier=auth.tenant_tier,
            outcome=outcome,
            citation_count_bucket=_bucket(len(normalized.citations)),
            finish_reason=normalized.choices[0].finish_reason,
        )
        return normalized
