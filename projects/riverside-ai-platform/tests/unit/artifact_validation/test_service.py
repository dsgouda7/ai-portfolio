from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from typing import cast

import pytest

from src.artifact_validation import (
    ChatCompletionRequest,
    GenerationResult,
    LifecycleState,
    ModelServingService,
    OverloadedError,
    RequestValidationError,
    VerifiedRelease,
)


class FakeVerifier:
    def verify_file(self, manifest_path: Path) -> VerifiedRelease:
        return cast(VerifiedRelease, object())


class FakeBackend:
    def __init__(self, *, prompt_tokens: int = 8, warmup_error: Exception | None = None) -> None:
        self.prompt_tokens = prompt_tokens
        self.warmup_error = warmup_error
        self.loaded = False
        self.warmed = False
        self.entered = Event()
        self.release = Event()
        self.block = False

    def load(self, release: VerifiedRelease) -> None:
        self.loaded = True

    def warmup(self) -> None:
        if self.warmup_error is not None:
            raise self.warmup_error
        self.warmed = True

    def count_input_tokens(self, request: ChatCompletionRequest) -> int:
        return self.prompt_tokens

    def generate(self, request: ChatCompletionRequest) -> GenerationResult:
        if self.block:
            self.entered.set()
            self.release.wait(timeout=2)
        return GenerationResult("A continuation.", self.prompt_tokens, 3, "stop")


def _request(**updates: object) -> ChatCompletionRequest:
    values: dict[str, object] = {
        "model": "riverside-editor",
        "messages": [{"role": "user", "content": "Continue."}],
        "max_input_tokens": 32,
        "max_tokens": 16,
        "stream": False,
    }
    values.update(updates)
    return ChatCompletionRequest.model_validate(values)


def _service(backend: FakeBackend) -> ModelServingService:
    service = ModelServingService(
        verifier=cast(object, FakeVerifier()),
        backend=backend,
        model_alias="riverside-editor",
        max_input_tokens=64,
        max_output_tokens=32,
        max_concurrency=1,
    )
    service.initialize(Path("unused.json"))
    return service


def test_readiness_requires_verification_load_and_warmup() -> None:
    backend = FakeBackend()

    service = _service(backend)

    assert service.state is LifecycleState.READY
    assert service.ready
    assert backend.loaded and backend.warmed


def test_warmup_failure_never_becomes_ready() -> None:
    backend = FakeBackend(warmup_error=RuntimeError("warmup failed"))
    service = ModelServingService(
        verifier=cast(object, FakeVerifier()),
        backend=backend,
        model_alias="riverside-editor",
        max_input_tokens=64,
        max_output_tokens=32,
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        service.initialize(Path("unused.json"))

    assert service.state is LifecycleState.FAILED
    assert not service.ready


def test_actual_prompt_tokens_are_bounded() -> None:
    service = _service(FakeBackend(prompt_tokens=33))

    with pytest.raises(RequestValidationError):
        service.generate(_request(max_input_tokens=32))


def test_concurrent_request_is_rejected_without_queueing() -> None:
    backend = FakeBackend()
    backend.block = True
    service = _service(backend)
    worker = Thread(target=service.generate, args=(_request(),))
    worker.start()
    assert backend.entered.wait(timeout=1)

    try:
        with pytest.raises(OverloadedError):
            service.generate(_request())
    finally:
        backend.release.set()
        worker.join(timeout=2)
