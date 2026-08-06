from __future__ import annotations

from enum import Enum
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Protocol

from .errors import LifecycleError, OverloadedError, RequestValidationError, ServiceNotReadyError
from .loader import GenerationResult
from .models import ChatCompletionRequest, VerifiedRelease
from .verification import ReleaseVerifier


class GenerationBackend(Protocol):
    def load(self, release: VerifiedRelease) -> None: ...

    def warmup(self) -> None: ...

    def count_input_tokens(self, request: ChatCompletionRequest) -> int: ...

    def generate(self, request: ChatCompletionRequest) -> GenerationResult: ...


class LifecycleState(str, Enum):
    NEW = "new"
    VERIFIED = "verified"
    LOADED = "loaded"
    WARMED = "warmed"
    READY = "ready"
    FAILED = "failed"


class ModelServingService:
    def __init__(
        self,
        *,
        verifier: ReleaseVerifier,
        backend: GenerationBackend,
        model_alias: str,
        max_input_tokens: int,
        max_output_tokens: int,
        max_concurrency: int = 1,
    ) -> None:
        if min(max_input_tokens, max_output_tokens, max_concurrency) < 1:
            raise ValueError("serving limits must be positive")
        self._verifier = verifier
        self._backend = backend
        self._model_alias = model_alias
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self._admission = BoundedSemaphore(max_concurrency)
        self._initialization_lock = Lock()
        self._state = LifecycleState.NEW
        self._release: VerifiedRelease | None = None

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def ready(self) -> bool:
        return self._state is LifecycleState.READY and self._release is not None

    @property
    def release(self) -> VerifiedRelease:
        if self._release is None:
            raise ServiceNotReadyError("release is not available")
        return self._release

    def initialize(self, manifest_path: Path) -> None:
        with self._initialization_lock:
            if self._state is not LifecycleState.NEW:
                raise LifecycleError("serving service can only be initialized once")
            try:
                release = self._verifier.verify_file(manifest_path)
                self._state = LifecycleState.VERIFIED
                self._backend.load(release)
                self._state = LifecycleState.LOADED
                self._backend.warmup()
                self._state = LifecycleState.WARMED
                self._release = release
                self._state = LifecycleState.READY
            except Exception:
                self._release = None
                self._state = LifecycleState.FAILED
                raise

    def generate(self, request: ChatCompletionRequest) -> GenerationResult:
        if not self.ready:
            raise ServiceNotReadyError("release is not ready")
        self._validate_request(request)
        if not self._admission.acquire(blocking=False):
            raise OverloadedError("bounded concurrency limit reached")
        try:
            prompt_tokens = self._backend.count_input_tokens(request)
            if prompt_tokens > request.max_input_tokens or prompt_tokens > self._max_input_tokens:
                raise RequestValidationError("messages exceed the bounded input token limit")
            return self._backend.generate(request)
        finally:
            self._admission.release()

    def _validate_request(self, request: ChatCompletionRequest) -> None:
        if request.model != self._model_alias:
            raise RequestValidationError("requested model alias is unavailable")
        if request.max_input_tokens > self._max_input_tokens:
            raise RequestValidationError("max_input_tokens exceeds the service limit")
        if request.max_tokens > self._max_output_tokens:
            raise RequestValidationError("max_tokens exceeds the service limit")
        if request.retrieval is not None and request.retrieval.enabled:
            raise RequestValidationError("retrieval must be performed by the RAG orchestrator")
