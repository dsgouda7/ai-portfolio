"""Local, Azure-shaped serving mechanisms for the Riverside tutorial.

This module uses only local HTTP and deterministic synthetic model work. It does
not emulate Azure control planes and the Azure adapter deliberately refuses to
send network traffic.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


JsonObject = dict[str, Any]


class ArtifactValidationError(ValueError):
    """Raised before readiness when a release artifact violates its contract."""


class BackendFailure(RuntimeError):
    """Raised for a retryable synthetic backend failure."""


class DeadlineExceeded(TimeoutError):
    """Raised when the request's absolute deadline is exhausted."""


class CloudCallBlocked(RuntimeError):
    """Raised when tutorial code attempts a production network call."""


def load_json(path: str | Path) -> JsonObject:
    """Load one JSON object from disk."""
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def validate_contract(
    instance: Mapping[str, Any],
    schema_path: str | Path,
    contract_dir: str | Path,
) -> None:
    """Validate against the frozen local v1 schema registry."""
    registry = Registry()
    for candidate in Path(contract_dir).glob("*.schema.json"):
        schema = load_json(candidate)
        registry = registry.with_resource(
            schema["$id"], Resource.from_contents(schema)
        )

    root_schema = load_json(schema_path)
    validator = Draft202012Validator(
        root_schema,
        registry=registry,
        format_checker=FormatChecker(),
    )
    errors = sorted(validator.iter_errors(dict(instance)), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(
            f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors[:5]
        )
        raise ArtifactValidationError(detail)


def validate_release_invariants(manifest: Mapping[str, Any]) -> None:
    """Enforce cross-field release invariants JSON Schema cannot compare."""
    runtime = manifest["serving_runtime"]
    if manifest["model_profile"] not in runtime["compatible_model_profiles"]:
        raise ArtifactValidationError("model_profile is incompatible with serving_runtime")
    if manifest["precision"] not in runtime["supported_precisions"]:
        raise ArtifactValidationError("precision is unsupported by serving_runtime")
    if manifest["version"] == "latest":
        raise ArtifactValidationError("release version must be immutable")


def stable_request_key(payload: Mapping[str, Any], release_id: str) -> str:
    """Hash a content-free canonical request representation for deduplication."""
    encoded = json.dumps(
        {"release_id": release_id, "request": payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def approximate_tokens(text: str) -> int:
    """Use a transparent local substitute for tokenizer accounting."""
    return max(1, len(text.split()))


def percentile(values: Iterable[float], q: float) -> float:
    """Return a linearly interpolated percentile without external packages."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class BackendResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    tpot_ms: float


class SyntheticBackend:
    """Deterministic latency and failure behavior; never generates manuscript text."""

    def __init__(
        self,
        name: str,
        *,
        first_token_ms: float = 18.0,
        tpot_ms: float = 1.5,
        completion_tokens: int = 12,
        fail_first: int = 0,
    ) -> None:
        self.name = name
        self.first_token_ms = first_token_ms
        self.tpot_ms = tpot_ms
        self.completion_tokens = completion_tokens
        self.fail_first = fail_first
        self.call_count = 0
        self._lock = threading.Lock()

    def warm(self) -> None:
        time.sleep(self.first_token_ms / 1000.0)

    def generate(
        self,
        payload: Mapping[str, Any],
        *,
        deadline_at: float,
    ) -> BackendResult:
        with self._lock:
            self.call_count += 1
            call_number = self.call_count

        if call_number <= self.fail_first:
            raise BackendFailure(f"{self.name} synthetic failure {call_number}")

        if time.perf_counter() >= deadline_at:
            raise DeadlineExceeded("deadline expired before backend work")

        ttft_seconds = self.first_token_ms / 1000.0
        time.sleep(ttft_seconds)
        if time.perf_counter() >= deadline_at:
            raise DeadlineExceeded("deadline expired before first token")

        requested = int(payload.get("max_tokens", self.completion_tokens))
        output_tokens = max(1, min(self.completion_tokens, requested))
        decode_seconds = max(0, output_tokens - 1) * self.tpot_ms / 1000.0
        if time.perf_counter() + decode_seconds > deadline_at:
            raise DeadlineExceeded("deadline would expire during decode")
        time.sleep(decode_seconds)

        prompt_text = " ".join(
            str(message.get("content", "")) for message in payload.get("messages", [])
        )
        return BackendResult(
            content="Riverside local substitute completed the contract-shaped request.",
            prompt_tokens=approximate_tokens(prompt_text),
            completion_tokens=output_tokens,
            ttft_ms=self.first_token_ms,
            tpot_ms=self.tpot_ms,
        )


@dataclass(frozen=True)
class ServicePolicy:
    max_concurrency: int = 2
    max_queue_ms: int = 20
    request_deadline_ms: int = 250
    max_retries: int = 1
    circuit_failure_threshold: int = 2
    circuit_cooldown_ms: int = 250
    enforce_readiness: bool = True
    cache_enabled: bool = True
    singleflight_enabled: bool = True
    account_tokens: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ServicePolicy":
        policy = config["service_policy"]
        return cls(**{field_name: policy[field_name] for field_name in cls.__dataclass_fields__})


@dataclass
class RequestMetric:
    status: str
    total_ms: float
    queue_ms: float
    ttft_ms: float
    tpot_ms: float
    prompt_tokens: int
    completion_tokens: int
    cache_result: str
    error_code: str
    retry_count: int
    release_id: str
    trace_id: str

    def metric_attributes(self, deployment: Mapping[str, Any]) -> JsonObject:
        """Return bounded labels only; trace_id remains trace context."""
        return {
            "service.name": "riverside-model-endpoint",
            "deployment.environment": deployment["environment"],
            "model.release_id": self.release_id,
            "model.alias": deployment["model_alias"],
            "deployment.name": deployment["deployment_name"],
            "cloud.region": deployment["region"],
            "http.route": "/v1/chat/completions",
            "outcome.status": self.status,
            "cache.result": self.cache_result,
            "gen_ai.usage.prompt_tokens_bucket": token_bucket(self.prompt_tokens),
            "gen_ai.usage.output_tokens_bucket": token_bucket(self.completion_tokens),
            "tenant.tier": "internal",
            "error.code": self.error_code,
            "retrieval.top_k_bucket": "none",
        }


def token_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 128:
        return "1-128"
    if count <= 512:
        return "129-512"
    if count <= 2048:
        return "513-2048"
    return "2049-8192"


class CircuitBreaker:
    """Small closed/open/half-open state machine for one backend."""

    def __init__(self, threshold: int, cooldown_ms: int) -> None:
        self.threshold = threshold
        self.cooldown_seconds = cooldown_ms / 1000.0
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            if time.perf_counter() - self.opened_at >= self.cooldown_seconds:
                self.opened_at = None
                self.failures = 0
                return True
            return False

    def success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = None

    def failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = time.perf_counter()

    @property
    def state(self) -> str:
        return "open" if self.opened_at is not None else "closed"


class RiversideService:
    """Contract-shaped service with explicit lifecycle and finite capacity."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        backend: SyntheticBackend,
        policy: ServicePolicy,
        *,
        fallback_backend: SyntheticBackend | None = None,
    ) -> None:
        self.manifest = deepcopy(dict(manifest))
        self.backend = backend
        self.fallback_backend = fallback_backend
        self.policy = policy
        self.validated = False
        self.warmed = False
        self.metrics: list[RequestMetric] = []
        self._metrics_lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(policy.max_concurrency)
        self._cache: dict[str, JsonObject] = {}
        self._cache_lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._breaker = CircuitBreaker(
            policy.circuit_failure_threshold, policy.circuit_cooldown_ms
        )

    @property
    def ready(self) -> bool:
        return self.validated and self.warmed

    @property
    def circuit_state(self) -> str:
        return self._breaker.state

    def validate_and_warm(
        self,
        schema_path: str | Path,
        contract_dir: str | Path,
    ) -> None:
        validate_contract(self.manifest, schema_path, contract_dir)
        validate_release_invariants(self.manifest)
        self.validated = True
        self.backend.warm()
        self.warmed = True

    def _deployment_metadata(self) -> JsonObject:
        release_id = self.manifest["release_id"]
        return {
            "contract_version": "1.0.0",
            "kind": "deployment_metadata",
            "environment": "dev",
            "release_id": release_id,
            "model_alias": "riverside-editor",
            "deployment_name": "riverside-dev-blue",
            "deployment_slot": "blue",
            "region": "eastus2",
            "runtime": "riverside-local-substitute",
            "runtime_version": "1.0.0",
            "index_version": "1.0.0",
            "deployed_at": "2026-08-05T12:00:00Z",
            "source_commit": self.manifest["source_commit"],
        }

    def _error(
        self,
        code: str,
        error_type: str,
        message: str,
        trace_id: str,
        *,
        retryable: bool,
        retry_after_seconds: int | None = None,
    ) -> JsonObject:
        error: JsonObject = {
            "code": code,
            "message": message,
            "type": error_type,
            "param": None,
            "retryable": retryable,
        }
        if retry_after_seconds is not None:
            error["retry_after_seconds"] = retry_after_seconds
        return {
            "error": error,
            "trace": {"trace_id": trace_id, "retry_count": 0},
            "deployment": self._deployment_metadata(),
        }

    def _record(self, metric: RequestMetric) -> None:
        with self._metrics_lock:
            self.metrics.append(metric)

    def _key_lock(self, key: str) -> threading.Lock:
        with self._cache_lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def handle(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None,
    ) -> tuple[int, JsonObject]:
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        release_id = self.manifest["release_id"]

        if self.policy.enforce_readiness and not self.ready:
            body = self._error(
                "release_unavailable",
                "service_error",
                "The release is not validated and warm.",
                trace_id,
                retryable=True,
            )
            self._record(
                RequestMetric(
                    "error", 0, 0, 0, 0, 0, 0, "bypass",
                    "release_unavailable", 0, release_id, trace_id
                )
            )
            return 503, body

        request_key = idempotency_key or stable_request_key(payload, release_id)
        with self._cache_lock:
            cached = deepcopy(self._cache.get(request_key))
        if self.policy.cache_enabled and cached is not None:
            total_ms = (time.perf_counter() - started) * 1000
            usage = cached["usage"]
            self._record(
                RequestMetric(
                    "success", total_ms, 0, 0, 0,
                    usage["prompt_tokens"], usage["completion_tokens"],
                    "hit", "none", 0, release_id, trace_id
                )
            )
            cached["trace"] = {"trace_id": trace_id, "retry_count": 0}
            return 200, cached

        singleflight = self._key_lock(request_key)
        if self.policy.singleflight_enabled:
            singleflight.acquire()
        try:
            if self.policy.cache_enabled:
                with self._cache_lock:
                    cached = deepcopy(self._cache.get(request_key))
                if cached is not None:
                    total_ms = (time.perf_counter() - started) * 1000
                    usage = cached["usage"]
                    self._record(
                        RequestMetric(
                            "success", total_ms, 0, 0, 0,
                            usage["prompt_tokens"], usage["completion_tokens"],
                            "hit", "none", 0, release_id, trace_id
                        )
                    )
                    cached["trace"] = {"trace_id": trace_id, "retry_count": 0}
                    return 200, cached

            queue_started = time.perf_counter()
            admitted = self._semaphore.acquire(
                timeout=self.policy.max_queue_ms / 1000.0
            )
            queue_ms = (time.perf_counter() - queue_started) * 1000
            if not admitted:
                total_ms = (time.perf_counter() - started) * 1000
                body = self._error(
                    "overloaded",
                    "capacity_error",
                    "The service is at its bounded concurrency limit.",
                    trace_id,
                    retryable=True,
                    retry_after_seconds=1,
                )
                self._record(
                    RequestMetric(
                        "rejected", total_ms, queue_ms, 0, 0, 0, 0,
                        "miss", "overloaded", 0, release_id, trace_id
                    )
                )
                return 429, body

            try:
                deadline_at = started + self.policy.request_deadline_ms / 1000.0
                retry_count = 0
                result: BackendResult | None = None
                last_failure: Exception | None = None
                primary_allowed = self._breaker.allow()
                if primary_allowed:
                    attempts = self.policy.max_retries + 1 if idempotency_key else 1
                    for attempt in range(attempts):
                        retry_count = attempt
                        try:
                            result = self.backend.generate(payload, deadline_at=deadline_at)
                            self._breaker.success()
                            break
                        except BackendFailure as error:
                            last_failure = error
                            self._breaker.failure()
                            if attempt + 1 < attempts and time.perf_counter() < deadline_at:
                                continue
                else:
                    last_failure = BackendFailure("primary circuit is open")

                if result is None and self.fallback_backend is not None:
                    result = self.fallback_backend.generate(payload, deadline_at=deadline_at)

                if result is None:
                    raise last_failure or BackendFailure("backend failed")

                prompt_tokens = result.prompt_tokens if self.policy.account_tokens else 0
                completion_tokens = result.completion_tokens if self.policy.account_tokens else 0
                response: JsonObject = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "riverside-editor",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": result.content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "citations": [],
                    "trace": {"trace_id": trace_id, "retry_count": retry_count},
                    "deployment": self._deployment_metadata(),
                }
                if self.policy.cache_enabled:
                    with self._cache_lock:
                        self._cache[request_key] = deepcopy(response)
                total_ms = (time.perf_counter() - started) * 1000
                self._record(
                    RequestMetric(
                        "success", total_ms, queue_ms, result.ttft_ms,
                        result.tpot_ms, prompt_tokens, completion_tokens,
                        "miss", "none", retry_count, release_id, trace_id
                    )
                )
                return 200, response
            except DeadlineExceeded:
                total_ms = (time.perf_counter() - started) * 1000
                body = self._error(
                    "timeout", "deadline_error", "The request deadline expired.",
                    trace_id, retryable=True
                )
                self._record(
                    RequestMetric(
                        "timeout", total_ms, queue_ms, 0, 0, 0, 0,
                        "miss", "timeout", 0, release_id, trace_id
                    )
                )
                return 504, body
            except BackendFailure:
                total_ms = (time.perf_counter() - started) * 1000
                body = self._error(
                    "backend_failure", "backend_error", "The backend failed safely.",
                    trace_id, retryable=True
                )
                self._record(
                    RequestMetric(
                        "error", total_ms, queue_ms, 0, 0, 0, 0,
                        "miss", "backend_failure", 0, release_id, trace_id
                    )
                )
                return 502, body
            finally:
                self._semaphore.release()
        finally:
            if self.policy.singleflight_enabled and singleflight.locked():
                singleflight.release()


class _ServiceHandler(BaseHTTPRequestHandler):
    service: RiversideService

    def _write(self, status: int, body: Mapping[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self._write(200, {"status": "alive"})
        elif self.path == "/ready":
            status = 200 if self.service.ready else 503
            self._write(status, {"status": "ready" if status == 200 else "not_ready"})
        else:
            self._write(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/chat/completions":
            self._write(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            status, body = self.service.handle(
                payload,
                idempotency_key=self.headers.get("Idempotency-Key"),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            status, body = 400, {"error": "invalid_json"}
        self._write(status, body)

    def log_message(self, format: str, *args: Any) -> None:
        return


@dataclass
class RunningServer:
    server: ThreadingHTTPServer
    thread: threading.Thread
    base_url: str

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def start_local_server(service: RiversideService) -> RunningServer:
    """Start an ephemeral loopback HTTP server for measured notebook requests."""
    handler = type("BoundServiceHandler", (_ServiceHandler,), {"service": service})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return RunningServer(server, thread, f"http://{host}:{port}")


@dataclass(frozen=True)
class AdapterResponse:
    status: int
    body: JsonObject
    elapsed_ms: float


class LocalSandboxAdapter:
    def __init__(self, endpoint_url: str, timeout_seconds: float = 5.0) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def invoke(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> AdapterResponse:
        headers = {"Content-Type": "application/json"}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        request = Request(
            f"{self.endpoint_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read())
                status = response.status
        except HTTPError as error:
            body = json.loads(error.read())
            status = error.code
        return AdapterResponse(status, body, (time.perf_counter() - started) * 1000)


class AzureMLAPIMAdapter:
    """Prepare a production request shape while blocking all network traffic."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = deepcopy(dict(config))

    def request_plan(self, payload: Mapping[str, Any]) -> JsonObject:
        return {
            "url": self.config["endpoint_url"],
            "authentication": self.config["authentication"],
            "headers": {
                "Authorization": "<workload-identity-token supplied at runtime>",
                "Content-Type": "application/json",
            },
            "body_keys": sorted(payload.keys()),
            "network_call": "BLOCKED_IN_TUTORIAL",
        }

    def invoke(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> AdapterResponse:
        del payload, idempotency_key
        raise CloudCallBlocked(
            "Azure ML/APIM calls are unvalidated and blocked in this tutorial."
        )


def build_endpoint_adapter(
    config: Mapping[str, Any],
    *,
    endpoint_override: str | None = None,
) -> LocalSandboxAdapter | AzureMLAPIMAdapter:
    adapter = config["endpoint_adapter"]
    kind = adapter["kind"]
    if kind == "local_sandbox":
        return LocalSandboxAdapter(endpoint_override or adapter["endpoint_url"])
    if kind == "azureml_apim":
        return AzureMLAPIMAdapter(adapter)
    raise ValueError(f"Unsupported endpoint adapter: {kind}")


def run_concurrent_requests(
    adapter: LocalSandboxAdapter,
    payloads: Iterable[Mapping[str, Any]],
    *,
    workers: int,
    shared_idempotency_key: str | None = None,
) -> list[AdapterResponse]:
    """Measure a finite concurrent workload against the local HTTP boundary."""
    responses: list[AdapterResponse] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                adapter.invoke,
                payload,
                idempotency_key=shared_idempotency_key,
            )
            for payload in payloads
        ]
        for future in as_completed(futures):
            responses.append(future.result())
    return responses


def summarize_metrics(metrics: Iterable[RequestMetric]) -> JsonObject:
    rows = list(metrics)
    statuses = Counter(row.status for row in rows)
    return {
        "requests": len(rows),
        "statuses": dict(statuses),
        "p50_total_ms": round(percentile((row.total_ms for row in rows), 0.50), 2),
        "p95_total_ms": round(percentile((row.total_ms for row in rows), 0.95), 2),
        "p95_queue_ms": round(percentile((row.queue_ms for row in rows), 0.95), 2),
        "p95_ttft_ms": round(percentile((row.ttft_ms for row in rows), 0.95), 2),
        "p95_tpot_ms": round(percentile((row.tpot_ms for row in rows), 0.95), 2),
        "successful_output_tokens": sum(
            row.completion_tokens for row in rows if row.status == "success"
        ),
        "cache_hits": sum(row.cache_result == "hit" for row in rows),
        "retries": sum(row.retry_count for row in rows),
    }
