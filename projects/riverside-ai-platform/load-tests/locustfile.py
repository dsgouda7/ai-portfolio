"""Staged streaming load test for the Riverside chat completions API."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Any

from azure.identity import ManagedIdentityCredential
from locust import HttpUser, LoadTestShape, between, task

STAGE_NAMES = ("warm", "steady", "target", "overload", "recovery")
CURRENT_STAGE = "warm"


def _load_json_file(environment_name: str, default_name: str) -> Any:
    file_name = os.environ.get(environment_name, default_name)
    return json.loads(Path(file_name).read_text(encoding="utf-8"))


def _load_stages() -> list[dict[str, int | str]]:
    stages = _load_json_file("RIVERSIDE_STAGE_CONFIG", "stages.json")
    if not isinstance(stages, list) or [stage.get("name") for stage in stages] != list(STAGE_NAMES):
        raise ValueError(f"load stages must be ordered as {STAGE_NAMES}")
    for stage in stages:
        for field in ("duration_seconds", "users", "spawn_rate"):
            if not isinstance(stage.get(field), int) or stage[field] <= 0:
                raise ValueError(f"stage {stage.get('name')} requires a positive integer {field}")
    return stages


def _load_requests() -> list[dict[str, Any]]:
    file_name = os.environ.get("RIVERSIDE_REQUESTS_FILE", "synthetic-requests.jsonl")
    requests = [json.loads(line) for line in Path(file_name).read_text(encoding="utf-8").splitlines() if line]
    if not requests:
        raise ValueError("the synthetic request fixture must not be empty")
    return requests


STAGES = _load_stages()
SYNTHETIC_REQUESTS = _load_requests()


def _metric_name(metric: str) -> str:
    return f"chat.completions.{metric}.{CURRENT_STAGE}"


class RiversideUser(HttpUser):
    """A user that consumes the complete SSE stream without retaining content."""

    host = os.environ.get("RIVERSIDE_TARGET_HOST")
    if not host:
        raise RuntimeError("RIVERSIDE_TARGET_HOST must be supplied by the materialized load test")
    wait_time = between(0.25, 1.0)

    def on_start(self) -> None:
        token_scope = os.environ.get("RIVERSIDE_TOKEN_SCOPE")
        if not token_scope or not token_scope.endswith("/.default"):
            raise RuntimeError("RIVERSIDE_TOKEN_SCOPE must be an explicit /.default scope")
        client_id = os.environ.get("RIVERSIDE_MANAGED_IDENTITY_CLIENT_ID") or None
        credential = ManagedIdentityCredential(client_id=client_id)
        token = credential.get_token(token_scope)
        self.headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        self.headers["Authorization"] = f"Bearer {token.token}"

    @task
    def stream_completion(self) -> None:
        payload = dict(random.choice(SYNTHETIC_REQUESTS))
        payload["stream"] = True
        started = perf_counter()
        first_token_at: float | None = None
        output_tokens: int | None = None

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=self.headers,
            name=_metric_name("total"),
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code >= 400:
                response.failure(f"bounded_http_{response.status_code}")
                return
            try:
                for raw_line in response.iter_lines(decode_unicode=True):
                    if isinstance(raw_line, bytes):
                        raw_line = raw_line.decode("utf-8")
                    if not raw_line or not raw_line.startswith("data:"):
                        continue
                    data = raw_line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    choices = event.get("choices", [])
                    content = choices[0].get("delta", {}).get("content") if choices else None
                    if content and first_token_at is None:
                        first_token_at = perf_counter()
                    usage = event.get("usage")
                    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                        output_tokens = usage["completion_tokens"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                response.failure(f"invalid_stream_contract:{type(exc).__name__}")
                return

            ended = perf_counter()
            if first_token_at is None:
                response.failure("stream_missing_output_token")
                return
            if output_tokens is None:
                response.failure("stream_missing_usage")
                return

            ttft_ms = (first_token_at - started) * 1000
            self.environment.events.request.fire(
                request_type="TELEMETRY",
                name=_metric_name("ttft"),
                response_time=ttft_ms,
                response_length=0,
                exception=None,
            )
            if output_tokens > 1:
                tpot_ms = ((ended - first_token_at) * 1000) / (output_tokens - 1)
                self.environment.events.request.fire(
                    request_type="TELEMETRY",
                    name=_metric_name("tpot"),
                    response_time=tpot_ms,
                    response_length=0,
                    exception=None,
                )
            response.success()


class RiversideLoadShape(LoadTestShape):
    """Warm, steady, target, overload, then verify bounded recovery."""

    def tick(self) -> tuple[int, float] | None:
        global CURRENT_STAGE

        elapsed = self.get_run_time()
        stage_end = 0
        engine_instances = int(os.environ.get("RIVERSIDE_ENGINE_INSTANCES", "1"))
        if engine_instances < 1:
            raise ValueError("RIVERSIDE_ENGINE_INSTANCES must be positive")
        for stage in STAGES:
            stage_end += int(stage["duration_seconds"])
            if elapsed < stage_end:
                CURRENT_STAGE = str(stage["name"])
                users_per_engine = math.ceil(int(stage["users"]) / engine_instances)
                spawn_per_engine = max(1.0, int(stage["spawn_rate"]) / engine_instances)
                return users_per_engine, spawn_per_engine
        return None
