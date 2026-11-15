"""Real GPU model experiments; the HTTP client itself uses only the standard library."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
from http.client import HTTPResponse
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Iterable, Mapping, Sequence, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


VLLM_VERSION = "0.11.0"
TRANSFORMERS_VERSION = "4.57.1"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SERVED_MODEL = "serving-lab"
PUBLIC_TEXT = (
    "A serving experiment measures memory, waiting time, and generated tokens. "
    "The same public input is used when comparing configurations. "
)
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
CACHE = "vllm:kv_cache_usage_perc"
SUCCESS = "vllm:request_success_total"
PREEMPTIONS = "vllm:num_preemptions_total"
GENERATED = "vllm:generation_tokens_total"
PREFIX_HITS = "vllm:prefix_cache_hits_total"
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"


def require_gpu_environment() -> dict[str, object]:
    if sys.platform != "linux":
        raise RuntimeError(
            "This lab requires Linux or WSL with an NVIDIA CUDA GPU; "
            "native Windows and CPU runs are not substituted for GPU measurements."
        )
    for package, expected in (("vllm", VLLM_VERSION), ("transformers", TRANSFORMERS_VERSION)):
        try:
            installed = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f"Missing {package}; run setup-serving.sh in Linux/WSL.") from error
        if installed != expected:
            raise RuntimeError(f"Expected {package}=={expected}, found {installed}.")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a supported NVIDIA GPU; there is no CPU fallback.")
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": properties.name,
        "gpu_total_bytes": properties.total_memory,
        "driver": driver,
        "vllm": VLLM_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
    }


@dataclass(frozen=True)
class ServerConfig:
    kv_cache_mib: int = 128
    prefix_caching: bool = False
    max_model_len: int = 2048
    max_sequences: int = 4

    def __post_init__(self) -> None:
        if min(self.kv_cache_mib, self.max_model_len, self.max_sequences) <= 0:
            raise ValueError("Cache budget, context length, and sequence limit must be positive.")
        if self.max_model_len > 8192:
            raise ValueError("The pinned model supports at most 8192 positions.")


def server_command(config: ServerConfig, port: int) -> list[str]:
    if not 0 < port < 65536:
        raise ValueError("The server port must be between 1 and 65535.")
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID, "--revision", MODEL_REVISION,
        "--tokenizer-revision", MODEL_REVISION,
        "--served-model-name", SERVED_MODEL,
        "--host", "127.0.0.1", "--port", str(port),
        "--dtype", "float16", "--load-format", "safetensors",
        "--generation-config", "vllm", "--seed", "31",
        "--enforce-eager", "--enable-chunked-prefill",
        "--max-num-batched-tokens", str(config.max_model_len),
        "--max-model-len", str(config.max_model_len),
        "--max-num-seqs", str(config.max_sequences),
        "--gpu-memory-utilization", "0.5",
        "--kv-cache-memory-bytes", str(config.kv_cache_mib * 1024**2),
        "--enable-prompt-tokens-details",
        "--enable-prefix-caching" if config.prefix_caching else "--no-enable-prefix-caching",
    ]


class ServingHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"Serving HTTP {status}: {body}")


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"Response is missing a nonnegative integer {key!r}.")
    return value


def sse_events(lines: Iterable[bytes]) -> Iterable[dict[str, object]]:
    data: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if not data:
                continue
            payload = "\n".join(data)
            data.clear()
            if payload == "[DONE]":
                return
            event = json.loads(payload)
            if not isinstance(event, dict):
                raise RuntimeError("Expected a JSON object in the completion stream.")
            if "error" in event:
                raise RuntimeError(f"Completion stream failed: {event['error']}")
            yield event
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    raise RuntimeError("Completion stream ended without [DONE]; partial text is not a success.")


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    output_tokens: int
    started_s: float
    first_text_s: float
    finished_s: float

    @property
    def first_text_ms(self) -> float:
        return 1000 * (self.first_text_s - self.started_s)

    @property
    def latency_ms(self) -> float:
        return 1000 * (self.finished_s - self.started_s)


@dataclass(frozen=True)
class Metrics:
    values: dict[str, list[float]]

    def samples(self, name: str) -> list[float]:
        if name not in self.values:
            raise RuntimeError(f"Missing metric {name!r}; check the pinned vLLM V1 runtime.")
        return self.values[name]

    def total(self, name: str) -> float:
        return sum(self.samples(name))

    def peak(self, name: str) -> float:
        return max(self.samples(name))


def parse_metrics(text: str) -> Metrics:
    values: dict[str, list[float]] = {}
    pattern = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+(\S+)(?:\s+\d+)?$")
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise RuntimeError(f"Invalid Prometheus sample: {line[:160]}")
        name, raw = match.groups()
        number = float(raw)
        if not math.isfinite(number):
            continue  # Unused empty-summary quantiles may be NaN; required metrics still fail explicitly.
        values.setdefault(name, []).append(number)
    return Metrics(values)


def counter_delta(before: Metrics, after: Metrics, name: str) -> float:
    difference = after.total(name) - before.total(name)
    if difference < 0:
        raise RuntimeError(f"Counter {name!r} reset during the experiment.")
    return difference


class ServingClient:
    def __init__(self, base_url: str, timeout_s: float = 180):
        parts = urlsplit(base_url)
        if (
            parts.scheme != "http" or parts.hostname != "127.0.0.1"
            or parts.username or parts.password or parts.query or parts.fragment
            or parts.path not in ("", "/")
        ):
            raise ValueError("The lab client only connects directly to a loopback HTTP server.")
        if timeout_s <= 0:
            raise ValueError("Request timeout must be positive.")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _open(self, path: str, payload: dict[str, object] | None = None) -> HTTPResponse:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = Request(
            self.base_url + path, data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            return urlopen(request, timeout=self.timeout_s)
        except HTTPError as error:
            with error:
                raise ServingHTTPError(error.code, error.read().decode("utf-8")) from error

    def json(self, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        with self._open(path, payload) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise RuntimeError("Expected a JSON response object.")
        return result

    def metrics(self) -> Metrics:
        with self._open("/metrics") as response:
            return parse_metrics(response.read().decode("utf-8"))

    def tokenize(self, text: str) -> list[int]:
        response = self.json("/tokenize", {
            "model": SERVED_MODEL, "prompt": text, "add_special_tokens": False,
        })
        tokens = response.get("tokens")
        if (
            not isinstance(tokens, list) or not tokens
            or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens)
            or _integer(response, "count") != len(tokens)
        ):
            raise RuntimeError("Tokenizer returned an invalid token list/count.")
        return tokens

    def complete(self, prompt: Sequence[int], output_tokens: int = 128) -> Completion:
        if not prompt or output_tokens <= 0:
            raise ValueError("Provide a nonempty tokenized prompt and positive output length.")
        started = time.perf_counter()
        first_text: float | None = None
        text: list[str] = []
        usage: dict[str, object] | None = None
        with self._open("/v1/completions", {
            "model": SERVED_MODEL, "prompt": list(prompt),
            "max_tokens": output_tokens, "temperature": 0, "seed": 31,
            "ignore_eos": True, "skip_special_tokens": False,
            "stream": True, "stream_options": {"include_usage": True},
        }) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise RuntimeError("Expected a streamed completion, not a buffered JSON response.")
            for event in sse_events(response):
                choices = event.get("choices")
                if not isinstance(choices, list):
                    raise RuntimeError("Stream event is missing its choices list.")
                for choice in choices:
                    if not isinstance(choice, dict) or not isinstance(choice.get("text"), str):
                        raise RuntimeError("Invalid streamed completion choice.")
                    chunk = choice["text"]
                    if chunk:
                        if first_text is None:
                            first_text = time.perf_counter()
                        text.append(chunk)
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage = event_usage
        finished = time.perf_counter()
        if first_text is None or usage is None:
            raise RuntimeError("Completion did not provide generated text and final token usage.")
        prompt_count = _integer(usage, "prompt_tokens")
        output_count = _integer(usage, "completion_tokens")
        if prompt_count != len(prompt) or output_count != output_tokens:
            raise RuntimeError("Actual token counts differ from the controlled workload.")
        return Completion("".join(text), prompt_count, output_count, started, first_text, finished)

    def settled_metrics(self, previous_successes: float, count: int, timeout_s: float = 15) -> Metrics:
        deadline = time.monotonic() + timeout_s
        while True:
            snapshot = self.metrics()
            successes = snapshot.values.get(SUCCESS)
            if successes is not None:
                completed = sum(successes) - previous_successes
                if completed == count:
                    return snapshot
                if completed > count:
                    raise RuntimeError("Unexpected requests reached this private experiment server.")
            if time.monotonic() >= deadline:
                raise RuntimeError("Server completion metrics did not settle; no zero-valued fallback used.")
            time.sleep(0.1)


class VllmServer:
    def __init__(self, config: ServerConfig, directory: Path, startup_timeout_s: float = 600):
        self.config = config
        self.directory = directory
        self.startup_timeout_s = startup_timeout_s
        self.process: subprocess.Popen[bytes] | None = None
        self.log: TextIO | None = None
        self.client: ServingClient | None = None

    def __enter__(self) -> ServingClient:
        require_gpu_environment()
        self.directory.mkdir(parents=True, exist_ok=False)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        command = server_command(self.config, port)
        (self.directory / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        self.log = (self.directory / "server.log").open("w", encoding="utf-8")
        environment = dict(os.environ, VLLM_USE_V1="1")
        ready = False
        try:
            self.process = subprocess.Popen(
                command, stdout=self.log, stderr=subprocess.STDOUT,
                env=environment, start_new_session=True,
            )
            self.client = ServingClient(f"http://127.0.0.1:{port}")
            self._wait_ready()
            ready = True
            return self.client
        finally:
            if not ready:
                self.close()

    def _wait_ready(self) -> None:
        assert self.process is not None and self.client is not None
        deadline = time.monotonic() + self.startup_timeout_s
        last_error = "not ready"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup; inspect {self.directory / 'server.log'}.")
            try:
                self.client.json("/version")
                break
            except (ServingHTTPError, URLError, TimeoutError) as error:
                last_error = str(error)
                time.sleep(0.5)
        else:
            raise RuntimeError(f"vLLM startup timed out: {last_error}; inspect {self.directory / 'server.log'}.")
        if self.client.json("/version").get("version") != VLLM_VERSION:
            raise RuntimeError("The endpoint version does not match the pinned runtime.")
        models = self.client.json("/v1/models").get("data")
        if not isinstance(models, list) or not any(
            isinstance(model, dict) and model.get("id") == SERVED_MODEL for model in models
        ):
            raise RuntimeError("The expected model is not ready.")
        seed = self.client.tokenize("Warm up this local model before collecting measurements.")
        self.client.complete(seed, output_tokens=8)
        self.client.settled_metrics(0, 1)

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                self.process.wait(timeout=15)
            else:
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=15)
        if self.log is not None:
            self.log.close()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def repeat_tokens(seed: Sequence[int], length: int, offset: int = 0) -> list[int]:
    if not seed or length <= 0 or offset < 0:
        raise ValueError("Provide seed tokens, positive length, and a nonnegative offset.")
    return [seed[(position + offset) % len(seed)] for position in range(length)]


@dataclass(frozen=True)
class BatchResult:
    requests: int
    concurrency: int
    output_tokens: int
    elapsed_s: float
    output_tokens_per_s: float
    mean_first_text_ms: float
    p95_request_ms: float
    sampled_max_running: float
    sampled_max_waiting: float
    sampled_max_cache_fraction: float
    preemptions: float
    counter_before: dict[str, float]
    counter_after: dict[str, float]
    metric_samples: list[dict[str, float]]
    completions: list[Completion]


def run_batch(
    client: ServingClient, prompts: Sequence[Sequence[int]], output_tokens: int, concurrency: int,
) -> BatchResult:
    if not prompts or not 1 <= concurrency <= 32:
        raise ValueError("Provide at least one request and concurrency between 1 and 32.")
    before = client.metrics()
    observations: list[dict[str, float]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending = [executor.submit(client.complete, prompt, output_tokens) for prompt in prompts]
        while True:
            sample = client.metrics()
            observations.append({
                "sampled_running": sample.total(RUNNING),
                "sampled_waiting": sample.total(WAITING),
                "sampled_cache_fraction": sample.peak(CACHE),
            })
            if all(future.done() for future in pending):
                break
            time.sleep(0.05)
        results = [future.result() for future in pending]
    after = client.settled_metrics(before.total(SUCCESS), len(results))
    generated_tokens = sum(result.output_tokens for result in results)
    if counter_delta(before, after, GENERATED) != generated_tokens:
        raise RuntimeError("Engine generation counters do not match completed request usage.")
    elapsed = max(result.finished_s for result in results) - min(result.started_s for result in results)
    latencies = sorted(result.latency_ms for result in results)
    return BatchResult(
        requests=len(results), concurrency=concurrency,
        output_tokens=generated_tokens,
        elapsed_s=elapsed,
        output_tokens_per_s=generated_tokens / elapsed,
        mean_first_text_ms=statistics.mean(result.first_text_ms for result in results),
        p95_request_ms=latencies[math.ceil(0.95 * len(latencies)) - 1],
        sampled_max_running=max(row["sampled_running"] for row in observations),
        sampled_max_waiting=max(row["sampled_waiting"] for row in observations),
        sampled_max_cache_fraction=max(row["sampled_cache_fraction"] for row in observations),
        preemptions=counter_delta(before, after, PREEMPTIONS),
        counter_before={name: before.total(name) for name in (SUCCESS, PREEMPTIONS, GENERATED)},
        counter_after={name: after.total(name) for name in (SUCCESS, PREEMPTIONS, GENERATED)},
        metric_samples=observations, completions=results,
    )


@dataclass(frozen=True)
class PrefixObservation:
    request: int
    first_text_ms: float
    generated_tokens: int
    engine_generated_tokens: float
    prefix_queried_tokens: float
    prefix_hit_tokens: float


def repeated_prefix(client: ServingClient, prompt: Sequence[int], repeats: int = 4) -> list[PrefixObservation]:
    if repeats < 2:
        raise ValueError("At least two requests are needed to compare cold and warm prefixes.")
    rows: list[PrefixObservation] = []
    warm_hits_observed = False
    for number in range(repeats):
        before = client.metrics()
        result = client.complete(prompt, output_tokens=32)
        after = client.settled_metrics(before.total(SUCCESS), 1)
        hits = counter_delta(before, after, PREFIX_HITS)
        generated = counter_delta(before, after, GENERATED)
        if generated != result.output_tokens:
            raise RuntimeError("Prefix reuse did not retain the expected native generation work.")
        warm_hits_observed |= number > 0 and hits > 0
        rows.append(PrefixObservation(
            request=number + 1, first_text_ms=result.first_text_ms,
            generated_tokens=result.output_tokens, engine_generated_tokens=generated,
            prefix_queried_tokens=counter_delta(before, after, PREFIX_QUERIES),
            prefix_hit_tokens=hits,
        ))
    if not warm_hits_observed:
        raise RuntimeError("No warm-prefix hits were observed; inspect the engine before claiming reuse.")
    return rows


def context_limit_rejection(client: ServingClient, prompt: Sequence[int], output_tokens: int) -> str:
    try:
        client.complete(prompt, output_tokens)
    except ServingHTTPError as error:
        if error.status != 400:
            raise
        response = json.loads(error.body)
        details = response.get("error") if isinstance(response, dict) else None
        if not isinstance(details, dict) or not isinstance(details.get("message"), str):
            raise RuntimeError("The rejection did not identify a context-length error.") from error
        if "maximum context length" not in details["message"].lower():
            raise RuntimeError("The HTTP 400 was not the expected context-length rejection.") from error
        return error.body
    raise RuntimeError("The over-context request was not rejected as expected.")


@dataclass(frozen=True)
class CacheMemory:
    prompt_tokens: int
    actual_kv_tensor_bytes: int


@dataclass(frozen=True)
class HfCacheResult:
    prompt_tokens: int
    output_tokens: int
    uncached_seconds: list[float]
    cached_seconds: list[float]
    outputs_equal: bool
    generated_token_ids: list[int]
    kv_memory: list[CacheMemory]


def hf_cache_experiment(prompt_tokens: int = 256, output_tokens: int = 32) -> HfCacheResult:
    """Compare the same real checkpoint with/without generation KV reuse, then release it."""
    require_gpu_environment()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    seed = tokenizer.encode(PUBLIC_TEXT, add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to("cuda").eval()
    try:
        inputs = torch.tensor([repeat_tokens(seed, prompt_tokens)], device="cuda")
        mask = torch.ones_like(inputs)

        def generate(cached: bool):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    inputs, attention_mask=mask, do_sample=False, use_cache=cached,
                    min_new_tokens=output_tokens, max_new_tokens=output_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            return time.perf_counter() - start, output[0, prompt_tokens:].tolist()

        for cached in (False, True):
            generate(cached)
        timings: dict[bool, list[float]] = {False: [], True: []}
        expected: list[int] | None = None
        for repetition in range(3):
            for cached in ((False, True) if repetition % 2 == 0 else (True, False)):
                elapsed, generated = generate(cached)
                if expected is not None and generated != expected:
                    raise RuntimeError("Cached/uncached greedy outputs differed; do not claim a pure speed comparison.")
                expected = generated
                timings[cached].append(elapsed)
        memory = []
        for length in (64, 256, 1024):
            ids = torch.tensor([repeat_tokens(seed, length)], device="cuda")
            with torch.inference_mode():
                forward = model(ids, attention_mask=torch.ones_like(ids), use_cache=True)
            cache_bytes = sum(
                key.numel() * key.element_size() + value.numel() * value.element_size()
                for key, value in forward.past_key_values
            )
            memory.append(CacheMemory(length, cache_bytes))
            del forward, ids
        assert expected is not None
        return HfCacheResult(
            prompt_tokens, output_tokens, timings[False], timings[True],
            True, expected, memory,
        )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Require the pinned Linux/CUDA environment.")
    arguments = parser.parse_args()
    if arguments.preflight:
        print(json.dumps(require_gpu_environment(), indent=2))
    else:
        parser.print_help()
