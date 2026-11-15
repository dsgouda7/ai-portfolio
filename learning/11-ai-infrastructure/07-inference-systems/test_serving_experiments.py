"""HTTP/protocol fixtures, not a GPU model or performance benchmark."""

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import Mock, patch

import serving_experiments as lab


@dataclass
class FixtureState:
    completed: int = 0
    generated: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_request: dict = field(default_factory=dict)


def fixture_handler(state: FixtureState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, body: bytes, status: int = 200, content_type: str = "application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            with state.lock:
                completed = state.completed
                generated = state.generated
            metrics = "\n".join([
                f'{lab.SUCCESS}{{finished_reason="length",model_name="fixture"}} {completed}',
                f"{lab.PREEMPTIONS} 0", f"{lab.RUNNING} 0", f"{lab.WAITING} 0",
                f"{lab.GENERATED} {generated}",
                f"{lab.CACHE} 0.25",
                f"{lab.PREFIX_QUERIES} {completed * 4}",
                f"{lab.PREFIX_HITS} {max(0, completed - 1) * 4}",
            ])
            self.send(metrics.encode(), content_type="text/plain")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with state.lock:
                state.last_request = body
            if self.path == "/tokenize":
                self.send(json.dumps({"count": 4, "tokens": [1, 2, 3, 4]}).encode())
                return
            prompt = body["prompt"]
            if prompt[0] == 95:
                self.send(json.dumps({"error": {
                    "message": "Unknown model identifier", "type": "BadRequestError", "code": 400,
                }}).encode(), status=400)
                return
            if len(prompt) + body["max_tokens"] > 64:
                self.send(json.dumps({
                    "error": {
                        "message": "This model's maximum context length is 64 tokens.",
                        "type": "BadRequestError", "code": 400,
                    },
                }).encode(), status=400)
                return
            events = [
                {"choices": [{"text": ""}], "usage": None},
                {"choices": [{"text": "hello "}]},
                {"choices": [{"text": "world"}]},
            ]
            if prompt[0] == 97:
                events = [{"error": {"message": "fixture stream failure"}}]
            elif prompt[0] != 98:
                events.append({
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(prompt),
                        "completion_tokens": body["max_tokens"] + (prompt[0] == 96),
                    },
                })
            data = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            if prompt[0] != 99:
                data += "data: [DONE]\n\n"
            with state.lock:
                state.completed += 1
                state.generated += body["max_tokens"]
            self.send(data.encode(), content_type="text/event-stream")

    return Handler


class ServingClientTests(unittest.TestCase):
    def setUp(self):
        self.state = FixtureState()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), fixture_handler(self.state))
        self.worker = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True,
        )
        self.worker.start()
        self.client = lab.ServingClient(f"http://127.0.0.1:{self.server.server_port}", timeout_s=3)

    def tearDown(self):
        self.server.shutdown()
        self.worker.join(timeout=3)
        self.server.server_close()

    def test_streamed_completion_uses_final_usage_and_ignores_empty_first_chunks(self):
        result = self.client.complete([1, 2, 3, 4], output_tokens=2)
        self.assertEqual(result.text, "hello world")
        self.assertEqual((result.prompt_tokens, result.output_tokens), (4, 2))
        self.assertGreaterEqual(result.first_text_ms, 0)
        self.assertGreaterEqual(result.latency_ms, result.first_text_ms)
        self.assertEqual(self.state.last_request["stream_options"], {"include_usage": True})
        self.assertTrue(self.state.last_request["ignore_eos"])

    def test_partial_or_malformed_success_streams_are_errors(self):
        for marker, message in (
            (99, "without \\[DONE\\]"), (98, "final token usage"),
            (97, "fixture stream failure"), (96, "controlled workload"),
        ):
            with self.subTest(marker=marker), self.assertRaisesRegex(RuntimeError, message):
                self.client.complete([marker], output_tokens=2)

    def test_context_limit_experiment_accepts_only_the_expected_rejection(self):
        reason = lab.context_limit_rejection(self.client, [1] * 64, output_tokens=2)
        self.assertIn("maximum context length", reason)
        with self.assertRaisesRegex(RuntimeError, "context"):
            lab.context_limit_rejection(self.client, [95], output_tokens=2)

    def test_batch_and_prefix_results_retain_counts_from_the_fixture(self):
        token_ids = self.client.tokenize(lab.PUBLIC_TEXT)
        prompts = [lab.repeat_tokens(token_ids, 8, offset=index) for index in range(4)]
        result = lab.run_batch(self.client, prompts, output_tokens=2, concurrency=2)
        self.assertEqual(result.requests, 4)
        self.assertEqual(result.output_tokens, 8)
        self.assertEqual(len(result.completions), 4)
        self.assertEqual(result.sampled_max_cache_fraction, 0.25)
        rows = lab.repeated_prefix(self.client, token_ids, repeats=2)
        self.assertTrue(all(row.generated_tokens == 32 for row in rows))
        self.assertTrue(all(row.engine_generated_tokens == 32 for row in rows))
        self.assertGreater(rows[-1].prefix_hit_tokens, 0)


class ExperimentContractTests(unittest.TestCase):
    def test_request_usage_cannot_stand_in_for_native_generation_work(self):
        samples = {
            lab.SUCCESS: [0.0], lab.GENERATED: [0.0], lab.PREEMPTIONS: [0.0],
            lab.RUNNING: [0.0], lab.WAITING: [0.0], lab.CACHE: [0.0],
        }
        before = lab.Metrics(samples)
        after = lab.Metrics(dict(samples, **{lab.SUCCESS: [1.0], lab.GENERATED: [1.0]}))
        client = Mock(spec=lab.ServingClient)
        client.metrics.return_value = before
        client.complete.return_value = lab.Completion("fixture", 1, 2, 0.0, 0.1, 0.2)
        client.settled_metrics.return_value = after
        with self.assertRaisesRegex(RuntimeError, "generation counters"):
            lab.run_batch(client, [[1]], output_tokens=2, concurrency=1)

    def test_notebook_root_startup_rejects_unsupported_host_before_writing_results(self):
        notebook = Path(__file__).with_name("real-llm-serving-lab.ipynb").resolve()
        script = f"""
import json
from pathlib import Path
from unittest.mock import patch
book = json.loads(Path({str(notebook)!r}).read_bytes())
code = [cell["source"] for cell in book["cells"] if cell["cell_type"] == "code"]
namespace = {{"__name__": "__main__"}}
exec(compile("".join(code[0]), "imports", "exec"), namespace)
with patch.object(namespace["lab"].sys, "platform", "win32"):
    try:
        exec(compile("".join(code[1]), "preflight", "exec"), namespace)
    except RuntimeError as error:
        assert "Linux or WSL" in str(error), error
    else:
        raise AssertionError("A CPU-only host must not manufacture GPU results.")
assert "RUN_DIR" not in namespace
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=notebook.parents[3],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unsettled_or_extra_request_counters_fail_instead_of_becoming_zero(self):
        client = lab.ServingClient("http://127.0.0.1:8080")
        for values, message in (({}, "did not settle"), ({lab.SUCCESS: [2]}, "Unexpected requests")):
            with self.subTest(values=values), \
                    patch.object(client, "metrics", return_value=lab.Metrics(values)), \
                    self.assertRaisesRegex(RuntimeError, message):
                client.settled_metrics(0, 1, timeout_s=0)

    def test_metrics_preserve_labels_and_missing_values_are_not_zero(self):
        before = lab.parse_metrics(
            '# comment\nvllm:example_total{kind="one"} 2\n'
            'vllm:example_total{kind="two"} 3\nempty_quantile NaN\n'
        )
        after = lab.parse_metrics("vllm:example_total 9\n")
        self.assertEqual(lab.counter_delta(before, after, "vllm:example_total"), 4)
        with self.assertRaisesRegex(RuntimeError, "Missing metric"):
            before.total(lab.CACHE)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            lab.counter_delta(after, before, "vllm:example_total")

    def test_native_cache_budget_and_prefix_switch_are_independent(self):
        baseline = lab.server_command(lab.ServerConfig(kv_cache_mib=64), 8080)
        prefix = lab.server_command(
            lab.ServerConfig(kv_cache_mib=64, prefix_caching=True), 8080,
        )
        self.assertIn("--no-enable-prefix-caching", baseline)
        self.assertIn("--enable-prefix-caching", prefix)
        index = baseline.index("--kv-cache-memory-bytes") + 1
        self.assertEqual(baseline[index], str(64 * 1024**2))
        self.assertEqual(prefix[index], baseline[index])
        self.assertEqual(baseline[baseline.index("--host") + 1], "127.0.0.1")

    def test_invalid_load_and_remote_targets_fail_explicitly(self):
        with self.assertRaises(ValueError):
            lab.ServingClient("https://example.com")
        with self.assertRaises(ValueError):
            lab.ServerConfig(kv_cache_mib=0)
        with self.assertRaises(ValueError):
            lab.repeat_tokens([], 10)
        with patch.object(lab.sys, "platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "Linux or WSL"):
                lab.require_gpu_environment()

    def test_failed_enter_cleans_up_its_own_server_even_for_protocol_errors(self):
        process = Mock(pid=12345)
        process.poll.side_effect = lambda: 0 if process.wait.called else None
        process.wait.return_value = 0
        with TemporaryDirectory() as temporary, \
                patch.object(lab, "require_gpu_environment", return_value={}), \
                patch.object(lab.subprocess, "Popen", return_value=process), \
                patch.object(lab.VllmServer, "_wait_ready", side_effect=ValueError("bad protocol")), \
                patch.object(lab.os, "killpg", create=True) as kill_group:
            server = lab.VllmServer(lab.ServerConfig(), Path(temporary) / "owned")
            try:
                with self.assertRaisesRegex(ValueError, "bad protocol"):
                    server.__enter__()
                kill_group.assert_called_once_with(12345, signal.SIGTERM)
                self.assertTrue(server.log.closed)
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
