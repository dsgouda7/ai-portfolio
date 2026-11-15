"""GPU-only companion. Run setup-serving.sh in Linux/WSL first."""

# %% serving-imports
import sys
import json
import hashlib
import statistics
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

CHAPTER_DIR = next(
    folder for folder in (
        Path(globals().get("__file__", globals().get("__vsc_ipynb_file__", ""))).resolve().parent,
        Path.cwd(),
        Path.cwd() / "learning" / "11-ai-infrastructure" / "07-inference-systems",
    ) if (folder / "serving_experiments.py").is_file()
)
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))
import serving_experiments as lab

# %% serving-preflight
environment = lab.require_gpu_environment()
RUN_DIR = CHAPTER_DIR / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
RUN_DIR.mkdir(parents=True, exist_ok=False)
BASELINE = lab.ServerConfig(kv_cache_mib=128, prefix_caching=False, max_model_len=2048, max_sequences=4)
print(json.dumps(environment, indent=2))
print("Evidence directory:", RUN_DIR)
print("Native engine settings:", asdict(BASELINE))

# %% serving-http-probe
with lab.VllmServer(BASELINE, RUN_DIR / "smoke") as client:
    print("Actual loopback endpoint:", client.base_url)
    print((RUN_DIR / "smoke" / "command.json").read_text())
    seed_tokens = client.tokenize(lab.PUBLIC_TEXT)
    smoke = lab.run_batch(client, [lab.repeat_tokens(seed_tokens, 128)], output_tokens=32, concurrency=1)

print("Generated text:", smoke.completions[0].text)
print("Input/output tokens:", smoke.completions[0].prompt_tokens, smoke.completions[0].output_tokens)
print("Client first-text milliseconds:", smoke.mean_first_text_ms)

# %% serving-hf-cache
hf_cache = lab.hf_cache_experiment(prompt_tokens=256, output_tokens=32)
print("Same generated token IDs:", hf_cache.outputs_equal)
print("Uncached seconds:", hf_cache.uncached_seconds)
print("Cached seconds:", hf_cache.cached_seconds)
print("Median uncached / cached:", statistics.median(hf_cache.uncached_seconds) / statistics.median(hf_cache.cached_seconds))

# %% serving-cache-bytes
import matplotlib.pyplot as plt

for row in hf_cache.kv_memory:
    print(row.prompt_tokens, "input tokens:", row.actual_kv_tensor_bytes, "actual KV tensor bytes")
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([row.prompt_tokens for row in hf_cache.kv_memory],
        [row.actual_kv_tensor_bytes / 1024**2 for row in hf_cache.kv_memory], "o-")
ax.set(xlabel="Input tokens", ylabel="Actual KV tensor MiB", title="Saved work has a storage cost")
fig.tight_layout()
plt.show()
plt.close(fig)

# %% serving-native-load
workload = [lab.repeat_tokens(seed_tokens, 1536, offset=index) for index in range(12)]
length_results: dict[int, lab.BatchResult] = {}
output_results: dict[int, lab.BatchResult] = {}
load_results: dict[int, lab.BatchResult] = {}
with lab.VllmServer(BASELINE, RUN_DIR / "baseline") as client:
    over_limit = lab.repeat_tokens(seed_tokens, BASELINE.max_model_len - 16)
    context_error = lab.context_limit_rejection(client, over_limit, output_tokens=64)
    print("Expected context-length rejection:", context_error)
    for length in (256, 768, 1536):
        prompts = [lab.repeat_tokens(seed_tokens, length, offset=index) for index in range(12)]
        length_results[length] = lab.run_batch(client, prompts, output_tokens=128, concurrency=1)
    for output_length in (32, 128, 512):
        prompts = [lab.repeat_tokens(seed_tokens, 768, offset=index) for index in range(12)]
        output_results[output_length] = lab.run_batch(client, prompts, output_tokens=output_length, concurrency=1)
    for concurrency in (1, 2, 4, 8):
        load_results[concurrency] = lab.run_batch(client, workload, output_tokens=128, concurrency=concurrency)

for length, row in length_results.items():
    print("Input tokens", length, "first-text ms", round(row.mean_first_text_ms, 2))
for length, row in output_results.items():
    print("Output tokens", length, "request p95 ms", round(row.p95_request_ms, 2),
          "sampled cache fraction", row.sampled_max_cache_fraction)
for concurrency, row in load_results.items():
    print(concurrency, "clients:", round(row.output_tokens_per_s, 2), "output tokens/s;",
          round(row.p95_request_ms, 2), "p95 ms; sampled waiting", row.sampled_max_waiting)

# %% serving-load-plot
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(list(load_results), [row.output_tokens_per_s for row in load_results.values()], "o-")
axes[0].set(xlabel="Concurrent clients", ylabel="Completed output tokens/s")
axes[1].plot(list(load_results), [row.p95_request_ms for row in load_results.values()], "o-")
axes[1].set(xlabel="Concurrent clients", ylabel="Observed request p95 (ms)")
fig.suptitle("One model and workload: throughput is not the same as waiting time")
fig.tight_layout()
plt.show()
plt.close(fig)

# %% serving-native-budget
SMALL_CACHE = replace(BASELINE, kv_cache_mib=64)
with lab.VllmServer(SMALL_CACHE, RUN_DIR / "cache-small") as client:
    small_cache = lab.run_batch(client, workload, output_tokens=128, concurrency=4)
large_cache = load_results[4]
for budget, row in ((BASELINE.kv_cache_mib, large_cache), (SMALL_CACHE.kv_cache_mib, small_cache)):
    print(budget, "MiB:", round(row.output_tokens_per_s, 2), "output tokens/s;",
          "sampled cache fraction", row.sampled_max_cache_fraction,
          "sampled waiting", row.sampled_max_waiting, "preemptions", row.preemptions)

# %% serving-prefix-reuse
PREFIX_CACHE = replace(BASELINE, prefix_caching=True)
prefix_prompt = lab.repeat_tokens(seed_tokens, 1024)
with lab.VllmServer(PREFIX_CACHE, RUN_DIR / "prefix-on") as client:
    prefix_results = lab.repeated_prefix(client, prefix_prompt, repeats=4)
for row in prefix_results:
    print("Request", row.request, "first-text ms", round(row.first_text_ms, 2),
          "prefix queried/hit tokens", row.prefix_queried_tokens, row.prefix_hit_tokens,
          "usage/native output tokens", row.generated_tokens, row.engine_generated_tokens)

# %% serving-save-evidence
report = {
    "evidence_class": "Local measured",
    "scope": "One public pretrained checkpoint, named GPU, controlled local workload; not production capacity.",
    "environment": environment,
    "helper_sha256": hashlib.sha256(Path(lab.__file__).read_bytes()).hexdigest(),
    "lesson_sha256": hashlib.sha256((CHAPTER_DIR / "real-llm-serving-lab.py").read_bytes()).hexdigest(),
    "profiles": {"baseline": asdict(BASELINE), "small_cache": asdict(SMALL_CACHE), "prefix": asdict(PREFIX_CACHE)},
    "workload_token_ids": workload,
    "seed_token_ids": seed_tokens,
    "smoke": asdict(smoke),
    "hf_cache": asdict(hf_cache),
    "by_input_length": {str(key): asdict(value) for key, value in length_results.items()},
    "by_output_length": {str(key): asdict(value) for key, value in output_results.items()},
    "by_concurrency": {str(key): asdict(value) for key, value in load_results.items()},
    "smaller_native_pool": asdict(small_cache),
    "context_limit_rejection": context_error,
    "prefix_reuse": [asdict(row) for row in prefix_results],
}
serialized = json.dumps(report, indent=2, allow_nan=False)
with (RUN_DIR / "experiment-results.json").open("x", encoding="utf-8") as destination:
    destination.write(serialized + "\n")
print("Retained evidence:", RUN_DIR)
