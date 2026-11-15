# Inference Systems: From Saved Work to a Real Model Server

Start with [Inference Systems](inference-systems.ipynb) to inspect the mechanisms.
Then use [Real GPU LLM Serving](real-llm-serving-lab.ipynb), with its
[theory companion](real-llm-serving-lab-theory.md), to measure an actual pretrained
model behind a local HTTP endpoint. No fine-tuning, adapter, or private dataset is required.

## What changes between the two notebooks?

| Notebook | What runs | What the evidence supports |
|---|---|---|
| Inference Systems | A small PyTorch model, timing experiments, and explicitly labeled scheduling/memory simulations | How the mechanisms work; not a production capacity forecast |
| Real GPU LLM Serving | A pinned public checkpoint, a cached/uncached Transformers comparison, and real vLLM HTTP requests | The results you retain from your named GPU, workload, and configuration |

The new lab follows a complaint chain:

```text
one real answer over HTTP
-> repeated generation work: measure saved token state
-> saved work occupies memory: inspect real cache tensors
-> longer prompts and more users: measure the native engine
-> a smaller cache pool: observe queueing and preemption, not an invented OOM
-> repeated input prefixes: verify reuse while the model still generates answers
```

Saving previous token calculations, reusing a common input prefix, and returning
a previously saved whole answer are different operations. The first two belong
here. Whole-answer caching belongs to the [gateway](../../09-llm-gateway/README.md).

## GPU setup: run these commands inside Linux or WSL

The real-serving notebook requires a CUDA-visible NVIDIA GPU and Python 3.10-3.12.
Native Windows and CPU execution are deliberately rejected rather than replaced
with synthetic timings. Use a quiet GPU with enough free memory for the model,
the native runtime, and its cache; a cache-pool limit is not a total-process limit.

```bash
cd learning/11-ai-infrastructure/07-inference-systems
bash setup-serving.sh
```

Choose **Python (real LLM serving GPU)** and run
[the real-serving notebook](real-llm-serving-lab.ipynb) from the top.
The notebook kernel must run inside that Linux/WSL environment too. Open the
folder through a WSL/SSH editor session; installing Linux packages does not
change an existing Windows notebook kernel.
The existing `setup.ps1`, `setup.sh`, and `requirements.txt` remain the setup for
the original mechanism notebook; they do not install vLLM.

The GPU lab uses a separate `.venv-serving`, vLLM **0.11.0**, Transformers **4.57.1**,
and `HuggingFaceTB/SmolLM2-135M` at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. This is a deliberately pinned compatibility
target, not a claim that 0.11.0 is the latest release. Match the NVIDIA driver to the
[pinned release's GPU requirements](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html).
The first run downloads public model weights. No account, private prompt, or cloud resource is needed.

The runner forces the V1 engine and uses its metric names. Its compatibility
references are the [versioned CLI](https://docs.vllm.ai/en/v0.11.0/cli/serve.html),
[V1 metrics source](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/metrics/loggers.py),
and [streaming/tokenization protocol](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/protocol.py).
The older V0 metrics table is not the contract for this lab.

## What you measure

- The same real checkpoint with and without generation KV reuse, including an output-equality check.
- Actual cached tensor bytes as the input grows.
- A real streamed completion, its token counts, and client-observed first-text latency.
- Separate input-length, output-length, and concurrency sweeps with native cache, queue, and preemption metrics.
- The difference between rejecting one over-context request and sharing a finite cache across users.
- Cold and warm prompt-prefix requests, native hit counters, and newly generated output tokens.

Every `VllmServer` context starts a new, loopback-only process group, waits for a
model probe, retains the launch command/log, and stops only that owned process group.
The notebook releases the standalone Transformers model before launching vLLM.
Do not expose these unauthenticated lab endpoints publicly.

Results and commands go under ignored `runs/` directories. They include model/runtime
identity, hardware information, exact input token IDs, configurations, individual requests,
and sampled metrics. Prefix caching is disabled in the length/concurrency/cache-budget
comparisons, so repeated inputs do not accidentally change those experiments.

The command also keeps `--gpu-memory-utilization 0.5` fixed for startup planning,
leaving room for the notebook's CUDA context. With the explicit byte budget,
`--kv-cache-memory-bytes` determines the pool size. Neither setting is a hard
ceiling on total process memory; do not vary both in the same comparison.

## Evidence boundary and validation

**Implemented source:** the GPU lab and runner. **Local fixture:** the HTTP/protocol
tests. **GPU execution: unvalidated on the authoring machine**, which has no NVIDIA GPU.
No GPU latency, cache-hit rate, capacity, or speedup result is supplied in source.
The lab produces **Local measured** evidence only after you run it successfully.

From the repository root, the portable contract tests are:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s learning\11-ai-infrastructure\07-inference-systems -p test_serving_experiments.py -v
```

The test HTTP server is an explicit protocol fixture. It is never a fallback
backend in the notebook and says nothing about GPU performance.

Small samples and sampled cache/queue peaks are diagnostic observations, not
production p95/p99 or a load-capacity certification. Do not multiply speedups
from different workloads or transfer the tiny checkpoint's throughput to a larger model.

Afterward, continue to [Systems Composition](../09-systems-composition/README.md):
the model engine's active-sequence limit does not bound the number of HTTP requests
waiting outside it. Admission, deadlines, and release control are the next problems.
