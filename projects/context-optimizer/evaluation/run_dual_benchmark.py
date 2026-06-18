from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"
OUT_DIR = EVAL_DIR / "out"
DATA_DIR = EVAL_DIR / "data"
LOG_FILE = DATA_DIR / "community_logs.txt"
RAW_METRICS_FILE = OUT_DIR / "raw_metrics.json"
OPT_METRICS_FILE = OUT_DIR / "optimized_metrics.json"
REPORT_FILE = OUT_DIR / "evaluation_report.md"
ARCH_GIF = OUT_DIR / "architecture_differences.gif"
METRICS_GIF = OUT_DIR / "metrics_animation.gif"

RAW_IMAGE = "context-optimizer-raw:cpu"
OPT_IMAGE = "context-optimizer-optimized:cpu"

COMMUNITY_LOG_URLS = [
    "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/nginx_logs/nginx_logs",
    "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
    "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/apache_logs/apache_logs",
]


def run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=True)


def ensure_community_logs(min_lines: int = 5000) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    aggregated: list[str] = []
    for url in COMMUNITY_LOG_URLS:
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                payload = response.read().decode("utf-8", errors="replace")
                lines = [line for line in payload.splitlines() if line.strip()]
                aggregated.extend(lines)
        except Exception:
            continue

    if len(aggregated) < min_lines:
        # Fallback: deterministic synthetic expansion for CPU benchmarks.
        for i in range(min_lines - len(aggregated)):
            aggregated.append(
                f"2026-06-16T02:{i % 60:02d}:{i % 60:02d}.000Z WARN ingress-nginx req-{i:06d} "
                "upstream timed out while reading response header from upstream service=order-service "
                "dependency=CosmosDB code=21012"
            )

    LOG_FILE.write_text("\n".join(aggregated[:max(min_lines, 6000)]), encoding="utf-8")


def build_images() -> None:
    run_cmd(["docker", "build", "-f", "Dockerfile.raw", "-t", RAW_IMAGE, "."], cwd=PROJECT_ROOT)
    run_cmd(["docker", "build", "-f", "Dockerfile.optimized", "-t", OPT_IMAGE, "."], cwd=PROJECT_ROOT)


def run_container(image: str, metrics_out_name: str, pipeline: str) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = OUT_DIR / metrics_out_name
    if metrics_file.exists():
        metrics_file.unlink()

    data_mount = f"{DATA_DIR.resolve()}:/app/evaluation/data:ro"
    out_mount = f"{OUT_DIR.resolve()}:/app/evaluation/out"

    cmd = [
        "docker",
        "run",
        "--rm",
        "--cpus",
        "2",
        "--memory",
        "4g",
        "-v",
        data_mount,
        "-v",
        out_mount,
        image,
        "python",
        "context_optimizer_benchmark.py",
        "--provider",
        "mock",
        "--pipeline",
        pipeline,
        "--log-file",
        "/app/evaluation/data/community_logs.txt",
        "--metrics-json",
        f"/app/evaluation/out/{metrics_out_name}",
    ]

    start = time.perf_counter()
    proc = run_cmd(cmd)
    wall_time = time.perf_counter() - start

    if not metrics_file.exists():
        raise RuntimeError(f"Expected metrics file not found: {metrics_file}")

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    metrics["container_wall_time_s"] = wall_time
    metrics["container_image"] = image
    metrics["container_stdout_tail"] = "\n".join(proc.stdout.splitlines()[-20:])
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _animate_bars(
    labels: list[str],
    raw_values: list[float],
    opt_values: list[float],
    title: str,
    ylabel: str,
    output_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(labels))

    def update(frame: int):
        ax.clear()
        progress = min(1.0, frame / 30.0)
        raw_frame = [v * progress for v in raw_values]
        opt_frame = [v * progress for v in opt_values]

        width = 0.35
        ax.bar([i - width / 2 for i in x], raw_frame, width, label="Raw Context", color="#D1495B")
        ax.bar([i + width / 2 for i in x], opt_frame, width, label="Optimized", color="#2E86AB")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=12)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    ani = FuncAnimation(fig, update, frames=36, interval=100, repeat=False)
    ani.save(str(output_file), writer=PillowWriter(fps=10))
    plt.close(fig)


def create_animations(raw: dict, optimized: dict) -> None:
    opt_reasoning = float(optimized.get("pipe_c_reasoning_s", 0.0))
    opt_log_lines = float(optimized.get("pipe_c_log_lines", 0))
    opt_tool_calls = float(optimized.get("pipe_c_tool_calls", 0))

    _animate_bars(
        labels=["Reasoning(s)", "Wall(s)", "Lines Processed"],
        raw_values=[
            float(raw.get("pipe_a_reasoning_s", 0.0)),
            float(raw.get("container_wall_time_s", 0.0)),
            float(raw.get("pipe_a_log_lines", 0)),
        ],
        opt_values=[
            opt_reasoning,
            float(optimized.get("container_wall_time_s", 0.0)),
            opt_log_lines,
        ],
        title="CPU Benchmark Metrics: Raw vs Optimized",
        ylabel="Value",
        output_file=METRICS_GIF,
    )

    _animate_bars(
        labels=["Prompt Chars", "Compression Latency(s)", "Tool Calls"],
        raw_values=[
            float(raw.get("raw_char_count", 0)),
            float(raw.get("compression_latency_s", 0.0)),
            0.0,
        ],
        opt_values=[
            float(optimized.get("compressed_char_count", 0)),
            float(optimized.get("compression_latency_s", 0.0)),
            opt_tool_calls,
        ],
        title="Architectural Difference Signals",
        ylabel="Value",
        output_file=ARCH_GIF,
    )


def write_report(raw: dict, optimized: dict) -> None:
    opt_reasoning = float(optimized.get("pipe_c_reasoning_s", 0.0))
    opt_log_lines = float(optimized.get("pipe_c_log_lines", 0))
    opt_tool_calls = int(optimized.get("pipe_c_tool_calls", 0))

    raw_lines = float(raw.get("pipe_a_log_lines", 1))
    opt_lines = float(max(1, opt_log_lines))
    line_reduction = ((raw_lines - opt_lines) / raw_lines) * 100.0 if raw_lines else 0.0

    raw_wall = float(raw.get("container_wall_time_s", 0.0))
    opt_wall = float(optimized.get("container_wall_time_s", 0.0))
    wall_delta = raw_wall - opt_wall

    report = textwrap.dedent(
        f"""
        # Context Optimization Docker Benchmark Report

        ## Run summary

        - Raw image: `{raw.get('container_image')}`
        - Optimized image: `{optimized.get('container_image')}`
        - Log corpus file: `evaluation/data/community_logs.txt`
        - CPU limits: `--cpus 2 --memory 4g` per container

        ## Key metrics

        | Metric | Raw Context | Optimized | Delta |
        |---|---:|---:|---:|
        | Container wall time (s) | {raw_wall:.4f} | {opt_wall:.4f} | {wall_delta:.4f} |
        | Reasoning latency (s) | {float(raw.get('pipe_a_reasoning_s', 0.0)):.4f} | {opt_reasoning:.4f} | {float(raw.get('pipe_a_reasoning_s', 0.0)) - opt_reasoning:.4f} |
        | Prompt chars | {int(raw.get('raw_char_count', 0))} | {int(optimized.get('compressed_char_count', 0))} | {int(raw.get('raw_char_count', 0)) - int(optimized.get('compressed_char_count', 0))} |
        | Log lines touched | {int(raw.get('pipe_a_log_lines', 0))} | {int(opt_log_lines)} | {line_reduction:.2f}% reduction |
        | Tool calls | 0 | {opt_tool_calls} | +{opt_tool_calls} |

        ## Animated charts

        - Metrics animation: ![Metrics animation](metrics_animation.gif)
        - Architectural differences animation: ![Architecture animation](architecture_differences.gif)

        ## Architectural differences

        | Dimension | Raw Context Container | Optimized Container |
        |---|---|---|
        | Pipeline mode | `raw` | `optimized` |
        | Prompt strategy | Sends full prompt + full log corpus | Sends compressed prompt with structured fields |
        | Retrieval strategy | None (eager context loading) | Dynamic retrieval via `query_log_cache` |
        | Log I/O profile | High upfront ingestion | Sparse targeted excerpts |
        | Context efficiency | Lower | Higher |

        ## Notes

        - Community logs are downloaded from public GitHub sources when available.
        - If remote fetch is unavailable, deterministic synthetic logs are generated as fallback so the benchmark remains runnable.
        - This run used mock provider mode for CPU-stable, dependency-free comparative execution.
        """
    ).strip()

    REPORT_FILE.write_text(report + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        ensure_community_logs()
        build_images()
        raw_metrics = run_container(RAW_IMAGE, RAW_METRICS_FILE.name, "raw")
        optimized_metrics = run_container(OPT_IMAGE, OPT_METRICS_FILE.name, "optimized")
        create_animations(raw_metrics, optimized_metrics)
        write_report(raw_metrics, optimized_metrics)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr + "\n")
        return 1
    except Exception as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1

    print(f"Report generated: {REPORT_FILE}")
    print(f"Metrics files: {RAW_METRICS_FILE}, {OPT_METRICS_FILE}")
    print(f"Animations: {METRICS_GIF}, {ARCH_GIF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
