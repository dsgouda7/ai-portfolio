# Improvement Plan — PyTorch Profiling

**Audited:** 2026-07-26 | **Audience fit:** 7/10

## Overall Assessment

Solid skeleton: compelling mystery hook, per-Part question map, "Predict first" before every major result, "Missing piece" transitions between Parts, and a closing bottleneck decision. The measurement discipline is professional — GPU synchronization at every timing boundary, medians over multiple runs, warmup passes before timing. Three critical moments where a first-time profiler *needs* a guide are absent: reading the profiler output table, navigating chrome://tracing, and hearing the 45-second mystery explicitly closed with arithmetic.

---

## Strengths (preserve these)

- **"🔮 Predict first" boxes** before every major result in Parts 1, 3, and 6 — the notebook's best pedagogical device
- **Phase-level bar charts** — bottleneck visible without mental arithmetic
- **"Missing piece" markdown transitions** — each Part ends naming what the current tool can't answer and points to the next Part
- **Break-even calculation for `torch.compile`** — "Break-even at N forward passes" is concrete and actionable
- **Tier 1/2/3 coverage map** — prevents "why didn't they cover Nsight Systems?" confusion
- **Prerequisite bridge placement** — `01-gpu-hardware` prerequisite anchored to Part 3 where it's actually needed
- **GPU synchronization discipline** — `torch.cuda.synchronize()` at every timing boundary throughout

---

## Gaps & Recommended Changes

### Gap 1 — Profiler table dumped without a reading guide — Priority: Critical

**Problem:** Part 1 calls `print(key_averages.table(...))` and the follow-up says "identified the most expensive phase." The table is never explained. Columns like `cpu_time_total` vs. `self_cpu_time_total` appear with no guidance on which to look at or what the distinction means. An op with high `cpu_time_total` but low `self_cpu_time_total` is a caller — the bottleneck is in its children.

**Recommendation:** Add a markdown cell immediately after the print output:

```markdown
#### How to read this table

| Column | Meaning | When to use |
|---|---|---|
| `self_cpu_time_total` | Time *inside* this op, excluding children | **Sort by this** to find the actual hot op |
| `cpu_time_total` | Total time including all child ops | High here + low `self` = wrapper, not the cause |
| `cuda_time_total` | Matching GPU kernel time | Large gap vs. CPU = async overlap |
| `count` | Times this op ran | High count + low self = loop overhead |

**Rule:** Sort by `self_cpu_time_total` to find where time is actually spent.
```

---

### Gap 2 — Chrome trace guidance is nearly absent — Priority: High

**Problem:** Part 5 closes with two print lines pointing to `chrome://tracing`. A first-timer who opens that URL sees a blank canvas with an unintuitive toolbar — no indication of how to load the file, what the rows represent, or how named `record_function` regions appear.

**Recommendation:** Add a markdown cell after the trace-save line:
```markdown
#### Reading the Chrome trace (5 steps)

1. Open `chrome://tracing` (or [ui.perfetto.dev](https://ui.perfetto.dev)) in Chrome
2. Click **Load** → select `profiler_trace.json`
3. **Rows** = CPU threads; GPU activity appears as additional CUDA-stream rows below
4. Your `record_function` names appear as **colored spans** — find `step_0/backward`, `step_0/forward`, etc.
5. **Navigate:** W/S = zoom in/out, A/D = pan; the **widest span** is your bottleneck

**What to look for first:** The widest span in the backward row. Click it — the bottom panel shows exact duration.
```

---

### Gap 3 — The 45-second mystery is never explicitly closed — Priority: Medium

**Problem:** The opening hook says "your fine-tuning loop takes 45 seconds per epoch." Part 6 outputs per-step milliseconds but never connects them back to epoch-level time. The mystery is not answered in the original units.

**Recommendation:** Add 5 lines to the Part 6 closing print block:
```python
steps_per_epoch = 1000  # substitute: len(train_loader)
epoch_s = (total_ms / 1000) * steps_per_epoch
print(f"\n--- Connecting back to the 45-second mystery ---")
print(f"  Assuming {steps_per_epoch} steps/epoch → epoch ≈ {epoch_s:.0f}s")
print(f"  (Substitute your real count: steps_per_epoch = len(train_loader))")
```

---

### Gap 4 — Compute vs. memory bound lacks hardware-ceiling anchoring — Priority: Medium

**Problem:** Part 3 prints GFLOPS and GB/s values unanchored. "42.3 GFLOPS" means nothing without knowing "my GPU's FP32 peak is 30,000 GFLOPS."

**Recommendation:** Add hardware peak output:
```python
if HAS_GPU:
    props = torch.cuda.get_device_properties(0)
    peak_tflops = props.multi_processor_count * 128 * 2 * props.clock_rate * 1e3 / 1e12
    print(f"  GPU peak (rough): ~{peak_tflops:.0f} TFLOPS FP32")
    util_pct = matmul_flops / t_matmul * 1000 / (peak_tflops * 1000) * 100
    print(f"  matmul at {util_pct:.1f}% of peak → {'compute-bound' if util_pct > 50 else 'not near compute ceiling'}")
```

---

### Gap 5 — Profiler overhead: mechanism missing — Priority: Low

**Problem:** Part 2 measures overhead correctly but doesn't explain *why* profiling has overhead (the profiler intercepts every PyTorch C++ dispatch call to record a timestamp).

**Recommendation:** Add one sentence: "Overhead scales with the number of op calls, not compute time — a loop with many small Python-level ops will see higher overhead than a simple matmul-only benchmark."

---

## Do NOT Change

- "🔮 Predict first" boxes in Parts 1, 3, and 6
- Phase-level bar charts (bottleneck visible without mental arithmetic)
- "Missing piece" markdown transitions between Parts
- Break-even calculation for `torch.compile`
- GPU synchronization discipline (`torch.cuda.synchronize()` at every boundary)
- Tier 1/2/3 coverage map
- Prerequisite bridge placement
