# ToT Benchmark Tests - Manual Execution Guide

## Overview

This directory contains scripts to test **Tree-of-Thought (ToT) multi-perspective retrieval** on both text and multimedia datasets using real public data.

## 📁 New Files Created

### Data Download Scripts
- **`download_test_data.py`** - Downloads real text datasets (Project Gutenberg books, etc.)
- **`download_multimedia_data.py`** - Creates multimedia corpus (image captions, audio transcripts, video descriptions)

### Benchmark Scripts
- **`run_fast_tot_benchmarks.py`** - Text corpus ToT benchmarks (books, code, wiki)
- **`run_multimedia_tot_benchmarks.py`** - Multimedia ToT benchmarks (images, audio, video)
- **`run_all_tot_benchmarks.py`** - Unified runner for all benchmarks
- **`run_manual_tests.py`** - Manual execution wrapper with file logging

### Support Scripts
- **`minimal_tot_test.py`** - Quick 5-minute proof-of-concept test
- **`hello_test.py`** - Simple environment validation test

## 🚀 How to Run Tests Manually

Since terminal output is not functioning in the current environment, run tests manually:

### Option 1: Quick Test (5 minutes)
```powershell
cd c:\repos\ai-portfolio\projects\context-optimizer
.venv\Scripts\Activate.ps1
python experiments\minimal_tot_test.py
```
**Output**: `experiments/minimal_test_log.txt`, `experiments/MINIMAL_TOT_RESULTS.json`

### Option 2: Full Text Benchmarks (20-30 minutes)
```powershell
python experiments\run_fast_tot_benchmarks.py
```
**Output**: `experiments/FAST_TOT_RESULTS.json`

### Option 3: Multimedia Benchmarks (30-40 minutes)
```powershell
python experiments\run_multimedia_tot_benchmarks.py
```
**Output**: `experiments/temp_results/multimedia_tot_*.md`, JSON files

### Option 4: Complete Suite (60-90 minutes)
```powershell
python experiments\run_manual_tests.py
```
**Output**: All temp results in `experiments/temp_results/`

## 📊 Test Structure

### Text Benchmarks
- **3 corpus sizes**: 100MB, 500MB, 1GB
- **3 domains**: books (Gutenberg), code (Python), wiki articles
- **6 queries total** (2 per size)
- **Metrics**: F1 quality, token overhead, latency, deduplication %

### Multimedia Benchmarks
- **3 corpus sizes**: 100MB, 500MB, 1GB
- **3 media types**: images (captions), audio (transcripts), video (descriptions)
- **9 queries total** (3 per type)
- **Same metrics** as text benchmarks

## 🎯 Success Criteria

ToT should demonstrate:
1. **F1 improvement > +0.05** (5% better quality than single-path)
2. **Token ratio < 3x** (acceptable overhead)
3. **Deduplication > 20%** (proves multi-perspective works)
4. **Scaling trend**: larger corpus → bigger F1 improvement

## 📈 Expected Results

### Hypothesis
**ToT efficiency gains INCREASE with corpus size** because multi-perspective search compensates for larger search space.

### Predictions
- **100MB**: +5-7% F1 improvement
- **500MB**: +7-10% F1 improvement
- **1GB**: +10-15% F1 improvement

### Token Overhead
- Target: <3x tokens vs single-path
- Deduplication should reduce overhead by 20-40%

## 📂 Output Files

All results saved to: `experiments/temp_results/`

### Generated Files
- `multimedia_tot_images_YYYYMMDD_HHMMSS.md` - Image benchmark results
- `multimedia_tot_audio_YYYYMMDD_HHMMSS.md` - Audio benchmark results
- `multimedia_tot_video_YYYYMMDD_HHMMSS.md` - Video benchmark results
- `multimedia_tot_results_YYYYMMDD_HHMMSS.json` - JSON data for all multimedia
- `FAST_TOT_RESULTS.json` - Text benchmark results (in experiments/)
- `MASTER_SUMMARY_YYYYMMDD_HHMMSS.md` - Overall summary
- `execution_log_YYYYMMDD_HHMMSS.txt` - Full execution log

## 🔍 How to Check Results

### 1. Check if tests ran
```powershell
ls experiments\temp_results\
```
Should show multiple `.md` and `.json` files.

### 2. Read markdown summaries
```powershell
cat experiments\temp_results\MASTER_SUMMARY_*.md
cat experiments\temp_results\multimedia_tot_images_*.md
```

### 3. Parse JSON programmatically
```python
import json
with open('experiments/temp_results/multimedia_tot_results_*.json') as f:
    results = json.load(f)
```

## 🐛 Troubleshooting

### Terminal output not working?
→ Use `run_manual_tests.py` which writes to files

### LLM not responding?
→ Check Ollama is running: `ollama list`, `ollama run qwen2.5-coder:7b`

### Download failures?
→ Scripts use fallback synthetic data if downloads fail

### Out of memory?
→ Reduce corpus sizes in scripts (e.g., change `* 500` to `* 50`)

## 📝 Next Steps After Running Tests

1. **Review temp markdown files** in `experiments/temp_results/`
2. **Check if hypothesis validated**:
   - Does F1 improvement increase with corpus size?
   - Is token overhead acceptable (<3x)?
   - Does deduplication work (>20%)?
3. **Consolidate results** into final experiment reports:
   - Update `EXPERIMENTS_GUIDE.md`
   - Update `ARCHITECTURE_EVOLUTION.md`
   - Update `proposed-whitepaper.md`
4. **Decision**: If validated → Plan production integration

## 🎓 Understanding the Tests

### Single-Path Retrieval (Baseline)
- Single query: `"How does authentication work?"`
- Returns top-6 chunks
- Measures: F1, tokens, latency

### Multi-Perspective ToT
- Generate 3 perspectives:
  - `"broad overview: How does authentication work?"`
  - `"specific details: How does authentication work?"`
  - `"related context: How does authentication work?"`
- Retrieve top-2 from each (6 chunks total before dedup)
- Deduplicate by chunk_id (keep highest relevance)
- Re-rank and return top-6
- Measures: same metrics + deduplication %

### Improvement Calculation
- `Δ F1 = ToT_F1 - Single_F1`
- `Token Ratio = ToT_tokens / Single_tokens`
- `Latency Ratio = ToT_latency / Single_latency`

## 📚 Related Documentation

- `TOT_BENCHMARKS_README.md` - Original ToT hypothesis and design
- `ARCHITECTURE_EVOLUTION.md` - Project evolution timeline
- `EXPERIMENTS_GUIDE.md` - All experiment documentation
- `proposed-whitepaper.md` - Theoretical foundations

## ✅ Quick Validation Checklist

After tests complete:

- [ ] Temp markdown files exist in `experiments/temp_results/`
- [ ] MASTER_SUMMARY shows success for both phases
- [ ] Average F1 improvement is positive across all corpus sizes
- [ ] Token ratio is below 3x for most tests
- [ ] Deduplication percentage is above 20%
- [ ] Scaling trend shows improvement: 100MB < 500MB < 1GB

If all checked → **Hypothesis validated, proceed with consolidation!**
