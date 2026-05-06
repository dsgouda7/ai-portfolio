#!/usr/bin/env python3
"""Fix remaining DECISION CHECKPOINT blocks that failed in the batch replacement."""
import os

def replace_in_file(path, old, new):
    with open(path, encoding='utf-8') as f:
        content = f.read()
    if old not in content:
        print(f"  NOT FOUND in {os.path.basename(path)}: {old[:60]!r}")
        return False
    content = content.replace(old, new, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  REPLACED in {os.path.basename(path)}")
    return True


# ──────────────────────────────────────────────────────────────────
# File 2: ch08-data-validation — 4.4 checkpoint (monitor verdict)
# ──────────────────────────────────────────────────────────────────
f2 = r'c:\repos\ai-portfolio\notes\01-ml\01-regression\ch08-data-validation\README.md'

old_4_4 = (
    "### 4.4 DECISION CHECKPOINT \u2014 Phase 2 Complete\n"
    "\n"
    "**What you just computed:**\n"
    "- KL divergence: D_KL(CA || PDX) = 0.275 for `MedInc` (> 0.20 threshold \u2192 investigate zone)\n"
    "- KS statistic: D = 0.385, p = 2.3e-41 (< 0.05 \u2192 distributions are significantly different)\n"
    "- PSI score: 0.339 for `MedInc` (\u2265 0.25 \u2192 severe drift, retrain required)\n"
    "- All three metrics agree: Portland income distribution has shifted drastically from California\n"
    "\n"
    "**What it means:**\n"
    "- Drift is not marginal \u2014 it is **severe** across all three metrics\n"
    "- The mean shifted +37% (5.30 vs 3.87), but that\u2019s just one symptom\n"
    "- The **entire shape** changed: low-income bins lost mass, high-income bins gained mass\n"
    "- Model trained on \"3.9-unit income \u2192 $X\" is now extrapolating to 5.3-unit districts it never saw\n"
    "\n"
    "**What to do next:**\n"
    "\u2192 **Log drift scores:** Record PSI + KS + D_KL per feature, per batch, with timestamp\n"
    "\u2192 **Set alert thresholds:** Use PSI \u2265 0.25 for BLOCK, 0.10 \u2264 PSI < 0.25 for WARN, < 0.10 for DEPLOY\n"
    "\u2192 **Track over time:** Plot PSI weekly \u2192 detects gradual drift before it crosses block threshold\n"
    "\u2192 **For Portland:** PSI = 0.339 triggers immediate **BLOCK** + retrain pipeline (Phase 4)"
)

new_4_4 = (
    "> \U0001f4a1 **Monitor verdict:** PSI = 0.339 (severe), KS D = 0.385 (p = 2.3e-41) \u2014 Portland income shifted +37%, all three drift metrics confirm severity.\n"
    "> \u27a1\ufe0f Log scores per batch and set PSI \u2265 0.25 = BLOCK threshold before routing to Phase 3 alert configuration."
)

replace_in_file(f2, old_4_4, new_4_4)

# Check what apostrophe is actually used
with open(f2, encoding='utf-8') as f:
    txt = f.read()
idx = txt.find("### 4.4 DECISION CHECKPOINT")
if idx >= 0:
    print("Still found at index:", idx)
    # Find the apostrophe around "that's"
    sym_idx = txt.find("that", idx)
    if sym_idx >= 0:
        print("Apostrophe char:", repr(txt[sym_idx+4]))
else:
    print("4.4 checkpoint successfully removed")
