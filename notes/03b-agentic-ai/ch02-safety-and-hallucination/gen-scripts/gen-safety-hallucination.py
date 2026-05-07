from pathlib import Path
"""Generate Safety and Hallucination.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Hallucination taxonomy
  (2) Mitigation stack (layered defences)
  (3) Jailbreak attack types
  (4) Production safety checklist
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Safety & Hallucination", fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_tax = fig.add_subplot(gs[0, 0])
ax_mit = fig.add_subplot(gs[0, 1])
ax_jail = fig.add_subplot(gs[1, 0])
ax_chk = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Taxonomy ═════════════════════════
ax_tax.set_title("1 · Hallucination Taxonomy", fontsize=13,
                 fontweight="bold", color=DARK)
ax_tax.set_xlim(0, 10); ax_tax.set_ylim(0, 10); ax_tax.axis("off")

# Root
ax_tax.add_patch(FancyBboxPatch((3.8, 8.2), 2.4, 1.1,
                                boxstyle="round,pad=0.1",
                                facecolor=DARK, edgecolor="white", lw=2))
ax_tax.text(5, 8.75, "Hallucination", ha="center", va="center",
            color="white", fontweight="bold", fontsize=12)

nodes = [
    (1.3, 6.0, "Intrinsic", BLUE,
     "contradicts\nsource"),
    (4.0, 6.0, "Extrinsic", ORANGE,
     "unverifiable\nfabrication"),
    (6.7, 6.0, "Confabulation", PURPLE,
     "fake citation\nfake API"),
    (9.2, 6.0, "Reasoning", RED,
     "wrong step\nright form"),
]
for x, y, name, c, body in nodes:
    ax_tax.add_patch(FancyBboxPatch((x - 1.1, y - 0.9), 2.2, 1.8,
                                    boxstyle="round,pad=0.08",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_tax.text(x, y + 0.3, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
    ax_tax.text(x, y - 0.35, body, ha="center", va="center",
                color="white", fontsize=8.5)
    ax_tax.annotate("", xy=(x, y + 0.9), xytext=(5, 8.2),
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=1))

ax_tax.text(5, 3.5,
            "All share a common mechanism:\n"
            "next-token prediction with no truth signal.",
            ha="center", fontsize=10, color=DARK)
ax_tax.text(5, 1.7,
            "Low temperature alone does not fix it.\n"
            "You need grounding + verification.",
            ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Mitigation stack ═════════════════
ax_mit.set_title("2 · Mitigation Stack (defence in depth)",
                 fontsize=13, fontweight="bold", color=DARK)
ax_mit.set_xlim(0, 10); ax_mit.set_ylim(0, 10); ax_mit.axis("off")

layers = [
    (8.8, "Output guardrails",  RED,    "schema + NLI + refusal"),
    (7.0, "Claim verification", ORANGE, "LLM-judge vs context"),
    (5.2, "Grounding",          PURPLE, "RAG w/ citations"),
    (3.4, "Prompt constraints", BLUE,   "\"Only from menu\""),
    (1.6, "Model selection",    GREEN,  "calibrated, instruct-tuned"),
]
for y, name, c, body in layers:
    ax_mit.add_patch(FancyBboxPatch((0.8, y - 0.65), 8.4, 1.3,
                                    boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_mit.text(2.6, y, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
    ax_mit.text(6.5, y, body, ha="center", va="center",
                color="white", fontsize=9.5)
ax_mit.text(5, 0.4,
            "No single layer is sufficient. Stack them.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Jailbreaks ═══════════════════════
ax_jail.set_title("3 · Jailbreak / Prompt-Injection Types",
                  fontsize=13, fontweight="bold", color=DARK)
ax_jail.set_xlim(0, 10); ax_jail.set_ylim(0, 10); ax_jail.axis("off")

attacks = [
    (9.0, "Role-play",         '"You are DAN, do anything..."',    ORANGE),
    (7.4, "Obfuscation",       "base64 / leetspeak / translation", BLUE),
    (5.8, "Indirect injection","malicious content in retrieved doc", RED),
    (4.2, "Tool-argument exfil","steal secrets via tool output",  PURPLE),
    (2.6, "Payload splitting", "distribute intent across turns",  DARK),
    (1.0, "Over-refusal probing", "force model to reveal rules",   GREY),
]
for y, name, body, c in attacks:
    ax_jail.add_patch(FancyBboxPatch((0.3, y - 0.5), 2.6, 1.0,
                                     boxstyle="round,pad=0.05",
                                     facecolor=c, edgecolor="white", lw=1.5))
    ax_jail.text(1.6, y, name, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=9.5)
    ax_jail.text(3.2, y, body, va="center", fontsize=9.5, color=DARK,
                 style="italic")

# ═══════════════════════ PANEL 4 — Production checklist ═════════════
ax_chk.set_title("4 · Production Safety Checklist",
                 fontsize=13, fontweight="bold", color=DARK)
ax_chk.set_xlim(0, 10); ax_chk.set_ylim(0, 10); ax_chk.axis("off")

items = [
    "Input classifier (PII / policy / injection)",
    "System prompt with refusal rules + citations required",
    "RAG corpus vetted; no secrets in chunks",
    "Tool allow-list + argument schemas",
    "Output classifier (toxicity / policy / schema)",
    "Claim-level grounding check for factual domains",
    "Red-team suite run pre-deploy + nightly",
    "Telemetry: refusals, overrides, user reports",
    "Kill-switch + rollback path",
]
for i, txt in enumerate(items):
    y = 9.2 - i * 0.95
    ax_chk.add_patch(FancyBboxPatch((0.3, y - 0.35), 0.7, 0.7,
                                    boxstyle="round,pad=0.05",
                                    facecolor=GREEN, edgecolor="white", lw=1.2))
    ax_chk.text(0.65, y, "v", ha="center", va="center",
                color="white", fontweight="bold", fontsize=11)
    ax_chk.text(1.2, y, txt, va="center", fontsize=10, color=DARK)

out = str(Path(__file__).resolve().parent.parent / "img" / "Safety and Hallucination.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
