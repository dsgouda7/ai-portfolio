"""
gen_ch10_monitoring_needle.py
Generates: ../img/ch10-monitoring-needle.gif

Progress animation showing detection time improvement:
From 2 weeks (manual checks) to 2 hours (automated monitoring)

Shows the "needle moving" - dramatic time savings from production monitoring.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Manual checks (14 days)",
        "value": 20160.0,  # 14 days in minutes
        "curve": [0.24, 0.02, 1.08, 0.32],
        "caption": "Weekly cron job dumps prediction logs; engineer manually compares distributions in notebook; drift discovered two weeks after it started.",
    },
    {
        "label": "Daily logs (3 days)",
        "value": 4320.0,  # 3 days in minutes
        "curve": [0.28, 0.06, 1.00, 0.22],
        "caption": "Automated daily reports compute basic statistics; team reviews metrics each morning but analysis requires manual correlation work.",
    },
    {
        "label": "Alerting system (12 hours)",
        "value": 720.0,  # 12 hours in minutes
        "curve": [0.33, 0.12, 0.94, 0.10],
        "caption": "Slack alerts trigger on threshold breaches; engineer investigates after next business-hours check but still needs to diagnose root cause.",
    },
    {
        "label": "Real-time monitoring (2 hours)",
        "value": 120.0,  # 2 hours in minutes
        "curve": [0.38, 0.18, 0.90, 0.02],
        "caption": "Evidently dashboard shows drift score + feature breakdown + prediction distribution in one view; automated rollback triggers immediately if critical.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent.parent / "img",
        "ch10-monitoring-needle",
        "Ch.10 — Production monitoring cuts drift detection from 14 days to 2 hours",
        "Detection Time (minutes)",
        STAGES,
        better="lower",
        style="regression",
    )
