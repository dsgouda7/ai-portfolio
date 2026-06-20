"""
Quality evaluation framework for comparing pipeline answers.

Two-track evaluation:
  1. Structural scoring   — deterministic, keyword-based, no LLM required.
                            Works in mock mode and with any real model.
  2. LLM-as-judge scoring — requires a live LLM (Ollama or Groq).
                            Returns semantic scores with JSON output.

Both tracks score 0–1 per dimension and emit an overall quality score.
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from experiments.shared_inputs import (
    BONUS_KEYWORDS,
    GROUND_TRUTH,
    REQUIRED_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Structural (keyword-based) scorer — deterministic, no LLM needed
# ---------------------------------------------------------------------------


@dataclass
class StructuralScore:
    pipe_name: str
    # Fraction of REQUIRED_KEYWORDS present in the answer (0–1)
    keyword_coverage: float = 0.0
    # Fraction of BONUS_KEYWORDS present (0–1)
    specificity: float = 0.0
    # Mentions CosmosDB as a root cause (binary)
    identified_root_cause: bool = False
    # Mentions at least 2 affected services
    service_coverage: float = 0.0
    # Mentions at least 1 mitigation action
    has_mitigations: bool = False
    # Mentions at least 1 next-check / observability step
    has_next_steps: bool = False
    # Composite 0–1
    overall: float = 0.0
    # Per-keyword hit list for debugging
    required_hits: list[str] = field(default_factory=list)
    bonus_hits: list[str] = field(default_factory=list)


def structural_score(pipe_name: str, answer: str) -> StructuralScore:
    """Score an answer deterministically against ground truth keywords."""
    lower = answer.lower()

    # Required keyword hits
    req_hits = [kw for kw in REQUIRED_KEYWORDS if kw.lower() in lower]
    kw_cov = len(req_hits) / max(1, len(REQUIRED_KEYWORDS))

    # Bonus keyword hits
    bonus_hits = [kw for kw in BONUS_KEYWORDS if kw.lower() in lower]
    spec = len(bonus_hits) / max(1, len(BONUS_KEYWORDS))

    # Root cause identification
    root_cause_ok = "cosmos" in lower and ("21012" in lower or "timeout" in lower)

    # Service coverage (how many affected services mentioned)
    services_mentioned = sum(
        1 for svc in GROUND_TRUTH["affected_services"] if svc.lower() in lower
    )
    svc_cov = services_mentioned / max(1, len(GROUND_TRUTH["affected_services"]))

    # Mitigations heuristic
    mitigation_signals = ["mitigation", "action", "increase", "tune", "cap retries",
                          "retry", "restart", "rollback", "fix", "adjust", "configure",
                          "check cosmos", "autoscale", "backoff", "circuit breaker"]
    has_mitigations = any(sig in lower for sig in mitigation_signals)

    # Next-steps heuristic
    next_step_signals = ["next check", "next step", "observe", "monitor", "query",
                         "investigate", "look at", "inspect", "verify", "prometheus",
                         "application insights", "ru consumption", "dashboard"]
    has_next_steps = any(sig in lower for sig in next_step_signals)

    # Composite: weighted average
    overall = (
        kw_cov * 0.30
        + spec * 0.20
        + (1.0 if root_cause_ok else 0.0) * 0.20
        + svc_cov * 0.15
        + (1.0 if has_mitigations else 0.0) * 0.10
        + (1.0 if has_next_steps else 0.0) * 0.05
    )

    return StructuralScore(
        pipe_name=pipe_name,
        keyword_coverage=round(kw_cov, 3),
        specificity=round(spec, 3),
        identified_root_cause=root_cause_ok,
        service_coverage=round(svc_cov, 3),
        has_mitigations=has_mitigations,
        has_next_steps=has_next_steps,
        overall=round(overall, 3),
        required_hits=req_hits,
        bonus_hits=bonus_hits,
    )


# ---------------------------------------------------------------------------
# LLM-as-judge scorer
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = textwrap.dedent(
    """
    You are a strict technical evaluator assessing incident-response analyses.

    You will receive:
    1. The GROUND TRUTH describing what a correct analysis must identify.
    2. An ANSWER produced by a pipeline under evaluation.

    Score the ANSWER on the following dimensions (each 1–5):
      - accuracy:      Are the identified root causes correct and evidence-backed?
      - completeness:  Does the answer cover all key entities, services, and error codes?
      - actionability: Are the suggested mitigations and next checks specific and useful?
      - precision:     Does the answer avoid vague statements, noise, or hallucinations?

    Respond ONLY with valid JSON in this exact format:
    {
      "accuracy": <1-5>,
      "completeness": <1-5>,
      "actionability": <1-5>,
      "precision": <1-5>,
      "overall": <float 1.0-5.0, weighted average>,
      "brief_justification": "<one sentence>"
    }
    """
).strip()


@dataclass
class LLMJudgeScore:
    pipe_name: str
    accuracy: float = 0.0
    completeness: float = 0.0
    actionability: float = 0.0
    precision: float = 0.0
    overall: float = 0.0
    justification: str = ""
    error: str = ""

    @property
    def normalised_overall(self) -> float:
        """Return overall as a 0–1 value for fair comparison with structural score."""
        return round(self.overall / 5.0, 3)


def llm_judge_score(
    judge_llm: Any,
    pipe_name: str,
    answer: str,
) -> LLMJudgeScore:
    """
    Ask a judge LLM to score a pipeline answer against the ground truth.

    Returns LLMJudgeScore with error field set if the LLM call fails or
    returns malformed JSON.
    """
    ground_truth_text = json.dumps(GROUND_TRUTH, indent=2)
    user_msg = (
        f"GROUND TRUTH:\n{ground_truth_text}\n\n"
        f"ANSWER TO EVALUATE:\n{answer}\n\n"
        "Respond with the JSON score only."
    )

    try:
        response = judge_llm.invoke(
            [
                SystemMessage(content=_JUDGE_SYSTEM),
                HumanMessage(content=user_msg),
            ]
        )
        raw = str(response.content).strip()

        # Extract JSON even if the LLM wraps it in markdown fences
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return LLMJudgeScore(pipe_name=pipe_name, error=f"No JSON in response: {raw[:200]}")

        data = json.loads(json_match.group())
        return LLMJudgeScore(
            pipe_name=pipe_name,
            accuracy=float(data.get("accuracy", 0)),
            completeness=float(data.get("completeness", 0)),
            actionability=float(data.get("actionability", 0)),
            precision=float(data.get("precision", 0)),
            overall=float(data.get("overall", 0)),
            justification=str(data.get("brief_justification", "")),
        )
    except Exception as exc:
        return LLMJudgeScore(pipe_name=pipe_name, error=str(exc))


# ---------------------------------------------------------------------------
# Unified comparison report
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    structural: list[StructuralScore] = field(default_factory=list)
    llm_judge: list[LLMJudgeScore] = field(default_factory=list)

    def add_structural(self, s: StructuralScore) -> None:
        self.structural.append(s)

    def add_llm_judge(self, s: LLMJudgeScore) -> None:
        self.llm_judge.append(s)

    def summary_table(self) -> str:
        """Return a markdown table comparing all pipes."""
        lines: list[str] = []

        # Structural table
        lines.append("### Structural Quality Scores (keyword / heuristic, no LLM)")
        lines.append("")
        lines.append(
            "| Pipeline | Keyword Coverage | Specificity | Root Cause Found | "
            "Service Coverage | Mitigations | Next Steps | Overall |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for s in self.structural:
            lines.append(
                f"| {s.pipe_name} "
                f"| {s.keyword_coverage:.0%} "
                f"| {s.specificity:.0%} "
                f"| {'✅' if s.identified_root_cause else '❌'} "
                f"| {s.service_coverage:.0%} "
                f"| {'✅' if s.has_mitigations else '❌'} "
                f"| {'✅' if s.has_next_steps else '❌'} "
                f"| **{s.overall:.3f}** |"
            )
        lines.append("")

        if self.llm_judge:
            judge_errors = [j for j in self.llm_judge if j.error]
            judge_ok = [j for j in self.llm_judge if not j.error]

            if judge_ok:
                lines.append("### LLM-as-Judge Scores (1–5 per dimension)")
                lines.append("")
                lines.append(
                    "| Pipeline | Accuracy | Completeness | Actionability | "
                    "Precision | Overall (1–5) | Overall (0–1) | Justification |"
                )
                lines.append("|---|---|---|---|---|---|---|---|")
                for j in judge_ok:
                    lines.append(
                        f"| {j.pipe_name} "
                        f"| {j.accuracy:.1f} "
                        f"| {j.completeness:.1f} "
                        f"| {j.actionability:.1f} "
                        f"| {j.precision:.1f} "
                        f"| **{j.overall:.2f}** "
                        f"| {j.normalised_overall:.3f} "
                        f"| {j.justification} |"
                    )
                lines.append("")

            if judge_errors:
                lines.append("_LLM judge errors (check Ollama availability):_")
                for j in judge_errors:
                    lines.append(f"- {j.pipe_name}: `{j.error[:120]}`")
                lines.append("")

        return "\n".join(lines)
