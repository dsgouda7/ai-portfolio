from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import typer
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from app.config import load_platform_config
from release_gates import (
    DatasetRef,
    Decision,
    Domain,
    EvaluatorRef,
    EvidenceRef,
    MetricRecord,
    MetricStatus,
    Observation,
    ReleaseContext,
    Slice,
    Threshold,
    ThresholdOperator,
    Uncertainty,
    build_release_report,
    evaluate_release,
    riverside_v1_policy,
    validate_decision_consistency,
    write_release_report,
)


cli = typer.Typer(
    name="riverside",
    no_args_is_help=True,
    help="Validate configuration and contracts, evaluate release evidence, and render reports.",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"{path} is not a readable JSON document") from error


def _schema_registry() -> Registry[Any]:
    paths = [*_project_root().joinpath("contracts", "v1").glob("*.schema.json")]
    paths.append(_project_root() / "config" / "schema.json")
    resources: list[tuple[str, Resource[Any]]] = []
    for path in paths:
        schema = _load_json(path)
        if not isinstance(schema, dict) or not isinstance(schema.get("$id"), str):
            raise typer.BadParameter(f"schema has no $id: {path}")
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def _validate_document(schema_path: Path, document: Any) -> None:
    schema = _load_json(schema_path)
    if not isinstance(schema, dict):
        raise typer.BadParameter("schema must be a JSON object")
    validator = Draft202012Validator(
        schema,
        registry=_schema_registry(),
        format_checker=FormatChecker(),
    )
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise typer.BadParameter(f"contract validation failed at {location}: {first.message}")


def _metric(raw: Mapping[str, Any]) -> MetricRecord:
    uncertainty = raw.get("uncertainty")
    return MetricRecord(
        domain=Domain(str(raw["domain"])),
        metric_id=str(raw["metric_id"]),
        dataset=DatasetRef(**raw["dataset"]),
        evaluator=EvaluatorRef(**raw["evaluator"]),
        slice=Slice(**raw["slice"]),
        threshold=Threshold(
            operator=ThresholdOperator(str(raw["threshold"]["operator"])),
            value=float(raw["threshold"]["value"]),
            unit=str(raw["threshold"]["unit"]),
        ),
        observed=Observation(
            value=float(raw["observed"]["value"]),
            unit=str(raw["observed"]["unit"]),
        ),
        uncertainty=Uncertainty(**uncertainty) if isinstance(uncertainty, dict) else None,
        status=MetricStatus(str(raw["status"])) if raw.get("status") is not None else None,
    )


def _release_context(raw: Mapping[str, Any]) -> ReleaseContext:
    evidence = tuple(
        EvidenceRef(
            name=str(item["name"]),
            uri=str(item["uri"]),
            digest=str(
                item["digest"]["value"]
                if isinstance(item.get("digest"), dict)
                else item["digest"]
            ),
        )
        for item in raw["evidence"]
    )
    return ReleaseContext(
        report_id=str(raw["report_id"]),
        release_id=str(raw["release_id"]),
        release_version=str(raw["release_version"]),
        generated_at=datetime.fromisoformat(str(raw["generated_at"]).replace("Z", "+00:00")),
        source_commit=str(raw["source_commit"]),
        baseline_release_id=(
            str(raw["baseline_release_id"])
            if raw.get("baseline_release_id") is not None
            else None
        ),
        evidence=evidence,
    )


@cli.command("validate")
def validate_command(
    config: Path | None = typer.Option(None, "--config", help="Environment YAML profile."),
    schema: Path | None = typer.Option(None, "--schema", help="Draft 2020-12 JSON Schema."),
    document: Path | None = typer.Option(None, "--document", help="JSON document to validate."),
) -> None:
    """Validate one resolved environment profile or one contract document."""

    if config is not None and schema is None and document is None:
        profile = load_platform_config(config)
        typer.echo(
            json.dumps(
                {
                    "status": "valid",
                    "environment": profile.environment,
                    "region": profile.region,
                    "endpoint_name": profile.serving.endpoint_name,
                },
                sort_keys=True,
            )
        )
        return
    if config is None and schema is not None and document is not None:
        _validate_document(schema, _load_json(document))
        typer.echo(json.dumps({"status": "valid", "schema": schema.name}, sort_keys=True))
        return
    raise typer.BadParameter(
        "use --config by itself, or supply both --schema and --document"
    )


@cli.command("evaluate")
def evaluate_command(
    metrics: Path = typer.Option(..., "--metrics", help="JSON array or object with a metrics array."),
    context: Path = typer.Option(..., "--context", help="Release report context JSON."),
    output: Path = typer.Option(..., "--output", help="Destination release-report JSON."),
) -> None:
    """Evaluate precomputed evidence with the v1 gate and write a release report."""

    raw_metrics = _load_json(metrics)
    if isinstance(raw_metrics, dict):
        raw_metrics = raw_metrics.get("metrics")
    if not isinstance(raw_metrics, list) or not all(isinstance(item, dict) for item in raw_metrics):
        raise typer.BadParameter("metrics input must be an array of metric objects")
    raw_context = _load_json(context)
    if not isinstance(raw_context, dict):
        raise typer.BadParameter("release context must be a JSON object")
    evaluation = evaluate_release(
        (_metric(item) for item in raw_metrics),
        riverside_v1_policy(),
    )
    report = build_release_report(_release_context(raw_context), evaluation)
    write_release_report(output, report)
    typer.echo(
        json.dumps(
            {
                "decision": evaluation.decision.value,
                "output": str(output),
                "policy": evaluation.policy_id,
                "policy_version": evaluation.policy_version,
            },
            sort_keys=True,
        )
    )
    if evaluation.decision is not Decision.PROMOTE:
        raise typer.Exit(code=2)


@cli.command("report")
def report_command(
    report: Path = typer.Option(..., "--report", help="Evaluation release-report JSON."),
    output: Path | None = typer.Option(None, "--output", help="Optional Markdown destination."),
) -> None:
    """Validate a release report and render a concise human-readable summary."""

    payload = _load_json(report)
    if not isinstance(payload, dict):
        raise typer.BadParameter("release report must be a JSON object")
    validate_decision_consistency(payload)
    _validate_document(
        _project_root() / "contracts" / "v1" / "evaluation-release-report.schema.json",
        payload,
    )
    lines = [
        f"# Release report: {payload['release_id']}",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Version: `{payload['release_version']}`",
        f"- Source commit: `{payload['source_commit']}`",
        "",
        "| Domain | Passed | Failed |",
        "|---|---:|---:|",
    ]
    for domain, domain_metrics in payload["domains"].items():
        passed = sum(metric["status"] == "pass" for metric in domain_metrics)
        failed = sum(metric["status"] == "fail" for metric in domain_metrics)
        lines.append(f"| {domain} | {passed} | {failed} |")
    lines.extend(["", "## Decision reasons", ""])
    lines.extend(f"- {reason}" for reason in payload["decision_reasons"])
    rendered = "\n".join(lines) + "\n"
    if output is None:
        typer.echo(rendered, nl=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        typer.echo(json.dumps({"status": "written", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    cli()
