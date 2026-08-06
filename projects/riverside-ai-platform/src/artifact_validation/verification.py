from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from .errors import (
    ArtifactResolutionError,
    CompatibilityError,
    DigestMismatchError,
    ManifestValidationError,
)
from .models import (
    ArtifactPaths,
    Digest,
    ModelReleaseManifest,
    RuntimeCompatibility,
    VerifiedRelease,
)


class ArtifactResolver:
    """Resolve portable manifest URIs without permitting network or root escape."""

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root.resolve(strict=True)

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    def resolve(self, uri: object) -> Path:
        parsed = urlsplit(str(uri))
        if parsed.query or parsed.fragment:
            raise ArtifactResolutionError("artifact URIs must not contain query strings or fragments")

        if parsed.scheme == "repo":
            relative = Path(unquote(parsed.netloc), unquote(parsed.path.lstrip("/")))
            candidate = self._artifact_root / relative
        elif parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                raise ArtifactResolutionError("remote file URIs are not supported")
            candidate = Path(unquote(parsed.path))
        else:
            raise ArtifactResolutionError(
                f"unsupported artifact URI scheme: {parsed.scheme or '<missing>'}"
            )

        return self.approve_local_file(candidate)

    def approve_local_file(self, candidate: Path) -> Path:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ArtifactResolutionError("referenced artifact is unavailable") from exc
        if not resolved.is_relative_to(self._artifact_root):
            raise ArtifactResolutionError("artifact URI resolves outside the approved artifact root")
        if not resolved.is_file():
            raise ArtifactResolutionError("artifact URI must resolve to a regular file")
        return resolved


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact:
            while chunk := artifact.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactResolutionError("artifact could not be read") from exc
    return digest.hexdigest()


def verify_digest(path: Path, expected: Digest) -> None:
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected.value):
        raise DigestMismatchError(f"sha256 digest mismatch for {path.name}")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"{label} is not a readable JSON document") from exc
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{label} must be a JSON object")
    return value


class ReleaseVerifier:
    def __init__(self, resolver: ArtifactResolver, runtime: RuntimeCompatibility) -> None:
        self._resolver = resolver
        self._runtime = runtime

    def verify_file(self, manifest_path: Path) -> VerifiedRelease:
        try:
            manifest_text = manifest_path.read_text(encoding="utf-8")
            manifest = ModelReleaseManifest.model_validate_json(manifest_text)
        except (OSError, UnicodeError, ValidationError) as exc:
            raise ManifestValidationError("release manifest does not satisfy contract v1") from exc
        return self.verify(manifest)

    def verify(self, manifest: ModelReleaseManifest) -> VerifiedRelease:
        self._verify_runtime(manifest)
        if manifest.evaluation.decision != "promote":
            raise CompatibilityError("release evaluation decision is not promote")

        adapter = self._resolver.resolve(manifest.adapter.uri)
        tokenizer = self._resolver.resolve(manifest.tokenizer.uri)
        training_manifest = self._resolver.resolve(manifest.training_provenance.manifest_uri)
        evaluation_report = self._resolver.resolve(manifest.evaluation.report_uri)

        verify_digest(adapter, manifest.adapter.digest)
        verify_digest(tokenizer, manifest.tokenizer.digest)
        verify_digest(training_manifest, manifest.training_provenance.manifest_digest)
        verify_digest(evaluation_report, manifest.evaluation.report_digest)

        self._verify_training_provenance(manifest, training_manifest)
        adapter_config = self._resolver.approve_local_file(adapter.parent / "adapter_config.json")
        self._verify_adapter_config(manifest, adapter_config)
        self._verify_evaluation_report(manifest, evaluation_report)

        return VerifiedRelease(
            manifest=manifest,
            paths=ArtifactPaths(
                adapter=str(adapter),
                tokenizer=str(tokenizer),
                training_manifest=str(training_manifest),
                evaluation_report=str(evaluation_report),
            ),
        )

    def _verify_runtime(self, manifest: ModelReleaseManifest) -> None:
        declared = manifest.serving_runtime
        expected = self._runtime
        mismatches = [
            declared.name != expected.name,
            declared.version != expected.version,
            declared.interface_version != expected.interface_version,
            manifest.model_profile != expected.model_profile,
            manifest.precision != expected.precision,
            expected.model_profile not in declared.compatible_model_profiles,
            expected.precision not in declared.supported_precisions,
            manifest.base_model.id != expected.base_model_id,
            manifest.base_model.revision != expected.base_model_revision,
            manifest.adapter.type != expected.adapter_type,
            manifest.tokenizer.revision != manifest.base_model.revision,
        ]
        if any(mismatches):
            raise CompatibilityError("release is incompatible with the configured serving runtime")

    def _verify_training_provenance(
        self, manifest: ModelReleaseManifest, training_manifest_path: Path
    ) -> None:
        training = _load_json_object(training_manifest_path, "training manifest")
        model = training.get("model")
        if not isinstance(model, dict):
            raise CompatibilityError("training manifest has no model provenance")
        if (
            model.get("id") != manifest.base_model.id
            or model.get("revision") != manifest.base_model.revision
            or training.get("stage") != manifest.adapter.stage
        ):
            raise CompatibilityError("training provenance does not match the release manifest")

    def _verify_adapter_config(
        self, manifest: ModelReleaseManifest, adapter_config_path: Path
    ) -> None:
        adapter_config = _load_json_object(adapter_config_path, "adapter configuration")
        revision = adapter_config.get("revision")
        if (
            adapter_config.get("peft_type") != "LORA"
            or adapter_config.get("task_type") != "CAUSAL_LM"
            or adapter_config.get("base_model_name_or_path") != manifest.base_model.id
            or (revision is not None and revision != manifest.base_model.revision)
        ):
            raise CompatibilityError("LoRA adapter configuration does not match the release manifest")

    def _verify_evaluation_report(
        self, manifest: ModelReleaseManifest, evaluation_report_path: Path
    ) -> None:
        report = _load_json_object(evaluation_report_path, "evaluation report")
        if (
            report.get("contract_version") != "1.0.0"
            or report.get("kind") != "evaluation_release_report"
            or report.get("release_id") != manifest.release_id
            or report.get("release_version") != manifest.version
            or report.get("source_commit") != manifest.source_commit
            or report.get("decision") != "promote"
        ):
            raise CompatibilityError("evaluation report does not match the promoted release")

        domains = report.get("domains")
        if not isinstance(domains, dict):
            raise CompatibilityError("evaluation report has no domain evidence")
        for threshold in manifest.evaluation.thresholds:
            metrics = domains.get(threshold.domain)
            if not isinstance(metrics, list):
                raise CompatibilityError("evaluation report is missing a required domain")
            matching = [
                metric
                for metric in metrics
                if isinstance(metric, dict) and metric.get("metric_id") == threshold.metric
            ]
            if not matching or any(metric.get("status") != "pass" for metric in matching):
                raise CompatibilityError("evaluation threshold lacks passing evidence")
            if not any(
                isinstance(metric.get("threshold"), dict)
                and metric["threshold"].get("operator") == threshold.operator
                and metric["threshold"].get("value") == threshold.value
                for metric in matching
            ):
                raise CompatibilityError("evaluation threshold differs from retained evidence")
