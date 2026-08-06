from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
APIM_ROOT = PROJECT_ROOT / "apim"
FRAGMENT_ROOT = APIM_ROOT / "policies" / "fragments"

NAMED_VALUE_PATTERN = re.compile(r"\{\{([a-z0-9-]+)\}\}")
URL_PATTERN = re.compile(r"https://[^\s\"'<>]+", re.IGNORECASE)
FORBIDDEN_METRIC_DIMENSION_PARTS = {
    "actor",
    "completion",
    "document",
    "prompt",
    "request",
    "source",
    "tenant.id",
    "trace",
    "user",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_xml(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def xml_assets() -> list[Path]:
    return sorted((APIM_ROOT / "policies").rglob("*.xml"))


def fragment_manifest() -> list[dict[str, Any]]:
    manifest = load_json(FRAGMENT_ROOT / "manifest.json")
    return manifest["fragments"]


def declared_named_values() -> set[str]:
    document = load_json(APIM_ROOT / "parameters" / "named-values.json")
    return {item["name"] for item in document["named_values"]}


def referenced_named_values() -> set[str]:
    references: set[str] = set()
    for path in [*xml_assets(), APIM_ROOT / "backends" / "backends.bicep"]:
        references.update(NAMED_VALUE_PATTERN.findall(path.read_text(encoding="utf-8")))
    return references


def included_fragment_ids() -> dict[str, list[str]]:
    root = parse_xml(APIM_ROOT / "policies" / "api-policy.xml")
    sections: dict[str, list[str]] = {}
    for section in ("inbound", "backend", "outbound", "on-error"):
        node = root.find(section)
        if node is None:
            sections[section] = []
            continue
        sections[section] = [
            element.attrib["fragment-id"]
            for element in node.findall("include-fragment")
        ]
    return sections


def expected_fragment_ids() -> dict[str, list[str]]:
    sections: dict[str, list[tuple[int, str]]] = {}
    for item in fragment_manifest():
        sections.setdefault(item["section"], []).append((item["order"], item["id"]))
    return {
        section: [fragment_id for _, fragment_id in sorted(values)]
        for section, values in sections.items()
    }


def token_metric_dimensions() -> set[str]:
    root = parse_xml(FRAGMENT_ROOT / "token-governance.xml")
    metric = root.find("llm-emit-token-metric")
    if metric is None:
        return set()
    return {dimension.attrib["name"] for dimension in metric.findall("dimension")}


def forbidden_metric_dimensions() -> set[str]:
    return {
        dimension
        for dimension in token_metric_dimensions()
        if any(part in dimension.casefold() for part in FORBIDDEN_METRIC_DIMENSION_PARTS)
    }


def committed_urls() -> set[str]:
    urls: set[str] = set()
    for path in APIM_ROOT.rglob("*"):
        if path.is_file():
            urls.update(URL_PATTERN.findall(path.read_text(encoding="utf-8")))
    return urls


def scenario_cases() -> list[dict[str, Any]]:
    document = load_json(Path(__file__).with_name("policy-cases.json"))
    return document["cases"]


def assert_scenario_markers(case: dict[str, Any]) -> None:
    for evidence in case["static_evidence"]:
        path = REPOSITORY_ROOT / evidence["file"]
        text = path.read_text(encoding="utf-8")
        marker = evidence["contains"]
        assert marker in text, f"{case['id']}: {marker!r} not found in {path}"
