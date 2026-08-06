from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-cloud",
        action="store_true",
        default=False,
        help="run tests that contact authorized Azure or Databricks resources",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_cloud = config.getoption("--run-cloud") and os.getenv("RIVERSIDE_CLOUD_TESTS") == "1"
    production_allowed = os.getenv("RIVERSIDE_ALLOW_PRODUCTION_CLOUD_TESTS") == "1"
    target_environment = os.getenv("RIVERSIDE_ENVIRONMENT", "")
    for item in items:
        if "cloud" not in item.keywords:
            continue
        if not run_cloud:
            item.add_marker(
                pytest.mark.skip(
                    reason="cloud tests require --run-cloud and RIVERSIDE_CLOUD_TESTS=1"
                )
            )
        elif target_environment == "production" and not production_allowed:
            item.add_marker(
                pytest.mark.skip(
                    reason="production cloud tests require RIVERSIDE_ALLOW_PRODUCTION_CLOUD_TESTS=1"
                )
            )
