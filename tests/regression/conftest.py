from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow regression tests (20-30+ minutes).",
    )
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Overwrite regression baselines with the newly produced outputs.",
    )
    parser.addoption(
        "--codeentropy-debug",
        action="store_true",
        default=False,
        help="Print CodeEntropy stdout/stderr and paths for easier debugging.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "regression: end-to-end regression tests")
    config.addinivalue_line("markers", "slow: long-running tests (20-30+ minutes)")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Regression tests are collected and runnable by default.

    Only @pytest.mark.slow tests are skipped unless you pass --run-slow.
    """
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="Skipped slow test (use --run-slow to run).")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
