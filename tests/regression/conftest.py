from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register custom command-line options for pytest.

    Adds options to control regression test execution, baseline updates,
    and debugging output.

    Args:
        parser (pytest.Parser): Pytest CLI parser.
    """
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
    """
    Register custom pytest markers.

    Args:
        config (pytest.Config): Pytest configuration object.
    """
    config.addinivalue_line("markers", "regression: end-to-end regression tests")
    config.addinivalue_line("markers", "slow: long-running tests (20-30+ minutes)")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Modify collected test items to skip slow tests unless explicitly enabled.

    Only tests marked with `@pytest.mark.slow` are skipped when the
    `--run-slow` flag is not provided.

    Args:
        config (pytest.Config): Pytest configuration object.
        items (list[pytest.Item]): Collected test items.
    """
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="Skipped slow test (use --run-slow to run).")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
