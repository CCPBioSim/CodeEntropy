from __future__ import annotations

import os
import random

import numpy as np
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register custom command-line options for pytest.
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
    parser.addoption(
        "--system",
        action="append",
        default=None,
        help="Run only tests for specified system(s). Can be passed multiple times.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """
    Register markers and enforce deterministic behavior.
    """
    config.addinivalue_line("markers", "regression: end-to-end regression tests")
    config.addinivalue_line("markers", "slow: long-running tests (20-30+ minutes)")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = "0"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Modify collected test items to:
    1. Skip slow tests unless --run-slow is provided
    2. Filter tests by --system if specified
    """
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(
            reason="Skipped slow test (use --run-slow to run)."
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    selected_systems = config.getoption("--system")
    if not selected_systems:
        return

    filtered_items = []

    for item in items:
        callspec = getattr(item, "callspec", None)

        case = None
        if callspec is not None:
            case = callspec.params.get("case")

        if case is None:
            filtered_items.append(item)
            continue

        if hasattr(case, "system") and case.system in selected_systems:
            filtered_items.append(item)

    items[:] = filtered_items
