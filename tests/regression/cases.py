from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True)
class RegressionCase:
    """
    Represents a single regression test case.

    A regression case corresponds to a specific system and a specific YAML
    configuration file, along with its associated baseline file.

    Attributes:
        system (str): Name of the system (e.g. "dna", "benzene").
        config_path (Path): Path to the YAML configuration file defining the scenario.
        baseline_path (Path): Path to the expected baseline JSON output.
    """

    system: str
    config_path: Path
    baseline_path: Path


def repo_root() -> Path:
    """
    Return the repository root directory.

    Returns:
        Path: Absolute path to the repository root.
    """
    return Path(__file__).resolve().parents[2]


def discover_cases() -> list[Any]:
    """
    Discover all regression test cases from the configs directory.

    This function scans the regression configuration directory structure and
    constructs a list of parametrized pytest cases. Each case corresponds to
    a single YAML configuration file and its associated baseline JSON file.

    Directory structure is expected to follow:

        tests/regression/configs/<system>/<config>.yaml
        tests/regression/baselines/<system>/<config>.json

    For each configuration file:
        - A `RegressionCase` instance is created.
        - A pytest parameter is generated with a unique ID.
        - Slow tests are automatically marked using `pytest.mark.slow`
          based on configuration naming conventions.

    Slow Test Heuristic:
        Configurations considered "slow" are those representing full-system
        or full-selection runs. These are explicitly identified by name
        (e.g. "default", "rad") and are marked with `pytest.mark.slow`.
        All other configurations (typically subset-based runs such as
        selection subsets) are treated as fast tests.

    Returns:
        list[Any]: A list of parametrized pytest cases, each wrapping a
        RegressionCase instance and optionally marked as slow.

    Notes:
        - Baseline files are not required to exist at discovery time.
        - Missing baselines will cause failures during test execution unless
          `--update-baselines` is used.
        - Test IDs are generated in the form: "{system}-{config_name}".
    """
    base_dir = Path(__file__).resolve().parent

    configs_root = base_dir / "configs"
    baselines_root = base_dir / "baselines"

    cases = []

    SLOW_CONFIGS = {"default", "rad"}

    for system_dir in sorted(configs_root.iterdir()):
        if not system_dir.is_dir():
            continue

        system = system_dir.name

        for config_path in sorted(system_dir.glob("*.yaml")):
            case_name = config_path.stem

            baseline_path = baselines_root / system / f"{case_name}.json"

            is_slow = case_name in SLOW_CONFIGS

            cases.append(
                pytest.param(
                    RegressionCase(system, config_path, baseline_path),
                    id=f"{system}-{case_name}",
                    marks=pytest.mark.slow if is_slow else (),
                )
            )

    return cases
