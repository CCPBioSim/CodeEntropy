from dataclasses import dataclass
from pathlib import Path

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


def discover_cases():
    """
    Discover all regression test cases from the configs directory.

    Iterates over each system directory under `tests/regression/configs`,
    and collects all YAML configuration files. Each configuration file is
    paired with a corresponding baseline JSON file.

    Returns:
        list[pytest.Param]: A list of parametrized pytest cases, each wrapping
        a RegressionCase instance.

    Notes:
        - Cases are automatically marked as slow unless the system is "dna".
        - Baseline files are not required to exist at discovery time.
    """
    base_dir = Path(__file__).resolve().parent

    configs_root = base_dir / "configs"
    baselines_root = base_dir / "baselines"

    cases = []

    for system_dir in sorted(configs_root.iterdir()):
        if not system_dir.is_dir():
            continue

        system = system_dir.name

        for config_path in sorted(system_dir.glob("*.yaml")):
            case_name = config_path.stem

            baseline_path = baselines_root / system / f"{case_name}.json"

            # DO NOT skip if baseline is missing
            cases.append(
                pytest.param(
                    RegressionCase(system, config_path, baseline_path),
                    id=f"{system}-{case_name}",
                    marks=pytest.mark.slow if system != "dna" else (),
                )
            )

    return cases
