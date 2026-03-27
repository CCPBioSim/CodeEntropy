from dataclasses import dataclass
from pathlib import Path
import pytest


@dataclass(frozen=True)
class RegressionCase:
    system: str
    config_path: Path
    baseline_path: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_cases():
    configs_root = repo_root() / "tests/regression/configs"
    baselines_root = repo_root() / "tests/regression/baselines"

    cases = []

    for system_dir in sorted(configs_root.iterdir()):
        if not system_dir.is_dir():
            continue

        system = system_dir.name
        config_path = system_dir / "config.yaml"
        baseline_path = baselines_root / f"{system}.json"

        if not config_path.exists():
            continue

        cases.append(
            pytest.param(
                RegressionCase(system, config_path, baseline_path),
                id=system,
                marks=pytest.mark.slow if system != "dna" else (),
            )
        )

    return cases