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