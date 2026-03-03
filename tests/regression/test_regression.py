from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from tests.regression.helpers import run_codeentropy_with_config


def _group_index(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return the groups mapping from a regression payload.

    Args:
        payload: Parsed JSON payload.

    Returns:
        Mapping of group id to group data.

    Raises:
        TypeError: If payload["groups"] is not a dict.
    """
    groups = payload.get("groups", {})
    if not isinstance(groups, dict):
        raise TypeError("payload['groups'] must be a dict")
    return groups


def _compare_grouped(
    *,
    got_payload: Dict[str, Any],
    baseline_payload: Dict[str, Any],
    rtol: float,
    atol: float,
) -> None:
    """Compare grouped regression outputs against baseline values.

    Args:
        got_payload: Newly produced payload.
        baseline_payload: Baseline payload.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Raises:
        AssertionError: If any required group/component differs from baseline.
    """
    got_groups = _group_index(got_payload)
    base_groups = _group_index(baseline_payload)

    missing_groups = sorted(set(base_groups.keys()) - set(got_groups.keys()))
    assert not missing_groups, f"Missing groups in output: {missing_groups}"

    mismatches: list[str] = []

    for gid, base_g in base_groups.items():
        got_g = got_groups[gid]

        base_components = base_g.get("components", {})
        got_components = got_g.get("components", {})

        if not isinstance(base_components, dict) or not isinstance(
            got_components, dict
        ):
            mismatches.append(f"group {gid}: components must be dicts")
            continue

        missing_keys = sorted(set(base_components.keys()) - set(got_components.keys()))
        if missing_keys:
            mismatches.append(f"group {gid}: missing component keys: {missing_keys}")
            continue

        for k, expected in base_components.items():
            actual = got_components[k]
            try:
                np.testing.assert_allclose(
                    float(actual), float(expected), rtol=rtol, atol=atol
                )
            except AssertionError:
                mismatches.append(
                    f"group {gid} component {k}: expected={expected} got={actual}"
                )

        if "total" in base_g:
            try:
                np.testing.assert_allclose(
                    float(got_g.get("total", 0.0)),
                    float(base_g["total"]),
                    rtol=rtol,
                    atol=atol,
                )
            except AssertionError:
                mismatches.append(
                    f"group {gid} total: expected={base_g['total']} "
                    f"got={got_g.get('total')}"
                )

    assert not mismatches, "Mismatches:\n" + "\n".join("  " + m for m in mismatches)


@pytest.mark.regression
@pytest.mark.parametrize(
    "system",
    [
        pytest.param("benzaldehyde", marks=pytest.mark.slow),
        pytest.param("benzene", marks=pytest.mark.slow),
        pytest.param("cyclohexane", marks=pytest.mark.slow),
        "dna",
        pytest.param("ethyl-acetate", marks=pytest.mark.slow),
        "methane",
        "methanol",
        pytest.param("octonol", marks=pytest.mark.slow),
        "water",
    ],
)
def test_regression_matches_baseline(
    tmp_path: Path, system: str, request: pytest.FixtureRequest
) -> None:
    """Run a regression test for one system and compare to its baseline.

    Args:
        tmp_path: Pytest-provided temporary directory.
        system: System name parameter.
        request: Pytest request fixture for reading CLI options.
    """
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root / "tests" / "regression" / "configs" / system / "config.yaml"
    )
    baseline_path = repo_root / "tests" / "regression" / "baselines" / f"{system}.json"

    assert config_path.exists(), f"Missing config: {config_path}"
    assert baseline_path.exists(), f"Missing baseline: {baseline_path}"

    baseline_payload = json.loads(baseline_path.read_text())
    run = run_codeentropy_with_config(workdir=tmp_path, config_src=config_path)

    if request.config.getoption("--codeentropy-debug"):
        print("\n[CodeEntropy regression debug]")
        print("workdir:", run.workdir)
        print("job_dir:", run.job_dir)
        print("output_json:", run.output_json)
        print("payload copy saved:", run.workdir / "codeentropy_output.json")

    if request.config.getoption("--update-baselines"):
        baseline_path.write_text(json.dumps(run.payload, indent=2))
        pytest.skip(f"Baseline updated for {system}: {baseline_path}")

    _compare_grouped(
        got_payload=run.payload,
        baseline_payload=baseline_payload,
        rtol=1e-9,
        atol=1e-8,
    )
