from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.regression.helpers import run_codeentropy_with_config
from tests.regression.cases import discover_cases


def _group_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the groups mapping from a regression payload."""
    groups = payload.get("groups", {})
    if not isinstance(groups, dict):
        raise TypeError("payload['groups'] must be a dict")
    return groups


def _compare_grouped(
    *,
    got_payload: dict[str, Any],
    baseline_payload: dict[str, Any],
    rtol: float,
    atol: float,
) -> None:
    """Compare grouped regression outputs against baseline values."""
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
@pytest.mark.parametrize("case", discover_cases())
def test_regression_matches_baseline(
    tmp_path: Path, case, request: pytest.FixtureRequest
) -> None:
    """Run a regression test for one system and compare to its baseline."""
    system = case.system
    config_path = case.config_path
    baseline_path = case.baseline_path

    assert config_path.exists(), f"Missing config: {config_path}"
    assert baseline_path.exists(), f"Missing baseline: {baseline_path}"

    baseline_payload = json.loads(baseline_path.read_text())

    run = run_codeentropy_with_config(
        workdir=tmp_path,
        config_src=config_path,
    )

    if request.config.getoption("--codeentropy-debug"):
        print("\n[CodeEntropy regression debug]")
        print("system:", system)
        print("workdir:", run.workdir)
        print("job_dir:", run.job_dir)
        print("output_json:", run.output_json)
        print("payload copy saved:", run.workdir / "codeentropy_output.json")

    if request.config.getoption("--update-baselines"):
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(run.payload, indent=2))
        pytest.skip(f"Baseline updated for {system}: {baseline_path}")

    _compare_grouped(
        got_payload=run.payload,
        baseline_payload=baseline_payload,
        rtol=1e-9,
        atol=0.5,
    )
