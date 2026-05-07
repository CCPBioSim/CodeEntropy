from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from tests.regression.cases import discover_cases
from tests.regression.helpers import (
    _abspathify_config_paths,
    ensure_testdata_for_system,
)


def unwrap_case(case_or_param: Any) -> Any:
    if hasattr(case_or_param, "values"):
        values = case_or_param.values
        if values:
            return values[0]
    return case_or_param


def required_paths_for_case(case: Any) -> list[Path]:
    raw = yaml.safe_load(case.config_path.read_text())
    cooked = _abspathify_config_paths(raw, base_dir=case.config_path.parent)

    required: list[Path] = []
    run1 = cooked.get("run1")

    if isinstance(run1, dict):
        ff = run1.get("force_file")
        if isinstance(ff, str) and ff:
            required.append(Path(ff))

        for p in run1.get("top_traj_file") or []:
            if isinstance(p, str) and p:
                required.append(Path(p))

    return required


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", action="append", default=None)
    args = parser.parse_args()

    selected = set(args.system or [])

    for case_or_param in discover_cases():
        case = unwrap_case(case_or_param)

        if selected and case.system not in selected:
            continue

        required = required_paths_for_case(case)
        if required:
            ensure_testdata_for_system(case.system, required_paths=required)

    print("Regression test data prepared.")


if __name__ == "__main__":
    main()
