from pathlib import Path

import numpy as np
import pytest
import yaml

import tests.regression.helpers as Helpers
from CodeEntropy.config.runtime import CodeEntropyRunner
from CodeEntropy.levels.mda import UniverseOperations
from CodeEntropy.levels.search import Search

# some dummy atom positions
a = np.array([0, 0, 1])
b = np.array([0, 1, 0])
c = np.array([1, 0, 0])
d = np.array([0, 1, 1])
e = np.array([0, 11, 11])
dimensions = np.array([10, 10, 10])

DEFAULT_TESTDATA_BASE_URL = "https://www.ccpbiosim.ac.uk/file-store/codeentropy-testing"


def test_get_RAD_neighbors(tmp_path: Path):
    """
    Args:
        tmp_path: Pytest provided temporatry directory
    """
    args = {}
    search = Search()
    system = "methane"
    repo_root = Path(__file__).resolve().parents[4]
    config_path = (
        repo_root / "tests" / "regression" / "configs" / system / "config.yaml"
    )

    tmp_path.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config must parse to a dict. Got {type(raw)} from {config_path}"
        )

    cooked = Helpers._abspathify_config_paths(raw, base_dir=config_path.parent)
    required: list[Path] = []
    run1 = cooked.get("run1")
    if isinstance(run1, dict):
        ff = run1.get("force_file")
        if isinstance(ff, str) and ff:
            required.append(Path(ff))
        for p in run1.get("top_traj_file") or []:
            if isinstance(p, str) and p:
                required.append(Path(p))

    if required:
        Helpers.ensure_testdata_for_system(system, required_paths=required)

    runner = CodeEntropyRunner(tmp_path)
    parser = runner._config_manager.build_parser()
    args, _ = parser.parse_known_args()
    args.end = run1.get("end")
    args.top_traj_file = run1.get("top_traj_file")
    args.file_format = run1.get("file_format")
    assert args.end == 1

    universe_operations = UniverseOperations()
    universe = CodeEntropyRunner._build_universe(args, universe_operations)

    neighbors = search.get_RAD_neighbors(universe=universe, mol_id=0)

    assert neighbors == [151, 3, 75, 219, 229, 488, 460, 118, 230, 326]


def test_get_grid_neighbors(tmp_path: Path):
    """
    Args:
        tmp_path: Pytest provided temporatry directory
    """
    args = {}
    search = Search()
    system = "methane"
    repo_root = Path(__file__).resolve().parents[4]
    config_path = (
        repo_root / "tests" / "regression" / "configs" / system / "config.yaml"
    )

    tmp_path.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config must parse to a dict. Got {type(raw)} from {config_path}"
        )

    cooked = Helpers._abspathify_config_paths(raw, base_dir=config_path.parent)
    required: list[Path] = []
    run1 = cooked.get("run1")
    if isinstance(run1, dict):
        ff = run1.get("force_file")
        if isinstance(ff, str) and ff:
            required.append(Path(ff))
        for p in run1.get("top_traj_file") or []:
            if isinstance(p, str) and p:
                required.append(Path(p))

    if required:
        Helpers.ensure_testdata_for_system(system, required_paths=required)

    runner = CodeEntropyRunner(tmp_path)
    parser = runner._config_manager.build_parser()
    args, _ = parser.parse_known_args()
    args.end = run1.get("end")
    args.top_traj_file = run1.get("top_traj_file")
    args.file_format = run1.get("file_format")
    assert args.end == 1

    universe_operations = UniverseOperations()
    universe = CodeEntropyRunner._build_universe(args, universe_operations)

    neighbors = search.get_grid_neighbors(
        universe=universe, mol_id=0, highest_level="united_atom"
    )

    assert (neighbors == [151, 3, 75, 219]).all


def test_get_angle():
    search = Search()
    result1 = search.get_angle(a, b, c, dimensions)
    result2 = search.get_angle(a, b, d, dimensions)

    assert result1 == 0.5
    assert result2 == pytest.approx(0.7071067811865477)


def test_angle_boundary_conditions():
    search = Search()

    result = search.get_angle(a, b, e, dimensions)

    assert result == pytest.approx(0.7071067811865477)


def test_distance():
    search = Search()
    distance1 = search.get_distance(a, b, dimensions)
    distance2 = search.get_distance(a, d, dimensions)
    distance3 = search.get_distance(c, d, dimensions)

    assert distance1 == pytest.approx(1.4142135623730951)
    assert distance2 == 1.0
    assert distance3 == pytest.approx(1.7320508075688772)


def test_distance_boundary_conditions():
    search = Search()

    distance4 = search.get_distance(c, e, dimensions)

    assert distance4 == pytest.approx(1.7320508075688772)
