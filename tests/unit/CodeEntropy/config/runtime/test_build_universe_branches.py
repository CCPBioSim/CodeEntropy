from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_build_universe_no_force(runner):
    args = SimpleNamespace(
        top_traj_file=["tpr", "trr"],
        force_file=None,
        file_format=None,
        kcal_force_units=False,
    )
    uops = MagicMock()

    with patch("CodeEntropy.config.runtime.mda.Universe", return_value="U"):
        out = runner._build_universe(args, uops)

    assert out == "U"
    uops.merge_forces.assert_not_called()


def test_build_universe_with_force(runner):
    args = SimpleNamespace(
        top_traj_file=["tpr", "trr"],
        force_file="force",
        file_format="gro",
        kcal_force_units=True,
    )
    uops = MagicMock()
    uops.merge_forces.return_value = "U2"

    out = runner._build_universe(args, uops)

    assert out == "U2"
