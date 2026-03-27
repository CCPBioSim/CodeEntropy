from unittest.mock import MagicMock

from CodeEntropy.levels.hbond_bias import HBondBias
from CodeEntropy.levels.mda import UniverseOperations


def test_get_hbond_bias_no_hbond():
    hbb = HBondBias()
    universe = MagicMock()
    groups = {0: [0]}
    group_id = 0
    donors = []
    acceptors = []

    UniverseOperations.extract_fragment = MagicMock(side_effect="rep_mol")

    HBondBias.get_possible_donors = MagicMock(side_effect=[[0, 0]])

    result = hbb.get_hbond_bias(universe, groups, group_id, donors, acceptors)

    assert result == (1, 1)


def test_get_hbond_bias():
    hbb = HBondBias()
    universe = MagicMock()
    universe.trajectory.__len__.return_value = 4
    groups = {0: [0, 1]}
    group_id = 0
    donors = [1, 2]
    acceptors = [4, 5]

    UniverseOperations.extract_fragment = MagicMock(side_effect="rep_mol")

    HBondBias.get_possible_donors = MagicMock(side_effect=[[2, 2]])
    HBondBias.get_hbond_info = MagicMock(side_effect=[[1, 1]])

    result = hbb.get_hbond_bias(universe, groups, group_id, donors, acceptors)

    assert result == (0.25, 1.0)


def test_get_hbond_bias_2():
    hbb = HBondBias()
    universe = MagicMock()
    universe.trajectory.__len__.return_value = 20
    groups = {0: [0, 1]}
    group_id = 0
    donors = [1, 2]
    acceptors = [4, 5]

    UniverseOperations.extract_fragment = MagicMock(side_effect="rep_mol")

    HBondBias.get_possible_donors = MagicMock(side_effect=[[2, 2]])
    HBondBias.get_hbond_info = MagicMock(side_effect=[[1, 1]])

    result = hbb.get_hbond_bias(universe, groups, group_id, donors, acceptors)

    assert result == (0.5, 2.0)
