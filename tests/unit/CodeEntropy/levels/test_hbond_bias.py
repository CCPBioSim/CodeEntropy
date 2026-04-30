from unittest.mock import MagicMock, patch

import pytest

from CodeEntropy.levels.hbond_bias import HBondFactor
from CodeEntropy.levels.mda import UniverseOperations


def test_get_hbond_factor_no_hbond_possible():
    hbb = HBondFactor()
    universe = MagicMock()
    groups = {0: [0]}
    group_id = 0
    donors = []
    acceptors = []

    UniverseOperations.extract_fragment = MagicMock(side_effect="rep_mol")

    HBondFactor.get_possible_donors = MagicMock(side_effect=[[0, 0]])

    result = hbb.get_hbond_factor(universe, groups, group_id, donors, acceptors)

    assert result == 1


def test_get_hbond_factor_no_hbond_occurs():
    hbb = HBondFactor()
    universe = MagicMock()
    groups = {0: [0]}
    group_id = 0
    donors = []
    acceptors = []

    UniverseOperations.extract_fragment = MagicMock(side_effect="rep_mol")

    HBondFactor.get_possible_donors = MagicMock(side_effect=[[1, 1]])
    HBondFactor.get_hbond_info = MagicMock(side_effect=[[0, 0]])

    result = hbb.get_hbond_factor(universe, groups, group_id, donors, acceptors)

    assert result == 1


def test_get_hbond_factor():
    hbb = HBondFactor()
    universe = MagicMock()
    universe.trajectory.__len__.return_value = 4
    groups = {0: [0, 1]}
    group_id = 0
    donors = [1, 2]
    acceptors = [4, 5]

    UniverseOperations.extract_fragment = MagicMock(side_effect="rep_mol")

    HBondFactor.get_possible_donors = MagicMock(side_effect=[[2, 2]])
    HBondFactor.get_hbond_info = MagicMock(side_effect=[[1, 1]])

    result = hbb.get_hbond_factor(universe, groups, group_id, donors, acceptors)

    assert result == pytest.approx(0.878906)


def test_get_hbond_factor_2():
    hbb = HBondFactor()
    universe = MagicMock()
    universe.trajectory.__len__.return_value = 20
    groups = {0: [0, 1]}
    group_id = 0
    donors = [1, 2]
    acceptors = [4, 5]

    UniverseOperations.extract_fragment = MagicMock(side_effect="rep_mol")

    HBondFactor.get_possible_donors = MagicMock(side_effect=[[2, 1]])
    HBondFactor.get_hbond_info = MagicMock(side_effect=[[1, 1]])

    result = hbb.get_hbond_factor(universe, groups, group_id, donors, acceptors)

    assert result == pytest.approx(0.9628125)


def test_too_many_hbond():
    hbb = HBondFactor()
    universe = MagicMock()
    universe.trajectory.__len__.return_value = 1
    groups = {0: [0]}
    group_id = 0
    donors = [1, 2]
    acceptors = [4, 5]

    UniverseOperations.extract_fragment = MagicMock(side_effect="rep_mol")

    HBondFactor.get_possible_donors = MagicMock(side_effect=[[1, 1]])
    HBondFactor.get_hbond_info = MagicMock(side_effect=[[2, 2]])

    result = hbb.get_hbond_factor(universe, groups, group_id, donors, acceptors)

    assert result == 0.5


def test_max_donor_zero():
    hbb = HBondFactor()
    universe = MagicMock()
    universe.trajectory.__len__.return_value = 20
    groups = {0: [0, 1]}
    group_id = 0
    donors = [1, 2]
    acceptors = [4, 5]

    UniverseOperations.extract_fragment = MagicMock(side_effect="rep_mol")

    HBondFactor.get_possible_donors = MagicMock(side_effect=[[0, 1]])
    HBondFactor.get_hbond_info = MagicMock(side_effect=[[1, 1]])

    result = hbb.get_hbond_factor(universe, groups, group_id, donors, acceptors)

    assert result == 0.975


def test_get_possible_donors_N():
    hbb = HBondFactor()
    universe = MagicMock()
    mol = MagicMock()

    class _FakeHBA(universe):
        def guess_acceptors():
            return "acceptors"

        def guess_hydrogens():
            return "hydrogens"

    a1 = MagicMock()
    a1.element = MagicMock(side_effect="N")
    mol.select_atoms = MagicMock(side_effect=([a1, a1], [1, 1, 1]))

    with patch("CodeEntropy.levels.hbond_bias.HBA", _FakeHBA):
        result = hbb.get_possible_donors(mol)

    assert result == (3, 2)


def test_get_possible_donors_O():
    hbb = HBondFactor()
    universe = MagicMock()
    mol = MagicMock()

    class _FakeHBA(universe):
        def guess_acceptors():
            return "acceptors"

        def guess_hydrogens():
            return "hydrogens"

    a1 = MagicMock()
    a1.element.side_effect = "O"
    mol.select_atoms = MagicMock(side_effect=([a1, a1], [1, 1, 1]))

    with patch("CodeEntropy.levels.hbond_bias.HBA", _FakeHBA):
        result = hbb.get_possible_donors(mol)

    assert result == (3, 4)


def test_get_hbond_info():
    hbf = HBondFactor()
    universe = MagicMock()
    universe.atoms.fragments[0].indices.return_value = [0, 1, 2]
    universe.atoms.fragments[1].indices.return_value = [3, 4, 5]

    mols = [0, 1]

    donors = [0, 3]
    acceptors = [1, 4]

    n_donors, n_acceptors = hbf.get_hbond_info(universe, mols, donors, acceptors)

    assert n_donors == 2
    assert n_acceptors == 2
