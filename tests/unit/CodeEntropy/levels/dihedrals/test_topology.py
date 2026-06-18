from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from CodeEntropy.levels.dihedrals.topology import DihedralTopologyDiscovery


class _AddableAG:
    """Minimal addable AtomGroup test double."""

    def __init__(self, name: str) -> None:
        """Initialize the fake AtomGroup.

        Args:
            name: Human-readable identifier used in composed names.
        """
        self.name = name

    def __add__(self, other: _AddableAG) -> _AddableAG:
        """Return a composed fake AtomGroup.

        Args:
            other: Fake AtomGroup to combine with this object.

        Returns:
            New fake AtomGroup containing a composed name.
        """
        return _AddableAG(f"({self.name}+{other.name})")


class _TopologyDiscovery(DihedralTopologyDiscovery):
    """Concrete topology-discovery helper for unit tests."""

    def __init__(self, universe_operations: MagicMock) -> None:
        """Initialize the test helper.

        Args:
            universe_operations: Mock universe-operation adapter.
        """
        self._universe_operations = universe_operations


def test_select_heavy_residue_builds_expected_selections():
    uops = MagicMock()
    helper = _TopologyDiscovery(universe_operations=uops)

    mol = MagicMock()
    mol.residues = [MagicMock()]
    mol.residues[0].atoms.indices = np.array([10, 11, 12], dtype=int)
    uops.select_atoms.side_effect = ["residue_atoms", "heavy_atoms"]

    out = helper._select_heavy_residue(mol, res_id=0)

    assert out == "heavy_atoms"
    assert uops.select_atoms.call_args_list == [
        ((mol, "index 10:12"),),
        (("residue_atoms", "prop mass > 1.1"),),
    ]


def test_get_dihedrals_united_atom_collects_atoms_from_dihedral_objects():
    helper = _TopologyDiscovery(universe_operations=MagicMock())

    d0 = MagicMock()
    d0.atoms = "A0"
    d1 = MagicMock()
    d1.atoms = "A1"

    container = MagicMock()
    container.dihedrals = [d0, d1]

    assert helper._get_dihedrals(container, level="united_atom") == ["A0", "A1"]


def test_get_dihedrals_residue_returns_empty_when_less_than_four_residues():
    helper = _TopologyDiscovery(universe_operations=MagicMock())

    mol = MagicMock()
    mol.residues = [MagicMock(), MagicMock(), MagicMock()]
    mol.select_atoms = MagicMock()

    assert helper._get_dihedrals(mol, level="residue") == []
    mol.select_atoms.assert_not_called()


def test_get_dihedrals_residue_builds_one_dihedral_when_four_residues():
    helper = _TopologyDiscovery(universe_operations=MagicMock())

    mol = MagicMock()
    mol.residues = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    mol.select_atoms = MagicMock(
        side_effect=[
            _AddableAG("a1"),
            _AddableAG("a2"),
            _AddableAG("a3"),
            _AddableAG("a4"),
        ]
    )

    out = helper._get_dihedrals(mol, level="residue")

    assert len(out) == 1
    assert isinstance(out[0], _AddableAG)
    assert mol.select_atoms.call_count == 4


def test_discover_group_dihedral_topology_builds_one_entry_per_molecule():
    uops = MagicMock()
    helper = _TopologyDiscovery(universe_operations=uops)

    mol0 = MagicMock()
    mol0.residues = [MagicMock(), MagicMock()]
    mol1 = MagicMock()
    mol1.residues = [MagicMock(), MagicMock()]
    uops.extract_fragment.side_effect = [mol0, mol1]

    helper._select_heavy_residue = MagicMock(
        side_effect=["heavy0", "heavy1", "heavy2", "heavy3"]
    )
    helper._get_dihedrals = MagicMock(
        side_effect=[
            ["ua0r0"],
            ["ua0r1"],
            ["res0"],
            ["ua1r0"],
            ["ua1r1"],
            ["res1"],
        ]
    )

    topologies = helper._discover_group_dihedral_topology(
        data_container="universe",
        group_id=3,
        molecules=[7, 8],
        level_list=["united_atom", "residue"],
    )

    assert [topology.molecule_id for topology in topologies] == [7, 8]
    assert [topology.molecule_order for topology in topologies] == [0, 1]
    assert topologies[0].group_id == 3
    assert topologies[0].ua_dihedrals_by_residue == {0: ["ua0r0"], 1: ["ua0r1"]}
    assert topologies[0].residue_dihedrals == ["res0"]
    assert topologies[1].ua_dihedrals_by_residue == {0: ["ua1r0"], 1: ["ua1r1"]}
    assert topologies[1].residue_dihedrals == ["res1"]
