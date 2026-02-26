from unittest.mock import MagicMock

import numpy as np

from CodeEntropy.levels.nodes.beads import BuildBeadsNode


def _bead(indices, heavy_resindex=None, empty=False):
    b = MagicMock()
    b.__len__.return_value = 0 if empty else len(indices)
    b.indices = np.asarray(indices, dtype=int)

    heavy = MagicMock()
    if heavy_resindex is None:
        heavy.__len__.return_value = 0
        heavy.__iter__.return_value = iter([])
    else:
        a0 = MagicMock()
        a0.resindex = int(heavy_resindex)
        heavy.__len__.return_value = 1
        heavy.__getitem__.side_effect = lambda i: a0
        heavy.__iter__.return_value = iter([a0])

    b.select_atoms.return_value = heavy
    return b


def test_build_beads_node_groups_united_atom_beads_into_local_residue_buckets():
    r0 = MagicMock()
    r0.resindex = 10
    r1 = MagicMock()
    r1.resindex = 11

    mol = MagicMock()
    mol.residues = [r0, r1]

    ua0 = _bead([1, 2], heavy_resindex=10)
    ua1 = _bead([3], heavy_resindex=11)
    ua_empty = _bead([], heavy_resindex=10, empty=True)

    hier = MagicMock()
    hier.get_beads.side_effect = lambda m, lvl: (
        [ua0, ua1, ua_empty] if lvl == "united_atom" else []
    )

    node = BuildBeadsNode(hierarchy=hier)

    u = MagicMock()
    u.atoms = MagicMock()
    u.atoms.fragments = [mol]

    shared = {"reduced_universe": u, "levels": [["united_atom"]]}

    out = node.run(shared)
    beads = out["beads"]

    assert (0, "united_atom", 0) in beads
    assert (0, "united_atom", 1) in beads
    assert len(beads[(0, "united_atom", 0)]) == 1
    assert len(beads[(0, "united_atom", 1)]) == 1

    np.testing.assert_array_equal(beads[(0, "united_atom", 0)][0], np.array([1, 2]))
    np.testing.assert_array_equal(beads[(0, "united_atom", 1)][0], np.array([3]))


def test_add_residue_beads_logs_error_if_none_kept(caplog):
    hier = MagicMock()
    # returns one empty bead -> skipped -> kept stays 0
    empty_bead = MagicMock()
    empty_bead.__len__.return_value = 0
    hier.get_beads.return_value = [empty_bead]

    node = BuildBeadsNode(hierarchy=hier)

    beads = {}
    mol = MagicMock()
    mol.residues = [MagicMock()]

    node._add_residue_beads(beads=beads, mol_id=0, mol=mol)

    assert (0, "residue") in beads
    assert beads[(0, "residue")] == []
    assert any("No residue beads kept" in rec.message for rec in caplog.records)


def test_infer_local_residue_id_returns_zero_if_no_heavy_atoms():
    mol = MagicMock()
    mol.residues = [MagicMock(resindex=10), MagicMock(resindex=11)]

    bead = MagicMock()
    heavy = MagicMock()
    heavy.__len__.return_value = 0
    bead.select_atoms.return_value = heavy

    out = BuildBeadsNode._infer_local_residue_id(mol=mol, bead=bead)
    assert out == 0


def test_infer_local_residue_id_returns_zero_if_resindex_not_found():
    mol = MagicMock()
    mol.residues = [MagicMock(resindex=10), MagicMock(resindex=11)]

    bead = MagicMock()
    heavy = MagicMock()
    heavy.__len__.return_value = 1
    heavy0 = MagicMock(resindex=999)
    heavy.__getitem__.return_value = heavy0
    bead.select_atoms.return_value = heavy

    out = BuildBeadsNode._infer_local_residue_id(mol=mol, bead=bead)
    assert out == 0


def test_build_beads_node_skips_when_no_levels():
    """
    Covers: early return when levels missing (92/95 style guard branches)
    """
    node = BuildBeadsNode(hierarchy=MagicMock())
    out = node.run({"reduced_universe": MagicMock(), "levels": []})
    assert out["beads"] == {}


def test_build_beads_node_residue_level_adds_residue_beads():
    """
    Covers: residue path + _add_residue_beads bookkeeping (log around 145 and 166-177)
    """
    r0 = MagicMock(resindex=10)
    r1 = MagicMock(resindex=11)

    mol = MagicMock()
    mol.residues = [r0, r1]

    res0 = _bead([100, 101], heavy_resindex=10)
    res1 = _bead([200], heavy_resindex=11)

    ua0 = _bead([1, 2], heavy_resindex=10)
    ua1 = _bead([3], heavy_resindex=11)

    hier = MagicMock()

    def _get_beads(m, lvl):
        if lvl == "residue":
            return [res0, res1]
        if lvl == "united_atom":
            return [ua0, ua1]
        return []

    hier.get_beads.side_effect = _get_beads

    node = BuildBeadsNode(hierarchy=hier)

    u = MagicMock()
    u.atoms = MagicMock()
    u.atoms.fragments = [mol]

    shared = {"reduced_universe": u, "levels": [["residue", "united_atom"]]}
    out = node.run(shared)

    beads = out["beads"]

    assert (0, "residue") in beads
    assert len(beads[(0, "residue")]) == 2
    assert np.array_equal(beads[(0, "residue")][0], np.array([100, 101]))
    assert np.array_equal(beads[(0, "residue")][1], np.array([200]))


def test_build_beads_node_polymer_level_adds_polymer_beads_and_skips_empty():
    mol0 = MagicMock()
    mol0.residues = [MagicMock(resindex=10)]

    u = MagicMock()
    u.atoms.fragments = [mol0]

    polymer_beads = [_bead([]), _bead([7, 8, 9])]

    hier = MagicMock()
    hier.get_beads.side_effect = lambda m, lvl: (
        polymer_beads if lvl == "polymer" else []
    )

    node = BuildBeadsNode(hierarchy=hier)

    shared = {"reduced_universe": u, "levels": [["polymer"]]}
    out = node.run(shared)

    beads = out["beads"]
    assert (0, "polymer") in beads

    assert len(beads[(0, "polymer")]) == 1
    np.testing.assert_array_equal(beads[(0, "polymer")][0], np.array([7, 8, 9]))
