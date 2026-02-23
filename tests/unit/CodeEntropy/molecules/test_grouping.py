import logging
from unittest.mock import MagicMock

import pytest

from CodeEntropy.molecules.grouping import MoleculeGrouper


def _universe_with_fragments(fragments):
    u = MagicMock()
    u.atoms.fragments = fragments
    return u


def _fragment(names):
    f = MagicMock()
    f.names = names
    return f


def test_get_strategy_returns_each_callable():
    g = MoleculeGrouper()
    fn = g._get_strategy("each")
    assert callable(fn)
    assert fn == g._group_each


def test_get_strategy_returns_molecules_callable():
    g = MoleculeGrouper()
    fn = g._get_strategy("molecules")
    assert callable(fn)
    assert fn == g._group_by_signature


def test_get_strategy_raises_value_error_for_unknown():
    g = MoleculeGrouper()
    with pytest.raises(ValueError, match="Unknown grouping strategy"):
        g._get_strategy("nope")


def test_fragments_returns_universe_fragments():
    g = MoleculeGrouper()
    frags = [MagicMock(), MagicMock()]
    u = _universe_with_fragments(frags)
    assert g._fragments(u) is frags


def test_num_molecules_counts_fragments_length():
    g = MoleculeGrouper()
    u = _universe_with_fragments([MagicMock(), MagicMock(), MagicMock()])
    assert g._num_molecules(u) == 3


def test_group_each_returns_one_group_per_molecule():
    g = MoleculeGrouper()
    u = _universe_with_fragments([MagicMock(), MagicMock(), MagicMock()])
    assert g._group_each(u) == {0: [0], 1: [1], 2: [2]}


def test_signature_uses_atom_count_and_ordered_names():
    g = MoleculeGrouper()
    frag = _fragment(["H", "O", "H"])
    assert g._signature(frag) == (3, ("H", "O", "H"))


def test_representative_id_first_seen_sets_rep_and_returns_candidate():
    g = MoleculeGrouper()
    cache = {}
    sig = (3, ("H", "O", "H"))
    rep = g._representative_id(cache, sig, candidate_id=5)
    assert rep == 5
    assert cache[sig] == 5


def test_representative_id_returns_existing_rep_when_seen_before():
    g = MoleculeGrouper()
    sig = (3, ("H", "O", "H"))
    cache = {sig: 2}
    rep = g._representative_id(cache, sig, candidate_id=9)
    assert rep == 2
    assert cache[sig] == 2


def test_group_by_signature_groups_identical_signatures_and_uses_first_id_as_group_id():
    g = MoleculeGrouper()
    f0 = _fragment(["H", "O", "H"])
    f1 = _fragment(["H", "O", "H"])  # same signature as f0
    f2 = _fragment(["C", "C", "H", "H"])  # different signature
    u = _universe_with_fragments([f0, f1, f2])

    out = g._group_by_signature(u)

    assert out == {0: [0, 1], 2: [2]}


def test_group_by_signature_is_deterministic_for_first_seen_representative():
    g = MoleculeGrouper()
    f0 = _fragment(["B"])
    f1 = _fragment(["A"])
    f2 = _fragment(["B"])
    u = _universe_with_fragments([f0, f1, f2])

    out = g._group_by_signature(u)

    assert out[0] == [0, 2]
    assert out[1] == [1]


def test_grouping_molecules_dispatches_each_and_logs_summary(caplog):
    g = MoleculeGrouper()
    u = _universe_with_fragments([MagicMock(), MagicMock()])

    caplog.set_level(logging.INFO)
    out = g.grouping_molecules(u, "each")

    assert out == {0: [0], 1: [1]}
    assert any("Number of molecule groups" in rec.message for rec in caplog.records)


def test_grouping_molecules_dispatches_molecules_strategy():
    g = MoleculeGrouper()
    f0 = _fragment(["H", "O", "H"])
    f1 = _fragment(["H", "O", "H"])
    u = _universe_with_fragments([f0, f1])

    out = g.grouping_molecules(u, "molecules")

    assert out == {0: [0, 1]}
