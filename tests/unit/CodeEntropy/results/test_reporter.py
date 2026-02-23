import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import CodeEntropy.results.reporter as reporter
from CodeEntropy.results.reporter import ResultsReporter


class _FakeTable:
    """Tiny Table stand-in: records columns and rows added."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.columns = []
        self.rows = []

    def add_column(self, *args, **kwargs):
        self.columns.append((args, kwargs))

    def add_row(self, *cells):
        self.rows.append(cells)


def test_clean_residue_name_removes_dash_like_characters():
    assert ResultsReporter.clean_residue_name("A-LA") == "ALA"
    assert ResultsReporter.clean_residue_name("A–LA") == "ALA"
    assert ResultsReporter.clean_residue_name("A—LA") == "ALA"
    assert ResultsReporter.clean_residue_name(123) == "123"


def test_add_results_data_appends_molecule_tuple():
    rr = ResultsReporter()
    rr.add_results_data(group_id=7, level="residue", entropy_type="vib", value=1.23)

    assert rr.molecule_data == [(7, "residue", "vib", 1.23)]


def test_add_residue_data_cleans_resname_and_converts_ndarray_count_to_list():
    rr = ResultsReporter()

    rr.add_residue_data(
        group_id=1,
        resname="A-LA",
        level="residue",
        entropy_type="conf",
        frame_count=np.array([1, 2, 3], dtype=int),
        value=9.0,
    )

    assert rr.residue_data == [[1, "ALA", "residue", "conf", [1, 2, 3], 9.0]]


def test_add_residue_data_keeps_non_ndarray_count_as_is():
    rr = ResultsReporter()

    rr.add_residue_data(
        group_id=2,
        resname="GLY",
        level="residue",
        entropy_type="conf",
        frame_count=5,
        value=3.14,
    )

    assert rr.residue_data == [[2, "GLY", "residue", "conf", 5, 3.14]]


def test_add_group_label_stores_metadata_with_optionals():
    rr = ResultsReporter()
    rr.add_group_label(group_id="G0", label="protein", residue_count=10, atom_count=100)

    assert rr.group_labels["G0"] == {
        "label": "protein",
        "residue_count": 10,
        "atom_count": 100,
    }


def test_save_dataframes_as_json_writes_expected_payload(tmp_path):
    rr = ResultsReporter()

    molecule_df = MagicMock()
    residue_df = MagicMock()

    molecule_df.to_dict.return_value = [{"a": 1}]
    residue_df.to_dict.return_value = [{"b": 2}]

    out = tmp_path / "out.json"
    rr.save_dataframes_as_json(
        molecule_df=molecule_df, residue_df=residue_df, output_file=str(out)
    )

    data = json.loads(out.read_text())
    assert data == {"molecule_data": [{"a": 1}], "residue_data": [{"b": 2}]}

    molecule_df.to_dict.assert_called_once_with(orient="records")
    residue_df.to_dict.assert_called_once_with(orient="records")


def test_log_tables_calls_each_internal_table_renderer(monkeypatch):
    rr = ResultsReporter()

    mol_spy = MagicMock()
    res_spy = MagicMock()
    grp_spy = MagicMock()

    monkeypatch.setattr(rr, "_log_molecule_table", mol_spy)
    monkeypatch.setattr(rr, "_log_residue_table", res_spy)
    monkeypatch.setattr(rr, "_log_group_label_table", grp_spy)

    rr.log_tables()

    mol_spy.assert_called_once()
    res_spy.assert_called_once()
    grp_spy.assert_called_once()


def test_log_molecule_table_returns_early_when_no_data(monkeypatch):
    rr = ResultsReporter()

    fake_console = SimpleNamespace(print=MagicMock())
    monkeypatch.setattr(reporter, "console", fake_console)
    monkeypatch.setattr(reporter, "Table", _FakeTable)

    rr._log_molecule_table()

    fake_console.print.assert_not_called()


def test_log_residue_table_returns_early_when_no_data(monkeypatch):
    rr = ResultsReporter()

    fake_console = SimpleNamespace(print=MagicMock())
    monkeypatch.setattr(reporter, "console", fake_console)
    monkeypatch.setattr(reporter, "Table", _FakeTable)

    rr._log_residue_table()

    fake_console.print.assert_not_called()


def test_log_group_label_table_returns_early_when_no_labels(monkeypatch):
    rr = ResultsReporter()

    fake_console = SimpleNamespace(print=MagicMock())
    monkeypatch.setattr(reporter, "console", fake_console)
    monkeypatch.setattr(reporter, "Table", _FakeTable)

    rr._log_group_label_table()

    fake_console.print.assert_not_called()


def test_log_molecule_table_builds_rows_and_prints_table(monkeypatch):
    rr = ResultsReporter()
    rr.molecule_data = [
        (1, "residue", "conf", 1.0),
        (2, "polymer", "vib", 2.0),
    ]

    fake_console = SimpleNamespace(print=MagicMock())
    monkeypatch.setattr(reporter, "console", fake_console)
    monkeypatch.setattr(reporter, "Table", _FakeTable)

    rr._log_molecule_table()

    fake_console.print.assert_called_once()
    table = fake_console.print.call_args.args[0]

    assert isinstance(table, _FakeTable)
    # 4 columns defined, 2 rows added
    assert len(table.columns) == 4
    assert len(table.rows) == 2
    # cells were stringified
    assert table.rows[0] == ("1", "residue", "conf", "1.0")


def test_log_residue_table_builds_rows_and_prints_table(monkeypatch):
    rr = ResultsReporter()
    rr.residue_data = [
        [1, "ALA", "residue", "conf", [1, 2], 9.0],
    ]

    fake_console = SimpleNamespace(print=MagicMock())
    monkeypatch.setattr(reporter, "console", fake_console)
    monkeypatch.setattr(reporter, "Table", _FakeTable)

    rr._log_residue_table()

    fake_console.print.assert_called_once()
    table = fake_console.print.call_args.args[0]

    assert isinstance(table, _FakeTable)
    # 6 columns defined, 1 row added
    assert len(table.columns) == 6
    assert len(table.rows) == 1
    assert table.rows[0] == ("1", "ALA", "residue", "conf", "[1, 2]", "9.0")


def test_log_group_label_table_adds_rows_for_each_label_and_prints(monkeypatch):
    rr = ResultsReporter()
    rr.group_labels = {
        7: {"label": "protein", "residue_count": 10, "atom_count": 100},
        8: {"label": "water", "residue_count": None, "atom_count": None},
    }

    fake_console = SimpleNamespace(print=MagicMock())
    monkeypatch.setattr(reporter, "console", fake_console)
    monkeypatch.setattr(reporter, "Table", _FakeTable)

    rr._log_group_label_table()

    fake_console.print.assert_called_once()
    table = fake_console.print.call_args.args[0]

    assert isinstance(table, _FakeTable)
    assert len(table.columns) == 4
    assert len(table.rows) == 2

    assert table.rows[0] == ("7", "protein", "10", "100")
    assert table.rows[1] == ("8", "water", "None", "None")
