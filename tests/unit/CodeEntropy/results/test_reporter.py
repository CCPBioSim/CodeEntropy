import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from rich.console import Console

import CodeEntropy.results.reporter as reporter_mod
from CodeEntropy.results.reporter import ResultsReporter, _RichProgressSink


class FakeTable:
    def __init__(self, title=None, show_lines=None, expand=None):
        self.title = title
        self.columns = []
        self.rows = []

    def add_column(self, *args, **kwargs):
        self.columns.append((args, kwargs))

    def add_row(self, *args, **kwargs):
        self.rows.append((args, kwargs))


def test_init_uses_provided_console():
    c = Console()
    rr = ResultsReporter(console=c)
    assert rr.console is c


def test_clean_residue_name_removes_dash_like():
    assert ResultsReporter.clean_residue_name("ALA-GLY") == "ALAGLY"
    assert ResultsReporter.clean_residue_name("ALA–GLY") == "ALAGLY"
    assert ResultsReporter.clean_residue_name("ALA—GLY") == "ALAGLY"
    assert ResultsReporter.clean_residue_name(123) == "123"


def test_add_results_data_appends():
    rr = ResultsReporter()
    rr.add_results_data(group_id=1, level="L", entropy_type="T", value=1.23)
    assert rr.molecule_data == [(1, "L", "T", 1.23)]


def test_add_residue_data_converts_ndarray_frame_count_to_list():
    rr = ResultsReporter()
    rr.add_residue_data(
        group_id=1,
        resname="ALA-1",
        level="L",
        entropy_type="T",
        frame_count=np.array([1, 2, 3]),
        value=9.0,
    )
    assert rr.residue_data == [[1, "ALA1", "L", "T", [1, 2, 3], 9.0]]


def test_add_residue_data_keeps_scalar_frame_count():
    rr = ResultsReporter()
    rr.add_residue_data(
        group_id=1,
        resname="ALA-1",
        level="L",
        entropy_type="T",
        frame_count=7,
        value=9.0,
    )
    assert rr.residue_data == [[1, "ALA1", "L", "T", 7, 9.0]]


def test_add_group_label_stores_metadata():
    rr = ResultsReporter()
    rr.add_group_label(1, "protein", residue_count=10, atom_count=100)
    assert rr.group_labels[1]["label"] == "protein"
    assert rr.group_labels[1]["residue_count"] == 10
    assert rr.group_labels[1]["atom_count"] == 100


def test_gid_sort_key_numeric_before_string_and_numeric_order():
    rr = ResultsReporter()
    gids = ["10", "2", "A", "1", "B"]
    out = sorted(gids, key=rr._gid_sort_key)
    assert out == ["1", "2", "10", "A", "B"]


def test_safe_float_valid_invalid():
    assert ResultsReporter._safe_float("1.25") == 1.25
    assert ResultsReporter._safe_float(3) == 3.0
    assert ResultsReporter._safe_float("bad") is None
    assert ResultsReporter._safe_float(None) is None
    assert ResultsReporter._safe_float(True) is None


def test_build_grouped_payload_components_and_total_from_sum(monkeypatch):
    rr = ResultsReporter()
    mol = pd.DataFrame(
        [
            {"Group ID": 1, "Level": "Trans", "Type": "A", "Result (J/mol/K)": 1.0},
            {"Group ID": 1, "Level": "Rovib", "Type": "B", "Result (J/mol/K)": 2.0},
        ]
    )
    res = pd.DataFrame([])
    monkeypatch.setattr(
        ResultsReporter, "_provenance", staticmethod(lambda: {"git_sha": None})
    )
    payload = rr._build_grouped_payload(
        molecule_df=mol, residue_df=res, args=None, include_raw_tables=False
    )
    comps = payload["groups"]["1"]["components"]
    assert comps == {"Trans:A": 1.0, "Rovib:B": 2.0} or comps == {
        "Rovib:B": 2.0,
        "Trans:A": 1.0,
    }
    assert payload["groups"]["1"]["total"] == 3.0


def test_build_grouped_payload_prefers_explicit_total(monkeypatch):
    rr = ResultsReporter()
    mol = pd.DataFrame(
        [
            {"Group ID": 1, "Level": "Trans", "Type": "A", "Result (J/mol/K)": 1.0},
            {
                "Group ID": 1,
                "Level": "Group Total",
                "Type": "Total",
                "Result (J/mol/K)": 99.0,
            },
            {"Group ID": 1, "Level": "Rovib", "Type": "B", "Result (J/mol/K)": 2.0},
        ]
    )
    res = pd.DataFrame([])
    monkeypatch.setattr(
        ResultsReporter, "_provenance", staticmethod(lambda: {"git_sha": None})
    )
    payload = rr._build_grouped_payload(
        molecule_df=mol, residue_df=res, args=None, include_raw_tables=False
    )
    assert payload["groups"]["1"]["total"] == 99.0
    assert payload["groups"]["1"]["components"]["Trans:A"] == 1.0
    assert payload["groups"]["1"]["components"]["Rovib:B"] == 2.0


def test_build_grouped_payload_skips_non_numeric_results(monkeypatch):
    rr = ResultsReporter()
    mol = pd.DataFrame(
        [
            {"Group ID": 1, "Level": "Trans", "Type": "A", "Result (J/mol/K)": 1.0},
            {"Group ID": 1, "Level": "Rovib", "Type": "B", "Result (J/mol/K)": "bad"},
            {
                "Group ID": 1,
                "Level": "Group Total",
                "Type": "Total",
                "Result (J/mol/K)": None,
            },
        ]
    )
    res = pd.DataFrame([])
    monkeypatch.setattr(
        ResultsReporter, "_provenance", staticmethod(lambda: {"git_sha": None})
    )
    payload = rr._build_grouped_payload(
        molecule_df=mol, residue_df=res, args=None, include_raw_tables=False
    )
    assert payload["groups"]["1"]["components"] == {"Trans:A": 1.0}
    assert payload["groups"]["1"]["total"] == 1.0


def test_build_grouped_payload_invalid_total_row_skipped(monkeypatch):
    rr = ResultsReporter()
    mol = pd.DataFrame(
        [
            {
                "Group ID": 1,
                "Level": "Group Total",
                "Type": "Total",
                "Result (J/mol/K)": "bad",
            },
        ]
    )
    res = pd.DataFrame([])
    monkeypatch.setattr(
        ResultsReporter, "_provenance", staticmethod(lambda: {"git_sha": None})
    )
    payload = rr._build_grouped_payload(
        molecule_df=mol, residue_df=res, args=None, include_raw_tables=False
    )
    assert payload["groups"] == {}


def test_build_grouped_payload_include_raw_tables(monkeypatch):
    rr = ResultsReporter()
    mol = pd.DataFrame(
        [{"Group ID": 1, "Level": "Trans", "Type": "A", "Result (J/mol/K)": 1.0}]
    )
    res = pd.DataFrame([{"Group ID": 1, "Residue": "ALA", "Result": 0.5}])
    monkeypatch.setattr(
        ResultsReporter, "_provenance", staticmethod(lambda: {"git_sha": None})
    )
    payload = rr._build_grouped_payload(
        molecule_df=mol, residue_df=res, args=None, include_raw_tables=True
    )
    assert "molecule_data" in payload
    assert "residue_data" in payload
    assert payload["molecule_data"][0]["Group ID"] == 1
    assert payload["residue_data"][0]["Group ID"] == 1


def test_serialize_args_dict_converts_ndarray_and_path():
    p = Path("x/y")
    args = {"arr": np.array([1, 2]), "p": p, "n": 3}
    out = ResultsReporter._serialize_args(args)
    assert out == {"arr": [1, 2], "p": str(p), "n": 3}


def test_serialize_args_namespace_converts_types():
    ns = SimpleNamespace(a=np.array([1]), b=Path("z"))
    assert ResultsReporter._serialize_args(ns) == {"a": [1], "b": "z"}


def test_serialize_args_falls_back_to_dict_protocol():
    class PairIterable:
        def __iter__(self):
            return iter([("k", 1)])

    assert ResultsReporter._serialize_args(PairIterable()) == {"k": 1}


def test_serialize_args_unserializable_returns_empty():
    class Unserializable:
        __slots__ = ()

    assert ResultsReporter._serialize_args(Unserializable()) == {}


def test_provenance_sets_version_none_on_failure(monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _: (_ for _ in ()).throw(Exception("nope")),
    )
    monkeypatch.setattr(ResultsReporter, "_try_get_git_sha", staticmethod(lambda: None))
    prov = ResultsReporter._provenance()
    assert "python" in prov
    assert "platform" in prov
    assert prov["codeentropy_version"] is None
    assert prov["git_sha"] is None


def test_provenance_sets_version_on_success(monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda _: "9.9.9")
    monkeypatch.setattr(
        ResultsReporter, "_try_get_git_sha", staticmethod(lambda: "sha")
    )
    prov = ResultsReporter._provenance()
    assert prov["codeentropy_version"] == "9.9.9"
    assert prov["git_sha"] == "sha"


def test_try_get_git_sha_env_override(monkeypatch):
    monkeypatch.setenv("CODEENTROPY_GIT_SHA", "abc123")
    assert ResultsReporter._try_get_git_sha() == "abc123"


def test_try_get_git_sha_subprocess_success(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEENTROPY_GIT_SHA", raising=False)
    fake_file = tmp_path / "a" / "b" / "c.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "a" / ".git").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(reporter_mod, "__file__", str(fake_file))
    mock_run = MagicMock()
    mock_run.return_value = SimpleNamespace(
        returncode=0, stdout="deadbeef\n", stderr=""
    )
    monkeypatch.setattr(reporter_mod.subprocess, "run", mock_run)
    assert ResultsReporter._try_get_git_sha() == "deadbeef"


def test_try_get_git_sha_subprocess_failure(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEENTROPY_GIT_SHA", raising=False)
    fake_file = tmp_path / "a" / "b" / "c.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "a" / ".git").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(reporter_mod, "__file__", str(fake_file))
    mock_run = MagicMock()
    mock_run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="err")
    monkeypatch.setattr(reporter_mod.subprocess, "run", mock_run)
    assert ResultsReporter._try_get_git_sha() is None


def test_try_get_git_sha_returns_none_when_no_git_anywhere(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEENTROPY_GIT_SHA", raising=False)
    fake_file = tmp_path / "a" / "b" / "c.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(reporter_mod, "__file__", str(fake_file))
    monkeypatch.setattr(
        reporter_mod.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("subprocess.run should not be called")
        ),
    )
    assert ResultsReporter._try_get_git_sha() is None


def test_try_get_git_sha_executes_subprocess_kwargs_block(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEENTROPY_GIT_SHA", raising=False)
    fake_file = tmp_path / "a" / "b" / "c.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "a" / ".git").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(reporter_mod, "__file__", str(fake_file))

    original_resolve = reporter_mod.Path.resolve

    def fake_resolve(self):
        if str(self) == str(fake_file):
            return fake_file
        return original_resolve(self)

    monkeypatch.setattr(reporter_mod.Path, "resolve", fake_resolve)

    mock_run = MagicMock()
    mock_run.return_value = SimpleNamespace(returncode=0, stdout="sha\n", stderr="")
    monkeypatch.setattr(reporter_mod.subprocess, "run", mock_run)

    assert ResultsReporter._try_get_git_sha() == "sha"
    _args, kwargs = mock_run.call_args
    assert "stdout" in kwargs
    assert "stderr" in kwargs
    assert kwargs.get("text") is True


def test_log_grouped_results_tables_hits_non_total_add_row(monkeypatch):
    rr = ResultsReporter()
    rr.add_results_data("1", "AAA", "BBB", 123.0)
    printed = []
    monkeypatch.setattr(reporter_mod, "Table", FakeTable)
    monkeypatch.setattr(reporter_mod.console, "print", lambda t: printed.append(t))
    rr._log_grouped_results_tables()
    assert len(printed) == 1
    assert printed[0].rows == [(("AAA", "BBB", "123.0"), {})]


def test_log_grouped_results_tables_prints_in_sorted_gid_order(monkeypatch):
    rr = ResultsReporter()
    rr.add_results_data("10", "L", "T", 1.0)
    rr.add_results_data("2", "L", "T", 2.0)
    rr.add_results_data("A", "L", "T", 3.0)
    printed_titles = []
    monkeypatch.setattr(
        reporter_mod.console,
        "print",
        lambda obj: printed_titles.append(getattr(obj, "title", None)),
    )
    rr._log_grouped_results_tables()
    assert printed_titles[0].startswith("Entropy Results — Group 2")
    assert printed_titles[1].startswith("Entropy Results — Group 10")
    assert printed_titles[2].startswith("Entropy Results — Group A")


def test_log_residue_table_grouped_prints_table(monkeypatch):
    rr = ResultsReporter()
    rr.add_group_label("2", "ResidLabel")
    rr.residue_data.append(["2", "ALA", "LevelX", "TypeY", 10, 0.5])
    printed = []
    monkeypatch.setattr(reporter_mod.console, "print", lambda obj: printed.append(obj))
    rr._log_residue_table_grouped()
    assert len(printed) == 1
    assert getattr(printed[0], "title", "").startswith("Residue Entropy — Group 2")


def test_log_group_label_table_hits_label_add_column(monkeypatch):
    rr = ResultsReporter()
    rr.add_group_label("1", "LabelHere", residue_count=2, atom_count=3)
    printed = []
    monkeypatch.setattr(reporter_mod, "Table", FakeTable)
    monkeypatch.setattr(reporter_mod.console, "print", lambda t: printed.append(t))
    rr._log_group_label_table()
    assert len(printed) == 1
    assert printed[0].columns[1][0][0] == "Label"


def test_log_tables_calls_all_subtables(monkeypatch):
    rr = ResultsReporter()
    m1 = MagicMock()
    m2 = MagicMock()
    m3 = MagicMock()
    monkeypatch.setattr(rr, "_log_grouped_results_tables", m1)
    monkeypatch.setattr(rr, "_log_residue_table_grouped", m2)
    monkeypatch.setattr(rr, "_log_group_label_table", m3)
    rr.log_tables()
    m1.assert_called_once()
    m2.assert_called_once()
    m3.assert_called_once()


def test_save_dataframes_as_json_writes_file(tmp_path, monkeypatch):
    rr = ResultsReporter()
    mol = pd.DataFrame(
        [{"Group ID": 1, "Level": "Trans", "Type": "A", "Result (J/mol/K)": 1.0}]
    )
    res = pd.DataFrame([])
    monkeypatch.setattr(
        ResultsReporter, "_provenance", staticmethod(lambda: {"git_sha": None})
    )
    out = tmp_path / "out.json"
    rr.save_dataframes_as_json(
        mol, res, str(out), args={"x": 1}, include_raw_tables=False
    )
    data = json.loads(out.read_text())
    assert data["args"] == {"x": 1}
    assert data["provenance"] == {"git_sha": None}
    assert data["groups"]["1"]["components"] == {"Trans:A": 1.0}
    assert data["groups"]["1"]["total"] == 1.0


def test_save_dataframes_as_json_uses_default_include_raw_tables(tmp_path, monkeypatch):
    rr = ResultsReporter()
    mol = pd.DataFrame(
        [{"Group ID": 1, "Level": "L", "Type": "T", "Result (J/mol/K)": 1.0}]
    )
    res = pd.DataFrame([])
    monkeypatch.setattr(
        ResultsReporter, "_provenance", staticmethod(lambda: {"git_sha": None})
    )
    out = tmp_path / "out.json"
    rr.save_dataframes_as_json(mol, res, str(out), args={"x": 1})
    assert out.exists()


def test_log_grouped_results_tables_returns_when_empty(monkeypatch):
    rr = ResultsReporter()
    monkeypatch.setattr(
        reporter_mod.console,
        "print",
        lambda *_: (_ for _ in ()).throw(AssertionError("should not print")),
    )
    rr._log_grouped_results_tables()


def test_log_grouped_results_tables_handles_total_row(monkeypatch):
    rr = ResultsReporter()
    rr.add_results_data("1", "A", "B", 1.0)
    rr.add_results_data("1", "Group Total", "Group Total", 3.0)

    printed = []
    monkeypatch.setattr(reporter_mod, "Table", FakeTable)
    monkeypatch.setattr(reporter_mod.console, "print", lambda t: printed.append(t))

    rr._log_grouped_results_tables()

    assert len(printed) == 1
    assert ("Group Total", "Group Total", "3.0") in [r[0] for r in printed[0].rows]


def test_log_residue_table_grouped_returns_when_empty(monkeypatch):
    rr = ResultsReporter()
    monkeypatch.setattr(
        reporter_mod.console,
        "print",
        lambda *_: (_ for _ in ()).throw(AssertionError("should not print")),
    )
    rr._log_residue_table_grouped()


def test_log_group_label_table_returns_when_empty(monkeypatch):
    rr = ResultsReporter()
    monkeypatch.setattr(
        reporter_mod.console,
        "print",
        lambda *_: (_ for _ in ()).throw(AssertionError("should not print")),
    )
    rr._log_group_label_table()


def test_try_get_git_sha_returns_none_on_exception(monkeypatch):
    monkeypatch.delenv("CODEENTROPY_GIT_SHA", raising=False)

    def boom(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(reporter_mod.Path, "resolve", boom)
    assert ResultsReporter._try_get_git_sha() is None


def test_progress_context_yields_progress_sink():
    rr = ResultsReporter()
    with rr.progress(transient=True) as p:
        assert hasattr(p, "add_task")
        assert hasattr(p, "update")
        assert hasattr(p, "advance")


def test_progress_sink_update_normalizes_none_title(monkeypatch):
    rr = ResultsReporter()

    with rr.progress(transient=True) as sink:
        inner = sink._progress
        spy = MagicMock()
        monkeypatch.setattr(inner, "update", spy)

        sink.update(1, title=None)

        spy.assert_called_once()
        _args, kwargs = spy.call_args
        assert kwargs["title"] == ""


def test_rich_progress_sink_add_task_sets_default_title():
    inner = MagicMock()
    inner.add_task.return_value = 7

    sink = _RichProgressSink(inner)
    task_id = sink.add_task("Stage", total=3)

    assert task_id == 7

    inner.add_task.assert_called_once()
    args, kwargs = inner.add_task.call_args

    assert args[0] == "Stage"
    assert kwargs["total"] == 3
    assert kwargs["title"] == ""


def test_rich_progress_sink_update_normalizes_title_none():
    inner = MagicMock()
    sink = _RichProgressSink(inner)

    sink.update(99, title=None)

    inner.update.assert_called_once_with(99, title="")


def test_gid_sort_key_handles_non_numeric_group_id():
    assert ResultsReporter._gid_sort_key("abc") == (1, "abc")


def test_rich_progress_sink_advance_forwards_to_inner_progress():
    inner = MagicMock()
    sink = _RichProgressSink(inner)

    sink.advance(123, step=5)

    inner.advance.assert_called_once_with(123, 5)
