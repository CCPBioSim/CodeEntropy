from unittest.mock import MagicMock

import pytest

import CodeEntropy.cli as entry


def test_main_creates_job_folder_and_runs_workflow(monkeypatch):
    fake_runner_cls = MagicMock()
    fake_runner_cls.create_job_folder.return_value = "/tmp/job"

    fake_runner = MagicMock()
    fake_runner_cls.return_value = fake_runner

    monkeypatch.setattr(entry, "CodeEntropyRunner", fake_runner_cls)

    entry.main()

    fake_runner_cls.create_job_folder.assert_called_once_with()
    fake_runner_cls.assert_called_once_with(folder="/tmp/job")
    fake_runner.run_entropy_workflow.assert_called_once_with()


def test_main_logs_and_exits_nonzero_on_exception(monkeypatch):
    fake_runner_cls = MagicMock()
    fake_runner_cls.create_job_folder.return_value = "/tmp/job"

    fake_runner = MagicMock()
    fake_runner.run_entropy_workflow.side_effect = RuntimeError("boom")
    fake_runner_cls.return_value = fake_runner

    monkeypatch.setattr(entry, "CodeEntropyRunner", fake_runner_cls)

    with pytest.raises(SystemExit) as exc:
        entry.main()

    assert exc.value.code == 1
    fake_runner.run_entropy_workflow.assert_called_once_with()
