import pytest

from CodeEntropy.config.runtime import CodeEntropyRunner


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    # keep filesystem effects isolated
    monkeypatch.chdir(tmp_path)
    return CodeEntropyRunner(folder=str(tmp_path))
