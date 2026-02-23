import runpy
from unittest.mock import MagicMock

import CodeEntropy.cli as cli


def test___main___invokes_cli_main(monkeypatch):
    main_spy = MagicMock()
    monkeypatch.setattr(cli, "main", main_spy)

    runpy.run_module("CodeEntropy", run_name="__main__")

    main_spy.assert_called_once_with()
