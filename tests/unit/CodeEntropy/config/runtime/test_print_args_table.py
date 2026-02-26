from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console


def test_print_args_table_prints_all_args(runner):
    args = SimpleNamespace(alpha=1, beta="two")

    buf = StringIO()
    test_console = Console(file=buf, force_terminal=False, width=120)

    with patch("CodeEntropy.config.runtime.console", test_console):
        runner.print_args_table(args)

    out = buf.getvalue()
    assert "Run Configuration" in out
    assert "alpha" in out
    assert "1" in out
    assert "beta" in out
    assert "two" in out
