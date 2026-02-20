from io import StringIO
from unittest.mock import patch

from rich.console import Console


def test_show_splash_with_citation(runner):
    citation = {
        "title": "TestProject",
        "version": "1.0",
        "date-released": "2025-01-01",
        "url": "https://example.com",
        "abstract": "This is a test abstract.",
        "authors": [{"given-names": "Alice", "family-names": "Smith"}],
    }

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)

    with (
        patch.object(runner, "load_citation_data", return_value=citation),
        patch("CodeEntropy.config.runtime.console", console),
    ):
        runner.show_splash()

    out = buf.getvalue()

    assert "Welcome to CodeEntropy" in out
    assert "Version 1.0" in out
    assert "2025-01-01" in out
    assert "https://example.com" in out
    assert "This is a test abstract." in out
    assert "Alice Smith" in out


def test_show_splash_without_citation(runner):
    buf = StringIO()
    console = Console(file=buf, force_terminal=False)

    with (
        patch.object(runner, "load_citation_data", return_value=None),
        patch("CodeEntropy.config.runtime.console", console),
    ):
        runner.show_splash()

    assert "Welcome to CodeEntropy" in buf.getvalue()
