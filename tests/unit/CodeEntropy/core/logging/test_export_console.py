import os
from unittest.mock import MagicMock


def test_export_console_writes_recorded_output(config):
    # Make export_text deterministic
    config.console.export_text = MagicMock(return_value="HELLO")

    config.export_console("out.txt")

    out_path = os.path.join(config.log_dir, "out.txt")
    assert os.path.exists(out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        assert f.read() == "HELLO"

    config.console.export_text.assert_called_once()
