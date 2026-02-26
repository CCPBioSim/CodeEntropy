from unittest.mock import MagicMock, patch


def test_write_universe(runner):
    u = MagicMock()

    with patch("pickle.dump") as mock_dump:
        name = runner.write_universe(u, name="test")

    assert name == "test"
    mock_dump.assert_called_once()


def test_read_universe(runner):
    mock_file = MagicMock()

    # Make open() context manager return mock_file
    mock_file.__enter__.return_value = mock_file

    with (
        patch("builtins.open", return_value=mock_file) as mock_open,
        patch("pickle.load", return_value="U") as mock_load,
    ):
        out = runner.read_universe("file.pkl")

    mock_open.assert_called_once_with("file.pkl", "rb")
    mock_load.assert_called_once_with(mock_file)
    assert out == "U"
