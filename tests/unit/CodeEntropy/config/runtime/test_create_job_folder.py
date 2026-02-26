import os
from unittest.mock import patch

from CodeEntropy.config.runtime import CodeEntropyRunner


def test_create_job_folder_empty_creates_job001():
    with (
        patch("os.getcwd", return_value="/cwd"),
        patch("os.listdir", return_value=[]),
        patch("os.makedirs") as mock_makedirs,
    ):
        path = CodeEntropyRunner.create_job_folder()

    assert path == os.path.join("/cwd", "job001")
    mock_makedirs.assert_called_once()


def test_create_job_folder_existing_creates_next():
    with (
        patch("os.getcwd", return_value="/cwd"),
        patch("os.listdir", return_value=["job001", "job002"]),
        patch("os.makedirs") as mock_makedirs,
    ):
        path = CodeEntropyRunner.create_job_folder()

    assert path == os.path.join("/cwd", "job003")
    mock_makedirs.assert_called_once()


def test_create_job_folder_ignores_invalid_names():
    with (
        patch("os.getcwd", return_value="/cwd"),
        patch("os.listdir", return_value=["job001", "abc", "job002"]),
        patch("os.makedirs") as _,
    ):
        path = CodeEntropyRunner.create_job_folder()

    assert path == os.path.join("/cwd", "job003")


def test_create_job_folder_skips_value_error_suffix():
    # jobABC triggers int("ABC") -> ValueError -> continue
    with (
        patch("os.getcwd", return_value="/cwd"),
        patch("os.listdir", return_value=["jobABC", "job001"]),
        patch("os.makedirs"),
    ):
        path = CodeEntropyRunner.create_job_folder()

    assert path == os.path.join("/cwd", "job002")
