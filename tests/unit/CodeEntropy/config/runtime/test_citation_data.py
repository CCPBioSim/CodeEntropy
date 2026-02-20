from unittest.mock import MagicMock, patch

import requests


def test_load_citation_data_success(runner):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.text = """
title: TestProject
authors:
  - given-names: Alice
"""

    with patch("requests.get", return_value=mock_response):
        data = runner.load_citation_data()

    assert data["title"] == "TestProject"


def test_load_citation_data_network_error_returns_none(runner):
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError()):
        assert runner.load_citation_data() is None
