import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

from CodeEntropy.main_mcc import main


class TestMainMcc(unittest.TestCase):
    """
    Unit tests for the main functionality of CodeEntropy.
    """

    def setUp(self):
        """
        Set up test environment.
        """
        self.test_dir = tempfile.mkdtemp(prefix="CodeEntropy_")
        self.config_file = os.path.join(self.test_dir, "config.yaml")
        self.code_entropy = main

        # Create a mock config file
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            self.setup_file(mock_file)
            with open(self.config_file, "w") as f:
                f.write(mock_file.return_value.read())

        # Change to test directory
        self._orig_dir = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """
        Clean up after each test.
        """
        os.chdir(self._orig_dir)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def setup_file(self, mock_file):
        """
        Mock the contents of a configuration file.
        """
        mock_file.return_value = mock_open(
            read_data="--- \n \nrun1:\n  "
            "top_traj_file: ['/path/to/tpr', '/path/to/trr']\n  "
            "selection_string: 'all'\n  "
            "start: 0\n  "
            "end: -1\n  "
            "step: 1\n  "
            "bin_width: 30\n  "
            "tempra: 298.0\n  "
            "verbose: False\n  "
            "thread: 1\n  "
            "output_file: 'output_file.json'\n  "
            "force_partitioning: 0.5\n  "
            "water_entropy: False"
        ).return_value

    def test_CodeEntropy_imported(self):
        """Sample test, will always pass so long as import statement worked."""
        assert "CodeEntropy" in sys.modules

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=b"--- \n top_traj_file: ['/path/to/tpr', '/path/to/trr'] \n",
    )
    @patch("os.path.exists", return_value=True)
    @patch("MDAnalysis.Universe")
    @patch("gettext.translation", return_value=MagicMock())
    def test_run(self, mock_translation, mock_universe, mock_exists, mock_file):
        """
        Test the execution of the main function with the necessary CLI argument.
        """
        with patch(
            "sys.argv",
            ["CodeEntropy", "--top_traj_file", "/path/to/tpr", "/path/to/trr"],
        ):
            self.setup_file(mock_file)
            mock_universe.return_value.trajectory = MagicMock()
            main()


if __name__ == "__main__":
    unittest.main()
