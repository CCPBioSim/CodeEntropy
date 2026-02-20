"""Utilities for logging entropy results and exporting data.

This module provides the DataLogger class, which is responsible for:

- Collecting molecule-level entropy results
- Collecting residue-level entropy results
- Storing group metadata labels
- Rendering rich tables to the console
- Exporting results to JSON
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from CodeEntropy.core.logging import LoggingConfig

logger = logging.getLogger(__name__)
console = LoggingConfig.get_console()


class DataLogger:
    """Collect, format, and output entropy calculation results."""

    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialize the logger.

        Args:
            console: Optional Rich Console instance. If None, the global
                console from LoggingConfig is used.
        """
        self.console: Console = console or Console()
        self.molecule_data: List[Tuple[Any, Any, Any, Any]] = []
        self.residue_data: List[List[Any]] = []
        self.group_labels: Dict[Any, Dict[str, Any]] = {}

    def save_dataframes_as_json(
        self, molecule_df, residue_df, output_file: str
    ) -> None:
        """Save molecule and residue DataFrames into a JSON file.

        Args:
            molecule_df: Pandas DataFrame containing molecule results.
            residue_df: Pandas DataFrame containing residue results.
            output_file: Path to JSON output file.
        """
        data = {
            "molecule_data": molecule_df.to_dict(orient="records"),
            "residue_data": residue_df.to_dict(orient="records"),
        }

        with open(output_file, "w") as out:
            json.dump(data, out, indent=4)

    @staticmethod
    def clean_residue_name(resname: Any) -> str:
        """Clean residue name by removing dash-like characters.

        Args:
            resname: Residue name input.

        Returns:
            Cleaned residue name string.
        """
        return re.sub(r"[-–—]", "", str(resname))

    def add_results_data(
        self,
        group_id: Any,
        level: str,
        entropy_type: str,
        value: Any,
    ) -> None:
        """Add molecule-level entropy result.

        Args:
            group_id: Group identifier.
            level: Hierarchy level (e.g., united_atom, residue).
            entropy_type: Entropy category.
            value: Computed entropy value.
        """
        self.molecule_data.append((group_id, level, entropy_type, value))

    def add_residue_data(
        self,
        group_id: Any,
        resname: str,
        level: str,
        entropy_type: str,
        frame_count: Any,
        value: Any,
    ) -> None:
        """Add residue-level entropy result.

        Args:
            group_id: Group identifier.
            resname: Residue name.
            level: Hierarchy level.
            entropy_type: Entropy category.
            frame_count: Frame count or array.
            value: Computed entropy value.
        """
        resname = self.clean_residue_name(resname)

        if isinstance(frame_count, np.ndarray):
            frame_count = frame_count.tolist()

        self.residue_data.append(
            [group_id, resname, level, entropy_type, frame_count, value]
        )

    def add_group_label(
        self,
        group_id: Any,
        label: str,
        residue_count: Optional[int] = None,
        atom_count: Optional[int] = None,
    ) -> None:
        """Store metadata label for a group.

        Args:
            group_id: Group identifier.
            label: Descriptive label.
            residue_count: Optional residue count.
            atom_count: Optional atom count.
        """
        self.group_labels[group_id] = {
            "label": label,
            "residue_count": residue_count,
            "atom_count": atom_count,
        }

    def log_tables(self) -> None:
        """Render all collected data as Rich tables."""

        self._log_molecule_table()
        self._log_residue_table()
        self._log_group_label_table()

    def _log_molecule_table(self) -> None:
        """Render molecule entropy table."""
        if not self.molecule_data:
            return

        table = Table(title="Molecule Entropy Results", show_lines=True, expand=True)
        table.add_column("Group ID", justify="center", style="bold cyan")
        table.add_column("Level", justify="center", style="magenta")
        table.add_column("Type", justify="center", style="green")
        table.add_column("Result (J/mol/K)", justify="center", style="yellow")

        for row in self.molecule_data:
            table.add_row(*[str(cell) for cell in row])

        console.print(table)

    def _log_residue_table(self) -> None:
        """Render residue entropy table."""
        if not self.residue_data:
            return

        table = Table(title="Residue Entropy Results", show_lines=True, expand=True)
        table.add_column("Group ID", justify="center", style="bold cyan")
        table.add_column("Residue Name", justify="center", style="cyan")
        table.add_column("Level", justify="center", style="magenta")
        table.add_column("Type", justify="center", style="green")
        table.add_column("Count", justify="center", style="green")
        table.add_column("Result (J/mol/K)", justify="center", style="yellow")

        for row in self.residue_data:
            table.add_row(*[str(cell) for cell in row])

        console.print(table)

    def _log_group_label_table(self) -> None:
        """Render group label metadata table."""
        if not self.group_labels:
            return

        table = Table(
            title="Group ID to Residue Label Mapping", show_lines=True, expand=True
        )
        table.add_column("Group ID", justify="center", style="bold cyan")
        table.add_column("Residue Label", justify="center", style="green")
        table.add_column("Residue Count", justify="center", style="magenta")
        table.add_column("Atom Count", justify="center", style="yellow")

        for group_id, info in self.group_labels.items():
            table.add_row(
                str(group_id),
                info["label"],
                str(info.get("residue_count", "")),
                str(info.get("atom_count", "")),
            )

        console.print(table)
