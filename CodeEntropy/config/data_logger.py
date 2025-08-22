import json
import logging
import re

import numpy as np
from rich.console import Console
from rich.table import Table

from CodeEntropy.config.logging_config import LoggingConfig

# Set up logger
logger = logging.getLogger(__name__)
console = LoggingConfig.get_console()


class DataLogger:
    def __init__(self, console=None):
        self.console = console or Console()
        self.molecule_data = []
        self.residue_data = []
        self.group_labels = {}

    def save_dataframes_as_json(self, molecule_df, residue_df, output_file):
        """Save multiple DataFrames into a single JSON file with separate keys"""
        data = {
            "molecule_data": molecule_df.to_dict(orient="records"),
            "residue_data": residue_df.to_dict(orient="records"),
        }

        # Write JSON data to file
        with open(output_file, "w") as out:
            json.dump(data, out, indent=4)

    def clean_residue_name(self, resname):
        """Ensures residue names are stripped and cleaned before being stored"""
        return re.sub(r"[-–—]", "", str(resname))

    def add_results_data(self, group_id, level, entropy_type, value):
        """Add data for molecule-level entries"""
        self.molecule_data.append((group_id, level, entropy_type, value))

    def add_residue_data(
        self, group_id, resname, level, entropy_type, frame_count, value
    ):
        """Add data for residue-level entries"""
        resname = self.clean_residue_name(resname)
        if isinstance(frame_count, np.ndarray):
            frame_count = frame_count.tolist()
        self.residue_data.append(
            [group_id, resname, level, entropy_type, frame_count, value]
        )

    def add_group_label(self, group_id, label, residue_count=None, atom_count=None):
        """Store a mapping from group ID to a descriptive label and metadata"""
        self.group_labels[group_id] = {
            "label": label,
            "residue_count": residue_count,
            "atom_count": atom_count,
        }

    def log_tables(self):
        """Display rich tables in terminal"""

        if self.molecule_data:
            table = Table(
                title="Molecule Entropy Results", show_lines=True, expand=True
            )
            table.add_column("Group ID", justify="center", style="bold cyan")
            table.add_column("Level", justify="center", style="magenta")
            table.add_column("Type", justify="center", style="green")
            table.add_column("Result (J/mol/K)", justify="center", style="yellow")

            for row in self.molecule_data:
                table.add_row(*[str(cell) for cell in row])

            console.print(table)

        if self.residue_data:
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

        if self.group_labels:
            label_table = Table(
                title="Group ID to Residue Label Mapping", show_lines=True, expand=True
            )
            label_table.add_column("Group ID", justify="center", style="bold cyan")
            label_table.add_column("Residue Label", justify="center", style="green")
            label_table.add_column("Residue Count", justify="center", style="magenta")
            label_table.add_column("Atom Count", justify="center", style="yellow")

            for group_id, info in self.group_labels.items():
                label_table.add_row(
                    str(group_id),
                    info["label"],
                    str(info.get("residue_count", "")),
                    str(info.get("atom_count", "")),
                )

            console.print(label_table)
