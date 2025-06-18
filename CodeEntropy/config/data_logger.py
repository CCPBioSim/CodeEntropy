import json
import logging
import re

from tabulate import tabulate

# Set up logger
logger = logging.getLogger(__name__)


class DataLogger:
    def __init__(self):
        self.molecule_data = []
        self.residue_data = []

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

    def add_results_data(self, resname, level, entropy_type, value):
        """Add data for molecule-level entries"""
        resname = self.clean_residue_name(resname)
        self.molecule_data.append((resname, level, entropy_type, value))

    def add_residue_data(self, resid, resname, level, entropy_type, value):
        """Add data for residue-level entries"""
        resname = self.clean_residue_name(resname)
        self.residue_data.append([resid, resname, level, entropy_type, value])

    def log_tables(self):
        """Log both tables at once"""
        # Log molecule data
        if self.molecule_data:
            logger.info("Molecule Data Table:")
            table_str = tabulate(
                self.molecule_data,
                headers=["Residue Name", "Level", "Type", "Result (J/mol/K)"],
                tablefmt="grid",
                numalign="center",
                stralign="center",
            )
            logger.info(f"\n{table_str}")

        # Log residue data
        if self.residue_data:
            logger.info("Residue Data Table:")
            table_str = tabulate(
                self.residue_data,
                headers=[
                    "Residue ID",
                    "Residue Name",
                    "Level",
                    "Type",
                    "Result (J/mol/K)",
                ],
                tablefmt="grid",
                numalign="center",
                stralign="center",
            )
            logger.info(f"\n{table_str}")
