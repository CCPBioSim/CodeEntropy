import json
import logging

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

    def add_results_data(self, molecule, level, type, S_molecule):
        """Add data for molecule-level entries"""
        self.molecule_data.append([molecule, level, type, f"{S_molecule}"])

    def add_residue_data(self, molecule, residue, type, S_trans_residue):
        """Add data for residue-level entries"""
        self.residue_data.append([molecule, residue, type, f"{S_trans_residue}"])

    def log_tables(self):
        """Log both tables at once"""
        # Log molecule data
        if self.molecule_data:
            logger.info("Molecule Data Table:")
            table_str = tabulate(
                self.molecule_data,
                headers=["Molecule ID", "Level", "Type", "Result (J/mol/K)"],
                tablefmt="grid",
            )
            logger.info(f"\n{table_str}")

        # Log residue data
        if self.residue_data:
            logger.info("Residue Data Table:")
            table_str = tabulate(
                self.residue_data,
                headers=["Molecule ID", "Residue", "Type", "Result (J/mol/K)"],
                tablefmt="grid",
            )
            logger.info(f"\n{table_str}")
