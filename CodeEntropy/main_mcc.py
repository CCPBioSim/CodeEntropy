import logging
import math
import os
import re
import sys

import MDAnalysis as mda
import pandas as pd

from CodeEntropy.calculations import EntropyFunctions as EF
from CodeEntropy.calculations import LevelFunctions as LF
from CodeEntropy.calculations import MDAUniverseHelper as MDAHelper
from CodeEntropy.config.arg_config_manager import ConfigManager
from CodeEntropy.config.data_logger import DataLogger
from CodeEntropy.config.logging_config import LoggingConfig


def create_job_folder():
    """
    Create a new job folder with an incremented job number based on existing folders.
    """
    # Get the current working directory
    base_dir = os.getcwd()

    # List all folders in the base directory
    existing_folders = [
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
    ]

    # Filter folders that match the pattern 'jobXXX'
    job_folders = [f for f in existing_folders if re.match(r"job\d{3}", f)]

    # Determine the next job number
    if job_folders:
        max_job_number = max([int(re.search(r"\d{3}", f).group()) for f in job_folders])
        next_job_number = max_job_number + 1
    else:
        next_job_number = 1

    # Format the new job folder name
    new_job_folder = f"job{next_job_number:03d}"
    new_job_folder_path = os.path.join(base_dir, new_job_folder)

    # Create the new job folder
    os.makedirs(new_job_folder_path, exist_ok=True)

    return new_job_folder_path


def main():
    """
    Main function for calculating the entropy of a system using the multiscale cell
    correlation method.
    """
    folder = create_job_folder()
    data_logger = DataLogger()
    arg_config = ConfigManager()

    # Load configuration
    config = arg_config.load_config("config.yaml")
    if config is None:
        raise ValueError(
            "No configuration file found, and no CLI arguments were provided."
        )

    parser = arg_config.setup_argparse()
    args, unknown = parser.parse_known_args()
    args.output_file = os.path.join(folder, args.output_file)

    try:
        # Initialize the logging system once
        logging_config = LoggingConfig(folder)
        logger = logging_config.setup_logging()

        # Process each run in the YAML configuration
        for run_name, run_config in config.items():
            if isinstance(run_config, dict):
                # Merging CLI arguments with YAML configuration
                args = arg_config.merge_configs(args, run_config)

                # Determine logging level
                log_level = logging.DEBUG if args.verbose else logging.INFO

                # Update the logging level
                logging_config.update_logging_level(log_level)

                # Capture and log the command-line invocation
                command = " ".join(sys.argv)
                logging.getLogger("commands").info(command)

                # Ensure necessary arguments are provided
                if not getattr(args, "top_traj_file"):
                    raise ValueError(
                        "The 'top_traj_file' argument is required but not provided."
                    )
                if not getattr(args, "selection_string"):
                    raise ValueError(
                        "The 'selection_string' argument is required but not provided."
                    )

                # Log all inputs for the current run
                logger.info(f"All input for {run_name}")
                for arg in vars(args):
                    logger.info(f" {arg}: {getattr(args, arg) or ''}")
            else:
                logger.warning(f"Run configuration for {run_name} is not a dictionary.")
    except ValueError as e:
        logger.error(e)
        raise

    # Get topology and trajectory file names and make universe
    tprfile = args.top_traj_file[0]
    trrfile = args.top_traj_file[1:]
    u = mda.Universe(tprfile, trrfile)

    # Define bin_width for histogram from inputs
    bin_width = args.bin_width

    # Define trajectory slicing from inputs
    start = args.start
    if start is None:
        start = 0
    end = args.end
    if end is None:
        end = -1
    step = args.step
    if step is None:
        step = 1
    # Count number of frames, easy if not slicing
    # MDAnalysis trajectory slicing only includes up to end-1
    # This works the way we want it to if the whole trajectory is being included
    if start == 0 and end == -1 and step == 1:
        end = len(u.trajectory)
        number_frames = len(u.trajectory)
    elif end == -1:
        end = len(u.trajectory)
        number_frames = math.floor((end - start) / step)
    else:
        end = end + 1
        number_frames = math.floor((end - start) / step)
    logger.debug(f"Number of Frames: {number_frames}")

    # Create pandas data frame for results
    results_df = pd.DataFrame(columns=["Molecule ID", "Level", "Type", "Result"])
    residue_results_df = pd.DataFrame(
        columns=["Molecule ID", "Residue", "Type", "Result"]
    )

    # Reduce number of atoms in MDA universe to selection_string arg
    # (default all atoms included)
    if args.selection_string == "all":
        reduced_atom = u
    else:
        reduced_atom = MDAHelper.new_U_select_atom(u, args.selection_string)
        reduced_atom_name = f"{len(reduced_atom.trajectory)}_frame_dump_atom_selection"
        MDAHelper.write_universe(reduced_atom, reduced_atom_name)

    # Scan system for molecules and select levels (united atom, residue, polymer)
    # for each
    number_molecules, levels = LF.select_levels(reduced_atom, args.verbose)

    # Loop over molecules
    for molecule in range(number_molecules):
        # molecule data container of MDAnalysis Universe type for internal degrees
        # of freedom getting indices of first and last atoms in the molecule
        # assuming atoms are numbered consecutively and all atoms in a given
        # molecule are together
        index1 = reduced_atom.atoms.fragments[molecule].indices[0]
        index2 = reduced_atom.atoms.fragments[molecule].indices[-1]
        selection_string = f"index {index1}:{index2}"
        molecule_container = MDAHelper.new_U_select_atom(reduced_atom, selection_string)

        # Calculate entropy for each relevent level
        for level in levels[molecule]:
            if level == levels[molecule][-1]:
                highest_level = True
            else:
                highest_level = False

            if level == "united_atom":
                # loop over residues, report results per residue + total united atom
                # level. This is done per residue to reduce the size of the matrices -
                # amino acid resiudes have tens of united atoms but a whole protein
                # could have thousands. Doing the calculation per residue allows for
                # comparisons of contributions from different residues
                num_residues = len(molecule_container.residues)
                S_trans = 0
                S_rot = 0
                S_conf = 0

                for residue in range(num_residues):
                    # molecule data container of MDAnalysis Universe type for internal
                    # degrees of freedom getting indices of first and last atoms in the
                    # molecule assuming atoms are numbered consecutively and all atoms
                    # in a given molecule are together
                    index1 = molecule_container.residues[residue].atoms.indices[0]
                    index2 = molecule_container.residues[residue].atoms.indices[-1]
                    selection_string = f"index {index1}:{index2}"
                    residue_container = MDAHelper.new_U_select_atom(
                        molecule_container, selection_string
                    )
                    residue_heavy_atoms_container = MDAHelper.new_U_select_atom(
                        residue_container, "not name H*"
                    )  # only heavy atom dihedrals are relevant

                    # Vibrational entropy at every level
                    # Get the force and torque matrices for the beads at the relevant
                    # level

                    force_matrix, torque_matrix = LF.get_matrices(
                        residue_container,
                        level,
                        args.verbose,
                        start,
                        end,
                        step,
                        number_frames,
                        highest_level,
                    )

                    # Calculate the entropy from the diagonalisation of the matrices
                    S_trans_residue = EF.vibrational_entropy(
                        force_matrix, "force", args.temperature, highest_level
                    )
                    S_trans += S_trans_residue
                    logger.debug(f"S_trans_{level}_{residue} = {S_trans_residue}")
                    new_row = pd.DataFrame(
                        {
                            "Molecule ID": [molecule],
                            "Residue": [residue],
                            "Type": ["Transvibrational (J/mol/K)"],
                            "Result": [S_trans_residue],
                        }
                    )
                    residue_results_df = pd.concat(
                        [residue_results_df, new_row], ignore_index=True
                    )
                    data_logger.add_residue_data(
                        molecule, residue, "Transvibrational", S_trans_residue
                    )

                    S_rot_residue = EF.vibrational_entropy(
                        torque_matrix, "torque", args.temperature, highest_level
                    )
                    S_rot += S_rot_residue
                    logger.debug(f"S_rot_{level}_{residue} = {S_rot_residue}")
                    new_row = pd.DataFrame(
                        {
                            "Molecule ID": [molecule],
                            "Residue": [residue],
                            "Type": ["Rovibrational (J/mol/K)"],
                            "Result": [S_rot_residue],
                        }
                    )
                    residue_results_df = pd.concat(
                        [residue_results_df, new_row], ignore_index=True
                    )
                    data_logger.add_residue_data(
                        molecule, residue, "Rovibrational", S_rot_residue
                    )

                    # Conformational entropy based on atom dihedral angle distributions
                    # Gives entropy of conformations within each residue

                    # Get dihedral angle distribution
                    dihedrals = LF.get_dihedrals(residue_heavy_atoms_container, level)

                    # Calculate conformational entropy
                    S_conf_residue = EF.conformational_entropy(
                        residue_heavy_atoms_container,
                        dihedrals,
                        bin_width,
                        start,
                        end,
                        step,
                        number_frames,
                    )
                    S_conf += S_conf_residue
                    logger.debug(f"S_conf_{level}_{residue} = {S_conf_residue}")
                    new_row = pd.DataFrame(
                        {
                            "Molecule ID": [molecule],
                            "Residue": [residue],
                            "Type": ["Conformational (J/mol/K)"],
                            "Result": [S_conf_residue],
                        }
                    )
                    residue_results_df = pd.concat(
                        [residue_results_df, new_row], ignore_index=True
                    )
                    data_logger.add_residue_data(
                        molecule, residue, "Conformational", S_conf_residue
                    )

                # Print united atom level results summed over all residues
                logger.debug(f"S_trans_{level} = {S_trans}")
                new_row = pd.DataFrame(
                    {
                        "Molecule ID": [molecule],
                        "Level": [level],
                        "Type": ["Transvibrational (J/mol/K)"],
                        "Result": [S_trans],
                    }
                )

                results_df = pd.concat([results_df, new_row], ignore_index=True)

                data_logger.add_results_data(
                    molecule, level, "Transvibrational", S_trans
                )

                logger.debug(f"S_rot_{level} = {S_rot}")
                new_row = pd.DataFrame(
                    {
                        "Molecule ID": [molecule],
                        "Level": [level],
                        "Type": ["Rovibrational (J/mol/K)"],
                        "Result": [S_rot],
                    }
                )
                results_df = pd.concat([results_df, new_row], ignore_index=True)

                data_logger.add_results_data(molecule, level, "Rovibrational", S_rot)
                logger.debug(f"S_conf_{level} = {S_conf}")

                new_row = pd.DataFrame(
                    {
                        "Molecule ID": [molecule],
                        "Level": [level],
                        "Type": ["Conformational (J/mol/K)"],
                        "Result": [S_conf],
                    }
                )
                results_df = pd.concat([results_df, new_row], ignore_index=True)

                data_logger.add_results_data(molecule, level, "Conformational", S_conf)

            if level in ("polymer", "residue"):
                # Vibrational entropy at every level
                # Get the force and torque matrices for the beads at the relevant level
                force_matrix, torque_matrix = LF.get_matrices(
                    molecule_container,
                    level,
                    args.verbose,
                    start,
                    end,
                    step,
                    number_frames,
                    highest_level,
                )

                # Calculate the entropy from the diagonalisation of the matrices
                S_trans = EF.vibrational_entropy(
                    force_matrix, "force", args.temperature, highest_level
                )
                logger.debug(f"S_trans_{level} = {S_trans}")

                # Create new row as a DataFrame for Transvibrational
                new_row_trans = pd.DataFrame(
                    {
                        "Molecule ID": [molecule],
                        "Level": [level],
                        "Type": ["Transvibrational (J/mol/K)"],
                        "Result": [S_trans],
                    }
                )

                # Concatenate the new row to the DataFrame
                results_df = pd.concat([results_df, new_row_trans], ignore_index=True)

                # Calculate the entropy for Rovibrational
                S_rot = EF.vibrational_entropy(
                    torque_matrix, "torque", args.temperature, highest_level
                )
                logger.debug(f"S_rot_{level} = {S_rot}")

                # Create new row as a DataFrame for Rovibrational
                new_row_rot = pd.DataFrame(
                    {
                        "Molecule ID": [molecule],
                        "Level": [level],
                        "Type": ["Rovibrational (J/mol/K)"],
                        "Result": [S_rot],
                    }
                )

                # Concatenate the new row to the DataFrame
                results_df = pd.concat([results_df, new_row_rot], ignore_index=True)

                data_logger.add_results_data(
                    molecule, level, "Transvibrational", S_trans
                )
                data_logger.add_results_data(molecule, level, "Rovibrational", S_rot)

                # Note: conformational entropy is not calculated at the polymer level,
                # because there is at most one polymer bead per molecule so no dihedral
                # angles.

            if level == "residue":
                # Conformational entropy based on distributions of dihedral angles
                # of residues. Gives conformational entropy of secondary structure

                # Get dihedral angle distribution
                dihedrals = LF.get_dihedrals(molecule_container, level)
                # Calculate conformational entropy
                S_conf = EF.conformational_entropy(
                    molecule_container,
                    dihedrals,
                    bin_width,
                    start,
                    end,
                    step,
                    number_frames,
                )
                logger.debug(f"S_conf_{level} = {S_conf}")
                new_row = pd.DataFrame(
                    {
                        "Molecule ID": [molecule],
                        "Level": [level],
                        "Type": ["Conformational (J/mol/K)"],
                        "Result": [S_conf],
                    }
                )
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                data_logger.add_results_data(molecule, level, "Conformational", S_conf)

            # Orientational entropy based on network of neighbouring molecules,
            #  only calculated at the highest level (whole molecule)
            #    if highest_level:
            #        neigbours = LF.get_neighbours(reduced_atom, molecule)
            #        S_orient = EF.orientational_entropy(neighbours)
            #        print(f"S_orient_{level} = {S_orient}")
            #        new_row = pd.DataFrame({
            #            'Molecule ID': [molecule],
            #            'Level': [level],
            #            'Type':['Orientational (J/mol/K)'],
            #            'Result': [S_orient],})
            #        results_df = pd.concat([results_df, new_row], ignore_index=True)
            #        with open(args.output_file, "a") as out:
            #    print(molecule,
            #          "\t",
            #          level,
            #          "\tOrientational\t",
            #          S_orient,
            #          file=out)

        # Report total entropy for the molecule
        S_molecule = results_df[results_df["Molecule ID"] == molecule]["Result"].sum()
        logger.debug(f"S_molecule = {S_molecule}")
        new_row = pd.DataFrame(
            {
                "Molecule ID": [molecule],
                "Level": ["Molecule Total"],
                "Type": ["Molecule Total Entropy "],
                "Result": [S_molecule],
            }
        )
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        data_logger.add_results_data(
            molecule, level, "Molecule Total Entropy", S_molecule
        )
        data_logger.save_dataframes_as_json(
            results_df, residue_results_df, args.output_file
        )

    logger.info("Molecules:")
    data_logger.log_tables()


if __name__ == "__main__":

    main()
