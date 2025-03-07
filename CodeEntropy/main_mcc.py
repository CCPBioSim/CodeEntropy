import argparse
import math

import MDAnalysis as mda

# import numpy as np
import pandas as pd

from CodeEntropy import EntropyFunctions as EF
from CodeEntropy import LevelFunctions as LF
from CodeEntropy import MDAUniverseHelper as MDAHelper

# from datetime import datetime


def main():
    """
    Main function for calculating the entropy of a system using the multiscale cell
    correlation method.
    """

    try:
        parser = argparse.ArgumentParser(
            description="""
        CodeEntropy-POSEIDON is a tool to compute entropy using the
        multiscale-cell-correlation (MCC) theory and force/torque covariance
        methods with the ablity to compute solvent entropy.
        Version:
            0.3.1;

        Authors:
            Arghya Chakravorty (arghya90),
            Jas Kalayan (jkalayan),
            Donald Chang,
            Sarah Fegan
            Ioana Papa;

        Output:
            *.csv = results from different calculateion,
            *.pkl - Pickled reduced universe for further analysis,
            *.out - detailed output such as matrix and spectra"""
        )

        parser.add_argument(
            "-f",
            "--top_traj_file",
            required=True,
            dest="filePath",
            action="store",
            nargs="+",
            help="Path to Structure/topology file (AMBER PRMTOP, GROMACS TPR which "
            "contains topology and dihedral information) followed by Trajectory "
            "file(s) (AMBER NETCDF or GROMACS TRR) you will need to output the "
            "coordinates and forces to the same file. Required.",
        )
        parser.add_argument(
            "-l",
            "--selectString",
            action="store",
            dest="selection_string",
            type=str,
            default="all",
            help="Selection string for CodeEntropy such as protein or resid, refer to "
            "MDAnalysis.select_atoms for more information.",
        )
        parser.add_argument(
            "-b",
            "--begin",
            action="store",
            dest="start",
            help="Start analysing the trajectory from this frame index. Defaults to 0",
            default=0,
            type=int,
        )
        parser.add_argument(
            "-e",
            "--end",
            action="store",
            dest="end",
            help="Stop analysing the trajectory at this frame index. Defaults to -1 "
            "(end of trajectory file)",
            default=-1,
            type=int,
        )
        parser.add_argument(
            "-d",
            "--step",
            action="store",
            dest="step",
            help="interval between two consecutive frames to be read index. "
            "Defaults to 1",
            default=1,
            type=int,
        )
        parser.add_argument(
            "-n",
            "--bin_width",
            action="store",
            dest="bin_width",
            default=30,
            type=int,
            help="Bin width in degrees for making the histogram of the dihedral angles "
            "for the conformational entropy. Default: 30",
        )
        parser.add_argument(
            "-k",
            "--tempra",
            action="store",
            dest="temp",
            help="Temperature for entropy calculation (K). Default to 298.0 K",
            default=298.0,
            type=float,
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store",
            dest="verbose",
            default=False,
            type=bool,
            help="True/False flag for noisy or quiet output. Default: False",
        )
        parser.add_argument(
            "-t",
            "--thread",
            action="store",
            dest="thread",
            help="How many multiprocess to use. Default 1 for single core execution.",
            default=1,
            type=int,
        )
        parser.add_argument(
            "-o",
            "--out",
            action="store",
            dest="outfile",
            default="outfile.out",
            help="Name of the file where the output will be written. "
            "Default: outfile.out",
        )
        parser.add_argument(
            "-r",
            "--resout",
            action="store",
            dest="resfile",
            default="res_outfile.out",
            help="Name of the file where the residue entropy output will be written. "
            "Default: res_outfile.out",
        )
        parser.add_argument(
            "-m",
            "--mout",
            action="store",
            dest="moutfile",
            default=None,
            help="Name of the file where certain matrices will be written "
            "(default: None).",
        )

        parser.add_argument(
            "-c",
            "--cutShell",
            action="store",
            dest="cutShell",
            default=None,
            type=float,
            help="include cutoff shell analysis, add cutoff distance in angstrom "
            "Default None will ust the RAD Algorithm",
        )
        parser.add_argument(
            "-p",
            "--pureAtomNum",
            action="store",
            dest="puteAtomNum",
            default=1,
            type=int,
            help="Reference molecule resid for system of pure liquid. " "Default to 1",
        )
        parser.add_argument(
            "-x",
            "--excludedResnames",
            dest="excludedResnames",
            action="store",
            nargs="+",
            default=None,
            help="exclude a list of molecule names from nearest non-like analysis. "
            "Default: None. Multiples are gathered into list.",
        )
        parser.add_argument(
            "-w",
            "--water",
            dest="waterResnames",
            action="store",
            default="WAT",
            nargs="+",
            help="resname for water molecules. "
            "Default: WAT. Multiples are gathered into list.",
        )
        parser.add_argument(
            "-s",
            "--solvent",
            dest="solventResnames",
            action="store",
            nargs="+",
            default=None,
            help="include resname of solvent molecules (case-sensitive) "
            "Default: None. Multiples are gathered into list.",
        )
        parser.add_argument(
            "--solContact",
            action="store_true",
            dest="doSolContact",
            default=False,
            help="Do solute contact calculation",
        )

        args = parser.parse_args()
    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the arguments")
        raise

    # REPLACE INPUTS
    print("printing all input")
    for arg in vars(args):
        print(" {} {}".format(arg, getattr(args, arg) or ""))

    # startTime = datetime.now()

    # Get topology and trajectory file names and make universe
    tprfile = args.filePath[0]
    trrfile = args.filePath[1:]
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
    if start == 0 and end == -1 and step == 1:
        number_frames = len(u.trajectory)
    elif end == -1:
        end = len(u.trajectory)
        number_frames = math.floor((end - start) / step) + 1
    else:
        number_frames = math.floor((end - start) / step) + 1
    print(number_frames)

    # Create pandas data frame for results
    results_df = pd.DataFrame(columns=["Molecule ID", "Level", "Type", "Result"])
    residue_results_df = pd.DataFrame(
        columns=["Molecule ID", "Residue", "Type", "Result"]
    )

    # printing headings for output files
    with open(args.outfile, "a") as out:
        print("Molecule\tLevel\tType\tResult (J/mol/K)\n", file=out)

    with open(args.resfile, "a") as res:
        print("Molecule\tResidue\tType\tResult (J/mol/K)\n", file=res)

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
                        force_matrix, "force", args.temp, highest_level
                    )
                    S_trans += S_trans_residue
                    print(f"S_trans_{level}_{residue} = {S_trans_residue}")
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
                    with open(args.resfile, "a") as res:
                        print(
                            molecule,
                            "\t",
                            residue,
                            "\tTransvibration\t",
                            S_trans_residue,
                            file=res,
                        )

                    S_rot_residue = EF.vibrational_entropy(
                        torque_matrix, "torque", args.temp, highest_level
                    )
                    S_rot += S_rot_residue
                    print(f"S_rot_{level}_{residue} = {S_rot_residue}")
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
                    with open(args.resfile, "a") as res:
                        #  print(new_row, file=res)
                        print(
                            molecule,
                            "\t",
                            residue,
                            "\tRovibrational \t",
                            S_rot_residue,
                            file=res,
                        )

                    # Conformational entropy based on atom dihedral angle distributions
                    # Gives entropy of conformations within each residue

                    # Get dihedral angle distribution
                    dihedrals = LF.get_dihedrals(residue_container, level)

                    # Calculate conformational entropy
                    S_conf_residue = EF.conformational_entropy(
                        residue_container,
                        dihedrals,
                        bin_width,
                        start,
                        end,
                        step,
                        number_frames,
                    )
                    S_conf += S_conf_residue
                    print(f"S_conf_{level}_{residue} = {S_conf_residue}")
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
                    with open(args.resfile, "a") as res:
                        print(
                            molecule,
                            "\t",
                            residue,
                            "\tConformational\t",
                            S_conf_residue,
                            file=res,
                        )

                # Print united atom level results summed over all residues
                print(f"S_trans_{level} = {S_trans}")
                new_row = pd.DataFrame(
                    {
                        "Molecule ID": [molecule],
                        "Level": [level],
                        "Type": ["Transvibrational (J/mol/K)"],
                        "Result": [S_trans],
                    }
                )
                with open(args.outfile, "a") as out:
                    print(
                        molecule, "\t", level, "\tTransvibration\t", S_trans, file=out
                    )

                results_df = pd.concat([results_df, new_row], ignore_index=True)

                print(f"S_rot_{level} = {S_rot}")
                new_row = pd.DataFrame(
                    {
                        "Molecule ID": [molecule],
                        "Level": [level],
                        "Type": ["Rovibrational (J/mol/K)"],
                        "Result": [S_rot],
                    }
                )
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                with open(args.outfile, "a") as out:
                    print(molecule, "\t", level, "\tRovibrational \t", S_rot, file=out)

                print(f"S_conf_{level} = {S_conf}")
                new_row = pd.DataFrame(
                    {
                        "Molecule ID": [molecule],
                        "Level": [level],
                        "Type": ["Conformational (J/mol/K)"],
                        "Result": [S_conf],
                    }
                )
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                with open(args.outfile, "a") as out:
                    print(molecule, "\t", level, "\tConformational\t", S_conf, file=out)

                # End united atom vibrational and conformational calculations #

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
                    force_matrix, "force", args.temp, highest_level
                )
                print(f"S_trans_{level} = {S_trans}")
                new_row = pd.DataFrame(
                    {
                        "Molecule ID": [molecule],
                        "Level": [level],
                        "Type": ["Transvibrational (J/mol/K)"],
                        "Result": [S_trans],
                    }
                )
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                with open(args.outfile, "a") as out:
                    print(
                        molecule, "\t", level, "\tTransvibrational\t", S_trans, file=out
                    )

                S_rot = EF.vibrational_entropy(
                    torque_matrix, "torque", args.temp, highest_level
                )
                print(f"S_rot_{level} = {S_rot}")
                new_row = pd.DataFrame(
                    {
                        "Molecule ID": [molecule],
                        "Level": [level],
                        "Type": ["Rovibrational (J/mol/K)"],
                        "Result": [S_rot],
                    }
                )
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                with open(args.outfile, "a") as out:
                    print(molecule, "\t", level, "\tRovibrational \t", S_rot, file=out)

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
                print(f"S_conf_{level} = {S_conf}")
                new_row = pd.DataFrame(
                    {
                        "Molecule ID": [molecule],
                        "Level": [level],
                        "Type": ["Conformational (J/mol/K)"],
                        "Result": [S_conf],
                    }
                )
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                with open(args.outfile, "a") as out:
                    print(molecule, "\t", level, "\tConformational\t", S_conf, file=out)

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
            #        with open(args.outfile, "a") as out:
            #    print(molecule,
            #          "\t",
            #          level,
            #          "\tOrientational\t",
            #          S_orient,
            #          file=out)

        # Report total entropy for the molecule
        S_molecule = results_df[results_df["Molecule ID"] == molecule]["Result"].sum()
        print(f"S_molecule = {S_molecule}")
        new_row = pd.DataFrame(
            {
                "Molecule ID": [molecule],
                "Level": ["Molecule Total"],
                "Type": ["Molecule Total Entropy "],
                "Result": [S_molecule],
            }
        )
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        with open(args.outfile, "a") as out:
            print(molecule, "\t Molecule\tTotal Entropy\t", S_molecule, file=out)


# END main function

if __name__ == "__main__":

    main()
