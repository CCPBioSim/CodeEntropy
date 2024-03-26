import MDAnalysis as mda
import numpy as nmp
import pandas as pd
from CodeEntropy import LevelFunctions as LF
from CodeEntropy import EntropyFunctions as EF
from CodeEntropy import MDAUniverseHelper as MDAHelper
from CodeEntropy import poseidon

def main(arg_dict):
    """
    Main function for calculating the entropy of a system using the multiscale cell correlation method.

    Parameters
    ----------
    arg_dict : the input arguments
    """

    u = arg_dict['universe']

    # Create pandas data frame for results
    results_df = pd.DataFrame(columns=['Molecule ID', 'Level','Type', 'Result'])
    residue_results_df = pd.DataFrame(columns=['Molecule ID', 'Residue','Type', 'Result'])

    # printing headings for output files
    with open(arg_dict['outfile'], "a") as out:
        print("Molecule ID\tLevel\tType\tResult\n", file=out)

    with open(arg_dict['resfile'], "a") as res:
        print("Molecule ID\tResidue\tType\tResult\n", file=res)

    # Reduce time frames in MDA universe using start/end/step args (default all frames included)
    reduced_frame = MDAHelper.new_U_select_frame(u, arg_dict['start'], arg_dict['end'], arg_dict['step'])
    reduced_frame_name = f"{(len(reduced_frame.trajectory))}_frame_dump"
    MDAHelper.write_universe(reduced_frame, reduced_frame_name)

    # Reduce number of atoms in MDA universe to selection_string arg (default all atoms included)
    reduced_atom = MDAHelper.new_U_select_atom(reduced_frame, arg_dict['selection_string'])
    reduced_atom_name = f"{len(reduced_atom.trajectory)}_frame_dump_atom_selection"
    MDAHelper.write_universe(reduced_atom, reduced_atom_name)

    # Scan system for molecules and select levels (united atom, residue, polymer) for each
    number_molecules, levels = LF.select_levels(reduced_atom, arg_dict['verbose'])

    number_frames = len(reduced_atom.trajectory)

    # Loop over molecules
    for molecule in range(number_molecules):
        # molecule data container of MDAnalysis Universe type for internal degrees of freedom
        # getting indices of first and last atoms in the molecule
        # assuming atoms are numbered consecutively and all atoms in a given molecule are together
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

            if level == 'united_atom':
                # loop over residues, report results per residue + total united atom level
                # This is done per residue to reduce the size of the matrices -
                # amino acid resiudes have tens of united atoms but a whole protein could have thousands
                # Doing the calculation per residue allows for comparisons of contributions from different residues
                num_residues = len(molecule_container.residues)
                S_trans = 0
                S_rot = 0
                S_conf = 0
                for residue in range(num_residues):
                    # molecule data container of MDAnalysis Universe type for internal degrees of freedom
                    # getting indices of first and last atoms in the molecule
                    # assuming atoms are numbered consecutively and all atoms in a given molecule are together
                    index1 = molecule_container.residues[residue].atoms.indices[0]
                    index2 = molecule_container.residues[residue].atoms.indices[-1]
                    selection_string = f"index {index1}:{index2}"
                    residue_container = MDAHelper.new_U_select_atom(molecule_container, selection_string)

                    ## Vibrational entropy at every level
                    # Get the force and torque matrices for the beads at the relevant level
                    force_matrix, torque_matrix = LF.get_matrices(residue_container, level, arg_dict['verbose'])

                    # Calculate the entropy from the diagonalisation of the matrices
                    S_trans_residue = EF.vibrational_entropy(force_matrix, "force", arg_dict['temper'],highest_level)
                    S_trans += S_trans_residue
                    print(f"S_trans_{level}_{residue} = {S_trans_residue}")
                    new_row = pd.DataFrame({'Molecule ID': [molecule], 'Residue': [residue],
                            'Type':['Transvibrational (J/mol/K)'],
                            'Result': [S_trans_residue],})
                    residue_results_df = pd.concat([residue_results_df, new_row], ignore_index=True)
                    with open(arg_dict['resfile'], "a") as res:
                        print(new_row, file=res)


                    S_rot_residue = EF.vibrational_entropy(torque_matrix, "torque", arg_dict['temper'], highest_level)
                    S_rot += S_rot_residue
                    print(f"S_rot_{level}_{residue} = {S_rot_residue}")
                    new_row = pd.DataFrame({'Molecule ID': [molecule], 'Residue': [residue],
                            'Type':['Rovibrational (J/mol/K)'],
                            'Result': [S_rot_residue],})
                    residue_results_df = pd.concat([residue_results_df, new_row], ignore_index=True)
                    with open(arg_dict['resfile'], "a") as res:
                        print(new_row, file=res)


                    ## Conformational entropy based on atom dihedral angle distributions
                    ## Gives entropy of conformations within each residue

                    # Get dihedral angle distribution
                    dihedrals = LF.get_dihedrals(residue_container, level)

                    # Calculate conformational entropy
                    S_conf_residue = EF.conformational_entropy(residue_container, dihedrals, number_frames)
                    S_conf += S_conf_residue
                    print(f"S_conf_{level}_{residue} = {S_conf_residue}")
                    new_row = pd.DataFrame({'Molecule ID': [molecule], 'Residue': [residue],
                            'Type':['Conformational (J/mol/K)'],
                            'Result': [S_conf_residue],})
                    residue_results_df = pd.concat([residue_results_df, new_row], ignore_index=True)
                    with open(arg_dict['resfile'], "a") as res:
                        print(new_row, file=res)


                # Print united atom level results summed over all residues
                print(f"S_trans_{level} = {S_trans}")
                new_row = pd.DataFrame({'Molecule ID': [molecule], 'Level': [level],
                            'Type':['Transvibrational (J/mol/K)'],
                            'Result': [S_trans],})
                with open(arg_dict['outfile'], "a") as out:
                    print(new_row, file=out)

                results_df = pd.concat([results_df, new_row], ignore_index=True)

                print(f"S_rot_{level} = {S_rot}")
                new_row = pd.DataFrame({'Molecule ID': [molecule], 'Level': [level],
                            'Type':['Rovibrational (J/mol/K)'],
                            'Result': [S_rot],})
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                with open(arg_dict['outfile'], "a") as out:
                    print(new_row, file=out)


                print(f"S_conf_{level} = {S_conf}")
                new_row = pd.DataFrame({'Molecule ID': [molecule], 'Level': [level],
                            'Type':['Conformational (J/mol/K)'],
                            'Result': [S_conf],})
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                with open(arg_dict['outfile'], "a") as out:
                    print(new_row, file=out)


                ## End united atom vibrational and conformational calculations ##

            if level in ('polymer', 'residue'):
                ## Vibrational entropy at every level
                # Get the force and torque matrices for the beads at the relevant level
                force_matrix, torque_matrix = LF.get_matrices(molecule_container, level, arg_dict['verbose'])

                # Calculate the entropy from the diagonalisation of the matrices
                S_trans = EF.vibrational_entropy(force_matrix, "force", arg_dict['temper'],highest_level)
                print(f"S_trans_{level} = {S_trans}")
                new_row = pd.DataFrame({'Molecule ID': [molecule], 'Level': [level],
                            'Type':['Transvibrational (J/mol/K)'],
                            'Result': [S_trans],})
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                with open(arg_dict['outfile'], "a") as out:
                    print(new_row, file=out)


                S_rot = EF.vibrational_entropy(torque_matrix, "torque", arg_dict['temper'], highest_level)
                print(f"S_rot_{level} = {S_rot}")
                new_row = pd.DataFrame({'Molecule ID': [molecule], 'Level': [level],
                            'Type':['Rovibrational (J/mol/K)'],
                            'Result': [S_rot],})
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                with open(arg_dict['outfile'], "a") as out:
                    print(new_row, file=out)


                # Note: conformational entropy is not calculated at the polymer level,
                # because there is at most one polymer bead per molecule so no dihedral angles.

            if level == 'residue':
                ## Conformational entropy based on distributions of dihedral angles of residues
                ## Gives conformational entropy of secondary structure

                # Get dihedral angle distribution
                dihedrals = LF.get_dihedrals(molecule_container, level)
                # Calculate conformational entropy
                S_conf = EF.conformational_entropy(molecule_container, dihedrals, number_frames)
                print(f"S_conf_{level} = {S_conf}")
                new_row = pd.DataFrame({'Molecule ID': [molecule], 'Level': [level],
                            'Type':['Conformational (J/mol/K)'],
                            'Result': [S_conf],})
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                with open(arg_dict['outfile'], "a") as out:
                    print(new_row, file=out)


            ## Orientational entropy based on network of neighbouring molecules,
            #  only calculated at the highest level (whole molecule)
   #        if highest_level:
   #            neigbours = LF.get_neighbours(reduced_atom, molecule)
   #            S_orient = EF.orientational_entropy(neighbours)
   #            print(f"S_orient_{level} = {S_orient}")
   #            new_row = pd.DataFrame({'Molecule ID': [molecule], 'Level': [level],
   #                         'Type':['Orientational (J/mol/K)'],
   #                         'Result': [S_orient],})
   #            results_df = pd.concat([results_df, new_row], ignore_index=True)
   #            with open(arg_dict['outfile'], "a") as out:
   #                print(new_row, file=out)

        # Report total entropy for the molecule
        S_molecule = results_df[results_df["Molecule ID"] == molecule]['Result'].sum()
        print(f"S_molecule = {S_molecule}")
        new_row = pd.DataFrame({'Molecule ID': [molecule], 'Level': ['Molecule Total'],
                            'Type':['Molecule Total Entropy (J/mol/K)'],
                            'Result': [S_molecule],})
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        with open(arg_dict['outfile'], "a") as out:
            print(new_row, file=out)


# END main function
