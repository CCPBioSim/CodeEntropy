import MDAnalysis as mda
import numpy as nmp
import pandas as pd
from CodeEntropy import LevelFunctions as LF
#from CodeEntropy import EntropyFunctions as EF
from CodeEntropy import MDAUniverseHelper as MDAHelper
from CodeEntropy import poseidon


def main(arg_dict):
    """
    Main function for calculating the entropy of a system using the multiscale cell correlation method.

    Parameters
    ----------
    arg_dict : the input arguments
    """

    u = mda.Universe(arg_dict['tprfile'], arg_dict['trrfile'])

    # Create pandas data frame for results
    results_df = pd.DataFrame(columns=['Molecule ID', 'Level','Type', 'Result'])
    residue_results_df = pd.DataFrame(columns=['Molecule ID', 'Residue','Type', 'Result'])

    # Reduce time frames in MDA universe using start/end/step args (default all frames included)
    reduced_frame = MDAHelper.new_U_select_frame(u, arg_dict['start'], arg_dict['end'], arg_dict['step'])
    reduced_frame_name = f"{(len(reduced_frame.trajectory))}_frame_dump"
    reduced_frame_filename = MDAHelper.write_universe(reduced_frame, reduced_frame_name)

    # Reduce number of atoms in MDA universe to selection_string arg (default all atoms included)
    reduced_atom = MDAHelper.new_U_select_atom(reduced_frame, arg_dict['selection_string'])
    reduced_atom_name = f"{len(reduced_atom.trajectory)}_frame_dump_atom_selection"
    reduced_atom_filename = MDAHelper.write_universe(reduced_atom, reduced_atom_name)

    # Scan system for molecules and select levels (united atom, residue, polymer) for each
    number_molecules, levels = LF.select_levels(reduced_atom)

    number_frames = len(reduced_atom.trajectory)

    # Loop over molecules
    for molecule in range(number_molecules):
        # molecule data container for internal degrees of freedom
        print(molecule)
        molecule_dataContainer = reduced_atom.atoms.fragments[molecule]
        # molecule_dataContainer = MDAHelper.new_U_select_atom(reduced_atom, f"group group_name, group_name={group_name}")
        
        # Calculate entropy for each relevent level
        for level in levels[molecule]:

            if level == 'polymer' or 'residue':
                ## Vibrational entropy at every level
                # Get the force and torque matrices for the beads at the relevant level
                force_matrix, torque_matrix = LF.get_matrices(molecule_dataContainer, level, number_frames)

                # Calculate the entropy from the diagonalisation of the matrices
                S_trans = vibrational_entropy(force_matrix, "force", arg_dict['temper'],level)
                print(f"S_trans_{level} = {S_trans}")
                newRow = pd.DataFrame({'Molecule ID': [molecule], 'Level': ['{level}'],
                            'Type':['Transvibrational Entropy (J/mol/K)'],
                            'Result': [S_trans],})
                results_df = pd.concat([results_df, newRow], ignore_index=True)

                S_rot = vibrational_entropy(torque_matrix, "torque", arg_dict['temper'], level)
                print(f"S_rot_{level} = {S_rot}")
                newRow = pd.DataFrame({'Molecule ID': [molecule], 'Level': ['{level}'],
                            'Type':['Rovibrational Entropy (J/mol/K)'],
                            'Result': [S_rot],})
                results_df = pd.concat([results_df, newRow], ignore_index=True)

                # Note: conformational entropy is not calculated at the polymer level, because there is at most one polymer bead per molecule so no dihedral angles.

            if level == 'residue':     
                ## Conformational entropy based on distributions of dihedral angles of residues
                ## Gives conformational entropy of secondary structure

                # Get dihedral angle distribution
                dihedrals = LF.get_dihedrals(molecule_dataContainer, level)

                # Calculate conformational entropy
                S_conf = conformational_entropy(dihedrals,level)
                print(f"S_conf_{level} = {S_conf}")
                newRow = pd.DataFrame({'Molecule ID': [molecule], 'Level': ['{level}'],
                            'Type':['Conformational Entropy (J/mol/K)'],
                            'Result': [S_conf],})
                results_df = pd.concat([results_df, newRow], ignore_index=True)
            
            elif level == 'united_atom':
                # loop over residues, report results per residue + total united atom level
                num_residues = len(molecule_dataContainer.residues)
                S_trans = 0
                S_rot = 0
                for residue in range(num_residues):
                    residue_dataContainer = molecule_dataContainer.residue[residue]
                    ## Vibrational entropy at every level
                    # Get the force and torque matrices for the beads at the relevant level
                    force_matrix, torque_matrix = LF.get_matrices(residue_dataContainer, level)

                    # Calculate the entropy from the diagonalisation of the matrices
                    S_trans_residue = vibrational_entropy(force_matrix, "force", arg_dict['temper'],level)
                    S_trans += S_trans_residue
                    print(f"S_trans_{level}_{residue} = {S_trans_residue}")
                    newRow = pd.DataFrame({'Molecule ID': [molecule], 'Residue': ['{residue}'],
                            'Type':['Transvibrational Entropy (J/mol/K)'],
                            'Result': [S_trans_residue],})
                    residue_results_df = pd.concat([residue_results_df, newRow], ignore_index=True)

                    S_rot_residue = vibrational_entropy(torque_matrix, "torque", arg_dict['temper'], level)
                    S_rot += S_rot_residue
                    print(f"S_rot_{level}_{residue} = {S_rot_residue}")
                    newRow = pd.DataFrame({'Molecule ID': [molecule], 'Residue': ['{residue}'],
                            'Type':['Rovibrational Entropy (J/mol/K)'],
                            'Result': [S_rot_residue],})
                    residue_results_df = pd.concat([residue_results_df, newRow], ignore_index=True)
                    ## Conformational entropy based on atom dihedral angle distributions
                    ## Gives entropy of conformations within each residue
                
                    # Get dihedral angle distribution
                    dihedrals = LF.get_dihedrals(molecule_dataContainer, level)

                    # Calculate conformational entropy
                    S_conf_residue = conformational_entropy(dihedrals)
                    S_conf += S_conf_residue
                    print(f"S_conf_{level}_{residue} = {S_conf_residue}")
                    newRow = pd.DataFrame({'Molecule ID': [molecule], 'Level': ['{level}'],
                            'Type':['Conformational Entropy (J/mol/K)'],
                            'Result': [S_conf_residue],})
                    residue_results_df = pd.concat([residue_results_df, newRow], ignore_index=True)

                # Print united atom level results summed over all residues
                print(f"S_trans_{level} = {S_trans}")
                newRow = pd.DataFrame({'Molecule ID': [molecule], 'Level': ['{level}'],
                            'Type':['Transvibrational Entropy (J/mol/K)'],
                            'Result': [S_trans],})
                results_df = pd.concat([results_df, newRow], ignore_index=True)

                print(f"S_rot_{level} = {S_rot}")
                newRow = pd.DataFrame({'Molecule ID': [molecule], 'Level': ['{level}'],
                            'Type':['Rovibrational Entropy (J/mol/K)'],
                            'Result': [S_rot],})
                results_df = pd.concat([results_df, newRow], ignore_index=True)

                print(f"S_conf_{level} = {S_conf}")
                newRow = pd.DataFrame({'Molecule ID': [molecule], 'Level': ['{level}'],
                            'Type':['Conformational Entropy (J/mol/K)'],
                            'Result': [S_conf],})
                results_df = pd.concat([results_df, newRow], ignore_index=True)
                
        ## Orientational entropy based on network of neighbouring molecules, only calculated at the highest level (whole molecule)
        level = levels[molecule][-1]
        neighbours_dict, neighbours_array = NF.get_neighbours (molecule_dataContainer, reduced_atom)
        S_orient = orientational_entropy(neighbours_dict)
        print(f"S_orient_{level} = {S_orient}")
        newRow = pd.DataFrame({'Molecule ID': [molecule], 'Level': ['{level}'],
                            'Type':['Orientational Entropy (J/mol/K)'],
                            'Result': [S_orient],})
        results_df = pd.concat([results_df, newRow], ignore_index=True)

        # Report total entropy for the molecule
        molecule_data = results_df[results_df["Molecule ID"] == molecule]
        S_molecule = molecule_data["Result"].sum
        print(f"S_molecule = {S_molecule}")
        newRow = pd.DataFrame({'Molecule ID': [molecule], 'Level': ['Molecule Total'],
                            'Type':['Molecule Total Entropy (J/mol/K)'],
                            'Result': [S_molecule],})
        results_df = pd.concat([results_df, newRow], ignore_index=True)

# END main function

