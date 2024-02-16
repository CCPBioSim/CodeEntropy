import numpy as nmp
import MDAnalysis as mda
from CodeEntropy import EntropyFunctions as EF
from CodeEntropy import CustomFunctions as CF
from CodeEntropy import GeometricFunctions as GF
from CodeEntropy import UnitsAndConversions as UAC
from CodeEntropy import Utils
from CodeEntropy import Writer

def select_levels(data_container):
    """
    Function to read input system and identify the number of molecules and the levels (i.e. united atom, residue and/or polymer) that should be used.
    The level refers to the size of the bead (atom or collection of atoms) that will be used in the entropy calculations.

    Input
    -----
       arg_DataContainer : MDAnalysis universe object containing the system of interest

    Returns
    -------
       number_molecules : integer
       levels : array of strings for each molecule
    """

    # fragments is MDAnalysis terminology for what chemists would call molecules
    number_molecules = len(data_container.atoms.fragments)
    print("The number of molecules is {}.".format(number_molecules))
    fragments = data_container.atoms.fragments
    levels = [[] for _ in range(number_molecules)]

    for molecule in range(number_molecules):
        levels[molecule].append("united_atom") # every molecule has at least one atom

        atoms_in_fragment = fragments[molecule].select_atoms("not name H*")
        number_residues = len(atoms_in_fragment.residues)

        # if a fragment has more than one atom assign residue level
        if len(atoms_in_fragment) > 1:
            levels[molecule].append("residue")

            #if assigned residue level and there is more than one residue assign polymer level
            if number_residues > 1:
                levels[molecule].append("polymer")

    print(levels)

    return number_molecules, levels

def get_matrices(data_container, level):
    """
    Function to create the force matrix needed for the transvibrational entropy calculation and the torque matrix for the rovibrational entropy calculation.

    Input
    -----
        data_container : MDAnalysis universe type with the information on the molecule of interest.
        level : string, which of the polymer, residue, or united atom levels are the matrices for.

    Returns
    -------
        force_matrix : force covariance matrix for transvibrational entropy
        torque_matrix : torque convariance matrix for rovibrational entropy
    """
   
    ## Make beads
    list_of_beads = GF.get_beads(data_container, level)
    
    # number of beads and frames in trajectory
    number_beads = len(list_of_beads)
    number_frames = len(data_container.trajectory)

    # initialize force and torque arrays
    weighted_forces = [[0 for x in range(number_frames)] for y in range(number_beads)]
    weighted_torques = [[0 for x in range(number_frames)] for y in range(number_beads)]

    ## Calculate forces/torques for each bead
    for bead_index in range(number_beads):
        for timestep in data_container.trajectory:
            ## Set up axes
            # translation and rotation use different axes
            # how the axes are defined depends on the level
            trans_axes, rot_axes = GF.get_axes(data_container, level, bead_index)
       
            ## Sort out coordinates, forces, and torques for each atom in the bead
            weighted_forces[bead_index][timestep.frame] = GF.get_weighted_forces(data_container, list_of_beads[bead_index], trans_axes)
            weighted_torques[bead_index][timestep.frame] = GF.get_weighted_torques(data_container, list_of_beads[bead_index], rot_axes)

    ## Make covariance matrices - looping over pairs of beads
    # list of pairs of indices
    indexPairList = [(i,j) for i in range(number_beads) for j in range(number_beads)]

    force_submatrix = [[0 for x in range(number_beads)] for y in range(number_beads)]
    torque_submatrix = [[0 for x in range(number_beads)] for y in range(number_beads)]

    for i, j in indexPairList:
        # for each pair of beads (but reducing effort because the matrix for [i][j] is the transpose of the one for [j][i])
        if i <= j:
            # calculate the force covariance segment of the matrix
            force_submatrix[i][j] = GF.create_submatrix(i, j, weighted_forces[i], weighted_forces[j], number_frames)
            force_submatrix[j][i] = nmp.transpose(force_submatrix[i][j])

            # calculate the torque covariance segment of the matrix
            torque_submatrix[i][j] = GF.create_submatrix(i, j, weighted_torques[i], weighted_torques[j], number_frames)
            torque_submatrix[j][i] = nmp.transpose(torque_submatrix[i][j])
    
    ## Tidy up
    # use nmp.block to make submatrices into one matrix
    force_matrix = nmp.block([   [  force_submatrix[i][j] for j in range(number_beads)  ] for i in range(number_beads)   ] )

    torque_matrix = nmp.block([   [  torque_submatrix[i][j] for j in range(number_beads)  ] for i in range(number_beads)   ] )

    # fliter zeros to remove any rows/columns that are all zero
    force_matrix = CF.filter_zero_rows_columns(force_matrix)
    torque_maxtrix = CF.filter_zero_rows_columns(torque_matrix)

    ### TODO temporary print for testing matrices
    with open("matrix.out", "a") as f:
        print("force_matrix \n", file=f)
        print(force_matrix, file=f)
        print("torque_matrix \n", file=f)
        print(torque_matrix, file=f)

    return force_matrix, torque_matrix
# END

def get_dihedrals(data_container, level):
    """
    Define the set of dihedrals for use in the conformational entropy function.
    If residue level, the dihedrals are defined from the atoms (4 bonded atoms for 1 dihedral).
    If polymer level, use the bonds between residues to cast dihedrals.
    Note: not using improper dihedrals only ones with 4 atoms/residues in a linear arrangement.

    Input
    -----
    data_container : system information
    level : level of the hierarchy (should be residue or polymer)

    Output
    ------
    dihedrals : set of dihedrals
    """
    # Start with empty array
    dihedrals = []

    # if united atom level, read dihedrals from MDAnalysis universe
    if level == "united_atom":
        # only use dihedrals made of heavy atoms
        heavy_atom_group = data_container.select_atoms('not name H*')
        dihedrals = heavy_atom_group.dihedrals

    # if residue level, looking for dihedrals involving residues
    if level == "residue":
        num_residues = len(data_container.residues)
        if num_residues < 4:
            print("no residue level dihedrals")

        else:
        # find bonds between residues N-3:N-2 and N-1:N
            for residue in range(3, num_residues):
                # Using MDAnalysis selection, assuming only one covalent bond between neighbouring residues
                # TODO test selection syntax
                # TODO not written for branched polymers
                atom1 = data_container.select(f"residue {residue}-3 bonded residue {residue}-2" )
                atom2 = data_container.select(f"residue {residue}-2 bonded residue {residue}-3" )
                atom3 = data_container.select(f"residue {residue}-1 bonded residue {residue}" )
                atom4 = data_container.select(f"residue {residue} bonded residue {residue}-1" )
                atom_group = atom1 + atom2 + atom3 + atom4
                dihedrals.append(atom_group.dihedral)

    return dihedrals
#END
