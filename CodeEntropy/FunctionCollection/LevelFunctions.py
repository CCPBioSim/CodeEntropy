import numpy as nmp
import MDAnalysis as mds
from CodeEntropy.ClassCollection import BeadClasses as BC
from CodeEntropy.ClassCollection import ConformationEntity as CONF
from CodeEntropy.ClassCollection import ModeClasses
from CodeEntropy.ClassCollection import CustomDataTypes
from CodeEntropy.FunctionCollection import EntropyFunctions as EF
from CodeEntropy.FunctionCollection import CustomFunctions as CF
from CodeEntropy.FunctionCollection import GeometricFunctions as GF
from CodeEntropy.FunctionCollection import UnitsAndConversions as UAC
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.IO import Writer
from CodeEntropy.FunctionCollection import UnitsAndConversions as CONST

def select_levels(arg_DataContainer):
    """
    Function to read input system and identify the number of molecules and the levels (i.e. united atom, residue and/or polymer) that should be used.

    Input
    -----
       arg_DataContainer : MDAnalysis universe object containing the system of interest

    Returns
    -------
       number_molecules : integer
       levels : array of strings for each molecule
    """

    # fragments is MDAnalysis terminology for what chemists would call molecules
    number_molecules = arg_DataContainer.n_fragments
    print("The number of molecules is {}.".format(number_molecules))
    fragments = arg_DataContainer.atoms.fragments
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

def get_matrices(arg_DataContainer, level):
    """
    Function to create the force matrix needed for the transvibrational entropy calculation and the torque matrix for the rovibrational entropy calculation.

    Input
    -----
        arg_DataContainer : CodeEntropy.ClassCollection.DataContainer DataContainer type with the information on the molecule of interest.
        level : string, which of the polymer, residue, or united atom levels are the matrices for.

    Returns
    -------
        force_matrix :
        torque_matrix :
    """

    # number of frames
    number_frames = len(arg_DataContainer.trajSnapshots)
   
    ## Make beads
    list_of_beads = GF.get_beads(arg_DataContainer, level)

    ## Calculate forces/torques for each bead
    for bead in list_of_beads:
        for frame in range(number_frames):
            ## Set up axes
            # translation and rotation use different axes
            # how the axes are defined depends on the level
            trans_axes, rot_axes = GF.get_axes(arg_DataContainer, level, bead.residue.ix, frame)
        
            ## Sort out coordinates, forces, and torques for each atom in the bead
            weighted_forces[bead][frame] = GF.get_weighted_forces(arg_DataContainer, bead, trans_axes, frame)
            weighted_torques[bead][frame] = GF.get_weighted_torques(arg_DataContainer, bead, rot_axes, frame)

    ## Make covariance matrices - looping over pairs of beads
    # number of beads
    num_beads = len(list_of_beads)
    # list of pairs of indices
    indexPairList = [(i,j) for i in range(num_beads) for j in range(num_beads)]

    for i, j in indexPairList:
        # for each pair of beads (but reducing effort because the matrix for [i][j] is the transpose of the one for [j][i])
        if i <= j:
            # calculate the force covariance segment of the matrix
            force_submatrix[i][j] = GF.create_submatrix(i, j, weighted_forces[i], weighted_forces[j])
            force_submatrix[j][i] = nmp.transpose(force_submatrix[i][j])

            # calculate the torque covariance segment of the matrix
            torque_submatrix[i][j] = GF.create_submatrix(i, j, weighted_torques[i], weighted_torques[j])
            torque_submatrix[j][i] = nmp.transpose(torque_submatrix[i][j])
    ## Tidy up
    # use nmp.block to make submatrices into one matrix

    # fliter zeros to remove any rows/columns that are all zero

    return force_matrix, torque_matrix
# END

def get_dihedrals(arg_hostDataContainer, level):
    """
    Define the set of dihedrals for use in the conformational entropy function.
    If residue level, the dihedrals are defined from the atoms (4 bonded atoms for 1 dihedral).
    If polymer level, use the bonds between residues to cast dihedrals.
    Note: not using improper dihedrals only ones with 4 atoms/residues in a linear arrangement.

    Input
    -----
    arg_hostDataContainer : system information
    level : level of the hierarchy (should be residue or polymer)

    Output
    ------
    dihedrals : set of dihedrals
    """
    # Start with empty array
    dihedrals = []

    # if residue level, read dihedrals from MDAnalysis universe
    if level == "residue":
        dihedrals.append(arg_hostDataContainer.dihedrals)

    # if polymer level, looking for dihedrals involving residues
    if level == "polymer":
        num_residues = len(arg_hostDataContainer.residues)
        if num_residues < 4:
            print("no polymer level dihedrals")

        else:
        # find bonds between residues N-3:N-2 and N-1:N
            for residue in range(3, num_residues):
                # Using MDAnalysis selection, assuming only one covalent bond between neighbouring residues
                # TODO test selection syntax
                # TODO not written for branched polymers
                atom1 = arg_hostDataContainer.select(f"residue {residue}-3 bonded residue {residue}-2" )
                atom2 = arg_hostDataContainer.select(f"residue {residue}-2 bonded residue {residue}-3" )
                atom3 = arg_hostDataContainer.select(f"residue {residue}-1 bonded residue {residue}" )
                atom4 = arg_hostDataContainer.select(f"residue {residue} bonded residue {residue}-1" )
                atom_group = atom1 + atom2 + atom3 + atom4
                dihedrals.append(atom_group.dihedral)

    return dihedrals
#END
