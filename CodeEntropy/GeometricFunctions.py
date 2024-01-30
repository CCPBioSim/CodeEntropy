import numpy as nmp
from CodeEntropy import CustomFunctions as CF 

def get_beads(arg_dataContainer, level):
    """
    Function to define beads depending on the level in the hierarchy.

    Input
    -----
    arg_dataContainer : the MDAnalysis universe
    level : the heirarchy level (polymer, residue, or united atom)

    Output
    ------
    list_of_beads : the relevent beads
    """

    if level == "polymer":
        list_of_beads = arg_dataContainer.atoms.fragments

    if level == "residue":
        list_of_beads = arg_dataContainer.residues

    if level == "united_atom":
        list_of_beads = []
        heavy_atoms = arg_dataContainer.select_atoms("not name H*")
        for atom in heavy_atoms:
            list_of_beads.append(arg_dataContainer.select_atoms(f"index {atom} or (name H* and bonded index {atom})"))

    return list_of_beads
#END

def get_axes(arg_dataContainer, level, index=0, frame=0):
    """
    Function to set the translational and rotational axes.
    The translational axes are based on the principal axes of the unit one level larger than the level we are interested in (except for the polymer level where there is no larger unit).
    The rotational axes use the covalent links between residues or atoms where possible to define the axes, or if the unit is not bonded to others of the same level the prinicpal axes of the unit are used.

    Input
    -----
    arg_dataContainer : the information about the molecule and trajectory
    level : the level (united atom, residue, or polymer) of interest
    index : residue index (integer)
    frame : frame index (integer)

    Output
    ------
    trans_axes : translational axes
    rot_axes : rotational axes
    """

    if level == "polymer":
        # for polymer use principle axis for both translation and rotation
        trans_axes = arg_dataContainer.principal_axes()
        rot_axes = arg_dataContainer.principal_axes()

    if level == "residue":
        ## Translation
        # for residues use principal axes of whole molecule for translation
        trans_axes = arg_dataContainer.principal_axes()

        ## Rotation
        # find bonds between atoms in residue of interest and other residues
        # we are assuming bonds only exist between adjacent residues (linear chains of residues)
        # TODO refine selection so that it will work for branched polymers
        atom_set = arg_dataContainer.select_atoms(f"(resid {index}-1 or resid {index}+1) and bonded resid {index}")

        if len(atom_set) == 0:
            # if no bonds to other residues use pricipal axes of residue
            rot_axes = arg_dataContainer.residue[index].principal_axes()

        else:
            # set center of rotation to center of mass of the residue
            center = arg_dataContainer.residue[index].center_of_mass()

            # get vector for average position of bonded atoms
            vector = get_avg_pos(atom_set, frame, center)

            # use spherical coordinates function to get rotational axes
            rot_axes = get_sphCoord_axes(vector)

    if level == "united_atom":
        ## Translation
        # for united atoms use principal axes of residue for translation
        trans_axes = arg_dataContainer.residue.principal_axes()
        
        ## Rotation
        # for united atoms use heavy atoms bonded to the heavy atom 
        atom_set = arg_dataContainer.select(f"name not H* and bonded index {index}")

        # center at position of heavy atom
        center = arg_dataContainer.atom[index].position()

        # get vector for average position of hydrogens
        vector = get_avg_pos(atom_set, frame, center)

        # use spherical coordinates function to get rotational axes
        rot_axes = get_sphCoord_axes(vector)


    return trans_axes, rot_axes
# END

def get_avg_pos(atom_set, arg_frame, center):
    """
    Function to get the average position of a set of atoms. 
    
    Input
    -----
    atoms : MDAnalysis atom group
    arg_frame : frame index (integer)
    center : position for center of rotation

    Output
    ------
    avg_position : three dimensional vector
    """
    # start with an empty vector
    avg_position = nmp.zeros((3))

    # get number of atoms
    number_atoms = len(atom_set.names)

    if number_atoms != 0:
        # sum positions for all atoms in the given set
        for atom_index in atom_set.atoms.indices:
            atom_position = atom_set.atoms.positions[arg_frame, atom_index]
            avg_position = nmp.add(avg_position, atom_position)

        avg_position /= number_atoms # divide by number of atoms to get average

    else:
        # if no atoms in set the unit has no bonds to restrict its rotational motion, so we can
        # use a random vector to get the spherical coordinates axes
        avg_position = nmp.random.random(3)

    # transform the average position to a coordinate system with the origin at center
    avg_position = avg_position - center

    return avg_position
#END

def get_sphCoord_axes(arg_r):
    """ For a given vector in space, treat it is a radial vector rooted at 0,0,0 and 
    derive a curvilinear coordinate system according to the rules of polar spherical 
    coordinates"""

    x2y2 = arg_r[0]**2 + arg_r[1]**2
    r2 = x2y2 + arg_r[2]**2

    if x2y2 != 0.:
        sinTheta = nmp.sqrt(x2y2/r2)    
        cosTheta = arg_r[2]/nmp.sqrt(r2)

        sinPhi = arg_r[1]/nmp.sqrt(x2y2)
        cosPhi = arg_r[0]/nmp.sqrt(x2y2)
        
    else:
        sinTheta = 0.
        cosTheta = 1

        sinPhi = 0.
        cosPhi = 1

    # if abs(sinTheta) > 1 or abs(sinPhi) > 1:
    #     print('Bad sine : T {} , P {}'.format(sinTheta, sinPhi))

    # cosTheta = nmp.sqrt(1 - sinTheta*sinTheta)
    # cosPhi = nmp.sqrt(1 - sinPhi*sinPhi)
    
    # print('{} {} {}'.format(*arg_r))
    # print('Sin T : {}, cos T : {}'.format(sinTheta, cosTheta))
    # print('Sin P : {}, cos P : {}'.format(sinPhi, cosPhi))

    sphericalBasis = nmp.zeros((3,3))
    
    # r^
    sphericalBasis[0,:] = nmp.asarray([sinTheta*cosPhi, sinTheta*sinPhi, cosTheta])
    
    # Theta^
    sphericalBasis[1,:] = nmp.asarray([cosTheta*cosPhi, cosTheta*sinPhi, -sinTheta])
    
    # Phi^
    sphericalBasis[2,:] = nmp.asarray([-sinPhi, cosPhi, 0.])
    
    return sphericalBasis    
# END

def get_weighted_forces(arg_DataContainer, bead, trans_axes, frame):
    """
    Function to calculate the mass weighted forces for a given bead.

    Input
    -----
    bead : the part of the system to be considered
    trans_axes : the axes relative to which the forces are located
    frame : the frame number from the trajectory

    Output
    ------
    weighted_force : the mass weighted sum of the forces in the bead
    """

    forces_trans = []

    # Sum forces from all atoms in the bead
    for atom in bead.atoms:
        # update local forces in translational axes
        forces_trans += trans_axes @ arg_DataContainer.trajectory[frame].atoms[atom].velocities[3:5]

    # divide the sum of forces by the mass of the bead to get the weighted forces
    mass = bead.mass()
    weighted_force = forces_trans / nmp.sqrt(mass)

    return weighted_force
#END

def get_weighted_torques(arg_dataContainer, bead, rot_axes, frame):
    """
    Function to calculate the moment of inertia weighted torques for a given bead.

    Input
    -----
    bead : the part of the molecule to be considered
    rot_axes : the axes relative to which the forces and coordinates are located
    frame : the frame number from the trajectory

    Output
    ------
    weighted_torque : the mass weighted sum of the forces in the bead
    """

    torques = []

    for atom in bead.atoms:

        # update local coordinates in rotational axes
        coords_rot = arg_dataContainer.trajectory[frame].atom[atom].positions - bead.center_of_mass()
        coords_rot = rot_axes @ coords_rot
        # update local forces in rotational frame
        forces_rot = rot_axes @ arg_dataContainer.trajectory[frame].atoms[atom].velocities[3:5]
        
        # define torques (cross product of coordinates and forces) in rotational axes
        torques += nmp.cross(coords_rot, forces_rot)

    # divide by moment of inertia to get weighted torques
    moment_of_inertia = bead.moment_of_inertia()
    weighted_torque = torques / nmp.sqrt(moment_of_inertia)

    return weighted_torque
#END

def create_submatrix(i, j, data_i, data_j, number_frames):
    """
    Function for making covariance matrices.

    Input
    -----
    i, j : indices for the two beads (can be the same)
    data_i : values for bead i
    data_j : valuees for bead j

    Output
    ------
    submatrix : 3x3 matrix for the covariance between i and j
    """

    # Start with 3 by 3 matrix of zeros
    submatrix = nmp.zeros( (3,3) )

    # For each frame calculate the outer product (cross product) of the data from the two beads and add the result to the submatrix
    for frame in range(number_frames):
        outer_product_matrix = nmp.outer(data_i[frame], data_j[frame])
        submatrix = nmp.add(submatrix, outer_product_matrix)

    # Divide by the number of frames to get the average 
    submatrix /= number_frames

    return submatrix
