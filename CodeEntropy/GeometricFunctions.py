import numpy as nmp

def get_beads(data_container, level):
    """
    Function to define beads depending on the level in the hierarchy.

    Input
    -----
    data_container : the MDAnalysis universe
    level : the heirarchy level (polymer, residue, or united atom)

    Output
    ------
    list_of_beads : the relevent beads
    """

    if level == "polymer":
        list_of_beads = []
        atom_group = "all"
        list_of_beads.append(data_container.select_atoms(atom_group))

    if level == "residue":
        list_of_beads = []
        num_residues = len(data_container.residues)
        for residue in range(num_residues):
            atom_group = "resindex " + str(residue)
            list_of_beads.append(data_container.select_atoms(atom_group))

    if level == "united_atom":
        list_of_beads = []
        heavy_atoms = data_container.select_atoms("not name H*")
        for atom in heavy_atoms:
            atom_group = "index " + str(atom.index) + " or (name H* and bonded index " + str(atom.index) +")"
            list_of_beads.append(data_container.select_atoms(atom_group))

    return list_of_beads
#END

def get_axes(data_container, level, index=0):
    """
    Function to set the translational and rotational axes.
    The translational axes are based on the principal axes of the unit one level larger than
    the level we are interested in (except for the polymer level where there is no larger unit).
    The rotational axes use the covalent links between residues or atoms where possible to 
    define the axes, or if the unit is not bonded to others of the same level the prinicpal 
    axes of the unit are used.

    Input
    -----
    data_container : the information about the molecule and trajectory
    level : the level (united atom, residue, or polymer) of interest
    index : residue index (integer)

    Output
    ------
    trans_axes : translational axes
    rot_axes : rotational axes
    """
    index = int(index)

    if level == "polymer":
        # for polymer use principle axis for both translation and rotation
        trans_axes = data_container.atoms.principal_axes()
        rot_axes = data_container.atoms.principal_axes()

    if level == "residue":
        ## Translation
        # for residues use principal axes of whole molecule for translation
        trans_axes = data_container.atoms.principal_axes()

        ## Rotation
        # find bonds between atoms in residue of interest and other residues
        # we are assuming bonds only exist between adjacent residues (linear chains of residues)
        # TODO refine selection so that it will work for branched polymers
        index_prev = index - 1
        index_next = index + 1
        atom_set = data_container.select_atoms(f"(resindex {index_prev} or resindex {index_next}) and bonded resid {index}")
        residue = data_container.select_atoms(f"resindex {index}")

        if len(atom_set) == 0:
            # if no bonds to other residues use pricipal axes of residue
            rot_axes = residue.atoms.principal_axes()

        else:
            # set center of rotation to center of mass of the residue
            center = residue.atoms.center_of_mass()

            # get vector for average position of bonded atoms
            vector = get_avg_pos(atom_set, center)

            # use spherical coordinates function to get rotational axes
            rot_axes = get_sphCoord_axes(vector)

    if level == "united_atom":
        ## Translation
        # for united atoms use principal axes of residue for translation
        trans_axes = data_container.residues.principal_axes()

        ## Rotation
        # for united atoms use heavy atoms bonded to the heavy atom
        atom_set = data_container.select_atoms(f"not name H* and bonded index {index}")

        # center at position of heavy atom
        atom_group = data_container.select_atoms(f"index {index}")
        center = atom_group.positions[0]

        # get vector for average position of hydrogens
        vector = get_avg_pos(atom_set, center)

        # use spherical coordinates function to get rotational axes
        rot_axes = get_sphCoord_axes(vector)

    return trans_axes, rot_axes
# END

def get_avg_pos(atom_set, center):
    """
    Function to get the average position of a set of atoms. 
    
    Input
    -----
    atoms : MDAnalysis atom group
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
        for atom_index in range(number_atoms):
            atom_position = atom_set.atoms[atom_index].position

            avg_position += atom_position

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
    """ 
    For a given vector in space, treat it is a radial vector rooted at 0,0,0 and 
    derive a curvilinear coordinate system according to the rules of polar spherical 
    coordinates
    """

    x2y2 = arg_r[0]**2 + arg_r[1]**2
    r2 = x2y2 + arg_r[2]**2

    if x2y2 != 0.:
        sin_theta = nmp.sqrt(x2y2/r2)
        cos_theta = arg_r[2]/nmp.sqrt(r2)

        sin_phi = arg_r[1]/nmp.sqrt(x2y2)
        cos_phi = arg_r[0]/nmp.sqrt(x2y2)

    else:
        sin_theta = 0.
        cos_theta = 1

        sin_phi = 0.
        cos_phi = 1

    # if abs(sin_theta) > 1 or abs(sin_phi) > 1:
    #     print('Bad sine : T {} , P {}'.format(sin_theta, sin_phi))

    # cos_theta = nmp.sqrt(1 - sin_theta*sin_theta)
    # cos_phi = nmp.sqrt(1 - sin_phi*sin_phi)

    # print('{} {} {}'.format(*arg_r))
    # print('Sin T : {}, cos T : {}'.format(sin_theta, cos_theta))
    # print('Sin P : {}, cos P : {}'.format(sin_phi, cos_phi))

    spherical_basis = nmp.zeros((3,3))

    # r^
    spherical_basis[0,:] = nmp.asarray([sin_theta*cos_phi, sin_theta*sin_phi, cos_theta])

    # Theta^
    spherical_basis[1,:] = nmp.asarray([cos_theta*cos_phi, cos_theta*sin_phi, -sin_theta])

    # Phi^
    spherical_basis[2,:] = nmp.asarray([-sin_phi, cos_phi, 0.])

    return spherical_basis
# END

def get_weighted_forces(data_container, bead, trans_axes, highest_level, force_partitioning=0.5):
    """
    Function to calculate the mass weighted forces for a given bead.

    Input
    -----
    bead : the part of the system to be considered
    trans_axes : the axes relative to which the forces are located

    Output
    ------
    weighted_force : the mass weighted sum of the forces in the bead
    """

    forces_trans = nmp.zeros((3,))

    # Sum forces from all atoms in the bead
    for atom in bead.atoms:
        # update local forces in translational axes
        forces_local = nmp.matmul(trans_axes, data_container.atoms[atom.index].force)
        forces_trans += forces_local

    if highest_level:
        # multiply by the force_partitioning parameter to avoid double counting
        # of the forces on weakly correlated atoms
        # the default value of force_partitioning is 0.5 (dividing by two)
        forces_trans = force_partitioning * forces_trans

    # divide the sum of forces by the mass of the bead to get the weighted forces
    mass = bead.total_mass()

    weighted_force = forces_trans / nmp.sqrt(mass)

    return weighted_force
#END

def get_weighted_torques(data_container, bead, rot_axes, force_partitioning=0.5):
    """
    Function to calculate the moment of inertia weighted torques for a given bead.

    Input
    -----
    bead : the part of the molecule to be considered
    rot_axes : the axes relative to which the forces and coordinates are located
    frame : the frame number from the trajectory

    Output
    ------
    weighted_torque : the mass weighted sum of the torques in the bead
    """

    torques = nmp.zeros((3,))
    weighted_torque = nmp.zeros((3,))

    for atom in bead.atoms:

        # update local coordinates in rotational axes
        coords_rot = data_container.atoms[atom.index].position - bead.center_of_mass()
        coords_rot = nmp.matmul(rot_axes, coords_rot)
        # update local forces in rotational frame
        forces_rot = nmp.matmul(rot_axes, data_container.atoms[atom.index].force)

        # multiply by the force_partitioning parameter to avoid double counting
        # of the forces on weakly correlated atoms
        # the default value of force_partitioning is 0.5 (dividing by two)
        forces_rot = force_partitioning * forces_rot

        # define torques (cross product of coordinates and forces) in rotational axes
        torques_local = nmp.cross(coords_rot, forces_rot)
        torques += torques_local

    # divide by moment of inertia to get weighted torques
    # moment of inertia is a 3x3 tensor
    # the weighting is done in each dimension (x,y,z) using the diagonal elements of
    # the moment of inertia tensor
    moment_of_inertia = bead.moment_of_inertia()

    for dimension in range(3):
        # cannot divide by zero
        if nmp.isclose(moment_of_inertia[dimension,dimension],0):
            weighted_torque[dimension] = torques[dimension]
        else:
            weighted_torque[dimension] = torques[dimension] / nmp.sqrt(moment_of_inertia[dimension,dimension])

    return weighted_torque
#END

def create_submatrix(data_i, data_j, number_frames):
    """
    Function for making covariance matrices.

    Input
    -----
    data_i : values for bead i
    data_j : valuees for bead j

    Output
    ------
    submatrix : 3x3 matrix for the covariance between i and j
    """

    # Start with 3 by 3 matrix of zeros
    submatrix = nmp.zeros( (3,3) )

    # For each frame calculate the outer product (cross product) of the data from the two beads
    # and add the result to the submatrix
    for frame in range(number_frames):
        outer_product_matrix = nmp.outer(data_i[frame], data_j[frame])
        submatrix = nmp.add(submatrix, outer_product_matrix)

    # Divide by the number of frames to get the average
    submatrix /= number_frames

    return submatrix
#END

def filter_zero_rows_columns(arg_matrix, verbose):
    """
    function for removing rows and columns that contain only zeros from a matrix

    Input
    -----
    arg_matrix : matrix

    Output
    ------
    arg_matrix : the reduced size matrix
    """

    #record the initial size
    init_shape = nmp.shape(arg_matrix)

    zero_indices = list(filter(lambda row : nmp.all(nmp.isclose(arg_matrix[row,:] , 0.0)) , nmp.arange(nmp.shape(arg_matrix)[0])))
    all_indices = nmp.ones((nmp.shape(arg_matrix)[0]), dtype=bool)
    all_indices[zero_indices] = False
    arg_matrix = arg_matrix[all_indices,:]

    all_indices = nmp.ones((nmp.shape(arg_matrix)[1]), dtype=bool)
    zero_indices = list(filter(lambda col : nmp.all(nmp.isclose(arg_matrix[:,col] , 0.0)) , nmp.arange(nmp.shape(arg_matrix)[1])))
    all_indices[zero_indices] = False
    arg_matrix = arg_matrix[:,all_indices]

    # get the final shape
    final_shape = nmp.shape(arg_matrix)

    if verbose and init_shape != final_shape:
        print('A shape change has occured ({},{}) -> ({}, {})'.format(*init_shape, *final_shape))

    return arg_matrix
#END
