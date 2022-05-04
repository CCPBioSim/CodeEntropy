#!/usr/bin/env python

import sys
import math
import numpy as np
from numpy import linalg as LA
import logging



def distance(x0, x1, dimensions):
    '''
    calculates distance and accounts for PBCs.
    '''

    x0 = np.array(x0)
    x1 = np.array(x1)
    delta = np.abs(x1 - x0)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    return dist



def vector(x0, x1, dimensions):
    '''
    get vector of two coordinates over PBCs.
    '''

    x0 = np.array(x0)
    x1 = np.array(x1)
    dimensions = np.array(dimensions)
    delta = x1 - x0
    delta2 = []
    for delt, box in zip(delta, dimensions):
        if delt < 0 and delt < 0.5 * box:
            delt = delt + box
        if delt > 0 and delt > 0.5 * box:
            delt = delt - box

        delta2.append(delt)

    delta2 = np.array(delta2)

    return delta2


def getMcoords(x0, x1, dimensions):
    '''
    Get the middle (M) coords of two coords.
    '''

    x0 = np.array(x0)
    x1 = np.array(x1)
    dimensions = np.array(dimensions)

    delta = x1 - x0
    
    delta2 = []
    for delt, box in zip(delta, dimensions):
        if delt < 0 and delt < 0.5 * box:
            delt = delt + box
        if delt > 0 and delt > 0.5 * box:
            delt = delt - box

        delta2.append(delt)


    delta2 = np.array(delta2)

    M_coords2 = 0.5 * delta2 + x0

    return M_coords2



def projectVectorToPlane(vector, plane_normal):
    '''
    Project a vector onto a plane, need normal of plane as input.
    '''

    plane_normal_magnitude = np.linalg.norm(plane_normal)
    
    project_vector = np.cross(vector, plane_normal)
    project_vector = np.divide(project_vector, plane_normal_magnitude)
    project_vector = np.cross(plane_normal, project_vector)
    projected_vector = np.divide(project_vector, plane_normal_magnitude)


    return projected_vector





def normaliseVector(x0, x1, dimensions):
    '''
    Get vector of two coordinates over PBCs and then normalise by the 
    magnitude of the vector
    '''

    x0 = np.array(x0)
    x1 = np.array(x1)
    dimensions = np.array(dimensions)
    delta = x1 - x0

    delta2 = []
    for delt, box in zip(delta, dimensions):
        if delt < 0 and delt < 0.5 * box:
            delt = delt + box
        if delt > 0 and delt > 0.5 * box:
            delt = delt - box

        delta2.append(delt)
        
    delta = np.array(delta2)

    delta_magnitude = (delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) ** 0.5
    delta_norm = np.divide(delta, delta_magnitude)

    return delta_norm


def angleBetweenVectors(x0, x1):
    '''
    Get the angle between two vectors to determine how orthogonal they are.
    '''

    top = np.dot(x0, x1)

    axis1_magnitude = np.linalg.norm(x0) #magnitude/ distance
    axis2_magnitude = np.linalg.norm(x1) #magnitude/ distance
    bottom = np.multiply(axis1_magnitude, axis2_magnitude)
    top_bottom = np.divide(top, bottom)
    if top_bottom > 1:
        top_bottom = 1
    if top_bottom < -1:
        top_bottom = -1
    theta = np.arccos(top_bottom)
    theta = np.degrees(theta)

    return theta



def angle2(a, b, c, dimensions):
    '''
    Just used as a check. Doesn't work as periodic boundary conditions 
    are not considered properly.
    '''

    a = np.array(a) #j
    b = np.array(b) #i
    c = np.array(c) #k
    dimensions = np.array(dimensions)
    ba = a - b
    bc = c - b
    ac = c - a
    ba = np.where(ba > 0.5 * dimensions, ba - dimensions, ba)
    bc = np.where(bc > 0.5 * dimensions, bc - dimensions, bc)
    ac = np.where(ac > 0.5 * dimensions, ac - dimensions, ac)

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)

    return angle



def angle(a, b, c, dimensions):

    a = np.array(a) #j
    b = np.array(b) #i
    c = np.array(c) #k
    dimensions = np.array(dimensions)
    ba = np.abs(a - b)
    bc = np.abs(c - b)
    ac = np.abs(c - a)
    ba = np.where(ba > 0.5 * dimensions, ba - dimensions, ba)
    bc = np.where(bc > 0.5 * dimensions, bc - dimensions, bc)
    ac = np.where(ac > 0.5 * dimensions, ac - dimensions, ac)
    dist_ba = np.sqrt((ba ** 2).sum(axis=-1))
    dist_bc = np.sqrt((bc ** 2).sum(axis=-1))
    dist_ac = np.sqrt((ac ** 2).sum(axis=-1))
    


    cosine_angle = ((dist_ac ** 2 - dist_bc ** 2 - dist_ba ** 2) /
                    (-2 * dist_bc * dist_ba))

    return cosine_angle





def com_vectors(coord_list, mass_list, dimensions):
    '''
    doesnt work
    '''
    cm = np.zeros(3)
    tot_mass = 0
    for coord, mass in zip(coord_list, mass_list):
        vec = vector(coord, [0, 0, 0], dimensions)
        cm += np.array(vec) * mass
        tot_mass += mass

    cm_weighted = np.array(cm) / float(tot_mass)

    return cm_weighted



def com(coord_list, mass_list):
    '''
    '''
    x_cm = 0
    y_cm = 0
    z_cm = 0
    tot_mass = 0
    for coord, mass in zip(coord_list, mass_list):
        x_cm += mass * coord[0]
        y_cm += mass * coord[1]
        z_cm += mass * coord[2]
        tot_mass += mass
    x_cm = float(x_cm) / float(tot_mass)
    y_cm = float(y_cm) / float(tot_mass)
    z_cm = float(z_cm) / float(tot_mass)

    cm = (x_cm, y_cm, z_cm)

    return cm




def MOI(cm, coord_list, mass_list):
    '''
    '''
    x_cm, y_cm, z_cm = cm[0], cm[1], cm[2]
    I_xx = 0
    I_xy = 0
    I_yx = 0

    I_yy = 0
    I_xz = 0
    I_zx = 0

    I_zz = 0
    I_yz = 0
    I_zy = 0

    for coord, mass in zip(coord_list, mass_list):
        I_xx += (abs(coord[1] - y_cm)**2 + abs(coord[2] - z_cm)**2) * mass
        I_xy += (coord[0] - x_cm) * (coord[1] - y_cm) * mass
        I_yx += (coord[0] - x_cm) * (coord[1] - y_cm) * mass

        I_yy += (abs(coord[0] - x_cm)**2 + abs(coord[2] - z_cm)**2) * mass
        I_xz += (coord[0] - x_cm) * (coord[2] - z_cm) * mass
        I_zx += (coord[0] - x_cm) * (coord[2] - z_cm) * mass

        I_zz += (abs(coord[0] - x_cm)**2 + abs(coord[1] - y_cm)**2) * mass
        I_yz += (coord[1] - y_cm) * (coord[2] - z_cm) * mass
        I_zy += (coord[1] - y_cm) * (coord[2] - z_cm) * mass

    I = np.array([[I_xx, -I_xy, -I_xz], [-I_yx, I_yy, -I_yz], 
        [-I_zx, -I_zy, I_zz]])
    return I



def UAforceTorque(bonded_atoms_list, F_axes, Fmoi_axes, MI_axes):
    '''
    Get UA force and torque from previously calculated Faxes and MIaxes
    '''

    '''
    Faxis_list = (Faxis1, Faxis2, Faxis3)
    f_list = (FO, FH1, FH2)
    MI_xyz = (axis1MI, axis2MI, axis3MI)
    water_torques = torque(c_list, m_list, Faxis_list, f_list, MI_xyz)
    '''

    c_list = []
    f_list = []
    forces_summed = np.zeros(3)
    m_list = []

    for atoms in bonded_atoms_list:
        c_list.append(atoms.coords)
        f_list.append(atoms.forces)
        forces_summed += atoms.forces
        m_list.append(atoms.mass)

    cm = com(c_list, m_list)
    UA_torque = torque_old(cm, c_list, Fmoi_axes, f_list, MI_axes)


    F1 = np.dot(forces_summed, F_axes[0])
    F2 = np.dot(forces_summed, F_axes[1])
    F3 = np.dot(forces_summed, F_axes[2])
    mass_sqrt = sum(m_list) ** 0.5
    UA_force = (float(F1)/float(mass_sqrt), 
            float(F2)/float(mass_sqrt), 
            float(F3)/float(mass_sqrt))


    return UA_force, UA_torque




def torque_old(cm, coord_list, axis_list, force_list, MI_coords):
    '''
    '''
    x_cm, y_cm, z_cm = cm[0], cm[1], cm[2]
    MI_coords = np.sqrt(MI_coords) #sqrt moi to weight torques

    O_torque = np.array([0, 0, 0])
    H1_torque = np.array([0, 0, 0])
    H2_torque = np.array([0, 0, 0])
    count_atom = 0
    for coord, force in zip(coord_list, force_list):
        count_atom +=1
        new_coords = []
        new_forces = []
        for axis in axis_list:
            new_c = axis[0]*(coord[0] - x_cm) + axis[1]*(coord[1] - y_cm) \
                    + axis[2]*(coord[2] - z_cm)
            new_f = axis[0]*force[0] + axis[1]*force[1] + axis[2]*force[2]
            new_coords.append(new_c)
            new_forces.append(new_f)

        torque_list = []
        torquex = float(new_coords[1]*new_forces[2] \
                - new_coords[2]*new_forces[1]) #/ float(1e10)
        torquey = float(new_coords[2]*new_forces[0] \
                - new_coords[0]*new_forces[2]) #/ float(1e10)
        torquez = float(new_coords[0]*new_forces[1] \
                - new_coords[1]*new_forces[0]) #/ float(1e10)

        if count_atom == 1:
            O_torque = (torquex, torquey, torquez)
            O_torque = np.divide(O_torque, MI_coords)
        elif count_atom == 2:
            H1_torque = (torquex, torquey, torquez)
            H1_torque = np.divide(H1_torque, MI_coords)
        elif count_atom == 3:
            H2_torque = (torquex, torquey, torquez)
            H2_torque = np.divide(H2_torque, MI_coords)
        else:
            ("torques outs of range")
            break

    W_torque = O_torque + H1_torque + H2_torque

    return (W_torque)



def principalAxesMOI(bonded_atoms_list):
    '''
    Calculate the principal axes of the MOI matrix, then calc forces
    '''

    coord_list = []
    mass_list = []
    forces_summed = np.zeros(3)

    for atom in bonded_atoms_list:
        coord_list.append(atom.coords)
        mass_list.append(atom.mass)
        forces_summed += atom.forces

    cm = com(coord_list, mass_list)
    moI = MOI(cm, coord_list, mass_list)


    eigenvalues, eigenvectors = LA.eig(moI) 
    #different values generated to Jon's code

    transposed = np.transpose(eigenvectors) #turn columns to rows

    min_eigenvalue = abs(eigenvalues[0])
    if eigenvalues[1] < min_eigenvalue:
        min_eigenvalue = eigenvalues[1]
    if eigenvalues[2] < min_eigenvalue:
        min_eigenvalue = eigenvalues[2]

    max_eigenvalue = abs(eigenvalues[0])
    if eigenvalues[1] > max_eigenvalue:
        max_eigenvalue = eigenvalues[1]
    if eigenvalues[2] > max_eigenvalue:
        max_eigenvalue = eigenvalues[2]

    #print (min_eigenvalue * const, max_eigenvalue * const) 
    #same as Jons

    FMIaxis1 = None
    FMIaxis2 = None
    FMIaxis3 = None

    axis1MI = None
    axis2MI = None
    axis3MI = None

    for i in range(0,3):
        if eigenvalues[i] == max_eigenvalue:
            FMIaxis1 = transposed[i]
            axis1MI = eigenvalues[i]
        elif eigenvalues[i] == min_eigenvalue:
            FMIaxis3 = transposed[i]
            axis3MI = eigenvalues[i]
        else:
            FMIaxis2 = transposed[i]
            axis2MI = eigenvalues[i]


    Fm_axes = [FMIaxis1, FMIaxis2, FMIaxis3]
    MI_axes = [axis1MI, axis2MI, axis3MI]

    return Fm_axes, MI_axes




def calcAngleWithNearestNonlike(neighbour, nearest, dimensions):
    '''
    Calc angle in plane and orthog to plane between any molecule
    and any other molecule.
    '''

    PAxes = np.array(neighbour.WMprincipalAxis[0])
    PAxis = np.array(neighbour.WMprincipalAxis[1])
    COM = np.array(neighbour.WMprincipalAxis[2])

    #plane_normal = vector(COM, PAxes[0], dimensions)
    #plane_normal2 = vector(COM, PAxes[1], dimensions)
    #OM_vector = vector(COM, PAxes[2], dimensions)

    plane_normal = PAxes[0]
    plane_normal2 = PAxes[1]
    OM_vector = PAxes[2]

    OSolute_vector = vector(nearest.coords, COM, dimensions)


    projected_OSolute = projectVectorToPlane(OSolute_vector, plane_normal)
    projected_orthog_OSolute = projectVectorToPlane(OSolute_vector, 
            plane_normal2)

    #print ('OM', OM_vector, O.atom_num)

    soluteOMangle = angleBetweenVectors(OM_vector, projected_OSolute)
    soluteOorthogAngle = angleBetweenVectors(OM_vector, 
            projected_orthog_OSolute)


    if np.dot(OM_vector, projected_OSolute) > 0 and \
            np.dot(plane_normal2, projected_OSolute) > 0:
        soluteOMangle = (90 - soluteOMangle) + 270

    if np.dot(-OM_vector, projected_OSolute) > 0 and \
            np.dot(plane_normal2, projected_OSolute) > 0:
        soluteOMangle = (180 - soluteOMangle) + 180


    if np.dot(-plane_normal, projected_orthog_OSolute) > 0 and \
            np.dot(-OM_vector, projected_orthog_OSolute) > 0:
        soluteOorthogAngle = (180 - soluteOorthogAngle) + 180

    if np.dot(-plane_normal, projected_orthog_OSolute) > 0 and \
            np.dot(OM_vector, projected_orthog_OSolute) > 0:
        soluteOorthogAngle = (90 - soluteOorthogAngle) + 270




    #cosAngle = angle(closest_H.coords, O.coords, solute.coords, dimensions)
    #arccosAngle = np.arccos(cosAngle)
    #angleDegrees = np.degrees(arccosAngle)
    #if math.isnan(angleDegrees) is False:
    #angleDegrees = int(round(angleDegrees, 0))

    if math.isnan(soluteOMangle) is False:
        soluteOMangle = int(round(soluteOMangle, 0))
    #else:
        #print ('Error1: NaN for solute resid: %s, '\
                #'water resid %s and angle %s.'\
                #% (nearest.resid, neighbour.resid, soluteOMangle))
        #continue

    if math.isnan(soluteOorthogAngle) is False:
        soluteOorthogAngle = int(round(soluteOorthogAngle, 0))
    #else:
        #print ('Error1: NaN for solute resid: %s, '\
                #'water resid %s and angle %s.'\
                #% (nearest.resid, neighbour.resid, soluteOorthogAngle))
        #continue


    return soluteOMangle, soluteOorthogAngle






def calcWaterSoluteAngle(solute, O, H1, H2, dimensions):
    '''
    Calc angle in plane and orthog to plane between any water
    and any solute atom.
    '''


    dist_solute_H1, dist_solute_H2 = None, None

    for atomDist in solute.nearest_all_atom_array:
        if atomDist[0] == H1.atom_num-1:
            dist_solute_H1 = atomDist[1]
        elif atomDist[0] == H2.atom_num-1:
            dist_solute_H2 = atomDist[1]
        else:
            continue

    if dist_solute_H1 == None:
        dist_solute_H1 = distance(solute.coords, H1.coords, dimensions)
    if dist_solute_H2 == None:
        dist_solute_H2 = distance(solute.coords, H2.coords, dimensions)
    closest_H = None
    if dist_solute_H1 < dist_solute_H2:
        closest_H = H1
    elif dist_solute_H1 > dist_solute_H2:
        closest_H = H2
    else:
        closest_H = H1


    H1O_vector = vector(O.coords, H1.coords, dimensions)
    H2O_vector = vector(O.coords, H2.coords, dimensions)
    plane_normal = np.cross(H1O_vector, H2O_vector)
    M_coords = getMcoords(H1.coords, H2.coords, dimensions)
    OM_vector = vector(M_coords, O.coords, dimensions)
    plane_normal2 = np.cross(OM_vector, plane_normal)
    OSolute_vector = vector(solute.coords, O.coords, dimensions)
    projected_OSolute = projectVectorToPlane(OSolute_vector, plane_normal)
    projected_orthog_OSolute = projectVectorToPlane(OSolute_vector, 
            plane_normal2)

    #print ('OM', OM_vector, O.atom_num)

    soluteOMangle = angleBetweenVectors(OM_vector, projected_OSolute)
    soluteOorthogAngle = angleBetweenVectors(OM_vector, 
            projected_orthog_OSolute)


    if np.dot(OM_vector, projected_OSolute) > 0 and \
            np.dot(plane_normal2, projected_OSolute) > 0:
        soluteOMangle = (90 - soluteOMangle) + 270
        ### only go up to 180 degrees rather than 360 as above
        #if soluteOMangle > 180:
            #soluteOMangle = 

    if np.dot(-OM_vector, projected_OSolute) > 0 and \
            np.dot(plane_normal2, projected_OSolute) > 0:
        soluteOMangle = (180 - soluteOMangle) + 180


    if np.dot(-plane_normal, projected_orthog_OSolute) > 0 and \
            np.dot(-OM_vector, projected_orthog_OSolute) > 0:
        soluteOorthogAngle = (180 - soluteOorthogAngle) + 180

    if np.dot(-plane_normal, projected_orthog_OSolute) > 0 and \
            np.dot(OM_vector, projected_orthog_OSolute) > 0:
        soluteOorthogAngle = (90 - soluteOorthogAngle) + 270




    cosAngle = angle(closest_H.coords, O.coords, solute.coords, dimensions)
    arccosAngle = np.arccos(cosAngle)
    angleDegrees = np.degrees(arccosAngle)
    if math.isnan(angleDegrees) is False:
        angleDegrees = int(round(angleDegrees, 0))
        soluteOMangle = int(round(soluteOMangle, 0))
        soluteOorthogAngle = int(round(soluteOorthogAngle, 0))
    else:
        logging.error('NaN for solute resid: %s, water resid %s and angle %s.'\
                % (solute.resid, O.resid, angleDegrees))


    return soluteOMangle, soluteOorthogAngle


