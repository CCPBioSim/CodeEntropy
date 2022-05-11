#!/usr/bin/env python

import sys
import logging
import math
import numpy as np
from numpy import linalg as LA

from CodeEntropy.poseidon.extractData.generalFunctions import com, MOI, vector, distance
from collections import defaultdict
nested_dict = lambda: defaultdict(nested_dict) 
        #create nested dict in one go





def calculateFTMatrix(all_data, dimensions):
    '''
    Consider force/torque calculations as discussed in Ali 2019 paper.
    UA level and molecule level.
    '''

    resid_list = []
    for x in range(0, len(all_data)):
        atom = all_data[x]
        if atom.mass > 1.1:
            if atom.resid not in resid_list: #iterate though resids once
                resid_list.append(atom.resid)
                all_molecule_atoms = [] #list of heavy atoms and Hs
                UAs_list = [] #list of lists [HA + Hs]

                for num in atom.molecule_atomNums: #atom nums in molecule
                    UA = all_data[num]

                    all_molecule_atoms.append(UA)
                    heavy_bonded = []
                    H_bonded = []

                    for bonded_atom in UA.bonded_to_atom_num:
                        bonded = all_data[bonded_atom]
                        if bonded.mass > 1.1 and bonded.atom_num != \
                                UA.atom_num:
                            heavy_bonded.append(bonded)

                        if bonded.mass < 1.1 and bonded.atom_num != \
                                UA.atom_num:
                            H_bonded.append(bonded)
                            all_molecule_atoms.append(bonded)
                            
                    UA_atoms_list = [UA] + H_bonded
                    UAs_list.append(UA_atoms_list)
                    bonded_atoms_list = [UA] + heavy_bonded + H_bonded
                    UA.bondedUA_H = [len(heavy_bonded), len(H_bonded)]


                molecule_coords = []
                molecule_masses = []
                for a in all_molecule_atoms:
                    molecule_coords.append(a.coords)
                    molecule_masses.append(a.mass)
                molecule_COM = com(molecule_coords, molecule_masses)


                #####****** MOLECULE LEVEL ******
                #if molecule is only one UA with bonded Hs, 
                #this works for one UA molecules
                if len(atom.molecule_atomNums) == 1 \
                        and len(atom.bonded_to_atom_num) > 0:
                    WM_principal_axes, WM_MI_axis = principalAxesMOI(
                            all_data, all_molecule_atoms, 
                            molecule_COM, Hs=True)
                            #use atom list containing Hs for P axes inc. Hs
                    WM_force, WM_torque = rotateFT(all_data, 
                            all_molecule_atoms, 
                            WM_principal_axes, WM_principal_axes, WM_MI_axis, 
                            molecule_COM, molecule_COM, Hs=True)

                    #'''
                    WM_force = np.outer(WM_force, WM_force)
                    WM_torque = np.outer(WM_torque, WM_torque)
                    atom.MweightedForces = np.round(np.divide(WM_force, 4), 3)
                    atom.MweightedTorques = np.round(np.divide(WM_torque, 4), 3)
                    atom.WMprincipalAxis = WM_principal_axes, \
                            WM_MI_axis, molecule_COM

                    ###check this with Richard, do you not halve here? 
                    ###For Jon's code, you do halve here
                    ###even though molecule == UA level
                    atom.UAweightedForces = np.round(np.divide(WM_force, 4), 3)
                    atom.UAweightedTorques = np.round(np.divide(WM_torque, 4), 3)
                    atom.molecule_UA_Fs = np.round(np.divide(WM_force, 4), 3)
                    atom.molecule_UA_Ts = np.round(np.divide(WM_torque, 4), 3)
                    #'''

                    '''
                    atom.MweightedForces = WM_force
                    atom.MweightedTorques = WM_torque
                    atom.UAweightedForces = WM_force
                    atom.UAweightedTorques = WM_torque
                    '''



                #if molecule contains more than one UA
                #forces work, torques work if Hs = True
                if len(atom.molecule_atomNums) > 1:

                    WM_principal_axes, WM_MI_axis = principalAxesMOI(
                            all_data, all_molecule_atoms, 
                            molecule_COM, Hs=False)
                            #only use heavy atom list for P axes calc
                            #we don't consider Hs as they result 
                            #in larger torques as Hs are further away

                    WM_force, WM_torque = rotateFT(all_data, 
                            all_molecule_atoms, 
                            WM_principal_axes, WM_principal_axes, WM_MI_axis, 
                            molecule_COM, molecule_COM, Hs=True)

                    #'''
                    #WM_FT = np.concatenate((WM_force, WM_torque), axis=None)
                    #WM_FT = np.outer(WM_FT, WM_FT)
                    WM_force = np.outer(WM_force, WM_force)
                    WM_torque = np.outer(WM_torque, WM_torque)
                    atom.MweightedForces = np.round(np.divide(WM_force, 4), 3)
                    atom.MweightedTorques = np.round(np.divide(WM_torque, 4), 3)
                    atom.WMprincipalAxis = WM_principal_axes, \
                            WM_MI_axis, molecule_COM

                    atom.molecule_UA_Fs = np.round(np.divide(WM_force, 4), 3)
                    atom.molecule_UA_Ts = np.round(np.divide(WM_torque, 4), 3)
                    #'''

                    #atom.MweightedForces = WM_force
                    #atom.MweightedTorques = WM_torque


                    ##### ****** UNITED-ATOM LEVEL ******
                    ## works!
                    UA_F_list = []
                    UA_T_list = []
                    for UA in UAs_list:
                        XX_principal_axes, UA_MI_axis \
                                = None, None

                        if UA[0].mass > 1.1:
                            bonded_HAs = [UA[0]] #inc itself
                            bonded_Hs = []
                            for b in UA[0].bonded_to_atom_num:
                                bonded = all_data[b]
                                if bonded.mass < 1.1:
                                    bonded_Hs.append(bonded)
                                elif bonded.mass > 1.1:
                                    bonded_HAs.append(bonded)
                                else:
                                    continue


                            if len(bonded_HAs) == 2 and len(bonded_Hs) > 0:
                                #num HAs including itself
                                UA_COM = bonded_HAs[0].coords

                                R = bonded_HAs[0]
                                X1 = bonded_HAs[1]
                                H = bonded_Hs[0]

                                RX1_vector = vector(R.coords, 
                                        X1.coords, dimensions)
                                RX1_dist = distance(R.coords, 
                                        X1.coords, dimensions)
                                Paxis1 = np.divide(RX1_vector, RX1_dist)

                                RH_vector = vector(R.coords, 
                                        H.coords, dimensions)
                                RH_dist = distance(R.coords, 
                                        H.coords, dimensions)
                                RH_vector_norm = np.divide(RH_vector, RH_dist)

                                Paxis2 = np.cross(Paxis1, RH_vector_norm)
                                Paxis3 = np.cross(Paxis1, Paxis2)


                                XX_principal_axes = [Paxis1, Paxis2, Paxis3]

                                UA_MI_axis, XX_principal_axes = \
                                        UA_MOI(all_data, 
                                        UA, UA_COM, XX_principal_axes, 
                                        dimensions, Hs=True) 

                                if len(bonded_Hs) == 1:
                                    MIx, MIy, MIz = UA_MI_axis[0], \
                                            UA_MI_axis[1], UA_MI_axis[2]
                                    if MIx < MIy and MIx < MIz:
                                        MIx = 0
                                    if MIy < MIx and MIy < MIz:
                                        MIy = 0
                                    if MIz < MIx and MIz < MIy:
                                        MIz = 0

                                    UA_MI_axis = [MIx, MIy, MIz]


                            elif len(bonded_HAs) > 2:
                                #num HAs including itself
                                UA_COM = bonded_HAs[0].coords

                                R = bonded_HAs[0]
                                X1 = bonded_HAs[1]
                                X2 = bonded_HAs[2]

                                Paxis1 = np.zeros(3) 
                                    #average of all HA covalent bond vectors
                                for HA in bonded_HAs[1:]:
                                    Paxis1 += vector(R.coords, 
                                            HA.coords, dimensions)
                                    
                                X1X2_vector = vector(X1.coords, 
                                        X2.coords, dimensions)
                                X1X2_dist = distance(X1.coords, 
                                        X2.coords, dimensions)
                                X1X2_norm = np.divide(X1X2_vector, X1X2_dist)
                                Paxis2 = np.cross(X1X2_norm, Paxis1)

                                Paxis3 = np.cross(Paxis1, Paxis2)


                                XX_principal_axes = [Paxis1, Paxis2, Paxis3]

                                UA_MI_axis, XX_principal_axes = \
                                        UA_MOI(all_data, 
                                        UA, UA_COM, XX_principal_axes, 
                                        dimensions, Hs=True)

                            elif len(bonded_HAs) == 2 and len(bonded_Hs) == 0:
                                #num use arbitrary coords for 2nd vector
                                UA_COM = bonded_HAs[0].coords

                                R = bonded_HAs[0]
                                X1 = bonded_HAs[1]

                                RX1_vector = vector(R.coords, 
                                        X1.coords, dimensions)
                                RX1_dist = distance(R.coords, 
                                        X1.coords, dimensions)
                                Paxis1 = np.divide(RX1_vector, RX1_dist)

                                RZ_vector = vector(R.coords, 
                                        [0, 0, 0], dimensions)
                                RZ_dist = distance(R.coords, 
                                        [0, 0, 0], dimensions)
                                RZ_vector_norm = np.divide(RZ_vector, RZ_dist)

                                Paxis2 = np.cross(Paxis1, RZ_vector_norm)
                                Paxis3 = np.cross(Paxis1, Paxis2)


                                XX_principal_axes = [Paxis1, Paxis2, Paxis3]

                                UA_MI_axis, XX_principal_axes = \
                                        UA_MOI(all_data, 
                                        UA, UA_COM, XX_principal_axes, 
                                        dimensions, Hs=True) 

                                if len(bonded_Hs) == 0:
                                    MIx, MIy, MIz = UA_MI_axis[0], \
                                            UA_MI_axis[1], UA_MI_axis[2]
                                    if MIx < MIy and MIx < MIz:
                                        MIx = 0
                                    if MIy < MIx and MIy < MIz:
                                        MIy = 0
                                    if MIz < MIx and MIz < MIy:
                                        MIz = 0

                                    UA_MI_axis = [MIx, MIy, MIz]

                            else:
                                continue



                                
                        if UA[0].mass < 1.1:
                            print ('Error: no HA in UA')

                        UA_force, UA_torque = rotateFT(all_data, UA, 
                                WM_principal_axes, 
                                XX_principal_axes, UA_MI_axis, 
                                molecule_COM, UA_COM, Hs=True)
                                #torques use HA com, 
                                #forces use molecule COM

                        UA_F_list += [UA_force]
                        UA_T_list += [UA_torque]


                        #'''
                        UA_force = np.outer(UA_force, UA_force)
                        UA_torque = np.outer(UA_torque, UA_torque)
                        UA[0].UAweightedForces = np.round(
                                np.divide(UA_force, 4), 3)
                        ##UA[0].UAweightedForces = UA_force
                                #2019 paper, dont halve forces
                        UA[0].UAweightedTorques = np.round(
                                np.divide(UA_torque, 4), 3)
                        #'''

                        #UA[0].UAweightedForces = UA_force
                        #UA[0].UAweightedTorques = UA_torque


                    #dont halve forces in molecule UAs (2019 paper),
                    #divide by 4 to compare with jons output
                    molecule_UA_Fs = np.concatenate(UA_F_list, axis=None)
                    molecule_UA_Fs = np.outer(molecule_UA_Fs, molecule_UA_Fs)
                    #atom.molecule_UA_Fs = np.divide(molecule_UA_Fs, 4)
                    atom.molecule_UA_Fs = np.round(molecule_UA_Fs)


                    #halve torques in molecule UAs
                    molecule_UA_Ts = np.concatenate(UA_T_list, axis=None)
                    molecule_UA_Ts = np.outer(molecule_UA_Ts, molecule_UA_Ts)
                    atom.molecule_UA_Ts = np.round(np.divide(molecule_UA_Ts, 4), 3)






                #if molecule is monoatomic (one UA and no bonded Hs)
                if len(atom.molecule_atomNums) == 1 \
                        and len(atom.bonded_to_atom_num) == 0:
                    mass_sqrt = float(atom.mass ** 0.5)
                    forces_sqrt = [float(atom.forces[0]) / mass_sqrt,
                            float(atom.forces[1]) / mass_sqrt,
                            float(atom.forces[2]) / mass_sqrt]
                    forces_sqrt = np.sort(forces_sqrt)

                    #'''
                    UA_force = np.outer(forces_sqrt, forces_sqrt)
                    UA_torque = np.outer([0, 0, 0], [0, 0, 0])
                    atom.UAweightedForces = np.round(np.divide(UA_force, 4), 3)
                    atom.UAweightedTorques = np.round(UA_torque, 3) #zero
                    atom.MweightedForces = np.round(np.divide(UA_force, 4), 3)
                    atom.MweightedTorques = np.round(UA_torque, 3) #zero
                    atom.WMprincipalAxis = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], \
                            [0, 0, 0], [0, 0, 0]

                    atom.molecule_UA_Fs = np.round(np.divide(UA_force, 4), 3)
                    atom.molecule_UA_Ts = np.round(np.divide(UA_torque, 4), 3)
                    #'''

                    '''
                    atom.UAweightedForces = forces_sqrt
                    atom.UAweightedTorques = [0, 0, 0] #zero
                    atom.MweightedForces = forces_sqrt
                    atom.MweightedTorques = [0, 0, 0] #zero
                    '''





def UA_MOI(all_data, bonded_atoms_list, COM, PIaxes, dimensions, Hs):
    '''
    Get MOI axis for UA level, where eigenvalues and vectors are
    not used
    '''


    coord_list = []
    mass_list = []

    for atom in bonded_atoms_list:
        if Hs == True:
            #print (atom.atom_name)
            coord_list.append(atom.coords)
            mass_list.append(atom.mass)
        elif Hs == False:
            if atom.mass > 1.1:
                UA_mass = 0 #add mass of HA and bonded Hs
                #print (atom.atom_name)
                UA_mass += atom.mass
                coord_list.append(atom.coords) #only include coords of UA
                for h in atom.bonded_to_atom_num:
                    H = all_data[h]
                    if H.mass < 1.1:
                        UA_mass += H.mass
                    else:
                        continue
                mass_list.append(UA_mass) 
                        #include mass of Hs on heavy atom mass
        else:
            continue


        ###### sorting out PIaxes for MoI for UA fragment

        modPIx = PIaxes[0][0] ** 2 + PIaxes[0][1] ** 2 + PIaxes[0][2] ** 2
        modPIy = PIaxes[1][0] ** 2 + PIaxes[1][1] ** 2 + PIaxes[1][2] ** 2
        modPIz = PIaxes[2][0] ** 2 + PIaxes[2][1] ** 2 + PIaxes[2][2] ** 2


        PIaxes[0][0] = PIaxes[0][0] / (modPIx ** 0.5)
        PIaxes[0][1] = PIaxes[0][1] / (modPIx ** 0.5)
        PIaxes[0][2] = PIaxes[0][2] / (modPIx ** 0.5)

        PIaxes[1][0] = PIaxes[1][0] / (modPIy ** 0.5)
        PIaxes[1][1] = PIaxes[1][1] / (modPIy ** 0.5)
        PIaxes[1][2] = PIaxes[1][2] / (modPIy ** 0.5)

        PIaxes[2][0] = PIaxes[2][0] / (modPIz ** 0.5)
        PIaxes[2][1] = PIaxes[2][1] / (modPIz ** 0.5)
        PIaxes[2][2] = PIaxes[2][2] / (modPIz ** 0.5)



        #get dot product of Paxis1 and CoM->atom1 vect
        #will just be [0,0,0]
        RRaxis = vector(coord_list[0], COM, dimensions)
        #flip each Paxis if its pointing out of UA
        dotProd1 = np.dot(np.array(PIaxes[0]), RRaxis)
        if dotProd1 < 0:
            PIaxes[0][0] = -PIaxes[0][0]
            PIaxes[0][1] = -PIaxes[0][1]
            PIaxes[0][2] = -PIaxes[0][2]

        dotProd2 = np.dot(np.array(PIaxes[1]), RRaxis)
        if dotProd2 < 0:
            PIaxes[1][0] = -PIaxes[1][0]
            PIaxes[1][1] = -PIaxes[1][1]
            PIaxes[1][2] = -PIaxes[1][2]

        dotProd3 = np.dot(np.array(PIaxes[2]), RRaxis)
        if dotProd3 < 0:
            PIaxes[2][0] = -PIaxes[2][0]
            PIaxes[2][1] = -PIaxes[2][1]
            PIaxes[2][2] = -PIaxes[2][2]


    #MI = moment of inertia
    axis1MI = 0 #x
    axis2MI = 0 #y
    axis3MI = 0 #z
    #'''

    for coord, mass in zip(coord_list, mass_list):
        dx = coord[0] - COM[0]
        dy = coord[1] - COM[1]
        dz = coord[2] - COM[2]

        PIaxes_xx = PIaxes[0][1] * dz - PIaxes[0][2] * dy
        PIaxes_xy = PIaxes[0][2] * dx - PIaxes[0][0] * dz
        PIaxes_xz = PIaxes[0][0] * dy - PIaxes[0][1] * dx

        PIaxes_yx = PIaxes[1][1] * dz - PIaxes[1][2] * dy
        PIaxes_yy = PIaxes[1][2] * dx - PIaxes[1][0] * dz
        PIaxes_yz = PIaxes[1][0] * dy - PIaxes[1][1] * dx

        PIaxes_zx = PIaxes[2][1] * dz - PIaxes[2][2] * dy
        PIaxes_zy = PIaxes[2][2] * dx - PIaxes[2][0] * dz
        PIaxes_zz = PIaxes[2][0] * dy - PIaxes[2][1] * dx

        daxisx = (PIaxes_xx ** 2 + PIaxes_xy ** 2 + PIaxes_xz ** 2) ** 0.5
        daxisy = (PIaxes_yx ** 2 + PIaxes_yy ** 2 + PIaxes_yz ** 2) ** 0.5
        daxisz = (PIaxes_zx ** 2 + PIaxes_zy ** 2 + PIaxes_zz ** 2) ** 0.5

        axis1MI += (daxisx ** 2) * mass
        axis2MI += (daxisy ** 2) * mass
        axis3MI += (daxisz ** 2) * mass



    UA_moI = [axis1MI, axis2MI, axis3MI]

    return UA_moI, PIaxes



def principalAxesMOI(all_data, bonded_atoms_list, COM, Hs):
    '''
    Calculate the principal axes of the MOI matrix, then calc forces.
    If Hs is True, then consider H coords in MOI calcs.
    If Hs is False, then don't include H coords, instead add their mass
    onto UA heavy atom (HA).
    '''

    coord_list = []
    mass_list = []

    for atom in bonded_atoms_list:
        if Hs == True:
            #print (atom.atom_name)
            coord_list.append(atom.coords)
            mass_list.append(atom.mass)
        elif Hs == False:
            if atom.mass > 1.1:
                UA_mass = 0 #add mass of HA and bonded Hs
                #print (atom.atom_name)
                UA_mass += atom.mass
                coord_list.append(atom.coords) #only include coords of UA
                for h in atom.bonded_to_atom_num:
                    H = all_data[h]
                    if H.mass < 1.1:
                        UA_mass += H.mass
                    else:
                        continue
                mass_list.append(UA_mass) 
                        #include mass of Hs on heavy atom mass
        else:
            continue


    moI = MOI(COM, coord_list, mass_list)
    #if Hs == False:
        #print ('masses', mass_list)
        #print ('moi', moI)


    eigenvalues, eigenvectors = LA.eig(moI) 

    #different values generated to Jon's code
    #print (eigenvalues)
    #print (eigenvectors)
    transposed = np.transpose(eigenvectors) #turn columns to rows

    #bonded_atoms_list[0].WMprincipalAxis = transposed, eigenvalues, COM
    #print (moI, eigenvalues, eigenvectors)

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

    #'''
    #PA = principal axes
    axis1PA = [0, 0, 0] #[x,y,z]
    axis2PA = [0, 0, 0] #[x,y,z]
    axis3PA = [0, 0, 0] #[x,y,z]

    #MI = moment of inertia
    axis1MI = 0 #x
    axis2MI = 0 #y
    axis3MI = 0 #z
    #'''


    for i in range(0,3):
        if eigenvalues[i] == max_eigenvalue:
            axis1PA = transposed[i]
            axis1MI = eigenvalues[i]
        elif eigenvalues[i] == min_eigenvalue:
            axis3PA = transposed[i]
            axis3MI = eigenvalues[i]
        else:
            axis2PA = transposed[i]
            axis2MI = eigenvalues[i]




    principal_axes = [axis1PA, axis2PA, axis3PA]
    MI_axis = [axis1MI, axis2MI, axis3MI]

    #print ('mI', MI_axis, bonded_atoms_list[0].atom_num)

    return principal_axes, MI_axis



def rotateFT(all_data, bonded_atoms_list, WM_principal_axes, 
        UA_principal_axes, MI_axis, molecule_COM, UA_COM, Hs):
    '''
    '''

    ###lists used for torque calcs
    coord_list = []
    mass_list = []
    force_list = []
    forces_summed = np.zeros(3)

    for atom in bonded_atoms_list:
        if Hs == True:
            coord_list.append(atom.coords)
            mass_list.append(atom.mass)
            force_list.append(atom.forces)
            forces_summed += atom.forces
        elif Hs == False:
            if atom.mass > 1.1:
                #print (atom.atom_name)
                UA_mass = 0 #add mass of HA and bonded Hs
                UA_forces = np.zeros(3)
                UA_mass += atom.mass
                UA_forces += np.array(atom.forces)
                coord_list.append(atom.coords) #only include coords of UA
                forces_summed += atom.forces
                for h in atom.bonded_to_atom_num:
                    H = all_data[h]
                    if H.mass < 1.1:
                        UA_mass += H.mass
                        UA_forces += np.array(H.forces)
                        forces_summed += H.forces
                    else:
                        continue
                mass_list.append(UA_mass) #include mass of Hs on heavy atom mass
                force_list.append(UA_forces)
        else:
            continue

    torque = [0, 0, 0] #default is no torque
    ### calc torques here with COM and PA of molecule or UA
    if UA_principal_axes != None and MI_axis != None:
        if Hs == True:
            torque = calcTorque(UA_COM, coord_list, UA_principal_axes, 
                    force_list, MI_axis)
        if Hs == False:
            torque = calcTorque(molecule_COM, coord_list, WM_principal_axes, 
                    force_list, MI_axis)



    ### Rotate forces here
    F1 = np.dot(forces_summed, WM_principal_axes[0])
    F2 = np.dot(forces_summed, WM_principal_axes[1])
    F3 = np.dot(forces_summed, WM_principal_axes[2])
    mass_sqrt = sum(mass_list) ** 0.5
    force = np.array([float(F1)/float(mass_sqrt), 
            float(F2)/float(mass_sqrt), 
            float(F3)/float(mass_sqrt)])


    return force, torque




def calcTorque(cm, coord_list, principal_axes, force_list, MI_axis):
    '''
    '''
    x_cm, y_cm, z_cm = cm[0], cm[1], cm[2]
    #MI_axis = np.sqrt(MI_axis) #sqrt moi to weight torques

    MI_axis_sqrt = []
    #print (MI_axis)
    for coord in MI_axis:
        coord_round = round(coord, 10)
        if coord_round != 0:
            c_sqrt = coord ** 0.5
            MI_axis_sqrt.append(c_sqrt)
        elif coord_round == 0:
            MI_axis_sqrt.append(coord)
        else:
            continue

    MI_axis_sqrt = np.array(MI_axis_sqrt)
    #print (MI_axis, MI_axis_sqrt)

    W_torque = np.zeros(3)
    for coord, force in zip(coord_list, force_list):
        atom_torque = np.array([0, 0, 0])
        #count_atom += 1
        new_coords = []
        new_forces = []
        for axis in principal_axes:
            new_c = axis[0]*(coord[0] - x_cm) + axis[1]*(coord[1] - y_cm) \
                    + axis[2]*(coord[2] - z_cm)
            new_f = axis[0]*force[0] + axis[1]*force[1] + axis[2]*force[2]
            new_coords.append(new_c)
            new_forces.append(new_f)
        #print (new_coords)
        #print (force)
        #CROSS PRODUCT
        torquex = float(new_coords[1]*new_forces[2] \
                - new_coords[2]*new_forces[1]) #/ float(1e10)
        torquey = float(new_coords[2]*new_forces[0] \
                - new_coords[0]*new_forces[2]) #/ float(1e10)
        torquez = float(new_coords[0]*new_forces[1] \
                - new_coords[1]*new_forces[0]) #/ float(1e10)
        #print (torquex, torquey, torquez)

        atom_torque = (torquex, torquey, torquez)
        #atom_torque = np.divide(atom_torque, MI_axis_sqrt)

        atom_torque2 = []
        for t, MI in zip(atom_torque, MI_axis_sqrt):
            if MI != 0:
                t_divide = float(t) / float(MI)
                atom_torque2.append(t_divide)
            elif MI == 0:
                atom_torque2.append(0)
            else:
                continue

        atom_torque2 = np.array(atom_torque2)

        W_torque += atom_torque2

    return (W_torque)






