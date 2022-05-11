#!/usr/bin/env python

import sys
import math
import numpy as np


from CodeEntropy.poseidon.extractData.generalFunctions import *


def calculateDihedrals(all_data, dimensions):
    '''
    In-place modification of all_data to calculate dihedrals
    For each dihedral in system, find what conformation it is in
    Dihedrals:


               X4
               /
              /
       X2---X3
      /
     /
    X1


    b1 = vector(X1, X2)
    b2 = vector(X2, X3)
    b3 = vector(X3, X4)

    source = https://math.stackexchange.com/questions/47059/
        how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates 

    Args:
        all_data (list of poseidon.atom_info): Container for all atom info
        dimensions (numpy array of size (1,3)): dimension of box of current frame
    '''

    rad2deg = float(180)/float(np.pi)

    dihedrals = []
    all_atom_nums_list = []

    max_atom_num = max(all_data[x].atom_num
            for x in range(0, len(all_data)))


    for x in range(0, len(all_data)):
        atom = all_data[x]
        if atom.mass > 1.1 and len(atom.dihedral_list) != 0 and \
                atom.atom_num not in all_atom_nums_list:

            Mdihedrals = [] #unique dihedrals for each atom in molecule
            lowest_atom_num = max_atom_num + 1
            for molecule_atom in atom.molecule_atomNums:
                Matom = all_data[molecule_atom]
                all_atom_nums_list.append(Matom.atom_num)
                #get lowest atom num for renumbering
                for dih in Matom.dihedral_list:
                    for d in dih:
                        if all_data[d].resid == atom.resid:
                            if d < lowest_atom_num:
                                lowest_atom_num = d
                            else:
                                continue

            for molecule_atom in atom.molecule_atomNums:
                Matom = all_data[molecule_atom]
                for dih in Matom.dihedral_list:

                    inside_resid = True
                    for d in dih:
                        if all_data[d].resid != atom.resid:
                            inside_resid = False

                    if dih not in dihedrals and inside_resid == True: 
                        #only analyse dihedral once
                        dihedrals.append(dih)

                        d_atoms = [] 
                        atom_nums = [] #1,2,3,4
                        for d in dih: #calc dih angles for each dih
                            atom2 = all_data[d]
                            d_atoms.append(atom2)
                            atom_nums.append(atom2.atom_num)

                        
                        #re-number atom_nums
                        renumbered_atom_nums = []
                        for num in atom_nums:
                            renumbered = (num - lowest_atom_num) + 1
                            renumbered_atom_nums.append(renumbered)

                        ##get vectors
                        b1_vector = vector(d_atoms[0].coords, 
                                d_atoms[1].coords, dimensions)
                        b1_dist = distance(d_atoms[0].coords, 
                                d_atoms[1].coords, dimensions)
                        b1_norm = np.divide(b1_vector, b1_dist)

                        b2_vector = vector(d_atoms[1].coords, 
                                d_atoms[2].coords, dimensions)
                        b2_dist = distance(d_atoms[1].coords, 
                                d_atoms[2].coords, dimensions)
                        b2_norm = np.divide(b2_vector, b2_dist)

                        b3_vector = vector(d_atoms[2].coords, 
                                d_atoms[3].coords, dimensions)
                        b3_dist = distance(d_atoms[2].coords, 
                                d_atoms[3].coords, dimensions)
                        b3_norm = np.divide(b3_vector, b3_dist)

                        ## get normals
                        n1 = np.cross(b1_norm, b2_norm)
                        n2 = np.cross(b2_norm, b3_norm)
                        m = np.cross(n1, b2_norm)

                        #get dot products
                        x = np.dot(n1, n2)
                        y = np.dot(m, n2)

                        phi = rad2deg * np.arctan2(y, x)
                        dih_type = None

                        ## only needed for fixed-dihedrals
                        if phi >= 120 or phi < -120:
                            dih_type = [0, 'trans']

                        if phi >= 0 and phi < 120:
                            dih_type = [1, 'g-']

                        if phi >= -120 and phi < 0:
                            dih_type = [2, 'g+']


                        Mdihedrals.append([renumbered_atom_nums, 
                            int(round(phi, 0)), 
                            dih_type])

                    else:
                        continue

            atom.dihedral_phi_type = Mdihedrals

        else:
            continue

