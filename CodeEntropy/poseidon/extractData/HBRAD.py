#!/usr/bin/env python

import sys
import math
import numpy as np
import logging

from CodeEntropy.poseidon.extractData.generalFunctions import angle

from datetime import datetime


def UALevelRAD(all_data, dimensions):
    '''
    UA level RAD algorithm implementation.
    Iterate through the 20 closest UAs and find which ones
    are not blocked by a closer UA.
    '''

    for x in range(0, len(all_data)):
        i = all_data[x]
        if i.mass > 1.1:
            RAD = []
            RAD_dist = []
            if i.nearest_sorted_array != None:
                range_limit = len(i.nearest_sorted_array)
                if len(i.nearest_sorted_array) > 30:
                    range_limit = 30
                for y in range(0, range_limit):
                    j = all_data[i.nearest_sorted_array[y][0]]
                    rij = float(i.nearest_sorted_array[y][1])
                    blocked = False
                    for z in range(0, y): #only closer UAs can block
                        k = all_data[i.nearest_sorted_array[z][0]]
                        rik = float(i.nearest_sorted_array[z][1])
                        costheta_jik = float(angle(j.coords, i.coords, 
                                k.coords, dimensions))
                        if math.isnan(costheta_jik):
                           break 

                        LHS = (float(1)/rij) ** 2 
                        RHS = (((float(1)/rik) ** 2) * costheta_jik)
                        if LHS < RHS:
                            blocked = True
                            break

                    if blocked == False:
                        RAD.append(j)
                        RAD_dist.append((j, rij))

            i.RAD = RAD
            i.RAD_dist = RAD_dist
        else:
            continue




    '''
    Make sure each UA is in the other UAs RAD shell if not, remove
    check RAD shell contains resids other than the reference id
    (used for systems containing surfaces that are all the same resid)
    '''

    for x in range(0, len(all_data)):
        atom = all_data[x]
        if atom.mass > 1.1:
            if len(atom.RAD) != 0:
                RAD = [] #symmetric RAD shell here
                RAD_resids = [] #checking if neighbours are diff molecules
                #make sure atom is in neighbours RAD shell
                for neighbour in atom.RAD:
                    if atom in neighbour.RAD:
                        RAD.append(neighbour)
                        RAD_resids.append(neighbour.resid)
                    else:
                        continue

                if len(RAD) != 0:
                    ##when RAD shell contains one resid only
                    if all(n == RAD_resids[0] for n in RAD_resids) == True:
                        #make sure atom is not surrounded just by itself
                        if RAD_resids[0] == atom.resid:
                            atom.RAD = []
                        else:
                            atom.RAD = RAD
                    else:
                        atom.RAD = RAD
                if len(RAD) == 0:
                    atom.RAD = RAD
            else:
                continue
        else:
            continue



def distCutoffNc(all_data, dimensions, cutoff):
    '''
    Instead of RAD, use a fixed distance cutoff
    to define the first coordination shell (Nc)
    '''


    for x in range(0, len(all_data)):
        i = all_data[x]
        if i.mass > 1.1 and i.nearest_sorted_array != None:
            range_limit = len(i.nearest_sorted_array)
            if len(i.nearest_sorted_array) > 30:
                range_limit = 30
            RAD = []
            RAD_dist = []
            for y in range(0, range_limit):
                rij = float(i.nearest_sorted_array[y][1])
                if rij <= cutoff:
                    near = \
                        all_data[i.nearest_sorted_array\
                                [y][0]]
                    RAD.append(near)
                    RAD_dist.append((near, rij))
                else:
                    continue

            i.RAD = RAD
            i.RAD_dist = RAD_dist

        else:
            continue





def HBCalc(all_data, waterTuple, dimensions):
    '''
    HB TYPE CALC METHOD 2 (including all hydrogens (not just 
    for water) and finding most closest and most negative 
    atom to donate to - (QA*QD)/r^2 approach)

    ***addition = H's can only HB to UAs that are in the RAD
    shell of their bonded heavy atom.
    '''

    for x in range(0, len(all_data)):
        atom = all_data[x]
        if atom.mass < 1.1 and atom.mass >= 1 and atom.charge > 0.1:
            acceptor_charge = 99
            H = atom
            O = None
            for bonded in H.bonded_to_atom_num:
                b = all_data[bonded]
                if b.mass > 1.1:
                    O = b
                else:
                    continue

            ##Needed for RAD shell HB ranking, HBing inside RAD
            #RAD_atom_nums = [neighbour.atom_num for neighbour in O.RAD]
            RAD_atom_nums = []
            if O != None:
                for neighbour in O.RAD:
                    RAD_atom_nums.append(neighbour.atom_num)
                    for bonded in neighbour.bonded_to_atom_num:
                        b = all_data[bonded]
                        if b.mass < 1.1:
                            RAD_atom_nums.append(b.atom_num)
                        else:
                            continue

            if H.nearest_all_atom_array != None:
                range_limit = len(H.nearest_all_atom_array)
                if len(atom.nearest_all_atom_array) > 50:
                    range_limit = 50
                for atom_dist in H.nearest_all_atom_array[:range_limit]:
                    near = all_data[atom_dist[0]]
                    HX_dist = atom_dist[1]

                    if near.atom_num not in H.bonded_to_atom_num \
                            and near.atom_num != H.atom_num \
                            and near.charge != 0 \
                            and HX_dist != 0 and O != None \
                            and near.atom_num in RAD_atom_nums:

                        X = near
                        QD = H.charge
                        QA = X.charge

                        r2 = HX_dist ** 2
                        relative_charge = (float(QD) * float(QA)) / float(r2)
                        cosine_angle = angle(O.coords, H.coords, X.coords, 
                                    dimensions)
                        angle1 = np.arccos(cosine_angle)
                        OHX_angle = np.degrees(angle1)

                        if relative_charge < acceptor_charge \
                                and float(OHX_angle) > 90:
                            acceptor_charge = relative_charge
                            H.nearest_atom = X
                            H.dist = HX_dist

                        else:
                            continue
                    else:
                        continue
            else:
                continue

        else:
            continue


    broken_HBs = False
    #find hydrogens with neighbouring eneg atoms and append those 
    #hydrogens to the nearest_Hs of those eneg atoms
    for x in range(0, len(all_data)):
        H = all_data[x]
        if H.nearest_atom != None:
            nearest_eneg = H.nearest_atom.atom_num
            all_data[nearest_eneg].nearest_Hs.append(H)

        if broken_HBs:
            ####Dealing with broken HBs - ND, bifurcated and cyclic
            ####Hard-coded for water resnames only for now
            atom = all_data[x]
            HHX = False
            for b in atom.bonded_to_atom_num:
                bonded = all_data[b]
                if atom.mass < 1.1 and bonded.mass > 1.1:
                    if bonded.bondedUA_H != None:
                        if bonded.bondedUA_H[0] == 0 and \
                                bonded.bondedUA_H[1] == 2:
                            HHX = True

            if atom.mass < 1.1 and atom.nearest_all_atom_array != None and HHX:
                for atom_dist in atom.nearest_all_atom_array[:30]:
                    atom2 = all_data[atom_dist[0]]
                    bonded_overlap = bool(set(atom.bonded_to_atom_num) & 
                            set(atom2.bonded_to_atom_num)) 
                            #check if both bonded to same atom
                    if atom2.atom_num != atom.atom_num and \
                            atom2.resid == atom.resid and \
                            atom.nearest_atom != None and \
                            atom2.nearest_atom != None and \
                            bonded_overlap == True:
                        if atom2.nearest_atom.atom_num == \
                                atom.nearest_atom.atom_num \
                                and atom2.dist < atom.dist:

                            atom.broken = ['Bifurcated', atom.atom_num,
                                    atom.atom_name,
                                    atom.nearest_atom.atom_num, 
                                    atom.nearest_atom.atom_name] #####

                            '''
                            print ('Bifurcated', atom.atom_num,
                                    atom.atom_name,
                                    atom.nearest_atom.atom_num, 
                                    atom.nearest_atom.atom_name)
                            '''

                            for b in atom.bonded_to_atom_num:
                                bonded = all_data[b]
                                bonded.broken = ['Bifurcated', atom.atom_num,
                                    atom.atom_name,
                                    atom.nearest_atom.atom_num, 
                                    atom.nearest_atom.atom_name] #####

                            #remove this broken donor from acceptor
                            new_Hlist = []
                            for H in atom.nearest_atom.nearest_Hs:
                                if H != atom:
                                    new_Hlist.append(H)
                                else:
                                    continue

                            atom.nearest_atom.nearest_Hs = new_Hlist


                            for b2 in atom.nearest_atom.bonded_to_atom_num:
                                bonded2 = all_data[b2]
                                if bonded2.mass < 1.1:
                                    if bonded2.nearest_atom != None and \
                                            bonded2.nearest_atom.resid == \
                                            atom2.resid and \
                                            bonded2.dist < atom.dist:

                                        atom.broken = ['ND', atom.atom_num, 
                                                atom.atom_name, 
                                                atom.nearest_atom.atom_num, 
                                                atom.nearest_atom.atom_name] 
                                                #####

                                        '''
                                        print ('ND', atom.atom_num, 
                                                atom.atom_name, 
                                                atom.nearest_atom.atom_num, 
                                                atom.nearest_atom.atom_name)
                                        '''

                                        for b3 in atom.bonded_to_atom_num:
                                            bonded3 = all_data[b3]
                                            bonded3.broken = ['ND', 
                                                atom.atom_num, 
                                                atom.atom_name, 
                                                atom.nearest_atom.atom_num, 
                                                atom.nearest_atom.atom_name] 
                                                #####

                                        #remove this broken donor from acceptor
                                        new_Hlist = []
                                        for H in atom.nearest_atom.nearest_Hs:
                                            if H != atom:
                                                new_Hlist.append(H)
                                            else:
                                                continue

                                        atom.nearest_atom.nearest_Hs = new_Hlist
                                        atom.nearest_atom = None

                                    else:
                                        continue
                                else:
                                    continue

                            atom.nearest_atom = None


                    if atom2.atom_num != atom.atom_num and \
                            atom2.resid == atom.resid and \
                            bonded_overlap == True and \
                            len(atom2.nearest_Hs) != 0:
                        for H in atom2.nearest_Hs:
                            if atom.nearest_atom != None and \
                                    H.resid == atom.nearest_atom.resid and \
                                    H.dist < atom.dist:

                                bonded_overlap2 = bool(set(
                                    atom.nearest_atom.bonded_to_atom_num) \
                                    & set(H.bonded_to_atom_num)) 
                                    #check if both bonded to same atom

                                if bonded_overlap2 == True:

                                    atom.broken = ['Cyclic', atom.atom_num, 
                                            atom.atom_name, 
                                            atom.nearest_atom.atom_num, 
                                            atom.nearest_atom.atom_name] #####

                                    '''
                                    print ('Cyclic', atom.atom_num, 
                                            atom.atom_name, 
                                            atom.nearest_atom.atom_num, 
                                            atom.nearest_atom.atom_name)
                                    '''

                                    for b4 in atom.bonded_to_atom_num:
                                        bonded4 = all_data[b4]
                                        bonded4.broken = ['Cyclic', 
                                            atom.atom_num, 
                                            atom.atom_name, 
                                            atom.nearest_atom.atom_num, 
                                            atom.nearest_atom.atom_name] #####

                                    #remove this broken donor from acceptor
                                    new_Hlist = []
                                    for H in atom.nearest_atom.nearest_Hs:
                                        if H != atom:
                                            new_Hlist.append(H)
                                        else:
                                            continue

                                    atom.nearest_atom.nearest_Hs = new_Hlist
                                    atom.nearest_atom = None

                            else:
                                continue



