#!/usr/bin/env python

import sys
import math
import numpy as np
import logging

import MDAnalysis
from MDAnalysis import *
from MDAnalysis.analysis import distances

from CodeEntropy.poseidon.extractData.generalFunctions import distance, calcWaterSoluteAngle

from collections import defaultdict
from collections import Counter

from datetime import datetime

nested_dict = lambda: defaultdict(nested_dict) 







def getShellAssignment(all_data, excludedResnames, dimensions, 
        startTime, verbosePrint):
    '''
    Assign solvent to solvation shells
    '''


    nearestNonlike_dict = {} #need a dictionary of all UAs that are 
        #assigned as nearest non-likes with values as a list of UAs

    if excludedResnames == None:
        excludedResnames = []

    for x in range(0, len(all_data)):
        atom = all_data[x]
        if atom.mass > 1.1 and atom.nearest_sorted_array != None:
            #and len(atom.RAD) != 0:
            nearestNonlike = None
            nearestNonlikeDist = None
            for atomDist in atom.nearest_sorted_array:
                if nearestNonlikeDist == None:
                    nonlike = all_data[atomDist[0]]
                    if nonlike.resid != atom.resid \
                            and nonlike.atom_num \
                            not in atom.bonded_to_atom_num \
                            and nonlike.atom_num != atom.atom_num \
                            and nonlike.resname != atom.resname \
                            and nonlike.resname not in \
                                    excludedResnames \
                            and len(nonlike.RAD) != 0:
                        nearestNonlike = nonlike
                        nearestNonlikeDist = atomDist[1]
                else:
                    break

            #### When nearest nonlike solute is found
            if nearestNonlikeDist != None:
                atom.nearestAnyNonlike = (nearestNonlike, 
                        nearestNonlikeDist)

                if nearestNonlike.resid not in nearestNonlike_dict:
                    nearestNonlike_dict[nearestNonlike.resid] = [[], []]
                if nearestNonlike not in \
                        nearestNonlike_dict[nearestNonlike.resid][0]:
                    nearestNonlike_dict[nearestNonlike.resid][0].append(
                            nearestNonlike)
                if atom not in nearestNonlike_dict[nearestNonlike.resid][1]:
                    nearestNonlike_dict[nearestNonlike.resid][1].append(atom)
            else:
                continue
        else:
            continue


    verbosePrint('NEAREST NON-LIKE ASSIGNMENT')
    verbosePrint(datetime.now() - startTime)
    sys.stdout.flush()




    #### Part 2: For each nonlike UA with assigned molecules, get prox shell
    #### This is the slow part of the code - prox shell assignment

    #### Find closest nonlike for each molecule, needed below
    #### so only one atom per molecule is used in prox assignment, rest
    #### of atoms in molecules are given the same prox as nearest atom
    molecule_resid_nearest_dict = {}
    for x in range(0, len(all_data)):
        atom = all_data[x]
        if atom.nearestAnyNonlike != None:
            nearestNonlike = atom.nearestAnyNonlike[0]
            dist = atom.nearestAnyNonlike[1]
            if atom.resid not in molecule_resid_nearest_dict:
                molecule_resid_nearest_dict[atom.resid] = \
                        [nearestNonlike.resid, dist]
            elif atom.resid in molecule_resid_nearest_dict:
                if dist < molecule_resid_nearest_dict[atom.resid][1]:
                    molecule_resid_nearest_dict[atom.resid] = \
                            [nearestNonlike.resid, dist]
                else:
                    continue
            else:
                continue


    multiple_shellList = []
    multiple_atomNumList = []
    localMols = 0

    for resid, nonlike_atom_lists in nearestNonlike_dict.items():
        nonlike_list = nonlike_atom_lists[0]
        atom_list = nonlike_atom_lists[1]

        for nonlike in nonlike_list:

            if len(nonlike.RAD) != 0:
                for neighbour in nonlike.RAD:
                    X = neighbour
                    if X in atom_list and X.atom_num not in \
                            multiple_atomNumList and \
                            molecule_resid_nearest_dict[X.resid][0] == resid:
                        for molecule_atom in X.molecule_atomNums:
                            ma = all_data[molecule_atom] 
                            multiple_shellList.append([ma, 1])
                            multiple_atomNumList.append(ma.atom_num)
                            ma.hydrationShellRAD_solute = 1
                            localMols += 1
                    else:
                        continue

            if len(nonlike.RAD) == 0:
                logging.warning('no RAD nonlike:', nonlike)



    ##Anything beyond the 1st shell is assigned to 2nd
    for x in range(0, len(all_data)):
        atom = all_data[x]
        if atom.hydrationShellRAD_solute == None and atom.mass > 1.1:
            atom.hydrationShellRAD_solute = 2
        else:
            continue








def moleculePositionRankingRAD(all_data, waterTuple, dimensions):
    '''
    For each UA, rank its RAD shell by proximity to that
    references UAs nearest Non-like molecule UA.
    num = RAD shell from same molecule type, when nearest nonlike 
        resid is the same as the reference.
    'X' = when same molecule type has different nearest nonlike resid.
    'RESNAME' = when molecule of different type is in RAD shell.
    '0' = closest different type molecule in RAD shell. 
            (the one its assigned to, its nearest non-like!)


    Extra:
    for each 'X' found in a RAD shell, use this information
    to find neighbouring zones. Think voronoi but with zones.
    For each molecule that defines a zone, find it's neighbouring zones
    from X's in shell type.


    The following should not occur with new HB defn (HB in RAD shell only):
    'AX' = acceptors not in RAD shell but same molecule type as ref.
    'AS' = acceptors not in RAD shell and diff mol. type to ref.
    'DX' = donor equivalent to 'AX'
    'DS' = donor equivalent to 'AS'
    '''

    for x in range(0, len(all_data)):
        atom = all_data[x]
        if atom.mass > 1.1:

            #### part 1: get ranking of RAD Nc according to extended shells.
            shell_list_ranked = []
            shell_list_atom_nums = []

            for neighbour in atom.RAD:
                shell_list_atom_nums.append(neighbour.atom_num)
                ##include Hs in shell_list_atom_nums
                for bonded in neighbour.bonded_to_atom_num:
                    b = all_data[bonded]
                    if b.mass < 1.1:
                        shell_list_atom_nums.append(b.atom_num)
                    else:
                        continue

            if len(atom.RAD) != 0 and \
                    atom.nearestAnyNonlike != None:

                for neighbour in atom.RAD:
                    if neighbour.nearestAnyNonlike != None:
                        if neighbour.nearestAnyNonlike[0].resid == \
                                atom.nearestAnyNonlike[0].resid \
                                and neighbour.resname == atom.resname:
                            shell_list_ranked.append(
                                (str(neighbour.hydrationShellRAD_solute),
                                    neighbour.atom_num))

                        if neighbour.nearestAnyNonlike[0].resid != \
                                atom.nearestAnyNonlike[0].resid \
                                and neighbour.resname == atom.resname:
                            shell_list_ranked.append(('X', 
                                neighbour.atom_num))


                        if neighbour.resname != atom.resname \
                                and neighbour.resid == \
                                atom.nearestAnyNonlike[0].resid:
                            if neighbour.atom_num == \
                                    atom.nearestAnyNonlike[0].atom_num:
                                shell_list_ranked.append(('0',
                                    neighbour.atom_num))
                            else:
                                shell_list_ranked.append(('%s_%s' % (
                                    neighbour.resname, neighbour.atom_name),
                                    neighbour.atom_num))


                        if neighbour.resid != \
                                atom.nearestAnyNonlike[0].resid \
                                and neighbour.resname != atom.resname:
                            shell_list_ranked.append(('%s_%s' % (
                                neighbour.resname, neighbour.atom_name),
                                neighbour.atom_num))


                    else:
                        continue


            #### part 2: get HBing info wrt RAD Nc ranking, only for water

            acceptor_list = []
            donor_list = []

            if atom.hydrationShellRAD_solute == 1 and \
                    atom.resname in waterTuple:

                #what Hs is atom accepting from:
                if len(atom.nearest_Hs) != 0:
                    for H in atom.nearest_Hs:
                        for bonded_atom_num in H.bonded_to_atom_num:
                            bonded = all_data[bonded_atom_num]
                            if bonded.mass > 1.1:
                                X = bonded
                                if X.atom_num in shell_list_atom_nums:
                                    for rank_num in shell_list_ranked:
                                        rank = rank_num[0]
                                        num = rank_num[1]
                                        if num == X.atom_num:
                                            acceptor_list.append((rank, num))
                                        else:
                                            continue
                                elif X.atom_num not in shell_list_atom_nums \
                                        and X.resname == atom.resname:
                                            acceptor_list.append(('AX', 
                                                X.atom_num))
                                            logging.warning('AX')

                                elif X.atom_num not in shell_list_atom_nums \
                                        and X.resname != atom.resname:
                                            acceptor_list.append(('AS', 
                                                X.atom_num))
                                            logging.warning('AS')
                                else:
                                    continue



                #what atoms are bonded Hs donating to:
                for bonded_atom_num in atom.bonded_to_atom_num:
                    H = all_data[bonded_atom_num]
                    if H.mass < 1.1 and H.nearest_atom != None:
                        X = H.nearest_atom
                        if X.mass > 1.1:
                            if X.atom_num in shell_list_atom_nums:
                                for rank_num in shell_list_ranked:
                                    rank = rank_num[0]
                                    num = rank_num[1]
                                    if num == X.atom_num:
                                        #treat a broken HB as a SD for now
                                        if (rank, num) not in donor_list:
                                            donor_list.append((rank, num))
                                            break 
                                    else:
                                        continue
                            #check donating inside RAD shell
                            elif X.atom_num not in shell_list_atom_nums \
                                    and X.resname == atom.resname:
                                        donor_list.append(('DX', 
                                            X.atom_num))
                                        logging.warning('DX', atom.atom_name, 
                                                X.atom_name)
                            elif X.atom_num not in shell_list_atom_nums \
                                    and X.resname != atom.resname:
                                        donor_list.append(('DS', 
                                            X.atom_num))
                                        logging.warning('DS', atom.atom_name,
                                                X.atom_name)
                            else:
                                continue

                        #### dealing with H-H interactions.
                        elif X.mass < 1.1:
                            UA_X = None
                            for bonded_atom_num2 in X.bonded_to_atom_num:
                                UA_X = all_data[bonded_atom_num2]
                                if UA_X.mass > 1.1 \
                                        and UA_X.atom_num in \
                                        shell_list_atom_nums:
                                    for rank_num in shell_list_ranked:
                                        rank = rank_num[0]
                                        num = rank_num[1]
                                        if num == UA_X.atom_num:
                                            #treat a H-H interaction as SD and
                                            #as unique to H-X interaction.
                                            if ('H', X.atom_num) \
                                                    not in donor_list:
                                                donor_list.append(('H', 
                                                    X.atom_num))
                                            else:
                                                continue
                                        else:
                                            continue
                                elif UA_X.mass > 1.1 \
                                        and UA_X.atom_num not in \
                                        shell_list_atom_nums:
                                    donor_list.append(('DH', 
                                        X.atom_num))
                                    logging.warning('DH', atom.atom_name,
                                            UA_X.atom_name)
                                else:
                                    continue
                                           

            shell_list_ranked.sort(key=lambda x: x[0])
            acceptor_list.sort(key=lambda x: x[0])
            donor_list.sort(key=lambda x: x[0])


            shell = [i[0] for i in shell_list_ranked]
            acceptors = [i[0] for i in acceptor_list]
            donors = [i[0] for i in donor_list]

            atom.RAD_shell_ranked = (shell, acceptors, donors)

            ### create a list for RAD neighbours and relative 
            ### distances (QQ/r^2)

            RAD_nAtoms = []
            for neighbour in atom.RAD:
                RAD_nAtoms.append([neighbour.atom_name, 
                    neighbour.resid, neighbour.resname])

            atom.RAD_nAtoms = RAD_nAtoms




