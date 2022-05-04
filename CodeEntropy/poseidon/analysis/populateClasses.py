#!/usr/bin/env python

import os
import sys
import psutil

import math
import numpy as np


from collections import Counter

from CodeEntropy.poseidon.analysis.classes import *
from CodeEntropy.poseidon.analysis.populateEE import SorPopulation, SvibPopulation, \
        SconfPopulation, contactPopulation, \
        contactPopulationUA


def classPopulation(atomList, entropyEnergy, 
        level_list, count, atom_count, 
        num_atoms, waterTuple, 
        temperature,  
        solvent, 
        EEclass, EEclass_residLevel_resname, EEclass_soluteContacts, 
        EEclass_atomLevel,
        weighting_info, verbosePrint):
    '''
    For each flag, populate dictionaries accordingly
    The objects read in are used for this.
    '''

    ##### Smix_methods
    step_list = None

    #'''
    if count == 1: 

        #initiate classes for each flag

        if entropyEnergy:
            EEclass = variables_dicts(True)
            EEclass_residLevel_resname = variables_dicts(True)
            EEclass_soluteContacts = variables_dicts(True)
            EEclass_atomLevel = variables_dicts(True)


        if weighting_info != None:
            if EEclass == None:
                EEclass = variables_dicts(True)
            EEclass.weighting_list = weighting_info[0]
            EEclass.weighting_chunks = weighting_info[1]


    #'''


    for i in atomList: #iterate through each atom
        ####### only need to change this if list order changes (and below)
        atom_count += 1
        globalFrame = math.ceil(atom_count / float(num_atoms)) #round UP
        frame = i[0] #not used
        cumulative_steps = ['All']
        if step_list != None:
            for n in range(1, len(step_list)):
                step = step_list[n]
                if int(globalFrame) <= step:
                    cumulative_step = str(step)
                    #print(globalFrame, cumulative_step)
                    cumulative_steps.append(cumulative_step)

        atom_name = i[1]
        atom_num = i[2]
        atom_resname = i[3]
        atom_resid = i[4]
        N_H = i[5][1]
        #molecule_atoms = i[6] #heavy atoms only
        molecule_size = len(i[6]) #heavy atoms only
        try:
            nearest_atomName = i[7][0]
            nearest_resname = i[7][1]
            nearest_resid = i[7][2]
            #dist = float(i[7][3])
            #dist = round(float(i[7][3]), 1)
        except IndexError:
            nearest_atomName = 'unassigned'
            nearest_resname = 'unassigned'
            nearest_resid = 0
            #dist = 0
        RADshell_num = i[8]
        if RADshell_num == None:
            RADshell_num = '0' #unassigned to a nearest
        RAD_list = sorted(i[9][0])
        As = i[9][1]
        Ds = i[9][2]
        ### Svib_Ecalcs
        MweightedForces = i[10] #array
        MweightedTorques = i[11] #array
        PE = None
        KE = None
        if i[12] != None:
            PE = i[12][0]
            KE = i[12][1]

        UAweightedForces = i[13]
        UAweightedTorques = i[14]
        #### FT POPULATION
        try:
            dihedral_phi_list = i[15]
            Ndih = len(i[15])
        except (IndexError, TypeError):
            dihedral_phi_list
            Ndih = None
        try:
            molecule_UA_F = i[16]
            molecule_UA_T = i[17]
        except IndexError:
            Ndih = None
            molecule_UA_F = []
            molecule_UA_T = []
        RAD_nAtoms = i[18] #closest atom in each RADshell and distance

        RAD_str = tuple(sorted(RAD_list))

        ### toggle between shells or distances as x variable
        RADshell_num_dist = str(RADshell_num)

        #### group all solute types together for easier analysis
        if atom_resname not in solvent and RADshell_num != '0':
            nearest_resname = 'Any'
            RADshell_num_dist = '1'


        #### find out weighting of each frame if simulation is biased
        weight = 1
        if weighting_info != None:
            if EEclass.weighting_list != None:
                weight = getWeight(EEclass, globalFrame)

        for level in level_list:

            nearest_resname_list = []
            atom_resnameLevel, nearest_resname_list, waterTuple \
                    = levelSelection(level, 
                    atom_resname, atom_resid, nearest_resname, 
                    nearest_resname_list, atom_name, 
                    nearest_atomName, nearest_resid, RAD_nAtoms, waterTuple)


            for nearest_resnameLevel in nearest_resname_list:

                if 'Any' in nearest_resnameLevel:
                    nearest_resnameLevel = 'Any'


                if level in [None, 'moleculeLevel']:

                    if entropyEnergy:

                        SorPopulation(EEclass, 
                                atom_name, atom_resnameLevel, 
                                nearest_resnameLevel, RAD_str, 
                                RADshell_num, RADshell_num_dist, 
                                waterTuple, N_H, 
                                As, Ds, weight)

                        SvibPopulation(EEclass, level, atom_name, 
                                atom_resnameLevel, 
                                N_H, atom_num, 
                                nearest_resnameLevel, 
                                RADshell_num_dist, waterTuple, 
                                UAweightedForces, UAweightedTorques, PE, KE, 
                                MweightedForces, 
                                MweightedTorques, Ndih, molecule_UA_F, 
                                molecule_UA_T, weight)

                        SconfPopulation(EEclass, nearest_resnameLevel, 
                                atom_name, atom_resnameLevel, 
                                N_H, dihedral_phi_list, molecule_size, 
                                RADshell_num_dist, weight)



                if level in ['residLevel_resname']:
                    if entropyEnergy:

                        contactPopulation(EEclass_residLevel_resname, 
                                RAD_nAtoms, 
                                atom_resnameLevel, atom_resid, waterTuple, 
                                weight)

                        SorPopulation(EEclass_residLevel_resname, 
                                atom_name, atom_resnameLevel, 
                                nearest_resnameLevel, RAD_str, 
                                RADshell_num, RADshell_num_dist, 
                                waterTuple, N_H, 
                                As, Ds, weight)

                        SvibPopulation(EEclass_residLevel_resname, 
                                level, atom_name, atom_resnameLevel, 
                                N_H, atom_num, 
                                nearest_resnameLevel, 
                                RADshell_num_dist, waterTuple, 
                                UAweightedForces, UAweightedTorques, PE, KE, 
                                MweightedForces, 
                                MweightedTorques, Ndih, molecule_UA_F, 
                                molecule_UA_T, weight)

                        SconfPopulation(EEclass_residLevel_resname, 
                                nearest_resnameLevel, 
                                atom_name, atom_resnameLevel, 
                                N_H, dihedral_phi_list, molecule_size, 
                                RADshell_num_dist, weight)


                if level in ['soluteContacts']:

                    if entropyEnergy:
                        SorPopulation(EEclass_soluteContacts, 
                                atom_name, atom_resnameLevel, 
                                nearest_resnameLevel, RAD_str, 
                                RADshell_num, RADshell_num_dist, 
                                waterTuple, N_H, 
                                As, Ds, weight)

                        SvibPopulation(EEclass_soluteContacts, 
                                level, atom_name, atom_resnameLevel, 
                                N_H, atom_num, 
                                nearest_resnameLevel, 
                                RADshell_num_dist, waterTuple, 
                                UAweightedForces, UAweightedTorques, PE, KE, 
                                MweightedForces, 
                                MweightedTorques, Ndih, molecule_UA_F, 
                                molecule_UA_T, weight)

                        SconfPopulation(EEclass_soluteContacts, 
                                nearest_resnameLevel, 
                                atom_name, atom_resnameLevel, 
                                N_H, dihedral_phi_list, molecule_size, 
                                RADshell_num_dist, weight)



                if level in ['atomLevel']:

                    if entropyEnergy:

                        contactPopulationUA(EEclass_atomLevel, 
                                RAD_nAtoms, atom_name,
                                atom_resnameLevel, atom_resid, waterTuple, 
                                weight)

                        SorPopulation(EEclass_atomLevel, 
                                atom_name, atom_resnameLevel, 
                                nearest_resnameLevel, RAD_str, 
                                RADshell_num, RADshell_num_dist, 
                                waterTuple, N_H, 
                                As, Ds, weight)

                        SvibPopulation(EEclass_atomLevel, 
                                level, atom_name, atom_resnameLevel, 
                                N_H, atom_num, 
                                nearest_resnameLevel, 
                                RADshell_num_dist, waterTuple, 
                                UAweightedForces, UAweightedTorques, PE, KE, 
                                MweightedForces, 
                                MweightedTorques, Ndih, molecule_UA_F, 
                                molecule_UA_T, weight)

                        SconfPopulation(EEclass_atomLevel, 
                                nearest_resnameLevel, 
                                atom_name, atom_resnameLevel, 
                                N_H, dihedral_phi_list, molecule_size, 
                                RADshell_num_dist, weight)

    sys.stdout.flush() 

    return EEclass, EEclass_residLevel_resname, \
            EEclass_soluteContacts, EEclass_atomLevel, \
            atom_count




def getWeight(EEclass, globalFrame):
    '''
    Find out what the weighting is for a given simualtion frame
    '''

    #chunk = int(EEclass.weighting_chunks)
    weighting_list = EEclass.weighting_list

    index = globalFrame-1
    try:
        weight = float(weighting_list[index])
        #print('weight: %s' % (weight))
    except (IndexError, ValueError):
        weight = 1

    #print(globalFrame, index, weight)

    return weight




def stripNumbers(atom_name, atom_resname):
    '''
    '''

    atom_name_letters = ''.join([i for i in atom_name 
            if not i.isdigit()])
    if atom_name != atom_resname:
        atom_name_letters = atom_name_letters[0]
    atom_resname = ('%s_%s' % (atom_resname, atom_name_letters))

    return atom_resname






def levelSelection(level, atom_resname, atom_resid, nearest_resname, 
        nearest_resname_list, 
        atom_name, nearest_atomName, nearest_resid, RAD_nAtoms, waterTuple):
    '''
    Depending on what level of analysis we want to do, dictionaries are
    populated according to how the nearest and assigned are defined.

    For example, we may want to analyse molecule types, the atoms 
    in each molecule, molecules by residue number or multiple solutes 
    in a RAD shell as a list of nearest molecules.
    '''

    ###keep as molecule
    if level == 'moleculeLevel' or level == None:
        atom_resname = atom_resname
        nearest_resname_list = [nearest_resname]


    ###use resname_atomName for both nearest and assigned
    if level == 'atomLevel': #res_atomLevel
        #'''
        water = False
        if atom_resname in waterTuple:
            water = True
        atom_resname = stripNumbers(atom_name, atom_resname)
        if atom_resname not in waterTuple and water:
            waterTuple.append(atom_resname)
        nearest_resname_list = [stripNumbers(nearest_atomName, 
            nearest_resname)]

        #'''

    '''
    ###Inclusion of atom names, no edits
    if level == 'atomLevel':
        atom_resname = ('%s_%s' % (atom_resname, atom_name))
        nearest_resname_list = [('%s_%s' % (nearest_resname, 
                nearest_atomName))]
    '''

    ###Inclusion of molecule resid
    if level == 'residLevel_resname':
        atom_resname = atom_resname
        if atom_resname not in waterTuple:
            atom_resname = '%s_%s' % (atom_resname, atom_resid)
        nearest_resname_list = [('%s_%s' % (nearest_resname, 
                nearest_resid))]
        #print(atom_resname, nearest_resname_list)

    ###Inclusion of molecule resid
    if level == 'residLevel_atom':
        atom_resname = ('%s_%s' % (atom_resname, atom_name))
        nearest_resname_list = [('%s_%s' % (nearest_resname, 
                nearest_resid))]


    if level == 'soluteContacts':
        atom_resname = atom_resname
        if atom_resname in waterTuple:
            if RAD_nAtoms != None:
                name_id_list = []
                for nInfo in RAD_nAtoms:
                    nResid = nInfo[1]
                    nResname = nInfo[2]

                    if nResname not in waterTuple:
                        name_id = ('%s_%s' % (nResname, nResid))
                        name_id_list.append(name_id)

                name_id_list = sorted(name_id_list)

                if len(name_id_list) > 1:
                    for solute in name_id_list:
                        list2 = name_id_list[:]
                        list2.remove(solute)
                        for solute2 in list2:
                            if ('%s_%s' % (solute, solute2)) not in \
                                    nearest_resname_list:
                                nearest_resname_list.append(
                                        ('%s_%s' % (solute, solute2)))


    return atom_resname, nearest_resname_list, waterTuple





