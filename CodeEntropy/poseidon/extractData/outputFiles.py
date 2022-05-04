#!/usr/bin/env python

import math
import numpy as np
import sys

from CodeEntropy.poseidon.extractData.generalFunctions import calcAngleWithNearestNonlike


def moleculeObjectPopulation(all_data, allMoleculeList, frame, dimensions):
    '''
    For all molecules in the system, populate a list with required
    info for further analysis across any trajectory frame.
    '''


    for x in range(0, len(all_data)):
        atom = all_data[x]
        if atom.mass > 1.1:
            nearestInfo = []
            if atom.nearestAnyNonlike != None:
                nearestInfo = [
                    atom.nearestAnyNonlike[0].atom_name,
                    atom.nearestAnyNonlike[0].resname,
                    atom.nearestAnyNonlike[0].resid,
                    round(atom.nearestAnyNonlike[1], 3) #dist
                    ]

            allMoleculeList.append([
                frame, #0
                atom.atom_name, #1
                atom.atom_num, #2
                atom.resname, #3
                atom.resid, #4
                atom.bondedUA_H, #5
                atom.molecule_atomNums, #6
                nearestInfo, #7
                atom.hydrationShellRAD_solute, #8
                atom.RAD_shell_ranked, #9
                atom.MweightedForces, #10
                atom.MweightedTorques, #11
                atom.UA_PKenergy, #12
                atom.UAweightedForces, #13
                atom.UAweightedTorques, #14
                atom.dihedral_phi_type, #15
                atom.molecule_UA_Fs, #16
                atom.molecule_UA_Ts, #17
                atom.RAD_nAtoms, #18
                ])


        else:
            continue


    return allMoleculeList





def pdbGenerate(all_data, fileName, dimensions):
    '''
    Output a pdb file to visualise different waters in system.
    Might have to also make an equivalent .csv file so that
    we have an easier time analysing the files.
    But append to one global file rather than each frame, 
    so need a column of frame number for analysis later.
    pdb file pure for producing images with vmd, 
    but could create these later anyway? 
    Then we can choose our beta column?
    '''

    data = open(fileName+'_solProx_dist.pdb', "w") 
        #solute resid, int(dist) from solute resid

    data3 = open(fileName+'_RADHBlengths.pdb', "w") 
        #Nc, N_acceptors/N_donors

    data6 = open(fileName+'_solProx_RAD.pdb', "w") 
        #solute resid, int(dist) from solute resid

    file_list = [data, data3, data6]
    
    for x in range(0, len(all_data)):
        atom = all_data[x]


        try:
            a = int(atom.nearestAnyNonlike[0].resid)
        except TypeError:
            a = 0
        try:
            b = int(atom.nearestAnyNonlike[1])
        except TypeError:
            b = 0
        try:
            e = len(atom.RAD)
        except TypeError:
            e = 0
        try:
            As = len(atom.RAD_shell_ranked[1])
            Ds = len(atom.RAD_shell_ranked[2])
            DAhh_type = '%s%s' % (Ds, As)
            f = str(DAhh_type)
        except TypeError:
            f = '00'
        try:
            i = int(atom.hydrationShellRAD_solute)
        except (TypeError, ValueError):
            i = 0

        data.write("\n".join(["{:6s}{:5d} {:^4s}{:1s}{:3s}" \
                " {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}" \
                "{:6.0f}{:6.0f}          {:>2s}{:2s}".format(
                    "ATOM", int(str(atom.atom_num)[0:5]), 
                    atom.atom_name, " ", atom.resname[0:3], 
                    "A", int(str(atom.resid)[0:4]), " ", 
                    float(atom.coords[0]), 
                    float(atom.coords[1]), 
                    float(atom.coords[2]), 
                    a, b, " ", " ")]) + "\n")

        data3.write("\n".join(["{:6s}{:5d} {:^4s}{:1s}{:3s}" \
                " {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}" \
                "{:6.0f}    {:2s}          {:>2s}{:2s}".format(
                    "ATOM", int(str(atom.atom_num)[0:5]), 
                    atom.atom_name, " ", atom.resname[0:3], 
                    "A", int(str(atom.resid)[0:4]), " ", 
                    float(atom.coords[0]), 
                    float(atom.coords[1]), 
                    float(atom.coords[2]), 
                    e, f, " ", " ")]) + "\n")
        
        data6.write("\n".join(["{:6s}{:5d} {:^4s}{:1s}{:3s}" \
                " {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}" \
                "{:6.0f}{:6.0f}          {:>2s}{:2s}".format(
                    "ATOM", int(str(atom.atom_num)[0:5]), 
                    atom.atom_name, " ", atom.resname[0:3], 
                    "A", int(str(atom.resid)[0:4]), " ", 
                    float(atom.coords[0]), 
                    float(atom.coords[1]), 
                    float(atom.coords[2]), 
                    a, i, " ", " ")]) + "\n")

    for d in file_list:
        d.close()


