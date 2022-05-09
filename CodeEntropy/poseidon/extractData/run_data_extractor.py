#!/usr/bin/env python

import os
import sys
from sys import argv
from glob import glob
import gzip
import logging

import pickle

from collections import Counter
from datetime import datetime

from CodeEntropy.poseidon.extractData.readFiles import *
from CodeEntropy.poseidon.extractData.HBRAD import *
from CodeEntropy.poseidon.extractData.forceTorques import calculateFTMatrix
from CodeEntropy.poseidon.extractData.dihedrals import *
from CodeEntropy.poseidon.extractData.nearestNonlike2 import * 
from CodeEntropy.poseidon.extractData.outputFiles import *
from CodeEntropy.poseidon.extractData.mainClass import *





def start(container, start, end, 
        step=1, pureAtomNum=1, cutShell=None, 
        excludedResnames=None,
        water='WAT', verbose=False):
    """This is a initialization function to collect information from a MDanalysis universe into a data container for analysis using POSEIDON
    Pending Work
        - rewrite this into a class initalization for easier understanding
        - rewrite the energy part to be compulsory

    Args:
        container (MDAnalysis.Universe): A MDAnalysis object with coordinates, forces and energy (loaded to velocity field with the value [Potential Energy(index 0), Kinetic Energy(1) and dummy(2)] respectively)
        start (int): Starting Frame ID
        end (int): Ending Frame ID, this frame is not included
        step (int, optional): Steps between frame. Defaults to 1.
        pureAtomNum (int, optional): Reference molecule resid for pure liquid. Defaults to 1.
        cutShell (float, optional): Explicit cut off shell (might be buggy since part of it is not defined). Default to None which uses the relative angular distance (RAD) algorithm. See https://aip.scitation.org/doi/10.1063/1.4961439
        excludedResnames (list, optional): List of resnames to exclude from nearest non alike analysis. Defaults to None.
        water (str, optional): Resname for water molecules. Defaults to 'WAT'.
        verbose (bool, optional): print out progress of each analysis step. Defaults to False.

    Returns:
        CodeEntropy.poseidon.extractData.allAtomList: a container for CodeEntropy.poseidon.analysis to analyse
    """
    startTime = datetime.now()
    print(startTime)
    verbosePrint = print if verbose else lambda *a, **k: None
    
    waterTuple = ('SOL', 'WAT', 'HOH', 'TIP3') #needed for pdb as top file 
    if water != 'WAT':
        waterTuple = (water)

    iterations = 0
    
    all_data = []

    populateTopology(container, all_data, waterTuple)
    verbosePrint('TOPOLOGY')
    verbosePrint(datetime.now() - startTime)
    sys.stdout.flush()


    resids = Counter([(i.resname) for i in all_data])
    verbosePrint(resids.keys())
    if len(resids.keys()) == 1:
        verbosePrint('Pure system with reference ID: %s' % (pureAtomNum))

    if excludedResnames != None:
        verbosePrint('EXCLUDED RESNAMES: %s' % (excludedResnames))
    

    dimensions = None
    start = int(start)-1
    allMoleculeList = []

    for frame in range(int(start), int(end), int(step)):

        iterations += 1

        clearClass(all_data)

        all_data, dimensions = getCoordsForces(container, 
                all_data, dimensions, frame, startTime, 
                verbosePrint)

        populateEnergy(container, all_data, 
                dimensions, frame, startTime, verbosePrint)
        UAEnergyGroup(all_data)


        calculateDihedrals(all_data, dimensions)
        verbosePrint('DIH')
        verbosePrint(datetime.now() - startTime)
        sys.stdout.flush()

        #'''
        traj = container.trajectory[frame]
        neighbour_coords = None
        neighbour_coords = traj.positions

        max_cutoff = 10
        for x in range(0, len(all_data)):
            atom = all_data[x]
            #start nearest array from solutes
            if atom.resname not in waterTuple:
                getDistArray(atom, all_data, traj, max_cutoff,
                        dimensions, neighbour_coords,
                        startTime, verbosePrint)
                #find nearest array for solute neighbours that are solvent
                for nearDist in atom.nearest_all_atom_array[0:20]:
                    neighbour = all_data[nearDist[0]]
                    if neighbour.resname in waterTuple and \
                            neighbour.nearest_all_atom_array == None:
                        getDistArray(neighbour, all_data, traj, max_cutoff,
                                dimensions, neighbour_coords,
                                startTime, verbosePrint)
                    #find solvent neighbours neighbours nearest array
                    if neighbour.nearest_all_atom_array != None:
                        for nearDist2 in neighbour.nearest_all_atom_array[0:20]:
                            neighbour2 = all_data[nearDist2[0]]
                            if neighbour2.resname in waterTuple and \
                                    neighbour2.nearest_all_atom_array == None:
                                getDistArray(neighbour2, all_data, 
                                        traj, max_cutoff,
                                        dimensions, 
                                        neighbour_coords,
                                        startTime, verbosePrint)
                            else:
                                continue
                    else:
                        continue
            else:
                continue
        verbosePrint('NEAREST ARRAYS')
        verbosePrint(datetime.now() - startTime)
        sys.stdout.flush()
        #'''



        if cutShell != None:
            #Used for fixed cut-off coordination shells
            try:
                cutoff_dist = float(cutShell)
                distCutoffNc(all_data, dimensions, cutoff_dist)
                # #don't know what this is it is not defined anywhere
                # NcPairs(all_data, dimensions)
                verbosePrint('distCutoffNc')
                verbosePrint(datetime.now() - startTime)
                sys.stdout.flush() 
            except ValueError:
                logging.error('Cutoff distance needs to be a float')


        if cutShell == None:
            UALevelRAD(all_data, dimensions)
            verbosePrint('RAD')
            verbosePrint(datetime.now() - startTime)


        #if inputType != 'pdb':
        HBCalc(all_data, waterTuple, dimensions)
        verbosePrint('HB')
        verbosePrint(datetime.now() - startTime)
        sys.stdout.flush() 


        getShellAssignment(all_data, excludedResnames, dimensions, 
                startTime, verbosePrint)
        verbosePrint('PROX')
        verbosePrint(datetime.now() - startTime)
        sys.stdout.flush()


        #if (force != None):
        calculateFTMatrix(all_data, dimensions)
        verbosePrint('FTMATRIX')
        verbosePrint(datetime.now() - startTime)
        sys.stdout.flush()


        moleculePositionRankingRAD(all_data, waterTuple, dimensions)
        verbosePrint('ORIENTS')
        verbosePrint(datetime.now() - startTime)
        sys.stdout.flush()

        '''
        if iterations == 1:
            verbosePrint('Generating .pdb file for frame %s' % (frame+1))
            pdbGenerate(all_data, ('frame_%s' % (frame+1)), dimensions)
        '''



        allMoleculeList = moleculeObjectPopulation(all_data, 
                allMoleculeList, frame, dimensions)


        print(datetime.now() - startTime)
        sys.stdout.flush() #write out to output file for csf in 
                #real time rather than at end of job


    #writing file here 
    with gzip.GzipFile('moleculeListAll.obj', 'wb') as pickleFile:
        pickle.dump((allMoleculeList), pickleFile, protocol=2)
        pickleFile.close()

    print(datetime.now() - startTime)
        
    return allMoleculeList


