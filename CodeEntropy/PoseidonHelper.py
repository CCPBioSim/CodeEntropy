import sys
import logging

from CodeEntropy.poseidon.extractData.readFiles import populateTopology, getCoordsForces, getDistArray
# # Energy is not needed
# from CodeEntropy.poseidon.extractData.readFiles import populateEnergy, UAEnergyGroup
from CodeEntropy.poseidon.extractData.HBRAD import distCutoffNc, UALevelRAD, HBCalc
from CodeEntropy.poseidon.extractData.forceTorques import calculateFTMatrix
from CodeEntropy.poseidon.extractData.dihedrals import calculateDihedrals
from CodeEntropy.poseidon.extractData.nearestNonlike2 import getShellAssignment, moleculePositionRankingRAD 
from CodeEntropy.poseidon.extractData.outputFiles import moleculeObjectPopulation
from CodeEntropy.poseidon.extractData.mainClass import clearClass

from CodeEntropy.poseidon.analysis.populateClasses import classPopulation
from CodeEntropy.poseidon.analysis.EECalculation import processEE
from CodeEntropy.poseidon.analysis.helper import memoryInfo, weightingPopulation

from datetime import datetime

def frame_iteration(container, all_data, dimensions, startTime, verbosePrint, waterTuple, cutShell, excludedResnames, frame):
    clearClass(all_data)
    print(f"frame = {frame}")

    all_data, dimensions = getCoordsForces(container, 
            all_data, dimensions, frame, startTime, 
            verbosePrint)
    # # Energy is not needed
    # populateEnergy(container, all_data, 
    #         dimensions, frame, startTime, verbosePrint)
    # UAEnergyGroup(all_data)


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
    #print(all_data, frame, dimensions)
    return (all_data, frame, dimensions)
