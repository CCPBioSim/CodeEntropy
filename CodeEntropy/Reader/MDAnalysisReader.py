# -*- coding: utf-8 -*-
"""
@author: bmm66251
"""

from CodeEntropy.Trajectory import TrajectoryConstants as TCON
from CodeEntropy.ClassCollection import DataContainer
from CodeEntropy.ClassCollection import BondStructs
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.FunctionCollection import UnitsAndConversions as UAC
from CodeEntropy.ClassCollection import BaseMolecule
from CodeEntropy.Trajectory import TrajectoryFrame as TF

import MDAnalysis as mda

import numpy as nmp

def read_MDAnalysis_universe(u):
    """
    

    Parameters
    ----------
    u : MDAnalysis.Universe
        DESCRIPTION.

    Returns
    -------
    newMolecule : ClassCollection.BaseMolecule
        Molecule See CodeEntropy.
    mainContainer : ClassCollection.DataContainer
        Data container See CodeEntropy.

    """
    # define an object of DataContainer class
    mainContainer = DataContainer.DataContainer()
    ####################
    newMolecule = BaseMolecule.BaseMolecule(
        arg_name='', arg_hostDataContainer=None)
    newMolecule.numCopies = 1
    newMolecule.numAtoms = len(u.atoms)
    newMolecule.numResidues = len(u.atoms.residues)
    newMolecule.numSegments = len(u.segments)
    # numCopies and numAtomsPerCopy is useless he neve read anything
    newMolecule.numCopies = 1
    newMolecule.numAtomsPerCopy = newMolecule.numAtoms // newMolecule.numCopies
    newMolecule.residArray = u.residues.resids
    newMolecule.resnameArray = u.residues.resnames
    newMolecule.segidArray = u.segments.segids
    newMolecule.atomResidueIdxArray = u.atoms.resindices
    newMolecule.atomSegmentIdxArray = u.atoms.segindices
    newMolecule.atomNameArray = u.atoms.names
    newMolecule.atomIndexArray = u.atoms.ix_array
    newMolecule.atomMassArray = u.atoms.masses
    newMolecule.atomChargeArray = u.atoms.charges
    newMolecule.hostDataContainer = mainContainer

    # #residue head (incorrect Block)
    # temp = []
    # for residsArr in u.residues.indices:
    #     temp.append(residsArr[-1])
    #
    # newMolecule.residueHead = np.array(temp)

    newMolecule.atomArray = nmp.zeros(newMolecule.numAtoms)

    newMolecule.residueHead = -1 * nmp.ones(newMolecule.numResidues)
    for aid, rid in enumerate(newMolecule.atomResidueIdxArray):
        newMolecule.atomArray[aid] = newMolecule.residueHead[rid]
        newMolecule.residueHead[rid] = aid

    # initialize
    newMolecule.initialize_element_info_arrays()
    # assign element types and priorities
    newMolecule.populate_element_info_arrays()
    #bonds
    newMolecule.bondList = u.bonds.indices
    for bond_pair in newMolecule.bondList:
        newMolecule.add_bond_tree(
            arg_aidI=bond_pair[0], arg_aidJ=bond_pair[1], arg_priorityLevelI=-1, arg_priorityLevelJ=-1)

    #Angles
    newMolecule.angleList = u.angles.indices

    #DIHEDRALS
    newMolecule.dihedList = u.dihedrals.indices
    for dihedrals in newMolecule.dihedList:
        newDih = BondStructs.Dihedral(dihedrals, newMolecule)
        newMolecule.add_dihedral(newDih)
        
    # impropers
    newMolecule.imprList = u.impropers.indices
    # skiping the other interactions as it doesn't seems to be needed in anyway see validation check in PSF file
    mainContainer.molecule = newMolecule
        
    ####################

    ### READ TRAJECTORY ###
    Utils.printflush('ReadingTrajectory')
    for ts in u.trajectory:
        newFrame = TF.TrajectoryFrame(arg_frameIndex = ts.frame, \
                                                  arg_vectorDim = TCON.VECTORDIM)
        newFrame.set_numAtoms(ts.n_atoms)
        newFrame.set_timeStep(ts.time)
        newFrame.initialize_matrices(arg_hasVelocities = ts.has_velocities, arg_hasForces = ts.has_forces)
        newFrame.value["coordinates"] = ts.positions.flatten()
        if ts.has_velocities:
            newFrame.value["velocities"] = ts.velocities.flatten()
        if ts.has_forces:
            newFrame.value["forces"] = ts.forces.flatten()
        mainContainer.trajSnapshots.append(newFrame)
        mainContainer.frameIndices.append(ts.frame)
        
        
        
    
    mainContainer.print_attributes()

    # read the coords and forces from the trajectory
    # and store them in the mainContainer
    # it is a gromacs trajectory
    mainContainer.initialize_ndarrays()

    frameIdx = 0
    #using Anstrong in MDanalysis so no need to change
    coordFactor = 1
    forceFactor = 1
    dim = TCON.VECTORDIM

    for iTrajSnapshot in mainContainer.trajSnapshots:
        mainContainer._labCoords[frameIdx] = nmp.multiply(coordFactor, nmp.reshape(
            nmp.asarray(iTrajSnapshot.value['coordinates']), (newMolecule.numAtoms, dim)))
        mainContainer._labForces[frameIdx] = nmp.multiply(forceFactor, nmp.reshape(
            nmp.asarray(iTrajSnapshot.value['forces']), (newMolecule.numAtoms, dim)))
        frameIdx += 1

    return (newMolecule, mainContainer)
