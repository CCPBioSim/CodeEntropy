#!/usr/bin/env python3

from ast import arg
import sys,os
import numpy as np
from numpy import linalg as la
np.set_printoptions (threshold=sys.maxsize)
from CodeEntropy.ClassCollection import BeadClasses as BC
from CodeEntropy.ClassCollection import ConformationEntity as CONF
from CodeEntropy.ClassCollection import ModeClasses
from CodeEntropy.ClassCollection import CustomDataTypes
from CodeEntropy.FunctionCollection import CustomFunctions as CF
from CodeEntropy.FunctionCollection import GeometricFunctions as GF
from CodeEntropy.FunctionCollection import UnitsAndConversions as UAC
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.IO import Writer
import multiprocessing as mp
from functools import partial
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt
import math
def frequency_calculation(lambdas,temp):
    '''
    
    frequencies are calculated from eq. (3) in Higham, S.-Y. Chou, F. Gräter and 
    R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976
    
    '''
    pi=np.pi
    kT=UAC.get_KT2(temp)
    frequencies=1/(2*pi)*np.sqrt(lambdas/kT)
    return frequencies
    
def vibrational_entropies(matrix, matrix_type, temp,level): #force or torque covariance matrix at a given level of entropy
    '''
    
    vibrational entropy calculated from eq. (4) in Higham, S.-Y. Chou, F. Gräter and 
    R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976
    
    '''
    lambdas=la.eigvals(matrix)
    frequencies=frequency_calculation(lambdas,temp)
    frequencies=frequencies.sort()
    kT=UAC.get_KT2J(temp)
    exponent=UAC.PLANCK_CONST*frequencies/kT
    power_positive= np.power(np.e,exponent)
    power_negative=np.power (np.e,-exponent)
    S_components=exponent/(power_positive-1)-np.ln(1-power_negative)
    S_components=S_components*UAC.GAS_CONST #multiply by R - get entropy in J mol^{-1} K^{-1}
    if matrix_type='FF':
        if level =='P': #polymer
            S_vib_total=sum(S_components) # 3x3 force covariance matrix => 6 eigenvalues
        else:
            S_vib_total=sum(S_components[6:]  #for the 'M' and 'UA' levels we discard the 6 lowest frequencies
    else: #for the torque covariance matrix, we take all values into account
        S_vib_total=sum(S_components)
    return S
        
def conformational_entropies(arg_hostDataContainer, arg_selector='all'):
    '''
    conformational entropies are calculated using eq (7) in Higham, S.-Y. Chou, F. Gräter and 
    R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976
    -used the Adaptive Enumeration Method (AEM) from previous EntropyFunctions script
    '''
    S_conf_total=0       
    allSel = arg_hostDataContainer.universe.select_atoms(arg_selector)
    #we browse through all residues in the system to get their dihedrals 
    resindices  allSel.residues.resindices:
        Utils.printflush('Working on resid : {} ({})'.format (arg_hostDataContainer.universe.residues.resids[resindices], 
                         arg_hostDataContainer.universe.residues.resnames[resindices]), end= '')
    resid = arg_hostDataContainer.universe.residues.resids[resindices]
    #we build a binary tree that hold unique dihedrals
    diheds_in_rid = CustomDataTypes.BinaryTree()
    iAtom_in_rid = np.flip(allSel.select_atoms(f"resid {resid}").atoms.indices)
    idx iAtom_in_rid:
        iDih arg_hostDataContainer.dihedralTable[idx]:
            iDih.is_from_same_residue() == resid  iDih.is_heavy():
            dihNode = CustomDataTypes.TreeNode (None, None, iDih)
            diheds_in_rid.add_node(dihNode) #if a dihedral is unique - we add it to the tree
    Utils.printflush('Found {} exclusive dihedrals in residue {}{}'.\format(len(diheds_in_rid),
                    arg_hostDataContainer.universe.residues.resnames[resindices],
                    arg_hostDataContainer.universe.residues.resids[resindices]))
    #we create an object of ClassConformationEntity 
    newEntity = CONF.ConformationEntity(arg_order = len(diheds_in_rid),arg_numFrames = numFrames)
    #we initialize a string array that will store the state in each frame as a distinct string - made from a coalesced character cast of numeric arrays
    ridDecimalReprArray = []
    #for each dihedral identified we get the state vector
    i, iDih enumerate(diheds_in_rid.list_in_order()):
    stateTS = iDih.get_states_ts(arg_verbose=arg_verbose)
    new_entity.timeSeries[i,:] = stateTS
    #now we coalesce the integer labels of the constituent dihedrals in each point to get an expression of the conformation at that time
    iFrame  range(numFrames):
        ridDecimalReprArray.append(Utils.coalesce_numeric_array(newEntity.timeSeries[:,iFrame]))
        #for each unique state we get the count and compute the topographical entropy for that residue
        setOfstates =set (ridDecimalArray)
        Utils.printflush('Found {} dihedrals which collectively acquire {} unique conformers'.format(len(diheds_in_rid), len(setOfstates)))
        print(ridDecimalReprArray)
        rid_conf_entropy=0
        iState  setOfstates:
        iCount = ridDecimalReprArray.count(iState)
        # p Log p for this state
        iPlogP = iCount * (np.log(iCount) - logNumFrames)
        rid_conf_entropy += iPlogP;
        rid_conf_entropy /=numFrames;
        rid_conf_entropy *= -UAC.GAS_CONST #multiply by R - get entropy in J mol^{-1} K^{-1}
        Utils.printflush('{:<40s}:{:.4f}'.format('Residue Topographical Entropy from AEM ({}{})'.format(arg_hostDataContainer.universe.residues.resnames[resindices],
                     arg_hostDataContainer.universe.residues.resids[resindices]), rid_conf_entropy ))
        arg_outFile != None:
        Utils.printOut(arg_outFile, '{:<40s} : {:.4f}'.format ('Residue Topographical Entropy from AEM ({} {})'.format(arg_hostDataContainer.universe.residues.resnames[resindices],
                       arg_hostDataContainer.universe.residues.reids[resindices]), rid_conf_entropy)
        S_conf_total += rid_conf_entropy
        return S_conf_total
        
def orientational_entropy(water_flag):
''' Ω calculated from eq. (8) in Higham, S.-Y. Chou, F. Gräter and R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976
    σ = 2 for water + divide by 4 OR 1 for other molecules = ligands we're concerned with
    -might need to add another case for molecules with high symmetry about 1 axis - e.g. methanol, ethane
 orientational entropies are calculated using eq. (10) in Higham, S.-Y. Chou, F. Gräter and 
    R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976
'''    
    S_or_total=0
    if water_flag:
        
    else:
        omega= np.sqrt((Nc**3)*math.pi) #array of neighbours
    S_component = p_Nc* np.log(omega)*UAC.GAS_CONST 
    S_or_total=sum(S_or_total)  
    
def total_entropy(matrix,T):
    S_vib_P = vibrational_entropies(matrix,T, 'P')
    print("$S^{vib}_P$:",S_vib_P)
    S_vib_M = vibrational_entropies(matrix,T, 'M')
    print("$S^{vib}_M$:",S_vib_M)
    S_vib_UA = vibrational_entropies(matrix,T, 'UA')
    print("$S^{vib}_{UA}$:",S_vib_UA)
    S_conf = conformational_entropy()
    print("$S^{conf}$:",S_conf)
    S_or = orientational_entropy()
    print("$S^{conf}$:",S_or)
    S_total = S_vib_P + S_vib_M + S_vib_UA + S_conf + S_or
    print(S_total)

total_entropy(matrix, 298)


        
    

        
