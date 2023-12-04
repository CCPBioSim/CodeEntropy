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

water={''}
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
    if matrix_type=='FF':
        if level =='P': #polymer
            S_vib_total=sum(S_components) # 3x3 force covariance matrix => 6 eigenvalues
        else:
            S_vib_total=sum(S_components[6:])  #for the 'M' and 'UA' levels we discard the 6 lowest frequencies
    else: #for the torque covariance matrix, we take all values into account
        S_vib_total=sum(S_components)
    return S_vib_total
        
def conformational_entropies(arg_hostDataContainer, arg_selector='all'):
    '''
    conformational entropies are calculated using eq (7) in Higham, S.-Y. Chou, F. Gräter and 
    R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976
    -used the Adaptive Enumeration Method (AEM) from previous EntropyFunctions script
    '''
    S_conf_total=0       
    allSel = arg_hostDataContainer.universe.select_atoms(arg_selector) #we select all atoms to go through
    #we browse through all residues in the system to get their dihedrals 
    for resindices in allSel.residues.resindices:
        resid = arg_hostDataContainer.universe.residues.resids[resindices]
        #we build a binary tree that hold unique dihedrals
        diheds_in_rid = CustomDataTypes.BinaryTree() #create the tree
        iAtom_in_rid = np.flip(allSel.select_atoms(f"resid {resid}").atoms.indices) 
        for idx in iAtom_in_rid: #we go trough atoms in the residue
            for iDih in arg_hostDataContainer.dihedralTable[idx]: #we go through each dihedral in the residue
                if iDih.is_from_same_residue() == resid and iDih.is_heavy(): #is_from_same_residue() and is_heavy() defined earlier - need to do that!!!!
                    dihNode = CustomDataTypes.TreeNode (None, None, iDih)
                    diheds_in_rid.add_node(dihNode) #if a dihedral is unique - we add it to the tree
        #we create an object of Class ConformationEntity for the dihedrals in each residue
        newEntity = CONF.ConformationEntity(arg_order = len(diheds_in_rid),arg_numFrames = numFrames)
        #we initialize a string array that will store the state in each frame as a distinct string - made from a coalesced character cast of numeric arrays
        ridDecimalReprArray = []
        #for each dihedral identified we get the state vector
        for i, iDih in enumerate(diheds_in_rid.list_in_order()): #need to defide this
            stateTS = iDih.get_states_ts() #define function
            new_entity.timeSeries[i,:] = stateTS
    #now we coalesce the integer labels of the constituent dihedrals in each point to get an expression of the conformation at that time
        for iFrame  in range(numFrames): #we go through all the frames
            ridDecimalReprArray.append(Utils.coalesce_numeric_array(newEntity.timeSeries[:,iFrame])) #we get all the states
        #for each unique state we get the count and compute the topographical entropy for that residue
        setOfstates =set (ridDecimalReprArray) #get a set of all states
        rid_conf_entropy=0 #conformational entropy for residue
        for iState in setOfstates:
            iCount = ridDecimalReprArray.count(iState) #we look at the degeneracy of states
            iPlogP = iCount * (np.log(iCount) - logNumFrames) #we calculate plog p for each state
            rid_conf_entropy += iPlogP
        rid_conf_entropy /=numFrames
        rid_conf_entropy *= -UAC.GAS_CONST #multiply by R - get entropy in J mol^{-1} K^{-1}
        S_conf_total += rid_conf_entropy
    return S_conf_total
        
def orientational_entropy():
''' Ω calculated from eq. (8) in Higham, S.-Y. Chou, F. Gräter and R. H. Henchman, Molecular Physics, 2018, 116,3' 1965–1976
    σ = 2 for water + divide by 4 OR 1 for other molecules = ligands we're concerned with
    -might need to add another case for molecules with high symmetry about 1 axis - e.g. methanol, ethane
    orientational entropies are calculated using eq. (10) in Higham, S.-Y. Chou, F. Gräter and 
    R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976 for molecules other than water
'''    
# assuming we identify neighbours before and could have a dictionary of neighbours or similar structure - e.g. neighbours= {'SOL': x, 'LIG': y} - identified using RAD for each orientation
    S_or_total=0
    neighbours_dict= RAD() #could do a separate function for identifying neighbours 
    for molecule in neighbours_dict: #we are going through neighbo
    #get the probabilities from somewhere - TBD
        if molecule in [] : #water flag'
            
             omega= np.sqrt((neighbours_dict[molecule]**3)*math.pi)*0.125 - #multiply by (HBav)/σ - HBav =0.25 as each water molecule is equally likely 
                                                                             #to donate and accept H bonds  
            
            
        else:
            omega= np.sqrt((neighbours_dict[molecule]**3)*math.pi) #always going to be larger than 1 as σ = 1
        S_molecule = probabilities_dict[molecule]* np.log(omega)*UAC.GAS_CONST
        S_or_total+=S_molecule 
    return S_or_total
                

    





        
    

        
