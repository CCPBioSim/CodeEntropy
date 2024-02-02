from ast import arg
import sys,os
import numpy as np
from numpy import linalg as la
np.set_printoptions (threshold=sys.maxsize)
from CodeEntropy import CustomFunctions as CF
from CodeEntropy import GeometricFunctions as GF
from CodeEntropy import UnitsAndConversions as UAC
from CodeEntropy import LevelFunctions as LF
from CodeEntropy import NeighbourFunctions as NF
from CodeEntropy import Utils
from CodeEntropy import Writer
import multiprocessing as mp
from functools import partial
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt
import math

def frequency_calculation(lambdas,temp):
   """
    Function to calculate an array of vibrational frequencies from the eigenvalues of the covariance matrix.
    Calculated from eq. (3) in Higham, S.-Y. Chou, F. Gräter and  R. H. Henchman, Molecular Physics, 2018, 116, 
    1965–1976//eq. (3) in A. Chakravorty, J. Higham and R. H. Henchman, 
    J. Chem. Inf. Model., 2020, 60, 5540–5551
    frequency=sqrt(λ/kT)/2π
    Input
    -----
       lambdas : array of floats - eigenvalues of the covariance matrix
       temp: float - temperature

    Returns
    -------
       frequencies : array of floats - corresponding vibrational frequencies
    """
    pi = np.pi
    kT=UAC.get_KT2(temp)
    frequencies=1/(2*pi)*nmp.sqrt(lambdas/kT)
    return frequencies
    
def vibrational_entropy(matrix, matrix_type, temp,level): 
    """
    Function to calculate the vibrational entropy for each level calculated from eq. (4) in J. Higham, S.-Y. Chou, F. Gräter and 
    R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976/eq. (2) in A. Chakravorty, J. Higham and R. H. Henchman, 
    J. Chem. Inf. Model., 2020, 60, 5540–5551.
    
        Input
    -----
       matrix : matrix - force/torque covariance matrix
       matrix_type: string
       temp: float - temperature
       level: string  - level of hierarchy - can be "polymer", "residue" or "united_atom"
    Returns
    -------
       S_vib_total : float - transvibrational/rovibrational entropy
    """
    lambdas=la.eigvals(matrix) #eigenvalues
    frequencies=frequency_calculation(lambdas,temp)
    frequencies=frequencies.sort()
    kT=UAC.get_KT2J(temp)
    exponent=UAC.PLANCK_CONST*frequencies/kT
    power_positive= np.power(np.e,exponent)
    power_negative=np.power (np.e,-exponent)
    S_components=exponent/(power_positive-1)-np.ln(1-power_negative) 
    S_components=S_components*UAC.GAS_CONST #multiply by R - get entropy in J mol^{-1} K^{-1}
    if matrix_type=='force': #force covariance matrix 
        if level =='polymer': #polymer level - we take all frequencies into account
            S_vib_total=sum(S_components) # 3x3 force covariance matrix => 3 eigenvalues
        else:
            S_vib_total=sum(S_components[6:])  #we discard the 6 lowest frequencies to discard translation and rotation at the upper level
    else: #torque covariance matrix - we always take all values into account
        S_vib_total=sum(S_components)
    return S_vib_total
        

def conformational_entropy(dihedrals, level):
    
    """
    Function to calculate conformational entropies using eq. (7) in Higham, S.-Y. Chou, F. Gräter and 
    R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976/ eq. (4) in A. Chakravorty, J. Higham and R. H. Henchman, 
    J. Chem. Inf. Model., 2020, 60, 5540–5551.
    Uses the adaptive enumeration method (AEM).
            Input
    -----
    dihedrals : array - array of dihedrals in the molecule
    level : string - level of the hierarchy - should be "residue" or "united_atom" here
    Returns
    -------
       S_conf_total : float - conformational entropy
    """
    
    S_conf_total=0     

    
    if level == 'residue': #we don't need to go through residues - we take the dihedrals between four residues with bonds between them
        diheds_in_polymer=BinaryTree() #we have one tree for the whole polymer
        for dihedral in dihedrals: #we go through dihedrals at polymer level
            dih_node= TreeNode (None, None, dihedral) 
            diheds_in_polymer.add_node(dih_node) #we add the dihedrals to the tree
        newEntity = CONF.ConformationEntity(arg_order = len(diheds_in_polymer),arg_numFrames = numFrames) #we initialize a string array that stores the state as a distinct string for each frame- made from a coalesced character cast of numeric arrays
        DecimalReprArray = []
        #we go through the dihedrals and find the corresponding state vectors
        for i, Dihedral in enumerate (diheds_in_polymer.list_in_order()):
            stateTS = iDih.get_state_ts()
            newEntity.timeSeries[i,:] = stateTS
        for iFrame in range (numFrames):
            DecimalReprArray.append (Utils.coalesce_numeric_array(newEntity.timeSeries[:,iFrame]))
        setOfstates = set(DecimalReprArray) #we get the unique states - we get the count and compute the topographical entropy
        for istate in setOfstates:
            icount = DecimalReprArray.count(istate) #we get the count of the state
            iPlogP = icount * (nmp.log(icount) - logNumFrames)
            S_conf_total+=iPlogP
        S_conf_total/= numFrames
        S_conf_total*= UAC.GAS_CONST 
            
    if level == 'united_atom':
        for residue in molecule_dataContainer.universe.residues: #we are going through the residues
            diheds_in_residue=BinaryTree() #we build a tree for each residue
            for dihedral in dihedrals: #we go through the dihedrals
                if dihedral.is_from_same_residue() == residue.resid and dihedral.is_heavy(): #we check if dihedral is from that residue - if yes, add to tree
                    dih_node= TreeNode (None, None, dihedral) 
                    diheds_in_residue.add_node() #we add the dihedrals in that residue to the tree
            newEntity = CONF.ConformationEntity(arg_order = len(diheds_in_residue),arg_numFrames = numFrames) 
            #we initialize a string array that stores the state as a distinct string for each frame- made from a coalesced character cast of numeric arrays
            DecimalReprArray = []
            for i, Dihedral in enumerate (diheds_in_residue.list_in_order()):
                stateTS = iDih.get_state_ts()
                newEntity.timeSeries[i,:] = stateTS
            for iFrame in range (numFrames):
                DecimalReprArray.append (Utils.coalesce_numeric_array(newEntity.timeSeries[:,iFrame]))
            setOfstates = set(DecimalReprArray) #we get the unique states - we get the count and compute the topographical entropy
            S_conf_residue = 0
            for istate in setOfstates:
                icount = DecimalReprArray.count(istate) #we get the count of the state
                iPlogP = icount * (nmp.log(icount) - logNumFrames)
                S_conf_residue+=iPlogP
            S_conf_residue/=numFrames
            S_conf_residue*=UAC.GAS_CONST 
            S_conf_total+=S_conf_residue
     
    return S_conf_total
   

    

def orientational_entropy(neighbours): 
    """
    Function to calculate orientational entropies from eq. (10) in J. Higham, S.-Y. Chou, F. Gräter and R. H. Henchman, 
    Molecular Physics, 2018, 116,3 1965–1976. Number of orientations, Ω, is calculated using eq. (8) in  J. Higham, 
    S.-Y. Chou, F. Gräter and R. H. Henchman,  Molecular Physics, 2018, 116,3 1965–1976. Neighbours are identified in 
    Input
    -----
      neighbours :  dictionary - dictionary of neighbours for the molecule - should contain the type of neighbour molecule and the number of neighbour
                                molecules of that species
    Returns
    -------
       S_or_total : float - orientational entropy
    """
    S_or_total=0 
    for neighbour in neighbours: #we are going through neighbours
        if molecule in [] : #water molecules - call POSEIDON functions          
        
        else:
            omega= np.sqrt((neighbours_dict[molecule]**3)*math.pi) #always going to be larger than 1 as σ = 1
        S_molecule = probabilities_dict[molecule]* np.log(omega)*UAC.GAS_CONST
        S_or_total+=S_molecule 
    #TODO also later implement a case for molecules with hydrogen bonds but to a lesser extent than water
    return S_or_total     
   



        
    

        
