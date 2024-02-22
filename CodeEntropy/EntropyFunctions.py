from ast import arg
import sys,os
import numpy as nmp
from numpy import linalg as la
nmp.set_printoptions (threshold=sys.maxsize)
from CodeEntropy import CustomFunctions as CF
from CodeEntropy import UnitsAndConversions as UAC
from CodeEntropy import NeighbourFunctions as NF
from CodeEntropy import Utils
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
    pi = nmp.pi
    kT = UAC.get_KT2(temp)
    frequencies = 1/(2*pi)*nmp.sqrt(lambdas/kT)
    return frequencies
#END frequency_calculation
    
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
    # Get eigenvalues of the given matrix
    lambdas = la.eigvals(matrix)
    # Calculate frequencies from the eigenvalues
    frequencies = frequency_calculation(lambdas,temp)
    # Sort frequencies lowest to highest
    frequencies = nmp.sort(frequencies)
    kT = UAC.get_KT2J(temp)
    exponent = UAC.PLANCK_CONST*frequencies/kT
    power_positive = nmp.power(np.e,exponent)
    power_negative = nmp.power (np.e,-exponent)
    S_components = exponent/(power_positive-1)-nmp.ln(1-power_negative) 
    S_components = S_components*UAC.GAS_CONST #multiply by R - get entropy in J mol^{-1} K^{-1}
    if matrix_type == 'force': #force covariance matrix 
        if level == 'polymer': #polymer level - we take all frequencies into account
            S_vib_total = sum(S_components) # 3x3 force covariance matrix => 3 eigenvalues
        else:
            S_vib_total = sum(S_components[6:])  #we discard the 6 lowest frequencies to discard translation and rotation at the upper level
    else: #torque covariance matrix - we always take all values into account
        S_vib_total = sum(S_components)
    return S_vib_total
#END vibrational_entropy        

def conformational_entropy(dihedrals, numFrames, level):
    
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

    diheds_in_unit=BinaryTree() #we have one tree for the whole unit
    for dihedral in dihedrals: #we go through dihedrals
        dih_node= TreeNode (None, None, dihedral) 
        diheds_in_unit.add_node(dih_node) #we add the dihedrals to the tree
    
    newEntity = CONF.ConformationEntity(arg_order = len(diheds_in_unit),arg_numFrames = numFrames) #we initialize a string array that stores the state as a distinct string for each frame- made from a coalesced character cast of numeric arrays
    DecimalReprArray = []
    
    #we go through the dihedrals and find the corresponding state vectors
    for i, Dihedral in enumerate (diheds_in_unit.list_in_order()):
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
            
    return S_conf_total
#END conformational_entropy

def orientational_entropy(neighbours_dict): 
    """
    Function to calculate orientational entropies from eq. (10) in J. Higham, S.-Y. Chou, F. Gräter and R. H. Henchman, 
    Molecular Physics, 2018, 116,3 1965–1976. Number of orientations, Ω, is calculated using eq. (8) in  J. Higham, 
    S.-Y. Chou, F. Gräter and R. H. Henchman,  Molecular Physics, 2018, 116,3 1965–1976. σ is assumed to be 1 for the molecules we're concerned with and hence, 
    max {1, (Nc^3*π)^(1/2)} will always be (Nc^3*π)^(1/2). 
    TODO future release - function for determing symmetry and symmetry numbers maybe?
    Input
    -----
      neighbours_dict :  dictionary - dictionary of neighbours for the molecule - should contain the type of neighbour molecule and the number of neighbour
                                molecules of that species
    Returns
    -------
       S_or_total : float - orientational entropy
    """
    S_or_total=0 
    for neighbour in neighbours_dict: #we are going through neighbours
        if molecule in [] : #water molecules - call POSEIDON functions          
        
        else:
            omega= np.sqrt((neighbours_dict[molecule]**3)*math.pi) #the bound ligand is always going to be a neighbour
            S_or_component =math.log(omega) #orientational entropy arising from each neighbouring species - we know the species is going to be a neighbour
            S_or_component*=UAC.GAS_CONST
        S_or_total+=S_or_component
    #TODO for future releases -  implement a case for molecules with hydrogen bonds but to a lesser extent than water
    return S_or_total     
   



        
    

        
