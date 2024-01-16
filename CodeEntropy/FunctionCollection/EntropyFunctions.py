#!/usr/bin/env python3

import numpy as nmp
import MDAnalysis as mds
from CodeEntropy.ClassCollection import BeadClasses as BC
from CodeEntropy.ClassCollection import ConformationEntity as CONF
from CodeEntropy.ClassCollection import ModeClasses
from CodeEntropy.ClassCollection import CustomDataTypes
from CodeEntropy.FunctionCollection import EntropyFunctions as EF
from CodeEntropy.FunctionCollection import CustomFunctions as CF
from CodeEntropy.FunctionCollection import GeometricFunctions as GF
from CodeEntropy.FunctionCollection import UnitsAndConversions as UAC
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.IO import Writer
from CodeEntropy.FunctionCollection import UnitsAndConversions as CONST

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
    pi=np.pi
    kT=UAC.get_KT2(temp)
    frequencies=1/(2*pi)*np.sqrt(lambdas/kT)
    return frequencies
    
def vibrational_entropies(matrix, matrix_type, temp,level): 
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
        

def conformational_entropies(dihedrals, level):
    
    """
    Function to calculate conformational entropies are calculated using eq. (7) in Higham, S.-Y. Chou, F. Gräter and 
    R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976/ eq. (4) in A. Chakravorty, J. Higham and R. H. Henchman, 
    J. Chem. Inf. Model., 2020, 60, 5540–5551.
    Uses the adaptive enumeration method (AEM).
            Input
    -----
    arg_hostDataContainer : MDAnalysis Universe object - system information
    level : string - level of the hierarchy - should be "polymer" or "residue" here
    Returns
    -------
       S_conf_total : float - conformational entropy
    """
    
    S_conf_total=0     

    
    if level == 'polymer': #we don't need to go through residues
        diheds_in_polymer=BinaryTree() #we have one tree for the whole polymer
        for dihedral in dihedrals: #we go through dihedrals at polymer level
            dih_node= TreeNode (None, None, dihedral) 
            diheds_in_polymer.add_node(dih_node) #we add the dihedrals to the tree
        newEntity = CONF.ConformationEntity(arg_order, arg_numFrames, kwargs)ConformationEntity(arg_order = len(diheds_in_polymer),arg_numFrames = numFrames) #change based on class collection
        #we initialize a string array that stores the state as a distinct string for each frame- made from a coalesced character cast of numeric arrays
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
            
    if level == 'residue':
        for residue in arg_hostDataContainer.universe.residues: #we are going through the residues
            diheds_in_residue=BinaryTree() #we build a tree for each residue
            for dihedral in dihedrals: #we go through the dihedrals
                if dihedral.is_from_same_residue() == residue.resid and dihedral.is_heavy(): #we check if dihedral is from that residue - if yes, add to tree
                    dih_node= TreeNode (None, None, dihedral) 
                    diheds_in_residue.add_node() #we add the dihedrals in that residue to the tree
            newEntity = CONF.ConformationEntity(arg_order, arg_numFrames, kwargs)ConformationEntity(arg_order = len(diheds_in_residue),arg_numFrames = numFrames) #change based on ClassCollection
            #we initialize a string array that stores the state as a distinct string for each frame- made from a coalesced character cast of numeric arrays
            DecimalReprArray = []
            for i, Dihedral in enumerate (diheds_in_residue.list_in_order()):
                stateTS = iDih.get_state_ts()
                newEntity.timeSeries[i,:] = stateTS
            for iFrame in range (numFrames):
                DecimalReprArray.append (Utils.coalesce_numeric_array(newEntity.timrSeries[:,iFrame]))
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
   
#TODO orientational entropy function
    
'''
def orientational_entropy():

    omega Ω calculated from eq. (8) in Higham, S.-Y. Chou, F. Gräter and R. H. Henchman, Molecular Physics, 2018, 116,3 1965–1976
    σ = 2 for water + divide by 4 OR 1 for other molecules = ligands we re concerned with
    -might need to add another case for molecules with high symmetry about 1 axis - e.g. methanol, ethane
    orientational entropies are calculated using eq. (10) in Higham, S.-Y. Chou, F. Gräter and 
    R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976 for molecules other than water

    # assuming we identify neighbours before and could have a dictionary of neighbours or similar structure - e.g. neighbours= {'SOL': x, 'LIG': y} - identified using RAD for each orientation
    S_or_total=0
    neighbours_dict= RAD() #could do a separate function for identifying neighbours 
    for molecule in neighbours_dict: #we are going through neighbo
    #get the probabilities from somewhere - TBD
        if molecule in [] : #water flag'
            
             omega= np.sqrt((neighbours_dict[molecule]**3)*math.pi)*0.125 - #multiply by (HBav)/σ - HBav =0.25 as each water molecule is equally likely to donate and accept H bonds  
            
        elif:  #things that are hydrogen bonds and aren't water
            
        
        else:
            omega= np.sqrt((neighbours_dict[molecule]**3)*math.pi) #always going to be larger than 1 as σ = 1
        S_molecule = probabilities_dict[molecule]* np.log(omega)*UAC.GAS_CONST
        S_or_total+=S_molecule 
    return S_or_total     
'''
   



        
    

        
