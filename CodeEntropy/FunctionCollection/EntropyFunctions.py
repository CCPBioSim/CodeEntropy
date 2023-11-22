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
    S_components=S_components*UAC.GAS_CONST
    if matrix_type='FF':
        if level =='P': #polymer
            S=sum(S_components) # 3x3 force covariance matrix => 6 eigenvalues
        else:
            S=sum(S_components[6:]  #for the 'M' and 'UA' levels we discard the 6 lowest frequencies
    else: #for the torque covariance matrix, we take all values into account
        S=sum(S_components)
        
def conformational_entropies        
        
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


        
    

        
