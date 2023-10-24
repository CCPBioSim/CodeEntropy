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
    
def vibrational_entropies(FTmatrix, temp,level): #could also separate the terms and use the separate force/torque covariance matrices
    '''
    
    vibrational entropy calculated from eq. (4) in Higham, S.-Y. Chou, F. Gräter and 
    R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976
    
    '''
    lambdas=la.eigvals(FTmatrix)
    frequencies=frequency_calculation(lambdas,temp)
    frequencies=frequencies.sort()
    kT=UAC.get_KT2J(temp)
    exponent=UAC.PLANCK_CONST*frequencies/kT
    power_positive= np.power(np.e,exponent)
    power_negative=np.power (np.e,-exponent)
    S_components=exponent/(power_positive-1)-np.ln(1-power_negative)
    S_components=S_components*UAC.GAS_CONST
    if level =='WM':
        S=sum(S_components) # 6x6 force-torque covariance matrix => 6 eigenvalues
    elif level == 'RES': #should expect 6M x 6M force torque covariance matrix, where M= no of residues 
         S=sum(S_components[6:])   
    elif level == 'UA': #6N x 6N force torque covariance matrix => 6N eigenvalues of which we take the largest 6N-6 frequencies
        S=sum(S_components[6:])
        
    

        
