from ast import arg
import sys,os
import numpy as np
from numpy import linalg as la
np.set_printoptions (threshold=sys.maxsize)
from CodeEntropy import CustomFunctions as CF
from CodeEntropy import GeometricFunctions as GF
from CodeEntropy import UnitsAndConversions as UAC
from CodeEntropy import LevelFunctions as LF
from CodeEntropy import Utils
from CodeEntropy import Writer
import multiprocessing as mp
from functools import partial
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt
import math

def get_neighbours(molecule, reduced_atom): #we use the relative angular distance (RAD) algorithm
    

    