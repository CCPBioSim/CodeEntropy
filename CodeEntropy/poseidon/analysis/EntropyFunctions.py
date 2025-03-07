#!/usr/bin/env python3

# import os
import sys

import numpy as np
from numpy import linalg as la

# from ast import arg


np.set_printoptions(threshold=sys.maxsize)
# from CodeEntropy.ClassCollection import BeadClasses as BC
# from CodeEntropy.ClassCollection import ConformationEntity as CONF
# from CodeEntropy.ClassCollection import ModeClasses
# from CodeEntropy.ClassCollection import CustomDataTypes
# from CodeEntropy.FunctionCollection import CustomFunctions as CF
# from CodeEntropy.FunctionCollection import GeometricFunctions as GF
# from CodeEntropy.FunctionCollection import UnitsAndConversions as UAC
# from CodeEntropy.FunctionCollection import Utils
# from CodeEntropy.IO import Writer
# import multiprocessing as mp
# from functools import partial

# import MDAnalysis as mda
# import matplotlib.pyplot as plt
# import pandas as pd

# def frequencies(lambdas): - to add!!!


def vibrational_entropies(FTmatrix):
    lambdas = la.eig(FTmatrix)
    exponent = 1
    exp_positive = np.power(np.e, exponent)
    exp_negative = np.power(np.e, -exponent)
    print(lambdas)


test_matrix = [
    [7, 0, 1, 1, 2, 2],
    [5, 5, 0, 0, 4, 5],
    [0, 1, 0, 0, 5, 4],
    [2, 0, 0, 0, 0, 2],
    [
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    [0, 0, 0, 0, 0, 1],
]
vibrational_entropies(test_matrix)
