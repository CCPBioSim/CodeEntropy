# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:13:07 2022

@author: bmm66251
"""
import os, sys
import MDAnalysis as mda
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.FunctionCollection import EntropyFunctions
from CodeEntropy.IO import InputParser, Writer 
from CodeEntropy.Reader import GromacsReader

if __name__ == "__main__":
    ############## REPLACE INPUTS ##############
    wd = os.path.dirname(os.path.abspath(__file__))
    tprfile = os.path.join(wd,"data/md_A4_dna.tpr")
    trrfile = os.path.join(wd,"data/md_A4_dna_xf.trr")
    outfile = os.path.join(wd,"dna_mcc.out")
    outfile0_SC = os.path.join(wd,"dna_mcc_0_SC.out")
    outfile0_BB = os.path.join(wd,"dna_mcc_0_BB.out")
    tScale = 1
    fScale = 1
    temper = 300 #K
    
    mainMolecule, mainContainer = GromacsReader.read_gromacs_input(
        arg_tprFile = tprfile, 
        arg_trajFile = trrfile, 
        arg_outFile = outfile, 
        arg_beginTime = -1, 
        arg_endTime = -1, 
        arg_stride = 1, 
        arg_verbose = 5
    )    
    EntropyFunctions.compute_topographical_entropy0_SC(
        arg_baseMolecule = mainMolecule, 
        arg_hostDataContainer = mainContainer, 
        arg_outFile = outfile0_SC, 
        arg_verbose = 5
    )

    EntropyFunctions.compute_topographical_entropy0_BB(
        arg_baseMolecule = mainMolecule, 
        arg_hostDataContainer = mainContainer, 
        arg_outFile = outfile0_BB, 
        arg_verbose = 5
    ) 
