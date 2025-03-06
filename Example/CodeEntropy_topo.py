# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:13:07 2022

@author: bmm66251
"""
import os
import sys
from datetime import datetime

import MDAnalysis as mda

from CodeEntropy.ClassCollection import DataContainer as DC
from CodeEntropy.FunctionCollection import EntropyFunctions

if __name__ == "__main__":
    startTime = datetime.now()
    ############## REPLACE INPUTS ##############
    wd = os.path.dirname(os.path.abspath(__file__))
    tprfile = os.path.join(wd, "data/1AKI_ws.tpr")
    trrfile = os.path.join(wd, "data/1AKI_ws.trr")
    outfile = "topo.out"
    tScale = 1
    fScale = 1
    temper = 300  # K

    u = mda.Universe(tprfile, trrfile)
    dataContainer = DC.DataContainer(u)

    result_entropy0_SC = EntropyFunctions.compute_topographical_entropy0_SC(
        arg_hostDataContainer=dataContainer,
        arg_selector="all",
        arg_outFile=outfile,
        arg_verbose=5,
    )

    print(f"result_entropy0_SC = {result_entropy0_SC}")

    result_entropy0_BB = EntropyFunctions.compute_topographical_entropy0_BB(
        arg_hostDataContainer=dataContainer,
        arg_selector="all",
        arg_outFile=outfile,
        arg_verbose=5,
    )

    print(f"result_entropy0_BB = {result_entropy0_BB}")

    result_entropy1_SC = EntropyFunctions.compute_topographical_entropy1_SC(
        arg_hostDataContainer=dataContainer,
        arg_selector="all",
        arg_outFile=outfile,
        arg_verbose=5,
    )

    print(f"result_entropy1_SC= {result_entropy1_SC}")

    result_entropy1_BB = EntropyFunctions.compute_topographical_entropy1_BB(
        arg_hostDataContainer=dataContainer,
        arg_selector="all",
        arg_outFile=outfile,
        arg_verbose=5,
    )
    print(f"result_entropy1_BB= {result_entropy1_BB}")
    # #work in progress
    # result_entropy4 = EntropyFunctions.compute_topographical_entropy_method4(
    #     arg_hostDataContainer = dataContainer,
    #     arg_selector = "all",
    #     arg_outFile = outfile1_SC,
    #     arg_verbose = 5
    # )

    # print(f"result_entropy4 = {result_entropy4}")

    result_entropyAEM = EntropyFunctions.compute_topographical_entropy_AEM(
        arg_hostDataContainer=dataContainer,
        arg_selector="all",
        arg_outFile=outfile,
        arg_verbose=5,
    )

    print(f"result_entropyAEM = {result_entropyAEM}")

    # # Demanding computation if large amount of Dihedral
    # result_entropy3 = EntropyFunctions.compute_topographical_entropy_method3(
    #     arg_hostDataContainer = dataContainer,
    #     arg_selector = "all",
    #     arg_outFile = outfile,
    #     arg_verbose = 5
    # )
    # print(f"result_entropy3 = {result_entropy3}")

    print(datetime.now() - startTime)
