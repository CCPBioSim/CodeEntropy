import os

# import sys
from datetime import datetime

import MDAnalysis as mda
import numpy as np
import pandas as pd

from CodeEntropy.ClassCollection import DataContainer as DC
from CodeEntropy.FunctionCollection import EntropyFunctions as EF
from CodeEntropy.IO import MDAUniverseHelper as MDAHelper

if __name__ == "__main__":
    # REPLACE INPUTS #
    startTime = datetime.now()
    data_dir = os.path.dirname(os.path.abspath(__file__))
    tprfile = os.path.join(data_dir, "data/1AKI_prod_60.tpr")
    trrfile = os.path.join(data_dir, "data/1AKI_prod_60.trr")
    outfile = None
    tScale = 1.0
    fScale = 1.0
    temper = 300.0  # K
    u = mda.Universe(tprfile, trrfile)
    topo_outfile = "topo_out.out"
    selection_string = "protein"
    start = 3
    end = 67
    step = 2
    thread = 8
    axis_list = ["C", "CA", "N"]
    if start is None:
        start = 0
    if end is None:
        end = len(u.trajectory)
    results_df = pd.DataFrame(columns=["Method and Level", "Type", "result"])
    reduced_frame = MDAHelper.new_U_select_frame(u, start, end, step)
    reduced_frame_name = f"{(end - start)//step}_frame_dump"
    reduced_frame_filename = MDAHelper.write_universe(reduced_frame, reduced_frame_name)
    reduced_atom = MDAHelper.new_U_select_atom(reduced_frame, selection_string)
    reduced_atom_name = f"{(end - start)//step}_frame_dump_strip_solvent"
    reduced_atom_filename = MDAHelper.write_universe(reduced_atom, reduced_atom_name)
    dataContainer = DC.DataContainer(reduced_atom)

    wm_entropyFF, wm_entropyTT = EF.compute_entropy_whole_molecule_level(
        arg_hostDataContainer=dataContainer,
        arg_outFile=outfile,
        arg_selector="protein",
        arg_moutFile="WholeMolecule_matrix.out",
        arg_nmdFile="WholeMolecule_mode_spectra.out",
        arg_fScale=fScale,
        arg_tScale=tScale,
        arg_temper=temper,
        arg_verbose=5,
    )

    print(f"wm_entropyFF = {wm_entropyFF}")
    print(f"wm_entropyTT = {wm_entropyTT}")
    newRow = pd.DataFrame(
        {
            "Method and Level": ["Whole molecule", "Whole molecule"],
            "Type": ["FF Entropy (J/mol/K)", "TT Entropy (J/mol/K)"],
            "result": [wm_entropyFF, wm_entropyTT],
        }
    )
    results_df = pd.concat([results_df, newRow], ignore_index=True)
    res_entropyFF, res_entropyTT = EF.compute_entropy_residue_level(
        arg_hostDataContainer=dataContainer,
        arg_outFile=outfile,
        arg_selector="all",
        arg_moutFile="ResidueLevel_matrix.out",
        arg_nmdFile="ResidueLevel_mode_spectra.out",
        arg_fScale=fScale,
        arg_tScale=tScale,
        arg_temper=temper,
        arg_verbose=5,
        arg_axis_list=axis_list,
    )
    print(res_entropyFF)
    print(res_entropyTT)
    newRow = pd.DataFrame(
        {
            "Method and Level": ["Residue", "Residue"],
            "Type": ["FF Entropy (J/mol/K)", "TT Entropy (J/mol/K)"],
            "result": [res_entropyFF, res_entropyTT],
        }
    )
    results_df = pd.concat([results_df, newRow], ignore_index=True)

    UA_entropyFF, UA_entropyTT, res_df = EF.compute_entropy_UA_level(
        arg_hostDataContainer=dataContainer,
        arg_outFile=outfile,
        arg_selector="all",
        arg_moutFile="AtomLevel_matrix.out",
        arg_nmdFile="AtomLevel_mode_spectra.out",
        arg_fScale=fScale,
        arg_tScale=tScale,
        arg_temper=temper,
        arg_verbose=1,
        arg_axis_list=axis_list,
        arg_csv_out="AtomLevel_bead_entropy.csv",
    )
    print(f"UA_entropyFF = {UA_entropyFF}")
    print(f"UA_entropyTT = {UA_entropyTT}")
    newRow = pd.DataFrame(
        {
            "Method and Level": ["United atom", "United atom"],
            "Type": ["FF Entropy (J/mol/K)", "TT Entropy (J/mol/K)"],
            "result": [UA_entropyFF, UA_entropyTT],
        }
    )
    results_df = pd.concat([results_df, newRow], ignore_index=True)

    UA_entropyFF, UA_entropyTT, res_df = EF.compute_entropy_UA_level_multiprocess(
        arg_hostDataContainer=dataContainer,
        arg_outFile=outfile,
        arg_selector="all",
        arg_moutFile="AtomLevel_matrix.out",
        arg_nmdFile="AtomLevel_mode_spectra.out",
        arg_fScale=fScale,
        arg_tScale=tScale,
        arg_temper=temper,
        arg_verbose=1,
        arg_csv_out="AtomLevel_bead_entropy.csv",
        arg_axis_list=axis_list,
        arg_thread=thread,
    )
    print(f"UA_entropyFF = {UA_entropyFF}")
    print(f"UA_entropyTT = {UA_entropyTT}")
    newRow = pd.DataFrame(
        {
            "Method and Level": ["United atom", "United atom"],
            "Type": ["FF Entropy (J/mol/K)", "TT Entropy (J/mol/K)"],
            "result": [UA_entropyFF, UA_entropyTT],
        }
    )
    results_df = pd.concat([results_df, newRow], ignore_index=True)

    result_entropy0_SC = EF.compute_topographical_entropy0_SC(
        arg_hostDataContainer=dataContainer,
        arg_selector="all",
        arg_outFile=topo_outfile,
        arg_verbose=5,
    )

    print(f"result_entropy0_SC = {result_entropy0_SC}")

    result_entropy0_BB = EF.compute_topographical_entropy0_BB(
        arg_hostDataContainer=dataContainer,
        arg_selector="all",
        arg_outFile=topo_outfile,
        arg_verbose=5,
    )

    print(f"result_entropy0_BB = {result_entropy0_BB}")

    newRow = pd.DataFrame(
        {
            "Method and Level": ["Topographical pLogp", "Topographical pLogp"],
            "Type": ["Total SC Topog. Entropy", "Total BB Topog. Entropy"],
            "result": [result_entropy0_SC, result_entropy0_BB],
        }
    )
    results_df = pd.concat([results_df, newRow], ignore_index=True)

    result_entropy1_SC = EF.compute_topographical_entropy1_SC(
        arg_hostDataContainer=dataContainer,
        arg_selector="all",
        arg_outFile=topo_outfile,
        arg_verbose=5,
    )

    print(f"result_entropy1_SC= {result_entropy1_SC}")

    result_entropy1_BB = EF.compute_topographical_entropy1_BB(
        arg_hostDataContainer=dataContainer,
        arg_selector="all",
        arg_outFile=topo_outfile,
        arg_verbose=5,
    )
    print(f"result_entropy1_BB= {result_entropy1_BB}")

    newRow = pd.DataFrame(
        {
            "Method and Level": [
                "Topographical correlation/pLogp",
                "Topographical correlation/pLogp",
            ],
            "Type": [
                "Total SC Topog. Entropy (corr. pLogP)",
                "Total BB Topog. Entropy (corr. pLogP)",
            ],
            "result": [result_entropy0_SC, result_entropy0_BB],
        }
    )
    results_df = pd.concat([results_df, newRow], ignore_index=True)

    # #work in progress
    # result_entropy4 = EntropyFunctions.compute_topographical_entropy_method4(
    #     arg_hostDataContainer = dataContainer,
    #     arg_selector = "all",
    #     arg_outFile = outfile1_SC,
    #     arg_verbose = 5
    # )

    # print(f"result_entropy4 = {result_entropy4}")

    result_entropyAEM = EF.compute_topographical_entropy_AEM(
        arg_hostDataContainer=dataContainer,
        arg_selector="all",
        arg_outFile=topo_outfile,
        arg_verbose=5,
    )
    newRow = pd.DataFrame(
        {
            "Method and Level": [
                "Topographical AEM",
            ],
            "Type": [
                "Total Topog. Entropy (AEM)",
            ],
            "result": [
                result_entropyAEM,
            ],
        }
    )
    results_df = pd.concat([results_df, newRow], ignore_index=True)

    print(f"result_entropyAEM = {result_entropyAEM}")

    print(f"total time {datetime.now() - startTime}")
    results_df = results_df.replace(np.nan, "nan")
    results_df.to_csv("molecule_entropy_result.csv", index=False)
