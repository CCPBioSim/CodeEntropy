import os
import sys
from datetime import datetime

import MDAnalysis as mda
import numpy as np
import pandas as pd

from CodeEntropy.ClassCollection import DataContainer as DC
from CodeEntropy.FunctionCollection import EntropyFunctions as EF
from CodeEntropy.IO import MDAUniverseHelper as MDAHelper

if __name__ == "__main__":
    ############## REPLACE INPUTS ##############
    startTime = datetime.now()
    data_dir = os.path.dirname(os.path.abspath(__file__))
    tprfile = os.path.join(data_dir, "data/md_A4_dna.tpr")
    trrfile = os.path.join(data_dir, "data/md_A4_dna_xf.trr")
    outfile = None
    tScale = 1.0
    fScale = 1.0
    temper = 300.0  # K
    u = mda.Universe(tprfile, trrfile)
    selection_string = "all"
    start = 3
    end = 40
    step = 1
    thread = 8
    axis_list = ["C5'", "C4'", "C3'"]
    if start == None:
        start = 0
    if end == None:
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
        arg_selector=selection_string,
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

    print(f"total time {datetime.now() - startTime}")
    results_df = results_df.replace(np.nan, "nan")
    results_df.to_csv(f"molecule_entropy_result.csv", index=False)
