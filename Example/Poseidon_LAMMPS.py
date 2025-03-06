import datetime as dt
import os
import sys

import MDAnalysis as mda
import pandas as pd
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader

from CodeEntropy.ClassCollection.PoseidonClass import Poseidon


def main():
    start = dt.datetime.now()
    # u2 = load_data() # this should drop all the intermediate values loaded which might benefit memory
    # select = u2.select_atoms("all")
    # select.write('data.trr', frames=u2.trajectory)
    wd = os.path.dirname(os.path.abspath(__file__))
    # loading files
    topo_file = os.path.join(wd, "data/molecules.prmtop")
    traj_file = os.path.join(wd, "data/data.trr")
    u = mda.Universe(topo_file, traj_file)
    load_data_time = dt.datetime.now()
    print(
        f"finished loading data: this step = {load_data_time - start}; total ={load_data_time - start}"
    )
    poseidon_object = Poseidon(container=u, start=0, end=20)
    populate_object_time = dt.datetime.now()
    print(
        f"finished populate object: this step = {populate_object_time - load_data_time}; total ={populate_object_time - start}"
    )
    analysis_time = dt.datetime.now()
    print(
        f"finished analysis: this step = {analysis_time - populate_object_time}; total ={analysis_time - start}"
    )
    result = poseidon_object.run_analysis(
        level_list=["moleculeLevel"], verbose=False, forceUnits="Kcal"
    )
    print(result.keys())
    print(result)


main()
