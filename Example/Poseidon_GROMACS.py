########################################################################################
# !!!!!!!!!!!The 1AKI_prod.trr trajectory is too big and can't be included in the repo
########################################################################################

import os

import MDAnalysis as mda

from CodeEntropy.ClassCollection.PoseidonClass import (  # Poseidon,; Poseidon_mp,
    Poseidon_dask,
)

# import pandas as pd
# from MDAnalysis.analysis.base import AnalysisFromFunction
# from MDAnalysis.coordinates.memory import MemoryReader


# import sys


def main():
    # hard coded for now
    wd = os.path.dirname(os.path.abspath(__file__))
    # loading files
    topo_file = os.path.join(wd, "data/poseidon_example.prmtop")
    traj_file = os.path.join(wd, "data/poseidon_example.trr")
    # loading data into individual universe
    main = mda.Universe(topo_file, traj_file)
    poseidon_object = Poseidon_dask(container=main, start=0, end=10, verbose=True)
    poseidon_object.run_analysis(
        level_list=[
            "moleculeLevel",
            "residLevel_resname",
            "atomLevel",
            "soluteContacts",
        ],
        verbose=False,
    )


if __name__ == "__main__":
    main()
