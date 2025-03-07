import os

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader

# import sys


# MDanalysis only supports reading force data from GROMACS TRR and AMBER NETCDF format
# For other data such as LAMMPS you will have to make a universe by loading the data
# into it. This is also useful for trimming existing trajectories to reduce the size
# of system for faster analysis. The key is to have a topology file that contains
# dihedral information and output the packaged position and force data as a trr file
# !!! you have to have a topology file that contains the dihedral information for
# MDAnalysis
def load_data():
    # This is to set the working directory
    wd = os.path.dirname(os.path.abspath(__file__))
    # This part is to set the path to the files
    topo_file = os.path.join(wd, "data/molecules.prmtop")
    traj_file = os.path.join(wd, "data/Trajectory_npt_1.data.gz")
    # remember to edit the format so that the header is "id mass x y z"
    # otherwise MDAnalysis won't load the data due to checks by LAMMPS parser
    force_file = os.path.join(wd, "data/Forces_npt_1.data")
    # loading data into individual universe
    main = mda.Universe(
        topo_file, traj_file, atom_style="id type x y z", format="LAMMPSDUMP"
    )
    force = mda.Universe(
        topo_file, force_file, atom_style="id type x y z", format="LAMMPSDUMP"
    )
    # selection for accessing values
    # the 'all' can be replaced by other selection string for
    select_atom = main.select_atoms("all")
    select_atom_force = force.select_atoms("all")
    # loading values from universe
    # this is done by generating a tuple from AnalysisFromFunction to traverse through
    # the entire data and loading the selected data into a tuple
    coordinates = (
        AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom)
        .run()
        .results["timeseries"]
    )
    forces = (
        AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom_force)
        .run()
        .results["timeseries"]
    )
    #  dimension is also required for poseidon analysis
    dimensions = (
        AnalysisFromFunction(lambda ag: ag.dimensions.copy(), select_atom)
        .run()
        .results["timeseries"]
    )
    # create a new universe
    u2 = mda.Merge(select_atom)
    # loading trajectory data using MemoryReader from tuples,
    # the system is not in memory
    u2.load_new(coordinates, format=MemoryReader, forces=forces, dimensions=dimensions)
    return u2


def main():
    u2 = load_data()
    # you can analyse the system or save trajectories for further analysis
    # selection
    select = u2.select_atoms("all")
    #  you can also slice the trajectories
    select.write("data.trr", frames=u2.trajectory[::2])
    #  reading data
    u_new = mda.Universe("molecules.prmtop", "data.trr")


if __name__ == "__main__":
    main()
