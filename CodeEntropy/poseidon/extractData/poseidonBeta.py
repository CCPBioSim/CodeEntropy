import os
import sys
import argparse
from pathlib import Path
from sys import argv
from glob import glob
import operator
import logging

import extractData

from datetime import datetime

from objectAnalysis import analysis

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader



def main():

    # try:
    #     usage = 'runPoseidon_beta.py [-h]'
    #     parser = argparse.ArgumentParser(description='Program for reading '\
    #             'in Molecular Dynamics Simulation files for: '\
    #             'Prediction Of a System\'s Entropy Including a '\
    #             'Determination Of its Nature - POSEIDON beta v2', usage=usage, 
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #     group = parser.add_argument_group('Options')
    #     group = parser.add_argument('-s', '--start', action='store', 
    #             default='1', help='starting frame number')
    #     group = parser.add_argument('-e', '--end', action='store', 
    #             default='1', help='end frame number')
    #     group = parser.add_argument('-dt', '--step', action='store', 
    #             default='1', help='steps between frames')
    #     group = parser.add_argument('-l', '--lammps', nargs='+', 
    #             metavar='file', help='list of lammps input files,'
    #             ' up to 5 files: amber_top, amber_crd, lammps_traj, '
    #             'lammps_force and lammps_energy files.')
    #     group = parser.add_argument('-a', '--amber', nargs='+', 
    #             metavar='file', help='list of amber input files,'
    #             ' up to 3 files: amber_top, amber_crd, amber_forces')
    #     group = parser.add_argument('-g', '--gromacs', nargs='+', 
    #             metavar='file', help='list of gromacs input files,'
    #             ' up to 2 files: gro_tpr, gro_trr')
    #     group = parser.add_argument('-p', '--pdb', nargs='+', 
    #             metavar='file', help='pdb file with multiple frames'
    #             ' up to 1 file: frames separated by ENDMDL')
    #     group = parser.add_argument('-pn', '--pureAtomNum', action='store', 
    #             default='1', help='reference molecule resid for pure liquid')
    #     group = parser.add_argument('-cs', '--cutShell', action='store', 
    #             default=None, help='include cutoff shell analysis, '\
    #             'add cutoff distance in angstrom')
    #     group = parser.add_argument('-ex', '--excludedResnames', 
    #             action='store', nargs='+',
    #             default=None, help='exclude a list of molecule names '\
    #             'from nearest non-like analysis')
    #     group = parser.add_argument('-obj', '--perFrameObj', 
    #             action='store_true', 
    #             help='output object every frame, use this for large systems')
    #     group = parser.add_argument('-wat', '--water', action='store', 
    #             default='WAT', help='resname for water molecules')
    #     group = parser.add_argument('-v', '--verbose', 
    #             action='store_true', 
    #             help='print out progress of each analysis step')

    #     op = parser.parse_args()
    # except argparse.ArgumentError:
    #     logging.error('Command line arguments are ill-defined, '
    #     'please check the arguments.')
    #     raise
    #     sys.exit(1)

    main = mda.Universe("molecules.prmtop", "Trajectory_npt_1.data.gz", atom_style='id type x y z' ,format="LAMMPSDUMP")
    force = mda.Universe("molecules.prmtop", "Forces_npt_1.data", atom_style='id type x y z' ,format="LAMMPSDUMP")
    energy = mda.Universe("molecules.prmtop", "Energy_npt_1.data", atom_style='id type x y z' ,format="LAMMPSDUMP")
    select_atom = main.select_atoms('all')
    select_atom_force = force.select_atoms('all')
    select_atom_energy = energy.select_atoms('all')
    u2 = mda.Merge(select_atom)
    coordinates = AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom).run().results['timeseries']
    forces = AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom_force).run().results['timeseries']
    energy = AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom_energy).run().results['timeseries']
    dimensions = AnalysisFromFunction(lambda ag: ag.dimensions.copy(), select_atom).run().results['timeseries']
    u2.load_new(coordinates, format=MemoryReader, forces=forces, velocities=energy, dimensions=dimensions)

    allMoleculeList = extractData.run(start=1, end=20, step=1, container=u2)
    analysis.run(allMoleculeList)

if __name__ == '__main__':
    main()
