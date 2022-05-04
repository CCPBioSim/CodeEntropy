#!/usr/bin/env python

import logging
import sys
import math
import numpy as np
from numpy import linalg as LA
from collections import Counter
from collections import defaultdict
import operator
from datetime import datetime

import MDAnalysis
from MDAnalysis import *
from MDAnalysis.analysis import distances
from MDAnalysis.lib.formats.libmdaxdr import TRRFile

from CodeEntropy.poseidon.extractData.generalFunctions import *
from CodeEntropy.poseidon.extractData.mainClass import *


nested_dict = lambda: defaultdict(nested_dict) 
        #create nested dict in one go



def readInputs(lammps, amber, gromacs, pdb, verbosePrint):
    '''

    Use MDAnalysis to read in files from several MD simulation
    packages: LAMMPS, AMBER and GROMACS
    Additionally, .pdb files can be read in, but also needs a 
    LAMMPS topology file to read in force field paramters.

    '''

    topology, trajectory, force, energy, inputType = \
            None, None, None, None, None

    if lammps != None:
        inputType = 'lammps'
        verbosePrint('LAMMPS FILES:', lammps)
        try:
            topology = Universe(lammps[0], lammps[1], format='TRJ')
            verbosePrint('AMBER topology')
        except (ValueError, IOError, IndexError, EOFError):
            try:
                topology = Universe(lammps[0], format='DATA')
                verbosePrint ('LAMMPS topology')
            except (ValueError, IOError, IndexError, EOFError):
                try:
                    topology = Universe(lammps[0], format='PDB')
                    verbosePrint ('PDB topology')
                except (ValueError, IOError, IndexError):
                    logging.error('No AMBER topology or trajectory/'\
                            ' LAMMPS topology file given or PDB given.')


        try:
            trajectory = Universe(lammps[2], atom_style='id type x y z', 
                    format='LAMMPSDUMP')
            verbosePrint('LAMMPS trajectory')
        except IndexError:
            logging.error('No LAMMPS trajectory file given.')

        try:
            force = Universe(lammps[3], atom_style='id type x y z', 
                    format='LAMMPSDUMP')
            verbosePrint('LAMMPS force')
        except IndexError:
            logging.warning('No LAMMPS force file given.')

        try:
            energy = Universe(lammps[4], atom_style='id type x y z', 
                    format='LAMMPSDUMP') #new file type, where x=PE, y=KE
            verbosePrint('LAMMPS energy')
        except IndexError:
            logging.warning('No LAMMPS energy file given.')
       

    
    if amber != None:
        inputType = 'amber'
        verbosePrint('AMBER FILES:', amber)
        try:
            topology = Universe(amber[0], amber[1], format='TRJ')
            trajectory = Universe(amber[0], amber[1], format='TRJ')
            verbosePrint('AMBER topology/trajectory (1)')
        except (ValueError, IOError, IndexError):
            try: #try binary format read in
                topology = Universe(amber[0], amber[1], format='NCDF')
                trajectory = Universe(amber[0], amber[1], format='NCDF')
                verbosePrint('AMBER topology/trajectory (2)')
            except (ValueError, IOError, IndexError):
                logging.error('No AMBER input topology or '\
                        'trajectory files given.')

        try:
            force = Universe(amber[0], amber[2], format='TRJ')
            verbosePrint('AMBER force (1)')
        except (ValueError, IOError, IndexError):
            try: #try binary format read in
                force = Universe(amber[0], amber[2], format='NCDF')
                verbosePrint('AMBER force (2)')
            except (ValueError, IOError, IndexError):
                logging.warning('No AMBER input topology or force files.')



    if gromacs != None:
        inputType = 'gromacs'
        verbosePrint('GROMACS FILES:', gromacs)
        try:
            topology = Universe(gromacs[0], format='TPR')
            verbosePrint('GROMACS topology (1)')
        except (ValueError, IOError, IndexError, EOFError):
            try:
                topology = Universe(gromacs[0], format='PDB')
                verbosePrint('GROMACS topology (2)')
            except (ValueError, IOError, IndexError):
                logging.error('No GROMACS topology '\
                        'or PDB given.')


        try:
            trajectory = Universe(gromacs[1], format='TRR')
            force = gromacs[1]
            verbosePrint('GROMACS trr trajectory')
        except (ValueError, IOError, IndexError, EOFError):
            try:
                trajectory = Universe(gromacs[1], format='XTC')
                force = gromacs[1]
                verbosePrint('GROMACS traj trajectory')
            except (ValueError, IOError, IndexError):
                logging.error('No gromacs trajectory file given.')




    if pdb != None:
        inputType = 'pdb'
        verbosePrint('PDB FILES:', pdb)
        try:
            topology = Universe(pdb[0], format='DATA')
            verbosePrint('LAMMPS topology')
        except (ValueError, IOError, IndexError):
            logging.warning('No LAMMPS topology for PDB coordinates (1).')

        try:
            topology = Universe(pdb[0], format='PDB')
            verbosePrint('PDB topology')
        except (ValueError, IOError, IndexError):
            logging.warning('No LAMMPS topology for PDB coordinates (2).')

        try:
            trajectory = Universe(pdb[1], format='PDB') 
            verbosePrint('PDB trajectory')
        except (ValueError, IOError, IndexError):
            logging.error('No PDB coordinates input file.')


    return topology, trajectory, force, energy, inputType





def populateTopology(container, all_data, waterTuple):
    '''
    After reading in topologies using MDAnalysis,
    relavant information from topology files are saved into a global
    class object. Properties are read in differently depeding on what
    file types were read in.
    '''

    all_resids = []
    mol = None

    for tp in container.atoms:
        bonded_atom_nums = []
        for b in tp.bonds:
            for x in b:
                if x.id != tp.id:
                    bonded_atom_nums.append(int(x.id))
                else:
                    continue

        try:
            dihedral_list = []
            if tp.mass > 1.1: 
                for dih in tp.dihedrals:
                    diha_list = []
                    for di in dih:
                        if di.mass > 1.1:
                            diha_list.append(int(di.id))
                        else:
                            break

                    if len(diha_list) == 4:
                        dihedral_list.append(diha_list)
                    else:
                        continue
        except AttributeError:
            dihedral_list = []


        inf = atom_info(int(tp.id), tp.name, tp.mass, 
                tp.charge, int(tp.resid), 
                tp.resname, bonded_atom_nums, dihedral_list)

        all_data.append(inf)

    ### Get and populate UA and molecule level information
    molecule_dict = nested_dict()
    molecule_resids_dict = nested_dict()
    for x in range(0, len(all_data)):
        atom = all_data[x]
        print(f"x = {x}")
        if atom.mass > 1.1:
            heavy_bonded = []
            H_bonded = []
            for bonded_atom in atom.bonded_to_atom_num:
                print(f"bonded_atom = {bonded_atom}")
                bonded = all_data[bonded_atom-1]
                if bonded.mass > 1.1:
                    heavy_bonded.append(bonded)
                elif bonded.mass < 1.1:
                    H_bonded.append(bonded)
                else:
                    continue

            bonded_atoms_list = [atom] + heavy_bonded + H_bonded
            atom.bondedUA_H = [len(heavy_bonded), len(H_bonded)]
           
            if atom.resid not in molecule_dict:
                molecule_dict[atom.resid] = []
                molecule_resids_dict[atom.resid] = []
            if atom.atom_num not in molecule_dict[atom.resid]:
                molecule_dict[atom.resid].append(atom.atom_num)
            for bonded2 in heavy_bonded:
                if bonded2.resid not in molecule_resids_dict[atom.resid]:
                    molecule_resids_dict[atom.resid].append(bonded2.resid)
                else:
                    continue
            

    for x in range(0, len(all_data)):
        atom = all_data[x]
        if atom.mass > 1.1:
            atom.molecule_atomNums = sorted(molecule_dict[atom.resid])
            atom.bonded_to_resids = sorted(molecule_resids_dict[atom.resid])
        else:
            continue





def getCoordsForces(container, all_data, dimensions, 
        frame, startTime, verbosePrint):
    '''
    Read in coordinate and force trajectories and populate mainClass.
    '''


    t = container.trajectory[frame]
    dimensions = np.array(t.dimensions[0:3])
    print(t)

    for x in range(0, len(t)):
        crds = np.array(t[x])
        all_data[x].coords = crds
        frcs = np.array(t.forces[x])
        all_data[x].forces = frcs
 
    verbosePrint('COORDS-FORCES')
    verbosePrint(datetime.now() - startTime)
    sys.stdout.flush() 

    return all_data, dimensions




def populateEnergy(container, all_data, dimensions, frame, startTime, 
        verbosePrint):
    '''
    read in energies from lammps input file.
    '''

    for e in container.trajectory:
        if e.frame == frame:
            dimensions = np.array(e.dimensions[0:3])
            for x in range(0, len(e)):
                energy = np.array(e.velocities[x])
                all_data[x].PKenergy = [energy[0], energy[1]]
            break
        else:
            continue


    verbosePrint('ENERGIES')
    verbosePrint(datetime.now() - startTime)
    sys.stdout.flush() 





def UAEnergyGroup(all_data):
    '''
    For energy on each atom, group together for each UA
    and sum
    '''

    for x in range(0, len(all_data)):
        atom = all_data[x]
        if atom.mass > 1.1:
            heavy_bonded = []
            H_bonded = []
            for bonded_atom in atom.bonded_to_atom_num:
                bonded = all_data[bonded_atom-1]
                if bonded.mass > 1.1:
                    heavy_bonded.append(bonded)
                elif bonded.mass < 1.1:
                    H_bonded.append(bonded)
                else:
                    continue

            bonded_atoms_list = [atom] + heavy_bonded + H_bonded
            atom.bondedUA_H = [len(heavy_bonded), len(H_bonded)]
            UA_atoms_list = [atom] + H_bonded

            UA_PE_list = []
            UA_KE_list = []
            for A in UA_atoms_list:
                if A.PKenergy != None:
                    UA_PE_list.append(A.PKenergy[0])
                    UA_KE_list.append(A.PKenergy[1])
                else:
                    continue

            if len(UA_PE_list) != 0:
                UA_PE = round(sum(UA_PE_list), 3)
                UA_KE = round(sum(UA_KE_list), 3)
                atom.UA_PKenergy = [UA_PE, UA_KE]






def getDistArray(atom, all_data, traj, max_cutoff,
        dimensions, neighbour_coords, startTime, verbosePrint):
    '''
    Find the NN list of an atom
    Important to use coords directly from MDAnalysis to run NN calc
    '''

    atom_coords = traj[atom.atom_num-1]


    array1, array2 = \
            MDAnalysis.lib.distances.capped_distance(atom_coords, 
                    neighbour_coords, max_cutoff=max_cutoff, 
                    min_cutoff=None, box=traj.dimensions, 
                    method=None, return_distances=True)


    try:
        array1, array2 = zip(*sorted(zip(array2, array1), 
            key=lambda x: x[0]))

    except ValueError:
        logging.error('Bad Arrays for Coordinate/ Atom Number '\
                'Nearest Neighbour Assignments')

    atomNumList = []
    allAtomList = []
    for atoms, dist in zip(array2, array1):
        near = atoms[1]
        atom_num = all_data[near].atom_num
        atom_resid = all_data[near].resid
        atom_mass = all_data[near].mass
        allAtomList.append((near, dist))

        #atom_resid != all_data[x].resid removed for quartz
        #surface that is all the same resid.
        if atom_num != atom.atom_num \
                and atom_num not in \
                    atom.bonded_to_atom_num \
                and float(atom_mass) > 1.1:
            atomNumList.append((near, dist))
        else:
            continue

    atom.nearest_sorted_array = atomNumList
    atom.nearest_all_atom_array = allAtomList



