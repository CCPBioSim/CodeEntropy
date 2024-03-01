from ast import arg
import sys,os
import numpy as np
from numpy import linalg as la
np.set_printoptions (threshold=sys.maxsize)
#from CodeEntropy import CustomFunctions as CF
#from CodeEntropy import GeometricFunctions as GF
#from CodeEntropy import UnitsAndConversions as UAC
#from CodeEntropy import LevelFunctions as LF
#from CodeEntropy import Utils
#from CodeEntropy import Writer
import multiprocessing as mp
from functools import partial
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import matplotlib.pyplot as plt
import math

def get_dihedrals(arg_hostDataContainer, level):
    """
    Define the set of dihedrals for use in the conformational entropy function.
    If residue level, the dihedrals are defined from the atoms (4 bonded atoms for 1 dihedral).
    If polymer level, use the bonds between residues to cast dihedrals.
    Note: not using improper dihedrals only ones with 4 atoms/residues in a linear arrangement.

    Input
    -----
    arg_hostDataContainer : system information
    level : level of the hierarchy (should be residue or polymer)

    Output
    ------
    dihedrals : set of dihedrals
    """
    # Start with empty array
    dihedrals = []

    # if residue level, read dihedrals from MDAnalysis universe
    if level == "united_atom":
        dihedrals.append(arg_hostDataContainer.dihedrals)

    # if polymer level, looking for dihedrals involving residues
    if level == "residue":
        num_residues = len(arg_hostDataContainer.residues)
        if num_residues < 4:
            print("no residue level dihedrals")

        else:
        # find bonds between residues N-3:N-2 and N-1:N
            for residue in range(3, num_residues):
                # Using MDAnalysis selection, assuming only one covalent bond between neighbouring residues
                # TODO test selection syntax
                # TODO not written for branched polymers
                atom1 = arg_hostDataContainer.select(f"residue {residue}-3 bonded residue {residue}-2" )
                atom2 = arg_hostDataContainer.select(f"residue {residue}-2 bonded residue {residue}-3" )
                atom3 = arg_hostDataContainer.select(f"residue {residue}-1 bonded residue {residue}" )
                atom4 = arg_hostDataContainer.select(f"residue {residue} bonded residue {residue}-1" )
                atom_group = atom1 + atom2 + atom3 + atom4
                dihedrals.append(atom_group.dihedral)

    return dihedrals

def get_neighbours(molecule_i, reduced_atom):
  """
  Molecules in the coordination shell are obtained using the relative angular distance (RAD) algorithm, as presented in
  J.Higham and R. H. Henchman, J. Chem. Phys., 2016, 145, 084108.
  Input
  -----
    molecule: DataContainer type with the information on the molecule of interest for which are finding the neighbours
    reduced_atom: MDAnalysis universe object containing the system of interest
  Returns
  -------
    neighbours_dict :  dictionary - dictionary of neighbours for the molecule - should contain the type of neighbour molecule and the number of neighbour
                                    molecules of that species
    neighbours_array: array containing all neighbouring molecules
  """
  number_molecules = len(reduced_atom.atoms.fragments)
  neighbours_dict = {}
  neighbours_array=[]
  for j in range(0,number_molecules): #go through molecules to see if they're in the coordination shell of the molecule of interest
          molecule_j=reduced_atom.atoms.fragments[j]
          if molecule_i!=molecule_j: #to avoid adding a molecule as its own neighbour 
              if len(neighbours_array)>0: #we have already found a neighbour 
                  for molecule_k in neighbours_array: #check if every closer molecule that is already in the neighbour list is unblocked
                      r_ij = mda.analysis.distances.dist (molecule_i, molecule_j) #distance between the molecule of interest and the molecule that we check if it's a neighbour
                      r_ik = mda.analysis.distances.dist (molecule_i, molecule_k) #distance between the molecule of interest and the neighbours already identified
                      r_jk = mda.analysis.distances.dist (molecule_j, molecule_k)
                      cos_theta_jik = (r_ij**2 + r_ik**2 - r_jk**2)/(2*r_ij*r_ik) #cosine of the angle subintended at i by j and k
                      if ((r_ij)**(-2) < (r_ik)**(-2)*cos_theta_jik):
                          break #j is not a neighbour as it is blocked by k
                      else:
                          neighbours_array.append(molecule_j.atoms.resids[0])
                          if molecule_j.smth in neighbours_dict.keys():
                              neighbours_dict[molecule_j.atoms.resnames[0]] +=1
                          else:
                              neighbours_dict[molecule_j.atoms.resnames[0]] =1
              else: #first neighbour
                  #r_ij = mda.analysis.distances.dist(molecule_i, molecule_j)
                  #print(r_ij)
                  neighbours_array.append(molecule_j.atoms.resids[0]) 
                  neighbours_dict[molecule_j.atoms.resnames[0]] =1
  return neighbours_dict, neighbours_array
                           