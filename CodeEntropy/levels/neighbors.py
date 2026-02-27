"""Neighbours info for orientational entropy.

This module finds the average number of neighbours, symmetry numbers, and
and linearity for each group.
These are used downstream to compute the orientational entropy.
"""

import logging

import MDAnalysis as mda
import numpy as np
from rdkit import Chem

from CodeEntropy.levels import search

logger = logging.getLogger(__name__)


class Neighbors:
    """
    Class to find the neighbors and any related information needed for the
    calculation of orientational entropy.
    """

    def __init__(self):
        """
        Initializes the LevelManager with placeholders for level-related data,
        including translational and rotational axes, number of beads, and a
        general-purpose data container.
        """

        self._universe = None
        self._groups = None
        self._levels = None

    def get_neighbors(self, universe, levels, groups, search_type):
        """
        Find the neighbors relative to the central molecule.

        Args:
            universe: MDAnalysis universe object for the system
            groups: list of groups for averaging
            levels: list of levels for each molecule
            search_type: str, how to find neigbours

        Returns:
            average_number_neighbors (dict): average number of neighbors
                at each frame for each group
        """

        number_neighbors = {}
        average_number_neighbors = {}

        number_frames = len(universe.trajectory)
        search_object = mda.lib.NeighborSearch.AtomNeighborSearch(universe.atoms)

        for group_id in groups.keys():
            molecules = groups[group_id]
            highest_level = levels[molecules[0]][-1]

            for mol_id in molecules:
                for timestep in range(number_frames):
                    # This is to get MDAnalysis to get the information from the
                    # correct frame of the trajectory
                    universe.trajectory[timestep]

                    if search_type == "RAD":
                        # Use the relative angular distance method to find neighbors
                        neighbors = search.get_RAD_neighbors(universe, mol_id)

                    elif search_type == "grid":
                        # Use MDAnalysis neighbor search.
                        neighbors = search.get_grid_neighbors(
                            universe, search_object, mol_id, highest_level
                        )
                    else:
                        # Raise error for unavailale search_type
                        raise ValueError(f"unknown search_type {search_type}")

                    if group_id in number_neighbors:
                        number_neighbors[group_id].append(len(neighbors))
                    else:
                        number_neighbors[group_id] = [len(neighbors)]

            # Get the average number of neighbors:
            # dividing the sum by the number of molecules and number of frames
            number = np.sum(number_neighbors[group_id])
            average_number_neighbors[group_id] = number / (
                len(molecules) * number_frames
            )
            logger.debug(
                f"group {group_id}:"
                f"number neighbors {average_number_neighbors[group_id]}"
            )

        return average_number_neighbors

    def get_symmetry(self, universe, groups):
        """
        Calculate symmetry number for the molecule.

        Args:
            universe: MDAnalysis object
            mol_id: index of the molecule of interest

        Returns:
            symmetry_number: dict, symmetry number of each group
            linear: dict, linear for each group
        """
        symmetry_number = {}
        linear = {}

        for group_id in groups.keys():
            molecules = groups[group_id]

            # Get rdkit object
            rdkit_mol, number_heavy, number_hydrogen = self._get_rdkit_mol(
                universe, molecules[0]
            )

            # Get symmetry number
            symmetry_number[group_id] = self._get_symmetry(
                rdkit_mol, number_heavy, number_hydrogen
            )

            # Is the molecule linear?
            linear[group_id] = self._get_linear(rdkit_mol, number_heavy)

        logger.debug(f"group: {group_id}, symmetry: {symmetry_number} linear: {linear}")

        return symmetry_number, linear

    def _get_rdkit_mol(self, universe, mol_id):
        """
        Convert molecule to rdkit object.

        Args:
            universe: MDAnalysis object
            mol_id: index of the molecule of interest

        Returns:
            rdkit_mol
            number_heavy
            number_hydrogen
        """

        # MDAnalysis convert_to(RDKIT) needs elements
        if not hasattr(universe.atoms, "elements"):
            universe.guess_TopologyAttrs(to_guess=["elements"])

        # pick molecule
        molecule = universe.atoms.fragments[mol_id]

        # Remove dummy atoms and convert to rkdit format.
        # If there are dummy atoms you need inferrer=None otherwise you
        # get errors from it getting the valence wrong.
        # If possible it is better to use the inferrer to get the bonds
        # and hybridization correct.
        dummy = molecule.select_atoms("prop mass < 0.1")
        if len(dummy) > 0:
            frag = molecule.select_atoms("prop mass > 0.1")
            rdkit_mol = frag.convert_to("RDKIT", force=True, inferrer=None)
            logger.debug("Warning: Dummy atoms found")
        else:
            rdkit_mol = molecule.convert_to("RDKIT", force=True)

        number_heavy = rdkit_mol.GetNumHeavyAtoms()
        number_hydrogen = rdkit_mol.GetNumAtoms() - number_heavy

        return rdkit_mol, number_heavy, number_hydrogen

    def _get_symmetry(self, rdkit_mol, number_heavy, number_hydrogen):
        """
        Calculate symmetry number for the molecule.

        Args:
            rdkit_mol: rdkit object for molecule of interest
            number_heavy (int): number of heavy atoms
            number_hydrogen (int): number of hydrogen atoms

        Returns:
            symmetry_number (int): symmetry number of molecule
        """

        # Find symmetry
        if number_heavy > 1:
            # if multiple heavy atoms remove hydrogens to prevent finding
            # too many permutations of atoms
            rdkit_heavy = Chem.RemoveHs(rdkit_mol)
            matches = rdkit_mol.GetSubstructMatches(
                rdkit_heavy, uniquify=False, useChirality=True
            )
            symmetry_number = len(matches)
        elif number_hydrogen > 0:
            # if only one heavy atom use the hydrogens
            matches = rdkit_mol.GetSubstructMatches(
                rdkit_mol, uniquify=False, useChirality=True
            )
            symmetry_number = len(matches)
        else:
            # one heavy atom and no hydrogens = spherical symmetry
            symmetry_number = 0

        return symmetry_number

    def _get_linear(self, rdkit_mol, number_heavy):
        """
        Determine if the molecule is linear.

        Args:
            rkdit_mol: rdkit object for molecule of interest
            number_heavy (int): number of heavy atoms

        Returns:
            linear (bool): True if molecule linear
        """
        # Don't consider hydrogens
        rdkit_heavy = Chem.RemoveHs(rdkit_mol)

        linear = False
        if number_heavy == 1:
            linear = False
        elif number_heavy == 2:
            linear = True
        else:
            sp_count = 0
            for x in rdkit_heavy.GetAtoms():
                if x.GetHybridization() == Chem.HybridizationType.SP:
                    sp_count += 1
            if sp_count >= (number_heavy - 2):
                linear = True

        return linear
