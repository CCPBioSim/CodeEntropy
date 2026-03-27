"""Neighbours info for orientational entropy.

This module finds the average number of neighbors, symmetry numbers, and
and linearity for each group.
These are used downstream to compute the orientational entropy.
"""

import logging

import numpy as np
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from rdkit import Chem

from CodeEntropy.levels.hbond_bias import HBondBias
from CodeEntropy.levels.search import Search

logger = logging.getLogger(__name__)


class Neighbors:
    """
    Class to find the neighbors and any related information needed for the
    calculation of orientational entropy.
    """

    def __init__(self):
        """
        Initializes the Neighbors class with placeholders for data,
        including the system trajectory, groups, and levels.
        """

        self._universe = None
        self._groups = None
        self._levels = None
        self._search = Search()
        self._bias = HBondBias()

    def get_neighbors(self, universe, levels, groups, search_type):
        """
        Find the neighbors relative to the central molecule.

        The search defaults to using RAD, but an MDAnalysis method based
        on grid searches is also available.
        The average number of neighbors is calculated.

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
                        neighbors = self._search.get_RAD_neighbors(
                            universe=universe, mol_id=mol_id
                        )

                    elif search_type == "grid":
                        # Use MDAnalysis neighbor search.
                        neighbors = self._search.get_grid_neighbors(
                            universe=universe,
                            mol_id=mol_id,
                            highest_level=highest_level,
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
                f"group: {group_id}"
                f"number neighbors {average_number_neighbors[group_id]}"
            )

        return average_number_neighbors

    def get_bias(self, universe, groups):
        """
        Calculate orientational bias factors.

        Args:
            universe: MDAnalysis object
            groups: list of molecules sorted into groups

        Returns:
            bias_factor: hydrogen bonding factor for each group
            n_factor: scaling factor for number of neighbors
        """
        bias_factor = {}
        n_factor = {}

        # Get H-bonds from MDAnalysis for whole universe
        hbonds = HBA(universe=universe)
        hbonds.run()
        donors = hbonds.results.hbonds[:, 1].astype(int)
        acceptors = hbonds.results.hbonds[:, 3].astype(int)

        for group_id in groups.keys():
            bias_factor[group_id], n_factor[group_id] = self._bias.get_hbond_bias(
                universe, groups, group_id, donors, acceptors
            )

        return bias_factor, n_factor

    def get_symmetry(self, universe, groups):
        """
        Calculate symmetry number for the molecule.

        This function converts the MDAnalysis instance of a molecule into
        an RDKit object and then calculates the symmetry number and
        determines if the molecule is linear.

        Args:
            universe: MDAnalysis object
            groups: indices of the molecules for each group

        Returns:
            symmetry_number: dict, symmetry number of each group
            linear: dict, linear for each group
        """
        symmetry_number = {}
        linear = {}

        for group_id in groups.keys():
            molecules = groups[group_id]

            rdkit_mol, number_heavy, number_hydrogen = self._get_rdkit_mol(
                universe, molecules[0]
            )

            symmetry_number[group_id] = self._get_symmetry_number(
                rdkit_mol, number_heavy, number_hydrogen
            )

            linear[group_id] = self._get_linear(rdkit_mol, number_heavy)

            logger.debug(
                f"group: {group_id}, symmetry: {symmetry_number}, linear: {linear}"
            )

        return symmetry_number, linear

    def _get_rdkit_mol(self, universe, mol_id):
        """
        Convert molecule to rdkit object.

        MDAnalysis convert_to(RDKIT) needs elements.
        We are removing dummy atoms and converting to rkdit format.
        If there are dummy atoms you need inferrer=None otherwise you
        get errors from it getting the valence wrong.
        If possible it is better to use the inferrer to get the bonds
        and hybridization correct.
        The convert_to argument force=True forces it to continue even when
        it cannot find hydrogens, this is needed to avoid errors for molecules
        like carbon dioxide which do not have hydrogens.

        Args:
            universe: MDAnalysis object
            mol_id: index of the molecule of interest

        Returns:
            rdkit_mol
            number_heavy
            number_hydrogen
        """

        if not hasattr(universe.atoms, "elements"):
            universe.guess_TopologyAttrs(to_guess=["elements"])

        molecule = universe.atoms.fragments[mol_id]

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

    def _get_symmetry_number(self, rdkit_mol, number_heavy, number_hydrogen):
        """
        Calculate symmetry number for the molecule.

        For larger molecules, removing the hydrogens reduces the over counting
        of symmetry states. When there is only one heavy atom the hydrogens
        are important to the symmetry. If there is a single heavy atom with
        no hydrogens then the molecule is spherically symmetric.

        Using the RDKit GetSubstructMatches function often works well as
        a guess for the symmetry number, but it occasionally returns a
        symmetry number 2x the expected value (for example, cyclohexane).

        Args:
            rdkit_mol: rdkit object for molecule of interest
            number_heavy (int): number of heavy atoms
            number_hydrogen (int): number of hydrogen atoms

        Returns:
            symmetry_number (int): symmetry number of molecule
        """

        if number_heavy > 1:
            rdkit_heavy = Chem.RemoveHs(rdkit_mol)
            matches = rdkit_mol.GetSubstructMatches(
                rdkit_heavy, uniquify=False, useChirality=True
            )
            symmetry_number = len(matches)
        elif number_hydrogen > 0:
            matches = rdkit_mol.GetSubstructMatches(
                rdkit_mol, uniquify=False, useChirality=True
            )
            symmetry_number = len(matches)
        else:
            symmetry_number = 0

        return symmetry_number

    def _get_linear(self, rdkit_mol, number_heavy):
        """
        Determine if the molecule is linear.

        We are not considering the hydrogens, just the united atom beads.
        So, molecules like methanol are treated as linear since they have only
        two united atoms.

        Args:
            rkdit_mol: rdkit object for molecule of interest
            number_heavy (int): number of heavy atoms

        Returns:
            linear (bool): True if molecule linear
        """
        linear = False
        if number_heavy == 1:
            linear = False
        elif number_heavy == 2:
            linear = True
        else:
            rdkit_heavy = Chem.RemoveHs(rdkit_mol)
            sp_count = 0
            for x in rdkit_heavy.GetAtoms():
                if x.GetHybridization() == Chem.HybridizationType.SP:
                    sp_count += 1
            if sp_count >= (number_heavy - 2):
                linear = True

        return linear
