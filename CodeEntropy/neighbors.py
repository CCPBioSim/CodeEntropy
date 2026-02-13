import logging

import MDAnalysis as mda
import numpy as np

# import rdkit
from rdkit import Chem

from CodeEntropy import RAD

# import numpy as np
# from rich.progress import (
#    BarColumn,
#    Progress,
#    SpinnerColumn,
#    TextColumn,
#    TimeElapsedColumn,
# )


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

    def get_neighbors(self, universe, groups, levels, use_RAD):
        """
        Find the neighbors relative to the central molecule.

        Args:
            universe: MDAnalysis universe object for the system
            groups: list of groups for averaging
            levels: list of levels for each molecule
            use_RAD (bool): if true use relative angular distance method for
                finding neighbors

        Returns:
            average_number_neighbors (dict): average number of neighbors
                at each frame for each group
        """

        number_neighbors = {}
        average_number_neighbors = {}
        #        biases = {}

        number_frames = len(universe.trajectory)
        search_object = mda.lib.NeighborSearch.AtomNeighborSearch(universe.atoms)

        for group_id in groups.keys():
            molecules = groups[group_id]
            highest_level = levels[molecules[0]][-1]

            for mol_id in molecules:
                fragment = universe.atoms.fragments[mol_id]
                selection_string = f"index {fragment.indices[0]}:{fragment.indices[-1]}"
                molecule_atom_group = universe.select_atoms(selection_string)

                for timestep in range(number_frames):
                    # This is to get MDAnalysis to get the information from the
                    # correct frame of the trajectory
                    universe.trajectory[timestep]

                    if use_RAD:
                        # Use the relative angular distance method to find neighbors
                        neighbors = RAD.get_RAD_neighbors(universe, mol_id)

                    else:
                        if highest_level == "united_atom":
                            # For united atom size molecules, use the grid search
                            # to find neighboring atoms
                            search_level = "A"
                            search = mda.lib.NeighborSearch.AtomNeighborSearch.search(
                                search_object,
                                molecule_atom_group,
                                radius=2.5,
                                level=search_level,
                            )
                            # Make sure that the neighbors list does not include
                            # atoms from the central molecule
                            #  neighbors = search - fragment.residues
                            neighbors = search - molecule_atom_group
                        else:
                            # For larger molecules, use the grid search
                            # to find neighboring residues
                            search_level = "R"
                            search = mda.lib.NeighborSearch.AtomNeighborSearch.search(
                                search_object,
                                molecule_atom_group,
                                radius=3.0,
                                level=search_level,
                            )
                            # Make sure that the neighbors list does not include
                            # residues from the central molecule
                            neighbors = search - fragment.residues

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

        # #TODO figure out best way to do bias calculation
        #            bias[group_id] = get_bias(
        #                    universe, molecules, average_number_neighbors[group_id],
        #                    number_frames
        #            )

        return average_number_neighbors

    def get_symmetry(self, universe, groups):
        """
        Calculate symmetry number for the molecule.

        Args:
            universe: MDAnalysis object
            mol_id: index of the molecule of interest

        Returns:
            symmetry_number
        """
        symmetry_number = {}
        linear = {}

        for group_id in groups.keys():
            molecules = groups[group_id]

            # Determine symmetry number
            # All molecules in the group have the same symmetry number
            symmetry_number[group_id], linear[group_id] = self._get_symmetry_number(
                universe, molecules[0]
            )

        return symmetry_number, linear

    def _get_symmetry_number(self, universe, mol_id):
        """
        Calculate symmetry number for the molecule.

        Args:
            universe: MDAnalysis object
            mol_id: index of the molecule of interest

        Returns:
            symmetry_number (int): symmetry number of molecule
            linear (bool): True if molecule linear
        """

        # MDAnalysis convert_to(RDKIT) needs elements
        if not hasattr(universe.atoms, "elements"):
            universe.guess_TopologyAttrs(to_guess=["elements"])

        # pick molecule and convert to rdkit format
        molecule = universe.atoms.fragments[mol_id]
        rdkit_mol = molecule.convert_to("RDKIT", force=True)
        number_heavy = rdkit_mol.GetNumHeavyAtoms()
        number_hydrogen = rdkit_mol.GetNumAtoms() - number_heavy

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

        # Check for linearity
        # Don't consider hydrogens
        # Think about how molecules of one united atom should be considered
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

        # TODO temp print
        print(f"symmetry {symmetry_number} linear {linear}")

        logger.debug(
            f"molecule: {mol_id}, symmetry: {symmetry_number} linear: {linear}"
        )

        return symmetry_number, linear

    def _get_bias(self, universe, molecules, average_number_neighbors, number_frames):
        """
        Define a bias factor to account for the intermolecular interactions
        constraining the orientations of the molecules.

        Args:
            universe: the MDAnalysis universe object for the system

        Returns:
            bias (float): the bias factor
        """
