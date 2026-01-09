import logging

import MDAnalysis as mda

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

        Returns
        -------
        average_number_neighbors
        """
        average_number_neighbors = {}
        number_frames = len(universe.trajectory)
        search_object = mda.lib.NeighborSearch.AtomNeighborSearch(universe.atoms)

        for group_id in groups.keys():
            molecules = groups[group_id]
            highest_level = levels[molecules[0]][-1]
            number_neighbors = 0

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

                    number_neighbors += len(neighbors)

            # Get the average number of neighbors:
            # dividing the sum by the number of molecules and number of frames
            average_number_neighbors[group_id] = number_neighbors / (
                len(molecules) * number_frames
            )
            logger.debug(
                f"group {group_id}:"
                f"number neighbors {average_number_neighbors[group_id]}"
            )
            # TODO temp print
            print(f"average number neighbors {average_number_neighbors}")

        return average_number_neighbors
