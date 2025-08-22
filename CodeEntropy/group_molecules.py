import logging

logger = logging.getLogger(__name__)


class GroupMolecules:
    """
    Groups molecules for averaging.
    """

    def __init__(self):
        """
        Initializes the class with relevant information.

        """
        self._molecule_groups = None

    def grouping_molecules(self, universe, grouping):
        """
        Grouping molecules by desired level of detail.

        Args:
            universe: MDAnalysis univers object for the system of interest.
            grouping (str): how to group molecules for averaging

        Returns:
            molecule_groups (dict): molecule indices for each group.
        """

        molecule_groups = {}

        if grouping == "each":
            molecule_groups = self._by_none(universe)

        if grouping == "molecules":
            molecule_groups = self._by_molecules(universe)

        number_groups = len(molecule_groups)

        logger.info(f"Number of molecule groups: {number_groups}")
        logger.debug(f"Molecule groups are: {molecule_groups}")

        return molecule_groups

    def _by_none(self, universe):
        """
        Don't group molecules. Every molecule is in its own group.

        Args:
            universe: MDAnalysis universe

        Returns:
            molecule_groups (dict): molecule indices for each group.
        """

        # fragments is MDAnalysis terminology for molecules
        number_molecules = len(universe.atoms.fragments)

        molecule_groups = {}

        for molecule_i in range(number_molecules):
            molecule_groups[molecule_i] = [molecule_i]

        number_groups = len(molecule_groups)

        logger.debug(f"Number of molecule groups: {number_groups}")
        logger.debug(f"Molecule groups are: {molecule_groups}")

        return molecule_groups

    def _by_molecules(self, universe):
        """
        Group molecules by chemical type.
        Based on number of atoms and atom names.

        Args:
            universe: MDAnalysis universe

        Returns:
            molecule_groups (dict): molecule indices for each group.
        """

        # fragments is MDAnalysis terminology for molecules
        number_molecules = len(universe.atoms.fragments)
        fragments = universe.atoms.fragments

        molecule_groups = {}

        for molecule_i in range(number_molecules):
            names_i = fragments[molecule_i].names
            number_atoms_i = len(names_i)

            for molecule_j in range(number_molecules):
                names_j = fragments[molecule_j].names
                number_atoms_j = len(names_j)

                # If molecule_i has the same number of atoms and same
                # atom names as molecule_j, then index i is added to group j
                # The index of molecule_j is the group key, the keys are
                # all integers, but may not be consecutive numbers.
                if number_atoms_i == number_atoms_j and all(
                    i == j for i, j in zip(names_i, names_j)
                ):
                    if molecule_j in molecule_groups.keys():
                        molecule_groups[molecule_j].append(molecule_i)
                    else:
                        molecule_groups[molecule_j] = []
                        molecule_groups[molecule_j].append(molecule_i)
                    break

        number_groups = len(molecule_groups)

        logger.debug(f"Number of molecule groups: {number_groups}")
        logger.debug(f"Molecule groups are: {molecule_groups}")

        return molecule_groups
