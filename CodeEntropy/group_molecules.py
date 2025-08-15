import logging

logger = logging.getLogger(__name__)


class GroupMolecules:
    """
    Groups molecules for averaging.
    """

    def __init__(self):
        """
        Initializes the class with relevant information.

        Args:
            run_manager: Manager for universe and selection operations.
            args: Argument namespace containing user parameters.
            universe: MDAnalysis universe representing the simulation system.
            data_logger: Logger for storing and exporting entropy data.
        """
        self._molecule_groups = None

    def grouping_molecules(self, universe, grouping):
        """
        Grouping molecules by desired level of detail.
        """

        molecule_groups = {}

        if grouping == "each":
            molecule_groups = self._by_none(universe)

        if grouping == "molecules":
            molecule_groups = self._by_molecules(universe)

        return molecule_groups

    def _by_none(self, universe):
        """
        Don't group molecules. Every molecule is in its own group.
        """

        # fragments is MDAnalysis terminology for molecules
        number_molecules = len(universe.atoms.fragments)

        molecule_groups = {}

        for molecule_i in range(number_molecules):
            molecule_groups[molecule_i] = [molecule_i]

        number_groups = len(molecule_groups)

        logger.info(f"Number of molecule groups: {number_groups}")
        logger.debug(f"Molecule groups are: {molecule_groups}")

        return molecule_groups

    def _by_molecules(self, universe):
        """
        Group molecules by chemical type.
        Based on number of atoms and atom names.
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

                if number_atoms_i == number_atoms_j and (names_i == names_j).all:
                    if molecule_j in molecule_groups.keys():
                        molecule_groups[molecule_j].append(molecule_i)
                    else:
                        molecule_groups[molecule_j] = []
                        molecule_groups[molecule_j].append(molecule_i)
                    break

        number_groups = len(molecule_groups)

        logger.info(f"Number of molecule groups: {number_groups}")
        logger.debug(f"Molecule groups are: {molecule_groups}")

        return molecule_groups
