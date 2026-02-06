import logging

logger = logging.getLogger(__name__)


class LevelHierarchy:
    """ """

    def __init__(self):
        """
        Initializes the LevelHierarchy with placeholders for level-related data,
        including translational and rotational axes, number of beads, and a
        general-purpose data container.
        """

    def select_levels(self, data_container):
        """
        Function to read input system and identify the number of molecules and
        the levels (i.e. united atom, residue and/or polymer) that should be used.
        The level refers to the size of the bead (atom or collection of atoms)
        that will be used in the entropy calculations.

        Args:
            arg_DataContainer: MDAnalysis universe object containing the system of
            interest

        Returns:
             number_molecules (int): Number of molecules in the system.
             levels (array): Strings describing the length scales for each molecule.
        """

        # fragments is MDAnalysis terminology for what chemists would call molecules
        number_molecules = len(data_container.atoms.fragments)
        logger.debug(f"The number of molecules is {number_molecules}.")

        fragments = data_container.atoms.fragments
        levels = [[] for _ in range(number_molecules)]

        for molecule in range(number_molecules):
            levels[molecule].append(
                "united_atom"
            )  # every molecule has at least one atom

            atoms_in_fragment = fragments[molecule].select_atoms("prop mass > 1.1")
            number_residues = len(atoms_in_fragment.residues)

            if len(atoms_in_fragment) > 1:
                levels[molecule].append("residue")

                if number_residues > 1:
                    levels[molecule].append("polymer")

        logger.debug(f"levels {levels}")

        return number_molecules, levels

    def get_beads(self, data_container, level):
        """
        Function to define beads depending on the level in the hierarchy.

        Args:
           data_container (MDAnalysis.Universe): the molecule data
           level (str): the heirarchy level (polymer, residue, or united atom)

        Returns:
           list_of_beads : the relevent beads
        """

        if level == "polymer":
            list_of_beads = []
            atom_group = "all"
            list_of_beads.append(data_container.select_atoms(atom_group))

        if level == "residue":
            list_of_beads = []
            num_residues = len(data_container.residues)
            for residue in range(num_residues):
                atom_group = "resindex " + str(residue)
                list_of_beads.append(data_container.select_atoms(atom_group))

        if level == "united_atom":
            list_of_beads = []
            heavy_atoms = data_container.select_atoms("prop mass > 1.1")
            if len(heavy_atoms) == 0:
                list_of_beads.append(data_container.select_atoms("all"))
            else:
                for atom in heavy_atoms:
                    atom_group = (
                        "index "
                        + str(atom.index)
                        + " or ((prop mass <= 1.1) and bonded index "
                        + str(atom.index)
                        + ")"
                    )
                    list_of_beads.append(data_container.select_atoms(atom_group))

        logger.debug(f"List of beads: {list_of_beads}")

        return list_of_beads
