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
        Identify the number of molecules and which levels (united atom, residue,
        polymer) should be used for each molecule.

        Args:
            data_container: MDAnalysis Universe for the system.

        Returns:
            number_molecules (int)
            levels (list[list[str]])
        """
        number_molecules = len(data_container.atoms.fragments)
        logger.debug(f"The number of molecules is {number_molecules}.")

        fragments = data_container.atoms.fragments
        levels = [[] for _ in range(number_molecules)]

        for molecule in range(number_molecules):
            levels[molecule].append("united_atom")

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
        Define beads depending on the hierarchy level.

        IMPORTANT FIX:
          - For "residue", DO NOT use "resindex i" selection strings.
            resindex is global to the universe and will often produce empty beads
            for molecules beyond the first one.
          - Instead, directly use the residues belonging to the data_container.

        Args:
            data_container: MDAnalysis AtomGroup (typically a molecule/fragment or
            residue.atoms) level (str): "polymer", "residue", or "united_atom"

        Returns:
            list_of_beads: list[AtomGroup]
        """
        if level == "polymer":
            return [data_container.select_atoms("all")]

        if level == "residue":
            list_of_beads = []
            for res in data_container.residues:
                bead = res.atoms
                list_of_beads.append(bead)
            logger.debug(f"Residue beads: {[len(b) for b in list_of_beads]}")
            return list_of_beads

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
                    bead = data_container.select_atoms(atom_group)
                    list_of_beads.append(bead)

            logger.debug(f"United-atom beads: {[len(b) for b in list_of_beads]}")
            return list_of_beads

        raise ValueError(f"Unknown level: {level}")
