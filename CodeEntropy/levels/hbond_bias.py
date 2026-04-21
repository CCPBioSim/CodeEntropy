"""These functions find hydrogen bonds.

There are functions for identifying hydrogen bonds and calculating
the bias factor for the orientational entropy calculation.
"""

import logging
from collections import Counter

from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA

from CodeEntropy.levels.mda import UniverseOperations

logger = logging.getLogger(__name__)


class HBondBias:
    """
    Class for functions to find hydrogen bond biases.
    """

    def __init__(self):
        """
        Initializes the Search class with a placeholder for the system
        trajectory.
        """

        self._universe_operations = UniverseOperations()
        self._universe = None
        self._mol_id = None

    def get_hbond_bias(
        self, universe, groups, group_id, donors, acceptors
    ) -> tuple[float, float]:
        """
        Calculate hydrogen bonding bias factors.

        Args:
            universe: The MDAnalysis universe of the system.
            mol_id (int): the index for the central molecule.

        Returns:
            hbond_factor (float): the hydrogen bond factor.
        """
        mols = groups[group_id]
        number_frames = len(universe.trajectory)

        mol_id = mols[0]
        rep_mol = self._universe_operations.extract_fragment(
            universe=universe, molecule_id=mol_id
        )

        # Find potential donors and acceptors per molecule
        max_donors, max_acceptors = self.get_possible_donors(rep_mol)

        if max_donors == 0 and max_acceptors == 0:
            # No hydrogen bonding possiblity
            hbond_factor = 1

            logger.debug(f"No hydrogen bond possible: {hbond_factor}")

            return hbond_factor

        normal_factor = 1 / (max_donors + max_acceptors)

        # Get probablities
        n_donor, n_acceptor = self.get_hbond_info(universe, mols, donors, acceptors)
        if n_donor == 0 and n_acceptor == 0:
            # No hydrogen bonding
            hbond_factor = 1

            logger.debug(f"No hydrogen bonds: {hbond_factor}")

            return hbond_factor

        if max_donors == 0:
            p_donor = 1
        else:
            p_donor = n_donor / (max_donors * number_frames * len(mols))
            p_donor = min(p_donor, 1)

        if max_acceptors == 0:
            p_acceptor = 1
        else:
            p_acceptor = n_acceptor / (max_acceptors * number_frames * len(mols))
            p_acceptor = min(p_acceptor, 1)

        hbond_factor = (1 - p_donor) * (1 - p_acceptor)
        hbond_factor = max(hbond_factor, normal_factor)

        # TODO temp print
        print(f"N donor {n_donor}, N acceptor {n_acceptor}")
        print(f"donor {p_donor}, acceptor {p_acceptor}, hbond {hbond_factor}")

        logger.debug(f"Hydrogen bond donations: {max_donors}, {n_donor}, {p_donor}")
        logger.debug(
            f"Hydrogen bond acceptors: {max_acceptors}, {n_acceptor}, {p_acceptor}"
        )
        logger.debug(f"Hydrogen bond factor: {hbond_factor}")

        return min(hbond_factor, 1)

    def get_possible_donors(self, mol) -> tuple[int, int]:
        """
        Find possible hydrogen bond donors and acceptors.

        Args:
            mol: MDAnalysis universe for the molecule in question.

        Returns:
            donors (int): number of H-bond donors
            number_acceptors (int): number of H-bond acceptors
        """
        mol.guess_TopologyAttrs(to_guess=["elements"])
        hbonds = HBA(universe=mol)
        hbonds.acceptors_sel = hbonds.guess_acceptors()
        hbonds.hydrogens_sel = hbonds.guess_hydrogens()

        acceptors = mol.select_atoms(hbonds.acceptors_sel)
        hydrogens = mol.select_atoms(hbonds.hydrogens_sel)

        number_acceptors = 0
        for atom in acceptors:
            if atom.element in ("N", "F"):
                number_acceptors += 1
            else:
                number_acceptors += 2  # Oxygen or water dummy atom

        return len(hydrogens), number_acceptors

    def get_hbond_info(self, universe, mols, donors, acceptors) -> tuple[int, int]:
        """
        Find the total number of hydrogen bond donations and acceptances.

        Args:
            universe: MDAnalysis object
            mols (list of ints): molecule ids of the group.
            donors (list of ints): atom indices of the hbond donors.
            acceptors (list of ints): atom indices of the hbond acceptors.

        Returns:
            number_donors (int)
            number_acceptors (int)
        """
        # Identify atom indices for the group
        indices = []
        for mol in mols:
            indices.extend(universe.atoms.fragments[mol].indices)

        # Count number of times the indices appear in the donor/acceptor data
        number_donors = 0
        number_acceptors = 0
        donor_counter = Counter(donors)
        acceptor_counter = Counter(acceptors)

        # TODO temp print
        print(len(donors))

        for index in indices:
            number_donors += donor_counter[index]
            number_acceptors += acceptor_counter[index]

        return number_donors, number_acceptors
