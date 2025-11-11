import logging

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

logger = logging.getLogger(__name__)


class DihedralAnalysis:
    """ """

    def __init__(self):
        """
        Initializes the DihedralAnalysis with placeholders for level-related data,
        including translational and rotational axes, number of beads, and a
        general-purpose data container.
        """

    def get_dihedrals(self, data_container, level):
        """
        Define the set of dihedrals for use in the conformational entropy function.
        If united atom level, the dihedrals are defined from the heavy atoms
        (4 bonded atoms for 1 dihedral).
        If residue level, use the bonds between residues to cast dihedrals.
        Note: not using improper dihedrals only ones with 4 atoms/residues
        in a linear arrangement.

        Args:
          data_container (MDAnalysis.Universe): system information
          level (str): level of the hierarchy (should be residue or polymer)

        Returns:
           dihedrals (array): set of dihedrals
        """
        # Start with empty array
        dihedrals = []

        # if united atom level, read dihedrals from MDAnalysis universe
        if level == "united_atom":
            dihedrals = data_container.dihedrals

        # if residue level, looking for dihedrals involving residues
        if level == "residue":
            num_residues = len(data_container.residues)
            logger.debug(f"Number Residues: {num_residues}")
            if num_residues < 4:
                logger.debug("no residue level dihedrals")

            else:
                # find bonds between residues N-3:N-2 and N-1:N
                for residue in range(4, num_residues + 1):
                    # Using MDAnalysis selection,
                    # assuming only one covalent bond between neighbouring residues
                    # TODO not written for branched polymers
                    atom_string = (
                        "resindex "
                        + str(residue - 4)
                        + " and bonded resindex "
                        + str(residue - 3)
                    )
                    atom1 = data_container.select_atoms(atom_string)

                    atom_string = (
                        "resindex "
                        + str(residue - 3)
                        + " and bonded resindex "
                        + str(residue - 4)
                    )
                    atom2 = data_container.select_atoms(atom_string)

                    atom_string = (
                        "resindex "
                        + str(residue - 2)
                        + " and bonded resindex "
                        + str(residue - 1)
                    )
                    atom3 = data_container.select_atoms(atom_string)

                    atom_string = (
                        "resindex "
                        + str(residue - 1)
                        + " and bonded resindex "
                        + str(residue - 2)
                    )
                    atom4 = data_container.select_atoms(atom_string)

                    atom_group = atom1 + atom2 + atom3 + atom4
                    dihedrals.append(atom_group.dihedral)

        logger.debug(f"Level: {level}, Dihedrals: {dihedrals}")

        return dihedrals

    def compute_dihedral_conformations(
        self,
        selector,
        level,
        number_frames,
        bin_width,
        start,
        end,
        step,
        ce,
    ):
        """
        Compute dihedral conformations for a given selector and entropy level.

        Parameters:
            selector (AtomGroup): Atom selection to compute dihedrals for.
            level (str): Entropy level ("united_atom" or "residue").
            number_frames (int): Number of frames to process.
            bin_width (float): Bin width for dihedral angle discretization.
            start (int): Start frame index.
            end (int): End frame index.
            step (int): Step size for frame iteration.
            ce : Conformational Entropy class

        Returns:
            states (list): List of conformation strings per frame.
        """
        # Identify the dihedral angles in the residue/molecule
        dihedrals = self.get_dihedrals(selector, level)

        # When there are no dihedrals, there is only one possible conformation
        # so the conformational states are not relevant
        if len(dihedrals) == 0:
            logger.debug("No dihedrals found; skipping conformation assignment.")
            states = []
        else:
            # Identify the conformational label for each dihedral at each frame
            num_dihedrals = len(dihedrals)
            conformation = np.zeros((num_dihedrals, number_frames))

            for i, dihedral in enumerate(dihedrals):
                conformation[i] = ce.assign_conformation(
                    selector, dihedral, number_frames, bin_width, start, end, step
                )

            # for all the dihedrals available concatenate the label of each
            # dihedral into the state for that frame
            states = [
                state
                for state in (
                    "".join(str(int(conformation[d][f])) for d in range(num_dihedrals))
                    for f in range(number_frames)
                )
                if state
            ]

        logger.debug(f"level: {level}, states: {states}")

        return states

    def build_conformational_states(
        self,
        entropy_manager,
        reduced_atom,
        levels,
        groups,
        start,
        end,
        step,
        number_frames,
        bin_width,
        ce,
    ):
        """
        Construct the conformational states for each molecule at
        relevant levels.

        Parameters:
            entropy_manager (EntropyManager): Instance of the EntropyManager
            reduced_atom (Universe): The reduced atom selection.
            levels (list): List of entropy levels per molecule.
            groups (dict): Groups for averaging over molecules.
            start (int): Start frame index.
            end (int): End frame index.
            step (int): Step size for frame iteration.
            number_frames (int): Total number of frames to process.
            bin_width (int): Width of histogram bins.
            ce: Conformational Entropy object

        Returns:
            tuple: A tuple containing:
                - states_ua (dict): Conformational states at the united-atom level.
                - states_res (list): Conformational states at the residue level.
        """
        number_groups = len(groups)
        states_ua = {}
        states_res = [None] * number_groups

        total_items = sum(
            len(levels[mol_id]) for mols in groups.values() for mol_id in mols
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[title]}", justify="right"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
        ) as progress:

            task = progress.add_task(
                "[green]Building Conformational States...",
                total=total_items,
                title="Starting...",
            )

            for group_id in groups.keys():
                molecules = groups[group_id]
                for mol_id in molecules:
                    mol = entropy_manager._get_molecule_container(reduced_atom, mol_id)

                    resname = mol.atoms[0].resname
                    resid = mol.atoms[0].resid
                    segid = mol.atoms[0].segid

                    mol_label = f"{resname}_{resid} (segid {segid})"

                    for level in levels[mol_id]:
                        progress.update(
                            task,
                            title=f"Building conformational states | "
                            f"Molecule: {mol_label} | "
                            f"Level: {level}",
                        )

                        if level == "united_atom":
                            for res_id, residue in enumerate(mol.residues):
                                key = (group_id, res_id)

                                res_container = (
                                    entropy_manager._run_manager.new_U_select_atom(
                                        mol,
                                        f"index {residue.atoms.indices[0]}:"
                                        f"{residue.atoms.indices[-1]}",
                                    )
                                )
                                heavy_res = (
                                    entropy_manager._run_manager.new_U_select_atom(
                                        res_container, "prop mass > 1.1"
                                    )
                                )
                                states = self.compute_dihedral_conformations(
                                    heavy_res,
                                    level,
                                    number_frames,
                                    bin_width,
                                    start,
                                    end,
                                    step,
                                    ce,
                                )

                                if key in states_ua:
                                    states_ua[key].extend(states)
                                else:
                                    states_ua[key] = states

                        elif level == "residue":
                            states = self.compute_dihedral_conformations(
                                mol,
                                level,
                                number_frames,
                                bin_width,
                                start,
                                end,
                                step,
                                ce,
                            )

                            if states_res[group_id] is None:
                                states_res[group_id] = states
                            else:
                                states_res[group_id].extend(states)

                        progress.advance(task)

        logger.debug(f"states_ua {states_ua}")
        logger.debug(f"states_res {states_res}")

        return states_ua, states_res
