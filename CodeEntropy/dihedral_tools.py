import logging

import numpy as np
from MDAnalysis.analysis.dihedrals import Dihedral
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

logger = logging.getLogger(__name__)


class DihedralAnalysis:
    """
    Functions for finding dihedral angles and analysing them to get the
    states needed for the conformational entropy functions.
    """

    def __init__(self, universe_operations=None):
        """
        Initialise with placeholders.
        """
        self._universe_operations = universe_operations
        self.data_container = None
        self.states_ua = None
        self.states_res = None

    def build_conformational_states(
        self,
        data_container,
        levels,
        groups,
        start,
        end,
        step,
        bin_width,
    ):
        """
        Build the conformational states descriptors based on dihedral angles
        needed for the calculation of the conformational entropy.
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
            mol = self._universe_operations.get_molecule_container(data_container, 0)
            num_residues = len(mol.residues)
            dihedrals_ua = [[] for _ in range(num_residues)]
            peaks_ua = [{} for _ in range(num_residues)]
            dihedrals_res = []
            peaks_res = {}

            # Identify dihedral AtomGroups
            for level in levels[molecules[0]]:
                if level == "united_atom":
                    for res_id in range(num_residues):
                        selection1 = mol.residues[res_id].atoms.indices[0]
                        selection2 = mol.residues[res_id].atoms.indices[-1]
                        res_container = self._universe_operations.new_U_select_atom(
                            mol,
                            f"index {selection1}:" f"{selection2}",
                        )
                        heavy_res = self._universe_operations.new_U_select_atom(
                            res_container, "prop mass > 1.1"
                        )

                        dihedrals_ua[res_id] = self._get_dihedrals(heavy_res, level)

                elif level == "residue":
                    dihedrals_res = self._get_dihedrals(mol, level)

            # Identify peaks
            for level in levels[molecules[0]]:
                if level == "united_atom":
                    for res_id in range(num_residues):
                        if len(dihedrals_ua[res_id]) == 0:
                            # No dihedrals means no histogram or peaks
                            peaks_ua[res_id] = []
                        else:
                            peaks_ua[res_id] = self._identify_peaks(
                                data_container,
                                molecules,
                                dihedrals_ua[res_id],
                                bin_width,
                                start,
                                end,
                                step,
                            )

                elif level == "residue":
                    if len(dihedrals_res) == 0:
                        # No dihedrals means no histogram or peaks
                        peaks_res = []
                    else:
                        peaks_res = self._identify_peaks(
                            data_container,
                            molecules,
                            dihedrals_res,
                            bin_width,
                            start,
                            end,
                            step,
                        )

            # Assign states for each group
            for level in levels[molecules[0]]:
                if level == "united_atom":
                    for res_id in range(num_residues):
                        key = (group_id, res_id)
                        if len(dihedrals_ua[res_id]) == 0:
                            # No conformational states
                            states_ua[key] = []
                        else:
                            states_ua[key] = self._assign_states(
                                data_container,
                                molecules,
                                dihedrals_ua[res_id],
                                peaks_ua[res_id],
                                start,
                                end,
                                step,
                            )

                elif level == "residue":
                    if len(dihedrals_res) == 0:
                        # No conformational states
                        states_res[group_id] = []
                    else:
                        states_res[group_id] = self._assign_states(
                            data_container,
                            molecules,
                            dihedrals_res,
                            peaks_res,
                            start,
                            end,
                            step,
                        )

            progress.advance(task)

        return states_ua, states_res

    def _get_dihedrals(self, data_container, level):
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
        atom_groups = []

        # if united atom level, read dihedrals from MDAnalysis universe
        if level == "united_atom":
            dihedrals = data_container.dihedrals
            num_dihedrals = len(dihedrals)
            for index in range(num_dihedrals):
                atom_groups.append(dihedrals[index].atoms)

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
                    atom_groups.append(atom_group)

        logger.debug(f"Level: {level}, Dihedrals: {atom_groups}")

        return atom_groups

    def _identify_peaks(
        self,
        data_container,
        molecules,
        dihedrals,
        bin_width,
        start,
        end,
        step,
    ):
        """
        Build a histogram of the dihedral data and identify the peaks.
        This is to give the information needed for the adaptive method
        of identifying dihedral states.
        """
        peak_values = [] * len(dihedrals)
        for dihedral_index in range(len(dihedrals)):
            phi = []
            # get the values of the angle for the dihedral
            # loop over all molecules in the averaging group
            # dihedral angle values have a range from -180 to 180
            for molecule in molecules:
                mol = self._universe_operations.get_molecule_container(
                    data_container, molecule
                )
                number_frames = len(mol.trajectory)
                R = Dihedral(dihedrals).run()
                for timestep in range(number_frames):
                    value = R.results.angles[timestep][dihedral_index]

                    # We want postive values in range 0 to 360 to make
                    # the peak assignment.
                    # works using the fact that dihedrals have circular symetry
                    # (i.e. -15 degrees = +345 degrees)
                    if value < 0:
                        value += 360
                    phi.append(value)

            # create a histogram using numpy
            number_bins = int(360 / bin_width)
            popul, bin_edges = np.histogram(a=phi, bins=number_bins, range=(0, 360))
            bin_value = [
                0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(0, len(popul))
            ]

            # identify "convex turning-points" and populate a list of peaks
            # peak : a bin whose neighboring bins have smaller population
            # NOTE might have problems if the peak is wide with a flat or
            # sawtooth top in which case check you have a sensible bin width

            peaks = []
            for bin_index in range(number_bins):
                # if there is no dihedrals in a bin then it cannot be a peak
                if popul[bin_index] == 0:
                    pass
                # being careful of the last bin
                # (dihedrals have circular symmetry, the histogram does not)
                elif (
                    bin_index == number_bins - 1
                ):  # the -1 is because the index starts with 0 not 1
                    if (
                        popul[bin_index] >= popul[bin_index - 1]
                        and popul[bin_index] >= popul[0]
                    ):
                        peaks.append(bin_value[bin_index])
                else:
                    if (
                        popul[bin_index] >= popul[bin_index - 1]
                        and popul[bin_index] >= popul[bin_index + 1]
                    ):
                        peaks.append(bin_value[bin_index])

            peak_values.append(peaks)

            logger.debug(f"Dihedral: {dihedral_index}, Peak Values: {peak_values}")

        return peak_values

    def _assign_states(
        self,
        data_container,
        molecules,
        dihedrals,
        peaks,
        start,
        end,
        step,
    ):
        """
        Turn the dihedral values into conformations based on the peaks
        from the histogram.
        Then combine these to form states for each molecule.
        """
        conformations = []
        states = []

        # get the values of the angle for the dihedral
        # dihedral angle values have a range from -180 to 180
        for molecule in molecules:
            mol = self._universe_operations.get_molecule_container(
                data_container, molecule
            )
            number_frames = len(mol.trajectory)
            R = Dihedral(dihedrals).run()
            for dihedral_index in range(len(dihedrals)):
                conformation = []
                for timestep in range(number_frames):
                    value = R.results.angles[timestep][dihedral_index]

                    # We want postive values in range 0 to 360 to make
                    # the peak assignment.
                    # works using the fact that dihedrals have circular symetry
                    # (i.e. -15 degrees = +345 degrees)
                    if value < 0:
                        value += 360

                    # Find the turning point/peak that the snapshot is closest to.
                    distances = [abs(value - peak) for peak in peaks[dihedral_index]]
                    conformation.append(np.argmin(distances))

                    logger.debug(
                        f"Dihedral: {dihedral_index} Conformations: {conformation}"
                    )
                conformations.append(conformation)

            # for all the dihedrals available concatenate the label of each
            # dihedral into the state for that frame
            mol_states = [
                state
                for state in (
                    "".join(
                        str(int(conformations[d][f])) for d in range(len(dihedrals))
                    )
                    for f in range(number_frames)
                )
                if state
            ]

            if states is None:
                states = mol_states
            else:
                states.extend(mol_states)

        logger.debug(f"States: {states}")

        return states
