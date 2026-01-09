import logging

import numpy as np
import waterEntropy.recipes.interfacial_solvent as GetSolvent

logger = logging.getLogger(__name__)


class WaterEntropy:

    def __init__(self):
        """"""

    def _calculate_water_entropy(self, universe, start, end, step, group_id=None):
        """
        Calculate and aggregate the entropy of water molecules in a simulation.

        This function computes orientational, translational, and rotational
        entropy components for all water molecules, aggregates them per residue,
        and maps all waters to a single group ID. It also logs the total results
        and labels the water group in the data logger.

        Parameters
        ----------
        universe : MDAnalysis.Universe
            The simulation universe containing water molecules.
        start : int
            The starting frame for analysis.
        end : int
            The ending frame for analysis.
        step : int
            Frame interval for analysis.
        group_id : int or str, optional
            The group ID to which all water molecules will be assigned.
        """
        Sorient_dict, covariances, vibrations, _, water_count = (
            GetSolvent.get_interfacial_water_orient_entropy(
                universe, start, end, step, self._args.temperature, parallel=True
            )
        )

        self._calculate_water_orientational_entropy(Sorient_dict, group_id)
        self._calculate_water_vibrational_translational_entropy(
            vibrations, group_id, covariances
        )
        self._calculate_water_vibrational_rotational_entropy(
            vibrations, group_id, covariances
        )

        water_selection = universe.select_atoms("resname WAT")
        actual_water_residues = len(water_selection.residues)
        residue_names = {
            resname
            for res_dict in Sorient_dict.values()
            for resname in res_dict.keys()
            if resname.upper() in water_selection.residues.resnames
        }

        residue_group = "_".join(sorted(residue_names)) if residue_names else "WAT"
        self._data_logger.add_group_label(
            group_id, residue_group, actual_water_residues, len(water_selection.atoms)
        )

    def _calculate_water_orientational_entropy(self, Sorient_dict, group_id):
        """
        Aggregate orientational entropy for all water molecules into a single group.

        Parameters
        ----------
        Sorient_dict : dict
            Dictionary containing orientational entropy values per residue.
        group_id : int or str
            The group ID to which the water residues belong.
        covariances : object
            Covariance object.
        """
        for resid, resname_dict in Sorient_dict.items():
            for resname, values in resname_dict.items():
                if isinstance(values, list) and len(values) == 2:
                    Sor, count = values
                    self._data_logger.add_residue_data(
                        group_id, resname, "Water", "Orientational", count, Sor
                    )

    def _calculate_water_vibrational_translational_entropy(
        self, vibrations, group_id, covariances
    ):
        """
        Aggregate translational vibrational entropy for all water molecules.

        Parameters
        ----------
        vibrations : object
            Object containing translational entropy data (vibrations.translational_S).
        group_id : int or str
            The group ID for the water residues.
        covariances : object
            Covariance object.
        """

        for (solute_id, _), entropy in vibrations.translational_S.items():
            if isinstance(entropy, (list, np.ndarray)):
                entropy = float(np.sum(entropy))

            count = covariances.counts.get((solute_id, "WAT"), 1)
            resname = solute_id.rsplit("_", 1)[0] if "_" in solute_id else solute_id
            self._data_logger.add_residue_data(
                group_id, resname, "Water", "Transvibrational", count, entropy
            )

    def _calculate_water_vibrational_rotational_entropy(
        self, vibrations, group_id, covariances
    ):
        """
        Aggregate rotational vibrational entropy for all water molecules.

        Parameters
        ----------
        vibrations : object
            Object containing rotational entropy data (vibrations.rotational_S).
        group_id : int or str
            The group ID for the water residues.
        covariances : object
            Covariance object.
        """
        for (solute_id, _), entropy in vibrations.rotational_S.items():
            if isinstance(entropy, (list, np.ndarray)):
                entropy = float(np.sum(entropy))

            count = covariances.counts.get((solute_id, "WAT"), 1)

            resname = solute_id.rsplit("_", 1)[0] if "_" in solute_id else solute_id
            self._data_logger.add_residue_data(
                group_id, resname, "Water", "Rovibrational", count, entropy
            )
