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


class ForceTorqueManager:
    """ """

    def __init__(self):
        """
        Initializes the ForceTorqueManager with placeholders for level-related data,
        including translational and rotational axes, number of beads, and a
        general-purpose data container.
        """

    def get_weighted_forces(
        self, data_container, bead, trans_axes, highest_level, force_partitioning
    ):
        """
        Compute mass-weighted translational forces for a bead.

        The forces acting on all atoms belonging to the bead are first transformed
        into the provided translational reference frame and summed. If this bead
        corresponds to the highest level of a hierarchical coarse-graining scheme,
        the total force is scaled by a force-partitioning factor to avoid double
        counting forces from weakly correlated atoms.

        The resulting force vector is then normalized by the square root of the
        bead's total mass.

        Parameters
        ----------
        data_container : MDAnalysis.Universe
            Container holding atomic positions and forces.
        bead : object
            Molecular subunit whose atoms contribute to the force.
        trans_axes : np.ndarray
            Transformation matrix defining the translational reference frame.
        highest_level : bool
            Whether this bead is the highest level in the length-scale hierarchy.
            If True, force partitioning is applied.
        force_partitioning : float
            Scaling factor applied to forces to avoid over-counting correlated
            contributions (typically 0.5).

        Returns
        -------
        weighted_force : np.ndarray
            Mass-weighted translational force acting on the bead.

        Raises
        ------
        ValueError
            If the bead mass is zero or negative.
        """
        forces_trans = np.zeros((3,))

        for atom in bead.atoms:
            forces_local = np.matmul(trans_axes, data_container.atoms[atom.index].force)
            forces_trans += forces_local

        if highest_level:
            forces_trans = force_partitioning * forces_trans

        mass = bead.total_mass()

        if mass <= 0:
            raise ValueError(
                f"Invalid mass value: {mass}. Mass must be positive to compute the "
                f"square root."
            )

        weighted_force = forces_trans / np.sqrt(mass)

        logger.debug(f"Weighted Force: {weighted_force}")

        return weighted_force

    def get_weighted_torques(self, data_container, bead, rot_axes, force_partitioning):
        """
        Compute moment-of-inertia weighted torques for a bead.

        Atomic coordinates and forces are transformed into the provided rotational
        reference frame. Torques are computed as the cross product of position
        vectors (relative to the bead center of mass) and forces, with a
        force-partitioning factor applied to reduce over-counting of correlated
        atomic contributions.

        The total torque vector is then weighted by the square root of the bead's
        principal moments of inertia. Weighting is performed component-wise using
        the sorted eigenvalues of the moment of inertia tensor.

        To ensure numerical stability:
        - Torque components that are effectively zero are skipped.
        - Zero moments of inertia result in zero weighted torque with a warning.
        - Negative moments of inertia raise an error.

        Parameters
        ----------
        data_container : object
            Container holding atomic positions and forces.
        bead : object
            Molecular subunit whose atoms contribute to the torque.
        rot_axes : np.ndarray
            Transformation matrix defining the rotational reference frame.
        force_partitioning : float
            Scaling factor applied to forces to avoid over-counting correlated
            contributions (typically 0.5).

        Returns
        -------
        weighted_torque : np.ndarray
            Moment-of-inertia weighted torque acting on the bead.

        Raises
        ------
        ValueError
            If a negative principal moment of inertia is encountered.
        """
        torques = np.zeros((3,))
        weighted_torque = np.zeros((3,))
        moment_of_inertia = np.zeros(3)

        for atom in bead.atoms:
            coords_rot = (
                data_container.atoms[atom.index].position - bead.center_of_mass()
            )
            coords_rot = np.matmul(rot_axes, coords_rot)
            forces_rot = np.matmul(rot_axes, data_container.atoms[atom.index].force)

            forces_rot = force_partitioning * forces_rot

            torques_local = np.cross(coords_rot, forces_rot)
            torques += torques_local

        eigenvalues, _ = np.linalg.eig(bead.moment_of_inertia())
        moments_of_inertia = sorted(eigenvalues, reverse=True)

        for dimension in range(3):
            if np.isclose(torques[dimension], 0):
                weighted_torque[dimension] = 0
                continue

            if np.isclose(moments_of_inertia[dimension], 0):
                weighted_torque[dimension] = 0
                logger.warning("Zero moment of inertia. Setting torque to 0")
                continue

            if moments_of_inertia[dimension] < 0:
                raise ValueError(
                    f"Negative value encountered for moment of inertia: "
                    f"{moment_of_inertia[dimension]} "
                    f"Cannot compute weighted torque."
                )

            weighted_torque[dimension] = torques[dimension] / np.sqrt(
                moments_of_inertia[dimension]
            )

        logger.debug(f"Weighted Torque: {weighted_torque}")

        return weighted_torque

    def build_covariance_matrices(
        self,
        entropy_manager,
        reduced_atom,
        levels,
        groups,
        start,
        end,
        step,
        number_frames,
        force_partitioning,
    ):
        """
        Construct average force and torque covariance matrices for all molecules and
        entropy levels.

        Parameters
        ----------
        entropy_manager : EntropyManager
            Instance of the EntropyManager.
        reduced_atom : Universe
            The reduced atom selection.
        levels : dict
            Dictionary mapping molecule IDs to lists of entropy levels.
        groups : dict
            Dictionary mapping group IDs to lists of molecule IDs.
        start : int
            Start frame index.
        end : int
            End frame index.
        step : int
            Step size for frame iteration.
        number_frames : int
            Total number of frames to process.
        force_partitioning : float
            Factor to adjust force contributions, default is 0.5.


        Returns
        -------
        tuple
            force_avg : dict
                Averaged force covariance matrices by entropy level.
            torque_avg : dict
                Averaged torque covariance matrices by entropy level.
        """
        number_groups = len(groups)

        force_avg = {
            "ua": {},
            "res": [None] * number_groups,
            "poly": [None] * number_groups,
        }
        torque_avg = {
            "ua": {},
            "res": [None] * number_groups,
            "poly": [None] * number_groups,
        }

        total_steps = len(reduced_atom.trajectory[start:end:step])
        total_items = (
            sum(len(levels[mol_id]) for mols in groups.values() for mol_id in mols)
            * total_steps
        )

        frame_counts = {
            "ua": {},
            "res": np.zeros(number_groups, dtype=int),
            "poly": np.zeros(number_groups, dtype=int),
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[title]}", justify="right"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
        ) as progress:

            task = progress.add_task(
                "[green]Processing...",
                total=total_items,
                title="Starting...",
            )

            indices = list(range(number_frames))
            for time_index, _ in zip(indices, reduced_atom.trajectory[start:end:step]):
                for group_id, molecules in groups.items():
                    for mol_id in molecules:
                        mol = self._universe_operations.get_molecule_container(
                            reduced_atom, mol_id
                        )
                        for level in levels[mol_id]:
                            resname = mol.atoms[0].resname
                            resid = mol.atoms[0].resid
                            segid = mol.atoms[0].segid

                            mol_label = f"{resname}_{resid} (segid {segid})"

                            progress.update(
                                task,
                                title=f"Building covariance matrices | "
                                f"Timestep {time_index} | "
                                f"Molecule: {mol_label} | "
                                f"Level: {level}",
                            )

                            self.update_force_torque_matrices(
                                entropy_manager,
                                mol,
                                group_id,
                                level,
                                levels[mol_id],
                                time_index,
                                number_frames,
                                force_avg,
                                torque_avg,
                                frame_counts,
                                force_partitioning,
                            )

                            progress.advance(task)

        return force_avg, torque_avg, frame_counts

    def update_force_torque_matrices(
        self,
        entropy_manager,
        mol,
        group_id,
        level,
        level_list,
        time_index,
        num_frames,
        force_avg,
        torque_avg,
        frame_counts,
        force_partitioning,
    ):
        """
        Update the running averages of force and torque covariance matrices
        for a given molecule and entropy level.

        This function computes the force and torque covariance matrices for the
        current frame and updates the existing averages in-place using the incremental
        mean formula:

            new_avg = old_avg + (value - old_avg) / n

        where n is the number of frames processed so far for that molecule/level
        combination. This ensures that the averages are maintained without storing
        all previous frame data.

        Parameters
        ----------
        entropy_manager : EntropyManager
            Instance of the EntropyManager.
        mol : AtomGroup
            The molecule to process.
        group_id : int
            Index of the group to which the molecule belongs.
        level : str
            Current entropy level ("united_atom", "residue", or "polymer").
        level_list : list
            List of entropy levels for the molecule.
        time_index : int
            Index of the current frame relative to the start of the trajectory slice.
        num_frames : int
            Total number of frames to process.
        force_avg : dict
            Dictionary holding the running average force matrices, keyed by entropy
            level.
        torque_avg : dict
            Dictionary holding the running average torque matrices, keyed by entropy
            level.
        frame_counts : dict
            Dictionary holding the count of frames processed for each molecule/level
            combination.
        force_partitioning : float
         Factor to adjust force contributions, default is 0.5.
        Returns
        -------
        None
            Updates are performed in-place on `force_avg`, `torque_avg`, and
            `frame_counts`.
        """
        highest = level == level_list[-1]

        # United atom level calculations are done separately for each residue
        # This allows information per residue to be output and keeps the
        # matrices from becoming too large
        if level == "united_atom":
            for res_id, residue in enumerate(mol.residues):
                key = (group_id, res_id)
                res = self._universe_operations.new_U_select_atom(
                    mol, f"index {residue.atoms.indices[0]}:{residue.atoms.indices[-1]}"
                )

                # This is to get MDAnalysis to get the information from the
                # correct frame of the trajectory
                res.trajectory[time_index]

                # Build the matrices, adding data from each timestep
                # Being careful for the first timestep when data has not yet
                # been added to the matrices
                f_mat, t_mat = self.get_matrices(
                    res,
                    level,
                    highest,
                    None if key not in force_avg["ua"] else force_avg["ua"][key],
                    None if key not in torque_avg["ua"] else torque_avg["ua"][key],
                    force_partitioning,
                )

                if key not in force_avg["ua"]:
                    force_avg["ua"][key] = f_mat.copy()
                    torque_avg["ua"][key] = t_mat.copy()
                    frame_counts["ua"][key] = 1
                else:
                    frame_counts["ua"][key] += 1
                    n = frame_counts["ua"][key]
                    force_avg["ua"][key] += (f_mat - force_avg["ua"][key]) / n
                    torque_avg["ua"][key] += (t_mat - torque_avg["ua"][key]) / n

        elif level in ["residue", "polymer"]:
            # This is to get MDAnalysis to get the information from the
            # correct frame of the trajectory
            mol.trajectory[time_index]

            key = "res" if level == "residue" else "poly"

            # Build the matrices, adding data from each timestep
            # Being careful for the first timestep when data has not yet
            # been added to the matrices
            f_mat, t_mat = self.get_matrices(
                mol,
                level,
                highest,
                None if force_avg[key][group_id] is None else force_avg[key][group_id],
                (
                    None
                    if torque_avg[key][group_id] is None
                    else torque_avg[key][group_id]
                ),
                force_partitioning,
            )

            if force_avg[key][group_id] is None:
                force_avg[key][group_id] = f_mat.copy()
                torque_avg[key][group_id] = t_mat.copy()
                frame_counts[key][group_id] = 1
            else:
                frame_counts[key][group_id] += 1
                n = frame_counts[key][group_id]
                force_avg[key][group_id] += (f_mat - force_avg[key][group_id]) / n
                torque_avg[key][group_id] += (t_mat - torque_avg[key][group_id]) / n

        return frame_counts
