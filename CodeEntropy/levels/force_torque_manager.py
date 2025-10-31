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
        self, data_container, bead, trans_axes, highest_level, force_partitioning=0.5
    ):
        """
        Function to calculate the mass weighted forces for a given bead.

        Args:
           data_container (MDAnalysis.Universe): Contains atomic positions and forces.
           bead : The part of the molecule to be considered.
           trans_axes (np.ndarray): The axes relative to which the forces are located.
           highest_level (bool): Is this the largest level of the length scale hierarchy
           force_partitioning (float): Factor to adjust force contributions to avoid
           over counting correlated forces, default is 0.5.

        Returns:
            weighted_force (np.ndarray): The mass-weighted sum of the forces in the
            bead.
        """

        forces_trans = np.zeros((3,))

        # Sum forces from all atoms in the bead
        for atom in bead.atoms:
            # update local forces in translational axes
            forces_local = np.matmul(trans_axes, data_container.atoms[atom.index].force)
            forces_trans += forces_local

        if highest_level:
            # multiply by the force_partitioning parameter to avoid double counting
            # of the forces on weakly correlated atoms
            # the default value of force_partitioning is 0.5 (dividing by two)
            forces_trans = force_partitioning * forces_trans

        # divide the sum of forces by the mass of the bead to get the weighted forces
        mass = bead.total_mass()

        # Check that mass is positive to avoid division by 0 or negative values inside
        # sqrt
        if mass <= 0:
            raise ValueError(
                f"Invalid mass value: {mass}. Mass must be positive to compute the "
                f"square root."
            )

        weighted_force = forces_trans / np.sqrt(mass)

        logger.debug(f"Weighted Force: {weighted_force}")

        return weighted_force

    def get_weighted_torques(
        self, data_container, bead, rot_axes, force_partitioning=0.5
    ):
        """
        Function to calculate the moment of inertia weighted torques for a given bead.

        This function computes torques in a rotated frame and then weights them using
        the moment of inertia tensor. To prevent numerical instability, it treats
        extremely small diagonal elements of the moment of inertia tensor as zero
        (since values below machine precision are effectively zero). This avoids
        unnecessary use of extended precision (e.g., float128).

        Additionally, if the computed torque is already zero, the function skips
        the division step, reducing unnecessary computations and potential errors.

        Parameters
        ----------
        data_container : object
            Contains atomic positions and forces.
        bead : object
            The part of the molecule to be considered.
        rot_axes : np.ndarray
            The axes relative to which the forces and coordinates are located.
        force_partitioning : float, optional
            Factor to adjust force contributions, default is 0.5.

        Returns
        -------
        weighted_torque : np.ndarray
            The mass-weighted sum of the torques in the bead.
        """

        torques = np.zeros((3,))
        weighted_torque = np.zeros((3,))

        for atom in bead.atoms:

            # update local coordinates in rotational axes
            coords_rot = (
                data_container.atoms[atom.index].position - bead.center_of_mass()
            )
            coords_rot = np.matmul(rot_axes, coords_rot)
            # update local forces in rotational frame
            forces_rot = np.matmul(rot_axes, data_container.atoms[atom.index].force)

            # multiply by the force_partitioning parameter to avoid double counting
            # of the forces on weakly correlated atoms
            # the default value of force_partitioning is 0.5 (dividing by two)
            forces_rot = force_partitioning * forces_rot

            # define torques (cross product of coordinates and forces) in rotational
            # axes
            torques_local = np.cross(coords_rot, forces_rot)
            torques += torques_local

        # divide by moment of inertia to get weighted torques
        # moment of inertia is a 3x3 tensor
        # the weighting is done in each dimension (x,y,z) using the diagonal
        # elements of the moment of inertia tensor
        moment_of_inertia = bead.moment_of_inertia()

        for dimension in range(3):
            # Skip calculation if torque is already zero
            if np.isclose(torques[dimension], 0):
                weighted_torque[dimension] = 0
                continue

            # Check for zero moment of inertia
            if np.isclose(moment_of_inertia[dimension, dimension], 0):
                raise ZeroDivisionError(
                    f"Attempted to divide by zero moment of inertia in dimension "
                    f"{dimension}."
                )

            # Check for negative moment of inertia
            if moment_of_inertia[dimension, dimension] < 0:
                raise ValueError(
                    f"Negative value encountered for moment of inertia: "
                    f"{moment_of_inertia[dimension, dimension]} "
                    f"Cannot compute weighted torque."
                )

            # Compute weighted torque
            weighted_torque[dimension] = torques[dimension] / np.sqrt(
                moment_of_inertia[dimension, dimension]
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
                        mol = entropy_manager._get_molecule_container(
                            reduced_atom, mol_id
                        )
                        for level in levels[mol_id]:
                            mol = entropy_manager._get_molecule_container(
                                reduced_atom, mol_id
                            )

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
                res = entropy_manager._run_manager.new_U_select_atom(
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
                    num_frames,
                    highest,
                    None if key not in force_avg["ua"] else force_avg["ua"][key],
                    None if key not in torque_avg["ua"] else torque_avg["ua"][key],
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
                num_frames,
                highest,
                None if force_avg[key][group_id] is None else force_avg[key][group_id],
                (
                    None
                    if torque_avg[key][group_id] is None
                    else torque_avg[key][group_id]
                ),
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
