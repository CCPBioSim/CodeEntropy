"""Axes utilities for entropy calculations.

This module contains the :class:`AxesCalculator`, a geometry-focused helper used by
the entropy pipeline to compute translational and rotational axes, centres, and
moments of inertia at different hierarchy levels (residue / united-atom).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from MDAnalysis.lib.mdamath import make_whole

logger = logging.getLogger(__name__)


class AxesCalculator:
    """Compute translation/rotation axes and inertia utilities used by entropy.

    Manages the structural and dynamic levels involved in entropy calculations.
    This includes selecting relevant levels, computing axes for translation and
    rotation, and handling bead-based representations of molecular systems.

    Provides utility methods to:

    - Extract averaged positions.
    - Convert coordinates to spherical systems (future/legacy scope).
    - Compute axes used to rotate forces around.
    - Compute custom moments of inertia.
    - Manipulate vectors under periodic boundary conditions (PBC).
    - Construct custom moment-of-inertia tensors and principal axes.

    Notes:
        This class deliberately does not:

        - Compute weighted forces/torques (that belongs in ForceTorqueCalculator).
        - Build covariances.
        - Compute entropies.
    """

    def __init__(self) -> None:
        """Initialize the AxesCalculator.

        The original implementation stored a few placeholders for level-related
        data (axes, bead counts, etc.). In the current design, AxesCalculator is a
        stateless helper, but we keep the attributes for compatibility and
        debugging/extension.

        Attributes:
            data_container: Optional container used by legacy workflows.
            _levels: Optional levels list (legacy/placeholder).
            _trans_axes: Optional cached translation axes (legacy/placeholder).
            _rot_axes: Optional cached rotation axes (legacy/placeholder).
            _number_of_beads: Optional bead count (legacy/placeholder).
        """
        self.data_container = None
        self._levels = None
        self._trans_axes = None
        self._rot_axes = None
        self._number_of_beads = None

    def get_residue_axes(
        self, data_container, index: int, relative_index: int, residue=None
    ):
        """Compute residue-level translational and rotational axes.

        The translational and rotational axes at the residue level.

        - Identify the residue (either provided or selected by `resindex index`).
        - Determine whether the residue is bonded to neighbouring residues
          (previous/next in sequence) using MDAnalysis bonded selections.
        - If there are *no* bonds to other residues:
            * Use a custom principal axes, from a moment-of-inertia (MOI) tensor
              that uses positions of heavy atoms only, but includes masses of
              heavy atom + bonded hydrogens.
            * Set translational axes equal to rotational axes (as per the original
              code convention).
        - If bonded to only one other residue:
            * Translational axes are principal axes of data_container.
            * Find edge heavy atom (i.e. heavy atoms bonded to neighbour residue).
              Find all heavy atoms bonded to edge heavy atom and compute their average
              position.
              Find all other heavy atoms in residue and compute their average position.
              The three points are now used to obtain determine residue rotational axes.
              (see get_residue_custom_axes)
        - If bonded to at least two other residues:
            * Translational axes are principal axes of data_container.
            * Find edge heavy atoms (i.e. heavy atoms bonded to neighbour residues)
              and find the shortest chain between them: the backbone. Edge
              atoms + backbone COM are used to determine residue rotational axes.
              (see get_residue_custom_axes).
        Compute a custom MOI, using heavy atom positions and
          heavy atom + hydrogen masses.

        Args:
            data_container (MDAnalysis.Universe or AtomGroup):
                Molecule and trajectory data (the fragment/molecule container).
            index (int):
                Residue index (resindex) within `data_container`.
            relative_index (int):
                Index of first residue within 'data_container'.
                This is used to obtain index in MDA Universe for atom selections.
            residue (MDAnalysis.AtomGroup, optional):
                If provided, this residue selection will be used rather than
                selecting again.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - trans_axes: Translational axes array of shape (3, 3).
                - rot_axes: Rotational axes array of shape (3, 3).
                - center: Center of mass array of shape (3,).
                - moment_of_inertia: Principal moments array of shape (3,).

        Raises:
            ValueError:
                If the residue selection is empty.
        """
        # TODO refine selection so that it will work for branched polymers

        index_prev = index + relative_index - 1
        index_next = index + relative_index + 1

        if residue is None:
            residue = data_container.select_atoms(f"resindex {index + relative_index}")
            # residue of interest
        if len(residue) == 0:
            raise ValueError(
                f"Empty residue selection for resindex={index + relative_index}"
            )

        edge_atom_set = data_container.atoms.select_atoms(
            f"resindex {index + relative_index} and "
            f"(bonded resindex {index_prev} or "
            f"resindex {index_next})"
        )

        uas = residue.select_atoms("mass 2 to 999")
        ua_masses = self.get_UA_masses(residue)

        if len(edge_atom_set) == 0:
            moi_tensor = self.get_moment_of_inertia_tensor(
                center_of_mass=np.array(residue.center_of_mass()),
                positions=uas.positions,
                masses=ua_masses,
                dimensions=data_container.dimensions[:3],
            )
            rot_axes, moment_of_inertia = self.get_custom_principal_axes(moi_tensor)
            trans_axes = rot_axes  # per original convention
            rot_center = np.array(residue.center_of_mass())
        else:
            make_whole(data_container.atoms)
            trans_axes = data_container.atoms.principal_axes()

            if len(edge_atom_set) == 1:
                edge_atom = edge_atom_set[0]
                bonded_atoms = uas.select_atoms(f"bonded index {edge_atom.index}")
                # find the average position of heavy atoms bonded to edge atom
                average_bonded_atom = np.zeros(3)
                for atom in bonded_atoms:
                    average_bonded_atom += atom.position
                average_bonded_atom /= len(bonded_atoms)
                # find the average position of all other heavy atoms in residue
                other_atoms = []
                for atom in uas:
                    if atom != edge_atom and atom not in bonded_atoms:
                        other_atoms.append(atom)
                average_other_atoms = np.zeros(3)
                for atom in other_atoms:
                    average_other_atoms += atom.position
                average_other_atoms /= len(other_atoms)
                rot_center, rot_axes = self.get_residue_custom_axes(
                    [edge_atom.position, average_other_atoms], average_bonded_atom
                )

            else:
                edges = [edge_atom_set[0].position, edge_atom_set[1].position]
                backbone = self.get_chain(residue, edge_atom_set[0], edge_atom_set[1])
                backbone_center = np.zeros(3)
                for heavy_atom in backbone:
                    backbone_center += heavy_atom.position
                backbone_center = backbone_center / len(backbone)
                rot_center, rot_axes = self.get_residue_custom_axes(
                    edges, backbone_center
                )

            moment_of_inertia = self.get_custom_residue_moment_of_inertia(
                center_of_mass=rot_center,
                positions=uas.positions,
                masses=ua_masses,
                custom_rot_axes=rot_axes,
                dimensions=data_container.dimensions[:3],
            )
        return trans_axes, rot_axes, rot_center, moment_of_inertia

    def get_residue_axes_from_topology(
        self,
        *,
        u,
        mol,
        residue_atoms,
        topology,
        box: np.ndarray | None,
    ):
        """Compute residue axes using cached static topology.

        This is the cached-index equivalent of ``get_residue_axes``. It keeps
        all frame-dependent numerical work frame-local, but avoids repeated
        MDAnalysis selections for residue heavy atoms, UA masses, and neighbour
        bond discovery.

        Args:
            u: Current-frame universe used to resolve cached atom indices.
            mol: Current-frame molecule fragment.
            residue_atoms: AtomGroup for the residue in the current frame.
            topology: Cached ``ResidueAxesTopology`` for this residue.
            box: Current periodic box lengths. If omitted, ``u.dimensions`` is used.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - trans_axes: Translational axes, shape ``(3, 3)``.
                - rot_axes: Rotational axes, shape ``(3, 3)``.
                - center: Residue centre, shape ``(3,)``.
                - moment_of_inertia: Principal moments, shape ``(3,)``.
        """
        dimensions = (
            np.asarray(box, dtype=float)
            if box is not None
            else np.asarray(u.dimensions[:3], dtype=float)
        )
        center = residue_atoms.center_of_mass(unwrap=True)

        if not topology.has_neighbor_bonds:
            heavy_atoms = u.atoms[topology.residue_heavy_indices]
            moment_of_inertia_tensor = self.get_moment_of_inertia_tensor(
                center_of_mass=center,
                positions=heavy_atoms.positions,
                masses=topology.residue_ua_masses,
                dimensions=dimensions,
            )
            rot_axes, moment_of_inertia = self.get_custom_principal_axes(
                moment_of_inertia_tensor
            )
            trans_axes = rot_axes
        else:
            make_whole(mol.atoms)
            trans_axes = mol.atoms.principal_axes()
            rot_axes, moment_of_inertia = self.get_vanilla_axes(residue_atoms)
            center = residue_atoms.center_of_mass(unwrap=True)

        return trans_axes, rot_axes, center, moment_of_inertia

    def get_UA_axes(self, data_container, index: int, res_position):
        """Compute united-atom-level translational and rotational axes.

        The translational and rotational axes at the united-atom level.

        This preserves the original behaviour and its rationale:

        - Translational axes:
            Use the same approach as residue level rotational.
            Identify residue of interest and neighbours, then select
            edge heavy atoms (i.e. heavy atoms bonded to neighbour residues).
            - If there are *no* bonds to other residues, use a custom principal axes
            from a moment-of-inertia (MOI) tensor that uses positions of heavy atoms
            only, but includes masses of heavy atom + bonded hydrogens.
            - If bonded to only one other residue, find edge heavy atom
            (i.e. heavy atom bonded to neighbour residue). Find all heavy atoms
            bonded to edge heavy atom and compute their average position.
            Find all other heavy atoms in residue and compute their average position.
            The three points are now used to obtain determine residue rotational axes.
            (see get_residue_custom_axes)
            - If bonded to at least two other residues, find edge heavy atoms
            (i.e. heavy atoms bonded to neighbour residues) and find the shortest
            chain between them: the backbone. Edge atoms + backbone COM are used
            to determine residue rotational axes. (see get_residue_custom_axes).

        - Rotational axes:
            Identify heavy atoms in the residue/molecule of interest and choose
            the `index`-th heavy atom (where index corresponds to the bead index).
            Use bonded topology around that heavy atom to determine UA rotational
            axes (see :meth:`get_bonded_axes`). Compute a custom MOI tensor.

        Args:
            data_container (MDAnalysis.Universe or AtomGroup):
                Molecule and trajectory data.
            index (int):
                Bead index (ordinal among heavy atoms).
            res_position: where the residue of interest is
                in data_container
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - trans_axes: Translational axes (3, 3).
                - rot_axes: Rotational axes (3, 3).
                - center: Rotation centre (3,) (heavy atom position).
                - moment_of_inertia: (3,) moments for the UA around rot_axes.

        Raises:
            IndexError:
                If `index` does not correspond to an existing heavy atom.
            ValueError:
                If axis construction fails.
        """

        index = int(index)  # UA bead index
        heavy_atoms = data_container.select_atoms("mass 2 to 999")
        # use the same customPI trans axes as the residue level
        if len(heavy_atoms) > 1:
            if len(data_container.residues) == 1:
                # only the one residue => use principal axes
                residue = data_container
                trans_center = data_container.atoms.center_of_mass(unwrap=True)
                trans_axes = data_container.atoms.principal_axes()
                residue_heavy_atoms = heavy_atoms
            else:
                # residue of interest has at least one neighbour
                if res_position == -1 or res_position == 1:
                    # look at a terminal residue
                    if res_position == -1:
                        # first residue
                        residue = data_container.residues[0]
                        resindex = residue.resindex
                        resindex_next = resindex + 1
                        edge_atom = data_container.select_atoms(
                            f"resindex {resindex} and bonded resindex {resindex_next}"
                        )
                    else:
                        # last residue
                        residue = data_container.residues[1]
                        resindex = residue.resindex
                        resindex_prev = resindex - 1
                        edge_atom = data_container.select_atoms(
                            f"resindex {resindex} and bonded resindex {resindex_prev}"
                        )
                    residue_heavy_atoms = residue.atoms.select_atoms("mass 2 to 999")
                    bonded_atoms = residue_heavy_atoms.select_atoms(
                        f"bonded index {edge_atom[0].index}"
                    )
                    # find the average position of heavy atoms bonded to edge atom
                    average_bonded_atom = np.zeros(3)
                    for atom in bonded_atoms:
                        average_bonded_atom += atom.position
                    average_bonded_atom /= len(bonded_atoms)
                    # find the average position of all other heavy atoms in residue
                    other_atoms = []
                    for atom in residue_heavy_atoms:
                        if atom != edge_atom and atom not in bonded_atoms:
                            other_atoms.append(atom)
                    average_other_atoms = np.zeros(3)
                    for atom in other_atoms:
                        average_other_atoms += atom.position
                    average_other_atoms /= len(other_atoms)
                    trans_center, trans_axes = self.get_residue_custom_axes(
                        [edge_atom.positions[0], average_other_atoms],
                        average_bonded_atom,
                    )
                else:
                    # between 2 residues
                    residue = data_container.residues[1]
                    resindex = residue.resindex
                    resindex_next = resindex + 1
                    resindex_prev = resindex - 1
                    residue_heavy_atoms = residue.atoms.select_atoms("mass 2 to 999")
                    edge_set = data_container.select_atoms(
                        f"resindex {resindex} and "
                        f"(bonded resindex {resindex_prev} or "
                        f"resindex {resindex_next})"
                    )

                    edges = edge_set.positions
                    backbone = self.get_chain(residue, edge_set[0], edge_set[1])
                    backbone_center = np.zeros(3)
                    for heavy_atom in backbone:
                        backbone_center += heavy_atom.position
                    backbone_center = backbone_center / len(backbone)

                    trans_center, trans_axes = self.get_residue_custom_axes(
                        edges, backbone_center
                    )

            # look for heavy atoms in residue of interest
            heavy_atom_indices = []
            for atom in residue_heavy_atoms:
                heavy_atom_indices.append(atom.index)
            # we find the nth heavy atom
            # where n is the bead index
            heavy_atom_index = heavy_atom_indices[index]
            heavy_atom = residue.atoms.select_atoms(f"index {heavy_atom_index}")[0]
            rot_center = heavy_atom.position
            rot_axes, moment_of_inertia = self.get_bonded_axes(
                system=data_container,
                atom=heavy_atom,
                dimensions=data_container.dimensions[:3],
            )

        else:
            # 1 heavy atom in the data_container
            heavy_atom = heavy_atoms[0]
            # trans and rot centres are centre of mass
            rot_center = data_container.center_of_mass()
            rot_axes, moment_of_inertia = self.get_bonded_axes(
                system=data_container,
                atom=heavy_atom,
                dimensions=data_container.dimensions[:3],
            )
            trans_center = rot_center
            # principal axes
            trans_axes = rot_axes

        if trans_axes is None:
            raise ValueError("Unable to compute translation axes for UA bead.")

        if rot_axes is None or moment_of_inertia is None:
            raise ValueError("Unable to compute bonded axes for UA bead.")

        logger.debug("Translational Axes: %s", trans_axes)
        logger.debug("Rotational Axes: %s", rot_axes)
        logger.debug("Translational center: %s", trans_center)
        logger.debug("Rotational center: %s", rot_center)
        logger.debug("Moment of Inertia: %s", moment_of_inertia)

        return trans_axes, rot_axes, rot_center, moment_of_inertia

    def get_UA_axes_from_topology(
        self,
        *,
        u,
        residue_atoms,
        topology,
        box: np.ndarray | None,
    ):
        """Compute UA axes using cached static topology.

        This is the cached-index equivalent of ``get_UA_axes``. It preserves the
        frame-dependent numerical calculations, but avoids repeated MDAnalysis
        selection strings for heavy atoms, bonded atoms, and UA masses.

        Args:
            u: Current-frame universe.
            residue_atoms: AtomGroup for the parent residue in the current frame.
            topology: Cached ``UAAxesTopology`` for this UA bead.
            box: Current periodic box lengths. If omitted, ``u.dimensions`` is used.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - trans_axes: Translational axes, shape ``(3, 3)``.
                - rot_axes: Rotational axes, shape ``(3, 3)``.
                - center: Rotation centre, shape ``(3,)``.
                - moment_of_inertia: Principal moments, shape ``(3,)``.

        Raises:
            ValueError: If cached bonded-axis construction fails.
        """
        dimensions = (
            np.asarray(box, dtype=float)
            if box is not None
            else np.asarray(u.dimensions[:3], dtype=float)
        )

        heavy_atoms = u.atoms[topology.residue_heavy_indices]
        heavy_atom = u.atoms[int(topology.heavy_atom_index)]

        if len(heavy_atoms) > 1:
            center = residue_atoms.center_of_mass(unwrap=True)
            moment_of_inertia_tensor = self.get_moment_of_inertia_tensor(
                center_of_mass=center,
                positions=heavy_atoms.positions,
                masses=topology.residue_ua_masses,
                dimensions=dimensions,
            )
            trans_axes, _moment_of_inertia = self.get_custom_principal_axes(
                moment_of_inertia_tensor
            )
        else:
            make_whole(residue_atoms)
            trans_axes = residue_atoms.principal_axes()

        center = heavy_atom.position
        rot_axes, moment_of_inertia = self.get_bonded_axes_from_topology(
            u=u,
            heavy_atom=heavy_atom,
            topology=topology,
            dimensions=dimensions,
        )
        if rot_axes is None or moment_of_inertia is None:
            raise ValueError("Unable to compute bonded axes for cached UA bead.")

        logger.debug("Translational Axes: %s", trans_axes)
        logger.debug("Rotational Axes: %s", rot_axes)
        logger.debug("Center: %s", center)
        logger.debug("Moment of Inertia: %s", moment_of_inertia)

        return trans_axes, rot_axes, center, moment_of_inertia

    def get_bonded_axes_from_topology(
        self,
        *,
        u,
        heavy_atom,
        topology,
        dimensions: np.ndarray,
    ):
        """Compute UA bonded axes using cached bonded atom indices.

        This mirrors ``get_bonded_axes`` but receives precomputed bonded atom
        memberships from ``UAAxesTopology`` instead of rediscovering them with
        MDAnalysis selection strings inside the frame loop.

        Args:
            u: Current-frame universe.
            heavy_atom: Current-frame heavy atom for the UA bead.
            topology: Cached ``UAAxesTopology`` for the UA bead.
            dimensions: Simulation box lengths, shape ``(3,)``.

        Returns:
            Tuple[np.ndarray | None, np.ndarray | None]:
                - custom_axes: Custom rotation axes, shape ``(3, 3)``, or ``None``.
                - custom_moment_of_inertia: Principal moments, shape ``(3,)``, or
                  ``None``.
        """
        if not heavy_atom.mass > 1.1:
            return None, None

        custom_moment_of_inertia = None
        custom_axes = None

        heavy_bonded = u.atoms[topology.bonded_heavy_indices]
        light_bonded = u.atoms[topology.bonded_light_indices]
        ua = u.atoms[topology.ua_atom_indices]
        ua_all = u.atoms[topology.ua_all_atom_indices]

        if len(heavy_bonded) == 0:
            custom_axes, custom_moment_of_inertia = self.get_vanilla_axes(ua_all)

        if len(heavy_bonded) == 1 and len(light_bonded) == 0:
            custom_axes = self.get_custom_axes(
                a=heavy_atom.position,
                b_list=[heavy_bonded[0].position],
                c=np.zeros(3),
                dimensions=dimensions,
            )

        if len(heavy_bonded) == 1 and len(light_bonded) >= 1:
            custom_axes = self.get_custom_axes(
                a=heavy_atom.position,
                b_list=[heavy_bonded[0].position],
                c=light_bonded[0].position,
                dimensions=dimensions,
            )

        if len(heavy_bonded) >= 2:
            custom_axes = self.get_custom_axes(
                a=heavy_atom.position,
                b_list=heavy_bonded.positions,
                c=heavy_bonded[1].position,
                dimensions=dimensions,
            )

        if custom_axes is None:
            return None, None

        if custom_moment_of_inertia is None:
            custom_moment_of_inertia = self.get_custom_moment_of_inertia(
                UA=ua,
                custom_rotation_axes=custom_axes,
                center_of_mass=heavy_atom.position,
                dimensions=dimensions,
            )

        custom_axes = self.get_flipped_axes(
            ua,
            custom_axes,
            heavy_atom.position,
            dimensions,
        )

        return custom_axes, custom_moment_of_inertia

    def get_residue_custom_axes(self, edges, center):
        """
        Compute rotation axes at the residue level, given
        two edge atoms of the residue (E1+E2),
        and the centre of geometry of backbone atoms
        that are not edges (C).
        x axis is O-E1
        y axis is O-C (perpendicular to O-E1 in the
        same plane as E2)
        z axis is perpendicular to the two other axes

        ::

                    C
                    |
                    |
            E1 ---- O --- E2

        Args:
            edges: (2,3) positions of two edge atoms
            center: (3,) coordinates of the inner backbone
            centre of geometry

        Returns:
            rot_center: (3,) rotation centre,
            lies on the E1-E2 vector
            rot_axes: (3,3) rotation axes of residue
        """
        first_edge_centre_of_geometry_vector = center - edges[0]
        # look for projection of E1-O onto E1-E2 (E1-C)
        first_edge_second_edge_vector = edges[1] - edges[0]
        first_edge_origin_vector = (
            np.dot(first_edge_second_edge_vector, first_edge_centre_of_geometry_vector)
            / (np.linalg.norm(first_edge_second_edge_vector) ** 2)
        ) * first_edge_second_edge_vector
        x_axis = -first_edge_origin_vector
        # O-C = O-E1 + E1-C
        origin_centre_of_geometry_vector = (
            -first_edge_origin_vector + first_edge_centre_of_geometry_vector
        )
        y_axis = origin_centre_of_geometry_vector
        z_axis = np.cross(x_axis, y_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis /= np.linalg.norm(z_axis)
        rot_axes = np.array([x_axis, y_axis, z_axis])
        rot_center = first_edge_origin_vector + edges[0]
        return rot_center, rot_axes

    def get_bonded_axes(self, system, atom, dimensions: np.ndarray):
        r"""Compute UA rotational axes from bonded topology around a heavy atom.

        For a given heavy atom, use its bonded atoms to get the axes for rotating
        forces around. Few cases for choosing united atom axes, which are dependent
        on the bonds to the atom:

        ::

            X -- H = bonded to zero or more light atom/s (case1)

            X -- R = bonded to one heavy atom (case2)

            R -- X -- H = bonded to one heavy and at least one light atom (case3)

            R1 -- X -- R2 = bonded to two heavy atoms (case4)

            R1 -- X -- R2 = bonded to more than two heavy atoms (case5)
                  |
                  R3

        Note that axis2 is calculated by taking the cross product between axis1 and
        the vector chosen for each case, dependent on bonding:

        - case1: if all the bonded atoms are hydrogens, use the principal axes.

        - case2: use XR vector as axis1, arbitrary axis2.

        - case3: use XR vector as axis1, vector XH to calculate axis2

        - case4: use vector XR1 as axis1, and XR2 to calculate axis2

        - case5: get the sum of all XR normalised vectors as axis1, then use vector
          R1R2 to calculate axis2

        axis3 is always the cross product of axis1 and axis2.

        Args:
            system:
                MDAnalysis selection containing all atoms in current frame.
            atom:
                MDAnalysis Atom for the heavy atom.
            dimensions:
                Simulation box dimensions (3,).

        Returns:
            Tuple[np.ndarray | None, np.ndarray | None]:
                - custom_axes: Custom axes (3, 3), or None if atom is not heavy.
                - custom_moment_of_inertia: (3,) moment of inertia around axes.

        Notes:
            If custom_moment_of_inertia is not produced by the chosen method, it is
            computed using :meth:`get_custom_moment_of_inertia` with the heavy atom
            as COM (matching original behaviour).
        """
        # check atom is a heavy atom
        if not atom.mass > 1.1:
            return None, None

        custom_moment_of_inertia = None
        custom_axes = None

        heavy_bonded, light_bonded = self.find_bonded_atoms(atom.index, system)
        ua = atom + light_bonded
        ua_all = atom + heavy_bonded + light_bonded

        # case1
        if len(heavy_bonded) == 0:
            custom_axes, custom_moment_of_inertia = self.get_vanilla_axes(ua_all)

        # case2
        if len(heavy_bonded) == 1 and len(light_bonded) == 0:
            custom_axes = self.get_custom_axes(
                a=atom.position,
                b_list=[heavy_bonded[0].position],
                c=np.zeros(3),
                dimensions=dimensions,
            )

        # case3
        if len(heavy_bonded) == 1 and len(light_bonded) >= 1:
            custom_axes = self.get_custom_axes(
                a=atom.position,
                b_list=[heavy_bonded[0].position],
                c=light_bonded[0].position,
                dimensions=dimensions,
            )

        # case4 (not used in original 2019 code; case5 used instead)
        # case5
        if len(heavy_bonded) >= 2:
            custom_axes = self.get_custom_axes(
                a=atom.position,
                b_list=heavy_bonded.positions,
                c=heavy_bonded[1].position,
                dimensions=dimensions,
            )

        if custom_axes is None:
            return None, None

        if custom_moment_of_inertia is None:
            custom_moment_of_inertia = self.get_custom_moment_of_inertia(
                UA=ua,
                custom_rotation_axes=custom_axes,
                center_of_mass=atom.position,
                dimensions=dimensions,
            )

        # flip axes to face correct way wrt COM
        custom_axes = self.get_flipped_axes(ua, custom_axes, atom.position, dimensions)

        return custom_axes, custom_moment_of_inertia

    def find_bonded_atoms(self, atom_idx: int, system):
        """Find bonded heavy and hydrogen atoms for a given atom.

        Args:
            atom_idx: Atom index to find bonded atoms for.
            system: MDAnalysis selection containing all atoms in current frame.

        Returns:
            Tuple[AtomGroup, AtomGroup]:
                - bonded_heavy_atoms: bonded heavy atoms (mass 2 to 999)
                - bonded_H_atoms: bonded hydrogen atoms (mass 1 to 1.1)
        """
        bonded_atoms = system.select_atoms(f"bonded index {atom_idx}")
        bonded_heavy_atoms = bonded_atoms.select_atoms("mass 2 to 999")
        bonded_H_atoms = bonded_atoms.select_atoms("mass 1 to 1.1")
        return bonded_heavy_atoms, bonded_H_atoms

    def get_vanilla_axes(self, molecule):
        """Get principal axes and sorted principal moments (vanilla method).

        Compute the principal axes and moments of inertia for a molecule using
        MDAnalysis built-in functionality.

        The original description is preserved:
        - The molecule is made whole to ensure correct handling of PBC.
        - The moments are obtained by diagonalising the moment of inertia tensor.
        - Eigenvalues are returned sorted from largest to smallest magnitude.

        Args:
            molecule (MDAnalysis.core.groups.AtomGroup):
                AtomGroup representing the molecule/bead.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - principal_axes: (3, 3) axes.
                - moment_of_inertia: (3,) moments sorted descending by absolute value.
        """
        moment_of_inertia_tensor = molecule.moment_of_inertia(unwrap=True)
        make_whole(molecule.atoms)
        principal_axes = molecule.principal_axes()

        eigenvalues, _ = np.linalg.eig(moment_of_inertia_tensor)
        order = np.argsort(np.abs(eigenvalues))[::-1]
        moment_of_inertia = eigenvalues[order]

        return principal_axes, moment_of_inertia

    def get_custom_axes(
        self,
        a: np.ndarray,
        b_list: Sequence[np.ndarray],
        c: np.ndarray,
        dimensions: np.ndarray,
    ) -> np.ndarray:
        r"""Compute custom rotation axes from bonded vectors (PBC-aware).

        For atoms a, b_list and c, calculate the axis to rotate forces around:

        - axis1: use the normalised vector ab as axis1. If there is more than one
          bonded heavy atom (HA), average over all the normalised vectors
          calculated from b_list and use this as axis1. b_list contains all the
          bonded heavy atom coordinates.

        - axis2: use the cross product of normalised vector ac and axis1 as axis2.
          If there are more than two bonded heavy atoms, then use normalised vector
          b[0]c to cross product with axis1. This gives the axis perpendicular
          to axis1.

        - axis3: the cross product of axis1 and axis2, which is perpendicular to
          axis1 and axis2.

        Args:
            a: Central united-atom coordinates (3,).
            b_list: Positions of heavy bonded atoms.
            c: Coordinates of a second heavy atom or a hydrogen atom.
            dimensions: Simulation box dimensions (3,).

        .. code-block:: text

            a          1 = norm_ab
           / \         2 = perpendicular to norm_ab and norm_ac (or bc if >2 HAs)
          /   \        3 = perpendicular to 1 and 2
         b     c

        Returns:
            np.ndarray: (3, 3) array of the axes used to rotate forces.

        Raises:
            ValueError: If axes cannot be normalized due to degeneracy.
        """
        unscaled_axis1 = np.zeros(3, dtype=float)
        for b in b_list:
            ab_vector = self.get_vector(a, b, dimensions)
            unscaled_axis1 += ab_vector

        if np.allclose(unscaled_axis1, 0.0):
            raise ValueError("Degenerate axis1: summed bonded vectors are zero.")

        if len(b_list) >= 2:
            ac_vector = self.get_vector(c, np.asarray(b_list)[0], dimensions)
        else:
            ac_vector = self.get_vector(c, a, dimensions)

        unscaled_axis2 = np.cross(ac_vector, unscaled_axis1)
        unscaled_axis3 = np.cross(unscaled_axis2, unscaled_axis1)

        unscaled_custom_axes = np.array(
            (unscaled_axis1, unscaled_axis2, unscaled_axis3), dtype=float
        )
        mod = np.sqrt(np.sum(unscaled_custom_axes**2, axis=1))
        if np.any(np.isclose(mod, 0.0)):
            raise ValueError("Degenerate custom axes: cannot normalize (zero norm).")

        scaled_custom_axes = unscaled_custom_axes / mod[:, np.newaxis]
        return scaled_custom_axes

    def get_custom_residue_moment_of_inertia(
        self,
        center_of_mass: np.ndarray,
        positions: np.ndarray,
        masses: np.ndarray,
        custom_rot_axes: np.ndarray,
        dimensions: np.ndarray,
    ):
        """
        Compute moment of inertia around custom axes for a bead
        formed of multiple UAs.

        Args:
            center_of_mass: (3, ) COM for bead
            positions: (N,3) positions of the UAs in the bead
            masses: (N,) masses of the UAs in the bead
            custom_rot_axes: (3,3) array of residue rotation axes
            dimensions: (3,) simulation_box_dimensions

        Returns:
            np.ndarray: (3,) moment of inertia array.

        """

        translated_coords = self.get_vector(center_of_mass, positions, dimensions)
        custom_moment_of_inertia = np.zeros(3, dtype=float)

        for coord, mass in zip(translated_coords, masses, strict=True):
            axis_component = np.sum(
                np.cross(custom_rot_axes, coord) ** 2 * mass, axis=1
            )
            custom_moment_of_inertia += axis_component

        return custom_moment_of_inertia

    def get_custom_moment_of_inertia(
        self,
        UA,
        custom_rotation_axes: np.ndarray,
        center_of_mass: np.ndarray,
        dimensions: np.ndarray,
    ) -> np.ndarray:
        """Compute moment of inertia around custom axes for a UA.

        Get the moment of inertia (specifically used for the united atom level)
        from a set of rotation axes and a given center of mass (COM is usually the
        heavy atom position in a UA).

        Original behaviour preserved:

        - Uses PBC-aware translated coordinates.
        - Sums contributions from each atom using the squared norm of (axis × r)
          multiplied by mass.
        - Removes the lowest MOI degree of freedom if the UA only has a single
          bonded H (i.e. UA has 2 atoms total).

        Args:
            UA: MDAnalysis AtomGroup for the UA (heavy + bonded H atoms).
            custom_rotation_axes: (3, 3) array of rotation axes.
            center_of_mass: (3,) COM for the UA (typically HA position).
            dimensions: (3,) simulation box dimensions.

        Returns:
            np.ndarray: (3,) moment of inertia array.
        """
        translated_coords = self.get_vector(center_of_mass, UA.positions, dimensions)
        custom_moment_of_inertia = np.zeros(3, dtype=float)

        for coord, mass in zip(translated_coords, UA.masses, strict=True):
            axis_component = np.sum(
                np.cross(custom_rotation_axes, coord) ** 2 * mass, axis=1
            )
            custom_moment_of_inertia += axis_component

        if len(UA) == 2:
            order = custom_moment_of_inertia.argsort()[::-1]  # descending order
            custom_moment_of_inertia[order[-1]] = 0.0

        return custom_moment_of_inertia

    def get_flipped_axes(
        self,
        UA,
        custom_axes: np.ndarray,
        center_of_mass: np.ndarray,
        dimensions: np.ndarray,
    ):
        """Flip custom axes to a consistent direction with respect to the UA.

        For a given set of custom axes, ensure the axes are pointing in the
        correct direction with respect to the heavy atom position and the chosen
        center of mass.

        Args:
            UA: MDAnalysis AtomGroup for the UA.
            custom_axes: (3, 3) array of rotation axes.
            center_of_mass: (3,) COM reference (usually HA position).
            dimensions: (3,) simulation box dimensions.

        Returns:
            np.ndarray: (3, 3) array of flipped/normalized axes.
        """
        rr_axis = self.get_vector(UA[0].position, center_of_mass, dimensions)

        axis_norm = np.sqrt(np.sum(custom_axes**2, axis=1))
        custom_axes_flipped = custom_axes / axis_norm[:, np.newaxis]

        for i in range(3):
            dot_prod = float(np.dot(custom_axes_flipped[i], rr_axis))
            if dot_prod < 0.0:
                custom_axes_flipped[i] *= -1.0

        return custom_axes_flipped

    def get_vector(self, a: np.ndarray, b: np.ndarray, dimensions: np.ndarray):
        """Compute PBC-wrapped displacement vector(s).

        For vector of two coordinates over periodic boundary conditions (PBCs).

        Args:
            a: (3,) or (N, 3) array of coordinates.
            b: (3,) or (N, 3) array of coordinates.
            dimensions: (3,) simulation box dimensions.

        Returns:
            np.ndarray: Wrapped displacement vector(s) with broadcasted shape.
        """
        delta = b - a
        delta -= dimensions * np.round(delta / dimensions)
        return delta

    def get_moment_of_inertia_tensor(
        self,
        center_of_mass: np.ndarray,
        positions: np.ndarray,
        masses: Sequence[float],
        dimensions: np.ndarray,
    ) -> np.ndarray:
        """Compute a custom moment of inertia tensor.

        Calculate a custom moment of inertia tensor.
        E.g., for cases where the mass list will contain masses of UAs rather than
        individual atoms and the positions will be those for the UAs only
        (excluding the H atoms coordinates).

        Args:
            center_of_mass: (3,) chosen centre for the tensor.
            positions: (N, 3) point positions.
            masses: (N,) point masses corresponding to positions.
            dimensions: (3,) simulation box dimensions.

        Returns:
            np.ndarray: (3, 3) moment of inertia tensor.
        """
        r = self.get_vector(center_of_mass, positions, dimensions)
        r2 = np.sum(r**2, axis=1)

        masses_arr = np.asarray(list(masses), dtype=float)
        moment_of_inertia_tensor = np.eye(3) * np.sum(masses_arr * r2)
        moment_of_inertia_tensor -= np.einsum("i,ij,ik->jk", masses_arr, r, r)

        return moment_of_inertia_tensor

    def get_custom_principal_axes(
        self, moment_of_inertia_tensor: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute principal axes and moments from a custom MOI tensor.

        Principal axes and centre of axes from the ordered eigenvalues and
        eigenvectors of a moment of inertia tensor. This function allows for a
        custom moment of inertia tensor to be used, which isn't possible with the
        built-in MDAnalysis principal_axes() function.

        Original behaviour preserved:

        - Eigenvalues are sorted by descending absolute magnitude.
        - Eigenvectors are transposed so axes are returned as rows.
        - Z axis is flipped to enforce the same handedness convention as the
          original implementation.

        Args:
            moment_of_inertia_tensor: (3, 3) custom inertia tensor.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - principal_axes: (3, 3) principal axes (rows).
                - moment_of_inertia: (3,) principal moments.
        """
        eigenvalues, eigenvectors = np.linalg.eig(moment_of_inertia_tensor)
        order = np.abs(eigenvalues).argsort()[::-1]  # descending order
        transposed = np.transpose(eigenvectors)  # columns -> rows
        moment_of_inertia = eigenvalues[order]
        principal_axes = transposed[order]

        # point z axis in correct direction, as per original code
        cross_xy = np.cross(principal_axes[0], principal_axes[1])
        dot_z = float(np.dot(cross_xy, principal_axes[2]))
        if dot_z < 0:
            principal_axes[2] *= -1

        return principal_axes, moment_of_inertia

    def get_UA_masses(self, molecule) -> list[float]:
        """Return united-atom (UA) masses for a molecule.

        For a given molecule, return a list of masses of UAs (combination of the
        heavy atoms + bonded hydrogen atoms). This list is used to get the moment
        of inertia tensor for molecules larger than one UA.

        Args:
            molecule: MDAnalysis AtomGroup representing the molecule.

        Returns:
            list[float]: UA masses for each heavy atom.
        """
        ua_masses: list[float] = []
        for atom in molecule:
            if atom.mass > 1.1:
                ua_mass = float(atom.mass)
                bonded_atoms = molecule.select_atoms(f"bonded index {atom.index}")
                bonded_h_atoms = bonded_atoms.select_atoms("mass 1 to 1.1")
                for h in bonded_h_atoms:
                    ua_mass += float(h.mass)
                ua_masses.append(ua_mass)
        return ua_masses

    def get_chain(self, residue, first, last):
        """
        For a given MDAnalysis AtomGroup and two given heavy atoms
        within that AtomGroup, return the
        shortest path between the two atoms.

        Args:
            residue: MDAnalysis AtomGroup representing
            the residue/monomer of interest.
            first: First heavy atom in the chain
            last: Last heavy atom in the chain

        Returns:
            chain: Array containing chain atoms.
        """

        chain = []
        # at the beggining we've only visited the first atom
        visited_dict = {first: True}
        # keep the previous atom to trace back the path
        prev = {}
        # queue of next heavy atoms to visit
        next_to_visit = [first]
        # all others heavy atoms in the residue, we have not yet visited
        remaining_heavy_atoms = residue.atoms.select_atoms(
            f"(mass 2 to 999) and not index {first.index}"
        )
        for atom in remaining_heavy_atoms:
            visited_dict[atom] = False
        current = first

        while not visited_dict[last]:
            # we haven't found a path to the last residue
            next_to_visit.pop(0)
            # we're visiting the current atom => we remove it from the queue
            bonded_atoms = residue.atoms.select_atoms(
                f"(mass 2 to 999) and bonded index {current.index}"
            )

            if last in bonded_atoms:
                # we found a path to the last atom
                visited_dict[last] = True
                chain.append(last)
                prev[last] = current

            else:
                for bonded_atom in bonded_atoms:
                    # look for unvisited bonded atoms to the current atom we're visiting
                    if not visited_dict[bonded_atom]:
                        # we're going to want to visit the atoms
                        next_to_visit.append(bonded_atom)
                        prev[bonded_atom] = current
                # we visit the next atom in the queue
                current = next_to_visit[0]
                visited_dict[current] = True

        # we track the previous atom back to the first atom now
        current = last
        chain = [last]
        # subtract index of first atom in resid
        # most likely will coincide with first
        # but this will work even if it doesn't
        # accout for in-residue index
        # start from last atom in chain
        while chain[-1] != first:
            # we haven't yet returned to the first atom
            current = prev[current]
            chain.append(current)

        chain = np.flip(chain)
        # only get in between residues
        chain = chain[1:-1]
        # accout for in-residue index
        return chain
