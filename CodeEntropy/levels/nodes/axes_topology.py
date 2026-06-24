"""Build static axes-topology metadata for frame covariance calculations.

This module caches topology-only atom-index relationships needed by customised
axes calculations. The cache avoids repeated MDAnalysis selection parsing inside
the frame-local covariance loop while preserving frame-dependent positions,
forces, centres, axes, torques, and moments of inertia.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

UAKey = tuple[int, int, int]
ResidueKey = tuple[int, int]


@dataclass(frozen=True)
class UAAxesTopology:
    """Static topology required to compute customised united-atom axes.

    Attributes:
        heavy_atom_index: Reduced-universe atom index for the UA heavy atom.
        ua_atom_indices: Atom indices for the UA heavy atom and its bonded
            hydrogens/light atoms.
        ua_all_atom_indices: Atom indices for the UA heavy atom, bonded heavy
            atoms, and bonded hydrogens/light atoms.
        bonded_heavy_indices: Heavy atoms bonded to the UA heavy atom.
        bonded_light_indices: Hydrogens/light atoms bonded to the UA heavy atom.
        residue_heavy_indices: Heavy atoms in the parent residue.
        residue_ua_masses: UA masses for heavy atoms in the parent residue.
    """

    heavy_atom_index: int
    ua_atom_indices: np.ndarray
    ua_all_atom_indices: np.ndarray
    bonded_heavy_indices: np.ndarray
    bonded_light_indices: np.ndarray
    residue_heavy_indices: np.ndarray
    residue_ua_masses: np.ndarray


@dataclass(frozen=True)
class ResidueAxesTopology:
    """Static topology required to compute customised residue axes.

    Attributes:
        residue_heavy_indices: Heavy atom indices in the residue.
        residue_ua_masses: UA masses for heavy atoms in the residue.
        has_neighbor_bonds: Whether the residue is bonded to a neighbouring
            residue according to the original customised residue-axis selection.
    """

    residue_heavy_indices: np.ndarray
    residue_ua_masses: np.ndarray
    has_neighbor_bonds: bool


@dataclass(frozen=True)
class AxesTopology:
    """Cached axes topology for frame covariance calculations.

    Attributes:
        ua: Mapping from ``(mol_id, local_residue_id, ua_id)`` to cached
            united-atom axes topology.
        residue: Mapping from ``(mol_id, local_residue_id)`` to cached
            residue axes topology.
    """

    ua: dict[UAKey, UAAxesTopology] = field(default_factory=dict)
    residue: dict[ResidueKey, ResidueAxesTopology] = field(default_factory=dict)


class BuildAxesTopologyNode:
    """Build static customised-axes topology before frame covariance execution."""

    def run(self, shared_data: dict[str, Any]) -> dict[str, Any]:
        """Build cached axes topology and write it into shared data.

        The cache is only populated when ``args.customised_axes`` is true. When
        customised axes are disabled, an empty cache is still written so later
        stages can read ``shared_data["axes_topology"]`` safely.

        Args:
            shared_data: Shared workflow data containing ``args`` and, when
                customised axes are enabled, ``reduced_universe``, ``levels``,
                and ``beads``.

        Returns:
            Dict containing the cached ``axes_topology`` object.
        """
        args = shared_data["args"]
        topology = AxesTopology()

        if not bool(getattr(args, "customised_axes", False)):
            shared_data["axes_topology"] = topology
            return {"axes_topology": topology}

        u = shared_data["reduced_universe"]
        levels = shared_data["levels"]
        beads = shared_data["beads"]

        ua_topology: dict[UAKey, UAAxesTopology] = {}
        residue_topology: dict[ResidueKey, ResidueAxesTopology] = {}
        fragments = u.atoms.fragments

        for mol_id, level_list in enumerate(levels):
            mol = fragments[mol_id]

            if "residue" in level_list:
                self._add_residue_topology(
                    mol=mol,
                    mol_id=mol_id,
                    beads=beads,
                    out=residue_topology,
                )

            if "united_atom" in level_list:
                self._add_ua_topology(
                    u=u,
                    mol=mol,
                    mol_id=mol_id,
                    beads=beads,
                    out=ua_topology,
                )

        topology = AxesTopology(ua=ua_topology, residue=residue_topology)
        shared_data["axes_topology"] = topology
        return {"axes_topology": topology}

    def _add_residue_topology(
        self,
        *,
        mol: Any,
        mol_id: int,
        beads: dict[Any, list[np.ndarray]],
        out: dict[ResidueKey, ResidueAxesTopology],
    ) -> None:
        """Cache static residue axes topology for one molecule.

        Args:
            mol: Molecule AtomGroup.
            mol_id: Molecule index.
            beads: Bead-index mapping produced by ``BuildBeadsNode``.
            out: Output residue topology mapping mutated in place.
        """
        bead_key = (mol_id, "residue")
        bead_idx_list = beads.get(bead_key, [])
        if not bead_idx_list:
            return

        for local_res_i, residue in enumerate(mol.residues):
            if local_res_i >= len(bead_idx_list):
                continue

            residue_atoms = residue.atoms
            residue_heavy = residue_atoms.select_atoms("mass 2 to 999")
            residue_heavy_indices = residue_heavy.indices.astype(int, copy=True)
            residue_ua_masses = np.asarray(
                self._get_ua_masses_from_topology(residue_atoms),
                dtype=float,
            )
            has_neighbor_bonds = self._has_neighbor_bonds(
                mol=mol,
                local_res_i=local_res_i,
            )

            out[(mol_id, local_res_i)] = ResidueAxesTopology(
                residue_heavy_indices=residue_heavy_indices,
                residue_ua_masses=residue_ua_masses,
                has_neighbor_bonds=has_neighbor_bonds,
            )

    def _add_ua_topology(
        self,
        *,
        u: Any,
        mol: Any,
        mol_id: int,
        beads: dict[Any, list[np.ndarray]],
        out: dict[UAKey, UAAxesTopology],
    ) -> None:
        """Cache static UA axes topology for one molecule.

        Args:
            u: Reduced universe used to resolve bead atom-index arrays.
            mol: Molecule AtomGroup.
            mol_id: Molecule index.
            beads: Bead-index mapping produced by ``BuildBeadsNode``.
            out: Output UA topology mapping mutated in place.
        """
        for local_res_i, residue in enumerate(mol.residues):
            bead_key = (mol_id, "united_atom", local_res_i)
            bead_idx_list = beads.get(bead_key, [])

            if not bead_idx_list:
                continue

            residue_atoms = residue.atoms
            residue_heavy = residue_atoms.select_atoms("prop mass > 1.1")
            residue_heavy_indices = residue_heavy.indices.astype(int, copy=True)
            residue_ua_masses = np.asarray(
                self._get_ua_masses_from_topology(residue_atoms),
                dtype=float,
            )

            for ua_i, bead_indices in enumerate(bead_idx_list):
                bead = u.atoms[bead_indices]
                heavy = bead.select_atoms("prop mass > 1.1")

                if len(heavy) == 0:
                    logger.warning(
                        "Skipping UA axes topology with no heavy atom: "
                        "mol=%s residue=%s ua=%s",
                        mol_id,
                        local_res_i,
                        ua_i,
                    )
                    continue

                heavy_atom = heavy[0]
                bonded_heavy, bonded_light = self._split_bonded_atoms(heavy_atom)

                heavy_index = np.asarray([int(heavy_atom.index)], dtype=int)
                bonded_heavy_indices = bonded_heavy.indices.astype(int, copy=True)
                bonded_light_indices = bonded_light.indices.astype(int, copy=True)

                ua_atom_indices = np.concatenate(
                    [heavy_index, bonded_light_indices],
                    axis=0,
                )
                ua_all_atom_indices = np.concatenate(
                    [heavy_index, bonded_heavy_indices, bonded_light_indices],
                    axis=0,
                )

                out[(mol_id, local_res_i, ua_i)] = UAAxesTopology(
                    heavy_atom_index=int(heavy_atom.index),
                    ua_atom_indices=ua_atom_indices,
                    ua_all_atom_indices=ua_all_atom_indices,
                    bonded_heavy_indices=bonded_heavy_indices,
                    bonded_light_indices=bonded_light_indices,
                    residue_heavy_indices=residue_heavy_indices,
                    residue_ua_masses=residue_ua_masses,
                )

    @staticmethod
    def _has_neighbor_bonds(*, mol: Any, local_res_i: int) -> bool:
        """Return whether a residue is bonded to neighbouring residues.

        Args:
            mol: Molecule AtomGroup used for the original bonded-neighbour
                selection.
            local_res_i: Residue index local to ``mol``.

        Returns:
            True when the residue has bonded atoms in the previous or next
            residue according to the original customised residue-axis query.
        """
        index_prev = local_res_i - 1
        index_next = local_res_i + 1
        atom_set = mol.select_atoms(
            f"(resindex {index_prev} or resindex {index_next}) "
            f"and bonded resid {local_res_i}"
        )
        return len(atom_set) > 0

    @staticmethod
    def _split_bonded_atoms(atom: Any) -> tuple[Any, Any]:
        """Return bonded heavy and light atoms for one atom.

        Args:
            atom: MDAnalysis Atom.

        Returns:
            Tuple containing bonded heavy atoms and bonded hydrogens/light atoms.
        """
        bonded_atoms = atom.bonded_atoms
        bonded_heavy = bonded_atoms.select_atoms("mass 2 to 999")
        bonded_light = bonded_atoms.select_atoms("mass 1 to 1.1")
        return bonded_heavy, bonded_light

    @staticmethod
    def _get_ua_masses_from_topology(atom_group: Any) -> list[float]:
        """Return UA masses using static bonded atom relationships.

        Args:
            atom_group: AtomGroup containing atoms from one residue.

        Returns:
            List of UA masses, one for each heavy atom in ``atom_group``.
        """
        ua_masses: list[float] = []

        for atom in atom_group:
            if atom.mass <= 1.1:
                continue

            ua_mass = float(atom.mass)
            bonded_atoms = getattr(atom, "bonded_atoms", None)
            if bonded_atoms is None:
                ua_masses.append(ua_mass)
                continue

            bonded_h_atoms = bonded_atoms.select_atoms("mass 1 to 1.1")
            for hydrogen in bonded_h_atoms:
                ua_mass += float(hydrogen.mass)

            ua_masses.append(ua_mass)

        return ua_masses
