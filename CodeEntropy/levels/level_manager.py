import logging

from CodeEntropy.levels.coordinate_system import CoordinateSystem
from CodeEntropy.levels.dihedral_tools import DihedralAnalysis
from CodeEntropy.levels.force_torque_manager import ForceTorqueManager
from CodeEntropy.levels.level_hierarchy import LevelHierarchy
from CodeEntropy.levels.matrix_operations import MatrixOperations
from CodeEntropy.levels.neighbours import Neighbours

logger = logging.getLogger(__name__)


class LevelManager:
    """
    High–level orchestrator for all 'level' computations.

    It does not implement physics itself. Instead it:
      - delegates to LevelHierarchy to decide which levels exist
      - delegates to ForceTorqueManager to build covariance matrices
      - delegates to DihedralAnalysis to build conformational states
      - delegates to MatrixOperations for matrix cleanup / utilities
    """

    def __init__(self):
        """
        Construct modular helpers and keep shared references.
        """
        self._hierarchy = LevelHierarchy()
        self._coords = CoordinateSystem()
        self._dihedrals = DihedralAnalysis()
        self._mat_ops = MatrixOperations()
        self._force_torque = ForceTorqueManager()
        self._neighbours = Neighbours()

    def select_levels(self, data_container):
        """
        Wrapper around LevelHierarchy.select_levels

        Parameters
        ----------
        data_container : MDAnalysis.Universe
            Reduced universe / selection.

        Returns
        -------
        number_molecules : int
        levels : list[list[str]]
            e.g. [["united_atom", "residue", "polymer"], ["united_atom"], ...]
        """
        number_molecules, levels = self._hierarchy.select_levels(data_container)
        logger.debug(f"[LevelManager] number_molecules={number_molecules}")
        logger.debug(f"[LevelManager] levels={levels}")
        return number_molecules, levels

    def get_beads(self, data_container, level):
        """
        Simple pass-through to LevelHierarchy.get_beads

        Kept here in case you later want the DAG to ask LevelManager for bead
        sets without touching LevelHierarchy directly.
        """
        return self._hierarchy.get_beads(data_container, level)

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
        Thin wrapper around ForceTorqueManager.build_covariance_matrices.

        All the heavy lifting (loops over frames, groups, levels, beads,
        physics of forces/torques, incremental averaging) stays inside
        ForceTorqueManager.

        Parameters
        ----------
        entropy_manager : EntropyManager
            Needed because ForceTorqueManager currently calls
            entropy_manager._get_molecule_container(...)
        reduced_atom : MDAnalysis.Universe
        levels : list[list[str]]
        groups : dict[int, list[int]]
        start, end, step : int
        number_frames : int

        Returns
        -------
        force_matrices : dict
        torque_matrices : dict
        frame_counts : dict
        """
        logger.debug(
            "[LevelManager] Delegating to ForceTorqueManager.build_covariance_matrices"
        )
        return self._force_torque.build_covariance_matrices(
            entropy_manager=entropy_manager,
            reduced_atom=reduced_atom,
            levels=levels,
            groups=groups,
            start=start,
            end=end,
            step=step,
            number_frames=number_frames,
        )

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
        conformational_entropy_obj,
    ):
        """
        Wrapper around DihedralAnalysis.build_conformational_states.

        Parameters
        ----------
        entropy_manager : EntropyManager
        reduced_atom : MDAnalysis.Universe
        levels : list[list[str]]
        groups : dict[int, list[int]]
        start, end, step : int
        number_frames : int
        bin_width : int
            Histogram bin width (degrees) for dihedral distributions.
        conformational_entropy_obj : ConformationalEntropy
            The CE object, passed through because your dihedral code
            may call its methods.

        Returns
        -------
        states_ua : dict
            e.g. {(group_id, residue_id): states_array}
        states_res : list
            e.g. [states_for_group0, states_for_group1, ...]
        """
        logger.debug(
            "[LevelManager] Delegating to DihedralAnalysis.build_conformational_states"
        )
        return self._dihedrals.build_conformational_states(
            entropy_manager=entropy_manager,
            reduced_atom=reduced_atom,
            levels=levels,
            groups=groups,
            start=start,
            end=end,
            step=step,
            number_frames=number_frames,
            bin_width=bin_width,
            conformational_entropy_obj=conformational_entropy_obj,
        )

    def filter_zero_rows_columns(self, matrix):
        """
        Wrapper around MatrixOperations.filter_zero_rows_columns.


        Physics and numerical behaviour are unchanged – MatrixOperations
        contains the original implementation.
        """
        return self._mat_ops.filter_zero_rows_columns(matrix)

    def get_axes(self, bead):
        """
        Convenience forwarder to CoordinateSystem.get_axes (if/when needed).
        Not used by EntropyManager right now but handy for future DAG nodes.
        """
        return self._coords.get_axes(bead)

    def find_neighbours(self, data_container, bead, cutoff):
        """
        Convenience wrapper around Neighbours.
        """
        return self._neighbours.find_neighbours(data_container, bead, cutoff)
