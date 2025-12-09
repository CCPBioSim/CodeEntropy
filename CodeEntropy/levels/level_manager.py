import logging
from typing import Any, Dict

from CodeEntropy.levels.hierarchy_graph import LevelDAG

logger = logging.getLogger(__name__)


class LevelManager:
    """
    High-level coordinator that runs the Level DAG and returns results required
    later for entropy nodes.

    Output from this class is forwarded into EntropyGraph.
    """

    def __init__(self, run_manager=None):
        self.run_manager = run_manager
        self.level_results = None

    def run(self, universe) -> Dict[str, Any]:
        """
        Execute the level-processing DAG and return all structural results.

        Input:
            universe (MDAnalysis.Universe)

        Output dictionary feeds forward into the entropy pipeline.
        """
        dag = LevelDAG().build()

        shared_data = {"universe": universe, "run_manager": self.run_manager}

        results = dag.execute(shared_data)

        self.level_results = {
            "levels": results["detect_levels"]["levels"],
            "molecule_count": results["detect_molecules"]["molecule_count"],
            "beads": results["build_beads"]["beads_by_mol_level"],
            "axes": results["compute_axes"]["axes"],
            "forces": results["compute_weighted_forces"]["forces"],
            "torques": results["compute_weighted_torques"]["torques"],
            "cov_matrices": results["build_covariance"]["covariance"],
            "dihedrals": results["compute_dihedrials"]["dihedrals"],
            "conformations": results["build_conformations"]["states"],
            "neighbours": results["compute_neighbours"]["neighbours"],
        }

        return self.level_results

    def get(self, key):
        return self.level_results.get(key)
