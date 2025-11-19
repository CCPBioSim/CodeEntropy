import logging
from typing import Any, Dict

from CodeEntropy.levels.hierarchy_graph import HierarchyGraph

logger = logging.getLogger(__name__)


class LevelManager:
    """
    Coordinates the DAG-based computation of molecular levels and beads.
    All physics/maths are delegated to lower-level classes and DAG nodes.
    """

    def __init__(self, universe, run_manager, args):
        """
        Parameters
        ----------
        universe : MDAnalysis.Universe
            The MD system being analysed.
        run_manager : RunManager
            Provides selection helpers and unit conversions.
        args : Namespace
            Parsed CLI arguments.
        """
        self.universe = universe
        self.run_manager = run_manager
        self.args = args

    def run_hierarchy(self) -> Dict[str, Any]:
        """
        Execute the structural hierarchy DAG (levels → beads).

        Returns
        -------
        dict
            Contains:
              - number_molecules
              - levels
              - beads_by_mol_level
        """

        shared_data = {
            "universe": self.universe,
            "args": self.args,
            "run_manager": self.run_manager,
        }

        graph = HierarchyGraph().build(self.run_manager)
        results = graph.execute(shared_data)

        logger.debug("[LevelManager] Hierarchy DAG results:")
        logger.debug(results)

        return results

    def run(self):
        """
        Placeholder: eventually will run all DAGs (hierarchy, matrices, entropy).
        For now, only run the hierarchy graph.
        """
        return self.run_hierarchy()
