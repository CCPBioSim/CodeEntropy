import logging

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader

logger = logging.getLogger(__name__)


class UniverseOperations:
    """
    Functions to create and manipulate MDAnalysis Universe objects.
    """

    def __init__(self):
        """
        Initialise class
        """
        self._universe = None

    def new_U_select_frame(self, u, start=None, end=None, step=1):
        """Create a reduced universe by dropping frames according to
        user selection.

        Parameters
        ----------
        u : MDAnalyse.Universe
            A Universe object will all topology, dihedrals,coordinates and force
            information
        start : int or None, Optional, default: None
            Frame id to start analysis. Default None will start from frame 0
        end : int or None, Optional, default: None
            Frame id to end analysis. Default None will end at last frame
        step : int, Optional, default: 1
            Steps between frame.

        Returns
        -------
            u2 : MDAnalysis.Universe
                reduced universe
        """
        if start is None:
            start = 0
        if end is None:
            end = len(u.trajectory)
        select_atom = u.select_atoms("all", updating=True)
        coordinates = (
            AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom)
            .run()
            .results["timeseries"][start:end:step]
        )
        forces = (
            AnalysisFromFunction(lambda ag: ag.forces.copy(), select_atom)
            .run()
            .results["timeseries"][start:end:step]
        )
        u2 = mda.Merge(select_atom)
        u2.load_new(coordinates, format=MemoryReader, forces=forces)
        logger.debug(f"MDAnalysis.Universe - reduced universe: {u2}")

        return u2

    def new_U_select_atom(self, u, select_string="all"):
        """Create a reduced universe by dropping atoms according to
        user selection.

        Parameters
        ----------
        u : MDAnalyse.Universe
            A Universe object will all topology, dihedrals,coordinates and force
            information
        select_string : str, Optional, default: 'all'
            MDAnalysis.select_atoms selection string.

        Returns
        -------
            u2 : MDAnalysis.Universe
                reduced universe

        """
        select_atom = u.select_atoms(select_string, updating=True)
        coordinates = (
            AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom)
            .run()
            .results["timeseries"]
        )
        forces = (
            AnalysisFromFunction(lambda ag: ag.forces.copy(), select_atom)
            .run()
            .results["timeseries"]
        )
        u2 = mda.Merge(select_atom)
        u2.load_new(coordinates, format=MemoryReader, forces=forces)
        logger.debug(f"MDAnalysis.Universe - reduced universe: {u2}")

        return u2

    def get_molecule_container(self, universe, molecule_id):
        """
        Extracts the atom group corresponding to a single molecule from the universe.

        Args:
            universe (MDAnalysis.Universe): The reduced universe.
            molecule_id (int): Index of the molecule to extract.

        Returns:
            MDAnalysis.Universe: Universe containing only the selected molecule.
        """
        # Identify the atoms in the molecule
        frag = universe.atoms.fragments[molecule_id]
        selection_string = f"index {frag.indices[0]}:{frag.indices[-1]}"

        return self.new_U_select_atom(universe, selection_string)

    def merge_forces(self, tprfile, trrfile, forcefile, fileformat=None, kcal=False):
        """
        Creates a universe by merging the coordinates and forces from
        different input files.

        Args:
            tprfile : Topology input file
            trrfile : Coordinate trajectory file
            forcefile : Force trajectory file
            format : Optional string for MDAnalysis identifying the file format
            kcal : Optional Boolean for when the forces are in kcal not kJ

        Returns:
           MDAnalysis Universe object
        """

        logger.debug(f"Loading Universe with {trrfile}")
        u = mda.Universe(tprfile, trrfile, format=fileformat)

        logger.debug(f"Loading Universe with {forcefile}")
        u_force = mda.Universe(tprfile, forcefile, format=fileformat)

        select_atom = u.select_atoms("all")
        select_atom_force = u_force.select_atoms("all")

        coordinates = (
            AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom)
            .run()
            .results["timeseries"]
        )
        forces = (
            AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom_force)
            .run()
            .results["timeseries"]
        )

        if kcal:
            # Convert from kcal to kJ
            forces *= 4.184

        logger.debug("Merging forces with coordinates universe.")
        new_universe = mda.Merge(select_atom)
        new_universe.load_new(coordinates, forces=forces)

        return new_universe
