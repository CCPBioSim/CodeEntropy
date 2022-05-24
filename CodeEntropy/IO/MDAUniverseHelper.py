import MDAnalysis as mda
import pickle
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader

def new_U_select_frame(u, start=None, end=None, step=1):
    """Create a reduced universe by dropping frames according to user selection

    Args:
        u (MDAnalysis.Universe): the Universe to reduce
        start (int, optional): Frame. frame id to start analysis. Default None will start from frame 0.
        end (int, optional): Frame id to end analysis. Default Nonw will end at last frame
        step (int, optional): Steps between frame. Defaults to 1.

    Returns:
        MDAnalysis.Universe: reduced universe
    """
    if start == None:
        start = 0
    if end == None:
        end = len(u.trajectory)
    select_atom = u.select_atoms('all')
    coordinates = AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom).run().results['timeseries'][start:end:step]
    forces = AnalysisFromFunction(lambda ag: ag.forces.copy(), select_atom).run().results['timeseries'][start:end:step]
    dimensions = AnalysisFromFunction(lambda ag: ag.dimensions.copy(), select_atom).run().results['timeseries'][start:end:step]
    u2 = mda.Merge(select_atom)
    u2.load_new(coordinates, format=MemoryReader, forces=forces, dimensions=dimensions)
    return u2

def new_U_select_atom(u, select_string='all'):
    """Create a reduced universe by dropping atoms according to user selection
    Args:
        u (MDAnalysis.Universe): the Universe to reduce
        select_string (str, optional): MDAnalysis.select_atoms selection string. Defaults to 'all'.

    Returns:
        MDAnalysis.Universe: reduced universe
    """
    select_atom = u.select_atoms(select_string)
    coordinates = AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom).run().results['timeseries']
    forces = AnalysisFromFunction(lambda ag: ag.forces.copy(), select_atom).run().results['timeseries']
    dimensions = AnalysisFromFunction(lambda ag: ag.dimensions.copy(), select_atom).run().results['timeseries']
    u2 = mda.Merge(select_atom)
    u2.load_new(coordinates, format=MemoryReader, forces=forces, dimensions=dimensions)
    return u2        

def write_universe(u, name='default'):
    """Write a universe to working directories as pickle

    Args:
        u (MDAnalysis.Universe): the Universe to save
        name (str, optional): the name of . Defaults to 'default'.

    Returns:
        str: filename
    """
    filename = f"{name}.pkl"
    pickle.dump(u, open(filename, "wb"))
    return name

def read_universe(path):
    """read a universe to working directories as pickle

    Args:
        path (str): the name of path.

    Returns:
        str: filename
    """
    u = pickle.load(open(path, "rb"))
    return u