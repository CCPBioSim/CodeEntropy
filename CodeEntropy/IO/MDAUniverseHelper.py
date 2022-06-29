import MDAnalysis as mda
import pickle
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader

def new_U_select_frame(u, start=None, end=None, step=1):
	"""Create a reduced universe by dropping frames according to user selection

	Parameters
	----------
	u : MDAnalyse.Universe
		A Universe object will all topology, dihedrals,coordinates and force information.
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

	Parameters
	----------
	u : MDAnalyse.Universe
		A Universe object will all topology, dihedrals,coordinates and force information.
	select_string : str, Optional, default: 'all'
		MDAnalysis.select_atoms selection string.

	Returns
	-------
		u2 : MDAnalysis.Universe
			reduced universe

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
	
	Parameters
	----------
	u : MDAnalyse.Universe
		A Universe object will all topology, dihedrals,coordinates and force information.
	name : str, Optional. default: 'default'
		The name of file with sub file name .pkl

	Returns
	-------
		name : str
			filename of saved universe
	"""
	filename = f"{name}.pkl"
	pickle.dump(u, open(filename, "wb"))
	return name

def read_universe(path):
	"""read a universe to working directories as pickle

	Parameters
	----------
	path : str
		The path to file.

	Returns
	-------
		u : MDAnalysis.Universe
			A Universe object will all topology, dihedrals,coordinates and force information.
	"""
	u = pickle.load(open(path, "rb"))
	return u