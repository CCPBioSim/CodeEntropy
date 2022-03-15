import numpy as nmp
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.Trajectory import TrajectoryConstants as TCON

def sort_modes(arg_modeSpectra):
	"""
	Sort the modes in ascending order using mode frequencies
	and return a new sorted spectrum
	"""
	retSpectra = []
	ascOrder = nmp.argsort([m for m in arg_modeSpectra])
	for im, midx in enumerate(ascOrder):
		arg_modeSpectra[midx].modeIdx = im + 1
		retSpectra.append(arg_modeSpectra[midx])

	return retSpectra
#END

class Mode(object):
	"""
	A class that contains information about a 
	fundamental mode of motion.
	"""

	def __init__(self, arg_modeIdx = 0, arg_modeEval = -1,\
	             arg_modeEvec = [], arg_modeFreq = -1,  **kwargs):
		self.modeIdx = arg_modeIdx
		self.modeEval = arg_modeEval
		self.modeEvec = arg_modeEvec
		self.modeFreq = arg_modeFreq

		for k, v in kwargs.items():
			setattr(self, k, v)

		self.modeAmpl = 0

	#END

	def __lt__(self, otherMode):
		return self.modeFreq < otherMode.modeFreq

	def __gt__(self, otherMode):
		return self.modeFreq > otherMode.modeFreq

	def __eq__(self, otherMode):
		return self.modeFreq == otherMode.modeFreq

	def get_mode_vecorder(self):
		return len(self.modeEvec)

	def get_line_nmd(self, arg_wfac):
		""" returns a string containing the mode info in the format
		suitable for NMWiz (NMD file)"""

		# mode - Normal mode array. Each normal mode array must be provided 
		# in one line as a list of decimal numbers. 
		# Mode array may be preceded by mode index and mode length (scaling factor) 
		# For FC: scaling factor is KT/sqrt(L_i*m_j) for mode `i` and bead `j` (mass weighting)
		# For TC: scaling factor is KT/sqrt(L_i*I_jj) for mode `i` and bead `j` (intertia weighting)
		# For NMA, QH, etc. this is different.
		#
		# For FC and TC, a common mass-independent mode amplitude factor is set.
		# therefore, when printing the eigenvectors out, they must be mass weighted (FC) or
		# inertia weighted for (TC).
		#
		# `arg_wfac` is an array of weighting factors
		#
		# Outpur format: "Mode %d %.2f x3 x3 ... x3"
		try:
			assert(int(self.get_mode_vecorder()/TCON.VECTORDIM) == len(arg_wfac))
		except:
			raise ValueError("Eigen vector and weight factor array are of different lengths ({} vs. {})"\
				.format(int(self.get_mode_vecorder()/TCON.VECTORDIM), len(arg_wfac)))

		nrow = int(self.get_mode_vecorder()/TCON.VECTORDIM)
		retStr = "Mode {:d} {:.2f} ".format(self.modeIdx, self.modeAmpl)
		evecs = nmp.reshape(a = self.modeEvec,\
			           newshape = (nrow, TCON.VECTORDIM))

		for uv, um in zip(evecs, arg_wfac):
			for uvm in nmp.divide(uv, nmp.sqrt(um)):
				retStr = "{} {:.3f}".format(retStr, uvm)

		return retStr
	#END