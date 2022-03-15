import numpy as nmp
import sys

from CodeEntropy.FunctionCollection import CustomFunctions, Utils

def write_file(arg_filename):
	"""
	Open a file stream for writing. It will overwrite any exisitng file
	by that name.
	"""
	fwrite = open(arg_filename,"w")
	fwrite.close()
	return 
#END

def append_file(arg_filename):
	"""
	Open a file stream for appending.
	"""
	fapp = open(arg_filename,"a")
	fapp.close()
#END

def write_atomic_forces( arg_baseMolecule, arg_hostDataContainer, arg_outFile ):
	""" function that prints the 
	1. average
	2. SD
	of the forces felt by each atom
	"""

	numFrames = len(arg_hostDataContainer.trajSnapshots)
	for iAtom in arg_baseMolecule.atomIndexArray:
		if not arg_baseMolecule.isHydrogenArray[iAtom]:
			iAtomname = arg_baseMolecule.atomNameArray[iAtom]
			iResid  = arg_baseMolecule.atomResidueIdxArray[iAtom]
			iResname = arg_baseMolecule.resnameArray[iResid]
			iChain = "X"

			iForces = arg_hostDataContainer._labForces[:,iAtom]
			try:
				assert(nmp.shape(iForces) == (numFrames, 3))
			except:
				Utils.printflush('Requested shape ', nmp.shape(iForces), 'does not allow proper broadcasting of the array of forces from the main data containter.')

			#coordinates
			iAvgX = nmp.mean(nmp.asarray([arg_hostDataContainer._labCoords[:, iAtom, 0]]))
			iAvgY = nmp.mean(nmp.asarray([arg_hostDataContainer._labCoords[:, iAtom, 1]]))
			iAvgZ = nmp.mean(nmp.asarray([arg_hostDataContainer._labCoords[:, iAtom, 2]]))
			
			# forces
			iForceMagnitudes = nmp.asarray([ nmp.linalg.norm(iForces[i]) for i in range(numFrames)])

			# average force
			iAvgForce = nmp.mean(iForceMagnitudes)

			# std dev of forces
			iSDevForce = nmp.std(iForceMagnitudes)

			#pdb format string
			atomStr = '{:<6s}{:>5d} {:>4s} {:3s} {:1s}{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}'.format("ATOM", iAtom, iAtomname, iResname, iChain, iResid, iAvgX, iAvgY, iAvgZ, iAvgForce, iSDevForce )
			Utils.printOut(arg_outFile, atomStr)

	return
#END

def write_a_matrix(arg_matrix, arg_descriptor, arg_outFile):
	""" This function writes a matrix to an ASCII outputfile. The elemetns are
	written in the format "%5d %5d %15.5f" in each line."""

	# fetch the shape (it must be two dimesnional)
	matShape = nmp.shape(arg_matrix)
	try:
		assert(len(matShape) == 2)
	except:
		print("A 2D matrix is expected. Received a matrix with {} dimensions instead. Will not write anything therefore.".format(len(matShape))) 
		return

	# write
	Utils.printOut(arg_outFile, "***MATRIX: {}".format(arg_descriptor))
	
	nrow, ncol = matShape
	Utils.printOut(arg_outFile, "***MATRIX: SHAPE {} {}".format(*matShape))
	Utils.printOut(arg_outFile, "***MATRIX: TRACE {:<15.5f}".format(nmp.trace(arg_matrix)))
	for iRow in range(nrow):
		for jCol in range(ncol):
			Utils.printOut(arg_outFile, "{:>5d}{:>5d}{:>15.5f}".format(iRow, jCol, arg_matrix[iRow][jCol]))
	Utils.printOut(arg_outFile, "***MATRIX: END")

	return

	
