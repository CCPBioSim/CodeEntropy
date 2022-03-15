import numpy as nmp
import sys

from CodeEntropy.FunctionCollection import CustomFunctions as CF

# output Forces
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
				CF.printflush('Requested shape ', nmp.shape(iForces), 'does not allow proper broadcasting of the array of forces from the main data containter.')

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
			CF.printOut(arg_outFile, atomStr)

	return
#END
