import numpy as nmp
import sys

from CodeEntropy.FunctionCollection import Utils, CustomFunctions
from CodeEntropy.FunctionCollection import UnitsAndConversions as CONST

class BeadCollection(object):
	""" 
	A class that containes all the properties of 
	the beads/particles that comprise it.
	Properties include values of local vectors and a connection to
	the main data container.
	
	A name is also provided while initializaion
	to be used for writing various outputs related 
	to it.
	"""
	
	def __init__(self, arg_descriptor, arg_hostDataContainer, **kwargs):
		self.name = arg_descriptor
		self.hostDataContainer = arg_hostDataContainer
		self.listOfBeads = [] 
		self.indexPairList = None

		self.subMatrixDict = {"FF" : None,
							  "TT" : None}
		
		# eigen values, frequencues and entropy per DOF
		self.lambdaFreqEntropyDict = {"FF" : None,
									  "TT" : None}
	#END	 

	def assign_attribute(self, arg_attrName, arg_attrValue):
		setattr(self, arg_attrName, arg_attrValue)
		return
	
	def reinitialize_matrices(self):
		N = len(self.listOfBeads)
		if N == 0:
			raise ValueError('There should be at least 1 bead in the system.')
		
		self.indexPairList = [(i,j) for i in range(N) for j in range(N)]
		
		for mKey in self.subMatrixDict.keys():
			self.subMatrixDict[mKey] = nmp.zeros( (N, N, 3, 3) )
			
		return
	#END	 
	
	def generate_quadrant(self, arg_pairString, arg_filterZeros):
		""" For an input sub matrix of 4 dimension where 
		each i-j element is another 3x3 matrix (outer product),
		it used nmp.block to return a 3N x 3N matrix which is called 'Quadrant'

		Sometimes there can be a row or column that is entirely zero. If arg_filterZeros is True, then they must be removed"""

		subMatrix = self.subMatrixDict[arg_pairString]
		nRowsMajor, nColsMajor, nRowsMinor, nColsMinor = nmp.shape(subMatrix)
		
		quadrant = nmp.block([   [  subMatrix[i,j] for j in range(nColsMajor)  ] for i in range(nRowsMajor)   ] )

		if arg_filterZeros:			 
			quadrant = self.filter_zero_rows_columns(quadrant)

		return quadrant
	#END	 

	def filter_zero_rows_columns(self, arg_matrix):
		#record the initial size
		initShape = nmp.shape(arg_matrix)

		zeroIdx = list(filter(lambda row : nmp.all(nmp.isclose(arg_matrix[row,:] , 0.0)) , nmp.arange(nmp.shape(arg_matrix)[0])))
		allIdx = nmp.ones((nmp.shape(arg_matrix)[0]), dtype=bool)
		allIdx[zeroIdx] = False 
		arg_matrix = arg_matrix[allIdx,:]

		allIdx = nmp.ones((nmp.shape(arg_matrix)[1]), dtype=bool)
		zeroIdx = list(filter(lambda col : nmp.all(nmp.isclose(arg_matrix[:,col] , 0.0)) , nmp.arange(nmp.shape(arg_matrix)[1])))
		allIdx[zeroIdx] = False
		arg_matrix = arg_matrix[:,allIdx]

		# get the final shape
		finalShape = nmp.shape(arg_matrix)

		if initShape != finalShape:
			Utils.printflush('A shape change has occured ({},{}) -> ({}, {})'.format(*initShape, *finalShape))

		return arg_matrix
	#END	 

	def update_subMatrix(self, arg_pairString, arg_verbose):
		"""Fill in the matrix with correlation data 
		corresponding to the value for the input arg_pairString (key)"""
	
		if len(self.indexPairList) == 0 or len(self.listOfBeads) == 0:
			raise ValueError("Index Pair List or the beads representing the molecule is not defined")
	
		N = len(self.listOfBeads)
		numFrames = len(self.hostDataContainer.trajSnapshots)
		self.subMatrixDict[arg_pairString] = nmp.zeros( (N, N, 3, 3) )
		
		for i,j in self.indexPairList:
			if i <= j:
				if arg_verbose >= 5:
					Utils.printflush('[{:>5},{:>5}]'.format(i,j), end = ' ')

				iBead = self.listOfBeads[i]
				jBead = self.listOfBeads[j]

				if arg_pairString == 'FF':
					self.subMatrixDict["FF"][i][j] = self.generate_block_matrix(arg_bead_i = iBead, arg_bead_j = jBead, arg_pairString="FF", arg_frameList = nmp.arange(numFrames))
					self.subMatrixDict["FF"][j][i] = self.subMatrixDict["FF"][i][j].T

				elif arg_pairString == "TT":
					self.subMatrixDict["TT"][i][j] = self.generate_block_matrix(arg_bead_i = iBead, arg_bead_j = jBead, arg_pairString="TT", arg_frameList = nmp.arange(numFrames))
					self.subMatrixDict["TT"][j][i] = self.subMatrixDict["TT"][i][j].T
		if arg_verbose >= 3:
			Utils.printflush("Finished updating submatrix {}".format(arg_pairString))
		return
	#END	 
	
	def generate_block_matrix(self, arg_bead_i, arg_bead_j, arg_pairString, arg_frameList):
		""" 
		For a pair of beads/UAs, it will return a 3x3 matrix for 
		each form of arg_pairString. The matrix and its transpose
		when placed correctly into the bigger force-torque covariance 
		matrix will create a block symmetric matrix in each quadrant.

		For each timestep get a cross matrix for the UA pair and then 
		add them serially. In the end, divide the sum by numFrames to
		obtain the average.

		Accepted arg_pairString : ['FF', 'TT']
		"""

		if not arg_pairString.upper() in self.subMatrixDict.keys():
			raise ValueError("Invalid pairString {}. Should be one of these : 'FF', 'TT' ".format(arg_pairString) )

		numFrames = len(self.hostDataContainer.trajSnapshots)

		if arg_pairString.upper() == "FF":
			ffMatrix_ij = nmp.zeros( (3,3) )
			for iFrame in arg_frameList:
				outerProdMatrix = nmp.outer(arg_bead_i.totalWeightedForces[iFrame], arg_bead_j.totalWeightedForces[iFrame])
				ffMatrix_ij = nmp.add(ffMatrix_ij, outerProdMatrix)

			ffMatrix_ij /= numFrames
			return ffMatrix_ij

		if arg_pairString.upper() == "TT":
			ttMatrix_ij = nmp.zeros( (3,3) )
			for iFrame in arg_frameList:
				outerProdMatrix = nmp.outer(arg_bead_i.totalWeightedTorques[iFrame,:], arg_bead_j.totalWeightedTorques[iFrame,:])
				ttMatrix_ij = nmp.add(ttMatrix_ij, outerProdMatrix)

			ttMatrix_ij /= numFrames
			return ttMatrix_ij
	#END

	def get_name(self):
		"""
		Returns the name/descriptor string of this bead collection.
		"""
		return self.name
	#END

	def get_bead_names(self):
		"""
		Returns a string with the identifying `atom names` of the individual comprising beads.
		"""
		retStr = ""
		for iBead in self.listOfBeads:
			retStr = "{} {}".format(retStr, iBead.beadName)

		return retStr
	#END

	def get_bead_resnames(self):
		"""
		Returns a string with the identifying `res names` of the individual comprising beads.
		"""
		retStr = ""
		for iBead in self.listOfBeads:
			retStr = "{} {}".format(retStr, iBead.beadResn)

		return retStr
	#END

	def get_bead_resids(self):
		"""
		Returns a string with the identifying `residue ids` of the individual comprising beads.
		"""
		retStr = ""
		for iBead in self.listOfBeads:
			retStr = "{} {}".format(retStr, iBead.beadResi)

		return retStr
	#END

	def get_bead_chids(self):
		"""
		Returns a string with the identifying `chain ids` of the individual comprising beads.
		"""
		retStr = ""
		for iBead in self.listOfBeads:
			retStr = "{} {}".format(retStr, iBead.beadChain)

		return retStr
	#END

	def get_bead_positions(self):
		"""
		Returns a string with the coordinates (representative) of the individual comprising beads.
		"""
		retStr = ""
		for iBead in self.listOfBeads:
			bx, by, bz = iBead.position
			retStr = "{} {:.3f} {:.3f} {:.3f}".format(retStr, bx, by, bz)

		return retStr
	#END 

	def write_nmd_file(self, arg_nmdfile, arg_spectrum, arg_wfac):
		"""
		write a NMD format file for visualization of the eigen modes in VMD (NMWIZ pugin).
		"""
		Utils.printOut(arg_nmdfile, "title {}".format(self.get_name()))
		Utils.printOut(arg_nmdfile, "names {}".format(self.get_bead_names()))
		Utils.printOut(arg_nmdfile, "resnames {}".format(self.get_bead_resnames()))
		Utils.printOut(arg_nmdfile, "chids {}".format(self.get_bead_chids()))
		Utils.printOut(arg_nmdfile, "resnums {}".format(self.get_bead_resids()))
		Utils.printOut(arg_nmdfile, "coordinates {}".format(self.get_bead_positions()))
		for mode in arg_spectrum:
			Utils.printOut(arg_nmdfile, "# *** {} {:.3f}".format(mode.modeIdx, mode.modeFreq/CONST.C_LIGHT))
			Utils.printOut(arg_nmdfile, mode.get_line_nmd(arg_wfac = arg_wfac))

	#END
		

# END
	


class Bead():
	"""
	It is a collection of atoms and it will contain 
	the total forces and torques (weighted) for each of its
	atoms per frame. It has a hostDataContainer from where 
	it fetches some of the information about its constituent atoms.
	"""

	def __init__(self, arg_atomList, arg_numFrames, arg_hostDataContainer\
					 , arg_beadName = "", arg_beadResn = "", arg_beadResi = ""\
					 , arg_beadChid = "", **kwargs):
		
		# set of atom indices that form this bead
		self.atomList = arg_atomList

		# a 3D coordinate representing its position in space
		self.position = None

		# dummy information about beads
		self.beadName = arg_beadName
		self.beadResn = arg_beadResn
		self.beadResi = arg_beadResi
		self.beadChain = arg_beadChid		
		
		#mass weighted local forces	
		self.totalWeightedForces = nmp.zeros( (arg_numFrames,3) )	 
		
		#mass weighted local torques
		self.totalWeightedTorques = nmp.zeros( (arg_numFrames,3) )   
		
		# data container
		self.hostDataContainer = arg_hostDataContainer
	# END

	def reset_totalWeightedVectors(self, arg_shape):
		self.totalWeightedForces = nmp.zeros( arg_shape )
		self.totalWeightedTorques = nmp.zeros( arg_shape )
		return	
	# END

	def get_num_atoms(self):
		return len(self.atomList)
	#ND

	def get_total_mass(self):
		totalMass = 0.
		for aid in self.atomList:
			iMass = self.hostDataContainer.universe.atoms.masses[aid]
			totalMass += iMass

		return totalMass
	# END

	def get_center_of_mass_lab(self, arg_frame):
		""" 
		compute and return the center of mass of 
		the input atoms in the lab frame
		"""
		
		com = nmp.zeros((3))

		totalMass = nmp.sum(nmp.asarray([self.hostDataContainer.universe.atoms.masses[iAtom] for iAtom in self.atomList]))

		for iAtom in self.atomList:
			iMassCoord = self.hostDataContainer.universe.atoms.masses[iAtom] * self.hostDataContainer._labCoords[arg_frame][iAtom]
			com = com + iMassCoord

		com /= totalMass
		return com
	# END

	def get_center_of_mass_local(self, arg_frame):
		""" 
		compute and return the center of mass of 
		the input atoms in the local frame
		"""
		
		com = nmp.zeros((3))

		totalMass = nmp.sum(nmp.asarray([self.hostDataContainer.universe.atoms.masses[iAtom] for iAtom in self.atomList]))

		for iAtom in self.atomList:
			iMassCoord = self.hostDataContainer.universe.atoms.masses[iAtom] * self.hostDataContainer.localCoords[arg_frame][iAtom]
			com = com + iMassCoord

		com /= totalMass
		return com
	# END

	def get_moment_of_inertia_tensor_local(self, arg_localBasis, arg_frame):
		""" 
		return the 3 x 3 moment of inertia tensor for 
		the body defined by the collection of atoms in 
		the bead. The MOIs are computed in the input basis. 
		The lab coordinates of the atoms will be transformed 
		into this basis and then the moment would be computed. 

		NB: Moment of inertia tensor expression taken from MDanalysis website.
		It agrees with the algebric expression provided elsewhere
		"""

		#check that the input basis is of shape 4,3
		try:
			assert(nmp.shape(arg_localBasis) == (4,3))
		except:
			Utils.printflush('The dimensions of the input basis must be 4 x 3')

		# now compute the local coords for each atom in the input list in that frame
		# store them
		atomLocalCoords = nmp.ndarray( (len(self.atomList),3) )
		atomMasses = nmp.zeros( (len(self.atomList)) )

		atIdx = 0
		for iAtom in self.atomList:
			atomLocalCoords[atIdx] = self.hostDataContainer._labCoords[arg_frame][iAtom] - arg_localBasis[-1,]
			atomLocalCoords[atIdx] = arg_localBasis[0:3,] @ atomLocalCoords[atIdx]

			atomMasses[atIdx] = self.hostDataContainer.universe.atoms.masses[iAtom]

			atIdx += 1


		# generate the  MOI tensor
		tensor = nmp.zeros((3,3))

		sumDyadicCoordBasis = nmp.zeros((3,3))
		for iAxis in range(3):
			sumDyadicCoordBasis = nmp.add(sumDyadicCoordBasis, nmp.outer(arg_localBasis[iAxis,], arg_localBasis[iAxis,]))

		for atIdx in range(len(self.atomList)):
			iMass = atomMasses[atIdx]
			iLocalCoord = atomLocalCoords[atIdx]

			iDyadicLocalCoord = nmp.outer(iLocalCoord, iLocalCoord)
			
			iMatrix = nmp.dot(iLocalCoord,iLocalCoord) * sumDyadicCoordBasis
			iMatrix = nmp.subtract(iMatrix,iDyadicLocalCoord)
			iMatrix = iMass * iMatrix
			
			tensor = nmp.add(tensor, iMatrix)


		return tensor
#END	

	def print_atomList(self):
		""" Prints the atom indices in a particular fashion"""
		baseMol = self.hostDataContainer.universe
		Utils.printflush('[{}]'.format(len(self.atomList)), end = ' : ')
		for iAtom in self.atomList:
			if iAtom in baseMol.select_atoms("name H*").indices:
				Utils.printflush('{}'.format(iAtom),end = ' ')
			else:
				Utils.printflush('({})'.format(iAtom), end = ' ')
		Utils.printflush('')
		return

#END				

