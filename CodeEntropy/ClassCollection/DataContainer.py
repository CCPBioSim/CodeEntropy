import numpy as nmp
from CodeEntropy.Trajectory import TrajectoryFrame as TF
from CodeEntropy.Trajectory import TrajectoryConstants as TCON
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.ClassCollection import BondStructs

#warining
class ComplexWarning(Warning):
	"""Waring for complex number result
	"""
	pass

class DataContainer(object):
	""" 
	This is the main data container for CodeEntropy Solute calculation
	This host framewise information of positions and forces on all the atoms, in lab and local frames. The vectors in the local frames are obtained after transforming the lab frame vectors  using orthonormal bases that also stored framewise for each atom. 

	The properties of its molecule (base molecule) is linked to a molecule class object that contains the info about its topology. 
	"""

	def __init__(self, u, start=None, end=None, step=1):
		"""
		Create a data container for CodeEntropy and load data from MDAnalysis.Universe
		
		Parameters
		----------
		u : MDAnalyse.Universe
			A Universe object will all topology, dihedrals,coordinates and force information Check ``Example/create_new_universe.py`` on how to create a universe from existing data
		start : int or None, Optional, default: None
			Frame id to start analysis. Default None will start from frame 0
		end : int or None, Optional, default: None
			Frame id to end analysis. Default None will end at last frame
		step : int, Optional, default: 1
			Steps between frame.
		"""
		self.universe = u
		self.numFrames = len(self.universe.trajectory)
		self.frameIndices = []     # a list of integer values
		self.trajSnapshots = []  # a list of instances of TrajectoryFrame class

		num_atom = self.universe.atoms.n_atoms

		self.isCAtomArray = nmp.zeros(num_atom)
		self.isOAtomArray = nmp.zeros(num_atom)
		self.isNAtomArray = nmp.zeros(num_atom)
		self.isCaAtomArray = nmp.zeros(num_atom)
		self.isBBAtomArray = nmp.zeros(num_atom)
		self.isHydrogenArray = nmp.zeros(num_atom)

		selection_pair = [
			(self.isCAtomArray, "name C"),
			(self.isOAtomArray, "name O"),
			(self.isNAtomArray, "name N"),
			(self.isCaAtomArray, "name CA"),
			(self.isBBAtomArray, "backbone"),
			(self.isHydrogenArray, "name H*")
		]
		for item in selection_pair:
			idx_list = u.atoms.select_atoms(item[1]).indices
			for id in idx_list:
				item[0][id] = 1

		self.dihedralArray = set()
		self.dihedralTable = dict()
		dihedList = u.dihedrals.indices
		for i in range(num_atom):
			self.dihedralTable[i] = set()
		
		for dihedrals in dihedList:
			newDih = BondStructs.Dihedral(dihedrals, self)
			self.add_dihedral(newDih)

		#reading trajectorys into memory because MDanalysis reads values on the fly which might slow down processing speed as these values are accessed multiple times
		if start == None:
			start = 0
		if end == None:
			end = len(u.trajectory)
		for i in range(int(start), int(end), int(step)):
			ts = u.trajectory[i]
			newFrame = TF.TrajectoryFrame(arg_frameIndex = ts.frame, \
													arg_vectorDim = TCON.VECTORDIM)
			newFrame.set_numAtoms(ts.n_atoms)
			newFrame.set_timeStep(ts.time)
			newFrame.initialize_matrices(arg_hasVelocities = ts.has_velocities, arg_hasForces = ts.has_forces)
			newFrame.value["coordinates"] = nmp.array(ts.positions)
			if ts.has_velocities:
				newFrame.value["velocities"] = nmp.array(ts.velocities)
			if ts.has_forces:
				newFrame.value["forces"] = nmp.array(ts.forces)
			self.trajSnapshots.append(newFrame)
			self.frameIndices.append(ts.frame)

		self.print_attributes()

		self._labCoords = nmp.ndarray( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 3) )
		self._labForces = nmp.ndarray( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 3) )
		
		self.translationAxesArray = nmp.ndarray ( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 1+3, 3) ) # 4rth row is the origin
		self.rotationAxesArray = nmp.ndarray ( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 1+3, 3) ) # 4rth row is the origin

		self.localCoords = nmp.ndarray( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 3) )
		self.localForces = nmp.ndarray( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 3) )
		self.localTorques = nmp.ndarray( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 3) )
		# read the coords and forces from the trajectory
		# and store them in the mainContainer
		# it is a gromacs trajectory
		self.initialize_ndarrays()

		for i in range(len(self.trajSnapshots)):
			self._labCoords[i] = self.trajSnapshots[i].value['coordinates']
			self._labForces[i] = self.trajSnapshots[i].value['forces']

		
	def print_attributes(self):
		# print("{:<20s} : {}".format("Molecule name", self.molecule.name))
		print("{:<20s} : {}".format("Number of atoms", self.universe.atoms.n_atoms))
		print("{:<20s} : {}".format("Number of frames", len(self.trajSnapshots)))
		
		return
  
	def initialize_ndarrays(self):
		
		""" The number of frames for which the container will hold the data must be provided before hand"""
		assert(len(self.trajSnapshots) >= 0 and len(self.frameIndices) == len(self.trajSnapshots))

		self._labCoords = nmp.ndarray( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 3) )
		self._labForces = nmp.ndarray( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 3) )
		
		self.translationAxesArray = nmp.ndarray ( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 1+3, 3) ) # 4rth row is the origin
		self.rotationAxesArray = nmp.ndarray ( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 1+3, 3) ) # 4rth row is the origin

		self.localCoords = nmp.ndarray( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 3) )
		self.localForces = nmp.ndarray( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 3) )
		self.localTorques = nmp.ndarray( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 3) )

		
		return
		
	def reset_translationAxesArray(self):
		"""
		Reset the translational axes value of every atom per frame while maintaining the shape of the array.
		""" 
		self.translationAxesArray = nmp.ndarray ( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 4, 3) )
		return
	
	def reset_rotationAxesArray(self):
		"""
		Reset the rotational axes value of every atom per frame while maintaining the shape of the array.
		"""
		self.rotationAxesArray = nmp.ndarray ( (len(self.trajSnapshots), self.universe.atoms.n_atoms, 4, 3) )
		return

	def update_translationAxesArray_at(self, arg_frame, arg_atomList, arg_pAxes, arg_orig):
		"""
		Update the translational axes at a given frame for the selected atoms with the input values.
		"""
		self.translationAxesArray[arg_frame,arg_atomList] = nmp.vstack((arg_pAxes, arg_orig))	   
		return
	
	def update_rotationAxesArray_at(self, arg_frame, arg_atomList, arg_pAxes, arg_orig):
		"""
		Update the rotational axes at a given frame for the selected atoms with the input values.
		"""
		self.rotationAxesArray[arg_frame,arg_atomList] = nmp.vstack((arg_pAxes, arg_orig))   
		return
	

	def update_localCoords(self, arg_type, arg_atomList):
		if arg_type.upper() not in ["T","R"]:
			raise ValueError('Type {} for axes system does not exist. Chose T(translation) or R(rotation)'.format(arg_type))
			return
		
		if arg_type.upper() == "T":
			for iFrame in range(len(self.trajSnapshots)):
				for iAtom in arg_atomList:
					#translation
					self.localCoords[iFrame,iAtom] = self._labCoords[iFrame,iAtom] - self.translationAxesArray[iFrame,iAtom][-1,]
					self.localCoords[iFrame,iAtom] = self.translationAxesArray[iFrame,iAtom][0:3,] @ self.localCoords[iFrame,iAtom]
		elif arg_type.upper() == "R":
			for iFrame in range(len(self.trajSnapshots)):
				for iAtom in arg_atomList:
					#rotation
					self.localCoords[iFrame,iAtom] = self._labCoords[iFrame,iAtom] - self.rotationAxesArray[iFrame,iAtom][-1,]
					self.localCoords[iFrame,iAtom] = self.rotationAxesArray[iFrame,iAtom][0:3,] @ self.localCoords[iFrame,iAtom]

		return
	#END

	def update_localCoords_of_all_atoms(self, arg_type):
		allAtoms = nmp.arange(self.universe.atoms.n_atoms)
		self.update_localCoords(arg_type, allAtoms)
		return
	#END

	def update_localForces(self, arg_type, arg_atomList):
		if arg_type.upper() not in ["T","R"]:
			raise ValueError('Type {} for axes system does not exist. Choose T(translation) or R(rotation)'.format(arg_type))
			return
	
		if arg_type.upper() == "T":
			for iFrame in range(len(self.trajSnapshots)):
				for iAtom in arg_atomList:
					self.localForces[iFrame, iAtom] = self.translationAxesArray[iFrame, iAtom][0:3,] @ self._labForces[iFrame, iAtom]
		elif arg_type.upper() == "R":
			for iFrame in range(len(self.trajSnapshots)):
				for iAtom in arg_atomList:
					self.localForces[iFrame, iAtom] = self.rotationAxesArray[iFrame, iAtom][0:3,] @ self._labForces[iFrame, iAtom]
								
		return
	#END

	def update_localForces_of_all_atoms(self, arg_type):
		allAtoms = nmp.arange(self.universe.atoms.n_atoms)
		self.update_localForces(arg_type, allAtoms)
		return
	#END

	def  get_minmax(self, arg_token, arg_indices, arg_frame = -1, arg_local=False):
		"""
		Return the minimum and maximum value of an array with name given  by 
		`arg_token` at frame `arg_frame` for selected indices.
		"""
		if arg_frame == -1:
			# assign the last frame
			arg_frame = self.frameIndices[-1]

		arg_token = arg_token.upper()
		if arg_token in ["X", "FX"]:
			axis = 0
		elif arg_token in ["Y", "FY"]:
			axis = 1
		elif arg_token in ["Z",  "FZ"]:
			axis = 2
		else:
			raise ValueError(f"Bad array name provided for  minmax operation. ({arg_token})")

		arrType = 0
		if arg_token in ['X', 'Y', 'Z']:
			arrType = 1
			if arg_local:
				arrType = 2
		elif arg_token in ['FX', 'FY', 'FZ']:
			arrType = 3
			if arg_local:
				arrType = 4

		try:
			assert(arrType > 0)
		except:
			raise AssertionError(f'Bad array type assigned for minmax operation.({arrType})')

		if arrType == 1:
			retArr = [v3[axis] for v3 in self._labCoords[arg_frame][arg_indices]]
		elif arrType == 2:
			retArr = [v3[axis] for v3 in self.localCoords[arg_frame][arg_indices]]
		elif arrType == 3:
			retArr = [v3[axis] for v3 in self._labForces[arg_frame][arg_indices]]
		elif arrType == 4:
			retArr = [v3[axis] for v3 in self.localForces[arg_frame][arg_indices]]

		return (nmp.min(retArr), nmp.max(retArr))
#END

	def get_center_of_mass(self, arg_atomList, arg_frame):
		""" compute and return the center of mass of the input atoms in the lab frame"""
		com = nmp.zeros((3))

		totalMass = nmp.sum(nmp.asarray([self.universe.atoms.masses[iAtom] for iAtom in arg_atomList]))

		for iAtom in arg_atomList:
			iMassCoord = self.universe.atoms.masses[iAtom] * self._labCoords[arg_frame][iAtom]
			com = com + iMassCoord

		com /= totalMass
		return com

		# replace this with 
		# u.select_atoms("all").center_of_mass()
	#END


	def get_moment_of_inertia_tensor_lab(self, arg_atomList, arg_frame):
		""" return the 3 x 3 moment of inertia tensor for the body defined by the 
		collection of atoms in the input atom List. The MOIs are computed in 
		the lab frame after subtracting the COM of the atom selection from their lab coordinates.

		Moment of inertia tensor expression taken from MDanalysis website. It agrees with
		the algebric expression provided elsewhere"""

		# first compute the local coords for each atom in the input list in that frame
		# store them
		atomLocalCoords = nmp.ndarray( (len(arg_atomList),3) )
		atomMasses = nmp.zeros( (len(arg_atomList)) )

		atomsCOM = self.get_center_of_mass(arg_atomList, arg_frame)

		atIdx = 0
		for iAtom in arg_atomList:
			iLabCoord = self._labCoords[arg_frame][iAtom]

			# deduct com
			atomLocalCoords[atIdx] = iLabCoord - atomsCOM

			atomMasses[atIdx] = self.universe.atoms.masses[iAtom]
			atIdx += 1


		# generate the  MOI tensor
		tensor = nmp.zeros((3,3))

		# using standard cartesian basis for the lab frame
		basis = nmp.identity(3)

		sumDyadicCoordBasis = nmp.zeros((3,3))
		for iAxis in range(3):
			sumDyadicCoordBasis = nmp.add(sumDyadicCoordBasis, nmp.outer(basis[iAxis,], basis[iAxis,]))

		for atIdx in range(len(arg_atomList)):
			iMass = atomMasses[atIdx]
			iLocalCoord = atomLocalCoords[atIdx]

			iDyadicLocalCoord = nmp.outer(iLocalCoord, iLocalCoord)
			
			iMatrix = nmp.dot(iLocalCoord,iLocalCoord) * sumDyadicCoordBasis
			iMatrix = nmp.subtract(iMatrix,iDyadicLocalCoord)
			iMatrix = iMass * iMatrix
			
			tensor = nmp.add(tensor, iMatrix)


		return tensor
	#END    

	def get_principal_axes(self, arg_atomList, arg_frame, arg_sorted):
		""" 
		Returns a pricipal moments of inertia and a 3 x 3 matrix 
		with each row as the principal axes sorted in the descending order of the 
		corresponding eigen values (principal moments of inertia).
		"""

		#compute the moment of inertia for the imput list
		momentOfInertiaTensor = self.get_moment_of_inertia_tensor_lab(arg_atomList = arg_atomList, arg_frame = arg_frame)

		#diagonlaize
		try:
			pMomentsOfInertia, pAxes = Utils.diagonalize(momentOfInertiaTensor)
		except ComplexWarning:
			print("Moments of Inertia : ", pMomentsOfInertia)
			print("Principal Axes matrix (cols are eigen vectors):")
			print(pAxes)
			
		# IMPORTANT :::: Columns of pAxes are the eigen vectors and therefore the pricipal axes
		# When returning, return the transpose so that the rows are the pricipal axes
		# and the format is consistent with 4x3 matrix convention used for coordinate frames
		# thoughout.
		if arg_sorted:
			#sorted order (descending)
			sortOrder = nmp.argsort(-pMomentsOfInertia)

			#create a new array of the sme shape as pAxes and copy columns from it in the order in sortOrder
			pAxesSorted = nmp.zeros(nmp.shape(pAxes))

			# transposing here
			for i,j in zip(range(3),sortOrder):
				pAxesSorted[i,:] = pAxes[:,j]

			return -nmp.sort(-pMomentsOfInertia), pAxesSorted

		else:
			return pMomentsOfInertia, nmp.transpose(pAxes)

	def add_dihedral(self, arg_dihedral):
		# add the input dihedral to the value list associated with the indices of the atoms that form it.
		for atIdx in arg_dihedral.atomList:
			# Utils.printflush('Adding dihedral {} to the list for atom {}'.format(arg_dihedral, atIdx))
			self.dihedralTable[atIdx].add(arg_dihedral)
		
		self.dihedralArray.add(arg_dihedral)