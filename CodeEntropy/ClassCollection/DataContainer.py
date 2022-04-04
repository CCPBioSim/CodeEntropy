import numpy as nmp
from CodeEntropy.Trajectory import TrajectoryFrame as TF
from CodeEntropy.Trajectory import TrajectoryConstants as TCON

class DataContainer(object):
	""" 
	This is the main data container that has the framewise information of
	positions and forces on all the atoms, in lab and local frames. The vectors
	in the local frames are obtained after transforming the lab frame vectors 
	using orthonormal bases that also stored framewise for each atom. 

	The properties of its molecule (base molecule) is linked to a molecule class object
	that contains the info about its topology. 
	"""

	def __init__(self, u):
		
		self.universe = u
		self.numFrames = len(self.universe.trajectory)
		self.frameIndices = []     # a list of integer values
		self.trajSnapshots = []  # a list of instances of TrajectoryFrame class
		
		#reading trajectorys into memory because MDanalysis reads values on the fly which might slow down processing speed as these values are accessed multiple times
		for ts in u.trajectory:
			newFrame = TF.TrajectoryFrame(arg_frameIndex = ts.frame, \
													arg_vectorDim = TCON.VECTORDIM)
			newFrame.set_numAtoms(ts.n_atoms)
			newFrame.set_timeStep(ts.time)
			newFrame.initialize_matrices(arg_hasVelocities = ts.has_velocities, arg_hasForces = ts.has_forces)
			newFrame.value["coordinates"] = ts.positions.flatten()
			if ts.has_velocities:
				newFrame.value["velocities"] = ts.velocities.flatten()
			if ts.has_forces:
				newFrame.value["forces"] = ts.forces.flatten()
			self.trajSnapshots.append(newFrame)
			self.frameIndices.append(ts.frame)

		self._labCoords = nmp.ndarray( (len(self.trajSnapshots), self.universe.n_atoms, 3) )
		self._labForces = nmp.ndarray( (len(self.trajSnapshots), self.universe.n_atoms, 3) )
		
		self.translationAxesArray = nmp.ndarray ( (len(self.trajSnapshots), self.universe.n_atoms, 1+3, 3) ) # 4rth row is the origin
		self.rotationAxesArray = nmp.ndarray ( (len(self.trajSnapshots), self.universe.n_atoms, 1+3, 3) ) # 4rth row is the origin

		self.localCoords = nmp.ndarray( (len(self.trajSnapshots), self.universe.n_atoms, 3) )
		self.localForces = nmp.ndarray( (len(self.trajSnapshots), self.universe.n_atoms, 3) )
		self.localTorques = nmp.ndarray( (len(self.trajSnapshots), self.universe.n_atoms, 3) )
		
	def print_attributes(self):
		# print("{:<20s} : {}".format("Molecule name", self.molecule.name))
		print("{:<20s} : {}".format("Number of atoms", self.universe.n_atoms))
		print("{:<20s} : {}".format("Number of frames", len(self.trajSnapshots)))
		
		return
  
	def initialize_ndarrays(self):
		
		""" The number of frames for which the container will hold the data must be provided before hand"""
		assert(len(self.trajSnapshots) >= 0 and len(self.frameIndices) == len(self.trajSnapshots))

		self._labCoords = nmp.ndarray( (len(self.trajSnapshots), self.universe.n_atoms, 3) )
		self._labForces = nmp.ndarray( (len(self.trajSnapshots), self.universe.n_atoms, 3) )
		
		self.translationAxesArray = nmp.ndarray ( (len(self.trajSnapshots), self.universe.n_atoms, 1+3, 3) ) # 4rth row is the origin
		self.rotationAxesArray = nmp.ndarray ( (len(self.trajSnapshots), self.universe.n_atoms, 1+3, 3) ) # 4rth row is the origin

		self.localCoords = nmp.ndarray( (len(self.trajSnapshots), self.universe.n_atoms, 3) )
		self.localForces = nmp.ndarray( (len(self.trajSnapshots), self.universe.n_atoms, 3) )
		self.localTorques = nmp.ndarray( (len(self.trajSnapshots), self.universe.n_atoms, 3) )

		
		return
		
	def reset_translationAxesArray(self):
		"""
		Reset the translational axes value of every atom per frame while maintaining the shape of the array.
		""" 
		self.translationAxesArray = nmp.ndarray ( (len(self.trajSnapshots), self.universe.n_atoms, 4, 3) )
		return
	
	def reset_rotationAxesArray(self):
		"""
		Reset the rotational axes value of every atom per frame while maintaining the shape of the array.
		"""
		self.rotationAxesArray = nmp.ndarray ( (len(self.trajSnapshots), self.universe.n_atoms, 4, 3) )
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
		allAtoms = nmp.arange(self.universe.n_atoms)
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
		allAtoms = nmp.arange(self.universe.n_atoms)
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






