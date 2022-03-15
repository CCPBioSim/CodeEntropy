import numpy as nmp

class TrajectoryFrame(object):
	""" A frame from a trajectory which contains 
	information about its coordinates (velecities and forces if present). 
	It also contains the metadata at that time."""

	def __init__(self, arg_frameIndex, arg_vectorDim = 3):

		""" Define an instance by providing a frame Index (arg_frameIndex),
		and the molecule, to which the data is related to """

		self.frameIndex = arg_frameIndex
		self.vectorDim = arg_vectorDim

		# other information/data about the frame will be received later
		# and they will populate the 'value' dictionary
		self.value = {}
		# self.valueKeys = ['numAtoms', 'timeStep', 'metaData', 'hasCoordinates' , 'coordinates', 'hasVelocities', 'velocities', 'hasForces', 'forces']
		# for vKey in self.valueKeys:
		# 	self.value[vKey] = None

		return

	def set_attribute(self, **kwargs):
		for prop, val in kwargs.items():
			setattr(self,prop, val)

	def set_numAtoms(self, arg_numAtoms):
		self.value['numAtoms'] = arg_numAtoms
		return

	def set_timeStep(self, arg_frameTimeStep):
		self.value['timeStep'] = arg_frameTimeStep
		return

	def set_metadata(self, arg_metadataDict):
		# storage for frame's metadata (dictionary)
		# keys will be determined the software that 
		# was used to generate the trajectory
		self.value['metaData'] = arg_metadataDict
		return

	def initialize_matrices(self, arg_hasVelocities, arg_hasForces):
		""" Designed for Gromacs's trajectory file """ 
		try:
			assert(self.value['numAtoms'] > 0 )
		except:
			raise ValueError('Number of atoms should be more than 0 ({} found)'.format(self.value['numAtoms']))
			return 

		self.value['hasCoordinates'] = True
		self.value['coordinates'] = []

		# velocity storage (optional)
		if arg_hasVelocities : 
			self.value['hasVelocities'] = True
			self.value['velocities'] = []

		# force storage (optional)
		if arg_hasForces : 
			self.value['hasForces'] = True
			self.value['forces'] = []

		return

	def print_frame_info(self):
		for vKey, vVal in self.value.items():
			print('{:<20} : {}'.format(vKey, vVal))

		return

	