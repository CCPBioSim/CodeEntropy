import numpy as nmp 

class ConformationVector(object):
	""" A single column vector of certain length and binary in nature that will tell the configuration of a certain instance of
	the ConformationEntity Class at some instant of time """

	def __init__(self, arg_vector):
		""" I am using quantum mechanics lingo here for fun """
		self.ket = bytearray(arg_vector)
		self.order = len(arg_vector)


class ConformationEntity(object):
	""" A class depicting an entity whose configurations and their distribution will be measured to estimate the probabilities 
	for computing the topographical entropy. An object of this class could be a Dihedral angle, an amino acids backbone phi/psi 
	angles, a whole protein, etc. This object will have a conformation/configuration vector (binary in 0 or 1) at each frame that
	will tell its status at that frame. E.g. A dihedral can be described by a conformation vector ordered as |g+, g-, t> and at some 
	time 't' its values is |0, 0, 1>. This means that at that time, te dihedral was in 'trans' state. Same can be done for a residue
	in a protein with flags indicating which quadrant in the Ramachandran plot the residue's backbone was in. """

	def __init__(self, arg_order, arg_numFrames, **kwargs):
		self.timeSeries = -1 * nmp.ones((arg_order, arg_numFrames))   # a 2D array where each column represents a ConformationVector at a certain time/snapshot (initialized with absurd -1's).
		for k, v in kwargs.items():
			setattr(self, k, v)
			# print('{} -> {}'.format(k, v))

		




