""" A module/collection of classes that store information 
about the bonded properties of a molecule. E.g. bonds, angles, dihedrals"""

import numpy as nmp
import sys

from CodeEntropy.FunctionCollection import GeometricFunctions as GF
from CodeEntropy.FunctionCollection import CustomFunctions as CF
from CodeEntropy.ClassCollection import CustomDataTypes as CDT
from CodeEntropy.FunctionCollection import Utils

class BondedStructure(object):
	"""
	A parent class designed to store information about 
	bonded atoms in a structure. Bonded atoms can form
	a bond, an angle, a dihedral and an improper.
	"""

	def __init__(self, atomIndices, arg_container):
		self.atomList = atomIndices
		self.container = arg_container
		self.qvalue = None

	def __hash__(self):
		return hash(('{}_'*len(self.atomList)).format(*sorted(self.atomList)))

	def set_quantity(self, value):
		"""Set the value for the relevant geomtric quantity associated
		with the bonded structure. For bonds - distance, angles - angle, 
		dihedral - angle, improper - angle.
		"""
		self.qvalue = value

	def get_quantity(self):
		"""Get the value for the relevant geomtric quantity associated
		with the bonded structure. For bonds - distance, angles - angle, 
		dihedral - angle, improper - angle.
		"""
		return self.qvalue

	def is_heavy(self):
		"""Return true if all of the atoms are not
		hydrogens"""
		retVal = True 
		for iAtom in self.atomList:
			if self.container.isHydrogenArray[iAtom]:
				retVal = False
				break
		return retVal
#END

class Bond(BondedStructure):
	""" Bond between two atoms (preferably covalent). 
	May serve as a parent of 'contact' """

	def __init__(self, arg_atomIndices, arg_container):
		super().__init__(arg_atomIndices, arg_container)
		self.atom1, self.atom2 = arg_atomIndices # indices of 2 atoms

#END

class Angle(BondedStructure):
	""" Angle of three atoms. """
	def __init__(self, arg_atomIndices, arg_container):
		super().__init__(arg_atomIndices, arg_container)
#END

class Dihedral(BondedStructure):
	""" Dihedral composed of 4 positions (2 planes with an intersecting line) """

	def __init__(self, arg_atomIndices, arg_container):
		super().__init__(arg_atomIndices, arg_container)
		self.atom1, self.atom2, self.atom3, self.atom4 = arg_atomIndices	 # indices of the 4 atoms

	def __str__(self):
		# return 'Dihedral: {} -> {} -> {} -> {}'.format(self.atom1, self.atom2, self.atom3, self.atom4)
		return '{} {} {} {} '.format(*self.atomList)


	def __lt__(self, arg_other):
		return sorted(self.atomList[1:3]) < sorted(arg_other.atomList[1:3])

	def __gt__(self, arg_other):
		return sorted(self.atomList[1:3]) > sorted(arg_other.atomList[1:3])

	# def __eq__(self, arg_other):
	#	 """ Two dihedrals are deemed equal if their 2,3 atoms are the same """
	#	 return sorted(self.atomList[1:3]) == sorted(arg_other.atomList[1:3])
	

	def is_BB_dihedral(self):
		""" Based on the information in the base molecule's arrays, see if 2-3 atoms 
		in this dihedral are BB atoms """
		retBool = self.container.isBBAtomArray[self.atom2] 
		retBool = retBool and self.container.isBBAtomArray[self.atom3] 
		return retBool
	#END

	def is_BB_psi(self):
		""" Checks if a heavy dihedral is BB psi (X-C-----Ca-Y)"""
		if self.is_BB_dihedral():
			if self.container.isCAtomArray[self.atom2] and self.container.isCaAtomArray[self.atom3]:
				return True
			elif self.container.isCAtomArray[self.atom3] and self.container.isCaAtomArray[self.atom2]:
				return True
			else:
				return False
		else:
			# if not a BB dihedral, it aint a PHI or PSI
			return False
	#END

	def is_BB_phi(self):
		""" Checks if a heavy dihedral is BB phi (X-Ca-----N-Y)"""
		if self.is_BB_dihedral():
			if self.container.isNAtomArray[self.atom2] and self.container.isCaAtomArray[self.atom3]:
				return True
			elif self.container.isNAtomArray[self.atom3] and self.container.isCaAtomArray[self.atom2]:
				return True
			else:
				return False
		else:
			# if not a BB dihedral, it aint a PHI or PSI
			return False
	#END

	def is_from_same_residue(self):
		""" Based on the information in the base molecule's arrays, see if the two central atoms (2,3)
		in this dihedral belong to the same residue. Return the residue index if yes, else -100"""
		if self.container.universe.atoms.resids[self.atom2] ==\
		self.container.universe.atoms.resids[self.atom3]:
			return self.container.universe.atoms.resids[self.atom2]
		else:
			return -10000
	#END

	def get_dihedral_angle_lab(self, arg_frame):
		""" Using the function compute_dihedral in GeometricFunctions module, return the dihedral for this quadruplet in the lab frame"""
		r1 = self.container._labCoords[arg_frame][self.atom1]
		r2 = self.container._labCoords[arg_frame][self.atom2]
		r3 = self.container._labCoords[arg_frame][self.atom3]
		r4 = self.container._labCoords[arg_frame][self.atom4]

		return GF.compute_dihedral(r1, r2, r3, r4)
	#END

	def get_dihedral_angle_local(self, arg_frame):
		""" Using the function compute_dihedral in GeometricFunctions module, return the dihedral for this quadruplet in the local frame"""
		r1 = self.container.localCoords[arg_frame][self.atom1]
		r2 = self.container.localCoords[arg_frame][self.atom2]
		r3 = self.container.localCoords[arg_frame][self.atom3]
		r4 = self.container.localCoords[arg_frame][self.atom4]

		return GF.compute_dihedral(r1, r2, r3, r4)
	#END

	def get_state_ts(self, arg_verbose = 3, arg_bw = 30):
		"""
		Create a state vector, showing the state in which the input dihedral is
		as a function of time.
		The function creates a histogram from the timeseries of the dihedral angle values
		and identifies points of dominant occupancy (called CONVEX TURNING POINTS).
		Based on the identified TPs, states are assigned to each configuration of the dihedral.
		
		Return
		------
		A timeseries with integer labels describing the state at each point in time.
		
		"""
		cosBW = nmp.cos(nmp.deg2rad(arg_bw))
		numFrames = len(self.container.trajSnapshots)
		phiTS = nmp.zeros(numFrames)
		stateTS = nmp.zeros(numFrames)
		
		for iFrame in range(numFrames):

			# fetch the dihedral value at that frame
			phi = self.get_dihedral_angle_lab(arg_frame = iFrame)
			if phi < 0:
				phi += 360
				
			phiTS[iFrame] = phi
			
		# create a histogram using numpy
		popul, binEdges = nmp.histogram(a = phiTS, \
										bins = nmp.linspace(start=0,stop=360, num = 1 + int(360/arg_bw), endpoint=True))
		binVal = [0.5 * (binEdges[i] + binEdges[i+1]) for i in range(0,len(popul))]

		# create a circular doubly linked list of the bin elements.
		# CCDL data structure will allow to iterate through the list
		# without worrying about histogram's end-points
		histCDLL = CDT.CircularDoublyLinkedList()
		for ele in zip(binVal, popul):
			newNode = CDT.DoublyLinkedListNode(arg_value=ele)
			histCDLL.append(newNode)
		
		# identify "convex turning-points" and populate a list of TPs
		# TP := a bin whose neighboring bins have smaller population
		tpList = []
		for iNode in histCDLL.iterator_cw():
			prevBin, prevPopul = iNode.previousNode.value
			currBin, currPopul = iNode.value
			nextBin, nextPopul = iNode.nextNode.value

			if currPopul >= prevPopul and currPopul >= nextPopul:
				# the equality condition is greedy
				# it is expected to put empty bins with 
				# empty neighbors in the list.
				# They will be filtered out in the next step

				# Another condition is that TPs should be at least arg_bw (default: 30 deg) apart
				# Keepin gin mind that angle is cicularly preiodic but the histogram's x-axis
				# is linear, the difference is measured by comparing the dot product of 
				# e^i(angle1) and e^i(agnle2) with cosine(arg_bw). Isnt that COOL?

				newTP = iNode.value
				newAngle = newTP[0]
				newVec = CF.euler_vector(arg_r = 1., arg_theta = newAngle)
				goodTP = True
				for tp in tpList:
					tpAngle = tp[0]
					tpVec = CF.euler_vector(arg_r = 1., arg_theta = tpAngle)
					
					dotProd = CF.dot_product(newVec, tpVec)
					if nmp.isclose(dotProd, cosBW) or dotProd > cosBW:
						goodTP = False

				if goodTP:
					tpList.append(newTP)
					
		# filter out the empty turning points
		tpList = list(filter(lambda tp: tp[1] != 0, tpList))

		# TP angles
		tpVals = [tp[0] for tp in tpList]
		
		if arg_verbose > 3:
			Utils.printflush('{} turning point(s) found for dihedral {}.'.format(len(tpVals), self.atomList), end=' ')
		
		if arg_verbose >= 5:
			Utils.printflush(':', end=' ')
			for v in tpVals:
				Utils.printflush(v, end=' ')
			Utils.printflush('')

		# go through each frame again and assign state
		for iFrame, iPhi in enumerate(phiTS):
			# find the TP that the snapshot is least distant from
			diffList = [abs(iPhi - tp) for tp in tpVals]
			stateTS[iFrame] = nmp.argmin(diffList)
		
		return stateTS