import numpy as nmp
from CodeEntropy.Trajectory import (TPRReader, TRRReader, TrajectoryFrame)
from CodeEntropy.Trajectory import TrajectoryConstants as TCON
from CodeEntropy.ClassCollection import DataContainer
from CodeEntropy.ClassCollection import BondStructs
from CodeEntropy.Reader import Constants as CONST
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.FunctionCollection import UnitsAndConversions as UAC


def read_gromacs_input(arg_tprFile, \
	                   arg_trajFile, \
	                   arg_outFile, \
	                   arg_beginTime, \
	                   arg_endTime, \
	                   arg_stride, \
	                   arg_verbose = 0 ):
	
	""" 
	A function supplied with input topology/structure 
	and trajectory file from gromacs.
	The topology file should be .tpr and 
	the trajectory file should be .trr. 
	It should return an instance of BaseMolecule 
	and DataContainer which are linked to each other.
	"""
	
	# define an object of DataContainer class
	mainContainer = DataContainer.DataContainer()

	# topology file
	print('{:<40s} : {}'.format('Reading struct/topology file',arg_tprFile))

	topol = TPRReader.GroTopology(arg_tprFile, mainContainer)

	topol.read_tprFile()
	# print(' Reader: GromacsReader: Metadata', topol.metadata)

	# base molecule
	iMolId = 0
	myProtein = topol.moleculeList[iMolId]  # expecting the first molecule in the list to be the one that is also sampled in the trajectory file
	
	# get the number of atoms and molecules associated with iMolId
	iNumMol   = topol.metadata['molblock_numMols'][iMolId]
	iNumAtoms = topol.metadata['molblock_numAtoms'][iMolId]
	totalAtomPopulation = iNumMol * iNumAtoms
	
	if arg_verbose:
		print('{:<40s} : {}'.format('Number of atoms expected in trajectory', totalAtomPopulation))

	# initialize
	myProtein.initialize_element_info_arrays()

	# assign element types and priorities								
	myProtein.populate_element_info_arrays()


	# >>> print molinfo
	# myProtein.print_molecule_info()
	
	### FETCH BOND INFORMATION (RELOCATE) ###
	# read the gromacs interaction dictionary and fetch bond information from the appropriate ENUM
	# bond information could be obrtained from the values correspoding to 
	# 'F_BONDS', 'F_G96BONDS', 'F_MORSE', 'F_CUBICBONDS', 'F_CONNBONDS', 
	# 'F_HARMONIC', 'F_FENEBONDS', 'F_TABBONDS', 'F_TABBONDSNC', 'F_RESTRBONDS', 
	# 'F_CONSTR'

	for bondKey in ['F_BONDS', 'F_G96BONDS', 'F_MORSE', 'F_CUBICBONDS', 'F_CONNBONDS', 
	'F_HARMONIC', 'F_FENEBONDS', 'F_TABBONDS', 'F_TABBONDSNC', 'F_RESTRBONDS', 
	'F_CONSTR']:
		# print(bondKey, TCON.FTYPE_ENUM[bondKey])
		lenBondList, bondList = myProtein.groInteractionDict[TCON.FTYPE_ENUM[bondKey]]
		i = 0
		while i < lenBondList:
			# get the bond type and the atomtype indices of a molecule involved in the bond 
			bondType, atom1, atom2 = bondList[i:(i+3)]

			# add that bond type across all the copies of the molecule in the system
			# add_bond_tree must be called only after atoms have been labelled as H or non-H
			# the function tries to classify heavy-heavy atom bonds and heavy-hydrogen atom bonds internally
			# if the two atoms are hydrogens, it will spit a warning but not add them.
			aidI, aidJ = atom1, atom2

			while aidI < totalAtomPopulation and aidJ < totalAtomPopulation:
				myProtein.add_bond_tree(arg_aidI=aidI, arg_aidJ=aidJ,arg_priorityLevelI=-1,arg_priorityLevelJ=-1)
				aidI += myProtein.numAtomsPerCopy
				aidJ += myProtein.numAtomsPerCopy

			i += 3
	
	# FETCH BOND INFORMATION (F_SETTLE) ###
	lenBondList, bondList = myProtein.groInteractionDict[TCON.FTYPE_ENUM['F_SETTLE']]
	i = 0
	while i < lenBondList:
		# get the bond type and the atomtype indices of a molecule involved in the settle constraint
		bondType, atom1, atom2, atom3 = bondList[i:(i+4)]
		
		# if the two atoms are hydrogens, it will spit a warning but not add them.
		aidI, aidJ, aidK = atom1, atom2, atom3

		while aidI < totalAtomPopulation and aidJ < totalAtomPopulation:
			myProtein.add_bond_tree(arg_aidI=aidI, arg_aidJ=aidJ,arg_priorityLevelI=-1,arg_priorityLevelJ=-1)
			myProtein.add_bond_tree(arg_aidI=aidJ, arg_aidJ=aidK,arg_priorityLevelI=-1,arg_priorityLevelJ=-1)
			myProtein.add_bond_tree(arg_aidI=aidK, arg_aidJ=aidI,arg_priorityLevelI=-1,arg_priorityLevelJ=-1)
			aidI += myProtein.numAtomsPerCopy
			aidJ += myProtein.numAtomsPerCopy
			aidK += myProtein.numAtomsPerCopy
		
		i += 4


	### FETCH DIHEDRAL INFORMATION (F_PDIHS) ###
	lenDihList, dihList = myProtein.groInteractionDict[TCON.FTYPE_ENUM['F_PDIHS']]
	i = 0
	while i < lenDihList:
		# get the dihedral type and the atomtype indices of a molecule involved in the sdihedral angle
		dihType, atom1, atom2, atom3, atom4 = dihList[i:(i+5)]
		
		aidI, aidJ, aidK, aidL = atom1, atom2, atom3, atom4

		while aidI < totalAtomPopulation and aidJ < totalAtomPopulation and aidK < totalAtomPopulation and aidL < totalAtomPopulation:
			newDih = BondStructs.Dihedral((aidI, aidJ, aidK, aidL), myProtein)
			myProtein.add_dihedral(newDih)

			aidI += myProtein.numAtomsPerCopy
			aidJ += myProtein.numAtomsPerCopy
			aidK += myProtein.numAtomsPerCopy
			aidL += myProtein.numAtomsPerCopy
		
		i += 5

	
	# assign host heavy atoms for each hydrogen (needed for entropy calculation at UA level)
	myProtein.hostHeavyAtomArray = nmp.zeros(totalAtomPopulation)
	for idx in myProtein.bondedHeavyAtomTable.keys():
		if myProtein.isHydrogenArray[idx]:
			# it must have ONE and ONLY ONE heavy atom bonded to it (it host heavy atom)
			try:
				assert(len(myProtein.bondedHeavyAtomTable[idx]) == 1)
			except:
				raise ValueError('Atom {} is a hydrogen with more than 1 covalent bonds ({})'.format(idx, len(myProtein.bondedHeavyAtomTable[idx])))
			
			# fetch that atom
			try:
				priority, heavyAtomIdx = myProtein.bondedHeavyAtomTable[idx].list_in_order()[0]
				myProtein.hostHeavyAtomArray[idx] = heavyAtomIdx
				myProtein.hostHeavyAtomArray[heavyAtomIdx] = heavyAtomIdx #self
			except:
				raise IndexError('Index {} is out of range'.format(idx))

	
	# >>> ASSIGN MYPROTEIN TO THE MAIN DATA CONTAINER
	mainContainer.molecule = myProtein


	### READ TRAJECTORY ###
	Utils.printflush('{:<40s} : {}'.format('Reading trajectory file',arg_trajFile))
	
	traj = TRRReader.GroTrajectory(arg_fileName = arg_trajFile, arg_beginTime = arg_beginTime, arg_endTime = arg_endTime, arg_stride = arg_stride, arg_verbose = arg_verbose)

	# assign the snapshots read in traj to mainContainer's trajSnapshot 
	frameIdx = 0
	for iFrame in traj.snapshots:
		mainContainer.trajSnapshots.append(iFrame)
		mainContainer.frameIndices.append(frameIdx)
		frameIdx += 1

	mainContainer.print_attributes()

	# read the coords and forces from the trajectory
	# and store them in the mainContainer
	# it is a gromacs trajectory
	mainContainer.initialize_ndarrays()

	frameIdx = 0
	coordFactor = UAC.NM2ANG
	forceFactor = 1.0/UAC.NM2ANG  
	dim = TCON.VECTORDIM
	
	for iTrajSnapshot in mainContainer.trajSnapshots:
		mainContainer._labCoords[frameIdx] = nmp.multiply(coordFactor, nmp.reshape(nmp.asarray(iTrajSnapshot.value['coordinates']), (totalAtomPopulation, dim)))
		mainContainer._labForces[frameIdx] = nmp.multiply(forceFactor, nmp.reshape(nmp.asarray(iTrajSnapshot.value['forces']), (totalAtomPopulation, dim)))
		frameIdx += 1


	return (myProtein, mainContainer)

#END


