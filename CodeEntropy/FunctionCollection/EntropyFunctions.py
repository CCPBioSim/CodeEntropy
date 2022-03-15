import sys, os
import numpy as nmp

from CodeEntropy.ClassCollection import BeadClasses as BC
from CodeEntropy.ClassCollection import ConformationEntity as CONF
from CodeEntropy.ClassCollection import ModeClasses
from CodeEntropy.ClassCollection import CustomDataTypes
from CodeEntropy.ClassCollection import Atomselection as SEL
from CodeEntropy.FunctionCollection import CustomFunctions as CF
from CodeEntropy.FunctionCollection import GeometricFunctions as GF
from CodeEntropy.FunctionCollection import UnitsAndConversions as UAC
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.IO import Writer
from CodeEntropy.Reader import Constants as CONST


def calculate_entropy_per_dof(arg_frequencies, arg_temper):
	"""
	For each frequency that corresponds to a given dof, 
	it computes the entropy
	using eqn (4) in the Higham et. al. 2018 paper and returns it
	"""

	
	exponent = CONST.PLANCK_CONST*arg_frequencies/UAC.get_KT2J(arg_temper)
	
	expTermPositive = nmp.power(nmp.e, exponent)
	expTermNegative = nmp.power(nmp.e, -exponent)
	
	
	DOFEntropy = exponent/(expTermPositive - 1)
	DOFEntropy -= nmp.log(1 - expTermNegative)
	DOFEntropy *= CONST.GAS_CONST
	
	return DOFEntropy
# END

def compute_frequency_from_lambda(arg_lambdas, arg_temper):
	""" 
	For each lambda, compute the frequency.
	F = sqrt(λ/kT)/2π 
	"""
	return nmp.sqrt((arg_lambdas)/UAC.get_KT2J(arg_temper))/(2*nmp.pi)
#END

def compute_ampfac_from_lambda(arg_lambdas, arg_temper):
	"""
	For each mode (lambda), the amplitude factor is computed.
	Amplitude 
	A_i = kT/sqrt(L_i) for all 'i' in 1:num. modes
	Dim of A_i: sqrt([M]).L
	Units of A_i: sqrt(amu).Ang
	Ref: Macromolecule entropy from force, R. Henchman JCTC 2014
	"""
	afac = UAC.M2ANG * UAC.sqrtKG2AMU * nmp.divide(UAC.get_KT2J(arg_temper), nmp.sqrt(arg_lambdas))
	# print("Ampl factor: ", afac)
	return afac
#END

def get_avg_hpos(arg_atom, arg_frame, arg_baseMolecule, arg_hostDataContainer):
	"""
	Compute the average of the coordinates of the hydrogen
	atoms covalently bonded to the atom with index `arg_atom` in a 
	given point in time `arg_frame` and return the value. 
	If no hydrogen is bonded to it, return a 
	random value for 3D cartesian coordinates.
	"""
	avgHPos = nmp.zeros((3))
	selH = SEL.Atomselection(arg_baseMolecule, f"BONDed {arg_atom}") & SEL.Atomselection(arg_baseMolecule, "hydrogen")

	if selH.nsel != 0:
		for iH in selH.get_indices():
			iHPosition = arg_hostDataContainer._labCoords[arg_frame, iH]
			avgHPos = nmp.add(avgHPos, iHPosition)

		avgHPos /= selH.nsel

	else:
		# assign random position because 
		# eventually the only atom using that 
		# NB: basis will be the heavy atom which 
		# simply lies on the origin
		avgHPos =  nmp.random.random(3)

	# transform the average H position to a 
	# coordinate system whose origin is the position of 
	# the heavy atom.
	avgHPos = avgHPos - arg_hostDataContainer._labCoords[arg_frame, arg_atom]

	return avgHPos
#END

def get_avg_apos(arg_atom, arg_frame, arg_baseMolecule, arg_hostDataContainer):
	"""
	Compute the average of the coordinates of the heavy 
	atoms covalently bonded to the atom with index `arg_atom` in a 
	given point in time `arg_frame` and return the value. 
	If no heavy atom is bonded to it, return a 
	random value for 3D cartesian coordinates.
	"""
	avgPos = nmp.zeros((3))
	selHeavy = SEL.Atomselection(arg_baseMolecule,f"BONDed {arg_atom}") & ~SEL.Atomselection(arg_baseMolecule,"hydrogen")

	if selHeavy.nsel != 0:
		for iA in selHeavy.get_indices():
			iPosition = arg_hostDataContainer._labCoords[arg_frame, iA]
			avgPos = nmp.add(avgPos, iPosition)

		avgPos /= selHeavy.nsel

	else:
		# assign random position because 
		# eventually the only atom using that 
		# NB: basis will be the heavy atom which 
		# simply lies on the origin
		avgPos =  nmp.random.random(3)

	# transform the average H position to a 
	# coordinate system whose origin is the position of 
	# the heavy atom.
	avgPos = avgPos - arg_hostDataContainer._labCoords[arg_frame, arg_atom]

	return avgPos
#END

def compute_entropy_whole_molecule_level(arg_baseMolecule, 
										 arg_hostDataContainer, 
										 arg_outFile, 
										 arg_moutFile,
										 arg_nmdFile,
										 arg_fScale,
										 arg_tScale,
										 arg_temper,
										 arg_verbose):
	""" 
	Conpute the entropy at the whole molecule level. 
	Determining translation and rotation axes is part of the function.
	Returns void
	"""

	Utils.hbar(60)
	Utils.printflush("{:^60}".format("Hierarchy level. --> Whole molecule <--"))
	Utils.hbar(60)

	Utils.printOut(arg_outFile,'-'*60)
	Utils.printOut(arg_outFile,"{:^60}".format("Hierarchy level. --> Whole molecule <--"))
	Utils.printOut(arg_outFile,'-'*60)

	# Define a bead collection at this level
	wholeMoleculeSystem = BC.BeadCollection("whole_mol_bead", arg_hostDataContainer)

	# number of frames
	numFrames = len(arg_hostDataContainer.trajSnapshots)

	# define a bead representing the whole molecule
	allSel = SEL.Atomselection(arg_baseMolecule, "ALL")
	allAtomList = allSel.get_indices()

	wholeProteinBead = BC.Bead(arg_atomList= allAtomList, \
							arg_numFrames=numFrames, \
							arg_hostDataContainer = arg_hostDataContainer,\
							arg_beadName = "WMOL",
							arg_beadResi = 0,
							arg_beadResn = "WMOL",
							arg_beadChid = "X")

	# add the bead to the bead colleciton
	wholeMoleculeSystem.listOfBeads = [wholeProteinBead]

	Utils.printflush(f"Total number of beads at the whole molecule level = {len(wholeMoleculeSystem.listOfBeads)}")
	Utils.printOut(arg_outFile,f"Total number of beads at the whole molecule level = {len(wholeMoleculeSystem.listOfBeads)}")

	# reset weighted vectors for each bead and
	for iBead in wholeMoleculeSystem.listOfBeads:
		iBead.reset_totalWeightedVectors( (numFrames, 3) )

	# reseting all the F-T combo matrices to zero
	wholeMoleculeSystem.reinitialize_matrices()

	# and assign a representative position
	for iBead in wholeMoleculeSystem.listOfBeads:
		iBead.position = iBead.get_center_of_mass_lab(arg_frame = 0)

	# setup translational and rotational axes
	Utils.printflush("Assigning Translation and Rotation Axes @ whole molecule level->", end = ' ' )
	# Use Princ. Axes COOR SYS.
	# USE whole molecule principal axes COOR SYS for each atom
	for iFrame in range(numFrames):
		selMOI, selAxes = arg_baseMolecule\
						  .get_principal_axes(arg_atomList = allAtomList,\
											  arg_frame = iFrame, arg_sorted=False)
		selCOM = arg_baseMolecule\
				 .get_center_of_mass(arg_atomList = allAtomList, \
									 arg_frame = iFrame)
		arg_hostDataContainer.update_translationAxesArray_at(arg_frame = iFrame, arg_atomList = allAtomList, arg_pAxes = selAxes, arg_orig = selCOM)
		arg_hostDataContainer.update_rotationAxesArray_at(arg_frame = iFrame, arg_atomList = allAtomList, arg_pAxes = selAxes, arg_orig = selCOM)
	Utils.printflush("Done")
	

	# update local coordinates
	Utils.printflush("Updating Local coordinates->",end = ' ')
	arg_hostDataContainer.update_localCoords_of_all_atoms(arg_type="R")
	Utils.printflush('Done')

	# update local forces
	Utils.printflush("Updating Local forces->", end = ' ' )
	arg_hostDataContainer.update_localForces_of_all_atoms(arg_type="T")
	Utils.printflush('Done')

	#update torques in the arg_hostDataContainer
	Utils.printflush("Updating Local torques->", end = ' ')
	for iFrame in range(numFrames):
		for iAtom in allSel.get_indices():
			coords_i = arg_hostDataContainer.localCoords[iFrame, iAtom]
			forces_i = arg_hostDataContainer.localForces[iFrame, iAtom]
			# arg_hostDataContainer.localTorques[iFrame,iAtom,:] = nmp.cross(coords_i,forces_i)
			arg_hostDataContainer.localTorques[iFrame,iAtom,:] = CF.cross_product(coords_i,forces_i)
	Utils.printflush('Done')

	Utils.printflush("Weighting forces and torques->", end = ' ')
	# mass weighting the forces and torques
	for iBead in wholeMoleculeSystem.listOfBeads:

		# mass weighting the forces for each bead (iBead) in each direction (j) 
		# inertia weighting the torques for each bead (iBead) in each direction (j)

		for iFrame in range(numFrames):
			# define local basis as the rotationalAxes of the first atom in the atomList of iBead 
			# doesnt matter because they all have the same R and T axes
			iLocalBasis = arg_hostDataContainer.rotationAxesArray[iFrame][iBead.atomList[0]]

			#get the moment of inertia tensor for ibead in thid local basis
			beadMOITensor = iBead.get_moment_of_inertia_tensor_local(arg_localBasis = iLocalBasis, arg_frame = iFrame)

			# get total force and torque in each direction and weight them
			for iAtom in iBead.atomList:
				iBead.totalWeightedForces[iFrame] += arg_hostDataContainer.localForces[iFrame,iAtom]
				iBead.totalWeightedTorques[iFrame] += arg_hostDataContainer.localTorques[iFrame,iAtom]

			iBead.totalWeightedForces[iFrame] /= nmp.sqrt(iBead.get_total_mass())
			
			# weight total torque in each direction by √beadMOITensor[jj]
			for j in range(3):

				if nmp.isclose(iBead.totalWeightedTorques[iFrame,j], 0.0):
					# then the beadMOITensor[j,j] must be close to 0 as well (machine precision wise)
					# ensure that
					assert(nmp.isclose(beadMOITensor[j,j] , 0.0))
				else:
					iBead.totalWeightedTorques[iFrame,j] /= nmp.sqrt(beadMOITensor[j,j])

	Utils.printflush('Done')

	# now fill in the matrices
	Utils.printflush("Updating the submatrices ... ")
	wholeMoleculeSystem.update_subMatrix(arg_pairString="FF",arg_verbose=arg_verbose)
	wholeMoleculeSystem.update_subMatrix(arg_pairString="TT",arg_verbose=arg_verbose)
	Utils.printflush('Done')

	#make quadrant from subMatrices
	# FF and TT quadrants must be symmetric
	Utils.printflush("Generating Quadrants->",end = ' ')
	ffQuadrant = wholeMoleculeSystem.generate_quadrant(arg_pairString="FF",arg_filterZeros=1)
	ttQuadrant = wholeMoleculeSystem.generate_quadrant(arg_pairString="TT",arg_filterZeros=1)

	# scale forces/torques of these quadrants
	ffQuadrant = nmp.multiply(arg_fScale**2, ffQuadrant)
	ttQuadrant = nmp.multiply(arg_tScale**2, ttQuadrant)
	Utils.printflush("Done")

	# print matrices if asked
	if arg_moutFile:
		Writer.write_a_matrix(arg_matrix = ffQuadrant, arg_descriptor = "FF COV AT WHOLE MOLECULE LEVEL", arg_outFile = arg_moutFile)
		Writer.write_a_matrix(arg_matrix = ttQuadrant, arg_descriptor = "TT COV AT WHOLE MOLECULE LEVEL", arg_outFile = arg_moutFile)
	

	# remove any row or column with zero axis
	# this could have been done while generating quadrants. Can be merged if wished for
	# ffQuadrant = wholeMoleculeSystem.filter_zero_rows_columns(ffQuadrant)
	# ttQuadrant = wholeMoleculeSystem.filter_zero_rows_columns(ttQuadrant)


	#diagnolaize
	Utils.printflush("Diagonalizing->", end = ' ' )
	lambdasFF, eigVectorsFF  = Utils.diagonalize(ffQuadrant)	
	lambdasTT, eigVectorsTT  = Utils.diagonalize(ttQuadrant)
	Utils.printflush('Done')

	# since eigen values can be complex numbers but with imag parts very close to zero
	# use numpy's real_if_close with some tolerance to mask the imag parts
	# Utils.printflush('Checking the nature of eigen values and conditioning them ...', end = ' ')
	# tol = 1e+5
	# lambdasFF = nmp.real_if_close(lambdasFF/1e+5, tol= tol)
	# lambdasTT = nmp.real_if_close(lambdasTT/1e+5, tol= tol)
	# Utils.printflush('Done')

	# change to SI units
	Utils.printflush('Changing the units of eigen values to SI units->', end = ' ')
	lambdasFF = UAC.change_lambda_units(lambdasFF)
	lambdasTT = UAC.change_lambda_units(lambdasTT)
	Utils.printflush('Done')


	# Create a spectrum to store these modes for 
	# proper output and analyses.
	modeSpectraFF = [] 
	modeSpectraTT = []
	for midx, mcombo in enumerate(zip(lambdasFF, eigVectorsFF)):
		fflmb, evec = mcombo
		# compute mode frequencies
		# nu = sqrt(lambda/kT)*(1/2pi)
		# Units: 1/s
		mfreq = compute_frequency_from_lambda(fflmb, arg_temper)
		newMode = ModeClasses.Mode(arg_modeIdx = midx + 1, \
			arg_modeEval = fflmb, \
			arg_modeEvec = evec, \
			arg_modeFreq = mfreq)
		newMode.modeAmpl = compute_ampfac_from_lambda(fflmb, arg_temper)
		modeSpectraFF.append(newMode)

	for midx, mcombo in enumerate(zip(lambdasTT, eigVectorsTT)):
		ttlmb, evec = mcombo
		# compute mode frequencies
		# nu = sqrt(lambda/kT)*(1/2pi)
		# Units: 1/s
		mfreq = compute_frequency_from_lambda(ttlmb, arg_temper)
		newMode = ModeClasses.Mode(arg_modeIdx = midx + 1, \
			arg_modeEval = ttlmb, \
			arg_modeEvec = evec, \
			arg_modeFreq = mfreq)
		newMode.modeAmpl = compute_ampfac_from_lambda(ttlmb, arg_temper)
		modeSpectraTT.append(newMode)

	# assign spectra to the bead collection 
	wholeMoleculeSystem.assign_attribute("modeSpectraFF", modeSpectraFF)
	wholeMoleculeSystem.assign_attribute("modeSpectraTT", modeSpectraTT)

	# sorting the spectrum
	Utils.printflush('Sorting spectrum in ascending order of frequencies->', end = ' ')
	wholeMoleculeSystem.modeSpectraFF = ModeClasses.sort_modes(wholeMoleculeSystem.modeSpectraFF)
	wholeMoleculeSystem.modeSpectraTT = ModeClasses.sort_modes(wholeMoleculeSystem.modeSpectraTT)
	Utils.printflush('Done')

	# Print modes if asked
	if arg_nmdFile:
		Writer.append_file(arg_nmdFile)
		wholeMoleculeSystem.write_nmd_file(arg_nmdfile = arg_nmdFile, \
										   arg_spectrum = wholeMoleculeSystem.modeSpectraFF,
										   arg_wfac = [iBead.get_total_mass() for iBead in wholeMoleculeSystem.listOfBeads])

	# compute entropy
	entropyFF = [calculate_entropy_per_dof(m.modeFreq, arg_temper) for m in wholeMoleculeSystem.modeSpectraFF]
	entropyTT = [calculate_entropy_per_dof(m.modeFreq, arg_temper) for m in wholeMoleculeSystem.modeSpectraTT]

	# print final outputs
	Utils.printflush("Entropy values:")
	Utils.printflush(f"{'FF Entropy (Whole mol level)':<40s} : {nmp.sum(entropyFF):.4f} J/mol/K")
	Utils.printOut(arg_outFile, f"{'FF Entropy (Whole mol level)':<40s} : {nmp.sum(entropyFF):.4f} J/mol/K")
	
	Utils.printflush(f"{'TT Entropy (Whole mol level)':<40s} : {nmp.sum(entropyTT):.4f} J/mol/K")
	Utils.printOut(arg_outFile, f"{'TT Entropy (Whole mol level)':<40s} : {nmp.sum(entropyTT):.4f} J/mol/K")
	

	return 
#END


def compute_entropy_residue_level(arg_baseMolecule, 
								  arg_hostDataContainer, 
								  arg_outFile, 
								  arg_moutFile,
								  arg_nmdFile,
								  arg_fScale,
								  arg_tScale,
								  arg_temper,
								  arg_verbose):
	""" 
	Computes the entropy calculations at the residue level
	where each residue is treated as a separate bead.
	Determining translation and rotation axes is part of the 
	function. A common translation axes are used for all residues
	which is the principal axes of the whole molecule. The rotation
	axes are specific to each residue, in that it is the C-Ca-N axes. 
	"""

	Utils.hbar(60)
	Utils.printflush("{:^60}".format("Hierarchy level. --> Residues <--"))
	Utils.hbar(60)

	Utils.printOut(arg_outFile,'-'*60)
	Utils.printOut(arg_outFile,"{:^60}".format("Hierarchy level. --> Residues <--"))
	Utils.printOut(arg_outFile,'-'*60)
	
	# define a bead collection at this level
	residueSystem = BC.BeadCollection("res_bead",arg_hostDataContainer)

	# number of frames
	numFrames = len(arg_hostDataContainer.trajSnapshots)

	# define the residue beads and add
	residueSystem.listOfBeads= []

	# all atom selection
	allSel = SEL.Atomselection(arg_baseMolecule, "ALL")

	for rid in range(arg_baseMolecule.numResidues):
		
		iResname = arg_baseMolecule.resnameArray[rid]
		iResid = arg_baseMolecule.residArray[rid]
		resLabel = "{}{}".format(iResname, iResid)

		resSel = SEL.Atomselection(arg_baseMolecule, f"RESId {iResid}")
		caSel = SEL.Atomselection(arg_baseMolecule, "NAME CA")
		caIdx = caSel.get_index()

		newBead = BC.Bead(arg_atomList = resSel.get_indices(), \
					   arg_numFrames = numFrames, \
					   arg_hostDataContainer = arg_hostDataContainer,\
						arg_beadName = resLabel,\
						arg_beadResi = iResid,\
						arg_beadResn = iResname,\
						arg_beadChid = "X" )
		newBead.position = arg_hostDataContainer._labCoords[0,caIdx]
		residueSystem.listOfBeads.append(newBead)

	Utils.printflush(f"Total number of beads at the residue level = {len(residueSystem.listOfBeads)}")

	# reset weighted vectors for each bead
	for iBead in residueSystem.listOfBeads:
		iBead.reset_totalWeightedVectors( (numFrames,3) )

	# reseting all the F-T combo matrices to zero
	residueSystem.reinitialize_matrices()

	# setup translation axes 
	Utils.printflush("Assigning Translation Axes @ residue level->", end = ' ')
	arg_hostDataContainer.reset_translationAxesArray()
	# Use Princ. Axes COOR SYS.
	for iFrame in range(numFrames):
		selMOI, selAxes = arg_baseMolecule\
						  .get_principal_axes(arg_atomList = allSel.get_indices(), \
							arg_frame = iFrame, \
							arg_sorted=False)
		selCOM = arg_baseMolecule\
				 .get_center_of_mass(arg_atomList = allSel.get_indices(), \
					arg_frame = iFrame)
		arg_hostDataContainer.update_translationAxesArray_at(arg_frame = iFrame, \
			arg_atomList = allSel.get_indices(), \
			arg_pAxes = selAxes, \
			arg_orig = selCOM)
	Utils.printflush("Done")

	# setup rotational axes
	Utils.printflush("Assigning Rotational Axes @ residue level->")
	arg_hostDataContainer.reset_rotationAxesArray()

	# for each residue, set the rotational axes to the c-ca-N axes
	for rid in range(arg_baseMolecule.numResidues):
		iResid = arg_baseMolecule.residArray[rid]
		resSel = SEL.Atomselection(arg_baseMolecule, f"RESID {iResid}")
		cIdx = (resSel & SEL.Atomselection(arg_baseMolecule,"name C")).get_index()
		caIdx = (resSel & SEL.Atomselection(arg_baseMolecule,"name CA")).get_index()
		nIdx = (resSel & SEL.Atomselection(arg_baseMolecule,"name N")).get_index()

		atoms_in_rid = resSel.get_indices()

		for iFrame in range(numFrames):
			cPosition = arg_hostDataContainer._labCoords[iFrame,cIdx]
			nPosition = arg_hostDataContainer._labCoords[iFrame,nIdx]
			caPosition = arg_hostDataContainer._labCoords[iFrame,caIdx]

			ridAxes, ridOrigin = GF.generate_orthonormal_axes_system(arg_coord1 = cPosition, \
				arg_coord2 = nPosition, \
				arg_coord3 = caPosition)

			arg_hostDataContainer.update_rotationAxesArray_at(arg_frame = iFrame, \
				arg_atomList = atoms_in_rid, \
				arg_pAxes = ridAxes, \
				arg_orig = ridOrigin)

		if arg_verbose >= 3:
			Utils.printflush('{:>5d}'.format(rid), end = ' ')
			if (rid+1) % 5 == 0:
				Utils.printflush('')

	Utils.printflush("")
	Utils.printflush("Done")

	# update local forces
	Utils.printflush("Updating Local forces->", end = ' ')
	arg_hostDataContainer.update_localForces_of_all_atoms(arg_type = "T")
	Utils.printflush('Done')

	# update local coordinates
	Utils.printflush("Updating Local coordinates->", end= ' ')
	arg_hostDataContainer.update_localCoords_of_all_atoms(arg_type="R")
	Utils.printflush('Done')


	#update torques in the arg_hostDataContainer if asked for (arg_tScale != 0)
	Utils.printflush("Updating Local torques->", end = ' ')
	for iFrame in range(numFrames):
		for iAtom in arg_baseMolecule.atomIndexArray:
			coords_i = arg_hostDataContainer.localCoords[iFrame, iAtom]
			forces_i = arg_hostDataContainer.rotationAxesArray[iFrame, iAtom][0:3,]@arg_hostDataContainer._labForces[iFrame,iAtom]
			arg_hostDataContainer.localTorques[iFrame,iAtom,:] = CF.cross_product(coords_i,forces_i)
	Utils.printflush('Done')


	# mass weighting the forces and torques
	Utils.printflush('Weighting forces and torques->', end = ' ')
	for iBead in residueSystem.listOfBeads:

		# mass weighting the forces for each bead (iBead) in each direction (j) 
		#inertia weighting the torques for each bead (iBead) in each direction (j)

		for iFrame in range(numFrames):
			
			# get total torque and force and weigh them
			for iAtom in iBead.atomList:
				iBead.totalWeightedForces[iFrame] += arg_hostDataContainer.localForces[iFrame,iAtom] 
				iBead.totalWeightedTorques[iFrame] += arg_hostDataContainer.localTorques[iFrame,iAtom]

			iBead.totalWeightedForces[iFrame] /= nmp.sqrt(iBead.get_total_mass())			

			# define local basis as the rotationalAxes of the first atom in the atomList of iBead 
			iLocalBasis = arg_hostDataContainer.rotationAxesArray[iFrame][iBead.atomList[0]]
			beadMOITensor = iBead.get_moment_of_inertia_tensor_local(arg_localBasis = iLocalBasis, arg_frame = iFrame)

			for j in range(3):
				if nmp.isclose(iBead.totalWeightedTorques[iFrame,j] , 0.0):
					# then the beadMOITensor[j,j] must be close to 0 as well (machine precision wise)
					# ensure that
					assert(nmp.isclose(beadMOITensor[j,j] , 0.0))
				else:
					iBead.totalWeightedTorques[iFrame,j] /= nmp.sqrt(beadMOITensor[j,j])
	Utils.printflush('Done')

	# now fill in the matrices
	Utils.printflush("Updating the submatrices ... ")
	residueSystem.update_subMatrix(arg_pairString="FF",arg_verbose=arg_verbose)
	residueSystem.update_subMatrix(arg_pairString="TT",arg_verbose=arg_verbose)
	Utils.printflush('Done')

	#make quadrant from subMatrices
	Utils.printflush("Generating Quadrants->",end = ' ')
	ffQuadrant = residueSystem.generate_quadrant(arg_pairString="FF",arg_filterZeros=0)
	ttQuadrant = residueSystem.generate_quadrant(arg_pairString="TT",arg_filterZeros=0)
	Utils.printflush("Done")

	# scale forces/torques of these quadrants
	ffQuadrant = nmp.multiply(arg_fScale**2, ffQuadrant)
	ttQuadrant = nmp.multiply(arg_tScale**2, ttQuadrant)
	
	# print matrices if asked
	if arg_moutFile:
		Writer.write_a_matrix(arg_matrix = ffQuadrant, arg_descriptor = "FF COV AT RESIDUE LEVEL", arg_outFile = arg_moutFile)
		Writer.write_a_matrix(arg_matrix = ttQuadrant, arg_descriptor = "TT COV AT RESIDUE LEVEL", arg_outFile = arg_moutFile)

	# remove any row or column with zero axis
	# this could have been done while generating quadrants. Can be merged if wished for
	# ffQuadrant = residueSystem.filter_zero_rows_columns(ffQuadrant)
	# ttQuadrant = residueSystem.filter_zero_rows_columns(ttQuadrant)


	#diagnolaize
	Utils.printflush("Diagonalizing->", end = ' ')
	lambdasFF, eigVectorsFF  = Utils.diagonalize(ffQuadrant)
	lambdasTT, eigVectorsTT  = Utils.diagonalize(ttQuadrant)
	Utils.printflush('Done')

	# since eigen values can be complex numbers but with imag parts very close to zero
	# use numpy's real_if_close with some tolerance to mask the imag parts
	# Utils.printflush('Checking the nature of eigen values and conditioning them ...', end = ' ')
	# tol = 1e+5
	# lambdasFF = nmp.real_if_close(lambdasFF/1e+5, tol= tol)
	# lambdasTT = nmp.real_if_close(lambdasTT/1e+5, tol= tol)
	# Utils.printflush('Done')

	# change to SI units
	Utils.printflush('Changing the units of eigen values to SI units->', end = ' ')
	lambdasFF = UAC.change_lambda_units(lambdasFF)
	lambdasTT = UAC.change_lambda_units(lambdasTT)
	Utils.printflush('Done')

	# Create a spectrum to store these modes for 
	# proper output and analyses.
	modeSpectraFF = []
	for midx, mcombo in enumerate(zip(lambdasFF, eigVectorsFF)):
		fflmb, evec = mcombo
		# compute mode frequencies
		# nu = sqrt(lambda/kT)*(1/2pi)
		# Units: 1/s
		mfreq = compute_frequency_from_lambda(fflmb, arg_temper)
		newMode = ModeClasses.Mode(arg_modeIdx = midx + 1, \
			arg_modeEval = fflmb, \
			arg_modeEvec = evec, \
			arg_modeFreq = mfreq)
		newMode.modeAmpl = compute_ampfac_from_lambda(fflmb, arg_temper)
		modeSpectraFF.append(newMode)

	residueSystem.assign_attribute("modeSpectraFF", modeSpectraFF)


	modeSpectraTT = []
	for midx, mcombo in enumerate(zip(lambdasTT, eigVectorsTT)):
		ttlmb, evec = mcombo
		# compute mode frequencies
		# nu = sqrt(lambda/kT)*(1/2pi)
		# Units: 1/s
		mfreq = compute_frequency_from_lambda(ttlmb, arg_temper)
		newMode = ModeClasses.Mode(arg_modeIdx = midx + 1, \
			arg_modeEval = ttlmb, \
			arg_modeEvec = evec, \
			arg_modeFreq = mfreq)
		newMode.modeAmpl = compute_ampfac_from_lambda(ttlmb, arg_temper)
		modeSpectraTT.append(newMode)
	
	residueSystem.assign_attribute("modeSpectraTT", modeSpectraTT)

	# sorting the spectrum
	Utils.printflush('Sorting spectrum in ascending order of frequencies->', end = ' ')
	residueSystem.modeSpectraFF = ModeClasses.sort_modes(residueSystem.modeSpectraFF)
	residueSystem.modeSpectraTT = ModeClasses.sort_modes(residueSystem.modeSpectraTT)
	Utils.printflush('Done')

	# Print modes if asked
	if arg_nmdFile:
		Writer.append_file(arg_nmdFile)
		residueSystem.write_nmd_file(arg_nmdfile = arg_nmdFile, \
										   arg_spectrum = residueSystem.modeSpectraFF,\
										   arg_wfac = [iBead.get_total_mass() for iBead in residueSystem.listOfBeads])


	# compute entropy
	# 1. remove the smallest 6 freqs from FF sprectrum 
	#     because they may be overlapping with whole molecule
	#     These 6 low frequency modes capture the translation and rotation at
	#     whole molecule level
	# 2. DO NOT remove any freq from TT spectrum because 
	#    they are uncoupled to any TT freq in any other hierarchy
	entropyFF = [calculate_entropy_per_dof(m.modeFreq, arg_temper) for m in residueSystem.modeSpectraFF[6:]]
	entropyTT = [calculate_entropy_per_dof(m.modeFreq, arg_temper) for m in residueSystem.modeSpectraTT[0:]]

	# print final outputs
	Utils.printflush("Entropy values:")

	# print final outputs
	Utils.printflush(f"{'FF Entropy (Residue level)':<40s} : {nmp.sum(entropyFF):.4f} J/mol/K")
	Utils.printflush(f"{'TT Entropy (Residue level)':<40s} : {nmp.sum(entropyTT):.4f} J/mol/K")
	
	Utils.printOut(arg_outFile,f"{'FF Entropy (Residue level)':<40s} : {nmp.sum(entropyFF):.4f} J/mol/K")
	Utils.printOut(arg_outFile,f"{'TT Entropy (Residue level)':<40s} : {nmp.sum(entropyTT):.4f} J/mol/K")
	
	return 
#END


def compute_entropy_UA_level(arg_baseMolecule, 
							 arg_hostDataContainer, 
							 arg_outFile, 
							 arg_moutFile,
							 arg_nmdFile,
							 arg_fScale,
							 arg_tScale,
							 arg_temper,
							 arg_verbose):
	""" 
	Computes the entropy calculations at the united atom (UA) level. 
	Each heavy atom with its covalently bonded H-atoms make a single bead. H-atoms
	are, however, treated explicitly.Determining translation and rotation axes is 
	part of the function. Translation axes for each bead is the C-Ca-N axes of the 
	residue the bead is part of. The rotation axes is a basis whose axes are directed
	along a sphereical-coordinate axes comprised of unit vectors along r,θ and Φ. 

	"""

	Utils.hbar(60)
	Utils.printflush("{:^60}".format("Hierarchy level. --> United Atom <--"))
	Utils.hbar(60)

	Utils.printOut(arg_outFile,'-'*60)
	Utils.printOut(arg_outFile,"{:^60}".format("Hierarchy level. --> United Atom <--"))
	Utils.printOut(arg_outFile,'-'*60)
	
	# preparing header for output file
	Utils.printOut(arg_outFile,f"	  {'RESNAME':<10s}{'RESID':>5s}{'FF_ENTROPY':>12s}{'TT_ENTROPY':>12s}")
	
	# initialize total entropy values
	totalUAEntropyFF = 0.
	totalUAEntropyTT = 0.
	
	# number of frames
	numFrames = len(arg_hostDataContainer.trajSnapshots)

	#reset
	arg_hostDataContainer.reset_rotationAxesArray()
	arg_hostDataContainer.reset_translationAxesArray()

	# for each residue:
	for rid in range(arg_baseMolecule.numResidues):
		iResname = arg_baseMolecule.resnameArray[rid]
		iResid = arg_baseMolecule.residArray[rid]
		resLabel = "{}{}".format(iResname, iResid)
		Utils.printflush('Working on resid : {}'.format(resLabel))

		# create a bead collection 
		ridBeadCollection = BC.BeadCollection("{}_bead".format(resLabel),arg_hostDataContainer)
		ridBeadCollection.listOfBeads = []

		# add UA beads to it (a heavy atom and its bonded hydrogens make a bead)
		resSel = SEL.Atomselection(arg_baseMolecule,f"RESId {iResid}") 
		cIdx = (resSel & SEL.Atomselection(arg_baseMolecule,"name C")).get_index()
		nIdx = (resSel & SEL.Atomselection(arg_baseMolecule,"name N")).get_index()
		caIdx = (resSel & SEL.Atomselection(arg_baseMolecule,"name CA")).get_index()

		resHeavySel = resSel & ~SEL.Atomselection(arg_baseMolecule,"hydrogen")

		for iheavy in resHeavySel.get_indices():
			# GRP := (a heavy atom and its bonded hydrogens make a bead)
			igrp = SEL.Atomselection(arg_baseMolecule,f"BONH {iheavy}") 

			# heavy atom name
			iName = arg_baseMolecule.atomNameArray[iheavy]

			# create a bead
			newBead = BC.Bead(arg_atomList=igrp.get_indices(),\
				arg_hostDataContainer=arg_hostDataContainer,\
				arg_numFrames=numFrames,\
				arg_beadName = iName,\
				arg_beadResi = iResid,\
				arg_beadResn = iResname,\
				arg_beadChid = "X")

			newBead.position = arg_hostDataContainer._labCoords[0, iheavy]
			ridBeadCollection.listOfBeads.append(newBead)


		# by this point, the UA beads for that residue have been created
		Utils.printflush('Total number of UA beads in residue {} : {}'\
					  .format(resLabel, len(ridBeadCollection.listOfBeads)))

		# reset weighted vectors for each bead
		for iBead in ridBeadCollection.listOfBeads:
			iBead.reset_totalWeightedVectors( (numFrames,3) )

		# reseting all the F-T combo matrices to zero
		ridBeadCollection.reinitialize_matrices()

		# setup Translation and Rotation axes
		# Translation axes : each atom is in the c-ca-n axes of its host residue
		Utils.printflush("Assigning Translation Axes at the UA level->", end = ' ')
		for iFrame in range(numFrames):
			cPosition = arg_hostDataContainer._labCoords[iFrame,cIdx]
			nPosition = arg_hostDataContainer._labCoords[iFrame,nIdx]
			caPosition = arg_hostDataContainer._labCoords[iFrame,caIdx]

			tAxes, tOrigin = GF.generate_orthonormal_axes_system(arg_coord1 = cPosition, \
				arg_coord2 = nPosition, \
				arg_coord3 = caPosition)
			arg_hostDataContainer.update_translationAxesArray_at(iFrame, resSel.get_indices(), tAxes, tOrigin)
			
		Utils.printflush('Done')

		Utils.printflush("Assigning Rotational Axes at the UA level->", end = ' ')
		# Rotation axes : 
		# the axes will have the geometry of a 
		# local spherical-polar coordinate system
		# assigned locally to each UA bead.
		# See Chakravorty et. al. 2020 on the math behind it.
		for iBead in ridBeadCollection.listOfBeads:
			# fetch its heavy atom
			iheavy = list(filter(lambda idx: not arg_baseMolecule.is_hydrogen(idx), iBead.atomList))
			try:
				# check that these is only one heavy atom in the bead
				assert(len(iheavy) == 1)
			except:
				raise ValueError(f"An united atom bead cannot have more than one heavy atom. {len(iheavy)} found.")

			iheavy = iheavy[0]

			for iFrame in range(numFrames):
				 
				# from each of the hydrogen atoms bonded to it 
				# get the average position lab coordinate
				avgHydrogenPosition = get_avg_hpos(arg_atom= iheavy, \
					arg_frame = iFrame, \
					arg_baseMolecule = arg_baseMolecule, \
					arg_hostDataContainer = arg_hostDataContainer)

				# use the resultant vector to generate an 
				# orthogonal local coordinate axes system
				# with origin at the heavy atom position
				heavyOrigin = arg_hostDataContainer._labCoords[iFrame, iheavy]
				iAtomBasis = GF.get_sphCoord_axes(arg_r=avgHydrogenPosition)

				arg_hostDataContainer.update_rotationAxesArray_at(arg_frame = iFrame, \
					arg_atomList = iBead.atomList, \
					arg_pAxes = iAtomBasis, \
					arg_orig = heavyOrigin)

			arg_hostDataContainer.update_localCoords("R", iBead.atomList)

		Utils.printflush('Done')

		# update local forces 
		Utils.printflush('Updating Local forces->',end=' ')
		arg_hostDataContainer.update_localForces("T", resSel.get_indices())
		Utils.printflush('Done')


		# update torques using the local rotational axes
		Utils.printflush('Updating Local torques->', end = ' ')
		for iAtom_in_rid in resSel.get_indices():
			for iFrame in range(numFrames):
				coords_i = arg_hostDataContainer.localCoords[iFrame, iAtom_in_rid]
				forces_i = arg_hostDataContainer.rotationAxesArray[iFrame, iAtom_in_rid][0:3,]@arg_hostDataContainer._labForces[iFrame,iAtom_in_rid]
				arg_hostDataContainer.localTorques[iFrame,iAtom_in_rid,:] = CF.cross_product(coords_i,forces_i)
		Utils.printflush('Done')


		# mass weighting the forces and torque
		Utils.printflush('Weighting forces and torques->', end = ' ')
		for iBead in ridBeadCollection.listOfBeads:

			for iFrame in range(numFrames):
			
				# mass weighting the forces for each bead (iBead) in each direction (j) 
				# inertia weighting the torques for each bead (iBead) in each direction (j)
				for iAtom in iBead.atomList:
					iBead.totalWeightedForces[iFrame] += arg_hostDataContainer.localForces[iFrame, iAtom]
					iBead.totalWeightedTorques[iFrame] += arg_hostDataContainer.localTorques[iFrame, iAtom]


				iBead.totalWeightedForces[iFrame] /= nmp.sqrt(iBead.get_total_mass())

				# define local basis as the rotationalAxes of the first atom in the atomList of iBead 
				iLocalBasis = arg_hostDataContainer.rotationAxesArray[iFrame][iBead.atomList[0]]
				beadMOITensor = iBead.get_moment_of_inertia_tensor_local(arg_localBasis = iLocalBasis, arg_frame = iFrame)

				# get total torque and force in each direction and weight them by √beadMOITensor[jj]
				for j in range(3):
					try:
						if nmp.isclose(iBead.totalWeightedTorques[iFrame,j] , 0.0):
							# then the beadMOITensor[j,j] must be 0 as well
							# ensure that
							assert(nmp.isclose(beadMOITensor[j,j] , 0.0))
						else:
							# inertia weight the total torque component
							iBead.totalWeightedTorques[iFrame,j] /= nmp.sqrt(beadMOITensor[j,j])
					except:
						raise AssertionError(f"Moment of Intertia is non-zero for a bead lying on axis {j}")
						
		Utils.printflush('Done')

		# now fill in the matrices
		Utils.printflush("Updating the submatrices ... ")
		ridBeadCollection.update_subMatrix(arg_pairString="FF",arg_verbose=arg_verbose)
		ridBeadCollection.update_subMatrix(arg_pairString="TT",arg_verbose=arg_verbose)
		Utils.printflush('Done')

		#make quadrant from subMatrices
		Utils.printflush("Generating Quadrants->",end = ' ')
		ffQuadrant = ridBeadCollection.generate_quadrant(arg_pairString="FF",arg_filterZeros=0)
		ttQuadrant = ridBeadCollection.generate_quadrant(arg_pairString="TT",arg_filterZeros=0)
		Utils.printflush("Done")

		# scale forces/torques of these quadrants
		ffQuadrant = nmp.multiply(arg_fScale**2, ffQuadrant)
		ttQuadrant = nmp.multiply(arg_tScale**2, ttQuadrant)

		# remove any row or column with zero axis
		# this could have been done while generating quadrants. Can be merged if wished for
		ffQuadrant = ridBeadCollection.filter_zero_rows_columns(ffQuadrant)
		ttQuadrant = ridBeadCollection.filter_zero_rows_columns(ttQuadrant)


		# print matrices if asked
		if arg_moutFile:
			Writer.write_a_matrix(arg_matrix = ffQuadrant\
								  , arg_descriptor = "FF COV AT UNITED ATOM LEVEL FOR RES {}".format(resLabel)\
								  , arg_outFile = arg_moutFile)
			Writer.write_a_matrix(arg_matrix = ttQuadrant\
								  , arg_descriptor = "TT COV AT UNITED ATOM LEVEL FOR RES {}".format(resLabel)\
								  , arg_outFile = arg_moutFile)
			
		#diagnolaize
		Utils.printflush("Diagonalizing->", end = ' ')
		lambdasFF, eigVectorsFF  = Utils.diagonalize(ffQuadrant)
		lambdasTT, eigVectorsTT  = Utils.diagonalize(ttQuadrant)
		Utils.printflush('Done')

		# since eigen values can be complex numbers 
		# but with imag parts very close to zero
		# use numpy's real_if_close with some tolerance to mask the imag parts
		# Utils.printflush('Checking the nature of eigen values and conditioning them ...', end = ' ')
		# tol = 1e+5
		# lambdasFF = nmp.real_if_close(lambdasFF/1e+5, tol= tol)
		# lambdasTT = nmp.real_if_close(lambdasTT/1e+5, tol= tol)
		# Utils.printflush('Done')

		# filter real zero values
		lambdasFF = nmp.asarray([lm for lm in lambdasFF if not nmp.isclose(lm, 0.0)])
		lambdasTT = nmp.asarray([lm for lm in lambdasTT if not nmp.isclose(lm, 0.0)])


		# change to SI units
		Utils.printflush('Changing the units of eigen values to SI units->', end = ' ')
		lambdasFF = UAC.change_lambda_units(lambdasFF)
		lambdasTT = UAC.change_lambda_units(lambdasTT)
		Utils.printflush('Done')

		# Create a spectrum to store these modes for 
		# proper output and analyses.
		modeSpectraFF = []
		for midx, mcombo in enumerate(zip(lambdasFF, eigVectorsFF)):
			fflmb, evec = mcombo
			# compute mode frequencies
			# nu = sqrt(lambda/kT)*(1/2pi)
			# Units: 1/s
			mfreq = compute_frequency_from_lambda(fflmb, arg_temper)
			newMode = ModeClasses.Mode(arg_modeIdx = midx + 1, \
				arg_modeEval = fflmb, \
				arg_modeEvec = evec, \
				arg_modeFreq = mfreq)
			newMode.modeAmpl = compute_ampfac_from_lambda(fflmb, arg_temper)
			modeSpectraFF.append(newMode)

		ridBeadCollection.assign_attribute("modeSpectraFF", modeSpectraFF)


		modeSpectraTT = []
		for midx, mcombo in enumerate(zip(lambdasTT, eigVectorsTT)):
			ttlmb, evec = mcombo
			# compute mode frequencies
			# nu = sqrt(lambda/kT)*(1/2pi)
			# Units: 1/s
			mfreq = compute_frequency_from_lambda(ttlmb, arg_temper)
			newMode = ModeClasses.Mode(arg_modeIdx = midx + 1, \
				arg_modeEval = ttlmb, \
				arg_modeEvec = evec, \
				arg_modeFreq = mfreq)
			newMode.modeAmpl = compute_ampfac_from_lambda(ttlmb, arg_temper)
			modeSpectraTT.append(newMode)
		
		ridBeadCollection.assign_attribute("modeSpectraTT", modeSpectraTT)

		# sorting the spectrum
		Utils.printflush('Sorting spectrum in ascending order of frequencies->', end = ' ')
		ridBeadCollection.modeSpectraFF = ModeClasses.sort_modes(ridBeadCollection.modeSpectraFF)
		ridBeadCollection.modeSpectraTT = ModeClasses.sort_modes(ridBeadCollection.modeSpectraTT)
		Utils.printflush('Done')

		# Print modes if asked
		if arg_nmdFile:
			Writer.append_file(arg_nmdFile)
			ridBeadCollection.write_nmd_file(arg_nmdfile = arg_nmdFile, \
										   arg_spectrum = ridBeadCollection.modeSpectraFF, \
										   arg_wfac = [iBead.get_total_mass() for iBead in ridBeadCollection.listOfBeads])

		# compute entropy
		# 1. remove the smallest 6 freqs from FF sprectrum 
		#     because they may be overlapping with residue level motions
		# 2. DO NOT remove any freq from TT spectrum because 
		#    they are uncoupled to any TT freq in any other hierarchy
		entropyFF = [calculate_entropy_per_dof(m.modeFreq, arg_temper) for m in ridBeadCollection.modeSpectraFF[6:]]
		entropyTT = [calculate_entropy_per_dof(m.modeFreq, arg_temper) for m in ridBeadCollection.modeSpectraTT[0:]]

		ridTotalEntropyFF = nmp.sum(entropyFF)
		ridTotalEntropyTT = nmp.sum(entropyTT)

		# print final outputs
		Utils.printflush("Entropy values:")

		Utils.printflush('{:<40s} : {:.4f} J/mol/K'.format('FF Entropy (UA for {})'.format(resLabel), ridTotalEntropyFF))
		Utils.printflush('{:<40s} : {:.4f} J/mol/K'.format('TT Entropy (UA for {})'.format(resLabel), ridTotalEntropyTT))
		Utils.printOut(arg_outFile,'UATOM {:<10}{:>5}{:>12.3f}{:>12.3f}'\
								.format(arg_baseMolecule.resnameArray[rid]\
								, arg_baseMolecule.residArray[rid]\
								, ridTotalEntropyFF\
								, ridTotalEntropyTT))
		Utils.printflush("\n\n")
		
		totalUAEntropyFF += ridTotalEntropyFF
		totalUAEntropyTT += ridTotalEntropyTT

	# Final information	
	Utils.hbar(60)
	Utils.printflush(f"{'Total Entropy FF (UA level)':<25} : {totalUAEntropyFF:>15.3f} J/mol/K")
	Utils.printflush(f"{'Total Entropy TT (UA level)':<25} : {totalUAEntropyTT:>15.3f} J/mol/K")
	Utils.hbar(60)

	Utils.printOut(arg_outFile,'_'*60)
	Utils.printOut(arg_outFile,f"{'Total Entropy FF (UA level)':<25} : {totalUAEntropyFF:>15.3f} J/mol/K")
	Utils.printOut(arg_outFile,f"{'Total Entropy TT (UA level)':<25} : {totalUAEntropyTT:>15.3f} J/mol/K")
	Utils.printOut(arg_outFile,'-'*60)
	
	return 
#END

def compute_topographical_entropy0_SC(arg_baseMolecule, arg_hostDataContainer, arg_outFile, arg_verbose):
	""" A code that computes the topographical entropy using the formula S = -Sum(pLog(p)). 
	Every SC dihedral from every residue will be scanned. 
	Each dihedral will be depicted using a vector of order 3 of the form |g+, g-, t> (arbitrarily chosen) and 
	so can have a maximum of three different configurations it can be in. Its probability of being in each of 
	these states will be computed and entropy will be coputed form that."""


	Utils.hbar(60)
	Utils.printflush("{:^60}".format("Topographical entropy of residue side chains \n computed using all the dihedrals with pLogp formalism"))
	Utils.hbar(60)

	Utils.printOut(arg_outFile,'-'*60)
	Utils.printOut(arg_outFile,"{:^60}".format("Topographical entropy of residue side chains \n computed using all the dihedrals with pLogp formalism"))
	Utils.printOut(arg_outFile,'-'*60)

	# number of frames
	numFrames = len(arg_hostDataContainer.trajSnapshots)

	# log of number of frames (a constant)
	logNumFrames = nmp.log(numFrames)

	# conformation vector order |g+, g-, t>
	vecOrder = 3

	# total SC entropy
	totalTopogEntropySC = 0.

	# browse through each residue in the system and get their dihedrals
	for rid in range(arg_baseMolecule.numResidues):
		Utils.printflush('-'*10,end='')
		Utils.printflush('Working on resid : {} ({})'.format(rid, arg_baseMolecule.resnameArray[rid]), end='')
		Utils.printflush('-'*10)
	

		# total SC entropy at the topographical level of thi residue
		ridTopogEntropy = 0.

		diheds_in_rid = set()
		iAtom_in_rid = int(arg_baseMolecule.residueHead[rid])
		while iAtom_in_rid != -1:

			for iDih in arg_baseMolecule.dihedralTable[iAtom_in_rid]:
				# see if it is exclusive to this resid because they could also be peptide bond diheds
				if iDih.is_from_same_residue() == rid and (iDih.is_heavy())  and (not iDih.is_BB_dihedral()):
					diheds_in_rid.add(iDih)

			iAtom_in_rid = int(arg_baseMolecule.atomArray[iAtom_in_rid])

		Utils.printflush('Found {} exclusive dihedrals in residue {}'.format(len(diheds_in_rid), arg_baseMolecule.resnameArray[rid]))

		# define a list of ConformationEntities for this residue
		conformationEntityList = []

		for iDih in diheds_in_rid:
			dihAtoms = {"atom1": iDih.atom1, 
					"atom2":     iDih.atom2, 
					"atom3":     iDih.atom3, 
					"atom4":     iDih.atom4,
					"isBB" :     iDih.is_BB_dihedral(),
					"isHeavy" :  iDih.is_heavy(),
					"isSameRes" : iDih.is_from_same_residue()}
			
			# make an entity from this dihedral
			newEntity = CONF.ConformationEntity(arg_order = vecOrder, arg_numFrames = numFrames, **dihAtoms)
		
			# generate a time series of the conformations it acquires.
			# at each frame
			for iFrame in range(numFrames):

				# fetch the dihedral value at that frame
				phi = iDih.get_dihedral_angle_lab(arg_dataContainer = arg_hostDataContainer, arg_frame = iFrame)

				# define its status
				# isGaucheP = ( 0 <= phi < 120)
				# isGaucheN = ( 0 > phi >= -120 )
				# isTrans   = ( phi >= 120 or phi < -120)

				# using a different categorisation because some dihedrals
				# hover around the zero-lines and that makes it incorectly flexible
				# e.g. aromatic ring planar dihedrals
				isGaucheP = ( -30 <= phi < 90)
				isGaucheN = ( -30 > phi >= -150 )
				isTrans   = ( phi >= 90 or phi < -150)

				# place it in the time series block appropriately
				newEntity.timeSeries[:,iFrame] = nmp.asarray([isGaucheP, isGaucheN, isTrans]).astype(int)

			# add this dihedral into the list of conformation entities
			conformationEntityList.append(newEntity)

		# go over each entity and find its entropy. Add its entropy to the total entropy.
		for iEntity in conformationEntityList:
			sEntity = 0.

			for iRow in range(vecOrder):
				# get the total number of occurences of '1' in that row ( count )
				iCount = nmp.sum(iEntity.timeSeries[iRow,:])

				if iCount != 0:
					# means that state was atained at least once
					# p Log(p) for this state
					iPlogP = iCount * (nmp.log(iCount) - logNumFrames)

					sEntity += iPlogP;


			sEntity /= numFrames
			sEntity *= -CONST.GAS_CONST #(R)

			# add entropy of this entity to the residue's SC topographical entropy
			ridTopogEntropy += sEntity

			Utils.printflush('Dihedral {:<5d}{:<5d}{:<5d}{:<5d} : {:.4f}'.format(iEntity.atom1, iEntity.atom2, iEntity.atom3, iEntity.atom4, sEntity))
			Utils.printOut(arg_outFile, 'Dihedral {:<5d}{:<5d}{:<5d}{:<5d} : {:.4f}'.format(iEntity.atom1, iEntity.atom2, iEntity.atom3, iEntity.atom4, sEntity))


		# Final residue SC information	
		Utils.printflush('{:<40s} : {:.4f}'.format('Side Chain Topographical Entropy ({} {})'.format(arg_baseMolecule.resnameArray[rid], arg_baseMolecule.residArray[rid]), ridTopogEntropy))

		Utils.printOut(arg_outFile, '{:<40s} : {:.4f}'.format('Side Chain Topographical Entropy ({} {})'.format(arg_baseMolecule.resnameArray[rid], arg_baseMolecule.residArray[rid]), ridTopogEntropy))

		# add this residue's SC entropy to the total SC entropy
		totalTopogEntropySC += ridTopogEntropy 

	# total SC topographical entropy
	Utils.hbar(60)
	Utils.printflush('{:<40} : {:>15.3f}'.format('Total SC Topog. Entropy ', totalTopogEntropySC))
	Utils.hbar(60)

	Utils.printOut(arg_outFile, '_'*60)
	Utils.printOut(arg_outFile, '{:<40} : {:>15.3f}'.format('Total SC Topog. Entropy ', totalTopogEntropySC))
	Utils.printOut(arg_outFile, '-'*60)

	return
#END

def compute_topographical_entropy0_BB(arg_baseMolecule, arg_hostDataContainer, arg_outFile, arg_verbose):
	""" A code that computes the topographical entropy using the formula S = -Sum(pLog(p)). 
	Every BB dihedral from the protein will be scanned. 
	Each dihedral will be depicted using a vector of order 3 of the form |g+, g-, t> (arbitrarily chosen) and 
	so can have a maximum of three different configurations it can be in. Its probability of being in each of 
	these states will be computed and entropy will be coputed form that."""


	Utils.hbar(60)
	Utils.printflush("{:^60}".format("Topographical entropy of BB dihedrals \n computed using the pLogp formalism"))
	Utils.hbar(60)

	Utils.printOut(arg_outFile,'-'*60)
	Utils.printOut(arg_outFile,"{:^60}".format("Topographical entropy of BB dihedrals \n computed using the pLogp formalism"))
	Utils.printOut(arg_outFile,'-'*60)

	# number of frames
	numFrames = len(arg_hostDataContainer.trajSnapshots)

	# log of number of frames (a constant)
	logNumFrames = nmp.log(numFrames)

	# conformation vector order |g+, g-, t>
	vecOrder = 3

	# total BB entropy
	totalTopogEntropyBB = 0.


	# fetch all the heavy BB dihedrals
	bbDiheds = list(filter(lambda dih: dih.is_BB_dihedral() and dih.is_heavy(), arg_baseMolecule.dihedralArray))
	Utils.printflush('Found a total of {} BB dihedrals.'.format(len(bbDiheds)))


	# define a list of ConformationEntities to store all the BB dihedrals
	conformationEntityList = []

	for iBBDih in bbDiheds:
		dihAtoms = {"atom1": iBBDih.atom1, 
				"atom2":     iBBDih.atom2, 
				"atom3":     iBBDih.atom3, 
				"atom4":     iBBDih.atom4,
				"isBB" :     iBBDih.is_BB_dihedral(),
				"isHeavy" :  iBBDih.is_heavy(),
				"isSameRes" : iBBDih.is_from_same_residue()}
		
		# make an entity from this dihedral
		newEntity = CONF.ConformationEntity(arg_order = vecOrder, arg_numFrames = numFrames, **dihAtoms)
		
		# generate a time series of the conformations it acquires.
		# at each frame
		for iFrame in range(numFrames):

			# fetch the dihedral value at that frame
			phi = iBBDih.get_dihedral_angle_lab(arg_dataContainer = arg_hostDataContainer, arg_frame = iFrame)

			# define its status
			# isGaucheP = ( 0 <= phi < 120)
			# isGaucheN = ( 0 > phi >= -120 )
			# isTrans   = ( phi >= 120 or phi < -120)

			# using a different categorisation because some dihedrals
			# hover around the zero-lines and that makes it incorectly flexible
			# e.g. aromatic ring planar dihedrals
			isGaucheP = ( -30 <= phi < 90)
			isGaucheN = ( -30 > phi >= -150 )
			isTrans   = ( phi >= 90 or phi < -150)

			# create an instance of ConformationVector
			v = nmp.asarray([isGaucheP, isGaucheN, isTrans]).astype(int)

			# place it in the time series block appropriately
			newEntity.timeSeries[:,iFrame] = v

		# add this dihedral into the list of conformation entities
		conformationEntityList.append(newEntity)

	# go over each entity and find its entropy. Add its entropy to the total BB topographical entropy.
	for iEntity in conformationEntityList:
		sEntity = 0.

		for iRow in range(vecOrder):
			# get the total number of occurences of '1' in that row ( count )
			iCount = nmp.sum(iEntity.timeSeries[iRow,:])

			if iCount != 0:
				# means that state was atained at least once
				# p Log(p) for this state
				iPlogP = iCount * (nmp.log(iCount) - logNumFrames)

				sEntity += iPlogP;


		sEntity /= numFrames
		sEntity *= -CONST.GAS_CONST #(R)

		Utils.printflush('Dihedral {:<5d}{:<5d}{:<5d}{:<5d} : {:.4f} ({:>5d})'.format(iEntity.atom1, iEntity.atom2, iEntity.atom3, iEntity.atom4, sEntity, iEntity.isSameRes))
		Utils.printOut(arg_outFile, 'Dihedral {:<5d}{:<5d}{:<5d}{:<5d} : {:.4f} ({:>5d})'.format(iEntity.atom1, iEntity.atom2, iEntity.atom3, iEntity.atom4, sEntity, iEntity.isSameRes))

		# add entropy of this entity to the residue's SC topographical entropy
		totalTopogEntropyBB += sEntity


	# total BB topographical entropy
	Utils.hbar(60)
	Utils.printflush('{:<40} : {:>15.3f}'.format('Total BB Topog. Entropy ', totalTopogEntropyBB))
	Utils.hbar(60)

	Utils.printOut(arg_outFile, '_'*60)
	Utils.printOut(arg_outFile, '{:<40} : {:>15.3f}'.format('Total BB Topog. Entropy ', totalTopogEntropyBB))
	Utils.printOut(arg_outFile, '-'*60)

	return
#END

def compute_topographical_entropy1_SC(arg_baseMolecule, arg_hostDataContainer, arg_outFile, arg_verbose):
	""" A function that computes the entropy over the states acquired by the a residue in terms of the states acquired by 
	its dihedrals by also accounting for their correlated motions. A residue is depicted as a vector of length N_d where N_d 
	is the number of dihedrals. Each dihedral is represented using an integer which is a decimal equivalent of its state of some order Q
	which is represented by a binary vector of that size. At each time frame, a vector of integers of size N_d is stored and it stores that
	time frame uniquely. All the different states acquired are then used to compute the entropy using p-logP. """

	Utils.hbar(60)
	Utils.printflush("{:^60}".format("Topographical entropy of residue side chains \ncomputed using all the dihedrals with correlation/pLogp formalism"))
	Utils.hbar(60)

	Utils.printOut(arg_outFile,'-'*60)
	Utils.printOut(arg_outFile,"{:^60}".format("Topographical entropy of residue side chains \ncomputed using all the dihedrals with correlation/pLogp formalism"))
	Utils.printOut(arg_outFile,'-'*60)

	# number of frames
	numFrames = len(arg_hostDataContainer.trajSnapshots)

	# log of number of frames (a constant)
	logNumFrames = nmp.log(numFrames)

	# conformation vector order |g+, g-, t>
	vecOrder = 3   # (= Q)

	# total SC entropy
	totalTopogEntropySC = 0.

	# define a list of ConformationEntities where each element corresponds to a residue
	conformationEntityList = []	

	# browse through each residue in the system and get their dihedrals
	for rid in range(arg_baseMolecule.numResidues):
		Utils.printflush('-'*10,end='')
		Utils.printflush('Working on resid : {} ({})'.format(rid, arg_baseMolecule.resnameArray[rid]), end='')
		Utils.printflush('-'*10)


		# build a binary tree that will hold unique dihedrals 
		# uniqueness is defined based on 2-3 atom indexes
		diheds_in_rid = CustomDataTypes.BinaryTree()
		iAtom_in_rid = int(arg_baseMolecule.residueHead[rid])
		while iAtom_in_rid != -1:

			for iDih in arg_baseMolecule.dihedralTable[iAtom_in_rid]:
				# see if it is a side chain dihedral exclusive to this resid 
				if iDih.is_from_same_residue() == rid and iDih.is_heavy() and not (iDih.is_BB_phi() or iDih.is_BB_psi()):
					dihNode = CustomDataTypes.TreeNode(None, None, iDih)
					diheds_in_rid.add_node(dihNode)

			iAtom_in_rid = int(arg_baseMolecule.atomArray[iAtom_in_rid])

		Utils.printflush('Found {} exclusive dihedrals in residue {}{}'.\
			format(len(diheds_in_rid), arg_baseMolecule.resnameArray[rid], arg_baseMolecule.residArray[rid]))

		# create an object of Class ConformationEntity corresponding to this residue
		newEntity = CONF.ConformationEntity(arg_order = len(diheds_in_rid), arg_numFrames = numFrames)

		# also initialize a string array that will store the state in each frame as a distinct string
		# made from coalesced character cast of numeric arrays
		ridDecimalReprArray = []

		# at each frame
		for iFrame in range(numFrames):

			# fetch the dihedral value of each of the dihedrals for this residue at that frame
			for i, iDih in enumerate(diheds_in_rid.list_in_order()):
				phi = iDih.get_dihedral_angle_lab(arg_dataContainer = arg_hostDataContainer, arg_frame = iFrame)

				# define its status
				# isGaucheP = ( 0 <= phi < 120)
				# isGaucheN = ( 0 > phi >= -120 )
				# isTrans   = ( phi >= 120 or phi < -120)

				# using a different categorisation because some dihedrals
				# hover around the zero-lines and that makes it incorectly flexible
				# e.g. aromatic ring planar dihedrals
				isGaucheP = ( -30 <= phi < 90)
				isGaucheN = ( -30 > phi >= -150 )
				isTrans   = ( phi >= 90 or phi < -150)

				v = bytearray([isGaucheP, isGaucheN, isTrans])
				newEntity.timeSeries[i,iFrame] = Utils.binary_to_dec_repr(v)

			# populate the ridDecimalReprArray appropriately
			ridDecimalReprArray.append(Utils.coalesce_numeric_array(newEntity.timeSeries[:,iFrame]))

		# for each of the unique state get their count and compute the topographical entropy for this residue
		setOfstates = set(ridDecimalReprArray)
		Utils.printflush('Found {} dihedrals which collectively acquire {} unique conformers'.format(len(diheds_in_rid), len(setOfstates)))

		# print(ridDecimalReprArray)

		# total SC entropy at the topographical level of this residue
		ridTopogEntropy = 0.

		for iState in setOfstates:
			iCount = ridDecimalReprArray.count(iState) 

			# p Log(p) for this state
			iPlogP = iCount * (nmp.log(iCount) - logNumFrames)
			ridTopogEntropy += iPlogP;

		ridTopogEntropy /= numFrames;
		ridTopogEntropy *= -CONST.GAS_CONST  #(R)

		# Final residue SC information	
		Utils.printflush('{:<40s} : {:.4f}'.format('Side Chain Topographical Entropy from corr. pLogP method ({} {})'.format(arg_baseMolecule.resnameArray[rid], arg_baseMolecule.residArray[rid]), ridTopogEntropy))

		Utils.printOut(arg_outFile, '{:<40s} : {:.4f}'.format('Side Chain Topographical Entropy from corr. pLogP method ({} {})'.format(arg_baseMolecule.resnameArray[rid], arg_baseMolecule.residArray[rid]), ridTopogEntropy))

		# add this residue's SC entropy to the total SC entropy
		totalTopogEntropySC += ridTopogEntropy
			
	# total SC topographical entropy
	Utils.hbar(60)
	Utils.printflush('{:<40} : {:>15.3f}'.format('Total SC Topog. Entropy (corr. pLogP) ', totalTopogEntropySC))
	Utils.hbar(60)

	Utils.printOut(arg_outFile, '_'*60)
	Utils.printOut(arg_outFile, '{:<40} : {:>15.3f}'.format('Total SC Topog. Entropy (corr. pLogP)', totalTopogEntropySC))
	Utils.printOut(arg_outFile, '-'*60)

	return
#END

def compute_topographical_entropy1_BB(arg_baseMolecule, arg_hostDataContainer, arg_outFile, arg_verbose):
	""" 
	A function that computes the entropy over the states acquired 
	collectively by the heavy BB dihedrals in a protein
	by also accounting for their correlated motions. 
	A protein's colleciton of BB diheds is depicted as 
	a vector of length N_d where N_d is the number of BB dihedrals. 
	Each dihedral's state is represented using 0/1 telling which state it was in. 
	Then at each time frame, the state of a dihedral is computed and 
	represented using a decimal equivalent of its buytearray form. 
	For the entire protein, each time 	frame has a tuple of integers 
	corresponding to it which describes it uniquely. All the different 
	states acquired are then used to compute the entropy using p-logP. 
	"""

	

	Utils.hbar(60)
	Utils.printflush("{:^60}".format("Topographical entropy of BB dihedrals \ncomputed using the correlated-pLogp formalism"))
	Utils.hbar(60)

	Utils.printOut(arg_outFile,'-'*60)
	Utils.printOut(arg_outFile,"{:^60}".format("Topographical entropy of BB dihedrals \ncomputed using the correlated-pLogp formalism"))
	Utils.printOut(arg_outFile,'-'*60)

	# number of frames
	numFrames = len(arg_hostDataContainer.trajSnapshots)

	# log of number of frames (a constant)
	logNumFrames = nmp.log(numFrames)

	# conformation vector order |g+, g-, t>
	vecOrder = 3

	# total BB entropy
	totalTopogEntropyBB = 0.

	# fetch all the heavy BB dihedrals
	bbDiheds = CustomDataTypes.BinaryTree()
	for iDih in arg_baseMolecule.dihedralArray:
		# see if it is a peptide bond dihedral
		if iDih.is_heavy() and iDih.is_BB_dihedral():
			dihNode = CustomDataTypes.TreeNode(None, None, iDih)
			bbDiheds.add_node(dihNode)

	# create an instance of Class ConformationEntity that will contain all of these BB diheds
	newEntity = CONF.ConformationEntity(arg_order = len(bbDiheds), arg_numFrames = numFrames)

	# also initialize a string array that will store the state in each frame as a distinct string
	# made from coalesced character cast of numeric arrays
	bbDecimalReprArray = []   

	# at each frame
	for iFrame in range(numFrames):

		# fetch the dihedral value of each of the BB dihedrals in the protein at that frame
		for i, iDih in enumerate(bbDiheds.list_in_order()):
			phi = iDih.get_dihedral_angle_lab(arg_dataContainer = arg_hostDataContainer, arg_frame = iFrame)

			# define its status
			# isGaucheP = ( 0 <= phi < 120)
			# isGaucheN = ( 0 > phi >= -120 )
			# isTrans   = ( phi >= 120 or phi < -120)

			# using a different categorisation because some dihedrals
			# hover around the zero-lines and that makes it incorectly flexible
			# e.g. aromatic ring planar dihedrals
			isGaucheP = ( -30 <= phi < 90)
			isGaucheN = ( -30 > phi >= -150 )
			isTrans   = ( phi >= 90 or phi < -150)

			v = bytearray([isGaucheP, isGaucheN, isTrans])
			newEntity.timeSeries[i,iFrame] = Utils.binary_to_dec_repr(v)


		# populate the bbDecimalReprArray appropriately
		bbDecimalReprArray.append(Utils.coalesce_numeric_array(newEntity.timeSeries[:,iFrame]))

	# for each of the unique state get their count and compute the topographical entropy for this residue
	setOfstates = set(bbDecimalReprArray)

	Utils.printflush('Found {} dihedrals which collectively acquire {} unique conformers'.format(len(bbDiheds), len(setOfstates)))

	# total BB entropy at the topographical level 
	totalTopogEntropyBB = 0.

	for iState in setOfstates:
		iCount = bbDecimalReprArray.count(iState) 

		# p Log(p) for this state
		iPlogP = iCount * (nmp.log(iCount) - logNumFrames)
		totalTopogEntropyBB += iPlogP;

	totalTopogEntropyBB /= numFrames;
	totalTopogEntropyBB *= -CONST.GAS_CONST  #(R)

	# total BB topographical entropy
	Utils.hbar(60)
	Utils.printflush('{:<40} : {:>15.3f}'.format('Total BB Topog. Entropy (corr. pLogP) ', totalTopogEntropyBB))
	Utils.hbar(60)

	Utils.printOut(arg_outFile, '_'*60)
	Utils.printOut(arg_outFile, '{:<40} : {:>15.3f}'.format('Total BB Topog. Entropy (corr. pLogP) ', totalTopogEntropyBB))
	Utils.printOut(arg_outFile, '-'*60)

	return
#END

def compute_topographical_entropy_method4(arg_baseMolecule, arg_hostDataContainer, arg_outFile, arg_verbose):
	"""
	Function that computes the topographical entropy using Method 4,
	a.k.a the dihedral-state-contingency method.
	"""

	Utils.hbar(60)
	Utils.printflush("{:^60}".format("Topographical entropy using dihedral-state-contingency method"))
	Utils.hbar(60)

	Utils.printOut(arg_outFile,'-'*60)
	Utils.printOut(arg_outFile,"{:^60}".format("Topographical entropy using dihedral-state-contingency method"))
	Utils.printOut(arg_outFile,'-'*60)

	# number of frames
	numFrames = len(arg_hostDataContainer.trajSnapshots)

	# conformation vector order |g+, g-, t>
	vecOrder = 3   # (= Q)

	# initialize total entropy from all residues
	totalTopogEntropy4 = 0

	# all the dihedrals will be computed using coordinates projected onto 
	# molecular principal axes frames. (It should howver not matter what 
	# axes system we chose because dihedrals are measured using vector differences
	# which should not depend on the choice of coordinate systems).
	CF.cast_translationAxesArray_at_molecule_level(arg_dataContainer=arg_hostDataContainer)

	# update local coordinates
	if arg_verbose >= 2:
		Utils.printflush("Updating Local coordinates based on new Principal Axes ... ",end= ' ')
	
	arg_hostDataContainer.update_localCoords_of_all_atoms(arg_type="T")
	
	if arg_verbose >= 2:
		Utils.printflush('Done')

	#
	#
	#      Residue wise calculation of topographical entropy
	#
	#
	for rid in range(arg_baseMolecule.numResidues):
		Utils.printflush('-'*10,end='')
		Utils.printflush('Working on resid : {} ({})'.format(rid, arg_baseMolecule.resnameArray[rid]), end='')
		Utils.printflush('-'*10)

		dihedsInRid = set()
		iAtomInRid = int(arg_baseMolecule.residueHead[rid])
		while iAtomInRid != -1:

			for iDih in arg_baseMolecule.dihedralTable[iAtomInRid]:
				# see if it is exclusive to this resid because they could also be peptide bond diheds
				if iDih.is_from_same_residue() == rid and iDih.is_heavy():
					dihedsInRid.add(iDih)

			iAtomInRid = int(arg_baseMolecule.atomArray[iAtomInRid])

		numDiheds = len(dihedsInRid)
		if arg_verbose >= 2:
			Utils.printflush('Found {} exclusive dihedrals in residue {}\
						  '.format(numDiheds, arg_baseMolecule.resnameArray[rid]))
	
		# treat each dihedral as a conformation entity
		# initialize a list of ConformationEntities for this molecule
		conformationEntityList = []


		# for each heavy dihedral
		for iDih in dihedsInRid:
				
			# make an entity from this dihedral
			newEntity = CONF.ConformationEntity(arg_order = vecOrder, arg_numFrames = numFrames)

			# generate a time series of the conformations it acquires.
			# at each frame
			for iFrame in range(numFrames):

				# fetch the dihedral value at that frame
				phi = iDih.get_dihedral_angle_local(arg_dataContainer = arg_hostDataContainer, arg_frame = iFrame)

				# define its status
				# isGaucheP = ( 0 <= phi < 120)
				# isGaucheN = ( 0 > phi >= -120 )
				# isTrans   = ( phi >= 120 or phi < -120)

				# using a different categorisation because some dihedrals
				# hover around the zero-lines and that makes it incorectly flexible
				# e.g. aromatic ring planar dihedrals
				isGaucheP = ( -30 <= phi < 90)
				isGaucheN = ( -30 > phi >= -150 )
				isTrans   = ( phi >= 90 or phi < -150)

				# place it in the time series block appropriately
				newEntity.timeSeries[:,iFrame] = nmp.asarray([isGaucheP, isGaucheN, isTrans], dtype = nmp.int8)

			# add this dihedral into the list of conformation entities
			conformationEntityList.append(newEntity)


		#-------------------------------------------------------------------------------------
		#
		#          initialize and populate the symmetric occupancy matrix (for the residue)
		#
		#-------------------------------------------------------------------------------------
		# initialize
		occuMatrix = -1000 * nmp.ones((numDiheds*vecOrder, numDiheds*vecOrder))
		Utils.printOut(arg_outFile, "Occupancy matrix for Residue {}".format(arg_baseMolecule.resnameArray[rid]))

		# populate
		for i in range(0,numDiheds):
			iDih = conformationEntityList[i]
			if arg_verbose >= 2: 
				Utils.printflush('Dihedral {} : |'.format(i), end = ' ' )	

			for j in range(i, numDiheds):
				jDih = conformationEntityList[j]
				if arg_verbose >= 2: 
					Utils.printflush('.',end='')

				for iState in range(vecOrder):
					idx = (vecOrder * i) + iState
					iDihTimeSeries = iDih.timeSeries[iState,:]

					for jState in range(vecOrder):
						jdx = (vecOrder * j) + jState
						jDihTimeSeries = jDih.timeSeries[jState,:]
				
						# get the determinant of the contingency matrix computed from 
						# the dihedral states for this pair of dihedrals
						ijElement = CF.phi_coeff(arg_v1 = iDihTimeSeries\
															 , arg_v2 = jDihTimeSeries)

						# add entry at position idx, jdx
						occuMatrix[idx, jdx] = (ijElement)

						# add same entry at the tranpose position because the matrix is symmetric
						occuMatrix[jdx, idx] = occuMatrix[idx, jdx]

			if arg_verbose >= 2: 
				Utils.printflush('|')	
				

		# diagonlaize the occupancy matrix 
		lambdasPhi, eigVectorsPhi  = Utils.diagonalize(occuMatrix)
		
		# normalize the eig values with number of states and return the absolute value
		lambdasPhi = nmp.abs(nmp.divide(lambdasPhi, vecOrder))

		# is the occupancy matrix symmetric-positive definite? (are all the eigen values positive?)
		for iLm, lm in enumerate(lambdasPhi):
			Utils.printOut(arg_outFile, "Eigen value {} = {}".format(iLm, lm))

		# compute residue topog. entropy from the eigen values using the `lm.log(lm)` formalism
		ridTopogEntropy4 = 0
		for lm in filter(lambda x: x != 0, lambdasPhi):
			ridTopogEntropy4 += (lm * nmp.log(lm) )

		ridTopogEntropy4 *= -CONST.GAS_CONST #(R)

		# Final residue entropy information	
		Utils.printflush('{:<40s} : {:.4f}'.format('Topog. Entropy using method4 ({} {})'.format(arg_baseMolecule.resnameArray[rid], arg_baseMolecule.residArray[rid]), ridTopogEntropy4))
		Utils.hbar(60)

		Utils.printOut(arg_outFile, '{:<40s} : {:.4f}'.format('Topog. Entropy using method4 ({} {})'.format(arg_baseMolecule.resnameArray[rid], arg_baseMolecule.residArray[rid]), ridTopogEntropy4))
		Utils.printOut(arg_outFile, '-'*60)

		# add this residue's topog. entropy to the total topog. entropy
		totalTopogEntropy4 += ridTopogEntropy4
		
	# print out the outputs
	if arg_verbose >= 0:
		Utils.printflush('{:<40} : {:>15.3f}'.format('Topog. Entropy (Method4) ', totalTopogEntropy4))
		Utils.hbar(60)

	Utils.printOut(arg_outFile, '{:<40} : {:>15.3f}'.format('Topog. Entropy (Method4) ', totalTopogEntropy4))
	Utils.printOut(arg_outFile, '-'*60)
		
	return
#END

def compute_topographical_entropy_AEM(arg_baseMolecule, arg_hostDataContainer, arg_outFile, arg_verbose):
	"""
	Compute entropy by Adaptive Enumeration Method (AEM).
	This method deals with each dihedral in a conformational entity on an individual basis. After that it coalesces
	the state vectors of each dihedral in the conformational entity to help compute entropy using p-logP formulation. 
	This function computes the total entropy from all residue in the base molecule.
	"""

	Utils.hbar(60)
	Utils.printflush("{:^60}".format("Topographical entropy of residue side chains \ncomputed using all the dihedrals with AEM method"))
	Utils.hbar(60)

	Utils.printOut(arg_outFile,'-'*60)
	Utils.printOut(arg_outFile,"{:^60}".format("Topographical entropy of residue side chains \ncomputed using all the dihedrals with AEM method"))
	Utils.printOut(arg_outFile,'-'*60)

	# number of frames
	numFrames = len(arg_hostDataContainer.trajSnapshots)

	# log of number of frames (a constant)
	logNumFrames = nmp.log(numFrames)

	# total SC entropy
	totalTopogEntropySC = 0.
	

	# browse through each residue in the system and get their dihedrals
	for rid in range(arg_baseMolecule.numResidues):
		Utils.printflush('-'*10,end='')
		Utils.printflush('Working on resid : {} ({})'.format(rid, arg_baseMolecule.resnameArray[rid]), end='')
		Utils.printflush('-'*10)


		# build a binary tree that will hold unique dihedrals 
		# uniqueness is defined based on 2-3 atom indexes
		diheds_in_rid = CustomDataTypes.BinaryTree()
		iAtom_in_rid = int(arg_baseMolecule.residueHead[rid])
		while iAtom_in_rid != -1:

			for iDih in arg_baseMolecule.dihedralTable[iAtom_in_rid]:
				# see if it is a side chain dihedral exclusive to this resid 
				if iDih.is_from_same_residue() == rid and iDih.is_heavy():
					dihNode = CustomDataTypes.TreeNode(None, None, iDih)
					diheds_in_rid.add_node(dihNode)

			iAtom_in_rid = int(arg_baseMolecule.atomArray[iAtom_in_rid])

		Utils.printflush('Found {} exclusive dihedrals in residue {}{}'.\
			format(len(diheds_in_rid), arg_baseMolecule.resnameArray[rid], arg_baseMolecule.residArray[rid]))

		# create an object of Class ConformationEntity corresponding to this residue
		newEntity = CONF.ConformationEntity(arg_order = len(diheds_in_rid), arg_numFrames = numFrames)

		# also initialize a string array that will store the state in each frame as a distinct string
		# made from coalesced character cast of numeric arrays
		ridDecimalReprArray = []

		# for each dihedral identified, get the state vector
		for i, iDih in enumerate(diheds_in_rid.list_in_order()):
			stateTS = iDih.get_state_ts(arg_dataContainer = arg_hostDataContainer, arg_verbose = arg_verbose)
			newEntity.timeSeries[i,:] = stateTS

		# Now coalesce integer labels of the constituent dihedrals in each time point to get 
		# an expression of the conformation at that time.
		for iFrame in range(numFrames):
			ridDecimalReprArray.append(Utils.coalesce_numeric_array(newEntity.timeSeries[:,iFrame]))


		# for each of the unique state get their count and compute the topographical entropy for this residue
		setOfstates = set(ridDecimalReprArray)
		Utils.printflush('Found {} dihedrals which collectively acquire {} unique conformers'.format(len(diheds_in_rid), len(setOfstates)))

		# print(ridDecimalReprArray)

		# total SC entropy at the topographical level of this residue
		ridTopogEntropy = 0.

		for iState in setOfstates:
			iCount = ridDecimalReprArray.count(iState) 

			# p Log(p) for this state
			iPlogP = iCount * (nmp.log(iCount) - logNumFrames)
			ridTopogEntropy += iPlogP;

		ridTopogEntropy /= numFrames;
		ridTopogEntropy *= -CONST.GAS_CONST  #(R)

		# Final residue SC information	
		Utils.printflush('{:<40s} : {:.4f}'.format('Residue Topographical Entropy from AEM ({} {})'.format(arg_baseMolecule.resnameArray[rid], arg_baseMolecule.residArray[rid]), ridTopogEntropy))

		Utils.printOut(arg_outFile, '{:<40s} : {:.4f}'.format('Residue Topographical Entropy from AEM ({} {})'.format(arg_baseMolecule.resnameArray[rid], arg_baseMolecule.residArray[rid]), ridTopogEntropy))

		# add this residue's SC entropy to the total SC entropy
		totalTopogEntropySC += ridTopogEntropy
			
	# total SC topographical entropy
	Utils.hbar(60)
	Utils.printflush('{:<40} : {:>15.3f}'.format('Total Topog. Entropy (AEM) ', totalTopogEntropySC))
	Utils.hbar(60)

	Utils.printOut(arg_outFile, '_'*60)
	Utils.printOut(arg_outFile, '{:<40} : {:>15.3f}'.format('Total Topog. Entropy (AEM)', totalTopogEntropySC))
	Utils.printOut(arg_outFile, '-'*60)

	return
#END


def compute_topographical_entropy_method3(arg_baseMolecule, arg_hostDataContainer, arg_outFile, arg_verbose):

	# number of frames
	numFrames = len(arg_hostDataContainer.trajSnapshots)

	# conformation vector order |g+, g-, t>
	vecOrder = 3   # (= Q)

	# treat each dihedral as a conformation entity
	# initialize a list of ConformationEntities for this molecule
	conformationEntityList = []

	# fetch all the heavy dihedrals
	nohDiheds = list(filter(lambda dih:  dih.is_heavy(), arg_baseMolecule.dihedralArray))
	
	# for iDih in arg_baseMolecule.dihedralArray:
	for iDih in nohDiheds:
		dihAtoms = {"atom1": iDih.atom1, 
					"atom2":     iDih.atom2, 
					"atom3":     iDih.atom3, 
					"atom4":     iDih.atom4,
					"isBB" :     iDih.is_BB_dihedral(),
					"isHeavy" :  iDih.is_heavy(),
					"isSameRes" : iDih.is_from_same_residue()}
			
		# make an entity from this dihedral
		newEntity = CONF.ConformationEntity(arg_order = vecOrder, arg_numFrames = numFrames, **dihAtoms)

		# generate a time series of the conformations it acquires.
		# at each frame
		for iFrame in range(numFrames):

			# fetch the dihedral value at that frame
			phi = iDih.get_dihedral_angle_lab(arg_dataContainer = arg_hostDataContainer, arg_frame = iFrame)

			# define its status
			# isGaucheP = ( 0 <= phi < 120)
			# isGaucheN = ( 0 > phi >= -120 )
			# isTrans   = ( phi >= 120 or phi < -120)

			# using a different categorisation because some dihedrals
			# hover around the zero-lines and that makes it incorectly flexible
			# e.g. aromatic ring planar dihedrals
			isGaucheP = ( -30 <= phi < 90)
			isGaucheN = ( -30 > phi >= -150 )
			isTrans   = ( phi >= 90 or phi < -150)

			# place it in the time series block appropriately
			newEntity.timeSeries[:,iFrame] = nmp.asarray([isGaucheP, isGaucheN, isTrans], dtype = nmp.int8)

		# add this dihedral into the list of conformation entities
		conformationEntityList.append(newEntity)

	
	# total number of conformational entities (or dihedrals)
	numDiheds = len(conformationEntityList)

	# for each pair of dihedrals, find a matrix \rho_ij = \p_ij * \r_ij for i,j = 1 .. Q
	# where \p_ij is the probability of seeing dihedral1 in state 'i' and dihedral 2 in state 'j'
	# and   \r_ij is the correlation of dihedral1 in state 'i' and dihedral 2 in state 'j'

	# initialize a density matrix with values that can never be!
	densityMatrix = -1000 * nmp.zeros((numDiheds*vecOrder, numDiheds*vecOrder))

	for i in range(0,numDiheds):
		iEntity = conformationEntityList[i]
		Utils.printflush('Dihedral {} : |'.format(i), end = ' ' )	

		for j in range(i, numDiheds):
			jEntity = conformationEntityList[j]
			Utils.printflush('.',end='')

			Utils.printOut(arg_outFile, 'Dihedral {}: ({} {} {} {}) and Dihedral {}: ({} {} {} {})'.format(i, iEntity.atom1, iEntity.atom2, iEntity.atom3, iEntity.atom4, \
				j, jEntity.atom1, jEntity.atom2, jEntity.atom3, jEntity.atom4))

			for iState in range(vecOrder):
				idx = (vecOrder * i) + iState
				iDihedralTimeSeries = iEntity.timeSeries[iState,:]
				iDihedralTimeSeriesSTD = nmp.std(iDihedralTimeSeries)

				for jState in range(vecOrder):
					jdx = (vecOrder * j) + jState
					jDihedralTimeSeries = jEntity.timeSeries[jState,:]
					jDihedralTimeSeriesSTD = nmp.std(jDihedralTimeSeries)

					# correlation (r_ij)
					ijCorrelation = -1000    #initialize with a number that can never be!

					if iDihedralTimeSeriesSTD == 0:
						if jDihedralTimeSeriesSTD == 0:
							#both are not changing => correlation is '1'
							ijCorrelation = 1
						elif jDihedralTimeSeriesSTD != 0:
							#one is changing irrespective of the other => no correlation
							ijCorrelation = 0

					elif iDihedralTimeSeriesSTD != 0:
						if jDihedralTimeSeriesSTD == 0:
							#one is changing irrespective of the other => no correlation
							ijCorrelation = 0
						else:
							#compute the correlation using covariance
							ijCovariance = CF.covariance(iDihedralTimeSeries, jDihedralTimeSeries)
							ijCorrelation = ijCovariance/(iDihedralTimeSeriesSTD * jDihedralTimeSeriesSTD)

					# probability of coexistence (p_ij)
					ijProb = CF.probability_of_coexistence(iDihedralTimeSeries, jDihedralTimeSeries)
					
					# add entry at position idx, jdx
					densityMatrix[idx, jdx] = ijProb * ijCorrelation

					# add same entry at the tranpose position because the matrix is symmetric
					densityMatrix[jdx, idx] = densityMatrix[idx, jdx]

					Utils.printOut(arg_outFile, "{:>15.8f}".format(densityMatrix[idx, jdx]), end = "")

					if jState == (vecOrder - 1):
						Utils.printOut(arg_outFile,'')
		
		Utils.printflush('|')	
			
	# filter rows and columns with all zero (which make the matrix singular)
	densityMatrix = CF.filter_zero_rows_columns(densityMatrix)
	
	# diagonlaize the density matrix 
	lambdasRho, eigVectorsRho  = Utils.diagonalize(densityMatrix)

	# is the density matrix symmetric-positive definite?
	for lr in lambdasRho:
		Utils.printOut(arg_outFile, lr)
	

	# plot the matrix with imshow
	if False:
		mplot = plt.figure()
		ax = mplot.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
		plt.imshow(densityMatrix, cmap = "jet", vmin = -1, vmax = +1)
		plt.show()

	return
#END

