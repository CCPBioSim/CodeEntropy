import sys
import logging
import gc

from collections import Counter
from datetime import datetime

from CodeEntropy.poseidon.extractData.readFiles import populateTopology, getCoordsForces, getDistArray
# # Energy is not needed
# from CodeEntropy.poseidon.extractData.readFiles import populateEnergy, UAEnergyGroup
from CodeEntropy.poseidon.extractData.HBRAD import distCutoffNc, UALevelRAD, HBCalc
from CodeEntropy.poseidon.extractData.forceTorques import calculateFTMatrix
from CodeEntropy.poseidon.extractData.dihedrals import calculateDihedrals
from CodeEntropy.poseidon.extractData.nearestNonlike2 import getShellAssignment, moleculePositionRankingRAD 
from CodeEntropy.poseidon.extractData.outputFiles import moleculeObjectPopulation
from CodeEntropy.poseidon.extractData.mainClass import clearClass

from CodeEntropy.poseidon.analysis.populateClasses import classPopulation
from CodeEntropy.poseidon.analysis.EECalculation import processEE
from CodeEntropy.poseidon.analysis.helper import memoryInfo, weightingPopulation

from functools import partial
from multiprocessing import Pool

class Poseidon():
	"""
	Container to host data from MDAnalysis.Universe and run Poseidon Analysis
	"""

	def __init__(self ,container, start=0, end=-1, 
		step=1, pureAtomNum=1, cutShell=None, 
		excludedResnames=None,
		water='WAT', verbose=False):
		"""
		This is a initialization function to collect information from a MDanalysis universe into a data container for analysis using POSEIDON
		
		Parameters
		----------
		container : MDAnalyse.Universe
			A Universe object will all topology, dihedrals,coordinates and force information Check ``Example/create_new_universe.py`` on how to create a universe from existing data.
		start : int or None, Optional, default: 0
			Frame id to start analysis.
		end : int or None, Optional, default: -1
			Frame id to end analysis.
		step : int, Optional, default: 1
			Steps between frame.
		pureAtomNum : int, Optional, default: 1
			Reference molecule resid for pure liquid.
		cutShell : float or None, Optional, default: 1
			Explicit cut off shell. Default to None which uses the relative angular distance (RAD) algorithm. See Jonathan Higham and Richard H. Henchman , "Locally adaptive method to define coordination shell", J. Chem. Phys. 145, 084108 (2016)
		excludedResnames : list of str or None, Optional, default: None
			List of resnames to exclude from nearest non alike analysis.
		water : list of str, Optional, default: "WAT"
			Resname for water molecules.
		verbose : bool, Optional, default: False
			Print out progress of each step.
		
		"""
		startTime = datetime.now()
		print(startTime)
		verbosePrint = print if verbose else lambda *a, **k: None
		
		waterTuple = ('SOL', 'WAT', 'HOH', 'TIP3') #needed for pdb as top file 
		if water != 'WAT':
			waterTuple = (water)

		iterations = 0
		
		all_data = []

		populateTopology(container, all_data, waterTuple)
		verbosePrint('TOPOLOGY')
		verbosePrint(datetime.now() - startTime)
		sys.stdout.flush()


		resids = Counter([(i.resname) for i in all_data])
		verbosePrint(resids.keys())
		if len(resids.keys()) == 1:
			verbosePrint('Pure system with reference ID: %s' % (pureAtomNum))

		if excludedResnames != None:
			verbosePrint('EXCLUDED RESNAMES: %s' % (excludedResnames))
		

		dimensions = None
		allMoleculeList = []
		
		#fix end frame
		if end == -1:
			end = len(container.trajectory)

		for frame in range(int(start), int(end), int(step)):

			iterations += 1

			clearClass(all_data)

			print(f"frame = {frame}")

			all_data, dimensions = getCoordsForces(container, 
					all_data, dimensions, frame, startTime, 
					verbosePrint)
			# # Energy is not needed
			# populateEnergy(container, all_data, 
			#         dimensions, frame, startTime, verbosePrint)
			# UAEnergyGroup(all_data)


			calculateDihedrals(all_data, dimensions)
			verbosePrint('DIH')
			verbosePrint(datetime.now() - startTime)
			sys.stdout.flush()

			#'''
			traj = container.trajectory[frame]
			neighbour_coords = None
			neighbour_coords = traj.positions

			max_cutoff = 10
			for x in range(0, len(all_data)):
				atom = all_data[x]
				#start nearest array from solutes
				if atom.resname not in waterTuple:
					getDistArray(atom, all_data, traj, max_cutoff,
							dimensions, neighbour_coords,
							startTime, verbosePrint)
					#find nearest array for solute neighbours that are solvent
					for nearDist in atom.nearest_all_atom_array[0:20]:
						neighbour = all_data[nearDist[0]]
						if neighbour.resname in waterTuple and \
								neighbour.nearest_all_atom_array == None:
							getDistArray(neighbour, all_data, traj, max_cutoff,
									dimensions, neighbour_coords,
									startTime, verbosePrint)
						#find solvent neighbours neighbours nearest array
						if neighbour.nearest_all_atom_array != None:
							for nearDist2 in neighbour.nearest_all_atom_array[0:20]:
								neighbour2 = all_data[nearDist2[0]]
								if neighbour2.resname in waterTuple and \
										neighbour2.nearest_all_atom_array == None:
									getDistArray(neighbour2, all_data, 
											traj, max_cutoff,
											dimensions, 
											neighbour_coords,
											startTime, verbosePrint)
								else:
									continue
						else:
							continue
				else:
					continue
			verbosePrint('NEAREST ARRAYS')
			verbosePrint(datetime.now() - startTime)
			sys.stdout.flush()
			#'''



			if cutShell != None:
				#Used for fixed cut-off coordination shells
				try:
					cutoff_dist = float(cutShell)
					distCutoffNc(all_data, dimensions, cutoff_dist)
					# #don't know what this is it is not defined anywhere
					# NcPairs(all_data, dimensions)
					verbosePrint('distCutoffNc')
					verbosePrint(datetime.now() - startTime)
					sys.stdout.flush() 
				except ValueError:
					logging.error('Cutoff distance needs to be a float')


			if cutShell == None:
				UALevelRAD(all_data, dimensions)
				verbosePrint('RAD')
				verbosePrint(datetime.now() - startTime)


			#if inputType != 'pdb':
			HBCalc(all_data, waterTuple, dimensions)
			verbosePrint('HB')
			verbosePrint(datetime.now() - startTime)
			sys.stdout.flush() 


			getShellAssignment(all_data, excludedResnames, dimensions, 
					startTime, verbosePrint)
			verbosePrint('PROX')
			verbosePrint(datetime.now() - startTime)
			sys.stdout.flush()


			#if (force != None):
			calculateFTMatrix(all_data, dimensions)
			verbosePrint('FTMATRIX')
			verbosePrint(datetime.now() - startTime)
			sys.stdout.flush()


			moleculePositionRankingRAD(all_data, waterTuple, dimensions)
			verbosePrint('ORIENTS')
			verbosePrint(datetime.now() - startTime)
			sys.stdout.flush()

			'''
			if iterations == 1:
				verbosePrint('Generating .pdb file for frame %s' % (frame+1))
				pdbGenerate(all_data, ('frame_%s' % (frame+1)), dimensions)
			'''



			allMoleculeList = moleculeObjectPopulation(all_data, 
					allMoleculeList, frame, dimensions)


			print(datetime.now() - startTime)
			sys.stdout.flush() #write out to output file for csf in 
					#real time rather than at end of job


		# #writing file here 
		# with gzip.GzipFile('moleculeListAll.obj', 'wb') as pickleFile:
		#     pickle.dump((allMoleculeList), pickleFile, protocol=2)
		#     pickleFile.close()

		print(datetime.now() - startTime)
			
		self.allMoleculeList = allMoleculeList

	def run_analysis(self, 
			temperature=298.0, 
			entropyEnergy=True, 
			level_list=['moleculeLevel'], 
			solvent=None, water='WAT',
			verbose=False, weighting=None, 
			forceUnits='kJ'):
		"""Perform analysis using poseidon

		Parameters
		-----------
		temperature : float, Optional, default: 298.0
			Temperature for system.
		entropyEnergy : Bool, Optional, default: True
			Run entropy and energy analysis.
		level_list : list of str, Optional, default: ['moleculeLevel']
			Choose and refine the level of analyiss: moleculeLevel, residLevel_resname, atomLevel, soluteContacts.
		solvent : str, Optional, default: None
			Resname for solvent.
		water : str or list of str, Optional, default: 'WAT'
			Resname for water.
		verbose : bool, Optional, default: False
			Print out progress of each step.
		weighting : str or None, Optional, default: None
			Get weighing for each frame if the simulation is biased.
		forceUnits : str, Optional, default: 'kJ'
			Units of forces, kJ or Kcal. 

		Returns
		---------
			Tuple of DataFrame, for each level analysed:
				solventData: dataframe that contains the result for solvent
				soluteData: dataframe that contains the result for solute
				contactMatrix: dataframe that contains the contact matrix
		"""

		verbosePrint = print if verbose else lambda *a, **k: None
		startTime = datetime.now()
		print(startTime)



		waterTuple = ['WAT', 'wat', 'SOL', 'H2O', 'h2o', 'WAT_O', 'TIP3']
		if water != 'WAT':
			waterTuple = [water]
		if solvent == None: ##when solvent is NOT water
			solvent = waterTuple




		print('\nsolvent: %s' % (solvent))
		print('\nwater: %s' % (waterTuple))
		print('\n1. Populate Dictionaries\n')

		count = 1
		totAtoms = 0
		list_len = None
		atom_nums = []
		atom_count = 0
		objectIteration = 0

		EEclass = None 
		EEclass_residLevel_resname = None
		EEclass_soluteContacts = None
		EEclass_atomLevel = None


		class_str_list = ['EEclass', 'EEclass_residLevel_resname', 
				'EEclass_soluteContacts', 'EEclass_atomLevel', 
				'totAtoms', 'atom_nums', 'atom_count', 'objectIteration']

		memoryInfo(verbosePrint)
		print(datetime.now() - startTime)
		sys.stdout.flush()
		if totAtoms != 0:
			count = 2

		weighting_info = None
		atomList = self.allMoleculeList
		list_len = len(atomList[0])
		totAtoms += len(atomList)
		if count == 1:
			for i in atomList:
				atom_num = i[2]
				if atom_num not in atom_nums:
					atom_nums.append(atom_num)
				else:
					break
			if weighting != None:
				weighting_info = weightingPopulation(weighting)

		EEclass, EEclass_residLevel_resname, \
				EEclass_soluteContacts, EEclass_atomLevel, \
				atom_count = \
					classPopulation(
				atomList, entropyEnergy, 
				level_list, count, 
				atom_count, len(atom_nums), 
				waterTuple, temperature, 
				solvent, EEclass, 
				EEclass_residLevel_resname, EEclass_soluteContacts, 
				EEclass_atomLevel, 
				weighting_info, verbosePrint)
		count += 1 #needed for class initiation
		gc.enable()
		memoryInfo(verbosePrint)
		print(datetime.now() - startTime)
		sys.stdout.flush()


		print(datetime.now() - startTime)
		memoryInfo(verbosePrint)
		sys.stdout.flush()

		
		try:
			totFrames = float(totAtoms) / float(len(atom_nums))
			print('\nTotal number of frames: %s' % (totFrames))
			print('Number of atoms in each frame: %s' % (len(atom_nums)))
			print('Number of variables in each list: %s' % (list_len))
			num_frames = totFrames
		except ZeroDivisionError:
			logging.error('No frames to analyse, please chose correct '\
					'path to .obj files')
			num_frames = None
			sys.exit()

		###once all the classes have been populated, calculate properties
		#and output to files
		
		print('\n2. Process Dictionaries')
		result_list = dict()
		print(level_list)
		for level in level_list:

			print('---level:', level)
			EEclass2, DSclass2 = None, None

			if level in [None, 'moleculeLevel']:
				EEclass2 = EEclass
			if level in ['residLevel_resname']:
				EEclass2 = EEclass_residLevel_resname
			if level in ['soluteContacts']:
				EEclass2 = EEclass_soluteContacts
			if level in ['atomLevel']:
				EEclass2 = EEclass_atomLevel

			
			if entropyEnergy:
				name = 'EE'
				result = processEE(num_frames, totFrames, EEclass2, 
						solvent, waterTuple, 
						temperature, level, name, forceUnits, verbosePrint)
				result_list[level] = result
		# #'''
		# ##Save each class as an object so that we can continue populating
		# #in stages.
		# if len(paths) != 0:
		#     objectIteration += 1
		#     if pathClasses:
		#         print('\n3. Save Dictionaries')
		#         for Aclass in class_str_list:
		#             with gzip.GzipFile('%s.obj' % (Aclass), 'wb') as pickleFile:
		#                 pickle.dump((locals()[Aclass]), pickleFile, protocol=2)
		#                 pickleFile.close()
		#                 #locals()[Aclass] = None
		#         print('Number of objectIteration cycles saved: %s' % 
		#                 (objectIteration))
		#         print('Total atoms processed: %s' % (totAtoms))

		# #'''


		sys.stdout.flush()
		print('\n')
		print(datetime.now() - startTime)
		return result_list

class Poseidon_mp(Poseidon):
	def __init__(self ,container, start=0, end=-1, 
		step=1, pureAtomNum=1, cutShell=None, 
		excludedResnames=None,
		water='WAT', verbose=False, thread=4):
		"""This is a initialization function to collect information from a MDanalysis universe into a data container for analysis using POSEIDON
		Pending Work
			- rewrite this into a class initalization for easier understanding
			- rewrite the energy part to be compulsory

		Args:
			container (MDAnalysis.Universe): A MDAnalysis object with coordinates, forces and energy (loaded to velocity field with the value [Potential Energy(index 0), Kinetic Energy(1) and dummy(2)] respectively)
			start (int): Starting Frame ID. Defaults to 0.
			end (int): Ending Frame ID, this frame is not included. Defaults to -1.
			step (int, optional): Steps between frame. Defaults to 1.
			pureAtomNum (int, optional): Reference molecule resid for pure liquid. Defaults to 1.
			cutShell (float, optional): Explicit cut off shell (might be buggy since part of it is not defined). Default to None which uses the relative angular distance (RAD) algorithm. See Jonathan Higham and Richard H. Henchman , "Locally adaptive method to define coordination shell", J. Chem. Phys. 145, 084108 (2016)
			excludedResnames (list, optional): List of resnames to exclude from nearest non alike analysis. Defaults to None.
			water (str, optional): Resname for water molecules. Defaults to 'WAT'.
			verbose (bool, optional): print out progress of each analysis step. Defaults to False.
		"""
		startTime = datetime.now()
		print(startTime)
		verbosePrint = print if verbose else lambda *a, **k: None
		
		waterTuple = ('SOL', 'WAT', 'HOH', 'TIP3') #needed for pdb as top file 
		if water != 'WAT':
			waterTuple = (water)

		iterations = 0
		
		all_data = []

		populateTopology(container, all_data, waterTuple)
		verbosePrint('TOPOLOGY')
		verbosePrint(datetime.now() - startTime)
		sys.stdout.flush()


		resids = Counter([(i.resname) for i in all_data])
		verbosePrint(resids.keys())
		if len(resids.keys()) == 1:
			verbosePrint('Pure system with reference ID: %s' % (pureAtomNum))

		if excludedResnames != None:
			verbosePrint('EXCLUDED RESNAMES: %s' % (excludedResnames))
		

		dimensions = None
		allMoleculeList = []
		
		#fix end frame
		if end == -1:
			end = len(container.trajectory)
		sys.setrecursionlimit(3000000)
		p = Pool(processes=thread)
		fi_partial = partial(self.frame_iteration, container, all_data, dimensions, startTime, verbosePrint, waterTuple, cutShell, excludedResnames)
		data = p.map(fi_partial, [frame for frame in range(int(start), int(end), int(step))])
		# data = []
		# for frame in range(int(start), int(end), int(step)):
		#     data.append(fi_partial(frame))
		for item in data:
			allMoleculeList = moleculeObjectPopulation(item[0], 
					allMoleculeList, item[1], item[2])
		# #writing file here 
		# with gzip.GzipFile('moleculeListAll.obj', 'wb') as pickleFile:
		#     pickle.dump((allMoleculeList), pickleFile, protocol=2)
		#     pickleFile.close()

		print(datetime.now() - startTime)
			
		self.allMoleculeList = allMoleculeList
	
	def frame_iteration(self, container, all_data, dimensions, startTime, verbosePrint, waterTuple, cutShell, excludedResnames, frame):
		clearClass(all_data)
		print(f"frame = {frame}")

		all_data, dimensions = getCoordsForces(container, 
				all_data, dimensions, frame, startTime, 
				verbosePrint)
		# # Energy is not needed
		# populateEnergy(container, all_data, 
		#         dimensions, frame, startTime, verbosePrint)
		# UAEnergyGroup(all_data)


		calculateDihedrals(all_data, dimensions)
		verbosePrint('DIH')
		verbosePrint(datetime.now() - startTime)
		sys.stdout.flush()

		#'''
		traj = container.trajectory[frame]
		neighbour_coords = None
		neighbour_coords = traj.positions

		max_cutoff = 10
		for x in range(0, len(all_data)):
			atom = all_data[x]
			#start nearest array from solutes
			if atom.resname not in waterTuple:
				getDistArray(atom, all_data, traj, max_cutoff,
						dimensions, neighbour_coords,
						startTime, verbosePrint)
				#find nearest array for solute neighbours that are solvent
				for nearDist in atom.nearest_all_atom_array[0:20]:
					neighbour = all_data[nearDist[0]]
					if neighbour.resname in waterTuple and \
							neighbour.nearest_all_atom_array == None:
						getDistArray(neighbour, all_data, traj, max_cutoff,
								dimensions, neighbour_coords,
								startTime, verbosePrint)
					#find solvent neighbours neighbours nearest array
					if neighbour.nearest_all_atom_array != None:
						for nearDist2 in neighbour.nearest_all_atom_array[0:20]:
							neighbour2 = all_data[nearDist2[0]]
							if neighbour2.resname in waterTuple and \
									neighbour2.nearest_all_atom_array == None:
								getDistArray(neighbour2, all_data, 
										traj, max_cutoff,
										dimensions, 
										neighbour_coords,
										startTime, verbosePrint)
							else:
								continue
					else:
						continue
			else:
				continue
		verbosePrint('NEAREST ARRAYS')
		verbosePrint(datetime.now() - startTime)
		sys.stdout.flush()
		#'''



		if cutShell != None:
			#Used for fixed cut-off coordination shells
			try:
				cutoff_dist = float(cutShell)
				distCutoffNc(all_data, dimensions, cutoff_dist)
				# #don't know what this is it is not defined anywhere
				# NcPairs(all_data, dimensions)
				verbosePrint('distCutoffNc')
				verbosePrint(datetime.now() - startTime)
				sys.stdout.flush() 
			except ValueError:
				logging.error('Cutoff distance needs to be a float')


		if cutShell == None:
			UALevelRAD(all_data, dimensions)
			verbosePrint('RAD')
			verbosePrint(datetime.now() - startTime)


		#if inputType != 'pdb':
		HBCalc(all_data, waterTuple, dimensions)
		verbosePrint('HB')
		verbosePrint(datetime.now() - startTime)
		sys.stdout.flush() 


		getShellAssignment(all_data, excludedResnames, dimensions, 
				startTime, verbosePrint)
		verbosePrint('PROX')
		verbosePrint(datetime.now() - startTime)
		sys.stdout.flush()


		#if (force != None):
		calculateFTMatrix(all_data, dimensions)
		verbosePrint('FTMATRIX')
		verbosePrint(datetime.now() - startTime)
		sys.stdout.flush()


		moleculePositionRankingRAD(all_data, waterTuple, dimensions)
		verbosePrint('ORIENTS')
		verbosePrint(datetime.now() - startTime)
		sys.stdout.flush()

		'''
		if iterations == 1:
			verbosePrint('Generating .pdb file for frame %s' % (frame+1))
			pdbGenerate(all_data, ('frame_%s' % (frame+1)), dimensions)
		'''

		return (all_data, frame, dimensions)
