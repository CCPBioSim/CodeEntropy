""" This is a X-PLOR format PSF file reader """
__author__ = "Arghya Chakravorty"

import sys, os, re
import xdrlib
import numpy as nmp
from datetime import datetime

from CodeEntropy.Trajectory import TrajectoryConstants as TCON
from CodeEntropy.ClassCollection import BaseMolecule
from CodeEntropy.ClassCollection import BondStructs
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.IO import Writer

VECTORDIM = 3

class PsfAtomType(object):
	""" A class that stores the information about certain physical properties of a type of atom. the structure is designed to suit the information
	stored in a PSF file."""
	def __init__(self, arg_atomType, **kwargs):
		self.atomType = arg_atomType

		for prop, val in kwargs.items():
			setattr(self, prop, val)

	def set_attribute(self, **kwargs):
		for prop, val in kwargs.items():
			setattr(self, prop, val)
		return

#END

class PsfAtom(PsfAtomType):
	"""A child class that will store records of an atom based on its serial and not atom type."""
	def __init__(self, arg_index, **kwargs):
		super().__init__(arg_index)
		self.record = kwargs['atomInfo']



#END

class PsfTopology(object):
	""" A class that reads X-PLOR or CHARMM format PSF file """

	def __init__(self, arg_fileName, arg_verbose = 3):
		self.psfFile = arg_fileName
		self.molecule = self.read_psfFile(arg_verbose)
		
	   
	def read_psfFile(self, arg_verbose = 3):
		""" Reads the input PSF file and returns an instance of BaseMolecule if successfully read"""

		# list of residue indices with residue name
		# designed in order to appropriately populate residue based information 
		# in the BaseMolecule instance
		self.residueList= []

		# just like residueList but for segment Ids
		self.segidList = []

		# Order of atoms
		self.atomOrder = []


		# local variables
		self.isValidPSF = False
		self.isFormatSTD = False
		self.isFormatEXT = False
		self.isFormatCMAP = False
		self.isFormatCHEQ = False
		self.isFormatDRUDE = False
		self.isFormatXPLOR = False
		self.isFormatNAMD = False

		canReadAtom = False
		canReadBond = False
		canReadAngle = False
		canReadDihed = False
		canReadImpr = False
		canReadDonor = False
		canReadAcc = False
		canReadNNB = False
		canReadGrp = False
		canReadMolnt = False
		canReadLP = False
		canReadCRTerm = False


		# create an instance of BaseMolecule
		# Unlike gromacs, everything is considered as one single molecule in PSF based packages
		newMolecule = BaseMolecule.BaseMolecule(arg_name = '', arg_hostDataContainer = None)

		# for the sake of compatibility with information in gromacs's TPR file, add these information in 'newMolecule'
		# which are essentially dummies
		newMolecule.numCopies = 1


		# open file handler and read all the contents into a buffer
		with open(self.psfFile, 'r') as psfFileHandler:
			psfBuffer = psfFileHandler.readlines()

		i = 0
		atIdx = 0

		lastResid = -9999
		currResid = -1
		lastResname = "___"

		lastSegid = -9999
		currSegid = -1 

		while ( i < len(psfBuffer)):

			# read the i-th line
			line = psfBuffer[i]

			# skip blank lines
			if len(line.strip()) == 0:
				i += 1
				continue
				
			# look for "PSF" in the very first line
			if line[0:3] == "PSF" and not self.isValidPSF:
				self.isValidPSF = True
				psfFormat = " "
				if arg_verbose >= 5: 
					Utils.printflush("Valid PSF found.")

				# learn the format
				if "EXT" in line.strip():
					# Extended format PSF  (CHARMM)
					self.isFormatEXT = True
					psfFormat = "{} EXT".format(psfFormat)
					if arg_verbose >= 5: 
						Utils.printflush(psfFormat)

				if "CHEQ" in line.strip():
					# Extended format PSF and CHEQ
					self.isFormatCHEQ = True
					psfFormat = "{} CHEQ".format(psfFormat)
					if arg_verbose >= 5:
						Utils.printflush(psfFormat)

				if "CMAP" in line.strip():
					# Extended format PSF and CMAP
					self.isFormatCMAP = True
					psfFormat = "{} CMAP".format(psfFormat)
					if arg_verbose >= 5:
						Utils.printflush(psfFormat)

				if "DRUDE" in line.strip():
					# Extended format PSF and DRUDE
					self.isFormatDRUDE = True
					psfFormat = "{} DRUDE".format(psfFormat)
					if arg_verbose >= 5:
						Utils.printflush(psfFormat)

				if "XPLOR" in line.strip():
					# Extended format PSF and XPLOR
					self.isFormatXPLOR = True
					psfFormat = "{} XPLOR".format(psfFormat)
					if arg_verbose >= 5:
						Utils.printflush(psfFormat)

				if "NAMD" in line.strip():
					# NAMD/VMD format PSF
					self.isFormatNAMD = True
					psfFormat = "{} NAMD".format(psfFormat)
					if arg_verbose >= 5:
						Utils.printflush(psfFormat)

				# if none of the above then it is standard format
				if not (self.isFormatEXT or self.isFormatNAMD):
					# std format
					self.isFormatSTD = True
					psfFormat = "{} STD".format(psfFormat)
					if arg_verbose >= 5:
						Utils.printflush(psfFormat)

				i += 1
				continue

					
			######### REMARKS #########
			if re.match("^[0-9]+ !NTITLE$", line.strip()) and self.isValidPSF:
				searchRes = re.search("^([0-9]+) !NTITLE$", line.strip())
				nTitles = int(searchRes.group(1))
				if arg_verbose >= 5:
					Utils.printflush(f'Found {nTitles} remark lines')
				
				self.title = []
				for ititle in range(nTitles):
					self.title.append(psfBuffer[i+1+ititle])

				i += (1 + nTitles)   # move seeker beyond all the title lines
				continue
			
			
			if re.match("^[0-9]+ !NATOM$", line.strip()) and self.isValidPSF and not canReadAtom:
				canReadAtom = True
				self.atomList = []
				
				searchRes = re.search("^([0-9]+) !NATOM$", line.strip())
				nAtoms = int(searchRes.group(1))
				if arg_verbose >= 3:
					Utils.printflush(f'Expecting {nAtoms} atoms in the system')

				i += 1  
				continue
			 
			######### ATOMS #########   
			if canReadAtom:

				# time to change flags to enable reading Bond information if NBOND is encountered
				if re.match("^[0-9]+ !NBOND", line.strip()):
					canReadAtom = False
					canReadBond = True
					self.bondList = []
					
					# Before proceeding 
					# 1. fill residue index and name, segment information
					newMolecule.numAtoms = nAtoms
					newMolecule.numAtomsPerCopy = newMolecule.numAtoms
					newMolecule.numResidues = len(self.residueList)
					newMolecule.numSegments = len(self.segidList)

					for rinfo in self.residueList:
						resid, resname = rinfo
						newMolecule.residArray.append(resid)
						newMolecule.resnameArray.append(resname)

					for segn in self.segidList:
						newMolecule.segidArray.append(segn)


					# 2. correctly fill the cell-linked-list formatted residue information 
					newMolecule.atomArray = nmp.zeros(newMolecule.numAtoms)
					newMolecule.residueHead = -1 * nmp.ones(newMolecule.numResidues)

					for aid, rid in enumerate(newMolecule.atomResidueIdxArray):
						newMolecule.atomArray[aid] = newMolecule.residueHead[rid]
						newMolecule.residueHead[rid] = aid


					# 3. Also before fetching information about bonds, it needs to know 
					#    the nature of the atoms involved in the bond.
					#    i.e. is it C, Ca, N, BB, H, etc.
					# So now that all the atom inforamtion is supposed to have been read, 
					# use it to prepare these arrays in the 
					# instance of BaseMolecule

					# initialize
					newMolecule.initialize_element_info_arrays()

					# assign element types and priorities
					newMolecule.populate_element_info_arrays()


					# make a final check on the above assignments to the BaseMolecule instance
					newMolecule.validate_assignments()

					searchRes = re.search("^([0-9]+) !NBOND", line.strip())
					nBonds = int(searchRes.group(1))
					if arg_verbose >= 3:
						Utils.printflush(f'Expecting {nBonds} bonds in the structure')


				else:
					# Utils.printflush(line)
					# STD format
					if self.isFormatSTD:
						# // (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,I4,1X,2G14.6,I8)
						# //  II,LSEGID,LRESID,LRES,TYPE(I),IAC(I),CG(I),AMASS(I),IMOVE(I)
						atomSegid = line[9:13].strip()
						atomResid = int(line[14:18])
						atomResname = line[19:23].strip()
						atomName = line[24:28].strip()
						atomMass = float(line[48:64])
						atomInfo = line[24:]

					elif self.isFormatEXT and self.isFormatCHEQ:
						# In charmm's src code, the detailed information of the
						#  correct string format of PSF based on PSF tags can be found.
						# This formatting here is based on that.
						# Obviosuly the control flow and the condition blocks are
						# not placed like it is the CHARMM code.

						if self.isFormatXPLOR:
							# (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,A6,1X,2G14.6,I8,2G14.6)
							# II,LSEGID,LRESID,LRES,TYPE(I),IAC(I),CG(I),AMASS(I),IMOVE(I)
							atomSegid = line[11:19].strip()
							atomResid = int(line[20:28])
							atomResname = line[29:37].strip()
							atomName = line[38:46].strip()
							atomMass = float(line[68:82])
							atomInfo = line[38:]

						elif not isFormatXPLOR:
							# (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,I4,1X,2G14.6,I8,2G14.6)
							# II,LSEGID,LRESID,LRES,TYPE(I),IAC(I),CG(I),AMASS(I),IMOVE(I)
							# SECRET: the region that is diff (A6 vs I4) is of no use here. 
							# but still, just to be clear!
							atomSegid = line[11:19].strip()
							atomResid = int(line[20:28])
							atomResname = line[29:37].strip()
							atomName = line[38:46].strip()
							atomMass = float(line[68:82])
							atomInfo = line[38:]

					# populate residueList
					if not (lastResid == atomResid and lastResname == atomResname):
						currResid += 1
						lastResid = atomResid
						lastResname = atomResname
						self.residueList.append((atomResid, atomResname))
						if arg_verbose >= 5:
							Utils.printflush(f"Added a new residue : {atomResname} {atomResid}")
						

					# populate segidList
					if not (lastSegid == atomSegid) :
						currSegid += 1
						lastSegid = atomSegid
						self.segidList.append(atomSegid)
						if arg_verbose >= 5:
							Utils.printflush(f"Added a new segment : {atomSegid}")
						

					# create an object of PsfAtom
					### trim the right white spaces of atom lines 
					### to avoid getting a new line carrier repeated
					pAtom = PsfAtom(atIdx, **{'atomInfo' : re.sub("\s+$","",atomInfo)}) 
					pAtom.set_attribute(**{'atomIndex' : atIdx})
					
					self.atomList.append(pAtom)
					self.atomOrder.append(atIdx)

					# populate atom-based indices of BaseMolecule instance
					# print('Adding entry: ATOM {} {} {} {}'.format(atomIdx, atomName, atomResname, lastResid))
					newMolecule.atomIndexArray.append(atIdx)
					newMolecule.atomResidueIdxArray.append(currResid)
					newMolecule.atomSegmentIdxArray.append(currSegid)
					newMolecule.atomNameArray.append(atomName)
					newMolecule.atomMassArray.append(atomMass)

					
					
					atIdx += 1


				i += 1
				continue
				
			######### BONDS #########
			if canReadBond:
				# read bond information
				if re.match("^[0-9]+\s+[0-9]+", line.strip()):
					# each line has at most 4 bond pairs
					# print('Reading bond', line)
					atomsInLine = line.strip().split()
					assert(len(atomsInLine) % 2 == 0 and 2 <= len(atomsInLine) <= 8)  # should have 2K (k=1, 2, 3 or 4) indices
					jdx = 0
					while jdx < len(atomsInLine):
						aidI, aidJ = nmp.sort([ (int(atomsInLine[jdx + i]) - 1) for i in range(2) ])
						self.bondList.append([aidI, aidJ])
						newMolecule.add_bond_tree(arg_aidI=aidI, arg_aidJ=aidJ,arg_priorityLevelI=-1,arg_priorityLevelJ=-1)
						jdx += 2
					
			
				# time to change flags   
				elif re.match("^[0-9]+ !NTHETA", line.strip()):
					canReadBond = False
					canReadAngle = True
					self.angleList = []

					searchRes = re.search("^([0-9]+) !NTHETA", line.strip())
					nAngles = int(searchRes.group(1))
					if arg_verbose >= 3:
						Utils.printflush(f'Expecting {nAngles} angles in the structure')
					
				i += 1
				continue
		
			######### ANGLES #########
			if canReadAngle:
				# read angle information
				if re.match("^[0-9]+\s+[0-9]+\s+[0-9]+", line.strip()):
					# each line has at most 3 angle triplets
					# print('Reading angle', line)
					atomsInLine = line.split()
					assert(len(atomsInLine) % 3 == 0 and 3 <= len(atomsInLine) <= 9)  # should have 3K (k=1,2 or 3) indices
					
					jdx = 0
					while jdx < len(atomsInLine):
						# maintain the order of atom indices when reading angles 
						aIndices = ([int(atomsInLine[jdx + i])-1 for i in range(3) ])
						self.angleList.append(aIndices)
						jdx += 3


				elif re.match("^[0-9]+ !NPHI", line.strip()):
					canReadAngle = False
					canReadDihed = True
					self.dihedList = []

					searchRes = re.search("^([0-9]+) !NPHI", line.strip())
					nDiheds = int(searchRes.group(1))
					if arg_verbose >= 3:
						Utils.printflush(f'Expecting {nDiheds} dihedrals in the structure')
					
				i += 1
				continue
			 
			######### DIHEDRALS #########   
			if canReadDihed:
				# read dihedral information
				if re.match("^[0-9]+\s+[0-9]+\s+[0-9]+\s+[0-9]+", line.strip()):
				   
					# each line has at most 2 dihedral quadruples
					# print('Reading dihedral', line)
					atomsInLine = line.strip().split()
					assert(len(atomsInLine) % 4 == 0 and 4 <= len(atomsInLine) <= 8)  # should have 4K (k=1 or 2) indices
					
					jdx = 0
					while jdx < len(atomsInLine):
						# maintain the order of atom indices when reading dihedrals
						aIndices = ([int(atomsInLine[jdx + i])-1 for i in range(4) ] )
						newDih = BondStructs.Dihedral(aIndices, newMolecule)
						# Utils.printflush(newDih.atomList)
						newMolecule.add_dihedral(newDih)
						self.dihedList.append(aIndices)
						jdx += 4
				
				elif re.match("^[0-9]+ !NIMPHI", line.strip()):
					canReadDihed = False
					canReadImpr = True
					self.imprList = []

					searchRes = re.search("^([0-9]+) !NIMPHI", line.strip())
					nimphi = int(searchRes.group(1))
					if arg_verbose >= 3:
						Utils.printflush(f'Expecting {nimphi} impropers in the structure')

				i += 1
				continue

			######## IMPROPERS ########
			if canReadImpr:
				# read improper information
				if re.match("^[0-9]+\s+[0-9]+\s+[0-9]+\s+[0-9]+", line.strip()):
				   
					# each line has at most 2 dihedral quadruples
					# print('Reading dihedral', line)
					atomsInLine = line.strip().split()
					assert(len(atomsInLine) % 4 == 0 and 4 <= len(atomsInLine) <= 8)  # should have 4K (k=1 or 2) indices
					
					jdx = 0
					while jdx < len(atomsInLine):
						# maintain the order of atom indices when reading impropers
						aIndices =  ([int(atomsInLine[jdx + i])-1 for i in range(4) ])
						self.imprList.append(aIndices)
						jdx += 4

				elif re.match("^([0-9]+) !NDON", line.strip()):
					canReadImpr = False
					canReadDonor = True
					self.donorList = []

					searchRes = re.search("^([0-9]+) !NDON", line.strip())
					nDon = int(searchRes.group(1))
					if arg_verbose >= 3:
						Utils.printflush(f'Expecting {nDon} donors in the structure')

				i += 1
				continue

			####### DONORS #######
			if canReadDonor:
				# read donors
				if re.match("^[0-9]+\s+[0-9]+", line.strip()):

					# each line has at most 8 donor-H tuples
					atomsInLine = line.strip().split()
					assert(len(atomsInLine) % 2 == 0 and 2 <= len(atomsInLine) <= 8) 

					jdx = 0
					while jdx < len(atomsInLine):
						aidX, aidH = ([ (int(atomsInLine[jdx + i]) - 1) for i in range(2) ])
						self.donorList.append([aidX, aidH])
						newMolecule.add_bond_tree(arg_aidI=aidX, arg_aidJ=aidH,arg_priorityLevelI=1,arg_priorityLevelJ=1)
						jdx += 2

				elif  re.match("^([0-9]+) !NACC", line.strip()):
					canReadDonor = False
					canReadAcc = True
					self.accList = []

					searchRes = re.search("^([0-9]+) !NACC", line.strip())
					nAcc = int(searchRes.group(1))
					if arg_verbose >= 3:
						Utils.printflush(f'Expecting {nAcc} acceptors in the structure')

				i += 1
				continue

			####### ACCEPTORS #######
			if canReadAcc:
				# read acceptors
				if re.match("^[0-9]+\s+[0-9]+", line.strip()):

					# each line has at most 8 tuples (of what???)
					atomsInLine = line.strip().split()
					assert(len(atomsInLine) % 2 == 0 and 2 <= len(atomsInLine) <= 8) 

					jdx = 0
					while jdx < len(atomsInLine):
						aid1, aid2 = ([ (int(atomsInLine[jdx + i]) - 1) for i in range(2) ])
						self.accList.append([aid1, aid2])
						jdx += 2

				elif re.match("^([0-9]+) !NNB", line.strip()):
					canReadAcc = False
					canReadNNB = True
					nnbRead = 0
					self.nbList = []
					self.blockList = []

					searchRes = re.search("^([0-9]+) !NNB", line.strip())
					nnb = int(searchRes.group(1))
					if arg_verbose >= 3:
						Utils.printflush(f'Expecting {nnb} nonbond exclusions')

				i += 1
				continue

			####### NNB #######
			if canReadNNB:
				# READ NONBOND EXCLUSIONS
				if re.match("^[0-9\s]+$", line.strip()): # contains only numbers
					if nnbRead < nnb:
						# each line has at most 8 integers 
						# each integer corresponds to some block number (maybe?)
						for innb in line.strip().split():
							self.nbList.append(int(innb))

						nnbRead += 1
					else:
						# each line has at most 8 integers
						# each integer corresponds to every atom in the structure
						for iblo in line.strip().split():
							self.blockList.append(int(iblo))

				elif re.match("^([0-9]+)\s+([0-9]+) !NGRP NST2",line.strip()):
					canReadNNB = False
					canReadGrp = True
					self.grpList = []

					searchRes = re.search("^([0-9]+)\s+([0-9]+) !NGRP NST2", line.strip())
					ngrp = int(searchRes.group(1))
					nst2 = int(searchRes.group(2))

					if nst2 > 0:
						raise ValueError(f"CHARMM should complain if nst2 > 0 ({nst2}). SRC code says 'ST2 code is not compiled' ")
					
					if arg_verbose >= 3:
						Utils.printflush(f'Expecting {ngrp} groups in the structure')

				i += 1
				continue

			####### NGRP #######
			if canReadGrp:
				# read group information
				if re.match("^[0-9]+\s+[0-9]+\s+[0-9]+", line.strip()):
					# each line must have at most three 3-tuples (9 integers)
					# based on charmm's source code
					# the first integer is 0-indexed atomId (NOT 1-indexed!!)
					# the second integer is group type 
					# the third integer is something called 'moveg' (??) 
					atomsInLine = line.strip().split()
					assert(len(atomsInLine) % 3 == 0 and 3 <= len(atomsInLine) <= 9)

					jdx = 0
					while jdx < len(atomsInLine):
						igpbs, igptyp, imoveg = [ int(atomsInLine[jdx + i])  for i in range(3) ]
						# info from charmm/source/nbonds/enbondg.F90
						# IGPBS  - base array for groups  (NGRP+1)
						# IGPTYP - group type array (0-no charges,1-neutral,2-charged,3-ST2)
						# IMOVEG - ???
						self.grpList.append((igpbs, igptyp, imoveg))
						jdx += 3

				elif re.match("^[0-9]+ !MOLNT", line.strip()) and self.isFormatCHEQ:
					canReadGrp = False
					canReadMolnt = True  # whatever  that is
					self.moltList = []

					searchRes = re.search("^([0-9]+) !MOLNT", line.strip())
					molnt = int(searchRes.group(1))

					if arg_verbose >= 3:
						Utils.printflush(f'Expecting {molnt} unique MOLT values for atoms in the structure')

				i += 1
				continue

			####### MOLNT #######
			if canReadMolnt:
				# read molt values for every atom in the structure
				if re.match("^[0-9\s]+$", line.strip()): 
					# each line has at most 8 integers
					# each  integer corresponds to every atom in the structure
					for imolt in line.strip().split():
						self.moltList.append(int(imolt))

				elif re.match("^[0-9]+\s+[0-9]+ !NUMLP NUMLPH", line.strip()):
					# enable reading lone pair stuff
					canReadMolnt = False
					canReadLP = True
					self.lpLines = []
					self.lpAtomList = []
					nLPRead = 0 # used below a counter of lines

					searchRes = re.search('^([0-9]+)\s+([0-9]+) !NUMLP NUMLPH', line.strip())
					nLP = int(searchRes.group(1))
					nlPH = int(searchRes.group(2))
					if arg_verbose == 3:
						Utils.printflush(f'Expecting {nLP} lonepair(s) and {nlPH} lonepair connections')


				i += 1
				continue

			####### LONEPAIR ########
			if canReadLP:
				# read lone pair stuff
				if nLPRead < nLP:
					self.lpLines.append(line) # will be printed out as is
					nLPRead += 1

				else:
					# read the next section of this directive
					# where a line can have at most 8 integers (all atom indices)
					if re.match("^[0-9\s]+$", line.strip()):
						atomsInLine = line.strip().split()
						jdx = 0
						while jdx < len(atomsInLine):
							self.lpAtomList.append(int(atomsInLine[jdx])-1)
							jdx += 1

				if re.match("^[0-9]+ !NCRTERM", line.strip()):
					# enable reading cross terms
					canReadLP = False
					canReadCRTerm = True
					self.crossTermList = []

					searchRes = re.search("^([0-9]+) !NCRTERM", line.strip())
					nCRTerm = int(searchRes.group(1))

					if arg_verbose >= 3:
						Utils.printflush(f'Expecting {nCRTerm} cross term maps in the structure.')

				i += 1
				continue

	 
			######### IGNORE EVERYTHING ELSE #########
			else:
				# if anything else shows up (like Improper, Cross-term, non-bonded params, etc. )
				# ignore them till the end of the PSF file
				#
				# Hopefully everything that is needed has been read in already!
				#
				break
		

		if (self.isValidPSF and nAtoms > 0 and nBonds > 0 and nAngles > 0 and nDiheds > 0):
			return newMolecule
		else:
			return None

	def write_psfFile(self, arg_outname,  arg_verbose = 3):
		"""
		Write a PSF file `arg_outname` .
		"""

		# open file for writing
		Writer.write_file(arg_outname)

		# start writing content
		if self.isFormatEXT:
			Utils.printOut(arg_outname, "PSF EXT", end='')
			IFMT = "10d"
		else:
			self.Utils.printOut(arg_outname, "PSF", end='')
			IFMT = "8d"
		
		if self.isFormatCMAP:
			Utils.printOut(arg_outname, " CMAP", end= '')
		if self.isFormatCHEQ:
			Utils.printOut(arg_outname," CHEQ", end =  '')
		if self.isFormatDRUDE:
			Utils.printOut(arg_outname, " DRUDE", end='')
		if self.isFormatXPLOR:
			Utils.printOut(arg_outname, " XPLOR", end= '')

		Utils.printOut(arg_outname,"")

		# NTITLE
		nTitle = len(self.title)
		timeNow = datetime.now()
		Utils.printOut(arg_outname,f"\n{(nTitle+1):{IFMT}} !NTITLE")
		Utils.printOut(arg_outname,f"* PSF suitable for DOMDEC written on DATE : {timeNow.strftime('%b %d, %Y')}")
		for strTitle in self.title:
			Utils.printOut(arg_outname,strTitle,end='')

		Utils.printOut(arg_outname,"")

		# ATOMS
		nAtoms = len(self.atomList)
		Utils.printOut(arg_outname,f"{nAtoms:{IFMT}} !NATOM")

		for idx, atIdx in enumerate(self.atomOrder):
			rdx = self.molecule.atomResidueIdxArray[atIdx]
			resi = str(self.molecule.residArray[rdx])
			resn = self.molecule.resnameArray[rdx]
			segi = self.segidList[self.molecule.atomSegmentIdxArray[atIdx]]

			atomLine = f"{idx+1 : {IFMT}} {segi:8s} {resi:8s} {resn:8s} {self.atomList[atIdx].record}"
			Utils.printOut(arg_outname,atomLine)

		# BONDS
		nBonds = len(self.bondList)
		Utils.printOut(arg_outname, f"\n{nBonds:{IFMT}} !NBOND: bonds")
		for bdx, iBond in enumerate(self.bondList):
			aid1, aid2 = iBond
			aid1 = self.atomList[aid1].atomIndex + 1
			aid2 = self.atomList[aid2].atomIndex + 1
			self.bondList[bdx] = (aid1, aid2)

		Utils.printList(arg_outname, nmp.asarray(self.bondList).flatten(), IFMT, 8)
			
			# # at most 4 bond 2-tuples per line
			# Utils.printOut(arg_outname,f"{aid1:{IFMT}}{aid2:{IFMT}}",end='')
			# if (bdx + 1) % 4 == 0:
			# 	Utils.printOut(arg_outname, "")

		#ANGLES
		nAngles = len(self.angleList)
		Utils.printOut(arg_outname,f"\n{nAngles:{IFMT}} !NTHETA: angles")
		for adx, iAngle in enumerate(self.angleList):
			aid1, aid2, aid3 = iAngle
			aid1 = self.atomList[aid1].atomIndex + 1
			aid2 = self.atomList[aid2].atomIndex + 1
			aid3 = self.atomList[aid3].atomIndex + 1
			self.angleList[adx] = (aid1, aid2, aid3)

		Utils.printList(arg_outname, nmp.asarray(self.angleList).flatten(), IFMT, 9)

			# # at most 3 angle 3-tuples per line
			# Utils.printOut(arg_outname,f"{aid1:{IFMT}}{aid2:{IFMT}}{aid3:{IFMT}}",end='')
			# if (adx + 1) % 3 == 0:
			# 	Utils.printOut(arg_outname,"")

		#DIHEDRALS
		nDiheds = len(self.dihedList)
		Utils.printOut(arg_outname,f"\n{nDiheds:{IFMT}} !NPHI: dihedrals")
		for ddx, iDihe in enumerate(self.dihedList):
			aid1, aid2, aid3, aid4 = iDihe
			aid1 = self.atomList[aid1].atomIndex + 1
			aid2 = self.atomList[aid2].atomIndex + 1
			aid3 = self.atomList[aid3].atomIndex + 1
			aid4 = self.atomList[aid4].atomIndex + 1
			self.dihedList[ddx] = (aid1, aid2, aid3, aid4)

		Utils.printList(arg_outname, nmp.asarray(self.dihedList).flatten(), IFMT, 8)

			# # at most 2 dihedral 4-tuples per line
			# Utils.printOut(arg_outname,f"{aid1:{IFMT}}{aid2:{IFMT}}{aid3:{IFMT}}{aid4:{IFMT}}",end='')
			# if (ddx + 1) % 2 == 0:
			# 	Utils.printOut(arg_outname,"")

		#IMPROPERS
		nimphi = len(self.imprList)
		Utils.printOut(arg_outname,f"\n{nimphi:{IFMT}} !NIMPHI: impropers")
		for mdx, iImpr in enumerate(self.imprList):
			aid1, aid2, aid3, aid4 = iImpr
			aid1 = self.atomList[aid1].atomIndex + 1
			aid2 = self.atomList[aid2].atomIndex + 1
			aid3 = self.atomList[aid3].atomIndex + 1
			aid4 = self.atomList[aid4].atomIndex + 1

			self.imprList[mdx] = (aid1, aid2, aid3, aid4)
		Utils.printList(arg_outname, nmp.asarray(self.imprList).flatten(), IFMT, 8)

			# # at most 2 improper 4-tuples per line
			# Utils.printOut(arg_outname,f"{aid1:{IFMT}}{aid2:{IFMT}}{aid3:{IFMT}}{aid4:{IFMT}}",end='')
			# if (mdx + 1) % 2 == 0:
			# 	Utils.printOut(arg_outname,"")

		#DONORS
		nDon = len(self.donorList)
		Utils.printOut(arg_outname,f"\n{nDon:{IFMT}} !NDON: donors")
		for hddx, iDon in enumerate(self.donorList):
			aid1, aid2 = iDon
			aid1 = self.atomList[aid1].atomIndex + 1
			aid2 = self.atomList[aid2].atomIndex + 1
			self.donorList[hddx] = (aid1, aid2)
		Utils.printList(arg_outname, nmp.asarray(self.donorList).flatten(), IFMT, 8)
			
			# at most 4 X-H 2-tuples per line
			# Utils.printOut(arg_outname,f"{aid1:{IFMT}}{aid2:{IFMT}}",end='')
			# if (hddx + 1) % 4 == 0:
			# 	Utils.printOut(arg_outname,"")

		# ACCEPTORS
		nAcc = len(self.accList)
		Utils.printOut(arg_outname, f"\n{nAcc:{IFMT}} !NACC: acceptors")
		for hadx, iAcc in enumerate(self.accList):
			aid1, aid2 = iAcc
			aid1 = self.atomList[aid1].atomIndex + 1
			aid2 = self.atomList[aid2].atomIndex + 1
			self.accList[hadx] = (aid1, aid2)
		Utils.printList(arg_outname, nmp.asarray(self.accList).flatten(), IFMT, 8)

			# at most 4 A-X 2-tuples per line
			# Utils.printOut(arg_outname,f"{aid1:{IFMT}}{aid2:{IFMT}}",end='')
			# if (hddx + 1) % 4 == 0:
			# 	Utils.printOut(arg_outname,"")

		# NNB
		numNNB = len(self.nbList)
		Utils.printOut(arg_outname, f"\n{numNNB:{IFMT}} !NNB")
		Utils.printList(arg_outname, nmp.asarray(self.nbList), IFMT, 8)
		Utils.printList(arg_outname, nmp.asarray(self.blockList), IFMT, 8)

		# for nbdx, nb in enumerate(self.nbList):
		# 	Utils.printOut(arg_outname,f"{nb:{IFMT}}",end='')

		# 	# 8 integers per line
		# 	if (nbdx + 1) % 8 == 0:
		# 		Utils.printOut(arg_outname,"")

		# for bldx, blo in enumerate(self.blockList):
		# 	Utils.printOut(arg_outname,f"{blo:{IFMT}}",end='')

		# 	# 8 integers per line
		# 	if (bldx + 1) % 8 == 0:
		# 		Utils.printOut(arg_outname,"")

		#NGRP
		ngrp = len(self.grpList)
		nst2 = 0 # just assume
		Utils.printOut(arg_outname,f"\n{ngrp:{IFMT}}{nst2:{IFMT}} !NGRP NST2")
		Utils.printList(arg_outname, nmp.asarray(self.grpList).flatten(), IFMT, 9)

		# for gdx, grp3 in enumerate(self.grpList):
		# 	igpbs, igptyp, imoveg = grp3
		# 	Utils.printOut(arg_outname,f"{igpbs:{IFMT}}{igptyp:{IFMT}}{imoveg:{IFMT}}",end='')

		# 	# at most 3 3-tuples per line
		# 	if (gdx + 1) % 3 == 0:
		# 		Utils.printOut(arg_outname,"")

		#MOLNT
		if self.isFormatCHEQ:
			nmolt = len(nmp.unique(self.moltList))
			Utils.printOut(arg_outname,f"\n{nmolt:{IFMT}} !MOLNT")
			Utils.printList(arg_outname, nmp.asarray(self.moltList), IFMT, 8)

			# for mldx, molt in enumerate(self.moltList):
			# 	Utils.printOut(arg_outname,f"{molt:{IFMT}}",end='')

			# 	# at most 8 integers per line
			# 	if (mldx + 1) % 8 == 0:
			# 		Utils.printOut(arg_outname,"")

		#LONEPAIR
		nLP = len(self.lpLines)
		nLPH = len(self.lpAtomList)
		if nLP == 0:
			Utils.printOut(arg_outname,f"\n{0:{IFMT}}{0:{IFMT}} !NUMLP NUMLPH")
		else:
			for lpl in self.lpLines:
				Utils.printOut(arg_outname, lpl)

			for lpdx, lpAtom in enumerate(self.lpAtomList):
				lpAtom = self.atomList[lpAtom].atomIndex + 1
				self.lpAtomList[lpdx] = lpAtom

			Utils.printList(arg_outname, nmp.asarray(self.lpAtomList), IFMT, 8)

				# Utils.printOut(arg_outname,f"{aid0:{IFMT}}",end='')

				# # at most 8 integers per line (why when its triplets?)
				# if (lpdx + 1) % 8 == 0:
				# 	Utils.printOut(arg_outname,"")

		#CMAP or CROSS TERM MAP
		if self.isFormatCMAP:
			Utils.printOut(arg_outname,"")
			nCRTerm = len(self.crossTermList)
			Utils.printOut(arg_outname,f"\n{nCRTerm:{IFMT}} !NCRTERM: cross-terms")


		return
	#END



	
