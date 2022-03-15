import sys, os
import xdrlib
import struct
import numpy as nmp

from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.Trajectory import TrajectoryConstants as TCON
from CodeEntropy.ClassCollection import BaseMolecule as BM

VECTORDIM = 3

class GroAtomType(object):
    """ A class that stores the information about certain physical properties of a type of atom. the structure is designed to suit the information
    stored by Gromacs."""
    def __init__(self, arg_index):
        self.atomTypeIndex = arg_index
        self.atomNumber    = -1

    def __str__(self):
        return '{} : atomnumber = {}'. \
        format(self.atomTypeIndex, self.atomNumber)


class GroTopology(object):
    """ A class that reads gromacs topology in TPR format """

    def __init__(self, arg_fileName, arg_hostDataContainer):
        self.tprFile = arg_fileName
        self.hostDataContainer = arg_hostDataContainer   # the container where everything is stored

        # list of molecules prsent in the topology
        self.moleculeList = []

        # dictionary <int : GroAtomType> of gromacs atomtypes
        self.groAtomTypeDict = dict()

        # metadata dictionary
        self.metadata = {}
        self.metadata['atom_counts'] = [] #list of number of atoms in molecule in the topology
        self.metadata['residue_counts'] = [] #list of number of residues in molecule in the topology
        self.metadata['molblock_types'] = [] #list of intergers indicating the molecule blocks
        self.metadata['molblock_numMols'] = [] #list of intergers indicating the number of molecules per molblock
        self.metadata['molblock_numAtoms'] = [] #list of intergers indicating the number of atoms per molecule

        

    def get_header_information(self, arg_xrdUnpacker):

        up = arg_xrdUnpacker

        # read version (string; preceeded by an unsigned integer iindicating its size + 1(?))
        up.unpack_int(); 
        self.metadata['version'] = up.unpack_string()

        # precision (int)
        self.metadata['precision'] = up.unpack_int()
        if self.metadata['precision'] not in [4,8]:
            raise ValueError('Bad precision for real data. Should be 4(float) or 8(double).{} found'.format(self.metadata['precision']))

        # file version (int)
        self.metadata['file_version'] = fileVersion = up.unpack_int()

        # fetching file tag before generation for certain versions 
        if 79 >= fileVersion >= 77:
            up.unpack_int()
            self.metadata['file_tag'] = up.unpack_string()

        # file generation (int)
        # exists for version >= 26 
        self.metadata['file_generation'] = up.unpack_int()

        # file_tag (string) 
        if fileVersion >= 81:
            up.unpack_int()
            self.metadata['file_tag'] = up.unpack_string()
        elif fileVersion < 77:
            self.metadata['file_tag'] = TCON.TPX_TAG_RELEASE

        # <n_atoms, n_gtc (number of groups for t-coupling)> (int) 
        for hKey in ('n_atoms', 'n_gtc'):
            self.metadata[hKey] = up.unpack_int()

        #idum and rdum (int)
        if fileVersion < 62:
            self.metadata['idum'] = up.unpack_int()
            self.metadata['rdum'] = up.unpack_int()

        # fep state (int)
        if fileVersion >= 79:
            self.metadata['fep_state'] = up.unpack_int()

        # lambda
        self.metadata['lambda'] = self.unpack_real(up)
        # if self.metadata['precision'] == 4:
        #     self.metadata['lambda'] = up.unpack_float()
        # else:
        #     self.metadata['lambda'] = up.unpack_double()

        # <bIr (input record?), bTop (topology?), bX (coords?), bV (velocities?), bF (forces?), bBox (box info?)> (bool)
        for hKey in ('bIr', 'bTop', 'bX', 'bV', 'bF', 'bBox'):
            self.metadata[hKey] = up.unpack_int()

        return
#END 

    def get_box_data(self, arg_xrdUnpacker):
        """ Reads the box vectors """
        try:
            fileVersion = self.metadata['file_version']
        except:
            raise KeyError('file_version should be set before reading box information')

        up = arg_xrdUnpacker
        if self.metadata['bBox']:
            self.metadata['box'] = nmp.asarray([self.unpack_real(up) for i in range(TCON.VECTORDIM * TCON.VECTORDIM)])
            if fileVersion >= 51:
                self.metadata['box_rel'] = nmp.asarray([self.unpack_real(up) for i in range(TCON.VECTORDIM * TCON.VECTORDIM)])

            self.metadata['box_v'] = nmp.asarray([self.unpack_real(up) for i in range(TCON.VECTORDIM * TCON.VECTORDIM)])
            if fileVersion < 56:
                self.metadata['mdum'] = nmp.asarray([self.unpack_real(up) for i in range(TCON.VECTORDIM * TCON.VECTORDIM)])
        return
#END

    def get_t_couple_groups(self, arg_xrdUnpacker):
        """ Get the index of temperature coupled groups """
        try:
            fileVersion = self.metadata['file_version']
        except:
            raise KeyError('file_version should be set before reading box information')

        up = arg_xrdUnpacker
        if self.metadata['n_gtc']:
            if self.metadata['precision'] == 4:
                if fileVersion < 69:
                    self.metadata['t_couple_groups'] = nmp.asarray([up.unpack_float() for i in range(self.metadata['n_gtc'])])

                # redunandant call??
                self.metadata['t_couple_groups'] = nmp.asarray([up.unpack_float() for i in range(self.metadata['n_gtc'])])

            else:
                if fileVersion < 69:
                    self.metadata['t_couple_groups'] = nmp.asarray([up.unpack_double() for i in range(self.metadata['n_gtc'])])

                # redunandant call??
                self.metadata['t_couple_groups'] = nmp.asarray([up.unpack_double() for i in range(self.metadata['n_gtc'])])
        return
#END

    def get_topology_symbols(self, arg_xrdUnpacker):
        """ Counts and reads the symbols in the topology file"""
        # gromacs's do_symtab in tpxio.c
        up = arg_xrdUnpacker
        self.metadata['n_symbols'] = up.unpack_int()
        if self.metadata['n_symbols']:
            self.metadata['symbols'] = []
            for i in range(self.metadata['n_symbols']):
                up.unpack_int()
                self.metadata['symbols'].append(up.unpack_string())
        return
#END    
    
    def get_symstr(self, arg_xrdUnpacker):
        up = arg_xrdUnpacker
        return up.unpack_int()
#END

    def get_ff_params(self, arg_xrdUnpacker):
        """ Inspired by the do_ffparam function of gromacs in tpxio.c """

        fileVersion = self.metadata['file_version']
        up = arg_xrdUnpacker

        for iFunctionId in self.metadata['funcTypes']:

            # whether update is required
            for ftupdElement in TCON.FTUPD:
                if fileVersion < ftupdElement[0] and iFunctionId >= ftupdElement[1]:
                    iFunctionId += 1

            # assign paramters to each functionType
            precision = self.metadata['precision']
            # 1. F_ANGLES, F_G96ANGLES, F_BONDS, F_G96BONDS, F_HARMONIC, F_IDIHS
            if iFunctionId in map(lambda ftype: TCON.FTYPE_ENUM[ftype], ['F_ANGLES', 'F_G96ANGLES', 'F_BONDS', 'F_G96BONDS', 'F_HARMONIC', 'F_IDIHS']):
                # do harmonic (unpack 4 real numbers sequentially)
                self.unpack_n_real(up, 4)
                pass

            elif iFunctionId == TCON.FTYPE_ENUM['F_RESTRANGLES']:
                self.unpack_n_real(up, 2)
                
            elif iFunctionId == TCON.FTYPE_ENUM['F_LINEAR_ANGLES']:
                self.unpack_n_real(up, 4)

            elif iFunctionId == TCON.FTYPE_ENUM['F_FENEBONDS']:
                self.unpack_n_real(up, 2)

            elif iFunctionId == TCON.FTYPE_ENUM['F_RESTRBONDS']:
                self.unpack_n_real(up, 8)

            elif iFunctionId in map(lambda ftype: TCON.FTYPE_ENUM[ftype], ['F_TABBONDS', 'F_TABBONDSNC', 'F_TABANGLES', 'F_TABDIHS']):
                self.unpack_real(up)
                up.unpack_int()
                self.unpack_real(up)

            elif iFunctionId == TCON.FTYPE_ENUM['F_CROSS_BOND_BONDS']:
                self.unpack_n_real(up, 3)

            elif iFunctionId == TCON.FTYPE_ENUM['F_CROSS_BOND_ANGLES']:
                self.unpack_n_real(up, 4)

            elif iFunctionId == TCON.FTYPE_ENUM['F_UREY_BRADLEY']:
                self.unpack_n_real(up, 4)
                if fileVersion >= 79:
                    self.unpack_n_real(up, 4)
                else:
                    # some internal assignments are done in Gromacs
                    pass

            elif iFunctionId == TCON.FTYPE_ENUM['F_QUARTIC_ANGLES']:
                self.unpack_n_real(up, 1 + 5)

            elif iFunctionId == TCON.FTYPE_ENUM['F_BHAM']:
                self.unpack_n_real(up, 3)

            elif iFunctionId == TCON.FTYPE_ENUM['F_MORSE']:
                self.unpack_n_real(up, 3)
                if fileVersion >= 79:
                    self.unpack_n_real(up, 3)
                else:
                    # some internal assignments are done in Gromacs
                    pass

            elif iFunctionId == TCON.FTYPE_ENUM['F_CUBICBONDS']:
                self.unpack_n_real(up, 3)

            elif iFunctionId == TCON.FTYPE_ENUM['F_CONNBONDS']:
                pass

            elif iFunctionId == TCON.FTYPE_ENUM['F_POLARIZATION']:
                self.unpack_real(up)

            elif iFunctionId == TCON.FTYPE_ENUM['F_ANHARM_POL']:
                self.unpack_n_real(up, 3)

            elif iFunctionId == TCON.FTYPE_ENUM['F_WATER_POL']:
                if fileVersion < 31:
                    # GROMACS ERROR : Old tpr files with water_polarization not supported. Make a new.
                    pass
                else:
                    self.unpack_n_real(up, 6)

            elif iFunctionId == TCON.FTYPE_ENUM['F_THOLE_POL']:
                self.unpack_n_real(up, 4)

            elif iFunctionId == TCON.FTYPE_ENUM['F_LJ']:
                self.unpack_n_real(up, 2)

            elif iFunctionId == TCON.FTYPE_ENUM['F_LJ14']:
                self.unpack_n_real(up, 4)

            elif iFunctionId == TCON.FTYPE_ENUM['F_LJC14_Q']:
                self.unpack_n_real(up, 5)

            elif iFunctionId == TCON.FTYPE_ENUM['F_LJC_PAIRS_NB']:
                self.unpack_n_real(up, 4)

            elif iFunctionId in map(lambda ftype: TCON.FTYPE_ENUM[ftype], ['F_PDIHS', 'F_PIDIHS', 'F_ANGRES', 'F_ANGRESZ']):
                self.unpack_n_real(up, 2)
                if iFunctionId in [TCON.FTYPE_ENUM['F_ANGRES'], TCON.FTYPE_ENUM['F_ANGRESZ']] and fileVersion < 42:
                    # GROMACS : Read the incorrectly stored multiplicity
                    self.unpack_n_real(up, 2)
                else:
                    self.unpack_n_real(up, 2)
                    self.unpack_n_int(up, 1)

            elif iFunctionId == TCON.FTYPE_ENUM['F_RESTRDIHS']:
                self.unpack_n_real(up, 2)

            elif iFunctionId == TCON.FTYPE_ENUM['F_DISRES']:
                self.unpack_n_int(up, 2)
                self.unpack_n_real(up, 4)

            elif iFunctionId == TCON.FTYPE_ENUM['F_ORIRES']:
                self.unpack_n_int(up, 3)
                self.unpack_n_real(up, 3)

            elif iFunctionId == TCON.FTYPE_ENUM['F_DIHRES']:
                if fileVersion < 82:
                    self.unpack_n_int(up, 2)

                self.unpack_n_real(up, 3)
                if fileVersion >= 82:
                    self.unpack_n_real(up, 3)
                else:
                    # some internal assignments are done in Gromacs
                    pass

            elif iFunctionId == TCON.FTYPE_ENUM['F_POSRES']:
                # positions
                self.unpack_n_real(up, 3) #A
                # forces
                self.unpack_n_real(up, 3) #A
                if fileVersion < 27:
                    # some internal assignments are done in Gromacs
                    pass
                else:
                    self.unpack_n_real(up, 3) #B
                    self.unpack_n_real(up, 3) #B

            elif iFunctionId == TCON.FTYPE_ENUM['F_FBPOSRES']:
                self.unpack_n_int(up, 1)
                self.unpack_n_real(up, 3) #vector
                self.unpack_n_real(up, 2)

            elif iFunctionId == TCON.FTYPE_ENUM['F_CBTDIHS']:
                self.unpack_n_real(up, TCON.NR_CBTDIHS)

            elif iFunctionId == TCON.FTYPE_ENUM['F_RBDIHS']:
                self.unpack_n_real(up, TCON.NR_RBDIHS)
                if fileVersion > 25:
                    self.unpack_n_real(up, TCON.NR_RBDIHS)

            elif iFunctionId == TCON.FTYPE_ENUM['F_FOURDIHS']:
                self.unpack_n_real(up, 2 * TCON.NR_RBDIHS)

            elif iFunctionId in map(lambda ftype: TCON.FTYPE_ENUM[ftype], ['F_CONSTR', 'F_CONSTRNC']):
                self.unpack_n_real(up, 2)

            elif iFunctionId == TCON.FTYPE_ENUM['F_SETTLE']:
                self.unpack_n_real(up, 2)

            elif iFunctionId == TCON.FTYPE_ENUM['F_VSITE2']:
                self.unpack_n_real(up, 1)

            elif iFunctionId in map(lambda ftype: TCON.FTYPE_ENUM[ftype], ['F_VSITE3', 'F_VSITE3FD', 'F_VSITE3FAD']):
                self.unpack_n_real(up, 2)

            elif iFunctionId in map(lambda ftype: TCON.FTYPE_ENUM[ftype], ['F_VSITE3OUT', 'F_VSITE4FD', 'F_VSITE4FDN']):
                self.unpack_n_real(up, 3)

            elif iFunctionId == TCON.FTYPE_ENUM['F_VSITEN']:
                self.unpack_n_int(up, 1)
                self.unpack_n_real(up, 1)

            elif iFunctionId in map(lambda ftype: TCON.FTYPE_ENUM[ftype], ['F_GB12', 'F_GB13', 'F_GB14']):
                if fileVersion < 68:
                    self.unpack_n_real(up, 4)
                self.unpack_n_real(up, 5)

            elif iFunctionId == TCON.FTYPE_ENUM['F_CMAP']:
                self.unpack_n_int(up, 2)

            else:
                Utils.printflush('Unknown function type {} '.format(iFunctionId))
        return
#END

    def read_atom_information(self, arg_baseMolecule, arg_xrdUnpacker):
        """ Equivalent of do_atoms (combines do_atom as well) """

        if not arg_baseMolecule.numAtoms:
            raise ValueError('No atom found in the molecule. That is not correct.')

        else:
            up = arg_xrdUnpacker
            fileVersion = self.metadata['file_version']

            # create the residueHead and atomArray which work in the cell-linked-list
            # see BaseMolecule.py
            arg_baseMolecule.residueHead = -1 * nmp.ones(arg_baseMolecule.numResidues)
            arg_baseMolecule.atomArray = nmp.zeros(arg_baseMolecule.numAtoms)


            for atIdx in range(arg_baseMolecule.numAtoms):
                arg_baseMolecule.atomIndexArray.append(atIdx)

                iMass = self.unpack_real(up)
                arg_baseMolecule.atomMassArray.append(iMass)

                iCharge = self.unpack_real(up)
                arg_baseMolecule.atomChargeArray.append(iCharge)

                iMassB = self.unpack_real(up)     # not our concern
                iChargeB = self.unpack_real(up)   # not our concern

                iType = up.unpack_uint()          # of concern
                arg_baseMolecule.atomGromacsTypeArray.append(iType)

                iTypeB = up.unpack_uint()         # not our concern

                iPType = up.unpack_int()   #'Atom' or 'vitual site', etc. (not our concern)

                iResidueIndex = up.unpack_int()   
                arg_baseMolecule.atomResidueIdxArray.append(iResidueIndex)

                # populates the residueHead and atomArray which work in the cell-linked-list
                arg_baseMolecule.atomArray[atIdx] = arg_baseMolecule.residueHead[iResidueIndex]
                arg_baseMolecule.residueHead[iResidueIndex] = atIdx


                if fileVersion >= 52:
                    iAtomNumber = up.unpack_int()
                else:
                    iAtomNumber = TCON.NOTSET

                # skipping parts where group numbers are assigned to atoms
                # not necessary here

            #do_strstr to fetch atomNames
            arg_baseMolecule.atomNameArray = [self.metadata['symbols'][up.unpack_int()].decode('ascii') for atIdx in range(arg_baseMolecule.numAtoms)]

            # atomtype assignment
            if fileVersion > 20:
                # element assignment that is useful
                # arg_baseMolecule.atomTypeArray = [self.metadata['symbols'][up.unpack_int()] for iAtom in range(arg_baseMolecule.numAtoms)]
                # Utils.printflush(arg_baseMolecule.atomTypeArray)
                self.unpack_n_int(up, arg_baseMolecule.numAtoms)

                # typeB assignemnt that is not useful
                self.unpack_n_int(up, arg_baseMolecule.numAtoms)

            else:
                # say that this is a very old version
                Utils.printflush('This is a primitive version of gromacs that is not read by this program!!!')
        
        return
#END

    def read_residue_information(self, arg_baseMolecule, arg_xrdUnpacker):
        """ Equivalent of do_resinfo for residue assignment """

        if not arg_baseMolecule.numAtoms:
            raise ValueError('No atom found in the molecule. That is not correct.')

        else:
            fileVersion = self.metadata['file_version']
            up = arg_xrdUnpacker

            # -> do_resinfo
            for iRes in range(arg_baseMolecule.numResidues):
                if fileVersion >= 63:
                    rname = self.metadata['symbols'][up.unpack_int()].decode('ascii')
                    rid = up.unpack_int()
                    up.unpack_fstring(1)  #unpack unsigned char

                    arg_baseMolecule.resnameArray.append(rname)
                    arg_baseMolecule.residArray.append(rid)

        return
#END

    def read_iLists_information(self, arg_baseMolecule, arg_xrdUnpacker):
        """ Equivalent of do_ilists """

        fileVersion = self.metadata['file_version']
        up = arg_xrdUnpacker

        for iEnerType in range(TCON.FTYPE_ENUM['F_NRE']):
            bClear = False

            for ftupdElement in TCON.FTUPD:
                if fileVersion < ftupdElement[0] and iEnerType == ftupdElement[1]:
                    bClear = True

            if bClear:
                lenAtomList = 0
                intAtomList = None

            else:
                # ->->-> do_ilist
                lenAtomList, intAtomList = self.read_iList_information(arg_baseMolecule = arg_baseMolecule, arg_xrdUnpacker = up, arg_interactionEnergyType = iEnerType)
        return
#END

    def read_iList_information(self, arg_baseMolecule, arg_xrdUnpacker, arg_interactionEnergyType):
        """ Equivalent to do_ilist """
        
        fileVersion = self.metadata['file_version']
        up = arg_xrdUnpacker

        if fileVersion < 44:
            self.unpack_n_int(up, TCON.MAXNODES) #idum

        lenAtomList = up.unpack_int() #length of iAtomList
        intAtomList = [up.unpack_int() for i in range(lenAtomList)]

        arg_baseMolecule.groInteractionDict[arg_interactionEnergyType] = (lenAtomList, intAtomList)

        if fileVersion < 78 and arg_interactionEnergyType == TCON.FTYPE_ENUM['F_SETTLE'] and lenAtomList > 0:
            # add_settle_atoms()
            # Utils.printflush('For SETTLE, len atom list (ilist.nr) = ', lenAtomList)
            pass

        return (lenAtomList, intAtomList)
#END

    def read_block_information(self, arg_xrdUnpacker):
        """ Equivalent to do_block (combines do_blocka) """

        fileVersion = self.metadata['file_version']
        up = arg_xrdUnpacker

        # will not pay attention to a whole lot of details here.
        # simply unpack values
        if fileVersion < 44:
            self.unpack_n_int(up, TCON.MAXNODES) #idum

        lenBlock = up.unpack_int()
        if fileVersion < 51:
            dumNra = up.unpack_int() #dum_nra

        self.unpack_n_real(up, lenBlock + 1)

        if fileVersion < 51 and dumNra > 0:
            self.unpack_n_int(up, dumNra)

        # ->-> do_blocka
        nr = up.unpack_int()    # using gromacs variable name
        nra = up.unpack_int()   # using gromacs variable name

        self.unpack_n_int(up, nr + 1)
        self.unpack_n_int(up, nra)

        return
#END

    def read_molblock_information(self, arg_xrdUnpacker, arg_molBlockId):
        """ Equivalent of do_molblock """
        
        fileVersion = self.metadata['file_version']
        up = arg_xrdUnpacker

        molBlockType = up.unpack_int()
        molBlockNumCopies = up.unpack_int()
        molBlockNumAtomsPerCopy = up.unpack_int()

        self.metadata['molblock_types'].append(molBlockType)  #molblock type
        self.metadata['molblock_numMols'].append(molBlockNumCopies)  #molblock number of atoms
        self.metadata['molblock_numAtoms'].append(molBlockNumAtomsPerCopy)  #atoms per molecule

        self.moleculeList[arg_molBlockId].numCopies = molBlockNumCopies
        self.moleculeList[arg_molBlockId].numAtomsPerCopy = molBlockNumAtomsPerCopy
        assert( self.moleculeList[arg_molBlockId].numAtomsPerCopy == self.moleculeList[arg_molBlockId].numAtoms )

        numPosresA = up.unpack_int() #number of posres A

        if numPosresA > 0:
            self.unpack_n_real(up, numPosresA * TCON.VECTORDIM)

        numPosresB = up.unpack_int() #number of posres B
        if numPosresB > 0:
            self.unpack_n_real(up, numPosresB * TCON.VECTORDIM)

        return
#END

    def read_atomtype_information(self, arg_xrdUnpacker):
        """ Equivalent to do_atomtypes """

        fileVersion = self.metadata['file_version']
        up = arg_xrdUnpacker

        # do_atomtypes
        if fileVersion > 25:
            self.metadata['n_atomtypes'] = numAtomTypes = up.unpack_int()

            atomtypeRadii  = [self.unpack_real(up) for i in range(numAtomTypes)]
            atomtypeVolume = [self.unpack_real(up) for i in range(numAtomTypes)]
            atomtypeSurfTens  = [self.unpack_real(up) for i in range(numAtomTypes)]

            if fileVersion >= 40:
                atomNumbers = [up.unpack_int() for i in range(numAtomTypes)]

            if fileVersion >= 60:
                # for GBIS simulations
                atomtypeGBRadii  = [self.unpack_real(up) for i in range(numAtomTypes)]
                atomtypeSHCT  = [self.unpack_real(up) for i in range(numAtomTypes)]

        for idx in range(self.metadata['n_atomtypes']):
            newGroAtomType = GroAtomType(idx)
            self.groAtomTypeDict[idx] = newGroAtomType


        return
#END

    def read_tprFile(self):
        """ Reads the input TPR file and calls subsequent functions to fetch information."""

        # open the tpr file for reading
        with open(self.tprFile, 'rb') as tprFileHandler:

            
            tprBuffer = tprFileHandler.read()

            # unpacker
            up = xdrlib.Unpacker(tprBuffer)

            ################ HEADER ################
            self.get_header_information(arg_xrdUnpacker = up)
            fileVersion = self.metadata['file_version']

            ################ BOX ################
            self.get_box_data(arg_xrdUnpacker = up)
                        
            ################ NGTC ################
            self.get_t_couple_groups(arg_xrdUnpacker = up)

            ################ SYMBOLS ################
            self.get_topology_symbols(arg_xrdUnpacker = up)

            self.get_symstr(up) #symstr

            ################ FF PARAMS ################
            atnr = up.unpack_int() # gromacs and mdanalysis call it atnr
            
            if fileVersion < 57:
                self.metadata['idum'] = up.unpack_int()

            # funciton types
            n_functionTypes = up.unpack_int()
            self.metadata['funcTypes'] = [up.unpack_int() for i in range(n_functionTypes)]

            # reppow (double based on mdanalysis TPRParser)
            if fileVersion >= 66: 
                self.metadata['reppow'] = up.unpack_double() 
            else:
                self.metadata['reppow'] = 12.0

            # fudgeQQ (real)
            self.metadata['fudgeQQ'] = self.unpack_real(up)

            # if self.metadata['precision'] == 4:
            #     self.metadata['fudgeQQ'] = up.unpack_float()
            # else:
            #     self.metadata['fudgeQQ'] = up.unpack_double()

            # Gromacs checks wether all the fucntions defined are compatible with the code.
            # this is for backward compatibility and the numbers are being altered from old
            # to new numbers for funcTypes

            # read parameters for all the function types read from the TPR file
            # (basically skip them because we dont need them now)
            self.get_ff_params(up)
            
            # number of mol type
            self.metadata['n_moltypes'] = up.unpack_int()

            #for each mol type
            # -> do_moltype
            for iMol in range(self.metadata['n_moltypes']):
                # create a base molecule which is linked to a host data container
                # this is one of those places where this code differs from MDanalysis or Gromacs
                # store atom information
                newMolecule = BM.BaseMolecule(arg_name = '', arg_hostDataContainer = self.hostDataContainer)

                # in essence  -> do_moltype
                if fileVersion >= 57:
                    self.get_symstr(up)

                # -> do_atoms
                newMolecule.numAtoms = up.unpack_int()
                # Utils.printflush('TPRREader: molecule with {} atoms found '.format(newMolecule.numAtoms))
                newMolecule.numResidues = up.unpack_int()


                self.metadata['atom_counts'].append(newMolecule.numAtoms)
                self.metadata['residue_counts'].append(newMolecule.numResidues)

                if fileVersion < 57:
                    groupNumber = up.unpack_int()

                    for iGrp in range(TCON.EGC_ENUM):
                        # assign numAtoms to some variable counting groups
                        pass

                # ->-> do_atoms
                self.read_atom_information(arg_baseMolecule = newMolecule, arg_xrdUnpacker = up)

                # ->-> do_resinfo
                self.read_residue_information(arg_baseMolecule = newMolecule, arg_xrdUnpacker = up)

                # ->-> do_ilists
                self.read_iLists_information(arg_baseMolecule = newMolecule, arg_xrdUnpacker = up)

                # ->-> do_block
                self.read_block_information(arg_xrdUnpacker = up)

                # add the molecule to the list
                self.moleculeList.append(newMolecule)

            # number of molblocks
            if fileVersion >= 57:
                self.metadata['n_molblocks'] = up.unpack_int()
                try:
                    assert( self.metadata['n_molblocks'] == len(self.moleculeList) )
                except:
                    raise ValueError(' Number of molecules ({}) is not equal to the number of molecular blocks ({})'.format(len(self.moleculeList), self.metadata['n_molblocks']))

            else:
                self.metadata['n_molblocks'] = 1

            if fileVersion >= 57:
                for iMolBlock in range(self.metadata['n_molblocks']):
                    # ->-> do_molblock
                    self.read_molblock_information(arg_xrdUnpacker = up, arg_molBlockId = iMolBlock)

                assert(self.metadata['n_atoms'] == up.unpack_int())  #natoms

            else:
                Utils.printflush('File Version should be greater than 57 if one is to access molecular block information. (Current Version = {})'.format(fileVersion))

            if fileVersion >= TCON.TPXV_ENUM['tpxv_IntermolecularBondeds']:
                hasInterMolInteractions = up.unpack_bool()
                if hasInterMolInteractions:
                    # dupicate read_iLists_information but do not assign anything to the molecular blocks

                    for iEnerType in range(TCON.FTYPE_ENUM['F_NRE']):
                        bClear = False

                        for ftupdElement in TCON.FTUPD:
                            if fileVersion < ftupdElement[0] and iEnerType == ftupdElement[1]:
                                bClear = True

                        if bClear:
                            lenAtomList = 0
                            intAtomList = None

                        else:
                            # ->->-> do_ilist
                            if fileVersion < 44:
                                self.unpack_n_int(up, TCON.MAXNODES) #idum

                            lenAtomList = up.unpack_int() #length of iAtomList
                            intAtomList = [up.unpack_int() for i in range(lenAtomList)]

                            if fileVersion < 78 and iEnerType == TCON.FTYPE_ENUM['F_SETTLE'] and lenAtomList > 0:
                                # add_settle_atoms()
                                pass
                            

            else:
                hasInterMolInteractions = False

            # gather information about the different atomtypes in the topology
            self.read_atomtype_information(arg_xrdUnpacker = up)

            # for each molecule, map atomtype information to its atoms
            for iMol in self.moleculeList:
                
                for i in range(iMol.numAtoms):
                    
                    iAtomType = iMol.atomGromacsTypeArray[i]

                    groAtomTypeObject = self.groAtomTypeDict[iAtomType]

                # validate that everything is properly assigned
                # iMol.validate_assignments()


        return
#END

    def unpack_n_int(self, arg_xrdUnpacker, arg_n):
        """ Unpack int arg_n times """
        for i in range(arg_n):
            arg_xrdUnpacker.unpack_int()
        return 
#END

    def unpack_n_real(self, arg_xrdUnpacker, arg_n):
        """ Unpack float/double arg_n times depending on precision"""
        if self.metadata['precision'] == 4:
            for i in range(arg_n):
                arg_xrdUnpacker.unpack_float()
        elif self.metadata['precision'] == 8:
            for i in range(arg_n):
                arg_xrdUnpacker.unpack_double()

        return
#END

    def unpack_real(self, arg_xrdUnpacker):
        """Unpack float/double depending on precision"""
        if self.metadata['precision'] == 4:
            return arg_xrdUnpacker.unpack_float()
        elif self.metadata['precision'] == 8:
            return arg_xrdUnpacker.unpack_double()
#END

    def print_tpr_info(self):
        for hKey, hVal in self.metadata.items():
            Utils.printflush('{:<20} : {}'.format(hKey, hVal))
        return
#END

