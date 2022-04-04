import numpy as nmp
import re
import sys
from copy import deepcopy


from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.FunctionCollection import GeometricFunctions
from CodeEntropy.FunctionCollection import CustomFunctions
from CodeEntropy.ClassCollection import CustomDataTypes as CDT
from CodeEntropy.ClassCollection import BondStructs
from CodeEntropy.ClassCollection import Atomselection

class BaseMolecule(object):
    """ A class that contains topographical and composition information of a molecule"""

    def __init__(self, arg_name, arg_hostDataContainer):
        self.name = arg_name
        self.numAtoms = 0
        self.numResidues = 0
        self.numSegments = 0
        self.numCopies = 0
        self.numAtomsPerCopy = 0

        self.residArray = []  #an array of residue indices as they appear in the input file 
        self.resnameArray = [] #an array of resnames "???" in the order of their resids as they appear in the input file
        
        self.segidArray = [] # an array of segment ids/names in the order they appear in input file

        self.atomResidueIdxArray =  [] # an array of residue indices in which they are read (0-indexed) of the length of the number of atoms
        self.atomSegmentIdxArray = [] # an array of segment indices in which they are read (0-indexed) of the length of the number of atoms

        self.atomNameArray = [] # an array of the name of atoms as they appear in the input file
        self.atomIndexArray = [] # an array of atom IDs as they appear in the input file
        self.atomMassArray = [] # an array of atom masses as they appear in the input 
        self.atomChargeArray = [] # an array of atom charges as they appear in the input
        
       

        # this is to help with the obtaining bonding information
        # the heads of a cell-linked-list structure
        # each entry is the last atom index as read from the input file that belongs to the residue on that index
        self.residueHead = None

        # the atoms array contains information about which residue the atoms belong to.
        # the idea is the same as cell-linked list
        self.atomArray = None

        # arrays storing element type information for atoms for easier access during runtime
        # length = totalNumAtom = numCopies * numAtomsPerCopy
        self.isCAtomArray = []
        self.isOAtomArray = []
        self.isNAtomArray = []
        self.isCaAtomArray = []
        self.isBBAtomArray = []
        self.isHydrogenArray = []

        # this is especially for united atom (UA) level of hierarchy
        # each atom is either a heavy atom or an atom bonded to a heavy atom
        # so each atom has an index associated with it which is the index
        # of the heavy atom it is bonded to (including itself)
        
        # this is an array that containes the information about 
        # which heavy atoms this certain atom is bonded to. (including itself if it is a heavy atom)
        self.hostHeavyAtomArray = []

        # this is a dictionary  <atom id : binary tree> that containes the information about 
        # which hydrogens atoms this certain atom is bonded to.
        # Each node contains a value which is a 2-tuple of the form (priority, atom_index)
        # The design of the tree is sorted by priority levels
        self.bondedHydrogenTable = dict()


        # this is a dictionary  <atom id : binary tree> that containes the information about 
        # which heavy atoms this certain atom is bonded to.
        # Each node contains a value which is a 2-tuple of the form (priority, atom_index)
        # The design of the tree is sorted by priority levels
        self.bondedHeavyAtomTable = dict()

        # bondArray
        # bonds (covalent) in the structure
        # array of instances of class Bond
        self.bondArray = []

        # dihedralArray
        # set of objects of Class Dihedral
		# this is to avoid storing dihedral information redundantly. Packages like gromacs have a single dihedral 
		# treated with more than one function and so the way we are reading the topology, we might overcount dihedrals.
        self.dihedralArray = set()

        # this is a dictionary  <atom id : binary tree> that containes the information about 
        # which  dihedral bonds an atom is associated with.
        # The key is atom index and its value is a binary tree where
        # each node is an instance of Class Dihedral.
        # It could very well have been a list of 'Dihedrals' but the idea was to introduce 
        # the use of 'Disjoint sets' to be able to find common dihedrals.
        # hopefully this will be time-effcient and importantnyl, elegent
        self.dihedralTable = dict()

        
        # residueDict
        # it is a compound dictionary
        # Keys of the main dictionary are interger (0-numResidues)
        # Key, value pair of the secondary dictionary are 
        #   Residue name : resid
        #   atom id      : atom name (notice the order is reversed now)
        self.residueDict = dict()

        # # atomOfResidueDict:
        # # it is a compound dictionary
        # # A dict <int (0-numResidues) :  dict(<str (atomname) : int (atom-index)>) >
        # self.atomsOfResidueDict = dict()

        # a dictionary of < int (0-numAtoms): Atom> format
        self.atomsDict = dict()

        # the host data container
        # it will use the vectorial information in it
        self.hostDataContainer = arg_hostDataContainer

        # an array of atom priority levels which is going to be used for
        # heap like addition of bond information
        self.atomPriorityLevelArray = None

        ################################################################
        #              G R O M A C S  
        ################################################################
        # a dictionary of interaction list (might just be unique to gromacs)
        # <INTERACTION_ENERGY_TYPE : ( numAtoms_in_list , [list of atoms interacting this way])>
        # not important now (Aug 2018) but might be for MM part 
        self.groInteractionDict = dict()

        self.atomGromacsTypeArray = [] # an array of atom types (uint based on gromacs' way of defining atomtypes) as they appear in the input
        #  |
        # \ /
        #  V
        self.atomRadiusArray = [] # an array of atom radii as they appear in the input
        self.atomSurftensArray = [] # an array of atoms surface tensions as they appear in the input
        

# END    
        
    def validate_initialization(self):
        """ Validate the entries by checking the identity of the lengths of the different arrays"""
        assert(len(self.residueIndexArray) == len(self.residArray))
        assert(len(self.residArray)        == len(self.resnameArray))

        assert(len(self.atomNameArray)     == len(self.atomIndexArray))
        assert(len(self.atomsDict)         == len(self.atomIndexArray))

        assert(len(self.bondedHydrogenTable) == len(self.atomIndexArray))
        assert(len(self.bondedHeavyAtomTable) == len(self.atomIndexArray))

        for iAtom in self.atomIndexArray:
            # that trees have no values
            assert(len(self.bondedHeavyAtomTable[iAtom]) == 0)
            assert(len(self.bondedHydrogenTable[iAtom]) == 0)

        return
# END
        
    def validate_assignments(self):
        assert(self.numResidues == len(self.residueHead))
        assert(self.numResidues == len(self.residArray))
        assert(self.numResidues == len(self.resnameArray))
        assert(self.numSegments == len(self.segidArray))

        assert(self.numAtoms == len(self.atomResidueIdxArray))
        assert(self.numAtoms == len(self.atomSegmentIdxArray))
        
        assert(self.numAtoms == len(self.atomArray))
        assert(self.numAtoms == len(self.atomNameArray))
        assert(self.numAtoms == len(self.atomIndexArray))
        assert(self.numAtoms == len(self.atomMassArray))
        
        return
# END

    def initialize_element_info_arrays(self):
        """ initialize arrays storing information about atom types and priority levels"""
        try:
            assert(self.numCopies * self.numAtomsPerCopy != 0)
        except:
            Utils.printflush('Arrays cannot be initialized if the molecule does not have any atom')
            return

        self.isCAtomArray = nmp.zeros(self.numCopies * self.numAtomsPerCopy)
        self.isOAtomArray = nmp.zeros(self.numCopies * self.numAtomsPerCopy)
        self.isNAtomArray = nmp.zeros(self.numCopies * self.numAtomsPerCopy)
        self.isCaAtomArray = nmp.zeros(self.numCopies * self.numAtomsPerCopy)
        self.isBBAtomArray = nmp.zeros(self.numCopies * self.numAtomsPerCopy)
        self.isHydrogenArray = nmp.zeros(self.numCopies * self.numAtomsPerCopy)
        self.atomPriorityLevelArray = -1 * nmp.ones(self.numCopies * self.numAtomsPerCopy) #initialize with -1

        return
#END

    def populate_element_info_arrays(self):
        """ After initialization, element info arrays are filled with appropriate values """
        # assign element types and priorities                                

        for atIdx in range(self.numCopies * self.numAtomsPerCopy):
            
            # set priority = -1 (meaningless value for now) for all atoms
            # We need priority level per atom to be able to work with
            # data types created in the first version of the code
            self.atomPriorityLevelArray[atIdx] = -1
            
                    
            # initialize bond information trees for each atom
            self.bondedHydrogenTable[atIdx] = CDT.BinaryTree()
            self.bondedHeavyAtomTable[atIdx] = CDT.BinaryTree()
                
            # initialize dihedral list for each atom
            self.dihedralTable[atIdx] = set()

            # assign atomtype using atomname
            atName = self.atomNameArray[atIdx % (self.numAtoms) ]
            
            if re.match("^C$",atName):
                self.isCAtomArray[atIdx] = 1
                self.isBBAtomArray[atIdx] = 1
                
            elif re.match("^N$",atName):
                self.isNAtomArray[atIdx] = 1
                self.isBBAtomArray[atIdx] = 1
                
            elif re.match("^CA$",atName):
                self.isCaAtomArray[atIdx] = 1
                self.isBBAtomArray[atIdx] = 1
                
            elif re.match("^O$",atName):
                self.isOAtomArray[atIdx] = 1
                self.isBBAtomArray[atIdx] = 1

            elif re.match("H[A-Z]{0,}[0-9]{0,}",atName):
                self.isHydrogenArray[atIdx] = 1

        # cast into integers
        self.atomArray = self.atomArray.astype(int)
        self.isCAtomArray = self.isCAtomArray.astype(int)
        self.isNAtomArray = self.isNAtomArray.astype(int)
        self.isCaAtomArray = self.isCaAtomArray.astype(int)
        self.isBBAtomArray = self.isBBAtomArray.astype(int)
        self.isHydrogenArray = self.isHydrogenArray.astype(int)

        return
    #END

    def add_bond_tree(self, arg_aidI, arg_priorityLevelI, arg_aidJ, arg_priorityLevelJ):

        # REPLACEMENT FOR HEAP LIKE LIST

        # add the input atoms to each others' list which is essentially a BINARY TREE
        # that way, atoms with a higher priority (low rank or closeness to BB) appear first

        arg_aidI = int(arg_aidI)
        arg_aidJ = int(arg_aidJ)

        # Utils.printflush("Adding {} and {}".format(arg_aidI, arg_aidJ))


        # make sure its a valid atom
        #try:
            #assert(arg_aidI >= min(self.atomIndexArray) and arg_aidI <= max(self.atomIndexArray))
            #assert(arg_aidJ >= min(self.atomIndexArray) and arg_aidJ <= max(self.atomIndexArray))
        #except:
            #Utils.printflush('Bad indices : {} {}'.format(arg_aidI, arg_aidJ))
            #return
        

        # make sure both atoms are not hydrogen atoms
        try:
            assert(not (self.isHydrogenArray[arg_aidI] and self.isHydrogenArray[arg_aidJ]))
        except:
            Utils.printflush('!!! WARNING : Requesting bond between two hydrogen atoms : ({:>3d}) and ({:>3d}) !!!'.format(arg_aidI, arg_aidJ))
            return
        
        # expect a pair to be noH-H or noH-noH 
        iH = -1
        iNonH1 = -1
        iNonH2 = -1
        priorityNonH1, priorityNonH2, priorityH = (-1, -1, -1)


        if self.isHydrogenArray[arg_aidI]:
            iH = arg_aidI
            priorityH = -1

            iNonH1 = arg_aidJ
            priorityNonH1 = arg_priorityLevelI

        elif self.isHydrogenArray[arg_aidJ]:
            iH = arg_aidJ
            priorityH = -1

            iNonH1 = arg_aidI
            priorityNonH1 = arg_priorityLevelI
        else:
            iNonH1 = arg_aidI
            priorityNonH1 = arg_priorityLevelI

            iNonH2 = arg_aidJ
            priorityNonH2 = arg_priorityLevelJ

        # Utils.printflush(iNonH1, iNonH2, iH)

        if iNonH1 != -1 and iNonH2 != -1:
            # Utils.printflush("Adding two heavy atom into a bond : {} {}".format(iNonH1, iNonH2))
            node1 = CDT.TreeNode(arg_leftChildNode = None, arg_rightChildNode = None, arg_value = (priorityNonH1, iNonH1) )
            node2 = CDT.TreeNode(arg_leftChildNode = None, arg_rightChildNode = None, arg_value = (priorityNonH2, iNonH2) )
            
            # add tuple-2 to the BT corresponding to 1
            self.bondedHeavyAtomTable[iNonH1].add_node(arg_node = node2)
            # Utils.printflush('New length of bondedHeavyAtomTable[{}] is {}'.format(iNonH1, len(self.bondedHeavyAtomTable[iNonH1])))

            # push 2 to the heap corresponding to 1
            self.bondedHeavyAtomTable[iNonH2].add_node(arg_node = node1)

        
        else:
            # Utils.printflush("Adding a hydrogen {} to heavy atom {} in a bond.".format(iH, iNonH1))
            node1 = CDT.TreeNode(arg_leftChildNode = None, arg_rightChildNode = None, arg_value = (priorityNonH1, iNonH1) )
            nodeH = CDT.TreeNode(arg_leftChildNode = None, arg_rightChildNode = None, arg_value = (priorityH, iH) )

            # add the hydrogen to the heavy atom
            self.bondedHydrogenTable[iNonH1].add_node(arg_node = nodeH)

            # add the heavy atom to the hydrogen's heavy atom bonding partners
            self.bondedHeavyAtomTable[iH].add_node(arg_node = node1)
    
        return
    # END

    def add_dihedral(self, arg_dihedral):
        # add the input dihedral to the value list associated with the indices of the atoms that form it.
        for atIdx in arg_dihedral.atomList:
            # Utils.printflush('Adding dihedral {} to the list for atom {}'.format(arg_dihedral, atIdx))
            self.dihedralTable[atIdx].add(arg_dihedral)
        
        self.dihedralArray.add(arg_dihedral)

    #END


    def get_center_of_mass(self, arg_atomList, arg_frame):
        """ compute and return the center of mass of the input atoms in the lab frame"""
        com = nmp.zeros((3))

        totalMass = nmp.sum(nmp.asarray([self.atomMassArray[iAtom] for iAtom in arg_atomList]))

        for iAtom in arg_atomList:
            iMassCoord = self.atomMassArray[iAtom] * self.hostDataContainer._labCoords[arg_frame][iAtom]
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
            iLabCoord = self.hostDataContainer._labCoords[arg_frame][iAtom]

            # deduct com
            atomLocalCoords[atIdx] = iLabCoord - atomsCOM

            atomMasses[atIdx] = self.atomMassArray[iAtom]
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

    # END

    def get_basis_protein_residueBB(self, arg_residueIndex,  arg_frame):
        """ For a protein residue, generate a C-Ca-N basis for a given frame """

        # fetch the indices of its C, Ca and N atoms
        # inititalize ridicuslously
        cIdx, caIdx, nIdx = -1, -1, -1

        #1. go to its residue head (index of its last atom)
        rAtom = int(self.residueHead[arg_residueIndex])
        while rAtom != -1:
            if self.isCAtomArray[rAtom]:
                cIdx = rAtom
            elif self.isNAtomArray[rAtom]:
                nIdx = rAtom
            elif self.isCaAtomArray[rAtom]:
                caIdx = rAtom
            rAtom = int(self.atomArray[rAtom])

        # basic checks (can remove later)
        try:
            assert(self.atomNameArray[cIdx] == "C")
        except:
            raise ValueError('Index {} deemed to be BB C is a pretend because it is a {}!'.format(cIdx, self.atomNameArray[cIdx]))
        assert(self.atomNameArray[nIdx] == "N")
        assert(self.atomNameArray[caIdx] == "CA")


        cPosition = self.hostDataContainer._labCoords[arg_frame,cIdx]
        nPosition = self.hostDataContainer._labCoords[arg_frame,nIdx]
        caPosition = self.hostDataContainer._labCoords[arg_frame,caIdx]

        return GeometricFunctions.generate_orthonormal_axes_system(arg_coord1 = cPosition, arg_coord2 = nPosition, arg_coord3 = caPosition)

    # END

    def print_residue_info(self, arg_resindex):
        """ print residue information in the prescribed format"""
        
        Utils.printflush("{:>6s}{:>8s}{:>4s}{:>4s}{:>4s}{:>4s}{:>4s}{:>10s}".format("Idx","Name","isC", "isN", "isCa","isBB","isH", "bondedTo"))
        Utils.printflush('-'*50)
        iAtom = int(self.residueHead[arg_resindex])
        while iAtom != -1:
            Utils.printflush("{:>6}{:>8}{:>4}{:>4}{:>4}{:>4}{:>4}{:>10}".format(iAtom, self.atomNameArray[iAtom], self.isCAtomArray[iAtom], self.isNAtomArray[iAtom], self.isCaAtomArray[iAtom], self.isBBAtomArray[iAtom], self.isHydrogenArray[iAtom],self.hostHeavyAtomArray[iAtom]))
            iAtom = self.atomArray[iAtom]
        
        return
    #END

    def print_molecule_info(self):
        """ Print the information of its constituent atoms in PDB style without the coordinates """

        Utils.printflush(' Name' , self.name)
        Utils.printflush(' numatoms' , self.numAtoms)
        Utils.printflush(' numRes' , self.numResidues)
        Utils.printflush(' numCopies' , self.numCopies)
        Utils.printflush(' numAtomsPercopy', self.numAtomsPerCopy)

        Utils.printflush(' resid array', self.residArray) 
        Utils.printflush(' resname array' ,self.resnameArray)

        
        for aIdx in range(self.numAtoms):
            rIdx = self.atomResidueIdxArray[aIdx]
            Utils.printflush('{:<6s}{:>5d}{:>4s} {:3s} {:>4d} {:>8.4f}'.format('ATOM',self.atomIndexArray[aIdx], self.atomNameArray[aIdx], self.resnameArray[rIdx], self.residArray[rIdx], self.atomMassArray[aIdx]))
        return
    #END

    def get_atoms_in_resid(self, arg_resid):
        """ Return the 0-indexed list of atoms in the residue with the input resid. """
        atomList = []
        iAtom = int(self.residueHead[arg_resid])
        while iAtom != -1:
            atomList.append(iAtom)
            iAtom = int(self.atomArray[iAtom])

        return atomList
    #END

    def is_hydrogen(self, arg_idx):
        """Return True if the atom with input index is a hydrogen atom."""
        return (self.isHydrogenArray[arg_idx] == 1)

    #END




  



