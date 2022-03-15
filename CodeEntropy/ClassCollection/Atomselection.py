import numpy as nmp
from copy import copy
from collections import deque
from CodeEntropy.ClassCollection import CustomDataTypes as CDT
from CodeEntropy.FunctionCollection import Utils


class Atomselection(object):
    """
    Objects of this class contain information about selected atoms
    in the input `baseMolecule'.
    
    """
    def __init__(self, arg_baseMolecule, arg_selstr, arg_autoparse = True, arg_verbosity = 0):
        """
        TO initialize a selection, a basemolecule must be provided along with
        a `selection string`. Value of `arg_byres` tells if the entire residues 
        of selected atoms should be selected. This is equivalent to ".BYRES." in CHARMM
        and "same residue as" in VMD.
        """
        
        self.baseMolecule = arg_baseMolecule
        self.selstr = arg_selstr
        self.beVerbose = (arg_verbosity > 3)
        
        #number of selected atoms
        self.nsel = -1    
        self.natoms = len(self.baseMolecule.atomIndexArray)
        
        # selection vector
        self.selVector = None
        self.subsetPointer = 0 

        # attributes of the first atom in the selection
        self.selatom = None
        self.selsegn = None
        self.selresi = None
        self.selresn = None
        self.selresd = None
        self.selname = None 
        
        # invoke parsing if automatically asked to
        if arg_autoparse:
            self.parse_selstr()
        
        return
    #END
    
        
    def __str__(self):
        return self.selstr
    #END
    
    def __and__(self, other):
        """
        Custom AND `&' operator for atom selections
        """
        andSelStr = f"({self.selstr} .AND. {other.selstr})"
        try:
            assert(self.selVector == other.selVector)
        except AssertionError:
            raise AssertionError(f"The selection object have different lengths ({self.nsel} and {other.nsel})")

        andSel = Atomselection(self.baseMolecule, andSelStr, arg_autoparse = False)
        andSel.selVector = (self.selVector & other.selVector)
        andSel.update_nsel()
        andSel.update_sel_attr()
        return andSel
    #END
    
    def __or__(self, other):
        """
        Custom OR `|' operator for atom selections
        """
        orSelStr = f"({self.selstr}) .OR. ({other.selstr})"
        try:
            assert(self.selVector == other.selVector)
        except AssertionError:
            raise AssertionError(f"The selection object have different lengths ({self.nsel} and {other.nsel})")

        orSel = Atomselection(self.baseMolecule, orSelStr, arg_autoparse = False)
        orSel.selVector = (self.selVector | other.selVector)
        orSel.update_nsel()
        orSel.update_sel_attr()
        return orSel
    #END
    
    def __invert__(self):
        """
        Custom NOT `~' operator for atom selections
        """
        notSelStr = f".NOT. ({self.selstr})"
        notSel = Atomselection(self.baseMolecule, notSelStr, arg_autoparse = False)
        notSel.selVector = ~self.selVector
        notSel.update_nsel()
        notSel.update_sel_attr()
        return notSel
    #END

    def get_indices(self):
        """Return a list of indices of atoms in this selection."""
        if self.selVector is None:
            Utils.printflush(f'Atom selection is empty')
            return []
        else:
            trueVector = [i for i in range(self.natoms) if self.selVector[i] ]
            return trueVector
    #END

    def get_index(self):
        """Return the index of atom at the subsetPointer if something is found.
           and increment the index by 1
        """
        try:
            rdx = self.get_indices()[self.subsetPointer]
            self.update_sel_attr()
            self.subsetPointer += 1
            return rdx

        except IndexError:
            Utils.printflush(f"Selection atom list exhausted.")
            Utils.printflush(f"Asking for atom at index {self.subsetPointer} when there are only {self.nsel} atoms in the selection.")
            return None
        
    #END  

    def subset(self, index):
        """
        Like CHARMM's .SUBSET. token, it returns a selection of the atom
        at the specified 0-indexed index.
        """
        try:
            rdx = self.get_indices()[index]
            subsetSel = Atomselection(self.baseMolecule, f"BYNU {rdx}")
            return subsetSel

        except IndexError:
            Utils.printflush(f'Requested index larger than the length of atoms in the selection ({self.nsel})')
    
    def update_nsel(self):
        """Update the number of atoms in the selection."""
        self.nsel = nmp.sum(self.selVector)
        return
    #END

    def around(self, other, arg_dist, arg_frame = -1):
        """
        Like CHARMM's .AROUND. token, it returns a selection of atoms within arg_dist
        of this selection.
        """
        return
    #END     

    def update_sel_attr(self):
        """Update the value of selected atom attributes based on the current position of subset pointer"""
        if self.nsel == 0:
            Utils.printflush("Selection is empty.")
            self.selatom = None
            self.selsegn = None
            self.selresi = None
            self.selresn = None
            self.selresd = None
            self.selname = None

        else:
            rdx = self.get_indices()[self.subsetPointer]
            self.selatom = rdx
            self.selsegn = self.baseMolecule.segidArray[self.baseMolecule.atomSegmentIdxArray[rdx]]
            self.selresi = self.baseMolecule.residArray[self.baseMolecule.atomResidueIdxArray[rdx]]
            self.selresn = self.baseMolecule.resnameArray[self.baseMolecule.atomResidueIdxArray[rdx]]
            self.selresd = self.baseMolecule.atomResidueIdxArray[rdx]
            self.selname = self.baseMolecule.atomNameArray[rdx]

        return 
    #END
    
    def update_selvector(self, arg_sdeque, tmpSelVector):
        """By parsing the selection string, create a selection vector and return it."""

        # read the selection's header
        seltokenHeader = arg_sdeque.popleft().upper()
        
        # use the first 4 characters like CHARMM if keyword is NOT "ALL"
        if seltokenHeader != "ALL":
            seltokenHeader = seltokenHeader[0:4]
        
        # check its validity
        try:
            selTokenObj = selHeaderDict[seltokenHeader]
        except:
            raise KeyError(f'Invalid atom selection token {seltokenHeader}')
        
        # gather the number of expected inputs
        ninputs = selTokenObj.nReqProps
        # Utils.printflush(f'Selection token {seltokenHeader} found. Looking for {ninputs} input(s)')

        # read the input property values
        propValueList = []
        for inputIdx in range(ninputs):
            try:
                propName = selTokenObj.propList[inputIdx]
                propValueList.append(arg_sdeque.popleft().upper())
            except:
                raise IndexError(f'Selection token {seltokenHeader} expects {ninputs} inputs. Only {inputIdx} provided.')
        
        
        # go through each approved selection tokens
        # and make the appropriate selection in appropriate order
        if seltokenHeader == "ALL":
            tmpSelVector = nmp.asarray([True for i in range(self.natoms)])
            
        elif seltokenHeader == "NONE":
            tmpSelVector = nmp.asarray([False for i in range(self.natoms)])
            
        elif seltokenHeader == "NAME":
            qName = propValueList[0]
            tmpSelVector = (nmp.array(self.baseMolecule.atomNameArray) == qName)
            
        elif seltokenHeader == "RESD":
            qResd0 = int(propValueList[0])
            tmpSelVector = (nmp.array(self.baseMolecule.atomResidueIdxArray) == qResd0)
            
        elif seltokenHeader == "RESI":
            qRid = int(propValueList[0])
            startAt = 0

            while True:
                try:
                    # find the first presence of query resid from some starting point
                    rid0 = self.baseMolecule.residArray.index(qRid, startAt)
                    if tmpSelVector is None:
                        tmpSelVector = (nmp.array(self.baseMolecule.atomResidueIdxArray) == rid0)
                    else:
                        tmpSelVector = tmpSelVector | (nmp.array(self.baseMolecule.atomResidueIdxArray) == rid0)
                        
                    startAt = rid0 + 1 #update the starting point to one after the last found index
                    
                except ValueError:
                    # nothing more to find (ran out of values)
                    break

        elif seltokenHeader == "RESN":
            qResn = propValueList[0]
            startAt = 0
            
            while True:
                try:
                    # find the first presence of query from some starting point
                    rid0 = self.baseMolecule.resnameArray.index(qResn, startAt)
                    if tmpSelVector is None:
                        tmpSelVector = (nmp.array(self.baseMolecule.atomResidueIdxArray) == rid0)
                    else:
                        tmpSelVector = tmpSelVector | (nmp.array(self.baseMolecule.atomResidueIdxArray) == rid0)
                        
                    startAt = rid0 + 1 #update the starting point to one after the last found index
                    
                except ValueError:
                    # nothing more to find (ran out of values)
                    break

        elif seltokenHeader == "SEGI" or seltokenHeader == "SEGN":
            qSegn = propValueList[0]
            segid0 = self.baseMolecule.segidArray.index(qSegn) #segid's position in segidArray
            tmpSelVector = (nmp.array(self.baseMolecule.atomSegmentIdxArray) == segid0)

        elif seltokenHeader == "BYSE":
            qNumber0 = int(propValueList[0])
            atomSegmentIdx = self.baseMolecule.atomSegmentIdxArray[qNumber0]
            segn = self.baseMolecule.segidArray[atomSegmentIdx]

            tmpSel = Atomselection(self.baseMolecule, f"SEGI {segn}")
            tmpSelVector = tmpSel.selVector

        elif seltokenHeader == "BYRE":
            qNumber0 = int(propValueList[0])
            atomResidueIdx = self.baseMolecule.atomResidueIdxArray[qNumber0]

            tmpSel = Atomselection(self.baseMolecule, f"RESD {atomResidueIdx}")
            tmpSelVector = tmpSel.selVector

                    
        elif seltokenHeader == "BYNU":
            qNumber0 = int(propValueList[0])
            tmpSelVector = (nmp.array(self.baseMolecule.atomIndexArray) == qNumber0)
                
        elif seltokenHeader == "HYDR":
            tmpSelVector = nmp.asarray(self.baseMolecule.isHydrogenArray, dtype = bool)
            
        elif seltokenHeader == "BOND":
            # output depends on whether supplied atomID0 is hydrogen or not
            qNumber0 = int(propValueList[0])
            bondedIdx = []
            
            # scan bondedHeavyAtom table for any atom
            for iPriority, idx in self.baseMolecule.bondedHeavyAtomTable[qNumber0].list_in_order():
                    bondedIdx.append(idx)
            
            # if not hydrogen, then scan the bondedHydrogen table
            if not self.baseMolecule.isHydrogenArray[qNumber0]:
                for iPriority, idx in self.baseMolecule.bondedHydrogenTable[qNumber0].list_in_order():
                    bondedIdx.append(idx)
            
            # initialize with all FALSE values
            tmpSelVector = nmp.asarray([False for i in range(self.natoms)])
            
            # update values at indices that bond
            for bdx in bondedIdx:
                tmpSelVector[bdx] = True

        elif seltokenHeader == "BONH":
            # output depends on whether supplied atomID0 is hydrogen or not
            # heavy atom ? it and its bonded hydrogen atoms
            # hydrogen atom ? then just the hydrogen

            qNumber0 = int(propValueList[0])
            bondedIdx = [qNumber0]
            
            # if not hydrogen, then scan the bondedHydrogen table
            if not self.baseMolecule.isHydrogenArray[qNumber0]:
                for iPriority, idx in self.baseMolecule.bondedHydrogenTable[qNumber0].list_in_order():
                    bondedIdx.append(idx)
            
            # initialize with all FALSE values
            tmpSelVector = nmp.asarray([False for i in range(self.natoms)])
            
            # update values at indices that bond
            for bdx in bondedIdx:
                tmpSelVector[bdx] = True

        else:
            Utils.printflush(f"Code for token = {seltokenHeader} is not yet written.")
            return
            
        return arg_sdeque, tmpSelVector
    #END


            
        
    def parse_selstr(self):
        """
        Parses the selection string and updates its selection vector which can be
        accessed by various member functions.
        """
        selstr = copy(self.selstr)
        selstr = selstr.strip()        
        strAsDeque = deque(selstr.split())
        selVector = None
        
        strAsDeque, selVector = self.update_selvector(strAsDeque, selVector)
        
        # finally
        self.selVector = selVector
        self.update_nsel()
        self.update_sel_attr()

        if self.beVerbose:
            Utils.printflush(f"{self.selstr} : {self.nsel} atoms of {self.natoms} selected.")

        return 
    #END

#END

class Seltoken(object):
    """
    Objects of this class contain information about different
    selection kinds and the properties values they should be provided
    in order to look for atoms with those properties.
    """
    def __init__(self, arg_header, arg_propList):
        self.tokenHeader = arg_header.upper()
        self.propList = arg_propList
        self.nReqProps = len(self.propList)

#END


"""
Dictionary mapping selection tokens encountered in a 
selection string with the corresponding object of Class `Seltoken'.
<STRING>:<Seltoken>
"""
selHeaderDict = dict()

# DEFINE ACCEPTED SELECTION TOKEN HEADERS
selHeaderDict["ALL"] = Seltoken("ALL","")
selHeaderDict["NONE"] = Seltoken("NONE","")
selHeaderDict["HYDR"] = Seltoken("HYDR","")   # all hydrogen atoms
selHeaderDict["NAME"] = Seltoken("NAME",["ATOMNAME"])   #NAME <ATOM NAME>
selHeaderDict["RESN"] = Seltoken("RESN",["RESNAME"])   #RESN <RESNAME>
selHeaderDict["RESI"] = Seltoken("RESI",["RESID"])   #RESI <RESID>
selHeaderDict["RESD"] = Seltoken("RESD",["RESIDUEIDX0"]) #RESD <0-indexed residue index>
selHeaderDict["BYNU"] = Seltoken("BYNU",["ATOMID0"])  #BYNUmber <0-indexed atom id>
selHeaderDict["BOND"] = Seltoken("BOND",["ATOMID0"])  #BONDed_to <0-indexed atom id>
selHeaderDict["BONH"] = Seltoken("BONH",["ATOMID0"])  #BONHydrogen <0-indexed atom id>
selHeaderDict["SEGN"] = Seltoken("SEGN",["SEGNAME"])  #SEGName <SEGNAME>
selHeaderDict["SEGI"] = Seltoken("SEGI",["SEGNAME"])  #alternative to SEGN to make it easy.
selHeaderDict["BYSE"] = Seltoken("BYSE",["ATOMID0"])  # BYSEGID <0-indexed atom id>
selHeaderDict["BYRE"] = Seltoken("BYRE",["ATOMID0"])  # BYRESID <0-indexed atom id> 



