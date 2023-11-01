import numpy as nmp
import MDAnalysis as mds
from CodeEntropy.ClassCollection import BeadClasses as BC
from CodeEntropy.ClassCollection import ConformationEntity as CONF
from CodeEntropy.ClassCollection import ModeClasses
from CodeEntropy.ClassCollection import CustomDataTypes
from CodeEntropy.FunctionCollection import EntropyFunctions as EF
from CodeEntropy.FunctionCollection import CustomFunctions as CF
from CodeEntropy.FunctionCollection import GeometricFunctions as GF
from CodeEntropy.FunctionCollection import UnitsAndConversions as UAC
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.IO import Writer
from CodeEntropy.FunctionCollection import UnitsAndConversions as CONST

def select_levels(arg_hostDataContainer):
    """
    Function to read input system and identify the number of molecules and the levels (i.e. united atom, residue and or whole molecule) that should be used
    """

    # fragments is MDAnalysis terminology for what chemists would call molecules
    number_molecules = len(arg_hostDataContainer.atoms.fragments)
    fragments = arg_hostDataContainer.atoms.fragments
    levels = []

    for i in range(number_molecules):
        levels[i].append("united_atom") # every molecule has at least one atom

        atoms_in_fragment = fragments[i].select_atoms("record type ATOM and not H*")
        number_residues = len(atoms_in_fragment.residues)

        # if a fragment has more than one atom assign residue level
        if len(atoms_in_fragment > 1):
            levels[i].append("residue")

            #if assigned residue level and there is more than one residue assign whole molecule level
            if number_residues > 1:
                levels[i].append("whole_molecule")

    return levels        
