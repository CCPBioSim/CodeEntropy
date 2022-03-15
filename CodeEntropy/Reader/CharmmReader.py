import numpy as nmp
from CodeEntropy.Trajectory import (PSFReader, DCDReader)
from CodeEntropy.ClassCollection import DataContainer
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.FunctionCollection import UnitsAndConversions as UAC

########################################################################################################################################
# NOTE: SIGNIFICANT CHANGES IN THE FLOW OF STEPS AND VARIABLE ASSIGNMENT BECAUSE OF THE FUNDAMENTAL DIFFERENCE BETWEEN NAMD AND GROMACS
########################################################################################################################################
def read_charmm_input(arg_psfFile,\
                    arg_xdcdFile, \
                    arg_fdcdFile, \
                    arg_outFile, \
                    arg_beginTime, \
                    arg_endTime, \
                    arg_stride, \
                    arg_verbose = 0 ):
    
    """ 
    Input :
    1. input topology/structure (PSF) 
    2. a coordinate trajectory file from CHARMM (arg_xdcdFile).
    3. a force trajectory file from CHARMM (arg_fdcdFile).

    Separate DCD files are required because unlike GROMACS, CHARMM does
    not print forces in a trajectory. A charmm script to write a FORCE
    trajectory file must be obtained.
    Function returns an instance of BaseMolecule and 
    DataContainer which are linked to each other.
    """


    
    # topology/PSF file
    Utils.printflush('{:<40s} : {}'.format('Reading PSF format topology file',arg_psfFile))
    topol = PSFReader.PsfTopology(arg_psfFile, arg_verbose)

    # base molecule
    myProtein = topol.molecule

    if arg_verbose >= 0:
        Utils.printflush('{:<40s} : {}'.format('Number of atoms expected in trajectory', myProtein.numAtoms))


    # assign host heavy atoms for each hydrogen (needed for entropy calculation at UA level)
    myProtein.hostHeavyAtomArray = nmp.zeros(myProtein.numAtoms)
    for idx in myProtein.bondedHeavyAtomTable.keys():

        if myProtein.isHydrogenArray[idx]:
            # it must have ONE and ONLY ONE heavy atom bonded to it (it host heavy atom)
            # NEW: SHAKE may impose >1 covalent bond on hydrogens. Allow exception. 
            try:
                assert(len(myProtein.bondedHeavyAtomTable[idx]) == 1)
            except AssertionError:
                Utils.printflush(f"Hydrogen at index {idx} has more than one covalent bond. Make sure it is because of SHAKE.")
            
            # fetch that atom
            try:
                for priority, heavyAtomIdx in myProtein.bondedHeavyAtomTable[idx].list_in_order():
                    if not myProtein.isHydrogenArray[heavyAtomIdx]:
                        myProtein.hostHeavyAtomArray[idx] = heavyAtomIdx
                        myProtein.hostHeavyAtomArray[heavyAtomIdx] = heavyAtomIdx #self
            except IndexError:
                raise IndexError('Index {} is out of range'.format(idx))

    # define an object of DataContainer class
    mainContainer = DataContainer.DataContainer()
    
    # >>> ASSIGN MYPROTEIN TO THE MAIN DATA CONTAINER
    mainContainer.molecule = myProtein


    ### READ DCD TRAJECTORY OF COORDINATES ###
    Utils.printflush('{:<40s} : {}'.format('Reading coordinate trajectory file',arg_xdcdFile))
    crdtraj = DCDReader.DCDTrajectory(arg_fileName=arg_xdcdFile, arg_dataContainer = mainContainer, arg_beginTime=arg_beginTime, arg_endTime=arg_endTime,arg_stride=arg_stride, arg_verbose=arg_verbose)

    ### READ DCD TRAJECTORY OF FORCES ###
    forceContainer = DataContainer.DataContainer()
    Utils.printflush('{:<40s} : {}'.format('Reading forces trajectory file',arg_fdcdFile))
    frctraj = DCDReader.DCDTrajectory(arg_fileName=arg_fdcdFile, arg_dataContainer = forceContainer, arg_beginTime=arg_beginTime, arg_endTime=arg_endTime,arg_stride=arg_stride, arg_verbose=arg_verbose)

    ### ASSERTION/ VALIDATION ###
    # both trajectories should have the same nu,ber of frames
    try:
        assert(len(mainContainer.trajSnapshots) == len(forceContainer.trajSnapshots))
    except:
        raise ValueError("Coordinate and Force trajectory are of different lengths ({} vs {}).".\
            format(len(mainContainer.snapshots), len(forceContainer.snapshots)))
    

    mainContainer.print_attributes()

    # read the coords and forces from the trajectory
    # and store them in the mainContainer
    mainContainer.initialize_ndarrays()

    coordFactor = 1.0
    forceFactor = 1.0 * UAC.KCAL2KJ

    for i, frame in enumerate(mainContainer.trajSnapshots):
        mainContainer._labCoords[i] = coordFactor * frame.value['coordinates']

    for i,frame in enumerate(forceContainer.trajSnapshots):
        mainContainer._labForces[i] = forceFactor * frame.value['coordinates']

    return (myProtein, mainContainer)

#END


