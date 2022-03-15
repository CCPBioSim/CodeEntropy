""" All the constants relevant to this module is here """

# function to define a cpp like ENUM
def define_enum(arg_enumKeys, arg_enumFirstValue = 0):
    """ For a list of keys, create an cpp-like enum in the form of a dictionary"""
    enumDict = {}
    for idx, tKey in enumerate(arg_enumKeys):
        enumDict[tKey] = idx + arg_enumFirstValue

    return enumDict

# CONSTANTS
VECTORDIM = 3
TPX_TAG_RELEASE  = "release"
NR_CBTDIHS = 6    #(look for idef.h)
NR_RBDIHS = 6     #(look for idef.h)
NR_FOURDIHS = 4     #(look for idef.h)
NOTSET = -12345   #(look for typedefs.h)
MAXNODES = 256    #(look for tpxio.c)

# define EGC "enum"
egcEnumKey = ['egcTC',    'egcENER',   'egcACC', 'egcFREEZE',  'egcUser1', 
'egcUser2',  'egcVCM', 'egcCompressedX', 'egcORFIT', 'egcQMMM', 'egcNR']
EGC_ENUM = define_enum(arg_enumKeys = egcEnumKey)

# define a TPXV "enum"
tpxvEnumKey = [
'tpxv_ComputationalElectrophysiology',
'tpxv_Use64BitRandomSeed',
'tpxv_RestrictedBendingAndCombinedAngleTorsionPotentials',
'tpxv_InteractiveMolecularDynamics',
'tpxv_RemoveObsoleteParameters1',
'tpxv_PullCoordTypeGeom'
'tpxv_PullGeomDirRel',
'tpxv_IntermolecularBondeds']
tpxvEnumFirstValue = 96
TPXV_ENUM = define_enum(arg_enumKeys = tpxvEnumKey, arg_enumFirstValue = tpxvEnumFirstValue )

# define function type enum
fTypeKeys = ['F_BONDS', 'F_G96BONDS', 'F_MORSE', 'F_CUBICBONDS', 'F_CONNBONDS', 
'F_HARMONIC', 'F_FENEBONDS', 'F_TABBONDS', 'F_TABBONDSNC', 'F_RESTRBONDS', 
'F_ANGLES', 'F_G96ANGLES', 'F_RESTRANGLES', 'F_LINEAR_ANGLES', 'F_CROSS_BOND_BONDS', 
'F_CROSS_BOND_ANGLES', 'F_UREY_BRADLEY', 'F_QUARTIC_ANGLES', 'F_TABANGLES', 'F_PDIHS', 
'F_RBDIHS', 'F_RESTRDIHS', 'F_CBTDIHS', 'F_FOURDIHS', 'F_IDIHS', 
'F_PIDIHS', 'F_TABDIHS', 'F_CMAP', 'F_GB12', 'F_GB13', 
'F_GB14', 'F_GBPOL', 'F_NPSOLVATION', 'F_LJ14', 'F_COUL14', 
'F_LJC14_Q', 'F_LJC_PAIRS_NB', 'F_LJ', 'F_BHAM', 'F_LJ_LR', 
'F_BHAM_LR', 'F_DISPCORR', 'F_COUL_SR', 'F_COUL_LR', 'F_RF_EXCL', 
'F_COUL_RECIP', 'F_LJ_RECIP', 'F_DPD', 'F_POLARIZATION', 'F_WATER_POL', 
'F_THOLE_POL', 'F_ANHARM_POL', 'F_POSRES', 'F_FBPOSRES', 'F_DISRES', 
'F_DISRESVIOL', 'F_ORIRES', 'F_ORIRESDEV', 'F_ANGRES', 'F_ANGRESZ', 
'F_DIHRES', 'F_DIHRESVIOL', 'F_CONSTR', 'F_CONSTRNC', 'F_SETTLE', 
'F_VSITE2', 'F_VSITE3', 'F_VSITE3FD', 'F_VSITE3FAD', 'F_VSITE3OUT', 
'F_VSITE4FD', 'F_VSITE4FDN', 'F_VSITEN', 'F_COM_PULL', 'F_EQM', 
'F_EPOT', 'F_EKIN', 'F_ETOT', 'F_ECONSERVED', 'F_TEMP', 
'F_VTEMP_NOLONGERUSED', 'F_PDISPCORR', 'F_PRES', 'F_DVDL_CONSTR', 'F_DVDL', 
'F_DKDL', 'F_DVDL_COUL', 'F_DVDL_VDW', 'F_DVDL_BONDED', 'F_DVDL_RESTRAINT', 
'F_DVDL_TEMPERATURE', 'F_NRE']
FTYPE_ENUM = define_enum(arg_enumKeys = fTypeKeys)

# define FTUPD list
# '<gromacs-dir>/src/gromacs/fileio/tpxio.c'
# i dont know the origin of this name but I will keep it 
# so that the reader knows where this was obtained from
FTUPD = [
    ( 20, 'F_CUBICBONDS'        ),
    ( 20, 'F_CONNBONDS'         ),
    ( 20, 'F_HARMONIC'          ),
    ( 34, 'F_FENEBONDS'         ),
    ( 43, 'F_TABBONDS'          ),
    ( 43, 'F_TABBONDSNC'        ),
    ( 70, 'F_RESTRBONDS'        ),
    ( TPXV_ENUM['tpxv_RestrictedBendingAndCombinedAngleTorsionPotentials'], 'F_RESTRANGLES' ),
    ( 76, 'F_LINEAR_ANGLES'     ),
    ( 30, 'F_CROSS_BOND_BONDS'  ),
    ( 30, 'F_CROSS_BOND_ANGLES' ),
    ( 30, 'F_UREY_BRADLEY'      ),
    ( 34, 'F_QUARTIC_ANGLES'    ),
    ( 43, 'F_TABANGLES'         ),
    ( TPXV_ENUM['tpxv_RestrictedBendingAndCombinedAngleTorsionPotentials'], 'F_RESTRDIHS' ),
    ( TPXV_ENUM['tpxv_RestrictedBendingAndCombinedAngleTorsionPotentials'], 'F_CBTDIHS' ),
    ( 26, 'F_FOURDIHS'          ),
    ( 26, 'F_PIDIHS'            ),
    ( 43, 'F_TABDIHS'           ),
    ( 65, 'F_CMAP'              ),
    ( 60, 'F_GB12'              ),
    ( 61, 'F_GB13'              ),
    ( 61, 'F_GB14'              ),
    ( 72, 'F_GBPOL'             ),
    ( 72, 'F_NPSOLVATION'       ),
    ( 41, 'F_LJC14_Q'           ),
    ( 41, 'F_LJC_PAIRS_NB'      ),
    ( 32, 'F_BHAM_LR'           ),
    ( 32, 'F_RF_EXCL'           ),
    ( 32, 'F_COUL_RECIP'        ),
    ( 93, 'F_LJ_RECIP'          ),
    ( 46, 'F_DPD'               ),
    ( 30, 'F_POLARIZATION'      ),
    ( 36, 'F_THOLE_POL'         ),
    ( 90, 'F_FBPOSRES'          ),
    ( 22, 'F_DISRESVIOL'        ),
    ( 22, 'F_ORIRES'            ),
    ( 22, 'F_ORIRESDEV'         ),
    ( 26, 'F_DIHRES'            ),
    ( 26, 'F_DIHRESVIOL'        ),
    ( 49, 'F_VSITE4FDN'         ),
    ( 50, 'F_VSITEN'            ),
    ( 46, 'F_COM_PULL'          ),
    ( 20, 'F_EQM'               ),
    ( 46, 'F_ECONSERVED'        ),
    ( 69, 'F_VTEMP_NOLONGERUSED'),
    ( 66, 'F_PDISPCORR'         ),
    ( 54, 'F_DVDL_CONSTR'       ),
    ( 76, 'F_ANHARM_POL'        ),
    ( 79, 'F_DVDL_COUL'         ),
    ( 79, 'F_DVDL_VDW',         ),
    ( 79, 'F_DVDL_BONDED',      ),
    ( 79, 'F_DVDL_RESTRAINT'    ),
    ( 79, 'F_DVDL_TEMPERATURE'  )]





