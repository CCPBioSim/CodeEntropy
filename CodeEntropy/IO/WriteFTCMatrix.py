import xdrlib
import numpy as nmp

def write_FTC_matrix_bin( arg_beadCollection ):
	""" This function writes the FTC matrix (FF + FT + TF + TT) for a given instance of BeadCollection in binary format.
	The sequence of the data is:
	N (# Beads), M x M (length of FTC matrix), MFF x MFF (length of FF matrix), MFT_row x MFT_col (length of FT matrix),
	MTF_row x MTF_col (length of TF matrix), MTT x MTT (length of TT matrix),
	FF matrix as 1D array, FT matrix as 1D array, TF matrix as 1D array, TT matrix as 1D array. Dimensions are unsigned int
	and arrays are in floats """

	fBin = open('FTC_matrix.bin', 'wb')

	xpa = xdrlib.Packer()

	# number of beads
	xpa.pack_uint( len(arg_beadCollection.listOfBeads) )

	# dim of FTC matrix
	pack_matrix_shape( arg_beadCollection.forceTorqueCovMatrix, xpa )
	
	# dim of FF, FT, TF, TT
	for iKey in arg_beadCollection.subMatrixDict.keys():
		iMat = arg_beadCollection.generate_quadrant( iKey, True)
		pack_matrix_shape(iMat, xpa)

	# flatten the submatrices and store them in 1D arrays
	for iKey in arg_beadCollection.subMatrixDict.keys():
		iMat = arg_beadCollection.generate_quadrant(iKey, True)
		pack_matrix_as_array(iMat, xpa)

	# write he buffer into fBin
	fBin.write(xpa.get_buffer())

	fBin.close()

	return
#END

def pack_matrix_shape( arg_matrix, arg_packer ):
	for iSize in nmp.shape(arg_matrix):
		# print("packing", iSize)
		arg_packer.pack_uint(iSize)

	return
#END

def pack_matrix_as_array ( arg_matrix, arg_packer ):
	for iVal in arg_matrix.flatten():
		arg_packer.pack_float(iVal)

	return
#END


