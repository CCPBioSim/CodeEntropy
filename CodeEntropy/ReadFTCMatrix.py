import xdrlib
import numpy as nmp
import sys

def print_FTC_matrix_bin( arg_filename ):
	""" This function writes the FTC matrix (FF + FT + TF + TT) for a given instance of BeadCollection in binary format.
	The sequence of the data is:
	N (# Beads), M x M (length of FTC matrix), MFF x MFF (length of FF matrix), MFT_row x MFT_col (length of FT matrix),
	MTF_row x MTF_col (length of TF matrix), MTT x MTT (length of TT matrix),
	FF matrix as 1D array, FT matrix as 1D array, TF matrix as 1D array, TT matrix as 1D array. Dimensions are unsigned int
	and arrays are in floats """

	with open(arg_filename, 'rb') as fBin:

		buff = fBin.read()
		xup = xdrlib.Unpacker(buff)

		# number of beads
		numBeads = xup.unpack_uint()
		print('Number of beads = {}'.format( numBeads )) 

		# dim of FTC matrix
		shapeFTC = (xup.unpack_uint(), xup.unpack_uint())
		print('FTC matrix size = {} x {}'.format( *shapeFTC ))
		
		# dim of FF, FT, TF, TT
		matrixShapeDict = dict()
		for iMat in ["FF", "FT", "TF", "TT"]:
			matrixShapeDict[iMat] = (xup.unpack_uint(), xup.unpack_uint())
			print('{} matrix size = {} x {}'.format( iMat, *matrixShapeDict[iMat] ))

		# reshape the flattened version of  the submatrices into appropriate sizes
		matrixDict = dict()
		for iMat in ["FF", "FT", "TF", "TT"]:
			matrixSize = nmp.product( matrixShapeDict[iMat] )
			flatArr = [ xup.unpack_float() for i in range(matrixSize) ]
			matrixDict[iMat] = nmp.reshape( flatArr,  matrixShapeDict[iMat] )
			print('Trace of {} : {:>10.4}'.format( iMat, nmp.trace(matrixDict[iMat]) ))
		# place them in correct order
		matrixFTC = nmp.block( [     [matrixDict["FF"]  ,  matrixDict["FT"]],
		                             [matrixDict["TF"]  ,  matrixDict["TT"]]  ] )

		# assert that this is a square matrix
		assert(nmp.shape(matrixFTC) == shapeFTC)

		# print
		for iRow in range(shapeFTC[0]):
			for iCol in range(shapeFTC[1]):
				print('{:>10.4}'.format(matrixFTC[iRow, iCol]), end = '')
			print('')




	return
#END

if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print('Input a bindary FTC matrix file to print its contents out in the terminal')
		sys.exit()

	print_FTC_matrix_bin(sys.argv[1])
