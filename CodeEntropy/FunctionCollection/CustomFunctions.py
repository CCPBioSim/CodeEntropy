import numpy as nmp
import sys

from bitarray import bitarray
from CodeEntropy import ClassCollection as CC 
from CodeEntropy.FunctionCollection import Utils

def cross_product(arg_v1, arg_v2):
	""" 
	Returns cross product of two vectors using analytical expressions 
	that one would have from using the determinant used to compute the product.
	It appears to be a tad bit faster than the numpy one.
	"""
	vecCross = nmp.zeros(3)

	vecCross[0] =  arg_v1[1]*arg_v2[2] - arg_v2[1]*arg_v1[2]
	vecCross[1] = -arg_v1[0]*arg_v2[2] + arg_v2[0]*arg_v1[2]
	vecCross[2] =  arg_v1[0]*arg_v2[1] - arg_v2[0]*arg_v1[1]
		 
	return vecCross

#END

def dot_product(arg_v1, arg_v2):
	"""
	Returns dot product of two vectors using numpy's dot function.
	""" 
	return nmp.dot(arg_v1, arg_v2)
#END

def euler_vector(arg_r, arg_theta):
	"""
	Create and return a vector of the form r*e^i(theta) in the argand plane.
	Angle should be degrees.
	"""
	vx = arg_r * nmp.cos(nmp.deg2rad(arg_theta))
	vy = arg_r * nmp.sin(nmp.deg2rad(arg_theta))

	return (vx, vy)
#END


def filter_zero_rows_columns(arg_matrix):
	
	#record the initial size
	initShape = nmp.shape(arg_matrix)

	zeroIdx = list(filter(lambda row : nmp.all(nmp.isclose(arg_matrix[row,:] , 0.0)) , nmp.arange(nmp.shape(arg_matrix)[0])))
	allIdx = nmp.ones((nmp.shape(arg_matrix)[0]), dtype=bool)
	allIdx[zeroIdx] = False 
	arg_matrix = arg_matrix[allIdx,:]

	allIdx = nmp.ones((nmp.shape(arg_matrix)[1]), dtype=bool)
	zeroIdx = list(filter(lambda col : nmp.all(nmp.isclose(arg_matrix[:,col] , 0.0)) , nmp.arange(nmp.shape(arg_matrix)[1])))
	allIdx[zeroIdx] = False
	arg_matrix = arg_matrix[:,allIdx]

	# get the final shape
	finalShape = nmp.shape(arg_matrix)

	if initShape != finalShape:
		Utils.printflush('A shape change has occured ({},{}) -> ({}, {})'.format(*initShape, *finalShape))

	return arg_matrix
#END


def probability_of_coexistence(arg_v1, arg_v2):
	""" From the input vectors/list , the function returns the probability
	 of co-occurence (True value for both vectors at the same index/time) of the same state across them """
	
	try:
		assert(len(arg_v1) == len(arg_v2))
	except:
		raise ValueError("Unequal vector lengths ({} and {}). The lengths should be the same.".format(len(arg_v1), len(arg_v2)))
	
	#prob   = nmp.dot(arg_v1, arg_v2)/len(arg_v1)
	prob = 0
	for x, y in zip(arg_v1, arg_v2):
		prob += (x * y)

	prob /= len(arg_v1)

	if prob < 0:
		Utils.printflush(arg_v1)
		Utils.printflush(arg_v2)

	return prob
#END

def covariance(arg_v1, arg_v2):
	""" Returns the covariance between two vectors/list """

	mean1 = nmp.mean(arg_v1)
	mean2 = nmp.mean(arg_v2)
	
	try:
		assert(len(arg_v1) == len(arg_v2))
	except:
		raise ValueError("Inconsistent vector lengths ({} and {}). The lengths should be the same.".format(len(arg_v1), len(arg_v2)))
		
	N = len(arg_v1)
	cov = 0
	for x, y in zip():
		cov += ((x - mean1) * (y - mean2))
	
	cov /= (N -  1)
	return cov

#END

def phi_coeff(arg_v1, arg_v2):
	"""
	From the input binary vector/list, creates a contingency matrix with the following entries:
	[1,1] <- probability that the order combination is 1/1 at the same time
	[1,2] <- probability that the order combination is 1/0 at the same time
	[2,1] <- probability that the order combination is 0/1 at the same time
	[2,2] <- probability that the order combination is 1/1 at the same time

	Computes and returns the phi coefficient from this matrix.
	"""
	contingencyMatrix = nmp.zeros((2,2))

	notV1 = nmp.array([not(bit) for bit in arg_v1], dtype = nmp.int8)
	notV2 = nmp.array([not(bit) for bit in arg_v2], dtype = nmp.int8)

	contingencyMatrix[0,0] = probability_of_coexistence(arg_v1, arg_v2)
	contingencyMatrix[0,1] = probability_of_coexistence(arg_v1, notV2)
	contingencyMatrix[1,0] = probability_of_coexistence(notV1, arg_v2)
	contingencyMatrix[1,1] = probability_of_coexistence(notV1, notV2)

	# print contingency matrix
	Utils.printflush('{:>8.6f} {:>8.6f} \n {:>8.6f} {:>8.6f}'. format(*nmp.ndarray.flatten(contingencyMatrix)))

	# column and row sums
	
	#   row sums
	sumR0 = nmp.sum(contingencyMatrix[0,:])
	sumR1 = nmp.sum(contingencyMatrix[1,:]) 

	# col sums
	sumC0 = nmp.sum(contingencyMatrix[:,0])
	sumC1 = nmp.sum(contingencyMatrix[:,1])
	
	Utils.printflush('r0 = {:>8.6f}, r1 = {:>8.6f}, c0 = {:>8.6f}, c1 = {:>8.6f}'.format(sumR0, sumR1, sumC0, sumC1))

	# return coeff
	if ( sumR0 * sumR1 * sumC0 * sumC1 == 0 ):
		return 0
	else:
		phi = nmp.linalg.det(contingencyMatrix)
		phi /= nmp.sqrt(sumR0 * sumR1 * sumC0 * sumC1)
		return phi
#END


