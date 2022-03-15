import numpy as nmp
from CodeEntropy.FunctionCollection import CustomFunctions as CF 

def transform_coordinates(arg_origVector, arg_oldBase, arg_newBase):
	"""Given a 3D vector (origVector; V) with components in the old_base (3,3 matrix), 
	transform it to a new 3D vector (return value) with components in the new_base (3x3 matrix).
	A 3X3 transformation matrix TM is created which gives new vector components V' = TM @ V. 
	TM[i,j] = dot(axis{j_new}, axis{i_old}).

	NOTE: Axes in both coordinate system have to be orthogonal"""

	#transformation matrix
	TM = [[nmp.dot(j,i) for i in arg_newBase] for j in arg_oldBase]

	return TM @ arg_origVector

#END


def get_cartesian_midpoint3D(arg_point1, arg_point2):
	""" Get the cartesian midpoint of the line joining two 3D points """
	return nmp.multiply(0.5, nmp.add(arg_point1, arg_point2))

#END

def generate_orthonormal_axes_system(arg_coord1, arg_coord2, arg_coord3):
	"""Generate a 3 x 3 matrix with columns as axes of the returning coordinate system. 
	They must be orthonormal (assert).
	Axis 1: Along the C-N line, from altitude bisector to C
	Axis 2: from altitde bisecting point of C-N to Ca
	Axis 3: Cross of Axis1 and Axis 2

	Return the basis and the origin"""

	basis = nmp.ndarray((3,3))

	r1 = arg_coord1
	r2 = arg_coord2
	r3 = arg_coord3

	r31 = r3 - r1
	r21 = r2 - r1

	#derived formula
	rAltitude = nmp.dot(r31, r21) * r21
	rAltitude /= nmp.linalg.norm(r21)**2
	rAltitude += r1

	# Axis 0
	axis0 = r2 - rAltitude
	axis0 /= nmp.linalg.norm(axis0)
	basis[0,:] = axis0


	# Axis 1
	axis1 = r3 - rAltitude
	axis1 /= nmp.linalg.norm(axis1)
	basis[1,:] = axis1

	# Axis 2
	# axis2 = nmp.cross(axis0, axis1)
	axis2 = CF.cross_product(axis0, axis1)
	axis2 /= nmp.linalg.norm(axis2)
	basis[2,:] = axis2

	# due to FP limitations, the check will verly likely fail
	# assert(nmp.dot(axis1, axis2) == 0)

	return basis, rAltitude

#END

def get_altitude(arg_lineEndPoint1, arg_lineEndPoint2, arg_sourcePoint):
	"""
	For the three points in the input, 
	2 form a line segement and 
	the thrid one is the point from
	where the altitude on the line segement is computed.

	Returns te altitude vector"""


	r1 = arg_lineEndPoint1
	r2 = arg_lineEndPoint2
	r3 = arg_sourcePoint

	r31 = r3 - r1
	r21 = r2 - r1

	#derived formula
	rAltitude = nmp.dot(r31, r21) * r21
	rAltitude /= nmp.linalg.norm(r21)**2
	rAltitude = r31 - rAltitude

	return rAltitude

# END


def get_sphCoord_axes(arg_r):
	""" For a given vector in space, treat it is a radial vector rooted at 0,0,0 and 
	derive a curvilinear coordinate system according to the rules of polar spherical 
	coordinates"""

	x2y2 = arg_r[0]**2 + arg_r[1]**2
	r2 = x2y2 + arg_r[2]**2

	if x2y2 != 0.:
		sinTheta = nmp.sqrt(x2y2/r2)	
		cosTheta = arg_r[2]/nmp.sqrt(r2)

		sinPhi = arg_r[1]/nmp.sqrt(x2y2)
		cosPhi = arg_r[0]/nmp.sqrt(x2y2)
		
	else:
		sinTheta = 0.
		cosTheta = 1

		sinPhi = 0.
		cosPhi = 1

	# if abs(sinTheta) > 1 or abs(sinPhi) > 1:
	#	 print('Bad sine : T {} , P {}'.format(sinTheta, sinPhi))

	# cosTheta = nmp.sqrt(1 - sinTheta*sinTheta)
	# cosPhi = nmp.sqrt(1 - sinPhi*sinPhi)
	
	# print('{} {} {}'.format(*arg_r))
	# print('Sin T : {}, cos T : {}'.format(sinTheta, cosTheta))
	# print('Sin P : {}, cos P : {}'.format(sinPhi, cosPhi))

	sphericalBasis = nmp.zeros((3,3))
	
	# r^
	sphericalBasis[0,:] = nmp.asarray([sinTheta*cosPhi, sinTheta*sinPhi, cosTheta])
	
	# Theta^
	sphericalBasis[1,:] = nmp.asarray([cosTheta*cosPhi, cosTheta*sinPhi, -sinTheta])
	
	# Phi^
	sphericalBasis[2,:] = nmp.asarray([-sinPhi, cosPhi, 0.])
	
	return sphericalBasis
	
# END

def compute_dihedral(arg_ri, arg_rj, arg_rk, arg_rl):
	""" 
	Return the dihedral angle in degrees formed by the atoms with the input coordinates.
	The formula to compute was obtained from: 
	https://www.math.fsu.edu/~quine/MB_10/6_torsion.pdf
	
	"""

	# define the directed line segments (vectors)
	a = nmp.subtract(arg_rj, arg_ri) # i->j
	b = nmp.subtract(arg_rk, arg_rj) # j->k
	c = nmp.subtract(arg_rl, arg_rk) # k->l

	# Complex number formalism for finding the angle
	# Z = X + jY
	X = -nmp.dot(a,c) * nmp.dot(b,b)
	X += ( nmp.dot(a,b) * nmp.dot(b,c) )

	Y = nmp.dot(a, nmp.cross(b,c))
	Y *= nmp.linalg.norm(b)

	# dihedral angle (in degrees)
	dih = nmp.angle(complex(X,Y), deg=True)

	return dih
# END
