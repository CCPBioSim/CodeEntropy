from CodeEntropy.Reader import Constants as CONST

# LENGTH
NM2ANG = 1e1
M2ANG = 1e10
ANG2M = 1e-10
NM2M = 1e-9
ANG2NM = 1e-1


# MASS
AMU2KG = 1.66053892110e-27
KG2AMU = 6.022141289754078e+26
sqrtKG2AMU = 24540051527562.199

# ENERGY
KJ2KCAL = 0.239006
KCAL2KJ = 4.184

#CHARMM units
AKMA2PS = 4.88882129e-02  

def change_lambda_units(arg_lambdas):
	"""Unit of lambdas : kJ2 mol-2 A-2 amu-1
	change units of lambda to J/s2"""
	# return arg_lambdas * N_AVOGADRO * N_AVOGADRO * AMU2KG * 1e-26
	return arg_lambdas * 1e+29 / CONST.N_AVOGADRO

def get_KT2J(arg_temper):
	"""A temperature dependent KT to Joule conversion"""
	return 4.11e-21 * arg_temper/CONST.DEF_TEMPER


