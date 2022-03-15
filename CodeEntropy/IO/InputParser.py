import optparse 
import re
import sys, os

from CodeEntropy.ClassCollection import Exceptions

class AdvancedOptionParser(optparse.OptionParser):
	""" An extended/advanced version of optionParser """
	def __init__(self):
		super().__init__()

	def validate_file(self, arg_filename):
		if os.path.isfile(arg_filename):
			pass
		else:
			raise OSError('File not found', ';', arg_filename)
		return
#END



def validate(arg_varname, arg_value, **kwargs):
	"""Validates values being asked to be assigned to a variable based on various criteria."""

	## checking if a value is in a list 
	try:
		if isinstance(kwargs['allowedValues'], list):
			if arg_value not in kwargs['allowedValues']:
				raise Exceptions.OutOfListError(arg_varname, arg_value, kwargs['allowedValues'])
			else:
				return True
	except KeyError:
		pass

	## chekcing if a value is less than the minimum allowed value
	try:
		if arg_value < kwargs['minval']:
			raise Exceptions.LessThanMinError(arg_varname, arg_value, kwargs['minval'])
		else:
			return True
	except KeyError:
		pass

	## chekcing if a value is greater than the maximum allowed value
	try:
		if arg_value > kwargs['maxval']:
			raise Exceptions.GreaterThanMaxError(arg_varname, arg_value, kwargs['maxval'])
		else:
			return True
	except KeyError:
		pass

	## chekcing if a value is within the allowed range of values
	try:
		rmin, rmax = kwargs['range']
		if arg_value < rmin or arg_value > rmax:
			raise Exceptions.OutOfRangeError(arg_varname, arg_value, kwargs['range'])
		else:
			return True
	except KeyError:
		pass


#END


