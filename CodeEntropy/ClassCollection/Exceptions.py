"""Subclasses of Exception used to define custom errors in the code."""


from CodeEntropy.FunctionCollection import Utils

class CustomBaseError(Exception):
	"""Base class for all the exceptions created for the program."""
	def __init__():
		self.error = ""

	def die(self):
		Utils.hbar(60)
		Utils.printflush("\nERROR\n")
		Utils.printflush(self.error)
		Utils.hbar(60)
		return
#END

class OutOfListError(CustomBaseError):
	"""Raised when an input value is not in the list of expected values."""
	def __init__(self, arg_varname, arg_value, arg_list):
		self.error = f"{arg_value} is not a valid value for {arg_varname}.\nIt only accepts the following values: "
		for _l in arg_list:
			self.error += f"\n\t{_l}"

		self.die()

#END

class LessThanMinError(CustomBaseError):
	"""Raised when an input value is less than the minimum allowed value."""
	def __init__(self, arg_varname, arg_value, arg_minval):
		self.error = f"Value for {arg_varname} cannot be less than {arg_minval}. {arg_value} provided."
		
		self.die()
#END

class GreaterThanMaxError(CustomBaseError):
	"""Raised when an input value is greater than the maximum allowed value."""
	def __init__(self, arg_varname, arg_value, arg_maxval):
		self.error = f"Value for {arg_varname} cannot be greater than {arg_maxval}. {arg_value} provided."

		self.die()
#END

class OutOfRangeError(CustomBaseError):
	"""Raised when an input value is out of the allowed range of values."""
	def __init__(self, arg_varname, arg_value, arg_range):
		rmin, rmax = arg_range
		self.error = f"Value for {arg_varname} must lie between {rmin} and {rmax}. {arg_value} provided."

		self.die()
#END 
