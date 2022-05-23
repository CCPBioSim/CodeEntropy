"""
	A collection of utility functions or wrappers that 
	are used throughout the package.
"""
from __future__ import print_function

import struct
import numpy as nmp
import re
from datetime import datetime, timedelta

from CodeEntropy.FunctionCollection import UnitsAndConversions as CONST

def printflush(arg_string, end = '\n'):
	""" Modified print statement with flush = True in the end """
	print(arg_string,flush = True, end = end)
	return

#END

def printOut(arg_outFile, arg_string, end = "\n"):
	""" Modified print statement that will write a tring into a outfile """
	with open(arg_outFile, 'a') as fout:
		fout.write('{}{}'.format(arg_string, end))
	return

#END

def printList(arg_outFile,arg_list, arg_fmt, arg_maxRowLen, arg_descriptor = None):
	""" Writes the elements of a 1D list into the output file.
		At most `arg_maxRowLen` values are written per line in addition to the
		descriptor in each line. """
	listLength = len(arg_list)
	numWholeRows = int(nmp.floor(listLength/arg_maxRowLen))

	# if an empty list, print an empty line
	if listLength == 0:
		if arg_descriptor:
			printOut(arg_outFile,f"{arg_descriptor} ", '')
		printOut(arg_outFile,"")
		return

	startIdx = 1
	listIdx = 0
	for i in range(numWholeRows):
		if arg_descriptor:
			printOut(arg_outFile,f"{arg_descriptor} ", '')
		
		# printOut(arg_outFile,"{:>5d} {:>5d}".format(startIdx, startIdx + 5 - 1), ':')
		while listIdx < (startIdx + arg_maxRowLen - 1):
			printOut(arg_outFile,f"{arg_list[listIdx]:{arg_fmt}}",'')
			listIdx += 1
		printOut(arg_outFile,"")
		startIdx += arg_maxRowLen
	
	# the last row (if any)
	if (listLength % arg_maxRowLen != 0):
		if arg_descriptor:
			printOut(arg_outFile,f"{arg_descriptor} ", '')
		# printOut(arg_outFile,"{:>5d} {:>5d}".format(startIdx, listLength), ':')
		while listIdx < listLength:
			printOut(arg_outFile,f"{arg_list[listIdx]:{arg_fmt}}",'')
			listIdx += 1
		printOut(arg_outFile,"")

	return
#END

def hbar(arg_len = 50):
	printflush('-'*arg_len)
	return
#END

def print_time(arg_datetime):
	"""
	Prints date and time in YYYY-MM-DD HH:MM:SS format
	"""
	year = arg_datetime.year
	month = arg_datetime.month
	day = arg_datetime.day
	hour = arg_datetime.hour
	minute = arg_datetime.minute
	second = arg_datetime.second
	
	datestr = f"{year:>4d}-{month:>02d}-{day:>02d}  "
	timestr = f"{hour:>02d}:{minute:>02d}:{second:>02d}"
	printflush(datestr + timestr)
	return
#END

def print_duration(arg_t0, arg_t1):
	"""
	Prints the duration of time between t0 (first argument) and t1 (second argument)
	in appropriate human-readable format.
	"""
	dur = datetime(year = 1, month = 1, day = 1) + (arg_t1 - arg_t0)
	if dur.year > 1:
		timestr = f"{(dur.year - 1):>4d}yr {(dur.month - 1):>2d}mo {(dur.day - 1):>2d}dy {dur.hour:>2d}hr {dur.minute:>2d}min {dur.second:>2d}sec"
	elif dur.month > 1:
		timestr = f"{(dur.month - 1):>2d}mo {(dur.day - 1):>2d}dy {dur.hour:>2d}hr {dur.minute:>2d}min {dur.second:>2d}sec"
	elif dur.day > 1:
		timestr = f"{(dur.day - 1):>2d}dy {dur.hour:>2d}hr {dur.minute:>2d}min {dur.second:>2d}sec"
	elif dur.hour > 1:
		timestr = f"{dur.hour:>2d}hr {dur.minute:>2d}min {dur.second:>2d}sec"
	elif dur.minute > 1:
		timestr = f"{dur.minute:>2d}min {dur.second:>2d}sec"
	else:
		timestr = f"{dur.second:>2d}sec"

	printflush(timestr)
	return
#END


def decode(arg_fmt, arg_fileHandler):
	""" Using the struct module, read/decode a segment of binary data into a type indicated by the input format """
	sfmt = struct.calcsize(arg_fmt)
	return struct.unpack(arg_fmt, arg_fileHandler.readline(sfmt))
#END

def binary_to_dec_repr(arg_byteArray):
	""" For an input sequence of 0/1, return its equivalent decimal form in base 10 """
	decEquivalent = 0
	for idx, iPow in enumerate(range(len(arg_byteArray))):
		decEquivalent += (arg_byteArray[idx] * nmp.power(2,iPow))

	return decEquivalent
#END

def coalesce_numeric_array(arg_numArray):
	""" Take the elements in a given input array with integer elements and coalesce them to returna string whose characters
	are string cast of teh elements """
	charList = [str(char) for char in arg_numArray.astype(int)]
	return ''.join(charList)
#END


def diagonalize(arg_matrix):
	""" 
	Find the eigen vectors and values for the arg_matrix 
	using the linalg module of numpy.
	"""
	eigenValues, eigenVectors = nmp.linalg.eigh(arg_matrix)
	# a work around for complex values changing all very small value and negative to zero
	# eigenValues[eigenValues < 0 ] = 0
	return eigenValues, eigenVectors
#END 

def get_frequencies_from_nmd(arg_nmdfile, wnum2freq = True):
	"""
	From the NMD spectra file, extract frequency values and return a list.
	"""
	freqList = []
	with open(arg_nmdfile, "r") as fnmd:
		for line in fnmd:
			if re.match("^# \*\*\*", line):
				freq = float(line.split()[-1])
				if wnum2freq:
					freq *= CONST.C_LIGHT
				freqList.append(freq)
	return freqList
#END