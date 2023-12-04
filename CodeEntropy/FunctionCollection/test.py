#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:11:14 2023

@author: ioana
"""

def coalesce_numeric_array(arg_numArray):
	""" Take the elements in a given input array with integer elements and coalesce them to return a string whose characters
	are string cast of teh elements """
	charList = [str(char) for char in arg_numArray]
	return ''.join(charList)
#END
    
arg_numArray=[10,5,8,3,45,600]
string = coalesce_numeric_array(arg_numArray)
print(string)