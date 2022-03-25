""" This is a Gromacs TRR file reader. 
http://www.ks.uiuc.edu/Research/vmd/plugins/doxygen/Gromacs_8h-source.html#l01300"""

import sys, os
import struct
import xdrlib
import numpy as nmp

from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.Trajectory import TrajectoryFrame as TF
from CodeEntropy.Trajectory import TrajectoryConstants as TCON


class GroTrajectory():
	""" A class that reads gromacs trajectory in TRR format"""

	def __init__(self, arg_fileName, arg_beginTime, arg_endTime, arg_stride, arg_verbose):
		self.trajFile = arg_fileName
		self.snapshots = [] 
		
		if arg_beginTime < 0:
			self.beginTime = 0
		else:
			self.beginTime = arg_beginTime #in ps
		
		if arg_endTime < 0:
			self.endTime = nmp.finfo(float).max  # set it to the maximum possible float
		else:
			self.endTime = arg_endTime  #in ps

		if arg_stride < 0:
			self.stride = 1
		else:
			self.stride = arg_stride  #in ps

		# int version;			// File version number
		# char title[MAX_TRX_TITLE + 1];  // File title : MAX_TRX_TITLE   80
		# int ir_size;
		# int e_size;
		# int box_size;
		# int vir_size;
		# int pres_size;
		# int top_size;
		# int sym_size;
		# int x_size;			 // Positions of atoms
		# int v_size;			 // Velocities of atoms
		# int f_size;
		# int natoms;			 // Number of atoms in the system
		# int step;
		# int nre;
		# float t;
		# float lambda;


		self.generate_frames(arg_verbose)
		return

	def generate_frames(self, arg_verbose = True):
		
		# traj file size to know when the EOF is reached
		fSize = os.stat(self.trajFile).st_size

		# generate new frames as we read the data
		iFrame = 0

		with open(self.trajFile, 'rb') as trajFileHandler:

			# create a header buffer and use it to examine if any more info can be read later on
			buff = trajFileHandler.read()
			
			# if verbose
			if arg_verbose : print('|',end='')

			# create an unpacker for the buffer
			up = xdrlib.Unpacker(buff)
			
			# read till the END where no more information can be read anymore
			while up.get_position() < fSize:

				version = up.unpack_int()

				# gromacs places an unsigned integer indicating the size of 
				# the string that is follow
				up.unpack_int(); 
				up.unpack_string() #title
				
				ir_size, e_size, box_size, vir_size, pres_size,top_size,sym_size, x_size,v_size, f_size, natoms, step, nre = [up.unpack_int() for ii in range(13)]

				# check if forces are present, die if not.
				if x_size != f_size:
					Utils.hbar()
					Utils.printflush("ERROR> TRR file does not contain forces that can be mapped to each frame.")
					Utils.printflush("\n\nSolution: Run MD with nstfout = <same value as nstxout> and/or")
					Utils.printflush("use -force flag when extracting trajectory using gmx TRJCONV.\n\n")
					sys.exit()

				# print(ir_size, e_size, box_size, vir_size, pres_size,top_size,sym_size, x_size,v_size, f_size, natoms, step, nre)

				t = up.unpack_float()
				# print(t)
				alchLambda = up.unpack_float()

				# determine precision
				precision = x_size/(3 * natoms)
				if precision == 4:
					precision = 'f'
				elif precision == 8:
					precision = 'd'


				# skip certain details
				matSize = sum([ir_size, e_size, box_size, vir_size, pres_size,top_size,sym_size])
				# offset the position further by the matrix sizes from the current position
				up.set_position(up.get_position() + matSize)

				# read the coordinates, velocities and forces (natoms x vectorDim) if they are present
				# they must be present in a sequence in the binary file
				
				# coordinates
				if precision == 'f':
					binCoords = [up.unpack_float() for i in range(natoms * TCON.VECTORDIM)]
				elif precision == 'd':
					binCoords = [up.unpack_double() for i in range(natoms * TCON.VECTORDIM)]

				
				# vel 
				if v_size > 0:
					velSize = 0  #default
					if precision == 'f':
						velSize = struct.calcsize('f')
					elif precision == 'd':
						velSize = struct.calcsize('d')
					# skip the velocity section
					up.set_position( up.get_position() + (velSize * natoms * TCON.VECTORDIM) )
					
				
				# forces 
				if f_size > 0:
					if precision == 'f':
						binForces = [up.unpack_float() for i in range(natoms * TCON.VECTORDIM)]
					elif precision == 'd':
						binForces = [up.unpack_double() for i in range(natoms * TCON.VECTORDIM)]
				
				# add the frame to the list of snapshots
				# only if the snapshot is in the demanded time frame
				# also check if it is position at integral multiple strides from the the beginTime
				if self.endTime > t >= self.beginTime and (0 == ((t - self.beginTime) % self.stride)):
					# create an instance of Trajectory frame which will store 3D vectors 
					newFrame = TF.TrajectoryFrame(arg_frameIndex = iFrame, \
					                              arg_vectorDim = TCON.VECTORDIM)
					newFrame.set_numAtoms(natoms)
					newFrame.set_timeStep(t)
					newFrame.initialize_matrices(arg_hasVelocities = (v_size > 0), arg_hasForces = (f_size > 0))
					newFrame.value["coordinates"] = binCoords
					newFrame.value["forces"] = binForces
					self.snapshots.append(newFrame)
					if arg_verbose >= 5: 
						Utils.printflush("Added a new snapshot at time t = {} with {} atoms".format(t, natoms))

				if arg_verbose and (iFrame % 100) == 0: 
					Utils.printflush('.',end='')
				
				iFrame += 1

				if t == self.endTime:
					break

		if arg_verbose: 
			Utils.printflush('|')
		
		Utils.printflush('Total number of frames in the input trajectory : {}'.format(len(self.snapshots)))

		return
#END


if __name__ == "__main__":
	trajFile = sys.argv[1]
	trajObj = GroTrajectory(trajFile, True)
