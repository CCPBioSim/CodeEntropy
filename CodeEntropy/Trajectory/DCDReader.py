""" This is a DCD format trajectory file reader. This is the most basic working version.
The code architcrue was obtianed from NAMD's dcdlib.C file in its source code.

NOTE TO DEVELOPERS: 
Due to the difference in the time between when the code for reding TRR and DCD (this one), severe changes in the coding style 
can be seen. At the time this was being written, efforts were made to make the two readers as consistent as possible. """

import struct, xdrlib
import os, sys
import numpy as nmp

from CodeEntropy.Trajectory import TrajectoryFrame as TF
from CodeEntropy.Trajectory import TrajectoryConstants as TCON
from CodeEntropy.Reader import Constants as CONST
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.FunctionCollection import UnitsAndConversions as UAC

class DCDTrajectory(object):
    """ A class that reads CHARMM/NAMD trajectory in DCD format. 
    
    In the DCD file, every block has an integer that starts it and the same integer that marks
    its end. this integer is the bytesize or size of that block, i.e. the number of bytes of information
    to be expected till before the end of the bloack is reached."""
    
    def __init__(self, arg_fileName, arg_dataContainer, arg_beginTime, arg_endTime, arg_stride, arg_verbose):
        self.trajFile = arg_fileName
        self.endianness = CONST.LITTLE_ENN

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

        if arg_dataContainer is None:
            raise SyntaxError('DataContainer will have to be defined.')
        else:
            self.container = arg_dataContainer

        self.generate_frames(arg_verbose)
        return

    #END

    def generate_frames(self, arg_verbose = True):
        """ Generate instances of Snapshot 
        Class by reading the trajectory in a DCD compatibe way """

        currPos = 0
        ifmt = '{}i'.format(self.endianness)
        ffmt = '{}f'.format(self.endianness)

        # traj file size to know when the EOF is reached
        fSize = os.stat(self.trajFile).st_size

        dcdFileHandler = open(self.trajFile, 'rb')
        self.bufferStr = dcdFileHandler.read()
        dcdFileHandler.close()
        # up = xdrlib.Unpacker(self.bufferStr)
         
        #
        #   
        # start reading the buffer with member decide funtion
        #
        #
        ################################################################################
        # FIRST BLOCK
        ################################################################################
        startBlockSize, currPos = self.decode(ifmt, currPos)
        try:
            assert(startBlockSize == 84)
        except:
            raise SyntaxError("Starting byte size is {}. It should be 84. BAD DCD ERROR".format(startBlockSize))

        # look for the word "CORD"
        fmt = '{}4s'.format(self.endianness)
        try:
            magicStr, currPos = self.decode(fmt, currPos)
            # magicStr = up.unpack_string()
            assert( magicStr == b'CORD')
        except:
            raise SyntaxError("Bad DCD File format. 'CORD' string not found.")

        # number of set of coordinates (a.k.a frames/ NSET)
        nframes, currPos = self.decode(ifmt, currPos)
        Utils.printflush("DCD file has {} frame(s)".format(nframes))

        # starting timestep (ISTART)
        istart, currPos = self.decode(ifmt, currPos)
        if arg_verbose >= 3:
            Utils.printflush("Trajectory starts at step {}".format(istart), end= ' ')

        # sampling freq (NSAVC)
        nsavc, currPos = self.decode(ifmt, currPos)
        if arg_verbose >= 3:
            Utils.printflush("with frames separated by {} steps.".format(nsavc))

        #skip 5 "blank" integers
        for i in range(5):
            randInt, currPos = self.decode(ifmt, currPos)

        # number of fixed atoms
        randNum, currPos = self.decode(ifmt, currPos)
        Utils.printflush("{} atoms in the structure are fixed atoms".format(randNum))

        # timestep (a.k.a delta)
        dt , currPos = self.decode('<f', currPos)
        dt *= UAC.AKMA2PS
        dt = nmp.round(dt,3)  # round up to 3 dec places to not have weird flops results
        Utils.printflush('Timestep = {} ps '.format(dt))

        # skip 10 blank integers
        for i in range(10):
            randInt, currPos = self.decode(ifmt, currPos)
            # print("Random Integer {}: {}".format(i, randInt))

        # end size of first block should be indicated by another '84' 
        # (probably indicating the number of bytes read so far)
        endBlocksize, currPos = self.decode(ifmt, currPos)
        try:
            assert(startBlockSize == endBlocksize)
        except:
            raise SyntaxError("Bad DCD File format because of block size mismatch: {}/{}".format(startBlockSize, endBlocksize))


        ################################################################################
        # TITLE BLOCK
        ################################################################################
        startBlockSize, currPos = self.decode(ifmt, currPos)
        if arg_verbose >= 5:
            Utils.printflush('Size of the title block = {}'.format(startBlockSize))

        try:
            assert((startBlockSize - 4)%80 == 0)
        except:
            raise SyntaxError("Bad DCD File format.")

        # numtitles
        numTitle, currPos = self.decode(ifmt, currPos)
        if arg_verbose >= 5:
            Utils.printflush('{} title(s) in the DCD file'.format(numTitle))

        for nt in range(numTitle):
            fmt = '{}80s'.format(self.endianness)
            strTitle, currPos = self.decode(fmt, currPos)
            if arg_verbose >= 5:
                print(strTitle)

        # skip the file seeker by 80*numTitle chars from current file position
        # arg_fileHandler.seek(numTitle * 80, os.SEEK_CUR)
        # currPos += (numTitle * 80)

        # ending size of title block
        endBlocksize, currPos = self.decode(ifmt, currPos)
        try:
            startBlockSize == endBlocksize
        except:
            raise SyntaxError("Bad DCD File format. Size mismatch in the TITLE section.")

        ################################################################################
        # NUMBER OF ATOMS BLOCK
        ################################################################################
        startblockSize, currPos = self.decode(ifmt, currPos)
        if arg_verbose >= 5: 
            Utils.printflush('Size of the atom number block = {}'.format(startBlockSize))

        # number of atoms
        natoms, currPos = self.decode(ifmt, currPos)
        if arg_verbose >= 3:
            Utils.printflush('{} atom(s) in the DCD file'.format(natoms))

        # ending size of title block
        endBlocksize, currPos = self.decode(ifmt, currPos)
        try:
            startBlockSize == endBlocksize
        except:
            raise SyntaxError("Bad DCD File format because of block size mismatch")
        

        Utils.printflush('|',end='')
        ################################################################################
        # SNAPSHOTS
        ################################################################################
        iFrame = 0
        frameIdx = 0
        while currPos < fSize:

            iTime = dt * (istart + (nsavc * iFrame))
            if arg_verbose >= 5:
                Utils.printflush('Reading frame {} timed at {:.3f} ps'.format(iFrame, iTime))
            
            # coord array which will be filled
            vecQuant = nmp.zeros( (natoms, TCON.VECTORDIM))

            startBlockSize, currPos = self.decode(ifmt, currPos)
            ################################################################################
            # CELL BLOCK
            ################################################################################    
            if startBlockSize == 48:
                # Utils.printflush('Will skip the cell information block')
                hasCellInfo = True
                
                #skip the next 48 bytes
                currPos += 48
                
                # end the cell block
                endBlocksize, currPos = self.decode(ifmt, currPos)
                
                try:
                    assert(startBlockSize == endBlocksize)
                except:
                    raise SyntaxError("Bad cell block entry.")
                
                # inititiate the beginning of the next block of coordinate arrays
                startBlockSize, currPos = self.decode(ifmt, currPos)
                
            ################################################################################
            # X Y Z BLOCK
            ################################################################################
            if startBlockSize == 4 * natoms:
                # Utils.printflush("startBlockSize is  = {}".format(startBlockSize))
                try:
                    # starting  x-block
                    # print('About to read x for the next {} bytes'.format(startBlockSize))
                    for idx in range(natoms):
                        x, currPos = self.decode(ffmt, currPos)
                        vecQuant[idx, 0] = x

                    endXBlock, currPos = self.decode(ifmt, currPos)
                except:
                    raise SyntaxError('Failed to read x values of atoms')
                
                try:
                    # starting  y-block
                    # print('About to read y for the next {} bytes'.format(startBlockSize))
                    startYBlock, currPos = self.decode(ifmt, currPos)
                    for jdx in range(natoms):
                        y, currPos = self.decode(ffmt, currPos)
                        vecQuant[jdx, 1] = y

                    endYBlock, currPos = self.decode(ifmt, currPos)
                except:
                    raise SyntaxError('Failed to read y values of atoms')
                
                try:
                    # starting  z-block
                    # print('About to read z for the next {} bytes'.format(startBlockSize))
                    startZBlock, currPos = self.decode(ifmt, currPos)
                    for kdx in range(natoms):
                        z, currPos = self.decode(ffmt, currPos)
                        vecQuant[kdx, 2] = z

                    endZBlock, currPos = self.decode(ifmt, currPos)
                except:
                    raise SyntaxError('Failed to read z values of atoms')
            

            # add the frame to the list of snapshots
            # only if the snapshot is in the demanded time frame
            # also check if it is position at integral multiple strides from the the beginTime
            if self.endTime > iTime >= self.beginTime and nmp.isclose(0.0, (iTime - self.beginTime) % self.stride):
                
                # create an instance of Trajectory frame which will store 3D vectors 
                newFrame = TF.TrajectoryFrame(arg_frameIndex = iFrame, \
                                              arg_vectorDim = TCON.VECTORDIM)
                newFrame.set_numAtoms(natoms)
                newFrame.set_timeStep(iTime)
                newFrame.initialize_matrices(arg_hasVelocities = False, arg_hasForces = False)
                newFrame.value["coordinates"] = vecQuant
                
                self.container.trajSnapshots.append(newFrame)
                self.container.frameIndices.append(frameIdx)
                frameIdx += 1

                if arg_verbose >= 5: 
                    Utils.printflush("Added a new snapshot at time t = {} with {} atoms".format(iTime, natoms))

                if arg_verbose and (iFrame % 100) == 0: 
                    Utils.printflush('.',end='')
            
            if iTime >= self.endTime:
                break
            
            iFrame += 1
        # while 

        if arg_verbose: 
            Utils.printflush('|')
        
        Utils.printflush('Total number of frames in the input trajectory : {}'.format(len(self.container.trajSnapshots)))

        return
  
    #END      

    def decode(self, arg_fmt, arg_startPos):
        """ Returns the first value in the unpacked tuple and the new position of the seeker. Use it for 
        unpacking one data at a time only."""
        
        sfmt = struct.calcsize(arg_fmt)
        val = struct.unpack(arg_fmt, self.bufferStr[arg_startPos: (arg_startPos  + sfmt)])
        newPos = arg_startPos + sfmt
        
        return (val[0], newPos)
    #END
