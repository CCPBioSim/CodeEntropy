import os, sys, re
from datetime import datetime, timedelta

from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.FunctionCollection import EntropyFunctions
from CodeEntropy.IO import InputParser, Writer 
from CodeEntropy.Reader import CharmmReader

def parse_input():
	inputParser = InputParser.AdvancedOptionParser()

	# Usage
	inputParser.usage = "The script uses CHARMM-generated coordinate and force DCD files and PSF topology file to compute entropy using the MCC method."

	# TOPOLOGY/STRUCTURE file in PSF format
	inputParser.add_option("--psf",
						   action="store",
						   dest  ="topology",
						   type  ="string",
						   help  ="Structure/topology file (.psf)")

	# COORDINATE TRAJECTORY file in DCD format
	inputParser.add_option("--dcd",
						   action="store",
						   dest  ="xtraj",
						   type  ="string",
						   help  ="coordinate trajectory file (.dcd)")

	# FORCE file in DCD format
	inputParser.add_option("--frc",
						   action="store",
						   dest  ="ftraj",
						   type  ="string",
						   help  ="force file (.dcd)")

	# start reading the trajectory from this timestep (ps)
	inputParser.add_option("-b","--begin",
						   action="store",
						   dest  = "beginTime",
						   default = 0,
						   type = float,
						   help = "Start reading the trajectory from this timestep (ps)")
	
	# stop reading the trajectory at this timestep (ps)
	inputParser.add_option("-e","--end",
						   action="store",
						   dest  = "endTime",
						   default = -1,
						   type = float,
						   help = "Stop reading the trajectory at this timestep (ps)") 

	# read frames with this stride (ps) 
	inputParser.add_option("-d","--stride",
						   action="store",
						   dest  = "stride",
						   default = 1,
						   type = float,
						   help = "The interval between two consecutive frames to be read (ps)") 

	# temperature for Entropy calc (K)
	inputParser.add_option("--temper",
		                   action="store",
		                   dest = "temper",
		                   default = 298,
		                   type = float,
		                   help = "Temperature for entropy calculation (K)")

	# COMPUTE WHOLE MOLECULE LEVEL ENTROPY
	inputParser.add_option("--molecule",
						   action="store_true",
						   dest  ="doMolEntropy",
						   default=False,
						   help  ="Do entropy calculation at whole molecule level (The whole molecule is treated as one single bead.)")
	
	# COMPUTE RESIDUE LEVEL ENTROPY
	inputParser.add_option("--residue",
						   action="store_true",
						   dest  ="doResEntropy",
						   default=False,
						   help  ="Do entropy calculation at residue level (A residue as a whole represents a bead.)")
	
	# COMPUTE UNITED ATOM LEVEL ENTROPY
	inputParser.add_option("--uatom",
						   action="store_true",
						   dest  ="doUAEntropy",
						   default=False,
						   help  ="Do entropy calculation at united atom level (A heavy atom and its covalently bonded H-atoms for an united atom and represent a bead.)")
	
	# COMPUTE TOPOGRAPHICAL ENTROPY
	inputParser.add_option("--topog",
							action="store",
							dest  = "topogEntropyMethod",
							type = "string",
							default="0",
							help = "Compute the topographical entropy using\n 1 : pLogP method \n 2: Corr. pLogP method \n 3: Corr. density function \n 4: Phi Coeff method \n aem: Corr. pLogP after adaptive enumeration of states")

	# VERBOSITY
	inputParser.add_option("-v","--verbosity",
						   action="store",
						   dest  ="verbose",
						   type=int,
						   default=0,
						   help  ="the level of verbosity (0 - no verbosity, 5 - speak your heart out)")
	
	# LOG FILE
	inputParser.add_option("-o","--out",
						   action="store",
						   dest  ="outFile",
						   default="outfile.dat",
						   help   ="Name of the file where all the output will be written.")
	
	# MATRIX FILE
	inputParser.add_option("--mout",
						   action="store",
						   dest  ="moutFile",
						   default=None,
						   help   ="Name of the file where all the covariance matrices will be printed (default: None).")

	# NMD FORMAT FILE
	inputParser.add_option("--nmd",
						   action="store",
						   dest  ="nmdFile",
						   default=None,
						   help   ="Name of the file where VMD compatible NMD format files with mode information will be printed (default: None).")
	
	# append to log file
	inputParser.add_option("--append",
						   action="store_true",
						   dest  = "doAppend",
						   default = False,
						   help = "Append output statments to an already existing output text file")

	# force scaling factor
	inputParser.add_option("--fscale",
	                       action="store",
						   dest="fScale",
						   default = 1,
						   type=float,
						   help = "Scale the atomic forces by this factor")

	# torque scaling factor
	inputParser.add_option("--tscale",
	                       action = "store",
						   dest = "tScale",
						   default = 1,
						   type=float,
						   help = "Scale the atomic torques by this factor")

	if len(sys.argv[1:]) == 0:
		inputParser.print_help()
		sys.exit(1)

	options, args = inputParser.parse_args()
	return (options, args)

#END

def mcc_charmm():

	Utils.printflush("Program started on : ", end = '')
	t0 = datetime.now()
	Utils.print_time(t0)

	# Fetch topology and coord/force trajectory file names 
	options, args = parse_input()

	# READ ALL THE INPUT OPTIONS
	# filenames
	topolFile	 = options.topology
	xtrajFile	 = options.xtraj
	ftrajFile    = options.ftraj

	# trajectory beginning, stride and end timestep
	beginTime = options.beginTime
	endTime   = options.endTime
	stride    = options.stride

	# temperature
	temper = options.temper

	# verbosity
	verbose	  = options.verbose
	
	# output
	outFile	  = options.outFile
	moutFile  = options.moutFile
	nmdFile   = options.nmdFile
	append	  = options.doAppend

	# force/torque scaling
	fScale = options.fScale
	tScale = options.tScale

	# levels for entropy calculation
	doMolEntropy = options.doMolEntropy
	doResEntropy = options.doResEntropy
	doUAEntropy  = options.doUAEntropy

	# method for TOPOGRAPHICAL entropy calculation
	topogEntropyMethod = options.topogEntropyMethod
	topogEntropyMethod = options.topogEntropyMethod
	

	##### VALIDATION #####
	InputParser.validate("Force scaling", fScale, minval = 0.01)
	InputParser.validate("Torque scaling", tScale, minval = 0.01)

	if InputParser.validate("Topographical entropy method", topogEntropyMethod, allowedValues = ["0","1", "2", "3", "4", "aem"]):
		doTopogEntropy = True
	else:
		doTopogEntropy = False


	# Deal with the input and output file's presence
	# output
	if append:
		Utils.printflush('{:<40s} : {}'.format('Output statements will be appended to', outFile))
		Writer.append_file(outFile)
	else:
		Utils.printflush('{:<40s} : {}'.format('Output statements will be written to', outFile))
		Writer.write_file(outFile)

	# open matrices and NMD format output files for writting if asked
	if moutFile:
		Writer.write_file(moutFile)

	if nmdFile:
		Writer.write_file(nmdFile)


	
	# create the base molecule and the host data container
	mainMolecule, mainContainer = CharmmReader.read_charmm_input(arg_xdcdFile=xtrajFile, \
		                                                         arg_fdcdFile=ftrajFile,\
		                                                         arg_psfFile=topolFile, \
		                                                         arg_beginTime=beginTime, \
		                                                         arg_endTime=endTime, \
		                                                         arg_stride = stride, \
		                                                         arg_verbose=verbose, \
		                                                         arg_outFile=outFile)

	# ALL THE CALCULATIONS OCCUR STARTING THIS POINT
	computeEntropy = True

	if computeEntropy:

		### ENTROPY AT THE WHOLE MOLECULE LEVEL (RELOCATE) ###
		if doMolEntropy:
			EntropyFunctions.compute_entropy_whole_molecule_level(arg_baseMolecule=mainMolecule\
			                                                         ,arg_hostDataContainer=mainContainer\
																	 , arg_outFile = outFile\
																	 , arg_moutFile = moutFile\
																	 , arg_nmdFile = nmdFile\
																	 , arg_fScale = fScale\
																	 , arg_tScale = tScale\
																	 , arg_temper = temper\
																	 , arg_verbose = verbose)

		### ENTROPY AT THE RESIDUE LEVEL (RELOCATE) ###
		if doResEntropy:
			EntropyFunctions.compute_entropy_residue_level(arg_baseMolecule=mainMolecule\
			                                                 , arg_hostDataContainer=mainContainer\
															 , arg_outFile = outFile\
															 , arg_moutFile = moutFile\
															 , arg_nmdFile = nmdFile\
															 , arg_fScale = fScale\
															 , arg_tScale = tScale\
															 , arg_temper = temper\
															 , arg_verbose = verbose)

		### ENTROPY AT THE UNITED ATOM LEVEL (RELOCATE) ###
		if doUAEntropy:
			EntropyFunctions.compute_entropy_UA_level(arg_baseMolecule = mainMolecule\
			                                             , arg_hostDataContainer = mainContainer\
														 , arg_outFile = outFile\
														 , arg_moutFile = moutFile\
														 , arg_nmdFile = nmdFile\
														 , arg_fScale = fScale\
														 , arg_tScale = tScale\
														 , arg_temper = temper\
														 , arg_verbose = verbose)


		### TOPOGRAPHICAL ENTROPY COMPUTED USING THE PROB DISTRO OF ALL THE DIHEDRALS
		if doTopogEntropy:

			if topogEntropyMethod == "1":
				EntropyFunctions.compute_topographical_entropy0_SC(arg_baseMolecule = mainMolecule\
				                                                      , arg_hostDataContainer = mainContainer\
																	  , arg_outFile = outFile \
																	  , arg_verbose = verbose)
				EntropyFunctions.compute_topographical_entropy0_BB(arg_baseMolecule = mainMolecule\
				                                                      , arg_hostDataContainer = mainContainer\
																	  , arg_outFile = outFile \
																	  , arg_verbose = verbose)

			elif topogEntropyMethod == "2":
				EntropyFunctions.compute_topographical_entropy1_SC(arg_baseMolecule = mainMolecule\
				                                                      , arg_hostDataContainer = mainContainer\
																	  , arg_outFile = outFile \
																	  , arg_verbose = verbose)
				EntropyFunctions.compute_topographical_entropy1_BB(arg_baseMolecule = mainMolecule\
					                                                  , arg_hostDataContainer = mainContainer\
					                                                  , arg_outFile = outFile\
					                                                  , arg_verbose = verbose)

			elif topogEntropyMethod == "3":
				EntropyFunctions.compute_topographical_entropy_method3(arg_baseMolecule = mainMolecule\
				                                                         , arg_hostDataContainer = mainContainer\
																		 , arg_outFile = outFile \
																		 , arg_verbose = verbose)

			elif topogEntropyMethod == "4":
				EntropyFunctions.compute_topographical_entropy_method4(arg_baseMolecule = mainMolecule\
				                                                         , arg_hostDataContainer = mainContainer\
																		 , arg_outFile = outFile \
																		 , arg_verbose = verbose)
			elif topogEntropyMethod == "aem":
				EntropyFunctions.compute_topographical_entropy_AEM(arg_baseMolecule = mainMolecule\
					                                               , arg_hostDataContainer = mainContainer\
					                                               , arg_outFile = outFile\
					                                               , arg_verbose = verbose)

	t1 = datetime.now()
	Utils.printflush('Program ended on : ', end = '')
	Utils.print_time(t1)

	Utils.printflush("{^40s}".format('Total time elpased : '), end = '')
	Utils.print_duration(t0, t1)
	return
#END

	

if __name__ == "__main__":
	mcc_charmm()
