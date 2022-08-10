Run As Script
=================
CodeEntropy can also be run as a script for more advanced operations.

.. _load-custom:

1. Loading data
-------------------
First you should load the data into an MDanalysis universe. For this code, MDanalysis must be able to read dihedral and force directly from the trajectory files 
See `MDAnalysis Format Overview <https://userguide.mdanalysis.org/stable/formats/index.html>`_ for more information.

.. code-block:: python

    import MDAnalysis as mda
    tprfile = "data/1AKI_ws_60.tpr"
    trrfile = "data/1AKI_ws_60.trr"
    u = mda.Universe(tprfile, trrfile)


Non-standard file
^^^^^^^^^^^^^^^^^^
You may be able to load non standard file by turning the force data into an array and create a new universe by combining the force and trajectory data.

.. code-block:: python

    import MDAnalysis as mda
    from MDAnalysis.analysis.base import AnalysisFromFunction
    from MDAnalysis.coordinates.memory import MemoryReader

    topo_file = "data/molecules.prmtop"
    traj_file = "data/Trajectory_npt_1.data.gz"
    ## remember to edit the format so that the header is "id mass x y z" otherwise MDAnalysis won't load the data due to checks by LAMMPS parser 
    force_file = "data/Forces_npt_1.data"
    # loading data into individual universe
    main = mda.Universe(topo_file, traj_file, atom_style='id type x y z' ,format="LAMMPSDUMP")
    force = mda.Universe(topo_file, force_file, atom_style='id type x y z' ,format="LAMMPSDUMP")
    # selection for accessing values 
    # the 'all' can be replaced by other selection string for
    select_atom = main.select_atoms('all')
    select_atom_force = force.select_atoms('all')
    # loading values from universe
    # this is done by generating a tuple from AnalysisFromFunction to traverse through the entire data and loading the selected data into a tuple
    coordinates = AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom).run().results['timeseries']
    forces = AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom_force).run().results['timeseries']
    ## dimension is also required for poseidon analysis 
    dimensions = AnalysisFromFunction(lambda ag: ag.dimensions.copy(), select_atom).run().results['timeseries']
    # create a new universe 
    u2 = mda.Merge(select_atom)
    # loading trajectory data using MemoryReader from tuples, the system is not in memory
    u2.load_new(coordinates, format=MemoryReader, forces=forces, dimensions=dimensions)    

2a. Solute Analysis
------------------------
To start solute analysis, you must load your data into the solute data container but you may want to trim the data using ``CodeEntropy.IO.MDAUniverseHelper`` functions.

.. code-block:: python

    from CodeEntropy.ClassCollection import DataContainer as DC
    from CodeEntropy.IO import MDAUniverseHelper as MDAHelper
    start = 0
    end = 21
    step = 2
    reduced_frame = MDAHelper.new_U_select_frame(u,  start, end, step)
    selection_string = "protein"
    reduced_atom = MDAHelper.new_U_select_atom(reduced_frame, selection_string)

    dataContainer = DC.DataContainer(reduced_atom)

You can now calculate entropy at different levels with the following function using information from the ``dataContainer``

.. code-block:: python

    from CodeEntropy.FunctionCollection import EntropyFunctions as EF
    tScale = 1.0
    fScale = 1.0
    temper = 300.0 #K
    axis_list = ['C', 'CA', 'N']
    outfile = 'outfile.out'
    # Whole molecule level
    wm_entropyFF, wm_entropyTT = EF.compute_entropy_whole_molecule_level(
        arg_hostDataContainer = dataContainer,
        arg_outFile = outfile,
        arg_selector = "protein", 
        arg_moutFile = 'WholeMolecule_matrix.out',
        arg_nmdFile = 'WholeMolecule_mode_spectra.out',
        arg_fScale = fScale,
        arg_tScale = tScale,
        arg_temper = temper,
        arg_verbose = 5
    )    

    #residue level
    res_entropyFF, res_entropyTT = EF.compute_entropy_residue_level(
        arg_hostDataContainer = dataContainer,
        arg_outFile = outfile,
        arg_selector = 'all', 
        arg_moutFile = 'ResidueLevel_matrix.out',
        arg_nmdFile = 'ResidueLevel_mode_spectra.out',
        arg_fScale = fScale,
        arg_tScale = tScale,
        arg_temper = temper,
        arg_verbose = 5,
        arg_axis_list = axis_list,
    )

    #United atom level
    UA_entropyFF, UA_entropyTT, res_df = EF.compute_entropy_UA_level(
        arg_hostDataContainer = dataContainer,
        arg_outFile = outfile,
        arg_selector = 'all', 
        arg_moutFile = 'AtomLevel_matrix.out',
        arg_nmdFile = 'AtomLevel_mode_spectra.out',
        arg_fScale = fScale,
        arg_tScale = tScale,
        arg_temper = temper,
        arg_verbose = 1,
        arg_axis_list = axis_list,
        arg_csv_out= 'AtomLevel_bead_entropy.csv',
    )
    UA_entropyFF, UA_entropyTT, res_df = EF.compute_entropy_UA_level_multiprocess(
        arg_hostDataContainer = dataContainer,
        arg_outFile = outfile,
        arg_selector = 'all', 
        arg_moutFile = 'AtomLevel_matrix.out',
        arg_nmdFile = 'AtomLevel_mode_spectra.out',
        arg_fScale = fScale,
        arg_tScale = tScale,
        arg_temper = temper,
        arg_verbose = 1,
        arg_csv_out= 'AtomLevel_bead_entropy.csv',
        arg_axis_list = axis_list,
        arg_thread= 4,
    )

    #Topographical Level
    result_entropy0_SC = EntropyFunctions.compute_topographical_entropy0_SC(
        arg_hostDataContainer = dataContainer, 
        arg_selector = "all",
        arg_outFile = outfile, 
        arg_verbose = 5
    )

    print(f"result_entropy0_SC = {result_entropy0_SC}")

    result_entropy0_BB = EntropyFunctions.compute_topographical_entropy0_BB(
        arg_hostDataContainer = dataContainer, 
        arg_selector = "all",
        arg_outFile = outfile, 
        arg_verbose = 5
    ) 

    print(f"result_entropy0_BB = {result_entropy0_BB}")


    result_entropy1_SC = EntropyFunctions.compute_topographical_entropy1_SC(
        arg_hostDataContainer = dataContainer, 
        arg_selector = "all",
        arg_outFile = outfile, 
        arg_verbose = 5
    )

    print(f"result_entropy1_SC= {result_entropy1_SC}")


    result_entropy1_BB = EntropyFunctions.compute_topographical_entropy1_BB(
        arg_hostDataContainer = dataContainer, 
        arg_selector = "all",
        arg_outFile = outfile, 
        arg_verbose = 5
    ) 
    print(f"result_entropy1_BB= {result_entropy1_BB}")

    result_entropyAEM = EntropyFunctions.compute_topographical_entropy_AEM(
        arg_hostDataContainer = dataContainer, 
        arg_selector = "all",
        arg_outFile = outfile, 
        arg_verbose = 5
    )

    print(f"result_entropyAEM = {result_entropyAEM}")

2b. Solvent Analysis
------------------------

To run solvent analysis you must load the data into the poseidon object

.. code-block:: python

    poseidon_object = Poseidon(container=main, start=0, end=10, water=('SOL',), excludedResnames=("CL",), verbose=False)

After that, add the level/ analysis you want to run to the `level_list` arguments and run with `run_analysis`. There are 4 levels ``['moleculeLevel', 'residLevel_resname', 'atomLevel', 'soluteContacts']`` 

.. code-block:: python

    poseidon_object.run_analysis(level_list=['moleculeLevel', 'residLevel_resname', 'atomLevel', 'soluteContacts'], verbose=False)

