Getting Started
===============

Requirements
----------------

* Python >= 3.11

Installation
----------------
To install the released version:

.. code-block:: bash
    
    pip install CodeEntropy

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/CCPBioSim/CodeEntropy.git

.. code-block:: bash

    cd CodeEntropy

.. code-block:: bash

    pip install .

Input
----------
For supported format (any topology and trajectory formats that can be read by `MDAnalysis <https://userguide.mdanalysis.org/stable/formats/index.html>`_) you will need to output the **coordinates** and **forces** to the **same file**.
Please consult the documentation for your MD simulation code if you need help outputting the forces.

Units
------------
The program assumes the following default unit

.. list-table:: Units
   :widths: 20 20
   :class: tight-table
   :header-rows: 1
   
   * - Quantity
     - Unit
   * - Length
     - Å
   * - Time
     - ps
   * - Charge
     - `e`
   * - Mass
     - u
   * - Force
     - kJ/(mol·Å)

Quick start guide
--------------------

A quick and easy way to get started is to use the command-line tool which you can run in bash by simply typing ``CodeEntropy``

For help
^^^^^^^^^^^
.. code-block:: bash
    
    CodeEntropy --help

Arguments
^^^^^^^^^^^^^
Arguments should go in a config.yaml file.
The values in the yaml file can be overridden by command line arguments.
The top_traj_file argument is necessary to identify your simulation data, the others can use default values.

.. list-table:: Arguments
   :widths: 20 30 10 10
   :class: tight-table
   :header-rows: 1
    
   * - Arguments
     - Description
     - Default
     - Type
   * - ``--top_traj_file`` 
     - Path to Structure/topology file followed by Trajectory file(s). Any MDAnalysis readable files should work  (for example ``GROMACS TPR and TRR`` or ``AMBER PRMTOP and NETCDF``). You will need to output the **coordinates** and **forces** to the **same file** . 
     - Required, no default value
     - list of ``str`` 
   * - ``--selection_string``
     - Selection string for CodeEntropy such as protein or resid, refer to ``MDAnalysis.select_atoms`` for more information.
     - ``"all"``: select all atom in trajectory
     - ``str``
   * - ``--start``
     - Start analysing the trajectory from this frame index.
     - ``0``: From begining
     - ``int``
   * - ``--end``
     - Stop analysing the trajectory at this frame index
     - ``-1``: end of trajectory
     - ``int``
   * - ``--step``
     - Interval between two consecutive frame indices to be read
     - ``1``
     - ``int``
   * - ``--bin_width``
     - Bin width in degrees for making the dihedral angle histogram
     - ``30``
     - ``int``
   * - ``--temperature``
     - Temperature for entropy calculation (K)
     - ``298.0``
     - ``float``
   * - ``--verbose``
     - Enable verbose output
     - ``False``
     - ``bool``
   * - ``--outfile``
     - Name of the file where the text format output will be written.
     - ``outfile.out``
     - ``str``
   * - ``--force_partitioning``
     - Factor for partitioning forces when there are weak correlations
     - ``0.5``
     - ``float``
   * - ``--water_entropy``
     - Use Jas Kalayan's waterEntropy code to calculate the water conformational entropy
     - ``False``
     - ``bool``
   * - ``--grouping``
     - How to group molecules for averaging
     - ``molecules``
     - ``str``

Averaging
^^^^^^^^^
The code is able to average over molecules of the same type.
The grouping arguement is used to control how the averaging is done.
The default is "molecules" which defines molecules by the number and names of the atoms and groups molecules that are the same.
You can also use "each" which makes each molecule its own group, effectively not averaging over molecules.

Example #1
^^^^^^^^^^
Example config.yaml file.

.. literalinclude:: config.yaml

You must specify the location of the topology/trajectory file(s) for the top_traj_file variable as there is no default and CodeEntropy cannot run without the data. The temperature variable should be adjusted to the temperature from the simulation. Changing the force_partitioning variable is possible, but not recommended unless you understand what it does and have a good reason to change it.

If you set end to -1, it will stop at the last frame of the trajectory. So, start = 0, end = -1, and step = 1 will use the whole trajectory.

To run CodeEntropy, you want to use the command line and change into the directory where your config.yaml file is located. As long as the file is named config.yaml, CodeEntropy will find it automatically.

.. code-block:: bash

  CodeEntropy

Example #2
^^^^^^^^^^
To use the same settings as in Example #1, but override trajectory information, you can use the command line flags.

.. code-block:: bash

  CodeEntropy --top_traj_file "md_A4_dna.tpr" "md_A4_dna_xf.trr"

Or as an alternative, you could edit the config.yaml file and use the CodeEntropy command as in the first example.

CodeEntropy creates job* directories for the output, where * is a job number choosen by the so that there are sequentially numbered directories when you rerun CodeEntropy in the same working directory.
Each job* directory contains the output json file and a subdirectory with the log files.

Data Files
^^^^^^^^^^
The example files mentioned above can be downloaded.

`Lysozyme example (~1.2GB) <https://ccpbiosim.ac.uk/file-store/codeentropy-examples/lysozyme_example.tar>`_

`DNA fragment example (~1MB) <https://ccpbiosim.ac.uk/file-store/codeentropy-examples/dna_example.tar>`_
